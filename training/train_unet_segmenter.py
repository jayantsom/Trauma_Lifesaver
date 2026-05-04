"""Low-disk RSNA mask-aware trainer for Trauma Lifesaver.

Uses the Hugging Face repo files directly because newer `datasets` releases no
longer allow this dataset's loading script. By default, downloaded NIfTI files
are deleted immediately after they are loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit("Install first: py -3.11 -m pip install -r requirements-unet.txt") from exc


DATASET_ID = "jherng/rsna-2023-abdominal-trauma-detection"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
MASK_CONFIGS = ("segmentation", "classification-with-mask")
INJURY_LABELS = ("bowel", "extravasation", "kidney", "liver", "spleen", "any_injury")


@dataclass(frozen=True)
class SliceExample:
    """One 2D training slice selected from a 3D CT volume."""

    img_url: str
    mask_url: str
    z_index: int
    labels: tuple[float, ...] | None


def truthy(value) -> bool:
    """Convert CSV-style truth values into a Python boolean."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return True
    try:
        return float(text) > 0
    except ValueError:
        return False


def number(value, default=0.0) -> float:
    """Parse a numeric CSV field while keeping malformed values harmless."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def argmax_label(row: dict[str, str], names: tuple[str, ...]) -> int:
    """Return the winning class index for one-hot RSNA label columns."""
    return int(np.argmax([number(row.get(name)) for name in names]))


class RepoCache:
    """Tiny file cache for RSNA metadata and NIfTI downloads."""

    def __init__(self, cache_dir: str | Path, keep_niftis: bool):
        """Create the cache folder and remember whether volumes should persist."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.keep_niftis = keep_niftis

    def path_for_url(self, url: str) -> Path:
        """Map a remote URL to a stable local cache filename."""
        suffix = ".nii.gz" if url.endswith(".nii.gz") else Path(url).suffix
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}{suffix}"

    def download_url(self, url: str) -> Path:
        """Download a NIfTI file if it is not already cached."""
        path = self.path_for_url(url)
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
            except OSError:
                path.unlink(missing_ok=True)
                raise
        return path

    def download_table(self, name: str) -> Path:
        """Download a small CSV metadata table from the HF dataset repo."""
        path = self.cache_dir / name
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        return path

    def read_csv(self, name: str) -> list[dict[str, str]]:
        """Read a cached or newly downloaded CSV table."""
        with self.download_table(name).open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def release(self, path: Path):
        """Remove a downloaded volume after use when low-disk mode is active."""
        if self.keep_niftis:
            return
        try:
            if path.is_file() and path.parent.resolve() == self.cache_dir.resolve():
                path.unlink()
        except OSError:
            pass


def read_nifti(path_text: str) -> np.ndarray:
    """Load a NIfTI file as a float32 H x W x D volume."""
    volume = nib.load(path_text).get_fdata(dtype=np.float32)
    while volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim == 2:
        volume = volume[:, :, None]
    return volume


def load_volume(cache: RepoCache, url: str) -> np.ndarray:
    """Download, load, and optionally release one CT or mask volume."""
    path = cache.download_url(url)
    try:
        volume = read_nifti(str(path))
    finally:
        cache.release(path)
    return volume


def split_rows(rows: list[dict], split: str, test_size: float, seed: int) -> list[dict]:
    """Reproduce the dataset's deterministic train/test split."""
    order = np.random.RandomState(seed).permutation(len(rows))
    test_count = int(np.ceil(len(rows) * test_size))
    wanted = order[test_count:] if split == "train" else order[:test_count]
    return [rows[int(i)] for i in wanted]


def load_items(config: str, split: str, args) -> list[dict]:
    """Build RSNA sample dictionaries without using deprecated dataset scripts."""
    cache = RepoCache(args.cache_dir, keep_niftis=True)
    series_rows = cache.read_csv("train_series_meta.csv")
    label_rows = cache.read_csv("train.csv") if config == "classification-with-mask" else []
    labels_by_patient = {int(number(r["patient_id"])): r for r in label_rows}
    items = []

    for row in series_rows:
        if not truthy(row.get("has_segmentation")):
            continue
        patient_id = int(number(row["patient_id"]))
        series_id = int(number(row["series_id"]))
        item = {
            "img_url": f"{BASE_URL}/train_images/{patient_id}/{series_id}.nii.gz",
            "mask_url": f"{BASE_URL}/segmentations/{series_id}.nii.gz",
        }
        if config == "classification-with-mask":
            labels = labels_by_patient.get(patient_id)
            if not labels:
                continue
            item["labels"] = (
                float(argmax_label(labels, ("bowel_healthy", "bowel_injury")) > 0),
                float(argmax_label(labels, ("extravasation_healthy", "extravasation_injury")) > 0),
                float(argmax_label(labels, ("kidney_healthy", "kidney_low", "kidney_high")) > 0),
                float(argmax_label(labels, ("liver_healthy", "liver_low", "liver_high")) > 0),
                float(argmax_label(labels, ("spleen_healthy", "spleen_low", "spleen_high")) > 0),
                float(truthy(labels.get("any_injury"))),
            )
        items.append(item)

    return split_rows(items, split, args.test_size, args.random_state)


def choose_slices(mask: np.ndarray, limit: int) -> list[int]:
    """Choose mask-positive slices, falling back to the center slice if needed."""
    depth = mask.shape[2]
    positive = np.flatnonzero(mask.reshape(-1, depth).sum(axis=0) > 0)
    candidates = positive.tolist() if positive.size else [depth // 2]
    if len(candidates) <= limit:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, limit)
    return [candidates[int(round(i))] for i in picks]


def to_uint8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply a CT window and scale one slice to uint8 image space."""
    low = center - width / 2
    high = center + width / 2
    scaled = (np.clip(slice_2d, low, high) - low) / max(high - low, 1.0)
    return (scaled * 255).astype(np.uint8)


class RSNASliceDataset(Dataset):
    """Lazy 2.5D slice dataset built from mask-aware RSNA volumes."""

    def __init__(self, config: str, split: str, args, train: bool):
        """Index useful slices once, then load image/mask volumes per batch."""
        self.cache = RepoCache(args.cache_dir, args.keep_nifti_cache)
        self.args = args
        self.transform = build_transform(args.image_size, train)
        self.examples: list[SliceExample] = []
        items = load_items(config, split, args)
        max_volumes = args.max_train_volumes if train else args.max_val_volumes
        if max_volumes:
            items = items[:max_volumes]

        for item in items:
            mask = load_volume(self.cache, item["mask_url"])
            for z in choose_slices(mask, args.slices_per_volume):
                self.examples.append(SliceExample(item["img_url"], item["mask_url"], z, item.get("labels")))

        if not self.examples:
            raise RuntimeError(f"No examples found for config={config} split={split}")

    def __len__(self):
        """Return the number of selected 2D slices."""
        return len(self.examples)

    def __getitem__(self, index: int):
        """Load one 2.5D image, binary mask, and optional injury-label vector."""
        ex = self.examples[index]
        image_volume = load_volume(self.cache, ex.img_url)
        mask_volume = load_volume(self.cache, ex.mask_url)
        z = min(ex.z_index, image_volume.shape[2] - 1, mask_volume.shape[2] - 1)
        zs = [max(0, z - 1), z, min(image_volume.shape[2] - 1, z + 1)]
        image = np.stack(
            [to_uint8(image_volume[:, :, zi], self.args.window_center, self.args.window_width) for zi in zs],
            axis=-1,
        )
        mask = (mask_volume[:, :, z] > 0).astype(np.float32)
        batch = self.transform(image=image, mask=mask)
        labels = torch.full((len(INJURY_LABELS),), -1.0)
        if ex.labels is not None:
            labels = torch.tensor(ex.labels, dtype=torch.float32)
        return {"image": batch["image"], "mask": batch["mask"].unsqueeze(0).float(), "labels": labels}


def build_transform(image_size: int, train: bool):
    """Create train/validation preprocessing and augmentation transforms."""
    ops = [A.Resize(image_size, image_size)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.92, 1.08), translate_percent=(-0.03, 0.03), rotate=(-12, 12), p=0.5),
            A.RandomBrightnessContrast(p=0.35),
        ]
    ops += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    return A.Compose(ops)


class Loss(nn.Module):
    """Dice+BCE segmentation loss with optional classification BCE."""

    def __init__(self, cls_weight: float):
        """Set up segmentation and classification loss terms."""
        super().__init__()
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.cls_bce = nn.BCEWithLogitsLoss()
        self.cls_weight = cls_weight

    def forward(self, mask_logits, masks, cls_logits, labels):
        """Compute multitask loss for one batch."""
        bce = self.seg_bce(mask_logits, masks)
        probs = torch.sigmoid(mask_logits)
        inter = (probs * masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * inter + 1) / (union + 1)).mean()
        loss = bce + dice_loss
        valid = labels[:, 0] >= 0
        if cls_logits is not None and valid.any():
            loss = loss + self.cls_weight * self.cls_bce(cls_logits[valid], labels[valid])
        return loss


def model_outputs(output):
    """Normalize SMP output into mask logits and optional class logits."""
    return output if isinstance(output, tuple) else (output, None)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, accum=1):
    """Run one train or validation epoch and return average loss/Dice."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_dice = 0.0
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train" if train_mode else "valid", leave=False)
    for step, batch in enumerate(progress, 1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.set_grad_enabled(train_mode):
            if scaler:
                with torch.amp.autocast("cuda"):
                    mask_logits, cls_logits = model_outputs(model(images))
                    loss = criterion(mask_logits, masks, cls_logits, labels)
            else:
                mask_logits, cls_logits = model_outputs(model(images))
                loss = criterion(mask_logits, masks, cls_logits, labels)
            if train_mode and scaler:
                scaler.scale(loss / accum).backward()
            elif train_mode:
                (loss / accum).backward()
        if train_mode and (step % accum == 0 or step == len(loader)):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pred = (torch.sigmoid(mask_logits) > 0.5).float()
            inter = (pred * masks).sum(dim=(1, 2, 3))
            union = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = ((2 * inter + 1) / (union + 1)).mean()
        total_loss += float(loss.detach())
        total_dice += float(dice.detach())
        progress.set_postfix(loss=f"{total_loss / step:.4f}", dice=f"{total_dice / step:.4f}")
    return total_loss / len(loader), total_dice / len(loader)


def make_dataset(args, split: str, train: bool):
    """Build one or more configured datasets for the requested split."""
    configs = MASK_CONFIGS if args.config == "both" else (args.config,)
    parts = [RSNASliceDataset(config, split, args, train) for config in configs]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def save_checkpoint(path: Path, model, args, val_loss, val_dice):
    """Persist the best model and the metadata needed for app inference."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "smp.Unet",
            "encoder": args.encoder,
            "in_channels": 3,
            "classes": 1,
            "image_size": args.image_size,
            "injury_labels": INJURY_LABELS if args.config in ("both", "classification-with-mask") else (),
            "val_loss": val_loss,
            "val_dice": val_dice,
        },
        path,
    )


def train(args):
    """Coordinate dataset setup, model creation, training, and checkpointing."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / args.checkpoint_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aux = None
    if args.config in ("both", "classification-with-mask"):
        aux = {"classes": len(INJURY_LABELS), "pooling": "avg", "dropout": 0.2, "activation": None}

    print("=" * 72)
    print("RSNA 2023 mask-aware training for Trauma Lifesaver")
    print("=" * 72)
    print(f"Dataset configs: {args.config}")
    print(f"Device: {device}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Keep NIfTI cache: {args.keep_nifti_cache}")
    print(f"Output: {ckpt}")

    print("\n[1/4] Preparing RSNA slice datasets...")
    train_ds = make_dataset(args, "train", True)
    val_ds = make_dataset(args, "test", False)
    print(f"Train slices: {len(train_ds)}")
    print(f"Valid slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n[2/4] Building U-Net...")
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
        aux_params=aux,
    ).to(device)
    criterion = Loss(args.classification_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.num_epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and args.amp else None

    print("\n[3/4] Training...")
    best_loss = float("inf")
    best_dice = 0.0
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        tr_loss, tr_dice = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, args.gradient_accumulation_steps
        )
        va_loss, va_dice = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} val_loss={va_loss:.4f} val_dice={va_dice:.4f}")
        if va_loss < best_loss:
            best_loss = va_loss
            best_dice = va_dice
            save_checkpoint(ckpt, model, args, va_loss, va_dice)
            print(f"Saved best checkpoint: {ckpt}")

    print("\n[4/4] Done.")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoint: {ckpt}")


def parse_args():
    """Parse CLI options for local or Colab training."""
    parser = argparse.ArgumentParser(description="Train U-Net on RSNA mask-aware configs.")
    parser.add_argument("--config", choices=("segmentation", "classification-with-mask", "both"), default="both")
    parser.add_argument("--output_dir", default="models/unet_hemorrhage")
    parser.add_argument("--checkpoint_name", default="unet_hemorrhage_rsna_mask_aware.pth")
    parser.add_argument("--cache_dir", default=os.path.join(tempfile.gettempdir(), "trauma_lifesaver_rsna_cache"))
    parser.add_argument("--keep_nifti_cache", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--classification_weight", type=float, default=0.25)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder_weights", default="imagenet")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--slices_per_volume", type=int, default=6)
    parser.add_argument("--max_train_volumes", type=int, default=None)
    parser.add_argument("--max_val_volumes", type=int, default=None)
    parser.add_argument("--window_center", type=float, default=50.0)
    parser.add_argument("--window_width", type=float, default=400.0)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
