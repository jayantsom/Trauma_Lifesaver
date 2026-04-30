"""Training script for the U-Net hemorrhage segmenter.

This script was used for Colab-based experiments on the RSNA abdominal trauma
dataset. It keeps memory usage low by loading NIfTI files lazily and training a
ResNet34 U-Net on 2.5D CT slices.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Installing segmentation_models_pytorch...")
    os.system("pip install segmentation-models-pytorch")
    import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Installing albumentations...")
    os.system("pip install albumentations")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Colab Drive Mounting Setup
# ---------------------------------------------------------------------------
def mount_drive_if_colab():
    """Return a Drive-backed model folder when running inside Colab."""
    if os.path.exists('/content/drive/MyDrive'):
        return "/content/drive/MyDrive/models"
    else:
        return "models"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ID = "jherng/rsna-2023-abdominal-trauma-detection"
CT_WINDOW_CENTER = 50
CT_WINDOW_WIDTH = 400


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_nifti_slice_and_mask(
    img_path: str,
    mask_path: str = None,
    center: int = CT_WINDOW_CENTER,
    width: int = CT_WINDOW_WIDTH,
):
    """Load one CT volume and prepare a middle 2.5D slice with its mask."""
    import nibabel as nib
    import tempfile

    def _load_nii(path):
        if not path.endswith((".nii", ".nii.gz")):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = os.path.join(tmpdir, "file.nii.gz")
                os.symlink(os.path.abspath(path), tmp)
                return nib.load(tmp).get_fdata()
        return nib.load(path).get_fdata()

    vol = _load_nii(img_path)
    while vol.ndim > 3:
        vol = vol[..., 0]
    if vol.ndim == 2:
        vol = vol[:, :, np.newaxis]

    depth = vol.shape[2]
    # Pick the center slice
    z = depth // 2
    
    # 2.5D approach: get z-1, z, z+1
    z_indices = [max(0, z-1), z, min(depth-1, z+1)]
    
    lo = center - width / 2.0
    hi = center + width / 2.0

    channels = []
    for zi in z_indices:
        sl = vol[:, :, zi].astype(np.float32)
        sl = np.clip(sl, lo, hi)
        sl = (sl - lo) / (hi - lo) * 255.0
        channels.append(sl.astype(np.uint8))
        
    rgb_image = np.stack(channels, axis=-1)  # (H, W, 3)

    # Process Mask
    if mask_path and os.path.exists(mask_path):
        mask_vol = _load_nii(mask_path)
        while mask_vol.ndim > 3:
            mask_vol = mask_vol[..., 0]
        if mask_vol.ndim == 2:
            mask_vol = mask_vol[:, :, np.newaxis]
            
        mask_slice = mask_vol[:, :, z].astype(np.float32)
        # Assuming hemorrhage label > 0 in the mask
        binary_mask = (mask_slice > 0).astype(np.float32)
    else:
        # If no mask path provided in this dataset sample, create a dummy empty mask.
        # Note: In a real RSNA segmentation run, ensure mask_path is correctly mapped.
        binary_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
        
    return rgb_image, binary_mask


class LazySegmentationDataset(Dataset):
    """Small dataset wrapper that loads CT volumes only when a batch asks for them."""

    def __init__(self, hf_dataset, transform=None):
        print("      Fetching metadata for selected samples (this may take a moment)...")
        self.dataset = list(hf_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        img_path = item.get("img_path")
        # For RSNA, segmentations are sometimes in a parallel directory or mapped in metadata
        mask_path = item.get("seg_path") or item.get("mask_path") 
        
        try:
            image, mask = load_nifti_slice_and_mask(img_path, mask_path)
        except Exception:
            # Fallback for failing NIfTIs
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            mask = np.zeros((224, 224), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask has channel dimension (1, H, W)
        mask = mask.unsqueeze(0)
        
        return image, mask


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combination loss commonly used for sparse medical segmentation masks."""

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return bce_loss + dice_loss


def train(args):
    """Run the training loop and save the best checkpoint to disk."""
    print(f"\n{'='*60}")
    print("U-Net (ResNet34) Fine-Tuning on RSNA Trauma Segmentation")
    print(f"{'='*60}\n")

    mount_drive_if_colab()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the directory name to use as the full .pth file name
    model_name = output_dir.name + ".pth"
    model_save_path = output_dir / model_name
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Device: {device}")

    # --- Load dataset ---
    print(f"\n[1/4] Loading the dataset ({DATASET_ID}) in streaming mode...")
    dataset = load_dataset(DATASET_ID, split="train", token=args.hf_token, trust_remote_code=True, streaming=True)
    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    # Augmentations
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = LazySegmentationDataset(dataset, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0  # 0 to avoid Colab multiprocessing issues
    )

    # --- Load model ---
    print(f"\n[2/4] Initializing U-Net model with ResNet34 backbone...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3, # 2.5D RGB input
        classes=1,     # Binary mask (hemorrhage vs background)
        activation=None
    ).to(device)

    # --- Configuration ---
    print(f"\n[3/4] Configuring the optimizer and loss function...")
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = torch.amp.GradScaler('cuda') if cuda else None

    # --- Train Loop ---
    print(f"\n[4/4] Starting the training process...")
    print(f"  Epochs:           {args.num_epochs}")
    print(f"  Batch size:       {args.batch_size} (Accumulation: {args.gradient_accumulation_steps})")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Output Model:     {model_save_path}\n")

    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(device), masks.to(device)

            # Mixed precision training
            if cuda:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss = loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  --> Saving best model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            
    print(f"\n[Done] Training complete. Best model saved to: {model_save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line options for segmentation training."""
    parser = argparse.ArgumentParser(description="Fine-tune U-Net for RSNA Hemorrhage Segmentation")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/Trauma_Lifesaver/models/unet-resnet34-rsna23-abd-ct-seg-ep10-lr1e4-v1",
                        help="Directory to save the trained model weights")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max training examples (use 200 for fast prototype, increase for production)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per forward pass")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients (effective batch size = batch_size * grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Peak learning rate")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
