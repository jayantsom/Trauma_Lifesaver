"""Layer 3: pixel-level hemorrhage segmentation.

The segmenter runs independently from the language model. It produces masks
and confidence maps that are later converted into rough volume estimates.
"""

import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from pathlib import Path


class HemorrhageSegmenter:
    """Layer 3: ResNet34 U-Net segmentation for pixel-level hemorrhage detection."""

    def __init__(
        self,
        encoder="resnet34",
        encoder_weights="imagenet",
        device="cpu",
        checkpoint_path="models/unet_hemorrhage/unet_hemorrhage_rsna_mask_aware.pth",
    ):
        """Load the trained mask-aware checkpoint when it is available."""
        self.device = device
        self.image_size = 256
        self.injury_labels = ()
        self.checkpoint_path = Path(checkpoint_path)

        checkpoint = None
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                encoder = checkpoint.get("encoder", encoder)
                self.image_size = int(checkpoint.get("image_size", self.image_size))
                self.injury_labels = tuple(checkpoint.get("injury_labels", ()))

        # The classification-with-mask checkpoint stores a segmentation head and
        # an auxiliary label head. If the labels are absent, this remains a pure
        # segmentation model.
        aux_params = None
        if self.injury_labels:
            aux_params = {
                "classes": len(self.injury_labels),
                "pooling": "avg",
                "dropout": 0.2,
                "activation": None,
            }

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None if checkpoint else encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
            aux_params=aux_params,
        ).to(device)

        if checkpoint is not None:
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            print(
                "[Layer 3 - HemorrhageSegmenter] Loaded trained checkpoint "
                f"{self.checkpoint_path} on {device}"
            )
        else:
            print(
                "[Layer 3 - HemorrhageSegmenter] Checkpoint not found; "
                f"loaded {encoder} ({encoder_weights}) on {device}"
            )

        self.model.eval()

    def _prepare_tensor(self, pil_image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        """Prepare an uploaded CT slice for the 2.5D-style trained U-Net."""
        original_size = pil_image.size
        image = pil_image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return x, original_size

    def _predict(self, pil_image: Image.Image, threshold: float) -> dict:
        """Run segmentation and optional injury-label classification."""
        x, original_size = self._prepare_tensor(pil_image)

        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, tuple):
                mask_logits, class_logits = output
            else:
                mask_logits, class_logits = output, None

            mask = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
            class_probs = None
            if class_logits is not None and self.injury_labels:
                class_probs = torch.sigmoid(class_logits).cpu().numpy()[0]

        # Resize the probability map back to the uploaded image size so the
        # downstream volume estimate uses the same dimensions as the input.
        mask_img = Image.fromarray((mask * 255.0).astype(np.uint8)).resize(original_size, Image.BILINEAR)
        mask = np.asarray(mask_img).astype(np.float32) / 255.0
        binary = (mask > threshold).astype(np.uint8)

        result = {
            "mask": binary,
            "probability_map": mask,
            "confidence": float(mask.max()),
        }
        if class_probs is not None:
            result["classification"] = {
                label: float(prob) for label, prob in zip(self.injury_labels, class_probs)
            }
        return result

    def segment_slice(self, image_path: str, threshold: float = 0.5) -> dict:
        """Segment a saved CT slice from disk."""
        return self._predict(Image.open(image_path), threshold)

    def segment_pil_image(self, pil_image: Image.Image, threshold: float = 0.5) -> dict:
        """Segment an in-memory CT slice, mainly useful for tests and notebooks."""
        return self._predict(pil_image, threshold)
