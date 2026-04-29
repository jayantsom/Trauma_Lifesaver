import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import config


class HemorrhageSegmenter:
    """Layer 3: ResNet34 U-Net segmentation for pixel-level hemorrhage detection."""

    def __init__(self, encoder="resnet34", encoder_weights="imagenet", device="cpu"):
        self.device = device
        self.model  = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=1,   # grayscale CT
            classes=1,
            activation=None,
        ).to(device)
        self.model.eval()
        print(f"[Layer 3 - HemorrhageSegmenter] Loaded {encoder} (ImageNet weights) on {device}")

    def segment_slice(self, image_path: str, threshold: float = 0.5) -> dict:
        img_array = np.array(Image.open(image_path).convert("L")).astype(np.float32) / 255.0
        x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = torch.sigmoid(self.model(x)).cpu().numpy()[0, 0]
        binary = (mask > threshold).astype(np.uint8)
        return {"mask": binary, "probability_map": mask, "confidence": float(mask.max())}

    def segment_pil_image(self, pil_image: Image.Image, threshold: float = 0.5) -> dict:
        img_array = np.array(pil_image.convert("L")).astype(np.float32) / 255.0
        x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = torch.sigmoid(self.model(x)).cpu().numpy()[0, 0]
        binary = (mask > threshold).astype(np.uint8)
        return {"mask": binary, "probability_map": mask, "confidence": float(mask.max())}
