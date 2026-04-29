import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import config


class HemorrhageSegmenter:
    """Layer 3: ResNet34 U-Net segmentation for pixel-level hemorrhage detection."""

    def __init__(self, encoder="resnet34", encoder_weights="imagenet", device="cpu"):
        self.device = device
        
        # We trained with 2.5D RGB images, so in_channels=3
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(device)
        
        # Attempt to load fine-tuned weights from Hugging Face
        try:
            from huggingface_hub import hf_hub_download
            print(f"[Layer 3 - HemorrhageSegmenter] Downloading fine-tuned U-Net from {config.UNET_SEGMENTER_ID}...")
            model_path = hf_hub_download(
                repo_id=config.UNET_SEGMENTER_ID,
                filename="model.pth",
                token=config.HF_TOKEN,
                local_files_only=config.UNET_LOCAL_FILES_ONLY
            )
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[Layer 3 - HemorrhageSegmenter] Successfully loaded fine-tuned RSNA weights!")
        except Exception as e:
            print(f"[Layer 3 - HemorrhageSegmenter] WARNING: Failed to load fine-tuned weights ({e}).")
            print(f"[Layer 3 - HemorrhageSegmenter] Falling back to base {encoder} (ImageNet weights) on {device}")
        
        self.model.eval()

    def segment_slice(self, image_path: str, threshold: float = 0.5) -> dict:
        # We trained on RGB (3 channels) for 2.5D spatial context.
        img_array = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
        # Convert HWC to CHW -> (3, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        # Add batch dimension -> (1, 3, H, W)
        x = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = torch.sigmoid(self.model(x)).cpu().numpy()[0, 0]
        binary = (mask > threshold).astype(np.uint8)
        return {"mask": binary, "probability_map": mask, "confidence": float(mask.max())}

    def segment_pil_image(self, pil_image: Image.Image, threshold: float = 0.5) -> dict:
        img_array = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        x = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = torch.sigmoid(self.model(x)).cpu().numpy()[0, 0]
        binary = (mask > threshold).astype(np.uint8)
        return {"mask": binary, "probability_map": mask, "confidence": float(mask.max())}
