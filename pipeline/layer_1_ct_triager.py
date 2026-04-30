"""Layer 1: fast zero-shot screening for uploaded CT slices.

MedSigLIP is used here as a lightweight first pass. It does not diagnose the
case by itself; it ranks slices and provides a simple label that later layers
can use for report context and PubMed search hints.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor, SiglipTokenizer

import config


class CTTriager:
    """Layer 1: MedSigLIP-448 zero-shot CT slice triage."""

    MODEL_ID = config.MEDSIGLIP_MODEL_ID

    def __init__(self, device="cpu", hf_token=None):
        token = hf_token or config.HF_TOKEN
        self.device = device
        self.threshold = config.TRIAGE_THRESHOLD

        print(f"[Layer 1 - CTTriager] Loading {self.MODEL_ID} on {device}...")
        self.image_processor = SiglipImageProcessor.from_pretrained(self.MODEL_ID, token=token)
        self.tokenizer = SiglipTokenizer.from_pretrained(self.MODEL_ID, token=token)
        self.model = AutoModel.from_pretrained(self.MODEL_ID, token=token).to(device)
        self.model.eval()
        print(f"[Layer 1 - CTTriager] Ready. Threshold={self.threshold}")

    def score_single_slice(self, pil_image: Image.Image) -> dict:
        """Score one CT slice against the configured trauma-related labels."""
        image = pil_image.convert("RGB").resize(
            (config.TRIAGE_IMAGE_SIZE, config.TRIAGE_IMAGE_SIZE), Image.BILINEAR
        )
        text_in = self.tokenizer(
            config.TRIAGE_LABELS, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        image_in = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**{**text_in, **image_in})

        probs = torch.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()
        suspicious_score = float(sum(probs[i] for i in config.TRIAGE_POSITIVE_INDICES))

        return {
            "scores": {label: float(probs[i]) for i, label in enumerate(config.TRIAGE_LABELS)},
            "suspicious_score": suspicious_score,
            "suspicious": suspicious_score > self.threshold,
            "top_label": config.TRIAGE_LABELS[int(np.argmax(probs))],
        }

    def triage_all_slices(self, pil_images: list) -> list:
        """Score all uploaded slices and return them highest-risk first."""
        results = []
        for i, img in enumerate(pil_images):
            result = self.score_single_slice(img)
            result["slice_index"] = i
            results.append(result)
        return sorted(results, key=lambda x: x["suspicious_score"], reverse=True)

    def get_top_suspicious(self, pil_images: list, max_slices: int = None):
        """Return suspicious images for heavier downstream analysis."""
        max_slices = max_slices or config.TRIAGE_MAX_SLICES
        all_results = self.triage_all_slices(pil_images)
        suspicious = [r for r in all_results if r["suspicious"]] or [all_results[0]]
        top = suspicious[:max_slices]
        indices = [r["slice_index"] for r in top]
        return [pil_images[i] for i in indices], all_results

    def summarize_triage(self, all_results: list) -> dict:
        """Summarize per-slice scores for the UI and report writer."""
        by_index = sorted(all_results, key=lambda x: x["slice_index"])
        per_slice_scores = [r["suspicious_score"] for r in by_index]
        return {
            "total_slices": len(all_results),
            "suspicious_count": sum(1 for r in all_results if r["suspicious"]),
            "max_score": float(max(per_slice_scores)) if per_slice_scores else 0.0,
            "mean_score": float(np.mean(per_slice_scores)) if per_slice_scores else 0.0,
            "per_slice_scores": per_slice_scores,
        }
