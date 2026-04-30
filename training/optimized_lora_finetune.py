"""LoRA fine-tuning script for MedGemma trauma experiments.

This is an offline training utility, separate from the Flask app. It prepares
RSNA abdominal trauma examples as image/text chat pairs and trains lightweight
LoRA adapters instead of updating the full MedGemma model.
"""

import argparse
import gzip
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "google/medgemma-1.5-4b-it"
DATASET_ID = "jherng/rsna-2023-abdominal-trauma-detection"

# LoRA targets for Gemma3 architecture (PaliGemma-style attention + MLP)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

# CT soft tissue windowing defaults (top Kaggle solutions, RSNA 2023)
CT_WINDOW_CENTER = 50
CT_WINDOW_WIDTH = 400


def load_nifti_as_pil(
    path: str,
    num_slices: int = 3,
    center: int = CT_WINDOW_CENTER,
    width: int = CT_WINDOW_WIDTH,
) -> Image.Image:
    """Load a NIfTI CT volume and convert sampled slices into one RGB image."""
    import nibabel as nib
    import tempfile

    if not path.endswith((".nii", ".nii.gz")):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = os.path.join(tmpdir, "ct.nii.gz")
            os.symlink(os.path.abspath(path), tmp)
            vol = nib.load(tmp).get_fdata()
    else:
        vol = nib.load(path).get_fdata()

    while vol.ndim > 3:
        vol = vol[..., 0]
    if vol.ndim == 2:
        vol = vol[:, :, np.newaxis]

    depth = vol.shape[2]

    # Sample from middle 60%
    z_start = int(depth * 0.2)
    z_end = max(z_start + 1, int(depth * 0.8))
    z_indices = np.linspace(z_start, z_end - 1, num_slices, dtype=int)

    lo = center - width / 2.0
    hi = center + width / 2.0

    channels = []
    for z in z_indices:
        sl = vol[:, :, z].astype(np.float32)
        sl = np.clip(sl, lo, hi)
        sl = (sl - lo) / (hi - lo) * 255.0
        channels.append(sl.astype(np.uint8))

    while len(channels) < 3:
        channels.append(channels[-1])

    rgb = np.stack(channels[:3], axis=-1)
    return Image.fromarray(rgb)


class LazyTraumaDataset(torch.utils.data.Dataset):
    """Dataset wrapper that delays NIfTI loading until a sample is requested."""
    def __init__(self, hf_dataset, max_samples=None, num_slices=3):
        # Consume the iterable dataset into a list so we can index it,
        # but because it's a 'take(200)' stream, it will only download those 200!
        print("      Fetching metadata for selected samples (this may take a moment)...")
        self.dataset = list(hf_dataset)
        self.num_slices = num_slices
        self.GRADE_NAMES = {0: "healthy", 1: "low-grade injury", 2: "high-grade injury"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load image lazily
        ct_image = None
        img_path = item.get("img_path")
        
        if img_path:
            try:
                ct_image = load_nifti_as_pil(img_path, num_slices=self.num_slices)
            except Exception:
                pass
                
        # Fallbacks for other dataset types (HF images, DICOMs)
        if ct_image is None:
            img_field = item.get("image")
            if img_field is not None and isinstance(img_field, Image.Image):
                ct_image = img_field.convert("RGB")
        
        # If absolutely failed to load, provide a blank placeholder to avoid crashing the batch
        if ct_image is None:
            ct_image = Image.new("RGB", (224, 224))
            
        # Extract Labels
        extravasation = int(item.get("extravasation", 0))
        liver   = int(item.get("liver",   0))
        spleen  = int(item.get("spleen",  0))
        kidney  = int(item.get("kidney",  0))
        bowel   = int(item.get("bowel",   0))
        any_injury = bool(item.get("any_injury", False))

        # Determine organs and severity
        organs_involved = []
        if liver   > 0: organs_involved.append(f"liver ({self.GRADE_NAMES[liver]})")
        if spleen  > 0: organs_involved.append(f"spleen ({self.GRADE_NAMES[spleen]})")
        if kidney  > 0: organs_involved.append(f"kidney ({self.GRADE_NAMES[kidney]})")
        if bowel   > 0: organs_involved.append("bowel (injury)")

        max_grade = max(liver, spleen, kidney)
        if max_grade == 2 or (extravasation == 1 and max_grade >= 1):
            severity = "severe"
        elif extravasation == 1 or max_grade == 1 or bowel == 1:
            severity = "moderate"
        else:
            severity = "none"

        # Construct textual JSON output
        if not any_injury:
            injury_pattern = "No acute intraabdominal injury identified"
            bleeding_description = "No active extravasation or hemoperitoneum detected"
            differential = ["No acute injury", "Clinically correlate with exam findings"]
        else:
            organ_str = ", ".join(organs_involved) if organs_involved else "abdomen"
            if extravasation == 1:
                injury_pattern = f"Active arterial extravasation with solid organ injury: {organ_str}"
                bleeding_description = "Active contrast extravasation consistent with ongoing hemorrhage"
            else:
                injury_pattern = f"Solid organ laceration without active extravasation: {organ_str}"
                bleeding_description = "Parenchymal laceration with hematoma; no active extravasation"
            differential = ["Solid organ laceration with contained hematoma", "Active arterial extravasation requiring intervention", "Subcapsular hematoma"]
            if bowel == 1:
                differential.append("Bowel perforation with mesenteric injury")

        response_json = json.dumps({
            "injury_pattern": injury_pattern,
            "organs_involved": [o.split(" (")[0] for o in organs_involved],
            "bleeding_description": bleeding_description,
            "severity_estimate": severity,
            "differential_diagnosis": differential,
        }, indent=2)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ct_image},
                    {"type": "text", "text": "You are a trauma radiologist. Analyze this abdominal CT angiogram slice for hemorrhage and solid organ injury. Respond in JSON with keys: injury_pattern, organs_involved, bleeding_description, severity_estimate, differential_diagnosis."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_json}]
            }
        ]
        
        return {"messages": messages}


def collate_fn(processor):
    """Return a collate function that applies the MedGemma chat template."""
    def _collate(batch):
        texts = []
        images_list = []

        for example in batch:
            text = processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            imgs = []
            for msg in example["messages"]:
                for part in msg.get("content", []):
                    if part.get("type") == "image":
                        imgs.append(part["image"])
            images_list.append(imgs if imgs else [Image.new("RGB", (224, 224))])

        flat_images = [img for imgs in images_list for img in imgs]

        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512, 
        )

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    return _collate


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    """Run the LoRA training job and save the adapter artifacts."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    print(f"\n{'='*60}")
    print("MedGemma 1.5 LoRA Fine-Tuning on RSNA Trauma Dataset (Optimized)")
    print(f"{'='*60}\n")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    bf16_supported = torch.cuda.is_bf16_supported() if cuda else False
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"Device: {'CUDA — ' + device_name if cuda else 'CPU (slow)'}")
    print(f"Bfloat16 Supported: {bf16_supported}")

    # --- Load dataset ---
    print(f"\n[1/4] Loading the dataset ({DATASET_ID}) in streaming mode...")
    dataset = load_dataset(DATASET_ID, split="train", token=hf_token, trust_remote_code=True, streaming=True)
    
    if args.max_samples:
        dataset = dataset.take(args.max_samples)
        
    # We do NOT build all examples into memory anymore. We create a lazy loader.
    train_dataset = LazyTraumaDataset(dataset, max_samples=args.max_samples, num_slices=args.num_slices)
    print(f"      Lazy dataset initialized. Files will be downloaded on-the-fly.")

    # --- Load model ---
    print(f"\n[2/4] Loading the model ({MODEL_ID})...")

    import gc
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    bnb_config = None
    if cuda:
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Check for flash attention support
    try:
        import flash_attn
        attn_implementation = "flash_attention_2" if cuda and bf16_supported else "sdpa"
    except ImportError:
        attn_implementation = "sdpa"
        
    print(f"      Using Attention Implementation: {attn_implementation}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if bf16_supported else (torch.float16 if cuda else torch.float32),
        device_map="auto" if cuda else "cpu",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True, 
        attn_implementation=attn_implementation,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if cuda:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # --- Apply LoRA ---
    print(f"\n[3/4] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training configuration ---
    print(f"\n[4/4] Configuring the trainer...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=args.learning_rate,
        fp16=cuda and not bf16_supported,
        bf16=bf16_supported,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0, # Set to 0 to prevent multiprocessing crashes in Colab
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        neftune_noise_alpha=5.0, # NEFTune noise for better generalization
    )

    # --- Train ---
    print(f"\nStarting the training process...")
    print(f"  Epochs:           {args.num_epochs}")
    print(f"  Batch size:       1 (effective: {8})")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Output dir:       {output_dir}\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn(processor),
    )

    print("Training the model...")
    trainer.train()

    # --- Save adapter weights ---
    print("Saving the adapter weights...")
    adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    
    print(f"\n[Done] LoRA adapter successfully saved to: {adapter_path}")

    # Save training summary
    summary = {
        "base_model": MODEL_ID,
        "dataset": DATASET_ID,
        "num_epochs": args.num_epochs,
        "training_samples": len(train_dataset),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "adapter_path": str(adapter_path),
        "optimizations_used": {
            "bfloat16": bf16_supported,
            "flash_attention_2": attn_implementation == "flash_attention_2",
            "lazy_dataset": True,
            "neftune_noise": 5.0
        }
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {output_dir / 'training_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line options for local or Colab training runs."""
    parser = argparse.ArgumentParser(description="Optimized LoRA fine-tune MedGemma 1.5")
    parser.add_argument("--output_dir", default="models/medgemma-1v5-4b-it-rsna23-abd-ct-peft-lora-r16-a32-ep3-lr2e4-v1",
                        help="Directory to save adapter weights")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max training examples")
    parser.add_argument("--num_slices", type=int, default=3,
                        help="Axial slices per NIfTI volume")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
