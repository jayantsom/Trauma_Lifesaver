import json
import torch
import threading
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import config

class CTVisualAnalyzer:
    """Layer 2: MedGemma 1.5 multimodal analysis of CT slices."""
    
    MODEL_ID = config.MEDGEMMA_MODEL_ID

    def __init__(self, device="auto", use_4bit=True, hf_token=None, lora_adapter=None):
        token = hf_token or config.HF_TOKEN
        cuda_available = torch.cuda.is_available()

        bnb_config = None
        if use_4bit and cuda_available:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"[Layer 2 - CTVisualAnalyzer] Loading {self.MODEL_ID} with 4-bit NF4...")
        else:
            print(f"[Layer 2 - CTVisualAnalyzer] Loading {self.MODEL_ID} without quantization...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16 if cuda_available else torch.float32,
            device_map=device if cuda_available else "cpu",
            quantization_config=bnb_config,
            token=token,
        )
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=token)

        self.lora_adapter = lora_adapter or config.LORA_ADAPTER
        if self.lora_adapter:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter)
                print(f"[Layer 2 - CTVisualAnalyzer] LoRA adapter loaded: {self.lora_adapter}")
            except Exception as e:
                print(f"[Layer 2 - CTVisualAnalyzer] WARNING: Failed to load LoRA: {e}")

        self.model.eval()
        self._device = next(self.model.parameters()).device
        
        self.max_qa_slices = 10
        if cuda_available:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gb < 20:
                self.max_qa_slices = 3
            elif total_vram_gb < 32:
                self.max_qa_slices = 6

        print(f"[Layer 2 - CTVisualAnalyzer] Ready on {self._device}. Max slices: {self.max_qa_slices}")

    def run_visual_analysis(self, pil_images: list, vitals: dict = None, patient_info: dict = None) -> dict:
        slices = pil_images[: self.max_qa_slices]
        content = self._build_content(slices, vitals, patient_info)
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=config.VISUAL_ANALYSIS_MAX_TOKENS,
                do_sample=False,
            )

        raw = self.processor.decode(generation[0][input_len:], skip_special_tokens=True)
        return self._parse_findings(raw)

    def _build_content(self, slices: list, vitals: dict = None, patient_info: dict = None) -> list:
        content = []
        for i, img in enumerate(slices):
            content.append({"type": "image", "image": img.convert("RGB")})
            content.append({"type": "text", "text": f"[CT Slice {i + 1} of {len(slices)}]"})

        context_parts = []
        if patient_info:
            if patient_info.get("age"):   context_parts.append(f"Age {patient_info['age']} years")
            if patient_info.get("state"): context_parts.append(f"Clinical state: {patient_info['state']}")
        if vitals:
            parts = [f"{k.upper()} {v}" for k, v in vitals.items() if v]
            if parts: context_parts.append(f"Vitals: {', '.join(parts)}")

        vitals_text = ("\nPatient context: " + "; ".join(context_parts) + ".") if context_parts else ""

        content.append({"type": "text", "text": config.visual_analysis_prompt(len(slices), vitals_text)})
        return content

    def _parse_findings(self, raw: str) -> dict:
        try:
            parsed = json.loads(raw.strip())
            if isinstance(parsed, dict):
                self._fill_defaults(parsed, raw)
                return parsed
        except json.JSONDecodeError:
            pass

        start = raw.find('{')
        if start != -1:
            depth = 0
            for i, ch in enumerate(raw[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(raw[start:i + 1])
                            if isinstance(parsed, dict):
                                self._fill_defaults(parsed, raw)
                                return parsed
                        except json.JSONDecodeError:
                            break

        # Aggressive text extraction fallback when JSON completely fails
        raw_lower = raw.lower()

        severity = "unknown"
        if any(w in raw_lower for w in ["no acute", "no injury", "unremarkable", "normal"]): severity = "none"
        elif "severe" in raw_lower: severity = "severe"
        elif "moderate" in raw_lower: severity = "moderate"
        elif "mild" in raw_lower: severity = "mild"

        # Extract injury pattern from first non-empty line
        import re
        injury_pattern = "Unable to determine from image"
        for line in raw.strip().split("\n"):
            line = line.strip().lstrip('{"').strip()
            if len(line) > 20 and not line.startswith(("{", "}", "[", "You", "CT", "Slice")):
                injury_pattern = line[:200].rstrip(',": ')
                break

        # Extract organs mentioned
        organ_keywords = ["liver", "spleen", "kidney", "bowel", "mesentery", "bladder",
                          "pancreas", "aorta", "stomach", "lung", "colon", "rectum"]
        organs = [o.capitalize() for o in organ_keywords if o in raw_lower]

        # Extract bleeding
        bleeding = "Not identified"
        for pat in [r'bleeding[:\s]+([^.\n]+)', r'hemorrhage[:\s]+([^.\n]+)',
                    r'hemoperitoneum[:\s]+([^.\n]+)', r'free fluid[:\s]+([^.\n]+)']:
            m = re.search(pat, raw_lower)
            if m:
                bleeding = raw[m.start(1):m.start(1)+150].strip()
                break

        return {
            "injury_pattern": injury_pattern,
            "organs_involved": organs,
            "bleeding_description": bleeding,
            "severity_estimate": severity,
            "differential_diagnosis": [],
            "raw_response": raw,
        }

    @staticmethod
    def _fill_defaults(parsed: dict, raw: str):
        parsed.setdefault("injury_pattern", "Unable to determine")
        parsed.setdefault("organs_involved", [])
        parsed.setdefault("bleeding_description", "Not identified")
        parsed.setdefault("severity_estimate", "unknown")
        parsed.setdefault("differential_diagnosis", [])
        parsed["raw_response"] = raw
