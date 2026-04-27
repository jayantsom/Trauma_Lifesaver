import torch
import config

class ClinicalReportWriter:
    """Layer 4: Synthesize EAST-aligned formal report."""

    def __init__(self, visual_analyzer):
        self.va = visual_analyzer

    def write_report(self, context: dict) -> str:
        try:
            return self._medgemma_synthesis(context)
        except Exception as e:
            print(f"[Layer 4 - ClinicalReportWriter] Synthesis failed: {e}. Using template.")
            return self._template_fallback(context)

    def _medgemma_synthesis(self, ctx: dict) -> str:
        risk = ctx.get("risk_level", "LOW")
        east_rec = config.EAST_RECOMMENDATIONS.get(risk, config.EAST_RECOMMENDATIONS["LOW"])
        shock_class = config.get_shock_class(ctx.get("volume_ml", 0))

        vitals = ctx.get("vitals") or {}
        vitals_str = ", ".join(f"{k.upper()} {v}" for k, v in vitals.items() if v) or "Not provided"

        prompt = config.report_synthesis_prompt(ctx, east_rec, shock_class, vitals_str)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = self.va.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.va._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Clear residual GPU memory from earlier layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            from peft import PeftModel
            use_disable = isinstance(self.va.model, PeftModel)
        except ImportError:
            use_disable = False

        def _generate():
            return self.va.model.generate(
                **inputs,
                max_new_tokens=700,   # reduced from 1024 for speed
                do_sample=False,
            )

        try:
            with torch.inference_mode():
                if use_disable:
                    with self.va.model.disable_adapter():
                        output = _generate()
                else:
                    output = _generate()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU OOM during report synthesis — falling back to template.")

        report = self.va.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

        import re
        # Strip MedGemma thinking tokens: <unused94>...<unused95>
        report = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', report).strip()

        # Find CLINICAL INDICATION anywhere in the output (no ^ anchor — thinking tag may precede it)
        m = re.search(r'CLINICAL INDICATION', report, re.IGNORECASE)
        if m:
            report = report[m.start():].strip()
        else:
            report = "CLINICAL INDICATION\n" + report

        return report


    def _template_fallback(self, ctx: dict) -> str:
        risk = ctx.get("risk_level", "LOW")
        east_rec = config.EAST_RECOMMENDATIONS.get(risk, config.EAST_RECOMMENDATIONS["LOW"])
        shock_class = config.get_shock_class(ctx.get("volume_ml", 0))
        organs = ", ".join(ctx.get("organs_involved", [])) or "None identified"
        volume = ctx.get("volume_ml", 0)

        return (
            f"CLINICAL INDICATION\nTrauma evaluation.\n\n"
            f"FINDINGS\n"
            f"Pattern: {ctx.get('injury_pattern', 'N/A')}\n"
            f"Organs: {organs}\n"
            f"Hemorrhage: {volume:.1f} mL ({shock_class})\n\n"
            f"IMPRESSION\n{ctx.get('severity_estimate', 'UNKNOWN').upper()} injury. Risk: {risk}.\n\n"
            f"EAST RECOMMENDATION\n{east_rec}\n\n"
            f"LABS & IMAGING\nCBC, BMP, type & screen. Repeat imaging based on clinical status."
        )
