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
                max_new_tokens=800,
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

        report_raw = self.va.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

        import re
        # Strip MedGemma thinking tokens
        report_raw = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', report_raw).strip()

        # The prompt pre-built CLINICAL INDICATION + FINDINGS and asked Gemma to start at AAST GRADING.
        # Reconstruct: prepend the pre-built block, then append Gemma's continuation.
        indication = ctx.get("clinical_notes") or "Abdominal CT angiogram for trauma evaluation."

        # Re-derive the findings text (same logic as prompt builder)
        triage    = ctx.get("triage_summary", {})
        organs_l  = ctx.get("organs_involved", [])
        volume    = ctx.get("volume_ml", 0)
        risk      = ctx.get("risk_level", "LOW")
        severity  = ctx.get("severity_estimate", "unknown")
        injury_p  = ctx.get("injury_pattern", "")
        bleeding  = ctx.get("bleeding_description", "")
        sus       = triage.get("suspicious_count", "?")
        total     = triage.get("total_slices", "?")
        max_sc    = triage.get("max_score", 0)

        fp = [f"MedSigLIP triage identified {sus}/{total} slices suspicious (max score {max_sc:.2f}/1.00)."]
        if injury_p and "Unable" not in injury_p and "raw response" not in injury_p:
            fp.append(f"Automated visual analysis: {injury_p}.")
        if organs_l:
            fp.append(f"Organs with potential involvement: {', '.join(organs_l)}.")
        if bleeding and "Not identified" not in bleeding:
            fp.append(f"Hemorrhage characterization: {bleeding}.")
        fp.append(f"Quantitative hemorrhage volume: {volume:.1f} mL ({shock_class}) — {risk} risk tier.")
        if severity not in ("unknown", "none", ""):
            fp.append(f"Overall injury severity: {severity}.")
        findings_text = " ".join(fp)

        # Find where Gemma's AAST GRADING output starts (strip any accidental preamble)
        m = re.search(r'AAST GRADING', report_raw, re.IGNORECASE)
        gemma_body = report_raw[m.start():].strip() if m else report_raw.strip()

        report = (
            f"CLINICAL INDICATION\n{indication}\n\n"
            f"FINDINGS\n{findings_text}\n\n"
            f"{gemma_body}"
        )

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
