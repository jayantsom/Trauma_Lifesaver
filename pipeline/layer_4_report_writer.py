import re
import torch
import config


# ── Template builders (deterministic, no Gemma) ─────────────────────────────

def _build_findings(ctx: dict, shock_class: str) -> str:
    triage   = ctx.get("triage_summary", {})
    sus      = triage.get("suspicious_count", "?")
    total    = triage.get("total_slices", "?")
    max_sc   = triage.get("max_score", 0)
    organs_l = ctx.get("organs_involved", [])
    injury_p = ctx.get("injury_pattern", "")
    bleeding = ctx.get("bleeding_description", "")
    volume   = ctx.get("volume_ml", 0)
    risk     = ctx.get("risk_level", "LOW")
    severity = ctx.get("severity_estimate", "unknown")
    top_label = ctx.get("top_triage_label", "")

    parts = [
        f"MedSigLIP triage flagged {sus}/{total} CT slices as suspicious for "
        f"intraabdominal pathology (peak suspicion score {max_sc:.2f}/1.00)."
    ]
    if top_label:
        parts.append(f"Primary automated classification: \"{top_label}\".")
    if injury_p and "Unable" not in injury_p and "raw response" not in injury_p:
        parts.append(f"Visual analysis: {injury_p}.")
    if organs_l:
        parts.append(f"Organs with potential involvement: {', '.join(organs_l)}.")
    if bleeding and bleeding not in ("Not identified", "None", ""):
        parts.append(f"Hemorrhage characterization: {bleeding}.")
    parts.append(
        f"Quantitative hemorrhage estimation: {volume:.1f} mL ({shock_class}), "
        f"{risk} risk tier, overall severity assessed as {severity}."
    )
    return " ".join(parts)


def _build_aast(organs_l: list, volume: float, risk: str) -> str:
    if risk == "HIGH" or volume > 500:
        grade, rationale = "III–IV", "significant hemorrhagic burden"
    elif risk == "MODERATE" or volume > 150:
        grade, rationale = "II–III", "moderate hemorrhage volume"
    else:
        grade, rationale = "I–II", "low hemorrhage volume"

    organ_grade_map = {
        "Liver":     ("Hepatic laceration",   grade),
        "Spleen":    ("Splenic laceration",   grade),
        "Kidney":    ("Renal laceration",     grade),
        "Bowel":     ("Bowel injury",         "I–II"),
        "Bladder":   ("Bladder injury",       "I–II"),
        "Pancreas":  ("Pancreatic injury",    "I–II"),
        "Aorta":     ("Aortic injury",        grade),
        "Mesentery": ("Mesenteric injury",    "I–II"),
        "Stomach":   ("Gastric injury",       "I"),
        "Lung":      ("Pulmonary contusion",  "I–II"),
        "Colon":     ("Colonic injury",       "I–II"),
        "Rectum":    ("Rectal injury",        "I"),
    }

    lines = []
    for organ in organs_l:
        desc, g = organ_grade_map.get(organ, (f"{organ} injury", grade))
        lines.append(f"- {organ} ({desc}): Grade {g} — {rationale}.")
    if not lines:
        lines.append(
            f"- No solid organ injury clearly delineated on automated analysis "
            f"(hemorrhage {volume:.0f} mL, {risk} risk)."
        )
    return "\n".join(lines)


def _deloop(text: str, min_len: int = 55, max_repeats: int = 2) -> str:
    parts = re.split(r'(?<=[.!?])\s+', text)
    seen: dict = {}
    kept = []
    for p in parts:
        key = p.strip().lower()
        if len(key) >= min_len:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > max_repeats:
                break
        kept.append(p)
    return ' '.join(kept).strip()


def _extract_section(text: str, start_marker: str, end_markers: list) -> str:
    """Extract text between start_marker and the first end_marker found."""
    pattern = re.compile(re.escape(start_marker), re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return ""
    content_start = m.end()
    content_end = len(text)
    for end in end_markers:
        em = re.compile(re.escape(end), re.IGNORECASE).search(text, content_start)
        if em:
            content_end = min(content_end, em.start())
    return text[content_start:content_end].strip()


# ── Main writer class ────────────────────────────────────────────────────────

class ClinicalReportWriter:
    """Layer 4: Formal report builder.

    Structure:
      CLINICAL INDICATION  — Python (patient notes)
      FINDINGS             — Python template (triage + visual + volume)
      AAST GRADING         — Python template (per-organ grade)
      IMPRESSION           — Gemma (2-3 sentences)   ┐
      PHYSICIAN ACTIONS    — Gemma (2 bullet points)  ├ one call, 250 tokens
      LABS & IMAGING       — Gemma (2 lines)          ┘
      EAST RECOMMENDATION  — Config lookup
    """

    def __init__(self, visual_analyzer):
        self.va = visual_analyzer

    def write_report(self, context: dict) -> str:
        try:
            return self._build_report(context)
        except Exception as e:
            import traceback
            print(f"[Layer 4] Report failed: {e}")
            print(traceback.format_exc())
            return self._template_fallback(context)

    # ── Assembler ─────────────────────────────────────────────────────────

    def _build_report(self, ctx: dict) -> str:
        risk       = ctx.get("risk_level", "LOW")
        east_rec   = config.EAST_RECOMMENDATIONS.get(risk, config.EAST_RECOMMENDATIONS["LOW"])
        shock      = config.get_shock_class(ctx.get("volume_ml", 0))
        volume     = ctx.get("volume_ml", 0)
        indication = ctx.get("clinical_notes") or "Abdominal CT angiogram for trauma evaluation."
        organs_l   = ctx.get("organs_involved", [])
        severity   = ctx.get("severity_estimate", "unknown")
        vitals     = ctx.get("vitals") or {}
        vitals_str = ", ".join(f"{k.upper()} {v}" for k, v in vitals.items() if v) or "Not provided"

        findings = _build_findings(ctx, shock)
        aast     = _build_aast(organs_l, volume, risk)

        # One Gemma call → fills IMPRESSION + PHYSICIAN ACTIONS + LABS
        gemma_out = self._gemma_clinical_sections(
            ctx, shock, risk, volume, severity, organs_l, vitals_str
        )
        impression = gemma_out.get("impression") or self._fallback_impression(volume, shock, risk, severity, organs_l)
        actions    = gemma_out.get("actions")    or f"• Immediate trauma surgery consultation.\n• Hemodynamic monitoring and IV access."
        labs       = gemma_out.get("labs")       or "Immediate: CBC, BMP, coagulation panel, lactate, type & screen.\nFollow-up: Repeat CT at 24–48 h."

        return (
            f"CLINICAL INDICATION\n{indication}\n\n"
            f"FINDINGS\n{findings}\n\n"
            f"AAST GRADING\n{aast}\n\n"
            f"IMPRESSION\n{impression}\n\n"
            f"PHYSICIAN ACTIONS\n{actions}\n\n"
            f"EAST RECOMMENDATION\n{east_rec}\n\n"
            f"LABS & IMAGING\n{labs}"
        )

    # ── Gemma: one bounded call for 3 sections ────────────────────────────

    def _gemma_clinical_sections(self, ctx, shock, risk, volume, severity, organs_l, vitals_str) -> dict:
        organs_str = ", ".join(organs_l) if organs_l else "no specific organ identified"
        top_label  = ctx.get("top_triage_label", "") or "intraabdominal trauma"
        injury_p   = ctx.get("injury_pattern", "") or "indeterminate"
        east_rec   = config.EAST_RECOMMENDATIONS.get(risk, "")

        # Fill-in-the-blank template — model completes each labeled slot
        prompt = (
            "You are a trauma radiologist. Complete the 3 labeled sections below.\n"
            "Write ONLY the text for each section. Stop immediately when all 3 are done.\n\n"
            f"PATIENT DATA:\n"
            f"CT label: {top_label}\n"
            f"Visual finding: {injury_p}\n"
            f"Organs: {organs_str}\n"
            f"Hemorrhage: {volume:.1f} mL ({shock}) | Risk: {risk} | Severity: {severity}\n"
            f"Vitals: {vitals_str}\n"
            f"EAST guideline: {east_rec}\n\n"
            "---\n\n"
            "IMPRESSION\n"
            "Write 2-3 sentences on: (1) clinical significance of findings, "
            "(2) hemorrhage severity and risk, (3) immediate management priority.\n\n"
            "PHYSICIAN ACTIONS\n"
            "Write exactly 2 bullet points (start each with •) on: most urgent action, second priority.\n\n"
            "LABS\n"
            "Immediate: list the specific urgent labs for this case\n"
            "Follow-up: imaging or monitoring recommendation\n"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.va.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        inputs    = {k: v.to(self.va._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                out = self.va.model.generate(
                    **inputs,
                    max_new_tokens=280,
                    do_sample=True,
                    temperature=0.25,
                    top_p=0.92,
                    repetition_penalty=1.3,
                )
            raw = self.va.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
            raw = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', raw).strip()
            raw = _deloop(raw)
            print(f"[Layer 4] Gemma raw:\n{raw[:500]}")
            return self._parse_gemma_sections(raw)
        except Exception as e:
            print(f"[Layer 4] Gemma call failed: {e}")
            return {}

    def _parse_gemma_sections(self, raw: str) -> dict:
        # Path A: header-based extraction (model repeated the section headers)
        impression = _extract_section(raw, "IMPRESSION", ["PHYSICIAN ACTIONS", "PHYSICIAN NOTE", "LABS"])
        actions    = _extract_section(raw, "PHYSICIAN ACTIONS", ["LABS", "EAST", "IMPRESSION"])
        if not actions:
            actions = _extract_section(raw, "PHYSICIAN NOTE", ["LABS", "EAST"])
        labs = _extract_section(raw, "LABS", ["EAST", "IMPRESSION", "PHYSICIAN"])

        # Path B: model output content without headers (common for chat models)
        if not impression and not actions and not labs:
            lines = raw.split('\n')
            bullet_lines = [l.strip() for l in lines if l.strip().startswith('•')]
            imm_lines    = [l.strip() for l in lines if re.match(r'immediate[:\s]', l, re.IGNORECASE)]
            fup_lines    = [l.strip() for l in lines if re.match(r'follow.?up[:\s]', l, re.IGNORECASE)]
            prose_paras  = [p.strip() for p in raw.split('\n\n')
                            if p.strip()
                            and not p.strip().startswith('•')
                            and not re.match(r'immediate[:\s]', p.strip(), re.IGNORECASE)]

            if prose_paras:
                impression = prose_paras[0]
            if bullet_lines:
                actions = '\n'.join(bullet_lines[:2])
            lab_parts = imm_lines[:1] + fup_lines[:1]
            if lab_parts:
                labs = '\n'.join(lab_parts)

        # Clean any accidental section bleed-in from impression
        if impression:
            impression = re.sub(r'\n?(PHYSICIAN|LABS|EAST|CLINICAL)[^\n]*',
                                '', impression, flags=re.IGNORECASE).strip()

        return {
            "impression": impression or None,
            "actions":    actions    or None,
            "labs":       labs       or None,
        }

    @staticmethod
    def _fallback_impression(volume, shock, risk, severity, organs_l) -> str:
        organs = ", ".join(organs_l) if organs_l else "no specific organ"
        return (
            f"CT angiogram demonstrates traumatic intraabdominal hemorrhage quantified at "
            f"{volume:.1f} mL ({shock}), consistent with a {risk.lower()} risk profile and "
            f"{severity} injury severity. Automated analysis identified potential involvement of {organs}. "
            f"Immediate trauma surgery evaluation and hemodynamic monitoring are recommended."
        )

    # ── Full template fallback ────────────────────────────────────────────

    def _template_fallback(self, ctx: dict) -> str:
        risk       = ctx.get("risk_level", "LOW")
        east_rec   = config.EAST_RECOMMENDATIONS.get(risk, config.EAST_RECOMMENDATIONS["LOW"])
        shock      = config.get_shock_class(ctx.get("volume_ml", 0))
        organs_l   = ctx.get("organs_involved", [])
        volume     = ctx.get("volume_ml", 0)
        severity   = ctx.get("severity_estimate", "unknown")
        indication = ctx.get("clinical_notes") or "Abdominal CT angiogram for trauma evaluation."
        return (
            f"CLINICAL INDICATION\n{indication}\n\n"
            f"FINDINGS\n{_build_findings(ctx, shock)}\n\n"
            f"AAST GRADING\n{_build_aast(organs_l, volume, risk)}\n\n"
            f"IMPRESSION\n{ClinicalReportWriter._fallback_impression(volume, shock, risk, severity, organs_l)}\n\n"
            f"PHYSICIAN ACTIONS\n• Immediate trauma surgery consultation.\n• Hemodynamic monitoring and resuscitation.\n\n"
            f"EAST RECOMMENDATION\n{east_rec}\n\n"
            f"LABS & IMAGING\nImmediate: CBC, BMP, PT/INR, aPTT, type & screen, lactate.\nFollow-up: Repeat CT at 24–48 h; serial abdominal exams every 4–6 h."
        )
