import os
import re

# ── Models ─────────────────────────────────────────────────────────────────
MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
MEDGEMMA_MODEL_ID  = "google/medgemma-1.5-4b-it"
MEDGEMMA_LOCAL_FILES_ONLY = os.environ.get("MEDGEMMA_LOCAL_FILES_ONLY", "true").lower() == "true"
MEDGEMMA_CPU_DTYPE = os.environ.get("MEDGEMMA_CPU_DTYPE", "bfloat16").lower()
CPU_SAFE_MODE = os.environ.get("CPU_SAFE_MODE", "true").lower() == "true"

HF_TOKEN     = os.environ.get("HF_TOKEN")
LORA_ADAPTER = os.environ.get("LORA_ADAPTER")  # e.g. "AryanMarwah/medgemma-trauma-lora"

# ── Layer 1 — Triage ────────────────────────────────────────────────────────
TRIAGE_IMAGE_SIZE       = 448
TRIAGE_THRESHOLD        = float(os.environ.get("TRIAGE_THRESHOLD", "0.25"))
TRIAGE_MAX_SLICES       = 1   # top-1 slice to Gemma — avoids OOM and tunnel timeout

TRIAGE_LABELS = [
    "CT scan showing intraabdominal hemorrhage or active bleeding",
    "CT scan with liver laceration, splenic injury, or solid organ trauma",
    "CT scan showing hemoperitoneum or free fluid in the abdomen",
    "Normal CT scan of the abdomen without hemorrhage or injury",
    "CT scan with bowel perforation or mesenteric injury",
]
TRIAGE_POSITIVE_INDICES = [0, 1, 2, 4]  # index 3 = "Normal" is negative

# ── Layer 2 — Visual Analysis ───────────────────────────────────────────────
VISUAL_ANALYSIS_MAX_TOKENS = 800

def visual_analysis_prompt(n: int, vitals_text: str = "") -> str:
    return (
        f"You are a trauma radiologist analyzing {n} abdominal CT angiogram slice(s).{vitals_text}\n\n"
        f"Examine ALL {n} images collectively. "
        "Identify: active hemorrhage, hemoperitoneum, solid organ injury (liver, spleen, kidneys), "
        "vascular injury, bowel/mesenteric injury.\n\n"
        "CRITICAL: Respond with ONLY a raw JSON object. "
        "No text before or after. No FINDINGS, IMPRESSION, or markdown. "
        "Start with { and end with }.\n\n"
        "{\n"
        '  "injury_pattern": "<one concise sentence>",\n'
        '  "organs_involved": ["<organ>"],\n'
        '  "bleeding_description": "<location and extent, or none>",\n'
        '  "severity_estimate": "<none|mild|moderate|severe>",\n'
        '  "differential_diagnosis": ["<dx1>", "<dx2>", "<dx3>"]\n'
        "}"
    )

# ── Layer 4 — Report Synthesis ──────────────────────────────────────────────
REPORT_MAX_TOKENS = 1024

def report_synthesis_prompt(ctx: dict, east_rec: str, shock_class: str, vitals_str: str) -> str:
    triage      = ctx.get("triage_summary", {})
    organs_list = ctx.get("organs_involved", [])
    organs      = ", ".join(organs_list) if organs_list else "None clearly identified on automated analysis"
    diffs       = ctx.get("differential_diagnosis", [])
    differentials = ", ".join(diffs) if diffs else "Not listed"
    indication  = ctx.get("clinical_notes") or "Abdominal CT angiogram for trauma evaluation."
    injury_pat  = ctx.get("injury_pattern", "")
    bleeding    = ctx.get("bleeding_description", "")
    severity    = ctx.get("severity_estimate", "unknown")
    volume      = ctx.get("volume_ml", 0)
    risk        = ctx.get("risk_level", "LOW")
    sus_count   = triage.get("suspicious_count", "?")
    total_sl    = triage.get("total_slices", "?")
    max_sc      = triage.get("max_score", 0)

    # Construct FINDINGS text from hard data regardless of Layer 2 success
    findings_parts = []
    findings_parts.append(
        f"MedSigLIP triage identified {sus_count}/{total_sl} slices as suspicious "
        f"for intraabdominal pathology (max suspicion score {max_sc:.2f}/1.00)."
    )
    if injury_pat and "Unable" not in injury_pat and "raw response" not in injury_pat:
        findings_parts.append(f"Automated visual analysis: {injury_pat}.")
    if organs_list:
        findings_parts.append(f"Organs with potential involvement: {organs}.")
    if bleeding and "Not identified" not in bleeding:
        findings_parts.append(f"Hemorrhage characterization: {bleeding}.")
    findings_parts.append(
        f"Quantitative hemorrhage analysis estimates {volume:.1f} mL of hemorrhagic volume "
        f"({shock_class}) with a {risk} risk tier."
    )
    if severity not in ("unknown", "none", ""):
        findings_parts.append(f"Overall injury severity assessed as {severity}.")
    if vitals_str and vitals_str != "Not provided":
        findings_parts.append(f"Presenting vitals: {vitals_str}.")
    findings_text = " ".join(findings_parts)

    return (
        "You are a trauma radiology attending. Continue this formal CT report.\n"
        "Use sophisticated medical terminology. Be specific and clinically accurate.\n"
        "Do NOT restate the instructions. Do NOT output reasoning. Continue directly from AAST GRADING.\n\n"
        f"=== CONTEXT FOR AAST GRADING ===\n"
        f"Organs involved: {organs}\n"
        f"Hemorrhage volume: {volume:.1f} mL ({shock_class})\n"
        f"Risk tier: {risk} | Severity: {severity}\n"
        f"Differentials: {differentials}\n"
        f"EAST recommendation: {east_rec}\n\n"
        "=== REPORT SO FAR (DO NOT REWRITE) ===\n\n"
        f"CLINICAL INDICATION\n{indication}\n\n"
        f"FINDINGS\n{findings_text}\n\n"
        "=== CONTINUE THE REPORT BELOW ===\n\n"
        "AAST GRADING\n"
    )


# ── Layer 5 — Q&A ───────────────────────────────────────────────────────────
QA_MAX_TOKENS = 400

def qa_context_summary(ctx: dict) -> str:
    bleeding = ctx.get("bleeding_description", "N/A")
    if len(bleeding) > 150 or "FINDINGS" in bleeding.upper() or "IMPRESSION" in bleeding.upper():
        bleeding = "See initial analysis"

    injury = ctx.get("injury_pattern", "N/A")
    # Strip report-header prefixes that Layer 2 fallback text sometimes includes
    injury = re.sub(r'^(FINDINGS|IMPRESSION|ASSESSMENT)[:\s]*', '', injury, flags=re.IGNORECASE).strip()
    if len(injury) > 200:
        injury = injury[:200] + "..."

    return (
        f"Patient scan summary:\n"
        f"- Injury pattern: {injury}\n"
        f"- Organs involved: {', '.join(ctx.get('organs_involved', [])) or 'None identified'}\n"
        f"- Bleeding: {bleeding}\n"
        f"- Severity: {ctx.get('severity_estimate', 'N/A')}\n"
        f"- Hemorrhage volume: {ctx.get('volume_ml', 0):.1f} mL\n"
        f"- Risk level: {ctx.get('risk_level', 'N/A')}\n"
    )

# ── EAST Guidelines ─────────────────────────────────────────────────────────
EAST_RECOMMENDATIONS = {
    "LOW": (
        "Non-operative management (NOM) is appropriate. "
        "Serial abdominal exams every 4–6 hours. "
        "Maintain hemodynamic stability monitoring. "
        "Repeat CT in 24–48 hours if clinical concern."
    ),
    "MODERATE": (
        "Consider angiography with selective embolization (SAE) if hemodynamically stable. "
        "If hemodynamically unstable despite resuscitation, proceed to OR. "
        "Massive transfusion protocol (MTP) activation if ongoing hemorrhage. "
        "Trauma surgery consultation required."
    ),
    "HIGH": (
        "EMERGENT intervention required. "
        "If hemodynamically unstable: immediate operative exploration. "
        "If transiently stable: angioembolization as bridge or definitive therapy. "
        "Activate massive transfusion protocol (1:1:1 ratio pRBC:FFP:PLT). "
        "REBOA (Zone III) may be considered for pelvic hemorrhage. "
        "Damage control surgery principles apply."
    ),
}

# ── Volume / Risk ───────────────────────────────────────────────────────────
def get_risk_level(volume_ml: float) -> str:
    if volume_ml < 10:
        return "LOW"
    elif volume_ml < 500:
        return "MODERATE"
    return "HIGH"

def get_shock_class(volume_ml: float) -> str:
    if volume_ml < 750:
        return "ATLS Class I"
    elif volume_ml < 1500:
        return "ATLS Class II"
    elif volume_ml < 2000:
        return "ATLS Class III"
    return "ATLS Class IV"

# ── App ─────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = "uploads"
MAX_CONTENT_LENGTH = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
SESSION_TTL        = 1800
PORT               = int(os.environ.get("PORT", 7860))
DEFAULT_SPACING    = (0.5, 0.5, 3.0)  # mm — typical abdominal CT
