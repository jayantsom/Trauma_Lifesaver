"""Layer 5: post-analysis clinical Q&A.

The chat layer answers only after an analysis has completed. It packages the
report, risk summary, and real PubMed citations so responses stay grounded in
the generated case context.
"""

import re
import time as _time
import json
import os
import urllib.error
import urllib.request
import torch
import config

NOT_ENOUGH_INFO = "I do not have enough information in the generated report to answer that."


def _strip_thinking(text: str) -> str:
    """Remove MedGemma thinking blocks. Two-pass: paired tags first, then unpaired."""
    text = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', text)  # paired <unusedX>...</unusedY>
    text = re.sub(r'<unused\d+>[\s\S]*', '', text)               # unpaired: strip to end
    return text.strip()


def _clean_qa_output(text: str) -> str:
    """Strip section headers and boilerplate the model might prepend."""
    # Remove any report-style headers the model echoes
    text = re.sub(
        r'^(FINDINGS|IMPRESSION|ASSESSMENT|RECOMMENDATION|ANSWER|CLINICAL\s+REASONING)[:\s]*',
        '', text, flags=re.IGNORECASE
    ).strip()
    # Remove disclaimer sentences
    text = re.sub(
        r'(?:I cannot|I am unable|please consult|disclaimer|note that I|as an AI)[^.!?]*[.!?]',
        '', text, flags=re.IGNORECASE
    ).strip()
    return text


def _clean_openai_answer(text: str) -> str:
    """Normalize OpenAI text while preserving the answer wording."""
    text = (text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text or NOT_ENOUGH_INFO


def _text(value) -> str:
    """Convert optional values into clean strings for prompt context."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_section(text: str, heading: str) -> str:
    """Extract one report section by heading from already-generated text."""
    if not text:
        return ""
    pattern = (
        rf"(?:^|\n)\s*{re.escape(heading)}\s*\n"
        rf"([\s\S]*?)(?=\n\s*[A-Z][A-Z /&-]{{3,}}\s*\n|\Z)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_agent_section(text: str, heading: str) -> str:
    """Extract a titled section from the agentic explanation text."""
    if not text:
        return ""
    known = (
        "Original AI Finding Summary",
        "Hemorrhage Location and Severity",
        "Volume and Risk Interpretation",
        "PubMed Research Support",
        "Clinical Considerations",
        "Model Limitations",
    )
    next_headings = "|".join(re.escape(h) for h in known if h.lower() != heading.lower())
    pattern = rf"{re.escape(heading)}\s*([\s\S]*?)(?=\n\s*(?:{next_headings})\s*\n|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _format_structured_report(structured: dict) -> str:
    """Render structured fields into compact text for the Q&A prompt."""
    if not isinstance(structured, dict) or not structured:
        return ""

    fields = [
        ("Hemorrhage detected", structured.get("hemorrhage_detected")),
        ("Hemorrhage type", structured.get("hemorrhage_type")),
        ("Anatomical location", structured.get("anatomical_location")),
        ("Volume", structured.get("volume_ml")),
        ("Risk level", structured.get("risk_level")),
        ("Confidence score", structured.get("confidence_score")),
        ("Suspicious slices", structured.get("suspicious_slices")),
        ("Model findings", structured.get("model_findings")),
        ("Limitations", structured.get("limitations")),
    ]
    lines = []
    for label, value in fields:
        if value in (None, "", [], {}):
            continue
        if label == "Volume":
            value = f"{value} mL"
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    """Limit prompt sections so one large report cannot dominate the request."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[Truncated for Q&A context]"


def format_citation_context(citations) -> str:
    """Format only real PubMed citations already returned by the research agent."""
    if not citations:
        return ""

    lines = []
    for i, article in enumerate(citations, 1):
        title = _text(article.get("title")) or "Untitled article"
        journal = _text(article.get("journal")) or "Journal not provided"
        year = _text(article.get("year")) or "Year not provided"
        pmid = _text(article.get("pmid")) or "PMID not provided"
        url = _text(article.get("url")) or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "PMID not provided" else "")
        why = _text(article.get("why_relevant")) or "Relevance note not provided."
        lines.append(
            f"{i}. {title}\n"
            f"   Journal: {journal}\n"
            f"   Year: {year}\n"
            f"   PMID: {pmid}\n"
            f"   Link: {url}\n"
            f"   Why relevant: {why}"
        )
    return "\n\n".join(lines)


def _openai_text(data: dict) -> str:
    """Extract text from the Responses API shape used by this project."""
    if data.get("output_text"):
        return data["output_text"].strip()
    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def build_chatbot_context(result_data: dict) -> dict:
    """Build the post-analysis Q&A context without changing inference output.

    This is a final context packaging layer for the Clinical Q&A chatbot. It
    consumes already-generated reports, agentic explanation, and PubMed
    citations so answers stay grounded in the completed analysis package.
    """
    result_data = result_data or {}
    clinical_report = _text(result_data.get("clinical_report") or result_data.get("report"))
    structured = result_data.get("structured_report") or {}
    enhanced = _text(result_data.get("research_enhanced_report"))
    citations = result_data.get("citations") or []
    patient_info = dict(result_data.get("patient_info") or {})
    if result_data.get("patient_id"):
        patient_info.setdefault("patient_id", result_data.get("patient_id"))
    if result_data.get("vitals"):
        patient_info.setdefault("vitals", result_data.get("vitals"))
    quant = result_data.get("quantification") or {}
    visual = result_data.get("visual_findings") or {}
    triage = result_data.get("triage") or {}

    risk_summary = {
        "risk_level": quant.get("risk_level") or structured.get("risk_level"),
        "volume_ml": quant.get("volume_ml") or structured.get("volume_ml"),
        "voxel_count": quant.get("num_voxels"),
        "suspicious_slices": triage.get("suspicious_count") or structured.get("suspicious_slices"),
        "max_triage_score": triage.get("max_score") or structured.get("confidence_score"),
        "east_recommendation": _extract_section(clinical_report, "EAST RECOMMENDATION"),
    }

    model_limitations = (
        _text(structured.get("limitations"))
        or _extract_agent_section(enhanced, "Model Limitations")
        or _text(visual.get("injury_pattern") if "skipped" in _text(visual.get("injury_pattern")).lower() else "")
    )

    return {
        "chatbot_context_version": 1,
        "clinical_report": clinical_report,
        "structured_report": structured,
        "structured_report_text": _format_structured_report(structured),
        "research_enhanced_report": enhanced,
        "citations": citations,
        "citation_context": format_citation_context(citations),
        "patient_info": patient_info,
        "risk_summary": risk_summary,
        "model_limitations": model_limitations,
        "clinical_review_note": "AI-assisted report. Requires physician/radiologist review.",
    }


def _risk_summary_text(risk_summary: dict) -> str:
    """Turn the risk payload into a short sentence for fallback answers."""
    if not risk_summary:
        return ""
    lines = []
    if risk_summary.get("volume_ml") not in (None, ""):
        lines.append(f"Hemorrhage volume: {risk_summary.get('volume_ml')} mL.")
    if risk_summary.get("risk_level"):
        lines.append(f"Risk level: {risk_summary.get('risk_level')}.")
    if risk_summary.get("suspicious_slices") not in (None, ""):
        lines.append(f"Suspicious slices: {risk_summary.get('suspicious_slices')}.")
    if risk_summary.get("max_triage_score") not in (None, ""):
        lines.append(f"Maximum triage score: {risk_summary.get('max_triage_score')}.")
    if risk_summary.get("east_recommendation"):
        lines.append(f"EAST recommendation: {risk_summary.get('east_recommendation')}")
    return " ".join(lines)


def _patient_info_text(patient_info: dict) -> str:
    """Format patient metadata for a compact human-readable summary."""
    if not patient_info:
        return ""
    labels = {
        "patient_id": "Patient ID",
        "age": "Age",
        "state": "Clinical state",
        "clinical_notes": "Clinical notes",
        "vitals": "Vitals",
    }
    lines = []
    for k, v in patient_info.items():
        if not v:
            continue
        if isinstance(v, dict):
            v = ", ".join(f"{vk.upper()}: {vv}" for vk, vv in v.items() if vv)
        lines.append(f"{labels.get(k, k.replace('_', ' ').title())}: {v}")
    return "; ".join(lines)


def _build_openai_qa_context(context: dict) -> str:
    """Build the grounded prompt payload used by the OpenAI Q&A call."""
    citations = context.get("citations") or []
    citation_context = context.get("citation_context") or "No PubMed citations were returned."
    payload = {
        "patient_info": context.get("patient_info") or {},
        "risk_summary": context.get("risk_summary") or {},
        "structured_report_fields": context.get("structured_report") or {},
        "model_limitations": context.get("model_limitations") or "",
        "clinical_review_note": context.get("clinical_review_note") or "",
        "citation_count": len(citations),
    }
    return (
        "FINAL ANALYSIS PACKAGE\n\n"
        "Patient / Risk / Structured Fields:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Clinical Structured Report:\n"
        f"{_truncate(context.get('clinical_report') or '', 6000)}\n\n"
        "Agentic Clinical Explanation:\n"
        f"{_truncate(context.get('research_enhanced_report') or '', 6000)}\n\n"
        "Suggested PubMed Articles / Citations:\n"
        f"{_truncate(citation_context, 6000)}"
    )


def _fallback_context_answer(question: str, context: dict) -> str:
    """Small safety fallback if OpenAI is unavailable."""
    question_l = (question or "").lower()
    if any(term in question_l for term in ("journal", "article", "pubmed", "pmid", "citation", "research")):
        citation_context = context.get("citation_context") or ""
        if citation_context:
            return "Based on the suggested PubMed articles:\n\n" + citation_context
    if any(term in question_l for term in ("final diagnosis", "diagnosis")):
        return "Based on the clinical review note, this should not be considered a final diagnosis. It is an AI-assisted report that requires physician/radiologist review."
    if any(term in question_l for term in ("summarize", "summary", "full case", "case")):
        report = _extract_section(context.get("clinical_report") or "", "IMPRESSION")
        risk = _risk_summary_text(context.get("risk_summary") or {})
        if report or risk:
            return "Based on the clinical structured report:\n\n- " + "\n- ".join([x for x in [risk, report] if x])
    return NOT_ENOUGH_INFO


def answer_question_from_context(question: str, context: dict) -> str:
    """Use OpenAI as a grounded clinical Q&A agent over the final report package."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_context_answer(question, context or {})

    context = context or {}
    prompt = (
        "You are the Trauma Lifesaver Clinical Q&A Agent.\n"
        "Answer the user's question using ONLY the final analysis package below.\n\n"
        "Strict grounding rules:\n"
        "- Do not invent findings, recommendations, article titles, journals, years, PMIDs, links, or citations.\n"
        "- If the answer is not available in the package, reply exactly: "
        f"\"{NOT_ENOUGH_INFO}\"\n"
        "- If journal support is requested, use ONLY the Suggested PubMed Articles / Citations section.\n"
        "- Do not provide a definitive diagnosis. If asked whether this is final, state that the report requires physician/radiologist review.\n"
        "- Be concise, clinically professional, and structured.\n"
        "- Use bullets or numbering when the question asks for a list, summary, contradictions, or articles.\n"
        "- For source awareness, start with one of these phrases when applicable:\n"
        "  Based on the clinical structured report...\n"
        "  Based on the enhanced clinical explanation...\n"
        "  Based on the suggested PubMed articles...\n"
        "- If multiple sources are used, say: Based on the final analysis package...\n\n"
        f"{_build_openai_qa_context(context)}\n\n"
        f"User question: {question}"
    )
    model = os.getenv("OPENAI_QA_MODEL", "gpt-4.1-mini")
    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You answer clinical Q&A only from provided report context and provided PubMed citations.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_output_tokens": 900,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = _clean_openai_answer(_openai_text(data))
        if "PMID" in answer and not context.get("citations"):
            return NOT_ENOUGH_INFO
        return answer
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"[Layer 5] OpenAI QA unavailable: {exc}")
        return _fallback_context_answer(question, context)


class QAStreamer:
    """Layer 5: Clinical Q&A — non-streaming generate + thinking strip + re-stream.

    Why non-streaming:
      MedGemma generates a thinking block first (~150 tokens) before the actual answer.
      With streaming we cannot strip the thinking block mid-generation.
      We generate the full response, strip thinking + cleanup, then re-stream the
      cleaned text word-by-word so the UI still shows a streaming effect.
    """

    def __init__(self, visual_analyzer):
        self.va = visual_analyzer

    def stream_qa_response(self, question: str, context: dict, pil_images: list):
        """Yield a complete answer or token-like chunks for the chat UI."""
        if context and context.get("chatbot_context_version"):
            answer = answer_question_from_context(question, context)
            yield answer
            return

        if not hasattr(self.va, "processor") or not hasattr(self.va, "model"):
            yield (
                "Clinical Q&A is running in CPU-safe mode, so MedGemma is not loaded. "
                "The available report is based on triage, quantification, and deterministic clinical templates."
            )
            return

        slices = pil_images[: self.va.max_qa_slices]
        context_summary = config.qa_context_summary(context)

        content = []
        for i, img in enumerate(slices):
            content.append({"type": "image", "image": img.convert("RGB")})
            content.append({"type": "text", "text": f"[Slice {i + 1}]"})

        content.append({"type": "text", "text": (
            f"{context_summary}\n"
            "You are a trauma radiologist. Answer this question in 2-3 sentences.\n"
            "Be direct, specific, and brief. No disclaimers. No section headers.\n\n"
            f"Question: {question}"
        )})

        messages = [{"role": "user", "content": content}]
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
                    max_new_tokens=config.QA_MAX_TOKENS,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.92,
                    repetition_penalty=1.3,
                )

            raw = self.va.processor.decode(
                out[0][input_len:], skip_special_tokens=True
            ).strip()

            # Strip thinking tokens (two-pass for paired and unpaired opening tags)
            raw = _strip_thinking(raw)

            # Strip report headers and disclaimers
            raw = _clean_qa_output(raw)

            print(f"[Layer 5] QA answer ({len(raw)} chars): {raw[:200]}")

            if not raw:
                raw = "Insufficient data to answer this question from the current scan."

            # Re-stream word-by-word for UX (8 words per chunk)
            words = raw.split(' ')
            for i in range(0, len(words), 8):
                chunk = ' '.join(words[i:i + 8])
                if i + 8 < len(words):
                    chunk += ' '
                yield chunk
                _time.sleep(0.015)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            yield "[Warning: GPU out of memory. Try with fewer images.]"
        except Exception as e:
            print(f"[Layer 5] QA error: {e}")
            yield f"[Error generating response: {str(e)}]"
