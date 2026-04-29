"""PubMed-backed post-processing explanation layer.

This module runs only after the existing CT hemorrhage pipeline has produced
its structured clinical report. It does not alter ML inference, segmentation,
quantification, or the original clinical report generation.

TODO: Add local vector database/RAG cache for faster repeated PubMed article retrieval.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
FALLBACK_TEXT = "PubMed research support unavailable at this time."
DISCLAIMER = (
    "This AI-generated report is for clinical decision support only and must be "
    "reviewed by a licensed radiologist or physician. It is not a final diagnosis."
)
ABDOMINAL_TERMS = [
    "abdomen", "abdominal", "intraabdominal", "intra-abdominal", "hemoperitoneum",
    "liver", "hepatic", "spleen", "splenic", "kidney", "renal", "bowel",
    "mesenteric", "solid organ",
]
NEURO_EXCLUSION_TERMS = [
    "brain", "cranio", "craniocerebral", "cerebral", "intracranial", "head injury",
    "traumatic brain injury", "tbi", "neuro", "neuropathological", "neuropathologica",
]


def _section(report: str, name: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(name)}\s*\n(?P<body>.*?)(?=^[A-Z][A-Z &]+$|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(report or "")
    return match.group("body").strip() if match else ""


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _infer_anatomical_location(clinical_report: str, model_findings: str | None, organs: list) -> str | None:
    if organs:
        return ", ".join(organs)
    haystack = f"{clinical_report} {model_findings or ''}".lower()
    matches = []
    if any(term in haystack for term in ("abdomen", "abdominal", "intraabdominal", "intra-abdominal", "hemoperitoneum")):
        matches.append("abdomen")
    if any(term in haystack for term in ("liver", "hepatic")):
        matches.append("liver")
    if any(term in haystack for term in ("spleen", "splenic")):
        matches.append("spleen")
    if any(term in haystack for term in ("kidney", "renal")):
        matches.append("kidney")
    if any(term in haystack for term in ("bowel", "mesenteric")):
        matches.append("bowel/mesentery")
    return ", ".join(dict.fromkeys(matches)) if matches else None


def extract_report_fields(report: Any) -> dict:
    """Extract report/context fields for final evidence retrieval.

    Missing fields remain None or empty so the explanation layer can state
    uncertainty instead of inventing report details.
    """
    payload = report if isinstance(report, dict) else {"report": report}
    clinical_report = payload.get("report") or payload.get("clinical_report") or ""
    visual = payload.get("visual_findings") or {}
    quant = payload.get("quantification") or {}
    triage = payload.get("triage") or payload.get("triage_summary") or {}
    context = payload.get("context") or {}

    volume_ml = quant.get("volume_ml", context.get("volume_ml"))
    risk_level = quant.get("risk_level", context.get("risk_level"))
    suspicious_count = triage.get("suspicious_count")
    total_slices = triage.get("total_slices")
    suspicious_slices = (
        f"{suspicious_count}/{total_slices}"
        if suspicious_count is not None and total_slices is not None
        else None
    )

    bleeding_description = visual.get("bleeding_description") or context.get("bleeding_description")
    model_findings = visual.get("injury_pattern") or context.get("injury_pattern")
    top_triage_label = context.get("top_triage_label")
    organs = visual.get("organs_involved") or context.get("organs_involved") or []
    anatomical_location = _infer_anatomical_location(
        f"{clinical_report} {top_triage_label or ''}",
        model_findings,
        organs,
    )
    hemorrhage_detected = None
    if volume_ml is not None:
        try:
            hemorrhage_detected = float(volume_ml) > 0
        except (TypeError, ValueError):
            hemorrhage_detected = None

    return {
        "clinical_report": clinical_report,
        "hemorrhage_detected": hemorrhage_detected,
        "hemorrhage_type": bleeding_description or None,
        "anatomical_location": anatomical_location,
        "volume_ml": volume_ml,
        "risk_level": risk_level,
        "confidence_score": triage.get("max_score"),
        "suspicious_slices": suspicious_slices,
        "model_findings": model_findings,
        "top_triage_label": top_triage_label,
        "limitations": (
            visual.get("raw_response")
            or context.get("raw_response")
            or "Automated AI output requires clinician review; missing fields indicate uncertainty."
        ),
        "original_findings_section": _section(clinical_report, "FINDINGS"),
        "original_impression_section": _section(clinical_report, "IMPRESSION"),
    }


def build_pubmed_queries(fields: dict) -> list[str]:
    """Build conservative PubMed search queries from extracted report fields."""
    queries: list[str] = []
    location = _clean_text(fields.get("anatomical_location"))
    risk_level = _clean_text(str(fields.get("risk_level") or ""))
    report_text = _clean_text(
        " ".join([
            fields.get("clinical_report") or "",
            fields.get("model_findings") or "",
            fields.get("top_triage_label") or "",
        ])
    ).lower()
    is_abdominal = bool(location) and any(term in location.lower() for term in ABDOMINAL_TERMS)
    is_abdominal = is_abdominal or any(term in report_text for term in ABDOMINAL_TERMS)
    neuro_not = ' NOT ("brain injuries"[MeSH Terms] OR "traumatic brain injury" OR craniocerebral OR intracranial OR neurotrauma)'

    if is_abdominal:
        queries.extend([
            f'("abdominal injuries"[MeSH Terms] OR abdominal trauma OR intraabdominal hemorrhage OR hemoperitoneum) AND (CT OR computed tomography) AND (management OR embolization OR nonoperative){neuro_not}',
            f'(liver injury OR splenic injury OR solid organ injury) AND trauma AND (CT OR computed tomography) AND (embolization OR management){neuro_not}',
        ])
    elif fields.get("hemorrhage_detected") is True:
        queries.append(f"traumatic intraabdominal hemorrhage CT volume management{neuro_not}")
    if location:
        queries.append(f"{location} trauma hemorrhage CT management{neuro_not}")
    if risk_level and is_abdominal:
        queries.append(f"{risk_level} risk abdominal hemorrhage trauma CT prognosis{neuro_not}")
    if not queries:
        queries.append(f"abdominal trauma CT hemorrhage management guidelines{neuro_not}")

    deduped = []
    for query in queries:
        if query.lower() not in {q.lower() for q in deduped}:
            deduped.append(query)
    return deduped


def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed with NCBI ESearch and return PMID strings."""
    api_key = os.getenv("NCBI_API_KEY")
    if not api_key:
        return []

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "relevance",
        "api_key": api_key,
        "tool": "TraumaLifesaverResearchAgent",
    }
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    return [str(pmid) for pmid in data.get("esearchresult", {}).get("idlist", [])]


def _article_text(article: ET.Element, path: str) -> str:
    node = article.find(path)
    if node is None:
        return ""
    return _clean_text("".join(node.itertext()))


def fetch_pubmed_details(pmids: list[str]) -> list[dict]:
    """Fetch PubMed article metadata and abstracts with NCBI EFetch."""
    api_key = os.getenv("NCBI_API_KEY")
    pmids = [str(pmid) for pmid in pmids if pmid]
    if not api_key or not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "api_key": api_key,
        "tool": "TraumaLifesaverResearchAgent",
    }
    url = f"{FETCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            root = ET.fromstring(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ET.ParseError, OSError):
        return []

    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _article_text(article, ".//MedlineCitation/PMID")
        title = _article_text(article, ".//Article/ArticleTitle")
        journal = _article_text(article, ".//Journal/Title")
        year = _article_text(article, ".//JournalIssue/PubDate/Year")
        if not year:
            medline_date = _article_text(article, ".//JournalIssue/PubDate/MedlineDate")
            match = re.search(r"\b(19|20)\d{2}\b", medline_date)
            year = match.group(0) if match else ""
        abstract_parts = [
            _clean_text("".join(node.itertext()))
            for node in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part)
        if pmid and title:
            articles.append({
                "title": title,
                "journal": journal,
                "year": year,
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "abstract": abstract,
            })
    return articles


def rank_articles(fields: dict, articles: list[dict]) -> list[dict]:
    """Rank retrieved articles by report relevance and add why_relevant text."""
    location_terms = [
        term.lower()
        for term in re.split(r"[,;/\s]+", fields.get("anatomical_location") or "")
        if len(term) > 2
    ]
    location = (fields.get("anatomical_location") or "").lower()
    is_abdominal_case = any(term in location for term in ABDOMINAL_TERMS)
    is_abdominal_case = is_abdominal_case or any(
        term in (fields.get("clinical_report") or "").lower()
        for term in ABDOMINAL_TERMS
    )
    keyword_groups = {
        "hemorrhage/trauma": ["hemorrhage", "haemorrhage", "bleeding", "trauma", "traumatic"],
        "CT/imaging": ["ct", "computed tomography", "imaging", "angiography"],
        "management/prognosis": ["management", "embolization", "operative", "nonoperative", "prognosis", "mortality"],
    }
    abdominal_words = ABDOMINAL_TERMS + ["embolization", "angioembolization", "nonoperative", "laparotomy"]

    ranked = []
    for article in articles:
        haystack = " ".join([
            article.get("title", ""),
            article.get("journal", ""),
            article.get("abstract", ""),
        ]).lower()
        score = 0
        reasons = []
        if is_abdominal_case and any(term in haystack for term in NEURO_EXCLUSION_TERMS):
            score -= 10
            reasons.append("penalized because it focuses on neurotrauma rather than abdominal trauma")
        if is_abdominal_case and any(term in haystack for term in abdominal_words):
            score += 8
            reasons.append("matches abdominal/solid-organ trauma context")
        if location_terms and any(term in haystack for term in location_terms):
            score += 4
            reasons.append("matches the reported anatomical location")
        for label, words in keyword_groups.items():
            if any(word in haystack for word in words):
                score += 2
                reasons.append(f"contains {label} concepts")
        try:
            year = int(article.get("year") or 0)
            if year >= 2020:
                score += 2
                reasons.append("recent publication")
            elif year >= 2015:
                score += 1
        except ValueError:
            pass

        if not is_abdominal_case or score > 0:
            ranked.append({
                **article,
                "_score": score,
                "why_relevant": "; ".join([r for r in reasons if not r.startswith("penalized")][:3])
                    or "related to abdominal trauma hemorrhage imaging or management",
            })

    ranked.sort(key=lambda item: item.get("_score", 0), reverse=True)
    return ranked[:3]


def _openai_text(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"].strip()
    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def _clean_explanation_sections(text: str) -> str:
    """Keep citations and repeated disclaimers out of the explanation panel."""
    cleaned = text or ""
    for heading in ("Suggested Relevant Journal Articles", "PubMed citations", "Safety Disclaimer"):
        cleaned = re.sub(
            rf"\n?{re.escape(heading)}\s*:?\s*\n.*?(?=\n(?:Clinical Considerations|Model Limitations|Safety Disclaimer|PubMed citations)\s*:?\s*\n|\Z)",
            "\n",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    cleaned = cleaned.replace(DISCLAIMER, "")
    return cleaned.strip()


def generate_enhanced_report_openai(fields: dict, articles: list[dict]) -> str:
    """Use OpenAI only to summarize retrieved real PubMed articles."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not articles:
        return FALLBACK_TEXT

    article_context = [
        {
            "title": article.get("title", ""),
            "journal": article.get("journal", ""),
            "year": article.get("year", ""),
            "pmid": article.get("pmid", ""),
            "url": article.get("url", ""),
            "abstract": article.get("abstract", ""),
            "why_relevant": article.get("why_relevant", ""),
        }
        for article in articles
    ]
    prompt = (
        "Only summarize the PubMed articles provided below. Do not create or invent citations. "
        "If evidence is limited, clearly state that.\n\n"
        "You are a cautious clinical decision-support explanation assistant. Do not override the "
        "original AI model findings and do not provide a definitive diagnosis.\n\n"
        "Return the report with these exact headings:\n"
        "Enhanced Clinical Explanation Report\n"
        "Original AI Finding Summary\n"
        "Hemorrhage Location and Severity\n"
        "Volume and Risk Interpretation\n"
        "PubMed Research Support\n"
        "Clinical Considerations\n"
        "Model Limitations\n\n"
        "Do not include article titles, numbered article lists, journal names, PMIDs, or links in this report. "
        "Do not include a Safety Disclaimer section. Those citations and the safety disclaimer are displayed "
        "separately elsewhere in the UI. In PubMed Research Support, "
        "summarize the overall retrieved evidence in 2-4 concise sentences only.\n\n"
        f"Structured report fields:\n{json.dumps(fields, indent=2, ensure_ascii=False)}\n\n"
        f"Retrieved PubMed articles:\n{json.dumps(article_context, indent=2, ensure_ascii=False)}"
    )
    model = os.getenv("OPENAI_RESEARCH_AGENT_MODEL", "gpt-4.1-mini")
    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You summarize only provided PubMed evidence for clinical decision support.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_output_tokens": 1200,
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
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return FALLBACK_TEXT

    return _clean_explanation_sections(_openai_text(data)) or FALLBACK_TEXT


def run_research_agent(report: Any) -> dict:
    """Run PubMed retrieval and OpenAI summarization after report generation."""
    try:
        fields = extract_report_fields(report)
        queries = build_pubmed_queries(fields)
        pmids = []
        for query in queries:
            pmids.extend(search_pubmed(query, max_results=5))
        deduped_pmids = list(dict.fromkeys(pmids))
        articles = fetch_pubmed_details(deduped_pmids)
        ranked_articles = rank_articles(fields, articles)
        enhanced = generate_enhanced_report_openai(fields, ranked_articles)
        citations = [
            {
                "title": article.get("title", ""),
                "journal": article.get("journal", ""),
                "year": article.get("year", ""),
                "pmid": article.get("pmid", ""),
                "url": article.get("url", ""),
                "why_relevant": article.get("why_relevant", ""),
            }
            for article in ranked_articles
        ]
        return {
            "structured_report": fields,
            "research_enhanced_report": enhanced or FALLBACK_TEXT,
            "citations": citations,
        }
    except Exception:
        return {
            "structured_report": {},
            "research_enhanced_report": FALLBACK_TEXT,
            "citations": [],
        }
