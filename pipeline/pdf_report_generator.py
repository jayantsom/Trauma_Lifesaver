"""Final PDF export layer for completed Trauma Lifesaver results.

This module only consumes already-generated result data. It does not run or
modify the ML inference pipeline, report writer, PubMed agent, or risk logic.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REVIEW_NOTE = "AI-assisted report. Requires physician/radiologist review."


def _text(value: Any, default: str = "Not provided") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _paragraph_text(text: str) -> str:
    escaped = escape(text or "")
    escaped = escaped.replace("\n", "<br/>")
    return escaped


def _section_title(title: str, styles) -> list:
    return [
        Spacer(1, 0.14 * inch),
        Paragraph(title, styles["SectionHeading"]),
        HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#9fd7c1"), spaceBefore=1, spaceAfter=6),
    ]


def _body_block(text: str, styles) -> Paragraph:
    return Paragraph(_paragraph_text(text or "Not available"), styles["BodyClinical"])


def _kv_table(rows: list[tuple[str, Any]], styles) -> Table:
    data = [[Paragraph(f"<b>{escape(label)}</b>", styles["TableCell"]), Paragraph(escape(_text(value)), styles["TableCell"])] for label, value in rows]
    table = Table(data, colWidths=[1.85 * inch, 4.55 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef6f2")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5d1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _risk_color(risk_level: str) -> colors.Color:
    risk = (risk_level or "").upper()
    if risk == "HIGH":
        return colors.HexColor("#fee2e2")
    if risk == "MODERATE":
        return colors.HexColor("#fef3c7")
    if risk == "LOW":
        return colors.HexColor("#dcfce7")
    return colors.HexColor("#f1f5f9")


def _metric_cards(rows: list[tuple[str, Any]], styles, risk_level: str = "") -> Table:
    data = []
    for label, value in rows:
        data.append([
            Paragraph(escape(label.upper()), styles["MetricLabel"]),
            Paragraph(escape(_text(value)), styles["MetricValue"]),
        ])
    table = Table(data, colWidths=[1.65 * inch, 4.75 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (1, 0), (1, 0), _risk_color(risk_level)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d7e3dd")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7e3dd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _split_report_sections(text: str) -> list[tuple[str | None, str]]:
    headings = [
        "CLINICAL INDICATION", "FINDINGS", "AAST GRADING", "IMPRESSION",
        "PHYSICIAN ACTIONS", "EAST RECOMMENDATION", "LABS & IMAGING",
        "ORIGINAL AI FINDING SUMMARY", "HEMORRHAGE LOCATION AND SEVERITY",
        "VOLUME AND RISK INTERPRETATION", "PUBMED RESEARCH SUPPORT",
        "CLINICAL CONSIDERATIONS", "MODEL LIMITATIONS",
    ]
    pattern = re.compile(rf"^({'|'.join(re.escape(h) for h in headings)})\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return [(None, text or "Not available")]
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1), (text[start:end] or "").strip()))
    return sections


def _report_blocks(text: str, styles) -> list:
    blocks = []
    for heading, body in _split_report_sections(text):
        if heading:
            blocks.append(Paragraph(escape(heading.title()), styles["SubHeading"]))
        blocks.append(_body_block(body, styles))
        blocks.append(Spacer(1, 0.04 * inch))
    return blocks


def _citation_blocks(citations: list[dict], styles) -> list:
    if not citations:
        return [_body_block("PubMed research support unavailable at this time.", styles)]
    blocks = []
    for i, article in enumerate(citations, 1):
        title = escape(article.get("title") or "Untitled article")
        meta_rows = [
            ("Journal", article.get("journal")),
            ("Year", article.get("year")),
            ("PMID", article.get("pmid")),
            ("PubMed URL", article.get("url")),
            ("Why relevant", article.get("why_relevant")),
        ]
        blocks.append(KeepTogether([
            Paragraph(f"{i}. <b>{title}</b>", styles["CitationTitle"]),
            _kv_table(meta_rows, styles),
            Spacer(1, 0.08 * inch),
        ]))
    return blocks


def format_citations(citations: list[dict]) -> str:
    """Format real PubMed citations already returned by the research agent."""
    if not citations:
        return "PubMed research support unavailable at this time."
    lines = []
    for i, article in enumerate(citations, 1):
        lines.extend([
            f"{i}. {article.get('title') or 'Untitled article'}",
            f"   Journal: {article.get('journal') or 'Not available'}",
            f"   Year: {article.get('year') or 'Not available'}",
            f"   PMID: {article.get('pmid') or 'Not available'}",
            f"   PubMed URL: {article.get('url') or 'Not available'}",
            f"   Why relevant: {article.get('why_relevant') or 'Relevant to retrieved clinical context.'}",
            "",
        ])
    return "\n".join(lines).strip()


def build_pdf_context(result_data: dict) -> dict:
    """Normalize completed analysis result data for PDF rendering."""
    quant = result_data.get("quantification") or {}
    triage = result_data.get("triage") or {}
    visual = result_data.get("visual_findings") or {}
    patient_info = result_data.get("patient_info") or {}
    vitals = result_data.get("vitals") or {}
    structured = result_data.get("structured_report") or {}

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "patient_id": result_data.get("patient_id") or "Not provided",
        "patient_age": patient_info.get("age") or structured.get("patient_age") or "Not provided",
        "clinical_state": patient_info.get("state") or "Not provided",
        "clinical_notes": patient_info.get("clinical_notes") or structured.get("clinical_report", ""),
        "vitals": vitals,
        "scan_summary": {
            "total_slices": triage.get("total_slices", "Not available"),
            "suspicious_slices": triage.get("suspicious_count", "Not available"),
            "max_triage_score": triage.get("max_score", "Not available"),
            "visual_pattern": visual.get("injury_pattern", "Not available"),
            "organs": ", ".join(visual.get("organs_involved") or []) or "None identified",
        },
        "risk_summary": {
            "risk_level": quant.get("risk_level", "Not available"),
            "hemorrhage_volume_ml": quant.get("volume_ml", "Not available"),
            "voxel_count": quant.get("num_voxels", "Not available"),
            "east_recommendation": quant.get("recommendation", "Not available"),
        },
        "clinical_report": result_data.get("clinical_report") or result_data.get("report") or "Not available",
        "agentic_explanation": result_data.get("research_enhanced_report") or "Not available",
        "citations": result_data.get("citations") or [],
        "model_limitations": structured.get("limitations") or visual.get("raw_response") or "AI outputs require clinical review and correlation with source imaging.",
    }


def _draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#53635c"))
    canvas.drawString(doc.leftMargin, 0.42 * inch, REVIEW_NOTE)
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(result_data: dict, output_path: str | None = None) -> str:
    """Generate a unified PDF report from an already-completed result object."""
    ctx = build_pdf_context(result_data)
    if output_path is None:
        safe_patient = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ctx["patient_id"])
        output_path = str(Path(tempfile.gettempdir()) / f"trauma_lifesaver_{safe_patient}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.white,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#047857"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SubHeading",
        parent=styles["Heading4"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#065f46"),
        spaceBefore=5,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="BodyClinical",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        name="MetricLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64746d"),
    ))
    styles.add(ParagraphStyle(
        name="MetricValue",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    ))
    styles.add(ParagraphStyle(
        name="CitationTitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    ))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title="Trauma Lifesaver CT Analysis Report",
    )

    header = Table(
        [[
            Paragraph("Trauma Lifesaver CT Analysis Report", styles["ReportTitle"]),
            Paragraph(
                f"<font color='white'><b>Generated</b><br/>{escape(ctx['generated_at'])}<br/><br/>"
                f"<b>Patient ID</b><br/>{escape(ctx['patient_id'])}</font>",
                styles["TableCell"],
            ),
        ]],
        colWidths=[4.35 * inch, 2.05 * inch],
        hAlign="LEFT",
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#064e3b")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#064e3b")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story = [header, Spacer(1, 0.18 * inch)]

    story += _section_title("Patient Information", styles)
    story.append(_kv_table([
        ("Patient ID", ctx["patient_id"]),
        ("Age", ctx["patient_age"]),
        ("Clinical State", ctx["clinical_state"]),
        ("Heart Rate", ctx["vitals"].get("hr")),
        ("Blood Pressure", ctx["vitals"].get("bp")),
        ("GCS", ctx["vitals"].get("gcs")),
    ], styles))

    story += _section_title("Scan Summary", styles)
    story.append(_kv_table([
        ("Total CT Slices", ctx["scan_summary"]["total_slices"]),
        ("Suspicious Slices", ctx["scan_summary"]["suspicious_slices"]),
        ("Max Triage Score", ctx["scan_summary"]["max_triage_score"]),
        ("Visual Pattern", ctx["scan_summary"]["visual_pattern"]),
        ("Organs", ctx["scan_summary"]["organs"]),
    ], styles))

    story += _section_title("Risk Summary", styles)
    story.append(_metric_cards([
        ("Risk Level", ctx["risk_summary"]["risk_level"]),
        ("Hemorrhage Volume", f"{ctx['risk_summary']['hemorrhage_volume_ml']} mL"),
        ("Voxel Count", ctx["risk_summary"]["voxel_count"]),
        ("EAST Recommendation", ctx["risk_summary"]["east_recommendation"]),
    ], styles, str(ctx["risk_summary"]["risk_level"])))

    story += _section_title("Clinical Structured Report", styles)
    story.extend(_report_blocks(ctx["clinical_report"], styles))

    story += _section_title("Agentic Clinical Explanation", styles)
    story.extend(_report_blocks(ctx["agentic_explanation"], styles))

    story += _section_title("Suggested Journal Articles / PubMed Citations", styles)
    story.extend(_citation_blocks(ctx["citations"], styles))

    story += _section_title("Model Limitations", styles)
    story.append(_body_block(ctx["model_limitations"], styles))

    story += _section_title("Clinical Review Note", styles)
    story.append(_body_block(REVIEW_NOTE, styles))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output_path


def create_pdf_download_response(pdf_path: str):
    """Create a Flask download response for a generated PDF file."""
    from flask import send_file

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=Path(pdf_path).name,
        mimetype="application/pdf",
    )
