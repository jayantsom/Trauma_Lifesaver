"""Pipeline coordinator for a single Trauma Lifesaver analysis run.

The orchestrator keeps the layers in order and passes only the needed context
between them. Individual model behavior stays inside the layer modules.
"""

import time
import uuid

import numpy as np
import torch
from PIL import Image

import config
from pipeline.layer_1_ct_triager import CTTriager
from pipeline.layer_2_ct_analyzer import CPUFallbackVisualAnalyzer, CTVisualAnalyzer
from pipeline.layer_3_hemorrhage_segmenter import HemorrhageSegmenter
from pipeline.layer_4_report_writer import ClinicalReportWriter
from pipeline.layer_5_qa_streamer import QAStreamer, build_chatbot_context
from pipeline.quantifier import quantify_hemorrhage
from pipeline.research_agent import run_research_agent


def _aggregate_classification(slice_results: list) -> dict:
    """Aggregate Layer 3 injury-label probabilities across uploaded slices."""
    rows = [r.get("classification") for r in slice_results if r.get("classification")]
    if not rows:
        return {"probabilities": {}, "positive_labels": [], "threshold": 0.5}

    # Max probability answers "did any uploaded slice look positive?", while
    # mean probability remains available for auditing smoother series-level
    # behavior later.
    labels = list(rows[0].keys())
    probabilities = {}
    mean_probabilities = {}
    for label in labels:
        values = [float(row.get(label, 0.0)) for row in rows]
        probabilities[label] = round(max(values), 4)
        mean_probabilities[label] = round(float(np.mean(values)), 4)

    positive_labels = [label for label, prob in probabilities.items() if prob >= 0.5]
    return {
        "probabilities": probabilities,
        "mean_probabilities": mean_probabilities,
        "positive_labels": positive_labels,
        "threshold": 0.5,
    }


class TraumaPipeline:
    """Orchestrates the five analysis layers used by the web app."""

    def __init__(self):
        """Load shared models once so each upload can reuse them."""
        cuda_available = torch.cuda.is_available()

        self.triager = CTTriager(device="cpu")
        if config.CPU_SAFE_MODE and not cuda_available:
            print("[TraumaPipeline] CPU_SAFE_MODE enabled: skipping MedGemma load on CPU.")
            self.visual_analyzer = CPUFallbackVisualAnalyzer()
        else:
            self.visual_analyzer = CTVisualAnalyzer(
                device="auto" if cuda_available else "cpu",
                use_4bit=cuda_available,
            )

        self.segmenter = HemorrhageSegmenter(device="cuda" if cuda_available else "cpu")
        self.report_writer = ClinicalReportWriter(self.visual_analyzer)
        self.qa_streamer = QAStreamer(self.visual_analyzer)
        self._sessions = {}

    def run_pipeline(
        self,
        image_paths: list,
        vitals: dict = None,
        patient_id: str = None,
        patient_info: dict = None,
    ) -> dict:
        """Run all analysis layers for one uploaded CT slice set."""
        session_id = str(uuid.uuid4())
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        # Layer 1: quick image/text screening to pick the most suspicious slice.
        suspicious_images, all_triage = self.triager.get_top_suspicious(pil_images)
        triage_summary = self.triager.summarize_triage(all_triage)

        # Layer 2 uses a capped slice list because MedGemma is the heaviest step.
        top_triage_label = ""
        for r in all_triage:
            if r["suspicious"]:
                top_triage_label = r.get("top_label", "")
                break
        if not top_triage_label and all_triage:
            top_triage_label = all_triage[0].get("top_label", "")

        try:
            visual_findings = self.visual_analyzer.run_visual_analysis(suspicious_images, vitals)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            visual_findings = self.visual_analyzer.run_visual_analysis(suspicious_images[:1], vitals)

        # Layer 3 runs across every uploaded slice, one at a time.
        masks = []
        layer3_results = []
        for path in image_paths:
            try:
                res = self.segmenter.segment_slice(path)
                masks.append(res["mask"])
                layer3_results.append(res)
            except Exception:
                masks.append(np.zeros((512, 512), dtype=np.uint8))

        # Quantification still comes from the mask. The classification payload is
        # carried alongside it for the UI and downstream report context.
        combined_mask = np.stack(masks, axis=0) if masks else np.zeros((1, 512, 512), dtype=np.uint8)
        quant_res = quantify_hemorrhage(combined_mask)
        classification_res = _aggregate_classification(layer3_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        context = {
            **visual_findings,
            **quant_res,
            "classification": classification_res,
            "vitals": vitals or {},
            "patient_info": patient_info or {},
            "clinical_notes": (patient_info or {}).get("clinical_notes", ""),
            "top_triage_label": top_triage_label,
            "triage_summary": triage_summary,
            "patient_id": patient_id,
        }

        # Layer 4 and the research layer consume the same consolidated context.
        report = self.report_writer.write_report(context)
        research_result = run_research_agent({
            "report": report,
            "context": context,
            "triage": triage_summary,
            "visual_findings": visual_findings,
            "quantification": quant_res,
            "classification": classification_res,
        })

        result = {
            "session_id": session_id,
            "patient_id": patient_id,
            "triage": triage_summary,
            "visual_findings": visual_findings,
            "quantification": quant_res,
            "classification": classification_res,
            "report": report,
            "clinical_report": report,
            "structured_report": research_result["structured_report"],
            "research_enhanced_report": research_result["research_enhanced_report"],
            "citations": research_result["citations"],
            "vitals": vitals or {},
            "patient_info": patient_info or {},
        }

        self._sessions[session_id] = {
            "images": suspicious_images,
            "context": build_chatbot_context(result),
            "timestamp": time.time(),
        }
        self._prune_sessions()
        return result

    def get_analysis_context(self, session_id: str):
        """Look up stored context for the chat widget."""
        session = self._sessions.get(session_id)
        return session.get("context") if session else None

    def run_layer5_qa_stream(self, session_id: str, question: str):
        """Stream a clinical Q&A answer for an existing analysis session."""
        session = self._sessions.get(session_id)
        if not session:
            yield "Session expired. Please re-upload."
            return
        yield from self.qa_streamer.stream_qa_response(question, session["context"], session["images"])

    def get_status(self):
        """Return the small health payload used by the Flask endpoint."""
        cuda = torch.cuda.is_available()
        return {
            "models_loaded": True,
            "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
            "sessions": len(self._sessions),
        }

    def _prune_sessions(self):
        """Remove stale chat contexts after the configured session TTL."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s["timestamp"] > config.SESSION_TTL]
        for sid in expired:
            del self._sessions[sid]
