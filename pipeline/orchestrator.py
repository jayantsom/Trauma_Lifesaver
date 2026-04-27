import time
import uuid
import numpy as np
import torch
from PIL import Image

from pipeline.layer_1_ct_triager import CTTriager
from pipeline.layer_2_ct_analyzer import CTVisualAnalyzer
from pipeline.layer_3_hemorrhage_segmenter import HemorrhageSegmenter
from pipeline.layer_4_report_writer import ClinicalReportWriter
from pipeline.layer_5_qa_streamer import QAStreamer
from pipeline.quantifier import quantify_hemorrhage
import config

class TraumaPipeline:
    """Orchestrates all 5 layers of the Trauma Lifesaver analysis."""

    def __init__(self):
        cuda_available = torch.cuda.is_available()
        
        self.triager = CTTriager(device="cpu")
        self.visual_analyzer = CTVisualAnalyzer(
            device="auto" if cuda_available else "cpu",
            use_4bit=cuda_available
        )
        self.segmenter = HemorrhageSegmenter(
            device="cuda" if cuda_available else "cpu"
        )
        self.report_writer = ClinicalReportWriter(self.visual_analyzer)
        self.qa_streamer = QAStreamer(self.visual_analyzer)
        
        self._sessions = {}

    def run_pipeline(self, image_paths: list, vitals: dict = None, patient_id: str = None, patient_info: dict = None) -> dict:
        session_id = str(uuid.uuid4())
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        # Layer 1
        suspicious_images, all_triage = self.triager.get_top_suspicious(pil_images)
        triage_summary = self.triager.summarize_triage(all_triage)

        # Layer 2 — top suspicious slices only (capped to avoid OOM)
        try:
            visual_findings = self.visual_analyzer.run_visual_analysis(
                suspicious_images, vitals
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            visual_findings = self.visual_analyzer.run_visual_analysis(
                suspicious_images[:1], vitals
            )

        # Layer 3 — U-Net segments ALL uploaded slices (one at a time, memory-safe)
        masks = []
        for path in image_paths:
            try:
                res = self.segmenter.segment_slice(path)
                masks.append(res["mask"])
            except:
                masks.append(np.zeros((512, 512), dtype=np.uint8))
        combined_mask = np.stack(masks, axis=0) if masks else np.zeros((1, 512, 512), dtype=np.uint8)
        quant_res = quantify_hemorrhage(combined_mask)


        # Free GPU memory before Layer 4 (report synthesis reuses MedGemma)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        context = {
            **visual_findings,
            **quant_res,
            "vitals": vitals or {},
            "patient_info": patient_info or {},
            "clinical_notes": (patient_info or {}).get("clinical_notes", ""),
            "triage_summary": triage_summary,
            "patient_id": patient_id,
        }

        # Layer 4
        report = self.report_writer.write_report(context)

        result = {
            "session_id": session_id,
            "patient_id": patient_id,
            "triage": triage_summary,
            "visual_findings": visual_findings,
            "quantification": quant_res,
            "report": report,
            "vitals": vitals or {},
        }

        self._sessions[session_id] = {
            "images": suspicious_images,
            "context": context,
            "timestamp": time.time(),
        }
        self._prune_sessions()
        return result

    def run_layer5_qa_stream(self, session_id: str, question: str):
        session = self._sessions.get(session_id)
        if not session:
            yield "Session expired. Please re-upload."
            return
        yield from self.qa_streamer.stream_qa_response(question, session["context"], session["images"])

    def get_status(self):
        cuda = torch.cuda.is_available()
        return {
            "models_loaded": True,
            "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
            "sessions": len(self._sessions),
        }

    def _prune_sessions(self):
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s["timestamp"] > config.SESSION_TTL]
        for sid in expired: del self._sessions[sid]
