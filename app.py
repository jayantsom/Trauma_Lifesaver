"""Flask entrypoint for the Trauma Lifesaver web app.

This file keeps the HTTP layer deliberately thin: it accepts uploads, starts
analysis jobs in the background, and returns completed pipeline results to the
browser. The model and reporting logic lives under ``pipeline/``.
"""

import threading
import time as _time
import os
import uuid

import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from pipeline.orchestrator import TraumaPipeline
from pipeline.pdf_report_generator import create_pdf_download_response, generate_pdf_report

app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

pipeline: TraumaPipeline = None


def allowed_file(filename: str) -> bool:
    """Check the upload extension before saving user-provided files."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def load_models():
    """Initialize the shared pipeline once before the Flask server starts."""
    global pipeline
    print(f"[app.py] CUDA available: {torch.cuda.is_available()}")
    pipeline = TraumaPipeline()


@app.route("/")
def index():
    return render_template("index.html")


# The app is single-user/demo oriented, so an in-memory job map is enough here.
_jobs: dict = {}
_JOBS_TTL = 1800


def _cleanup_jobs():
    """Drop old jobs so long-running demos do not keep growing memory usage."""
    cutoff = _time.time() - _JOBS_TTL
    stale = [jid for jid, j in _jobs.items() if j.get("ts", 0) < cutoff]
    for jid in stale:
        del _jobs[jid]


@app.route("/upload", methods=["POST"])
def upload():
    """Save uploaded slices and start a background analysis job."""
    if pipeline is None:
        return jsonify({"success": False, "error": "Models not loaded yet."}), 503

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        single = request.files.get("file")
        if single and single.filename:
            files = [single]
        else:
            return jsonify({"success": False, "error": "No files uploaded."}), 400

    image_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            image_paths.append(filepath)

    if not image_paths:
        return jsonify({"success": False, "error": "No valid image files found."}), 400

    vitals = {k: request.form.get(k) for k in ("hr", "bp", "gcs")}
    vitals = {k: v for k, v in vitals.items() if v}
    patient_info = {
        "age": request.form.get("age"),
        "state": request.form.get("clinical_state"),
        "clinical_notes": request.form.get("clinical_notes"),
    }
    patient_info = {k: v for k, v in patient_info.items() if v}
    patient_id = request.form.get("patient_id") or f"PT-{uuid.uuid4().hex[:6].upper()}"

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "ts": _time.time()}

    def _run():
        try:
            result = pipeline.run_pipeline(image_paths, vitals, patient_id, patient_info)
            _jobs[job_id] = {"status": "done", "result": result, "ts": _time.time()}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _jobs[job_id] = {"status": "error", "error": "GPU out of memory.", "ts": _time.time()}
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            _jobs[job_id] = {"status": "error", "error": str(e), "ts": _time.time()}
        finally:
            _cleanup_jobs()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "processing"})

@app.route("/status/<job_id>")
def job_status(job_id):
    """Return job progress or final analysis output for polling clients."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job not found or expired."}), 404
    if job["status"] == "done":
        return jsonify({"status": "done", "success": True, "result": job["result"]})
    if job["status"] == "error":
        return jsonify({"status": "error", "success": False, "error": job["error"]})
    return jsonify({"status": "processing"})


@app.route("/download-report/<job_id>")
def download_report(job_id):
    """Generate a PDF for a completed job and stream it to the browser."""
    job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"success": False, "error": "PDF report generation unavailable at this time."}), 404
    try:
        pdf_path = generate_pdf_report(job["result"])
        return create_pdf_download_response(pdf_path)
    except Exception:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": "PDF report generation unavailable at this time."}), 500


@app.route("/qa-stream")
def qa_stream():
    """Stream clinical Q&A tokens over Server-Sent Events."""
    if pipeline is None:
        return jsonify({"error": "Models not loaded."}), 503
    session_id = request.args.get("session_id", "")
    question = request.args.get("q", "").strip()

    if not session_id or not question:
        return jsonify({"error": "session_id and q required."}), 400

    def sse_data(token) -> str:
        """Encode multiline text as valid SSE data lines."""
        text = str(token).replace("\r\n", "\n").replace("\r", "\n")
        return "".join(f"data: {line}\n" for line in text.split("\n")) + "\n"

    def generate():
        try:
            for token in pipeline.run_layer5_qa_stream(session_id, question):
                yield sse_data(token)
        except Exception as e:
            yield sse_data(f"Error: {str(e)}")
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/health")
def health():
    """Small readiness endpoint used by Docker, Colab, and local testing."""
    if pipeline is None:
        return jsonify({"status": "loading", "models_loaded": False}), 503
    return jsonify({"status": "ok", **pipeline.get_status()})


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(traceback.format_exc())
    return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File too large (max 100 MB)."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    load_models()
    port = config.PORT
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
