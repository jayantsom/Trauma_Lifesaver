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

app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

pipeline: TraumaPipeline = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def load_models():
    global pipeline
    print(f"[app.py] CUDA available: {torch.cuda.is_available()}")
    pipeline = TraumaPipeline()

@app.route("/")
def index():
    return render_template("index.html")

import threading
import time as _time

# In-memory job store: job_id -> {status, result, error, ts}
_jobs: dict = {}
_JOBS_TTL = 1800  # 30 min

def _cleanup_jobs():
    cutoff = _time.time() - _JOBS_TTL
    stale = [jid for jid, j in _jobs.items() if j.get("ts", 0) < cutoff]
    for jid in stale:
        del _jobs[jid]

@app.route("/upload", methods=["POST"])
def upload():
    if pipeline is None: return jsonify({"success": False, "error": "Models not loaded yet."}), 503

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        single = request.files.get("file")
        if single and single.filename: files = [single]
        else: return jsonify({"success": False, "error": "No files uploaded."}), 400

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
        "age":            request.form.get("age"),
        "state":          request.form.get("clinical_state"),
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
            import traceback; print(traceback.format_exc())
            _jobs[job_id] = {"status": "error", "error": str(e), "ts": _time.time()}
        finally:
            _cleanup_jobs()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "processing"})

@app.route("/status/<job_id>")
def job_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job not found or expired."}), 404
    if job["status"] == "done":
        return jsonify({"status": "done", "success": True, "result": job["result"]})
    if job["status"] == "error":
        return jsonify({"status": "error", "success": False, "error": job["error"]})
    return jsonify({"status": "processing"})


@app.route("/qa-stream")
def qa_stream():
    if pipeline is None: return jsonify({"error": "Models not loaded."}), 503
    session_id = request.args.get("session_id", "")
    question = request.args.get("q", "").strip()

    if not session_id or not question: return jsonify({"error": "session_id and q required."}), 400

    def generate():
        try:
            for token in pipeline.run_layer5_qa_stream(session_id, question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/health")
def health():
    if pipeline is None: return jsonify({"status": "loading", "models_loaded": False}), 503
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

    if os.environ.get("USE_NGROK", "").lower() == "true":
        try:
            from pyngrok import ngrok
            ngrok_token = os.environ.get("NGROK_TOKEN")
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            public_url = ngrok.connect(port)
            print(f"\n[ngrok] Public URL: {public_url}\n")
        except ImportError:
            print("[ngrok] pyngrok not installed — run: pip install pyngrok")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
