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

    vitals = {
        "hr": request.form.get("hr"),
        "bp": request.form.get("bp"),
        "gcs": request.form.get("gcs"),
    }
    vitals = {k: v for k, v in vitals.items() if v}
    patient_info = {
        "age":           request.form.get("age"),
        "state":         request.form.get("clinical_state"),
        "clinical_notes": request.form.get("clinical_notes"),
    }
    patient_info = {k: v for k, v in patient_info.items() if v}
    patient_id = request.form.get("patient_id") or f"PT-{uuid.uuid4().hex[:6].upper()}"

    try:
        result = pipeline.run_pipeline(image_paths, vitals, patient_id, patient_info)
        return jsonify({"success": True, "result": result})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return jsonify({"success": False, "error": "GPU out of memory."}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
