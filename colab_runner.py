# ============================================================
# Trauma Lifesaver — Google Colab Runner
# Paste each block as a SEPARATE Colab cell and run in order.
# Runtime → Change runtime type → T4 GPU (before running)
# ============================================================

# ── CELL 1: Clone repo ────────────────────────────────────────
# !git clone https://github.com/YOUR_USERNAME/Trauma_Lifesaver.git
# %cd Trauma_Lifesaver

# ── CELL 2: Install dependencies + cloudflared ───────────────
# !pip install -q -r requirements.txt
# !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
# !chmod +x cloudflared

# ── CELL 3: Set HuggingFace token ────────────────────────────
# import os
# os.environ["HF_TOKEN"]     = "hf_YOUR_TOKEN_HERE"
# os.environ["LORA_ADAPTER"] = "AryanMarwah/medgemma-trauma-lora"   # optional

# ── CELL 4: Launch app + get public URL ──────────────────────
# Paste everything below (from import to the last break) into ONE Colab cell.
#
# import subprocess, threading, time, re, urllib.request
#
# flask_proc = subprocess.Popen(
#     ["python", "app.py"],
#     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
# )
#
# def _stream():
#     for line in flask_proc.stdout:
#         print(line.decode(), end="", flush=True)
# threading.Thread(target=_stream, daemon=True).start()
#
# print("Waiting for models to load (first run ~10-15 min)...")
# for _ in range(600):
#     try:
#         urllib.request.urlopen("http://localhost:7860/health", timeout=2)
#         break
#     except Exception:
#         time.sleep(2)
# else:
#     print("ERROR: Flask did not start - check output above.")
#
# tunnel = subprocess.Popen(
#     ["./cloudflared", "tunnel", "--url", "http://localhost:7860"],
#     stderr=subprocess.PIPE, stdout=subprocess.PIPE,
# )
# for raw in tunnel.stderr:
#     line = raw.decode()
#     m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
#     if m:
#         print(f"\n==> Open in your browser: {m.group(0)}\n")
#         break
