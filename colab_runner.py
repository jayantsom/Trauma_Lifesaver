"""Notebook helper notes for running Trauma Lifesaver in Google Colab.

This file is intentionally written as commented cells. Copy each section into
Colab in order, choose a T4 GPU runtime, and replace the placeholder secrets
with your own tokens before launching the Flask app.
"""
#CELL 1: Clone the repository.
#import os, subprocess, threading, time, re, urllib.request

#CELL 2: Clone the repository.
#!git clone https://github.com/jayantsom/Trauma_Lifesaver.git
#!%cd Trauma_Lifesaver

#CELL 3: Install dependencies
#!pip install -q -r requirements.txt

#CELL 4: Install cloudflared.
#!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
#!chmod +x cloudflared

#CELL 5: Set required model/API tokens.
#os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
#os.environ["NCBI_API_KEY"] = "YOUR_NCBI_KEY_HERE"
#os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY_HERE"

#CELL 6: Launch Flask and open a temporary public URL.
# flask_proc = subprocess.Popen(
#     ["python", "app.py"],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.STDOUT,
#     bufsize=1,
# )

# def _stream():
#     for line in flask_proc.stdout:
#         print(line.decode(), end="", flush=True)

# threading.Thread(target=_stream, daemon=True).start()

# print("Waiting for models to load. First run can take 10-15 minutes.")
# for _ in range(600):
#     try:
#         urllib.request.urlopen("http://localhost:7860/health", timeout=2)
#         break
#     except Exception:
#         time.sleep(2)
# else:
#     print("ERROR: Flask did not start. Check the logs above.")

# tunnel = subprocess.Popen(
#     ["./cloudflared", "tunnel", "--url", "http://localhost:7860"],
#     stderr=subprocess.PIPE,
#     stdout=subprocess.PIPE,
# )

# for raw in tunnel.stderr:
#     line = raw.decode()
#     match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
#     if match:
#         print(f"\nOpen in your browser: {match.group(0)}\n")
#         break
