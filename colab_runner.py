# ============================================================
# Trauma Lifesaver — Google Colab Runner
# Paste each block into a separate Colab cell and run in order
# Runtime → Change runtime type → T4 GPU
# ============================================================

# ── CELL 1: Clone repo from GitHub ──────────────────────────
# Replace with YOUR GitHub repo URL after you push
!git clone https://github.com/YOUR_USERNAME/Trauma_Lifesaver.git
%cd Trauma_Lifesaver

# ── CELL 2: Install dependencies ────────────────────────────
!pip install -q -r requirements.txt

# ── CELL 3: Set secrets ──────────────────────────────────────
import os

# Your HuggingFace token (must have read access to the gated models)
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"

# Optional: enable LoRA adapter
os.environ["LORA_ADAPTER"] = "AryanMarwah/medgemma-trauma-lora"

# Enable ngrok tunnel
os.environ["USE_NGROK"] = "true"

# Your ngrok token from https://dashboard.ngrok.com/get-started/your-authtoken
os.environ["NGROK_TOKEN"] = "YOUR_NGROK_TOKEN_HERE"

# ── CELL 4: Run the app ──────────────────────────────────────
# The public ngrok URL will be printed below — open it in your browser
!python app.py
