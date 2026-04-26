# HuggingFace Spaces Deployment Dockerfile
# Hardware target: T4 Small (16 GB VRAM)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads

EXPOSE 7860

ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV HF_HUB_CACHE=/tmp/huggingface/hub
ENV TORCH_HOME=/tmp/torch
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

CMD ["python", "app.py"]
