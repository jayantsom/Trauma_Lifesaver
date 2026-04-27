import re
import threading
import torch
from transformers import TextIteratorStreamer
import config

class QAStreamer:
    """Layer 5: Real-time clinical Q&A using MedGemma streaming."""

    def __init__(self, visual_analyzer):
        self.va = visual_analyzer

    def stream_qa_response(self, question: str, context: dict, pil_images: list):
        slices = pil_images[: self.va.max_qa_slices]
        context_summary = config.qa_context_summary(context)

        content = []
        for i, img in enumerate(slices):
            content.append({"type": "image", "image": img.convert("RGB")})
            content.append({"type": "text", "text": f"[Slice {i + 1}]"})

        content.append({"type": "text", "text": (
            f"{context_summary}\n"
            f"You are a trauma radiologist. Answer the following clinical question "
            f"accurately and in detail based on these CT slices and the analysis above. "
            f"Use proper medical terminology. Do not repeat yourself.\n\n"
            f"Question: {question}"
        )})

        messages = [{"role": "user", "content": content}]
        inputs = self.va.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.va._device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.va.processor.tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": config.QA_MAX_TOKENS,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.92,
            "repetition_penalty": 1.3,
        }

        def generation_task():
            try:
                self.va.model.generate(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                streamer.text_queue.put("\n\n[Warning: GPU OOM. Context too large.]")
                streamer.text_queue.put(streamer.stop_signal)
            except Exception as e:
                streamer.text_queue.put(f"\n\n[Error: {str(e)}]")
                streamer.text_queue.put(streamer.stop_signal)

        thread = threading.Thread(target=generation_task)
        thread.start()

        # Accumulate a small prefix to strip any leading thinking block,
        # then stream the remainder token-by-token.
        prefix_buf = ""
        prefix_done = False
        for token in streamer:
            if not prefix_done:
                prefix_buf += token
                # Once we have enough text to detect (or rule out) a thinking block:
                if len(prefix_buf) > 20 or '<unused' not in prefix_buf:
                    # Strip any <unusedXX>...</unusedXX> thinking block from prefix
                    cleaned = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', prefix_buf).strip()
                    if cleaned:
                        yield cleaned
                    prefix_done = True
            else:
                yield token

        # Flush any remaining prefix if stream ended before threshold
        if not prefix_done and prefix_buf:
            cleaned = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', prefix_buf).strip()
            if cleaned:
                yield cleaned

        thread.join()
