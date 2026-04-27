import re
import time as _time
import torch
import config


def _strip_thinking(text: str) -> str:
    """Remove MedGemma thinking blocks. Two-pass: paired tags first, then unpaired."""
    text = re.sub(r'<unused\d+>[\s\S]*?<unused\d+>', '', text)  # paired <unusedX>...</unusedY>
    text = re.sub(r'<unused\d+>[\s\S]*', '', text)               # unpaired: strip to end
    return text.strip()


def _clean_qa_output(text: str) -> str:
    """Strip section headers and boilerplate the model might prepend."""
    # Remove any report-style headers the model echoes
    text = re.sub(
        r'^(FINDINGS|IMPRESSION|ASSESSMENT|RECOMMENDATION|ANSWER|CLINICAL\s+REASONING)[:\s]*',
        '', text, flags=re.IGNORECASE
    ).strip()
    # Remove disclaimer sentences
    text = re.sub(
        r'(?:I cannot|I am unable|please consult|disclaimer|note that I|as an AI)[^.!?]*[.!?]',
        '', text, flags=re.IGNORECASE
    ).strip()
    return text


class QAStreamer:
    """Layer 5: Clinical Q&A — non-streaming generate + thinking strip + re-stream.

    Why non-streaming:
      MedGemma generates a thinking block first (~150 tokens) before the actual answer.
      With streaming we cannot strip the thinking block mid-generation.
      We generate the full response, strip thinking + cleanup, then re-stream the
      cleaned text word-by-word so the UI still shows a streaming effect.
    """

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
            "You are a trauma radiologist. Answer this question in 2-3 sentences.\n"
            "Be direct, specific, and brief. No disclaimers. No section headers.\n\n"
            f"Question: {question}"
        )})

        messages = [{"role": "user", "content": content}]
        inputs = self.va.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        inputs    = {k: v.to(self.va._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                out = self.va.model.generate(
                    **inputs,
                    max_new_tokens=config.QA_MAX_TOKENS,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.92,
                    repetition_penalty=1.3,
                )

            raw = self.va.processor.decode(
                out[0][input_len:], skip_special_tokens=True
            ).strip()

            # Strip thinking tokens (two-pass for paired and unpaired opening tags)
            raw = _strip_thinking(raw)

            # Strip report headers and disclaimers
            raw = _clean_qa_output(raw)

            print(f"[Layer 5] QA answer ({len(raw)} chars): {raw[:200]}")

            if not raw:
                raw = "Insufficient data to answer this question from the current scan."

            # Re-stream word-by-word for UX (8 words per chunk)
            words = raw.split(' ')
            for i in range(0, len(words), 8):
                chunk = ' '.join(words[i:i + 8])
                if i + 8 < len(words):
                    chunk += ' '
                yield chunk
                _time.sleep(0.015)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            yield "[Warning: GPU out of memory. Try with fewer images.]"
        except Exception as e:
            print(f"[Layer 5] QA error: {e}")
            yield f"[Error generating response: {str(e)}]"
