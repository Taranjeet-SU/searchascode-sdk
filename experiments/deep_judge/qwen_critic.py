"""An INDEPENDENT local critic — a ~32B Qwen (4-bit) — to rewrite the judge prompt.

The judge stays gpt-4.1-mini (the harness's production judge). Only the CRITIC swaps to Qwen, to test
whether an independent, larger model finds prompt improvements that same-model self-critique misses. The
critic is called ~once per tuning round (a handful of times total), so we load with plain transformers +
bitsandbytes 4-bit — no server needed. GPU memory is capped so we stay a good citizen on the shared card.

Exposes `.complete(prompt, system=...)` matching phase1.llm.LLM so tune_judge can use it interchangeably.
"""
from __future__ import annotations

MODEL = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"   # pre-quantized bnb 4-bit (~19GB), loads without autoawq
GPU_CAP_GIB = 22   # leave headroom on the shared 32GB card for other users


class QwenCritic:
    def __init__(self, model: str = MODEL, max_new_tokens: int = 1200, gpu_cap_gib: int = GPU_CAP_GIB):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map={"": 0}, dtype=torch.float16,
            max_memory={0: f"{gpu_cap_gib}GiB", "cpu": "40GiB"},
        )
        self.model.eval()

    def complete(self, prompt: str, system: str | None = None) -> str:
        import torch
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


if __name__ == "__main__":   # smoke test: load + one generation
    import time
    t0 = time.time()
    c = QwenCritic()
    print(f"[qwen] loaded in {time.time() - t0:.0f}s")
    print(c.complete("Reply with exactly the word: READY", system="You are terse."))
