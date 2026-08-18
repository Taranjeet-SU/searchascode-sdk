"""OpenAI (gpt-4.1-mini) wrapper shared by the rephraser, tool-calling baseline,
and the SAC code-gen agent. Tracks token usage and USD cost for the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phase1 import common

common.load_env()


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, prompt_toks: int, completion_toks: int, cached: int = 0) -> None:
        self.input_tokens += prompt_toks - cached
        self.cached_input_tokens += cached
        self.output_tokens += completion_toks
        self.calls += 1

    @property
    def cost_usd(self) -> float:
        p = common.LLM_PRICE
        return (
            self.input_tokens * p["input"]
            + self.cached_input_tokens * p["cached_input"]
            + self.output_tokens * p["output"]
        ) / 1_000_000

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class LLM:
    """Thin OpenAI chat client with usage accounting."""

    def __init__(self, model: str = common.LLM_MODEL, temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.usage = Usage()

    def chat(self, messages: list[dict], tools: list | None = None, **kw):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            tools=tools, **kw,
        )
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        self.usage.add(u.prompt_tokens, u.completion_tokens, cached)
        return resp

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.model.startswith("qwen3"):
            prompt = prompt + " /no_think"           # Qwen3 soft switch: no <think> preamble
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        text = self.chat(msgs).choices[0].message.content or ""
        if "<think>" in text:                        # strip any residual reasoning block
            import re as _re
            text = _re.sub(r"<think>.*?</think>\s*", "", text, flags=_re.DOTALL)
        return text

    def as_generator(self):
        """Adapt to the SDK's ``generate(prompt) -> list[str]`` contract
        (for rephrase / expand / decompose primitives)."""
        def generate(prompt: str) -> list[str]:
            text = self.complete(prompt)
            return [ln.strip("-*0123456789. \t") for ln in text.splitlines() if ln.strip()]
        return generate
