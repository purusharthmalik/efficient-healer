"""A rollout engine for AgentTune that generates through a vLLM server.

AgentTune keeps the whole agent loop: prompt building, the Aya tool-prompt
fallback, tool-call parsing, tool execution. Only token generation is sent to
an OpenAI-compatible vLLM server, which batches concurrent requests.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from agenttune.agentic.rollout_engines.base import RolloutEngine
from agenttune.agentic.rollout_engines.rollout_factory import (
    _render_chat_template,
    _strip_special_tokens,
)
from openai import BadRequestError, OpenAI
from transformers import AutoTokenizer


class VLLMServerEngine(RolloutEngine):
    def __init__(
        self,
        model_path: str,
        served_model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 384,
        stop: list[str] | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.served_model = served_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        # Optional harness stop rule: end the turn at the end of the first tool-call
        # list or at the closing answer tag, instead of letting the model run on.
        self.stop = stop
        self.client = OpenAI(base_url=base_url, api_key="unused", timeout=300, max_retries=2)
        self._local = threading.local()

    # AgentTune reads the tokenizer to decide whether the chat template renders
    # tools. For Aya it does not, so AgentTune moves the tools into the system prompt.
    def _get_tokenizer(self):
        return self.tokenizer

    # Per-thread counters: one trajectory runs entirely inside one thread.
    def begin(self) -> None:
        self._local.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "gen_seconds": 0.0}

    def usage(self) -> dict[str, Any]:
        return dict(getattr(self._local, "usage", {}))

    def generate(self, prompts: Any, tools: list[Any], gen_cfg: dict[str, Any]) -> dict[str, Any]:
        conversation = prompts
        if isinstance(prompts, str):
            conversation = [{"role": "user", "content": prompts}]
        elif prompts and isinstance(prompts[0], list):
            conversation = prompts[0]

        # The library's own render helper, with its repair chain for Cohere templates.
        text = _render_chat_template(self.tokenizer, conversation, tools or None, tokenize=False)

        t0 = time.time()
        raw, n_prompt, n_completion = "", 0, 0
        try:
            resp = self.client.completions.create(
                model=self.served_model,
                prompt=text,
                max_tokens=gen_cfg.get("max_new_tokens", self.max_new_tokens),
                temperature=gen_cfg.get("temperature", self.temperature),
                top_p=gen_cfg.get("top_p", self.top_p),
                # The rendered text already starts with BOS; keep special tokens
                # in the output because the tool-call parser reads them.
                stop=self.stop,
                extra_body={
                    "add_special_tokens": False,
                    "skip_special_tokens": False,
                    "include_stop_str_in_output": True,
                },
            )
            raw = resp.choices[0].text or ""
            if resp.usage:
                n_prompt, n_completion = resp.usage.prompt_tokens, resp.usage.completion_tokens
        except BadRequestError as e:  # usually: conversation longer than the context window
            raw = ""
            self._local.last_error = str(e)[:300]

        u = getattr(self._local, "usage", None)
        if u is not None:
            u["prompt_tokens"] += n_prompt
            u["completion_tokens"] += n_completion
            u["calls"] += 1
            u["gen_seconds"] += time.time() - t0

        visible = _strip_special_tokens(raw, self.tokenizer)
        ids = self.tokenizer.encode(raw, add_special_tokens=False) if raw else []
        return {
            "completions": [visible],
            "completions_raw": [raw],
            "logprobs": [None],
            "metadata": {"backend": "vllm-server", "completion_ids": [ids]},
        }
