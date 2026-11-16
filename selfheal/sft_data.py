"""Turn selected trajectories into SFT rows in the exact format the student sees.

Every row is a conversation: system prompt with the tool list (AgentTune's Aya
fallback prompt), the task, then alternating assistant tool calls and folded
tool results, then the final answer. Hints used during repair are removed, so
the student is trained on the prompt it gets at inference time.

Target shape: a tool-call turn is only the JSON call list. The
final turn keeps the model's written working and ends at `</answer>`.
`clean=False` keeps tool-call turns exactly as generated instead.
"""

from __future__ import annotations

import json
import re

from .summary import sh
from .tasks import SYSTEM_PROMPT


def _call_list(tool_calls: list[dict]) -> str:
    """The call format AgentTune's Aya prompt asks for: [{"tool_name": ..., "parameters": {...}}]."""
    out = []
    for c in tool_calls:
        fn = c.get("function") or c
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({"tool_name": fn.get("name"), "parameters": args})
    return json.dumps(out)


def _final_turn(content: str) -> str:
    """Keep the written working, drop any thinking block, and end at the closing answer tag."""
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    end = text.rfind("</answer>")
    return text[: end + len("</answer>")] if end != -1 else text


def _strip_hint(system_text: str, hint: str | None) -> str:
    return system_text.replace(f"\n\nHint: {hint}", "") if hint else system_text


def student_messages(rec: dict, clean: bool, tool_result_format: str, system_text: str | None = None) -> list[dict]:
    """`system_text` is required for teacher records, whose own system prompt has no tool list."""
    s = sh(rec)
    conv = rec["metadata"]["conversation"]
    messages: list[dict] = []
    for i, m in enumerate(conv):
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            messages.append({"role": "system", "content": system_text or _strip_hint(content, s.get("hint"))})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "tool":  # teacher conversations keep tool turns; the student sees them folded
            text = tool_result_format.format(name=m.get("name") or "the tool", content=content)
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += f"\n\n{text}"
            else:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            is_last = i == len(conv) - 1
            if m.get("tool_calls"):
                text = _call_list(m["tool_calls"]) if (clean or system_text) else content
            elif is_last:
                text = _final_turn(content)
            else:
                text = content
            messages.append({"role": "assistant", "content": text})
    if messages and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_text or SYSTEM_PROMPT})
    return drop_exact_repeats(messages)


def drop_exact_repeats(messages: list[dict]) -> list[dict]:
    """Remove a tool call and its result when both are identical to the call and
    result immediately before them. The repeat returned the same data, so the run
    stays verified; it only gets shorter."""
    out: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        is_repeat = (
            m["role"] == "assistant"
            and i + 1 < len(messages)
            and len(out) >= 2
            and out[-2]["role"] == "assistant"
            and out[-2]["content"] == m["content"]
            and out[-1]["content"] == messages[i + 1]["content"]
            and m["content"].startswith("[{")
        )
        if is_repeat:
            i += 2
            continue
        out.append(m)
        i += 1
    return out


def valid_row(messages: list[dict]) -> bool:
    """Roles must alternate user/assistant after the system turn and end on an answer."""
    roles = [m["role"] for m in messages[1:]]
    alternating = all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles))
    return bool(roles) and alternating and roles[-1] == "assistant" and "<answer>" in messages[-1]["content"]
