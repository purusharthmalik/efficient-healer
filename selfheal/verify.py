"""Success check for FinQA answers.

`strict` is AgentTune's numerical_match_reward (first number, exact match).
`success` is a superset that also accepts the two harmless format differences
FinQA answers have: rounding to the gold's shown precision (or within 0.5%), and
percent vs fraction (14.1% vs 0.141) when the gold is a percentage or a fraction.
Base and candidate are always scored by the same rule.
"""

from __future__ import annotations

import re

from agenttune.agentic.rewards.builtin_rewards.use_case import numerical_match_reward

_NUM = re.compile(r"-?\d+(?:,\d{3})*\.?\d*")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _first_number(text: str) -> tuple[float, int] | None:
    """(value, decimals shown) of the first number in text."""
    m = _NUM.search(text)
    if not m or not any(ch.isdigit() for ch in m.group(0)):
        return None
    tok = m.group(0).replace(",", "")
    decimals = len(tok.split(".")[1]) if "." in tok else 0
    try:
        return float(tok), decimals
    except ValueError:
        return None


def extract_answer(response: str) -> str | None:
    """Text inside the last <answer> tag, or None when the model gave no tagged answer."""
    found = _ANSWER.findall(response or "")
    return found[-1].strip() if found else None


def verify(response: str, gold: str) -> dict:
    strict = float(numerical_match_reward(None, [response or ""], answer=[gold])[0]) >= 1.0
    answer = extract_answer(response)
    out = {"success": False, "strict": strict, "rule": "no_answer_tag", "pred": answer}
    if answer is None:
        return out
    g, p = _first_number(str(gold)), _first_number(answer)
    if g is None:
        out["rule"] = "gold_not_numeric"
        out["success"] = answer.strip().lower() == str(gold).strip().lower()
        return out
    if p is None:
        out["rule"] = "pred_not_numeric"
        return out
    (gv, gd), (pv, _) = g, p
    rel = abs(gv) * 0.005  # 0.5% of the gold value
    # Same scale: equal after rounding to the gold's shown precision, or within 0.5%.
    if abs(pv - gv) <= max(0.5 * 10 ** (-gd), rel):
        out.update(success=True, rule="match")
        return out
    # Percent vs fraction (14.1% vs 0.141). Only when the gold is written as a percentage
    # or is a fraction below 1, never for a gold of 0, and only within 0.5%: the rounding
    # tolerance is not used here, or a whole-number gold would match values 100x off.
    percent_like = "%" in str(gold) or 0 < abs(gv) < 1
    if percent_like and gv != 0:
        for scale in (100.0, 0.01):
            if abs(pv * scale - gv) <= rel:
                out.update(success=True, rule="match_percent_scale")
                return out
    out["rule"] = "wrong_number"
    return out
