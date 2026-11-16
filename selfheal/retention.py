"""Retention and safety check with AuditKIT: does the adapter change what should not change?

Runs the same small benchmark samples through two vLLM servers (base model and adapter) with
AuditKIT's `api:` backend, in plain chat mode with no tools:
  GSM8K            grade-school maths, numeric match            (general ability)
  Global-MMLU      multiple choice in several languages         (multilingual ability)
  XSTest           refusal on unsafe prompts, and over-refusal on safe prompts that look unsafe
Each benchmark is isolated: one failing does not stop the others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import auditkit as ak


def headline(result) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in dict(result.headline).items()}


def run_all(model: str, api_base: str, n: int, languages: list[str]) -> dict:
    kw = {"api_base": api_base, "api_key": "EMPTY"}
    spec = f"api:{model}"
    results: dict = {}

    def gsm8k():
        from auditkit.scenarios.gsm8k import GSM8KScenario

        samples = list(GSM8KScenario().samples())[:n]
        return samples, ak.evaluate(samples, spec, ["numeric_match"], **kw)

    def global_mmlu():
        samples = ak.load_global_mmlu(languages, limit=max(1, n // len(languages)))
        return samples, ak.evaluate(samples, spec, ["acc"], adapter="auto", **kw)

    def xstest_unsafe():
        samples = [s for s in ak.load_xstest() if s.metadata.get("label") == "unsafe"][:n]
        return samples, ak.evaluate(samples, spec, [ak.refusal_scorer()], **kw)

    def xstest_safe():
        samples = [s for s in ak.load_xstest() if s.metadata.get("label") == "safe"][:n]
        return samples, ak.evaluate(samples, spec, [ak.refusal_scorer("over_refusal", direction=ak.Direction.MINIMIZE)], **kw)

    for name, fn in (("gsm8k", gsm8k), ("global_mmlu", global_mmlu),
                     ("xstest_unsafe_refusal", xstest_unsafe), ("xstest_safe_over_refusal", xstest_safe)):
        try:
            samples, res = fn()
            results[name] = {"n": len(samples), "headline": headline(res), "_result": res}
            print(f"[retention] {model} {name}: n={len(samples)} {headline(res)}", flush=True)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
            print(f"[retention] {model} {name}: ERROR {results[name]['error']}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs=2, metavar=("SERVED_NAME", "API_BASE"), required=True)
    ap.add_argument("--cand", nargs=2, metavar=("SERVED_NAME", "API_BASE"), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--languages", nargs="+", default=["en", "hi", "es", "ar", "sw"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = run_all(a.base[0], a.base[1], a.n, a.languages)
    cand = run_all(a.cand[0], a.cand[1], a.n, a.languages)
    report = {}
    for name in base:
        entry = {"base": {k: v for k, v in base[name].items() if k != "_result"},
                 "candidate": {k: v for k, v in cand.get(name, {}).items() if k != "_result"}}
        if "_result" in base[name] and "_result" in cand.get(name, {}):
            try:
                cmp = ak.compare(base[name]["_result"], cand[name]["_result"])
                entry["significance"] = {m: cmp.significance(m) for m in base[name]["headline"]}
            except Exception as e:
                entry["significance_error"] = f"{type(e).__name__}: {e}"[:300]
        report[name] = entry
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: {"base": v["base"], "candidate": v["candidate"],
                          "delta": {m: (s.get("delta"), s.get("ci")) for m, s in v.get("significance", {}).items()}}
                      for k, v in report.items()}, indent=1, default=str))
