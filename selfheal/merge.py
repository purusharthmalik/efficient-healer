"""Merge a LoRA adapter into the base model so vLLM can serve it as a plain model."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    print(f"[merge] wrote {a.out}")
