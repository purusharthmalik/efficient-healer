"""Publish the exploration results: copy them from the private ops repo into the public repos.

    uvx --from "huggingface_hub>=2.0" python tools/publish_exploration.py [--dry-run]

Run on a laptop; no GPU. The two GPU jobs saved their outputs under outa/ and outb/ and their
adapters under adapters/ in the ops repo. Only the files the exploration produced are copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

OPS = "Purusharth/selfheal-ops"
DATASET = "Purusharth/efficient-healer-trajectories"
TINY = "Purusharth/tiny-aya-efficient-healer-lora"
EXPANSE = "Purusharth/aya-expanse-8b-efficient-healer-lora"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "train_log.json", "train_rows.json"]

A_EVAL = ["setB_final", "setB_r3", "confirm_r3", "setB_r3dpo"]
A_CMP = ["cmp_r3_gate", "cmp_confirm_r3", "cmp_setB_r3dpo"]
A_GATES = ["gate_round3.json", "gate_r3dpo_setB.json"]
B_EVAL = ["confirm_expanse_base", "confirm_expanse_ft", "confirm_dpo", "setB_dpo", "setB_expanse_ft"]
B_CMP = ["cmp_expanse_vs_tiny", "cmp_dpo", "cmp_setB_dpo", "cmp_setB_expanse"]
B_GATES = ["gate_dpo.json", "gate_dpo_setB.json"]


def cp(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  missing: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors", "*.bin"))
    else:
        shutil.copy2(src, dst)
    return True


def adapter(src: Path, dst: Path) -> bool:
    if not (src / "adapter_model.safetensors").exists():
        print(f"  missing adapter: {src}")
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="runs/explore")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    patterns = ["adapters/**", "logsa/*", "logsb/*", "outa/r3/**", "outa/dpo3/pairs.jsonl", "outa/splits.json",
                "outa/splits_fresh2.json", "outb/dpo/pairs.jsonl", "outb/void/**", "outb/eval/retention/**"]
    patterns += [f"outa/eval/{n}.*" for n in A_EVAL] + [f"outa/eval/{n}/**" for n in A_CMP] + [f"outa/eval/{n}" for n in A_GATES]
    patterns += [f"outb/eval/{n}.*" for n in B_EVAL] + [f"outb/eval/{n}/**" for n in B_CMP] + [f"outb/eval/{n}" for n in B_GATES]
    src = Path(snapshot_download(OPS, repo_type="dataset", allow_patterns=patterns, local_dir=str(work / "ops")))

    stage = work / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    data, tiny, expanse = stage / "dataset", stage / "tiny", stage / "expanse"

    # ---- dataset ----
    cp(src / "outa/r3", data / "round3")
    cp(src / "outa/splits.json", data / "splits_pool1400.json")
    cp(src / "outa/splits_fresh2.json", data / "splits_heldout_B.json")
    for side, names, cmps, gates in (("outa", A_EVAL, A_CMP, A_GATES), ("outb", B_EVAL, B_CMP, B_GATES)):
        for n in names:
            for ext in (".jsonl", ".summary.json"):
                cp(src / side / "eval" / f"{n}{ext}", data / "eval" / f"{n}{ext}")
        for n in cmps:
            cp(src / side / "eval" / n, data / "eval" / n)
        for n in gates:
            cp(src / side / "eval" / n, data / "eval" / n)
    cp(src / "outb/eval/retention", data / "exploration/retention")
    cp(src / "outb/dpo/pairs.jsonl", data / "exploration/dpo_on_round2/pairs.jsonl")
    cp(src / "outa/dpo3/pairs.jsonl", data / "exploration/dpo_on_round3/pairs.jsonl")
    cp(src / "outb/void", data / "exploration/void_expanse_run_without_database")
    for side in ("logsa", "logsb"):
        for f in sorted((src / side).glob("*.log")):
            cp(f, data / "worker_logs_exploration" / f"{side[-1]}_{f.name}")

    # ---- Tiny Aya model repo: round 3 at the root if its gate said deploy ----
    gate3 = src / "outa/eval/gate_round3.json"
    deploy3 = gate3.exists() and json.loads(gate3.read_text())["deploy"]
    adapter(src / "adapters/round3", tiny / "round3")
    adapter(src / "adapters/dpo", tiny / "round2_dpo")
    adapter(src / "adapters/round3_dpo", tiny / "round3_dpo")
    if deploy3:
        adapter(src / "adapters/round3", tiny)
        (tiny / "deployed.json").write_text(json.dumps({"deployed": "round3"}, indent=1))
    # ---- Expanse model repo ----
    adapter(src / "adapters/expanse", expanse)

    for name, folder in (("dataset", data), ("tiny", tiny), ("expanse", expanse)):
        files = [f for f in folder.rglob("*") if f.is_file()]
        print(f"{name}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")
    print("round 3 gate deploy:", deploy3)
    if a.dry_run:
        return

    api = HfApi()
    api.create_repo(EXPANSE, repo_type="model", private=False, exist_ok=True)
    for repo, kind, folder, msg in ((DATASET, "dataset", data, "Round 3 and further experiments"),
                                    (TINY, "model", tiny, "Round 3 adapter (deployed) and DPO adapters"),
                                    (EXPANSE, "model", expanse, "Aya Expanse 8B adapter")):
        api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder), commit_message=msg)
        info = api.repo_info(repo, repo_type=kind)
        print(f"uploaded to {repo}: private={info.private}, files={len(info.siblings)}")


if __name__ == "__main__":
    main()
