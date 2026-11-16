"""Stage the run's artifacts and upload them to two Hugging Face repos (private by default).

Model repo:    the deployed adapter at the root, plus every adapter trained, each in its own folder.
Dataset repo:  every rollout, failure label, training set, evaluation log and gate decision.
Visibility is changed separately, by hand, after review.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "train_log.json", "train_rows.json"]


def copy_adapter(src: Path, dst: Path) -> bool:
    if not (src / "adapter_model.safetensors").exists():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return True


def copy_tree(src: Path, dst: Path, skip_suffixes=(".safetensors", ".bin", ".pt")) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for f in src.rglob("*"):
        if f.is_file() and f.suffix not in skip_suffixes and "merged" not in f.parts:
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--stage", default="publish_stage")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    out, stage = Path(a.out), Path(a.stage)
    shutil.rmtree(stage, ignore_errors=True)
    model_dir, data_dir = stage / "model", stage / "dataset"

    # Which adapter is deployed is decided by the gates, not by us.
    def gate(name: str) -> bool:
        p = out / "eval" / name
        return p.exists() and json.loads(p.read_text())["deploy"]

    deployed = "base"
    if gate("gate_round1.json"):
        deployed = "round1"
    # Round 2: the single confirmatory evaluation, when it was run, has the last word.
    round2_gate = "gate_round2_confirm.json" if (out / "eval" / "gate_round2_confirm.json").exists() else "gate_round2.json"
    if deployed == "round1" and gate(round2_gate):
        deployed = "round2"
    adapters = {"round1": out / "r1" / "adapter", "round2": out / "final" / "adapter",
                "negative_all_token_loss": out / "neg_alltokens" / "adapter"}
    for name, src in adapters.items():
        copy_adapter(src, model_dir / name)
    if deployed in adapters:
        copy_adapter(adapters[deployed], model_dir)
    (model_dir / "deployed.json").write_text(json.dumps({"deployed": deployed}, indent=1))

    layout = {
        "splits.json": "splits.json",
        "splits_confirm.json": "splits_confirm.json",
        "eval": "eval",
        "r1": "round1",
        "r2": "round2",
        "pilot": "pilot",
        "neg_alltokens": "negative_all_token_loss",
        "pool": "pilot/base_no_stop_first_20_tasks",
    }
    for src, dst in layout.items():
        copy_tree(out / src, data_dir / dst)
    copy_tree(Path("logs"), data_dir / "worker_logs")
    for f in (data_dir / "worker_logs").glob("*.srv"):  # raw server logs, large and not useful
        f.unlink()

    api = HfApi()
    for repo, kind, folder in ((a.model_repo, "model", model_dir), (a.dataset_repo, "dataset", data_dir)):
        api.create_repo(repo, repo_type=kind, private=not a.public, exist_ok=True)
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
        print(f"[publish] uploading {size:.0f} MB to {kind} repo {repo} (private={not a.public})", flush=True)
        api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                          commit_message="Efficient Healer run artifacts")
        info = api.repo_info(repo, repo_type=kind)
        print(f"[publish] {repo}: private={info.private}, files={len(info.siblings)}", flush=True)
    print(f"[publish] deployed adapter: {deployed}")


if __name__ == "__main__":
    main()
