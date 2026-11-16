"""Re-apply the current verifier to rollout files in place (after a verifier fix)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .verify import verify

if __name__ == "__main__":
    for path in sys.argv[1:]:
        p = Path(path)
        recs = [json.loads(line) for line in p.open() if line.strip()]
        before = after = 0
        for r in recs:
            s = r["metadata"]["selfheal"]
            before += bool(s["success"])
            if not s.get("error"):
                v = verify(r.get("final_response", ""), s["gold"])
                s.update(success=v["success"], strict=v["strict"], verify_rule=v["rule"], pred=v["pred"])
                r["reward"] = 1.0 if v["success"] else 0.0
            after += bool(s["success"])
        p.write_text("".join(json.dumps(r, default=str) + "\n" for r in recs))
        print(f"[rescore] {path}: successes {before} -> {after} of {len(recs)}")
