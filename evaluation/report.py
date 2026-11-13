"""Machine-readable reports include environment, code and source identities."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from flybrain.graph import digest


def provenance(root):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for pattern in ("*.py", "*.md", "*.csv", "requirements.txt")
               for path in Path(root).rglob(pattern)
               if not any(part in {".git", "artifacts", ".venv"} for part in path.relative_to(root).parts)}
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
            "source_sha256": digest(sources), "source_files": sources}


def write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summary(values):
    values = np.array(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.
    # Descriptive normal approximation, not a significance claim at n=5.
    margin = 1.96 * std / float(np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sample_std": std,
            "normal_approx_95_ci": [mean - margin, mean + margin]}
