"""Compact review evidence for the measured v2 studies; full traces stay in artifacts/."""

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from evaluation.report import provenance


ROOT = Path(__file__).resolve().parents[1]


def review_json(value, level=0):
    """Keep scalar arrays and ordinary case records on one line for review."""
    flat = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    scalar_array = isinstance(value, (list, tuple)) and all(
        not isinstance(v, (dict, list, tuple)) for v in value
    )
    case = isinstance(value, dict) and "id" in value and len(flat) < 3500
    if (
        not isinstance(value, (dict, list, tuple))
        or len(flat) < 160
        or scalar_array
        or case
    ):
        return flat
    padding = "  " * (level + 1)
    if isinstance(value, dict):
        rows = [
            json.dumps(key, ensure_ascii=False) + ": " + review_json(item, level + 1)
            for key, item in value.items()
        ]
        start, end = "{", "}"
    else:
        rows = [review_json(item, level + 1) for item in value]
        start, end = "[", "]"
    return (
        start
        + "\n"
        + padding
        + (",\n" + padding).join(rows)
        + "\n"
        + "  " * level
        + end
    )


def write_review(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(review_json(payload) + "\n", encoding="utf-8")


def checksum(text):
    return hashlib.sha256(text.encode()).hexdigest()


def model_metadata(directory):
    payload = json.loads((directory / "best.json").read_text())["payload"]
    result = {
        key: payload[key]
        for key in (
            "role",
            "encoder",
            "config",
            "seed",
            "output_nodes",
            "updates",
            "steps_seen",
            "contract",
        )
    }
    result["graph"] = {
        key: value for key, value in payload["graph"].items() if key != "node_ids"
    }
    return result


def code_summary(report):
    result = {
        key: deepcopy(value)
        for key, value in report.items()
        if key not in {"validation", "hidden", "frozen", "composition"}
    }
    for name in ("validation", "hidden", "frozen", "composition"):
        if name not in report:
            continue
        group = report[name]
        summary = {key: value for key, value in group.items() if key != "results"}
        summary["results"] = []
        for case in group["results"]:
            generation = case["generation"]
            grade = case["grade"]
            row = {
                "id": case["id"],
                "shape": case["shape"],
                "status": generation["status"],
                "passed": case["passed"],
                "syntax_ok": grade["syntax_ok"],
                "runtime_status": grade["status"],
                "generated_sha256": checksum(generation["code"]),
                "reference_sha256": checksum(case["target"]),
                "stdout": grade.get("stdout"),
                "returns": grade.get("returns"),
                "variables": grade.get("variables"),
            }
            if not case["passed"] and name != "frozen":
                row.update(
                    source=case["source"],
                    generated_code=generation["code"],
                    reference=case["target"],
                    grade=grade,
                )
            summary["results"].append(row)
        result[name] = summary
    result["neural_trace_sample"] = report["hidden"]["results"][0]["generation"][
        "trace"
    ][:12]
    return result


def nl_summary(report):
    result = deepcopy(report)
    for group in ("validation", "hidden", "frozen"):
        for row in result.get(group, {}).get("results", []):
            row["generation"]["trace"] = row["generation"]["trace"][:12]
    return result


def snapshot(code_dir, nl_dir, e2e_dir, toy_dir, output, *, test_log=None):
    code_dir, nl_dir, e2e_dir, toy_dir, output = map(
        Path, (code_dir, nl_dir, e2e_dir, toy_dir, output)
    )
    code = json.loads((code_dir / "report.json").read_text())
    nl = json.loads((nl_dir / "report.json").read_text())
    e2e = json.loads((e2e_dir / "report.json").read_text())
    toy = json.loads((toy_dir / "report.json").read_text())
    for run in toy["runs"]:
        run.pop("history", None)
        run.pop("example_trace", None)
    code_report = code_summary(code)
    code_report["model"] = model_metadata(code_dir)
    nl_report = nl_summary(nl)
    nl_report["model"] = model_metadata(nl_dir)
    for row in e2e["results"]:
        for role in ("fly_nl", "fly_code"):
            if row["pipeline"].get(role):
                row["pipeline"][role]["trace"] = row["pipeline"][role]["trace"][:12]
    write_review(output / "fly-code-v2.json", code_report)
    write_review(output / "fly-nl-v2.json", nl_report)
    write_review(output / "end-to-end-v2.json", e2e)
    write_review(output / "toy-signed-v2.json", toy)
    verification = {
        "provenance": provenance(ROOT),
        "tests": Path(test_log).read_text() if test_log else None,
        "p3_gate": code["gate"],
        "p3_restore_exact": code["restore_exact"],
        "nl_restore_exact": nl["restore_exact"],
        "e2e_weights_unchanged": e2e["weights_unchanged_during_evaluation"],
    }
    write_review(output / "verification-v2.json", verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=ROOT / "artifacts/code-v2")
    parser.add_argument("--nl-dir", type=Path, default=ROOT / "artifacts/nl-v2")
    parser.add_argument("--e2e-dir", type=Path, default=ROOT / "artifacts/e2e-v2")
    parser.add_argument("--toy-dir", type=Path, default=ROOT / "artifacts/toy-signed")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports")
    parser.add_argument("--test-log", type=Path)
    args = parser.parse_args()
    snapshot(
        args.code_dir,
        args.nl_dir,
        args.e2e_dir,
        args.toy_dir,
        args.output,
        test_log=args.test_log,
    )


if __name__ == "__main__":
    main()
