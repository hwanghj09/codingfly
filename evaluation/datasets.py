"""Content-locked, group-aware splits. Test examples never drive training or selection."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from flybrain.graph import digest


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    source: str
    target: str


def split_cases(cases, *, seed=1729):
    cases = sorted(cases, key=lambda case: case.id)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    if len({case.source for case in cases}) != len(cases):
        raise ValueError("Duplicate input (including conflicting labels)")
    families = sorted({case.family for case in cases}, key=lambda family: digest([seed, family]))
    if len(families) < 3:
        raise ValueError("At least three independent groups required")
    validation_count = max(1, len(families) // 5)
    test_count = max(1, len(families) // 5)
    assignments = {family: "validation" if i < validation_count else
                    "test" if i < validation_count + test_count else "train"
                    for i, family in enumerate(families)}
    return {name: [asdict(case) for case in cases if assignments[case.family] == name]
            for name in ("train", "validation", "test")}


def lock_manifest(path, *, splits, graph, ir_hash, seed, kind):
    all_cases = [case for cases in splits.values() for case in cases]
    ids = [case["id"] for case in all_cases]
    sources = [case["source"] for case in all_cases]
    if len(set(ids)) != len(ids) or len(set(sources)) != len(sources):
        raise ValueError("Cross-split duplicate ID/input")
    family_sets = {name: {case["family"] for case in cases} for name, cases in splits.items()}
    names = list(family_sets)
    if any(family_sets[a] & family_sets[b] for i, a in enumerate(names) for b in names[i + 1:]):
        raise ValueError("Cross-split group leakage")
    payload = {"version": 1, "seed": seed, "kind": kind, "graph": graph.metadata(),
               "ir_sha256": ir_hash, "splits": splits}
    manifest = {"sha256": digest(payload), "payload": payload}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(manifest)):
            raise ValueError("Locked evaluation contract changed; use a new run directory")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def toy_cases():
    """First byte classification with unseen nuisance bytes, NOT language understanding."""
    # Lock on nuisance identity so variants of the same stimulus cannot cross splits.
    cases = [Case(f"byte-{first}-{noise}", f"noise-{noise}", bytes([first, noise]).hex(), str(label))
             for label, first in enumerate(b"ABCD") for noise in range(48)]
    return split_cases(cases)
