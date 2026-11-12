from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from evaluation.code_data import code_cases
from evaluation.datasets import Case, lock_manifest, split_cases, toy_cases
from evaluation.toy import TOY_VERSION
from flybrain.graph import Graph, digest
from ir.tokens import spec_hash
from evaluation.curriculum_v2 import DATA_VERSION, code_cases_v2, composition_cases
from evaluation.nl_data import NL_DATA_VERSION, nl_cases
from evaluation.retries import (
    DATA_VERSION as RETRY_DATA_VERSION,
    retry_cases,
    run as run_retries_study,
)
from models.train_sequence import fit, polish_saved


ROOT = Path(__file__).resolve().parents[1]


class HoldoutTest(unittest.TestCase):
    def test_v2_locks_freeze_held_out_names_paraphrases_and_compositions(self):
        graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
        for filename, splits, kind in (
            ("code-v2.json", code_cases_v2(), DATA_VERSION),
            ("nl-v2.json", nl_cases(), NL_DATA_VERSION),
            ("retry-v1.json", retry_cases(), RETRY_DATA_VERSION),
            (
                "compositions-v2.json",
                {"composition": composition_cases()},
                "held-out-compositions-v2",
            ),
        ):
            path = ROOT / "experiments/locked" / filename
            self.assertTrue(path.is_file())
            lock_manifest(
                path,
                splits=splits,
                graph=graph,
                ir_hash=spec_hash(),
                seed=314159,
                kind=kind,
            )
        nl = nl_cases()
        self.assertEqual({r["paraphrase"] for r in nl["train"]}, {0, 1, 2, 3})
        self.assertEqual({r["paraphrase"] for r in nl["validation"]}, {4})
        self.assertEqual({r["paraphrase"] for r in nl["test"]}, {5})

    def test_opened_hidden_study_cannot_be_tuned_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "hidden.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Cannot train"):
                fit(None, {}, path, {})
            (path / "report.json").write_text('{"test_evaluated":true}')
            with self.assertRaisesRegex(ValueError, "Cannot tune"):
                polish_saved(path)
            with self.assertRaisesRegex(ValueError, "already opened"):
                run_retries_study(None, None, path)

    def test_committed_evaluation_locks_match_current_generators(self):
        for filename, splits, graph, ir_hash, kind in (
            (
                "toy-v1.json",
                toy_cases(),
                "flywire_sample.csv",
                digest(TOY_VERSION),
                TOY_VERSION,
            ),
            (
                "code-v1.json",
                code_cases(),
                "flywire_lab_2048.csv",
                spec_hash(),
                "P3-LET-PRINT-surface-holdout-v1",
            ),
        ):
            path = ROOT / "experiments/locked" / filename
            self.assertTrue(
                path.is_file(), "Do not silently recreate a missing evaluation lock"
            )
            lock_manifest(
                path,
                splits=splits,
                graph=Graph.from_csv(ROOT / graph),
                ir_hash=ir_hash,
                seed=1729,
                kind=kind,
            )

    def test_splits_are_order_independent_and_group_disjoint(self):
        cases = [Case(str(i), f"group-{i // 2}", f"input-{i}", "x") for i in range(20)]
        self.assertEqual(split_cases(cases), split_cases(list(reversed(cases))))
        for splits in (split_cases(cases), toy_cases(), code_cases()):
            names = list(splits)
            for i, name in enumerate(names):
                self.assertTrue(splits[name])
                for other in names[i + 1 :]:
                    self.assertFalse(
                        {r["family"] for r in splits[name]}
                        & {r["family"] for r in splits[other]}
                    )
                    self.assertFalse(
                        {r["source"] for r in splits[name]}
                        & {r["source"] for r in splits[other]}
                    )

    def test_duplicate_inputs_and_cross_split_family_leakage_are_rejected(self):
        cases = [Case(str(i), str(i), "same", "different") for i in range(4)]
        with self.assertRaises(ValueError):
            split_cases(cases)
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        with tempfile.TemporaryDirectory() as directory:
            splits = toy_cases()
            splits["test"][0]["family"] = splits["train"][0]["family"]
            with self.assertRaisesRegex(ValueError, "group leakage"):
                lock_manifest(
                    Path(directory) / "data.json",
                    splits=splits,
                    graph=graph,
                    ir_hash="test",
                    seed=1,
                    kind="test",
                )

    def test_locked_contract_cannot_be_silently_replaced(self):
        graph = Graph.from_csv(ROOT / "flywire_sample.csv")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            kwargs = {
                "splits": toy_cases(),
                "graph": graph,
                "ir_hash": "test",
                "seed": 1,
                "kind": "test",
            }
            first = lock_manifest(path, **kwargs)
            self.assertEqual(first, lock_manifest(path, **kwargs))
            changed = deepcopy(kwargs)
            changed["splits"]["test"][0]["target"] = "wrong"
            with self.assertRaisesRegex(ValueError, "Locked"):
                lock_manifest(path, **changed)
