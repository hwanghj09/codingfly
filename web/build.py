"""Build self-hosted GitHub Pages assets and a verified, inference-only Python bundle."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain.graph import Graph
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code_v2 import CODE_VOCABULARY
from models.fly_nl import NL_VOCABULARY
from models.neural_sequence import NeuralSequence

PYODIDE_VERSION = "0.27.7"
PYODIDE_BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
MODULES = {
    "flybrain": "__init__ graph dynamics checkpoint encoder decoder plasticity clock_encoder policy retry_input sequence_io sequence_state",
    "models": "__init__ fly_code fly_code_v2 fly_nl neural_sequence pipeline",
    "ir": "__init__ grammar serializer tokens",
}


def verified_release():
    release = json.loads((ROOT / "web/checkpoints/release.json").read_text())
    report = json.loads((ROOT / "docs/reports/accuracy-v3.json").read_text())
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    if (
        release["graph_sha256"] != graph.topology_hash
        or release["graph_source_sha256"] != graph.source_hash
        or release["ir_version"] != IR_VERSION
        or release["ir_sha256"] != spec_hash()
    ):
        raise ValueError("Released model/graph/IR mismatch")
    for role, vocabulary, expected in (
        ("fly-nl", NL_VOCABULARY, report["final_weights_sha256"]),
        ("fly-code", CODE_VOCABULARY, report["b_weights_sha256"]),
    ):
        entry = release["models"][role]
        path = ROOT / "web" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Released checkpoint file checksum mismatch")
        model, _ = NeuralSequence.restore(
            path, graph, vocabulary, role=role, contract=entry["contract"]
        )
        if (
            model.brain.weights_hash != entry["weights_sha256"]
            or entry["weights_sha256"] != expected
        ):
            raise ValueError("Released synapses differ from evaluated synapses")
    return release


def model_archive(release):
    files = {
        f"{package}/{name}.py": ROOT / package / f"{name}.py"
        for package, names in MODULES.items()
        for name in names.split()
    }
    files.update(
        {
            entry["file"]: ROOT / "web" / entry["file"]
            for entry in release["models"].values()
        }
    )
    files["flywire_lab_2048.csv"] = ROOT / "flywire_lab_2048.csv"
    files["browser_bridge.py"] = ROOT / "web/bridge.py"
    files["docs/IR_SPEC.md"] = ROOT / "docs/IR_SPEC.md"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("browser_manifest.json", date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(release, ensure_ascii=False))
    return data.getvalue()


def vendor_runtime(destination):
    destination.mkdir(parents=True, exist_ok=True)

    def download(name, checksum=None):
        path = destination / name
        if path.exists() and (
            checksum is None
            or hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        ):
            return path.read_bytes()
        with urllib.request.urlopen(PYODIDE_BASE + name, timeout=60) as response:
            data = response.read()
        if checksum and hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"Runtime package checksum mismatch: {name}")
        path.write_bytes(data)
        return data

    lock = json.loads(download("pyodide-lock.json"))
    if lock["info"]["version"] != PYODIDE_VERSION:
        raise ValueError("Unexpected browser Python version")
    for name in (
        "pyodide.js",
        "pyodide.asm.js",
        "pyodide.asm.wasm",
        "python_stdlib.zip",
    ):
        download(name)
    visited = set()

    def package(name):
        if name in visited:
            return
        visited.add(name)
        entry = lock["packages"][name]
        for dependency in entry["depends"]:
            package(dependency)
        download(entry["file_name"], entry["sha256"])

    package("numpy")


def build(output, *, runtime=True, source_export=False):
    output = Path(output)
    protected = [
        ROOT / name
        for name in (
            ".git",
            "web",
            "flybrain",
            "models",
            "ir",
            "tests",
            "evaluation",
            "docs",
            "experiments",
        )
    ]
    if not (source_export and output.resolve() == ROOT / "docs/site") and (
        output.resolve() == ROOT
        or any(
            path == output.resolve() or path in output.resolve().parents
            for path in protected
        )
    ):
        raise ValueError("Build output cannot overwrite source")
    release = verified_release()
    bundle = model_archive(release)
    identity = hashlib.sha256(bundle).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "index.html",
        "styles.css",
        "app.js",
        "inference-worker.js",
        "execution-worker.js",
        "runner.py",
        "favicon.svg",
    ):
        shutil.copyfile(ROOT / "web" / name, output / name)
    filename = f"model-{identity[:16]}.zip"
    for old in output.glob("model-*.zip"):
        if old.name != filename:
            old.unlink()
    (output / filename).write_bytes(bundle)
    (output / "site-manifest.json").write_text(
        json.dumps(
            {
                **release,
                "bundle": filename,
                "bundle_sha256": identity,
                "pyodide_version": PYODIDE_VERSION,
                "runtime_base": PYODIDE_BASE if source_export else "./runtime/",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (output / ".nojekyll").touch()
    (output / "LICENSES.txt").write_text(
        "FlyWire Consortium, Dorkenwald et al., Neuronal wiring diagram of an adult brain (2024).\n"
        "Derived from FlyWire Codex FAFB v783, selected and aggregated as described in the repository README.\n"
        "https://www.nature.com/articles/s41586-024-07558-y\n"
        "https://home.flywire.ai/guidelines\n"
        "Data license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/\n"
        "Pyodide: MPL-2.0; CPython and NumPy retain their upstream licenses.\n"
        "https://github.com/pyodide/pyodide\nhttps://github.com/hwanghj09/codingfly\n"
    )
    if runtime:
        vendor_runtime(output / "runtime")
    print(f"Built {output}; {len(bundle):,} byte verified model bundle")


def check_source():
    with tempfile.TemporaryDirectory() as directory:
        # Match the legacy export without granting permission to overwrite source.
        build(directory, runtime=False)
        path = Path(directory) / "site-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["runtime_base"] = PYODIDE_BASE
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        expected = {
            p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()
        }
        actual = {
            p.name: p.read_bytes()
            for p in (ROOT / "docs/site").iterdir()
            if p.is_file()
        }
        if expected != actual:
            raise ValueError(
                "Tracked Pages export is stale; run python web/build.py --export-source"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    parser.add_argument(
        "--skip-runtime", action="store_true", help="For offline unit checks only"
    )
    parser.add_argument("--export-source", action="store_true")
    parser.add_argument("--check-source", action="store_true")
    arguments = parser.parse_args()
    if arguments.check_source:
        check_source()
    elif arguments.export_source:
        build(ROOT / "docs/site", runtime=False, source_export=True)
    else:
        build(arguments.output, runtime=not arguments.skip_runtime)
