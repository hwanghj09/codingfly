import argparse
import hashlib
import json
import random
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

from fly import BLANK, HERE, Fly, load_lessons
from algorithm_phrases import ALGORITHM_LESSONS, CHECKS, GENERALIZATION_CHECKS
from program_model import ProgramModel
from reward import RewardMemory


GRAPH = HERE / "flywire_lab_2048.csv"
PORT = 8765
STAGE_ROUNDS = 12
PRACTICE_ROUNDS = 12
BATCH_ROUNDS = 3


def attempt(stage, prompt, chosen, expected, reward=None):
    return {
        "stage": stage, "prompt": prompt, "chosen": chosen, "expected": expected,
        "code": prompt.replace(BLANK, chosen) if chosen is not None else prompt,
        "reward": reward if reward is not None else (1 if chosen == expected else -1) if chosen is not None else None,
    }


class Experiment:
    def __init__(self, checkpoint, delay=0.13, *, graph=GRAPH, prioritized=True):
        self.lessons = load_lessons()
        self.tokens = [lesson["token"] for lesson in self.lessons]
        self.fly = Fly(self.tokens, graph, extended_context=True)
        self.graph_hash = hashlib.sha256(graph.read_bytes()).hexdigest()
        self.prioritized = prioritized
        self.checkpoint = checkpoint
        self.delay = delay
        self.lock = Lock()
        self.running = False
        self.pause_requested = False
        self.error = ""
        self.state = {"stage": 1, "round": 0, "batches": 0, "positive": 0, "negative": 0, "bonus": 0,
                      "history": [], "events": [], "results": [], "feedback": {}}
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved["version"] != 2 or saved["graph"] != self.graph_hash:
                raise ValueError("저장된 모델과 현재 연결망 또는 모델 버전이 다릅니다.")
            if saved["tokens"] != self.tokens[:len(saved["tokens"])]:
                raise ValueError("저장된 수업 순서를 확인해 주세요. 새 수업은 마지막에 추가할 수 있습니다.")
            self.state = saved["state"]
            if len(saved["tokens"]) < len(self.tokens) and self.state["stage"] > len(saved["tokens"]):
                self.state["round"] = 0
        self.train_examples = self.examples("train")
        self.check_examples = self.examples("check")
        self.memory = RewardMemory(self.fly, self.train_examples, self.check_examples, self.state["feedback"])
        if self.state["history"]:
            last = self.state["history"][-1]
            last["correct"], last["total"] = self.evaluate(last["stage"])

    def examples(self, split):
        return [(index, prompt, lesson["token"])
                for index, lesson in enumerate(self.lessons, 1) for prompt in lesson[split]]

    def mastered(self):
        last = self.state["history"][-1] if self.state["history"] else None
        return (self.state["stage"] > len(self.lessons) and last is not None
                and last["correct"] == last["total"] == len(self.check_examples))

    def evaluate(self, stage):
        results = [attempt(index, prompt, self.memory.predict(prompt), expected)
                   for index, prompt, expected in self.check_examples if index <= stage]
        self.state["results"] = results
        return sum(result["reward"] == 1 for result in results), len(results)

    def start(self):
        with self.lock:
            if self.running or self.mastered():
                return False
            self.running = True
            self.pause_requested = False
            self.error = ""
        Thread(target=self.run, daemon=True).start()
        return True

    def run(self):
        try:
            while True:
                with self.lock:
                    if self.pause_requested or self.mastered():
                        break
                    self.batch(self.state["stage"] > len(self.lessons))
                if self.delay:
                    time.sleep(self.delay)
        except (OSError, ValueError) as error:
            with self.lock:
                self.error = str(error)
        finally:
            with self.lock:
                self.running = False
                self.pause_requested = False

    def pause(self):
        with self.lock:
            self.pause_requested = self.running

    def batch(self, practice):
        state = self.state
        stage = min(state["stage"], len(self.lessons))
        target_rounds = PRACTICE_ROUNDS if practice else STAGE_ROUNDS
        rng = random.Random(42 + state["batches"])
        training = [example for example in self.train_examples if example[0] <= stage]
        tokens = tuple(dict.fromkeys(self.tokens[:min(stage + 1, len(self.tokens))]))
        samples = {}
        for _ in range(min(BATCH_ROUNDS, target_rounds - state["round"])):
            rng.shuffle(training)
            weights = [self.memory.priority(prompt) for _, prompt, _ in training] if self.prioritized else None
            replay = rng.choices(training, weights=weights, k=max(1, len(training) // 3))
            for index, prompt, expected in [*training, *replay]:
                chosen, reward = self.memory.attempt(prompt, expected, tokens, rng)
                state["positive" if reward > 0 else "negative"] += 1
                state["bonus"] += max(0, reward - 1)
                key = index, reward > 0
                if key not in samples or reward >= samples[key]["reward"]:
                    samples[key] = attempt(index, prompt, chosen, expected, reward)
            state["round"] += 1

        state["batches"] += 1
        counts = Counter()
        recent = []
        for event in reversed([*state["events"], *samples.values()]):
            index = event["stage"]
            if counts[index] < 8:
                recent.append(event)
                counts[index] += 1
        state["events"] = list(reversed(recent))
        correct, total = self.evaluate(stage)
        state["history"] = [*state["history"], {"stage": stage, "correct": correct,
                            "total": total, "batch": state["batches"]}][-240:]
        done = state["round"] == target_rounds
        if done:
            state["round"] = 0
            if not practice:
                state["stage"] += 1
        self.save()

    def save(self):
        temporary = self.checkpoint.with_name(f".{self.checkpoint.name}.tmp")
        temporary.write_text(json.dumps({"version": 2, "graph": self.graph_hash, "tokens": self.tokens,
                                        "state": self.state},
                                       separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.checkpoint)

    def snapshot(self):
        with self.lock:
            visible = {key: value for key, value in self.state.items() if key != "feedback"}
            return json.dumps({**visible, "running": self.running, "mastered": self.mastered(), "error": self.error,
                               "stageRounds": STAGE_ROUNDS, "practiceRounds": PRACTICE_ROUNDS,
                               "nodes": self.fly.node_count, "edges": len(self.fly.edges),
                               "lessons": [{"title": lesson["title"], "token": lesson["token"]}
                                           for lesson in self.lessons]}).encode()


def serve(lab):
    program_model = ProgramModel(lab.fly)
    categories = {
        "기존 문법 종합": "program_checks.json",
        "추가 문법 종합": "advanced_checks.json",
        "알고리즘": "algorithm_checks.json",
        "사용자 예제": "example_checks.json",
    }
    examples = {group: json.loads((HERE / name).read_text(encoding="utf-8")) for group, name in categories.items()}
    examples["알고리즘"].extend(json.loads((HERE / "algorithm_transfer.json").read_text(encoding="utf-8")))
    examples["알고리즘 학습 예제"] = [
        {"title": title, "source": "\n".join(
            f"{'    ' * depth}{index}. {phrase}: {operands}".rstrip()
            for index, (depth, phrase, _, operands) in enumerate(steps, 1))}
        for title, steps in ALGORITHM_LESSONS
    ]
    program_checks = [*examples["기존 문법 종합"], *examples["추가 문법 종합"]]
    program_summary = program_model.evaluate(program_checks)
    program_summary["examples"] = {group: [{"title": check["title"], "source": check["source"]} for check in checks]
                                   for group, checks in examples.items()}
    program_summary["curriculum"] = program_model.curriculum
    phrase_checks = [(phrase, expected) for split in (CHECKS, GENERALIZATION_CHECKS)
                     for expected, phrases in split.items() for phrase in phrases]
    program_summary["phrase_correct"] = sum(program_model.algorithm.ranks(phrase)[0][0] == expected
                                           for phrase, expected in phrase_checks)
    program_summary["phrase_total"] = len(phrase_checks)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send((HERE / "lab.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send(lab.snapshot(), "application/json")
            elif self.path == "/api/program":
                self.send(json.dumps(program_summary).encode(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in ("/api/train", "/api/pause", "/api/generate"):
                self.send_error(404)
            elif (self.headers.get("Origin") != f"http://127.0.0.1:{PORT}"
                  or self.headers.get("Content-Type") != "application/json"):
                self.send_error(403)
            elif self.path == "/api/generate":
                self.generate()
            elif self.path == "/api/train":
                started = lab.start()
                self.send(b'{"started":true}' if started else b'{"started":false}', "application/json")
            else:
                lab.pause()
                self.send(b'{}', "application/json")

        def generate(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65_536:
                    raise ValueError("요청의 크기를 확인하세요.")
                try:
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, RecursionError) as error:
                    raise ValueError("의사코드 요청의 형식을 확인하세요.") from error
                if not isinstance(payload, dict):
                    raise ValueError("의사코드 요청의 형식을 확인하세요.")
                data = program_model.generate(payload.get("source"))
                status = 200
            except ValueError as error:
                data = {"error": str(error)}
                status = 400
            self.send(json.dumps(data).encode(), "application/json", status)

    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as server:
        print(f"http://127.0.0.1:{PORT}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="초파리 Python 학습 대시보드")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "lab_state_v2.json")
    args = parser.parse_args()
    serve(Experiment(args.checkpoint))
