from pathlib import Path
from tempfile import TemporaryDirectory

from lab import HERE, Experiment
from reward import RewardMemory


# These prompts use shapes outside the 100-point curriculum checks.
TRANSFER = [
    ("limit □ {'min': 1, 'max': 9}", "="),
    ("if ready and count < 9□\n    move()", ":"),
    ("if rows and active:\n□read(rows)", "    "),
    ("def get():\n    □ sorted(items)", "return"),
    ("colors = □'red', 'blue', 'green']", "["),
    ("for item □ sorted(rows):\n    print(item.name)", "in"),
    ("□ counter(x=1):\n    return x", "def"),
    ("from pathlib □ Path, PurePath as P", "import"),
    ("□ def process(items):\n    await send(items)", "async"),
    ("try:\n    result = get(x)\n□ (ValueError, TypeError):\n    log()", "except"),
    ("□ decorate(2, 3)\ndef run():\n    pass", "@"),
    ("match value:\n    □ (x, y) if x > 0:\n        plot(x, y)", "case"),
]


def main():
    for prompt, token in TRANSFER:
        compile(prompt.replace("□", token), "transfer", "exec")
    transfer = [(0, prompt, token) for prompt, token in TRANSFER]
    for graph in ("flywire_lab.csv", "flywire_lab_2048.csv"):
        for prioritized in (False, True):
            with TemporaryDirectory() as folder:
                lab = Experiment(Path(folder) / "state.json", delay=0,
                                 graph=HERE / graph, prioritized=prioritized)
                for _ in range(260):
                    if lab.mastered():
                        break
                    lab.batch(practice=lab.state["stage"] > len(lab.lessons))
                state = lab.state
                score = state["history"][-1]
                transfer_model = RewardMemory(lab.fly, lab.train_examples, transfer, state["feedback"])
                extra = sum(transfer_model.predict(prompt) == token for _, prompt, token in transfer)
                policy = "prioritized" if prioritized else "uniform"
                trials = state["positive"] + state["negative"]
                print(f"{lab.fly.node_count:>4} neurons | {policy:11} | {trials:>6} attempts | "
                      f"{score['correct']}/{score['total']} curriculum | {extra}/{len(transfer)} transfer")


if __name__ == "__main__":
    main()
