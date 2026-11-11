"""Non-AI teachers ONLY. Never imported by FlyCode inference."""

from evaluation.datasets import Case, split_cases
from flybrain.decoder import EOS
from ir.grammar import validate


def code_cases():
    cases, details = [], {}
    # Eight held-out groups x three shapes = 24 hidden programs. This is surface
    # generalization within LET/PRINT, not 24 independent algorithm families.
    for group in range(40):
        name = f"value_{group}"
        value = group * 7 - 51
        for shape in ("let", "print", "let_print"):
            identity = f"code-{group}-{shape}"
            if shape == "let":
                source = f"LET {name} {value}\n"
                target = f"{name} = {value}\n"
                actions = [8, 1, 3, 1, 9, 2, EOS]
                expected = {"stdout": "", "variables": {name: value}}
            elif shape == "print":
                source = f"PRINT {value}\n"
                target = f"print({value})\n"
                actions = [4, 5, 8, 6, 2, EOS]
                expected = {"stdout": f"{value}\n", "variables": {}}
            else:
                source = f"LET {name} {value}\nPRINT {name}\n"
                target = f"{name} = {value}\nprint({name})\n"
                actions = [8, 1, 3, 1, 9, 2, 4, 5, 11, 6, 2, EOS]
                expected = {"stdout": f"{value}\n", "variables": {name: value}}
            validate(source)
            compile(target, "<teacher>", "exec")
            cases.append(Case(identity, f"name-value-group-{group}", source, target))
            details[identity] = {"actions": actions, "expected": expected, "shape": shape}
    splits = split_cases(cases)
    return {name: [{**case, **details[case["id"]]} for case in records] for name, records in splits.items()}
