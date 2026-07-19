from __future__ import annotations

import json
from pathlib import Path


class ScalarAverager:
    def __init__(self):
        self.total = {}
        self.count = {}

    def update(self, values, n=1, prefix=None):
        for key, value in values.items():
            if value is None:
                continue
            name = f"{prefix}_{key}" if prefix else key
            self.total[name] = self.total.get(name, 0.0) + float(value) * int(n)
            self.count[name] = self.count.get(name, 0) + int(n)

    def mean(self):
        return {key: self.total[key] / max(self.count[key], 1) for key in sorted(self.total)}


def tensor_dict_to_float(losses):
    out = {}
    for key, value in losses.items():
        if hasattr(value, "detach"):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)
    return out


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

