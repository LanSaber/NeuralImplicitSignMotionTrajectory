from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(handle)
        else:
            config = json.load(handle)
    base_path = config.pop("_base_", None)
    if base_path is None:
        return config
    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return deep_update(load_config(base_path), config)


def deep_update(base, updates):
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out
