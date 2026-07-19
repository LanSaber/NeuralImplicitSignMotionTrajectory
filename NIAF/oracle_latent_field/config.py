from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def deep_update(base, updates):
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_latent_stats_from_config(path):
    cfg = load_config(path)
    latent_cfg = cfg.get("latent_config") or {}
    stats = latent_cfg.get("stats") or {}
    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"{path} does not contain latent_config.stats.mean/std")
    return stats
