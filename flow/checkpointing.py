import json
import math
import re
import shutil
from pathlib import Path


def copy_checkpoint_file(src, dst):
    """Copy checkpoint bytes without preserving filesystem metadata.

    Some shared mounts reject timestamp updates used by shutil.copy2/copystat.
    """

    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f"{dst.name}.tmp")
    try:
        shutil.copyfile(src, tmp_path)
        tmp_path.replace(dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def maybe_save_top_k_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    args,
    epoch,
    global_step,
    score,
    top_k,
    top_checkpoints,
    save_checkpoint_fn,
    metric_name="flow",
):
    """Keep the best K validation checkpoints by a scalar score.

    Lower score is better. Stable aliases are written as best_01.pt, best_02.pt,
    and so on. best.pt is kept as a backward-compatible alias of best_01.pt.
    """

    top_k = int(top_k)
    score = float(score)
    if top_k <= 0 or not math.isfinite(score):
        return top_checkpoints, False

    checkpoint_dir = Path(checkpoint_dir)
    source_dir = checkpoint_dir / "top_k_sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    should_save = len(top_checkpoints) < top_k
    if not should_save and top_checkpoints:
        should_save = score < max(item["score"] for item in top_checkpoints)
    if not should_save:
        return top_checkpoints, False

    safe_metric = str(metric_name).replace("/", "_")
    source_path = source_dir / f"epoch_{epoch:04d}_step_{global_step:08d}_{safe_metric}_{score:.6f}.pt"
    save_checkpoint_fn(source_path, model, optimizer, args, epoch, global_step)

    updated = list(top_checkpoints)
    updated.append(
        {
            "score": score,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "source_path": source_path,
        }
    )
    updated.sort(key=lambda item: (item["score"], item["epoch"], item["global_step"]))

    overflow = updated[top_k:]
    updated = updated[:top_k]
    for item in overflow:
        path = Path(item["source_path"])
        if path.exists():
            path.unlink()

    for rank, item in enumerate(updated, start=1):
        alias_path = checkpoint_dir / f"best_{rank:02d}.pt"
        copy_checkpoint_file(Path(item["source_path"]), alias_path)
        item["alias_path"] = alias_path
        if rank == 1:
            copy_checkpoint_file(alias_path, checkpoint_dir / "best.pt")

    for rank in range(len(updated) + 1, top_k + 1):
        stale_path = checkpoint_dir / f"best_{rank:02d}.pt"
        if stale_path.exists():
            stale_path.unlink()

    metadata = []
    for rank, item in enumerate(updated, start=1):
        metadata.append(
            {
                "rank": rank,
                "metric": metric_name,
                "score": item["score"],
                "epoch": item["epoch"],
                "global_step": item["global_step"],
                "path": f"best_{rank:02d}.pt",
                "source_path": str(Path(item["source_path"]).relative_to(checkpoint_dir)),
            }
        )
    metadata_path = checkpoint_dir / "best_top_k.json"
    tmp_metadata_path = metadata_path.with_name(f"{metadata_path.name}.tmp")
    with tmp_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    tmp_metadata_path.replace(metadata_path)

    return updated, True


def load_top_k_checkpoints(checkpoint_dir, top_k, metric_name="flow"):
    """Recover existing validation-best checkpoint metadata for resumed runs."""

    top_k = int(top_k)
    if top_k <= 0:
        return []

    checkpoint_dir = Path(checkpoint_dir)
    candidates = []
    metadata_path = checkpoint_dir / "best_top_k.json"

    if metadata_path.is_file() and metadata_path.stat().st_size > 0:
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            for item in metadata:
                source_path = Path(item.get("source_path", ""))
                if not source_path.is_absolute():
                    source_path = checkpoint_dir / source_path
                score = float(item["score"])
                if source_path.is_file() and math.isfinite(score):
                    candidates.append(
                        {
                            "score": score,
                            "epoch": int(item.get("epoch", 0)),
                            "global_step": int(item.get("global_step", 0)),
                            "source_path": source_path,
                        }
                    )
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            candidates = []

    if not candidates:
        safe_metric = str(metric_name).replace("/", "_")
        pattern = re.compile(
            rf"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)_{re.escape(safe_metric)}_(?P<score>[-+0-9.eE]+)\.pt$"
        )
        source_dir = checkpoint_dir / "top_k_sources"
        if source_dir.is_dir():
            for path in source_dir.glob("*.pt"):
                match = pattern.match(path.name)
                if match is None:
                    continue
                try:
                    score = float(match.group("score"))
                except ValueError:
                    continue
                if math.isfinite(score):
                    candidates.append(
                        {
                            "score": score,
                            "epoch": int(match.group("epoch")),
                            "global_step": int(match.group("step")),
                            "source_path": path,
                        }
                    )

    candidates.sort(key=lambda item: (item["score"], item["epoch"], item["global_step"]))
    return candidates[:top_k]
