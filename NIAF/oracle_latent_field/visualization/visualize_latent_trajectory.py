#!/usr/bin/env python
"""Visualize temporal latent trajectories stored in oracle latent-field .npz outputs."""
import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np


DEFAULT_COLORS = {
    "z_sent": "#d4a017",
    "z_word": "#2fb344",
    "z_prior": "#37b24d",
    "z_adapt": "#1c7ed6",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project latent-token sequences from .npz files to PCA space and draw "
            "time-ordered trajectories with adjacent-step diagnostics."
        )
    )
    parser.add_argument(
        "--npz",
        type=Path,
        nargs="+",
        required=True,
        help="Input .npz file(s), or directories containing .npz files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob used when an --npz argument is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directory inputs with --pattern.",
    )
    parser.add_argument(
        "--latent_key",
        "--latent_keys",
        nargs="+",
        default=["z_sent"],
        help="Latent array key(s) to plot. Default: z_sent.",
    )
    parser.add_argument(
        "--mask_key",
        default="latent_mask",
        help="Boolean valid-token mask key. If absent or mismatched, all tokens are used.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments/NIAF/oracle_latent_field/latent_trajectory_viz"),
        help="Directory for PNG, JSON metrics, and projected coordinates.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2,
        choices=[2],
        help="Projection dimensions. Currently only 2D PCA plots are supported.",
    )
    parser.add_argument(
        "--no_step_plot",
        action="store_true",
        help="Only draw the PCA trajectory, without the adjacent-step distance panel.",
    )
    parser.add_argument(
        "--annotate_every",
        type=int,
        default=5,
        help="Annotate every Nth latent token in the PCA plot. Use 0 to disable.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title override.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=13.5,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=5.8,
        help="Figure height in inches.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def collect_npz_paths(paths, pattern, recursive=False):
    collected = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            collected.extend(sorted(p for p in iterator if p.is_file()))
        elif path.is_file():
            collected.append(path)
        else:
            raise FileNotFoundError(f"Missing input: {path}")
    if not collected:
        raise FileNotFoundError("No .npz inputs found.")
    return collected


def scalar_to_str(value):
    if value is None:
        return ""
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def scalar_to_int(value, default=0):
    try:
        array = np.asarray(value)
        if array.shape == ():
            return int(array.item())
        return int(array.reshape(-1)[0])
    except Exception:
        return int(default)


def safe_stem(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "latent"


def load_latent_sequence(data, latent_key, mask_key):
    if latent_key not in data:
        raise KeyError(f"Missing latent key '{latent_key}'. Available keys: {', '.join(data.files)}")
    latent = np.asarray(data[latent_key], dtype=np.float64)
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(
            f"Latent key '{latent_key}' must have shape [T,D] or [1,T,D], got {latent.shape}."
        )
    if latent.shape[0] < 2:
        raise ValueError(f"Latent key '{latent_key}' needs at least two tokens.")

    mask = None
    if mask_key in data:
        mask = np.asarray(data[mask_key]).astype(bool)
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 1 or len(mask) != len(latent):
            mask = None
    if mask is None:
        mask = np.ones(len(latent), dtype=bool)

    latent = latent[mask]
    if latent.shape[0] < 2:
        raise ValueError(f"Latent key '{latent_key}' has fewer than two valid tokens.")
    return latent, mask


def pca_fit_transform(sequences, dims=2):
    stacked = np.concatenate(sequences, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    component_count = min(dims, vt.shape[0])
    basis = vt[:component_count]
    coords = [(seq - mean) @ basis.T for seq in sequences]
    if component_count < dims:
        coords = [
            np.pad(item, ((0, 0), (0, dims - component_count)), mode="constant")
            for item in coords
        ]

    if len(stacked) > 1:
        variances = (singular_values ** 2) / float(len(stacked) - 1)
    else:
        variances = np.zeros_like(singular_values)
    total = float(variances.sum())
    explained = variances / total if total > 0.0 else np.zeros_like(variances)
    if len(explained) < dims:
        explained = np.pad(explained, (0, dims - len(explained)), mode="constant")

    return coords, mean.reshape(-1), basis, explained[:dims]


def pairwise_diameter(sequence):
    if len(sequence) < 2:
        return 0.0
    distances = np.linalg.norm(sequence[:, None, :] - sequence[None, :, :], axis=-1)
    return float(distances.max())


def step_metrics(sequence):
    step = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    diameter = pairwise_diameter(sequence)
    return {
        "mean": float(step.mean()),
        "median": float(np.median(step)),
        "std": float(step.std()),
        "min": float(step.min()),
        "max": float(step.max()),
        "p90": float(np.percentile(step, 90)),
        "p95": float(np.percentile(step, 95)),
        "max_transition_index": int(np.argmax(step) + 1),
        "trajectory_diameter": float(diameter),
        "max_step_over_diameter": float(step.max() / diameter) if diameter > 0.0 else None,
    }


def read_metadata(data):
    return {
        "name": scalar_to_str(data["name"]) if "name" in data else "",
        "text": scalar_to_str(data["text"]) if "text" in data else "",
        "gloss": scalar_to_str(data["gloss"]) if "gloss" in data else "",
        "label_word": scalar_to_str(data["label_word"]) if "label_word" in data else "",
        "length": scalar_to_int(data["length"], default=0) if "length" in data else 0,
        "source_index": scalar_to_int(data["source_index"], default=-1) if "source_index" in data else -1,
    }


def color_for_key(key, index):
    if key in DEFAULT_COLORS:
        return DEFAULT_COLORS[key]
    palette = [
        "#1c7ed6",
        "#f08c00",
        "#7048e8",
        "#0ca678",
        "#e03131",
        "#1098ad",
    ]
    return palette[index % len(palette)]


def plot_single_time_colored(ax, coords, annotate_every):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    points = coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.arange(len(segments))
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=plt.Normalize(0, max(len(segments) - 1, 1)),
    )
    collection.set_array(colors)
    collection.set_linewidth(2.8)
    ax.add_collection(collection)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.arange(len(coords)),
        cmap="viridis",
        s=34,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        s=95,
        marker="o",
        color="#2fb344",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="start",
    )
    ax.scatter(
        coords[-1, 0],
        coords[-1, 1],
        s=110,
        marker="X",
        color="#e03131",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="end",
    )
    annotate_tokens(ax, coords, annotate_every)
    return collection


def annotate_tokens(ax, coords, annotate_every):
    if annotate_every <= 0:
        return
    important = {0, len(coords) - 1}
    for idx in range(len(coords)):
        if idx % annotate_every == 0 or idx in important:
            ax.text(coords[idx, 0], coords[idx, 1], str(idx), fontsize=7, ha="left", va="bottom")


def plot_multi_key(ax, keys, coords_by_key, annotate_every):
    for index, key in enumerate(keys):
        coords = coords_by_key[key]
        color = color_for_key(key, index)
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2.4, label=key)
        ax.scatter(coords[:, 0], coords[:, 1], color=color, s=24, edgecolor="white", linewidth=0.5)
        ax.scatter(
            coords[0, 0],
            coords[0, 1],
            s=80,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            coords[-1, 0],
            coords[-1, 1],
            s=90,
            marker="X",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
        annotate_tokens(ax, coords, annotate_every)


def draw_figure(path, keys, sequences, coords, explained, metadata, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords_by_key = dict(zip(keys, coords))
    sequences_by_key = dict(zip(keys, sequences))
    width = args.fig_width
    height = args.fig_height
    if args.no_step_plot:
        fig, ax = plt.subplots(1, 1, figsize=(width * 0.62, height), dpi=args.dpi)
        step_ax = None
    else:
        fig = plt.figure(figsize=(width, height), dpi=args.dpi)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)
        ax = fig.add_subplot(grid[0, 0])
        step_ax = fig.add_subplot(grid[0, 1])

    if len(keys) == 1:
        collection = plot_single_time_colored(ax, coords[0], args.annotate_every)
        colorbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("latent token index")
        ax.legend(loc="best", fontsize=8)
    else:
        plot_multi_key(ax, keys, coords_by_key, args.annotate_every)
        ax.legend(loc="best", fontsize=8)

    ax.autoscale()
    ax.set_title("Latent trajectory (PCA 2D)", fontsize=12)
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}% var)")
    ax.grid(True, alpha=0.22)

    metrics_by_key = {}
    if step_ax is not None:
        for index, key in enumerate(keys):
            sequence = sequences_by_key[key]
            step = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
            token_index = np.arange(1, len(sequence))
            color = color_for_key(key, index)
            step_ax.plot(token_index, step, color=color, linewidth=2.0, label=key)
            step_ax.scatter(token_index, step, color=color, s=16)
            metrics_by_key[key] = step_metrics(sequence)

        if len(keys) == 1:
            key = keys[0]
            metrics = metrics_by_key[key]
            step_ax.axhline(
                metrics["mean"],
                color="#495057",
                linestyle="--",
                linewidth=1.2,
                label=f"mean {metrics['mean']:.3f}",
            )
            step_ax.axhline(
                metrics["p95"],
                color="#f08c00",
                linestyle=":",
                linewidth=1.5,
                label=f"95% {metrics['p95']:.3f}",
            )
            step_ax.scatter(
                [metrics["max_transition_index"]],
                [metrics["max"]],
                color="#e03131",
                s=70,
                zorder=3,
                label=f"max {metrics['max']:.3f}",
            )
        step_ax.set_title("Adjacent latent-token distance", fontsize=12)
        step_ax.set_xlabel("transition t-1 -> t")
        step_ax.set_ylabel("L2 distance in original latent space")
        step_ax.grid(True, alpha=0.22)
        step_ax.legend(loc="best", fontsize=8)
    else:
        metrics_by_key = {key: step_metrics(seq) for key, seq in zip(keys, sequences)}

    title = args.title
    if not title:
        name = metadata.get("name") or path.stem
        length = metadata.get("length") or 0
        title = f"{name} | {len(sequences[0])} latent tokens"
        if length:
            title += f" from {length} SMPL-X frames"
    fig.suptitle(title, fontsize=13, y=0.99)

    footer = metadata.get("text") or metadata.get("gloss") or metadata.get("label_word") or ""
    if footer:
        fig.text(
            0.5,
            0.01,
            textwrap.shorten(footer, width=160, placeholder="..."),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#495057",
        )
        fig.subplots_adjust(bottom=0.16, top=0.88)
    else:
        fig.subplots_adjust(bottom=0.12, top=0.88)

    return fig, metrics_by_key


def visualize_one(path, args):
    with np.load(path, allow_pickle=False) as data:
        metadata = read_metadata(data)
        sequences = []
        used_keys = []
        masks = {}
        for key in args.latent_key:
            sequence, mask = load_latent_sequence(data, key, args.mask_key)
            sequences.append(sequence)
            used_keys.append(key)
            masks[key] = {
                "total_tokens": int(len(mask)),
                "valid_tokens": int(mask.sum()),
            }

    coords, pca_mean, pca_components, explained = pca_fit_transform(sequences, dims=args.dims)

    key_stem = "-".join(safe_stem(key) for key in used_keys)
    out_stem = f"{safe_stem(path.stem)}_{key_stem}_pca{args.dims}d_trajectory"
    png_path = args.out_dir / f"{out_stem}.png"
    json_path = args.out_dir / f"{out_stem}_metrics.json"
    projection_path = args.out_dir / f"{out_stem}_projection.npz"

    fig, metrics_by_key = draw_figure(path, used_keys, sequences, coords, explained, metadata, args)
    fig.savefig(png_path, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

    projection_arrays = {
        "pca_mean": pca_mean.astype(np.float32),
        "pca_components": pca_components.astype(np.float32),
        "explained_variance_ratio": explained.astype(np.float32),
        "latent_keys": np.asarray(used_keys, dtype=str),
    }
    for key, sequence, coord in zip(used_keys, sequences, coords):
        safe_key = safe_stem(key)
        projection_arrays[f"{safe_key}_latent"] = sequence.astype(np.float32)
        projection_arrays[f"{safe_key}_pca"] = coord.astype(np.float32)
    np.savez_compressed(projection_path, **projection_arrays)

    result = {
        "source_npz": str(path),
        "output_png": str(png_path),
        "projection_npz": str(projection_path),
        "latent_keys": used_keys,
        "mask_key": args.mask_key,
        "masks": masks,
        "metadata": metadata,
        "pca": {
            "fit": "all selected valid latent tokens from this NPZ",
            "dims": int(args.dims),
            "explained_variance_ratio": [float(value) for value in explained],
        },
        "step_l2": metrics_by_key,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["metrics_json"] = str(json_path)
    return result


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_npz_paths(args.npz, args.pattern, recursive=args.recursive)

    results = []
    for path in paths:
        result = visualize_one(path, args)
        results.append(result)
        print(f"Saved: {result['output_png']}")
        print(f"Saved: {result['metrics_json']}")
        print(f"Saved: {result['projection_npz']}")

    summary_path = args.out_dir / "latent_trajectory_summary.json"
    summary_path.write_text(json.dumps({"renders": results}, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
