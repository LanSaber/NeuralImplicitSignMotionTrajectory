#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from NIAF.oracle_smplx_field.config import deep_update, load_config
from NIAF.oracle_smplx_field.experiment import expand_grid, run_experiment


def comma_list(value):
    if value is None:
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit oracle continuous SMPL-X rot6D fields.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("NIAF/oracle_smplx_field/configs/how2sign_rot6d_fk_pilot.yaml"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--models", default=None, help="Comma-separated model IDs.")
    parser.add_argument("--time_modes", default=None, help="Comma-separated: uniform,joint_arclength,hand_arclength.")
    parser.add_argument(
        "--fit_modes",
        default=None,
        help="Comma-separated: fit_all,even_odd,odd_even,stride_4,block_middle_25,keyframe_sparse_8.",
    )
    parser.add_argument("--loss_schedules", default=None, help="Comma-separated: S1,S2,S3,S4.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_points", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--no_save_npz", action="store_true")
    parser.add_argument("--print_grid", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    updates = {}
    if args.out_dir is not None:
        updates.setdefault("output", {})["out_dir"] = str(args.out_dir)
    if args.max_sequences is not None:
        updates.setdefault("data", {})["max_sequences"] = int(args.max_sequences)
    if args.device is not None:
        updates["device"] = args.device
    grid_updates = {}
    for attr, key in [
        ("models", "models"),
        ("time_modes", "time_modes"),
        ("fit_modes", "fit_modes"),
        ("loss_schedules", "loss_schedules"),
    ]:
        value = comma_list(getattr(args, attr))
        if value is not None:
            grid_updates[key] = value
    if args.max_runs is not None:
        grid_updates["max_runs"] = int(args.max_runs)
    if grid_updates:
        updates["grid"] = grid_updates
    fit_updates = {}
    if args.steps is not None:
        fit_updates["steps"] = int(args.steps)
    if args.batch_points is not None:
        fit_updates["batch_points"] = int(args.batch_points)
    if fit_updates:
        updates["fit"] = fit_updates
    if args.save_npz:
        updates.setdefault("output", {})["save_npz"] = True
    if args.no_save_npz:
        updates.setdefault("output", {})["save_npz"] = False
    return deep_update(cfg, updates)


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    if args.print_grid:
        runs = expand_grid(cfg)
        print(json.dumps({"count": len(runs), "runs": runs}, indent=2, default=str))
        return
    summary = run_experiment(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
