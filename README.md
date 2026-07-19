# Neural Implicit Sign Motion Trajectory

This repository is the standalone home of the neural implicit sign-motion
trajectory work originally developed inside SOKE. The extraction is
non-destructive: the source repository remains in place, while the NIAF code,
experiment archive, evaluations, and visualizations are copied here.

## Current Pipeline

The primary PHOENIX model produces a continuous trajectory instance rather than
a fixed frame tensor:

1. Sentence text is encoded with the local FLAN-T5 model.
2. A frozen SoftArranger adapter supplies retrieval and motion-plan context.
3. A hypernetwork predicts a finite `TrajectoryInstance`, including duration and
   grouped modulations for shared implicit fields.
4. Modulated SIREN prior and residual fields map arbitrary continuous query times
   to compact 256-D SMPL-X poses.
5. Sampling the same trajectory instance at any time grid produces motion for
   rendering, FK, or quantitative evaluation.

The detailed design is in
[`docs/NIAF/continuous_trajectory_field/implementation_plan.md`](docs/NIAF/continuous_trajectory_field/implementation_plan.md).

## Repository Layout

```text
NIAF/                 Neural implicit field implementations
flow/                 Shared adapter, dataset, SMPL-X, evaluation, and render code
mGPT/                 Joint metrics and rendering support used by evaluation
scripts/NIAF/         Slurm launchers for NIAF experiments
scripts/flow/         Launchers for frozen flow dependencies
docs/NIAF/            Designs, experiment summaries, and visualization guides
tests/                 Focused NIAF and GUAVA tests
experiments/NIAF/     Migrated checkpoints, evaluations, NPZ exports, plots, and videos
experiments/flow/     Exact frozen adapter/VAE dependencies used by NIAF
deps/                 Local FLAN-T5 and SMPL-X assets
logs/sbatch/          Migrated NIAF-related Slurm logs
```

Large artifacts are present on disk but ignored by Git intentionally. See
[`docs/MIGRATION.md`](docs/MIGRATION.md) for the exact migration boundary and
integrity notes.

## Environment

The migrated launchers default to the existing SOKE environment:

```bash
export PROJECT_DIR=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
export PYTHON_ENV=/media/cvpr/haomian/python_envs/SOKE
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
```

The motion datasets and generated FK/scaffold caches remain at
`/media/cvpr/haomian/data/SOKE_FLOW`. They were not duplicated because they are
shared data inputs, not project outputs.

## Common Commands

Run focused tests after installing the optional `pytest` dependency:

```bash
"$PYTHON_ENV/bin/python" -m pip install pytest
"$PYTHON_ENV/bin/python" -m pytest -q tests/test_niaf_continuous_trajectory_field.py
```

Launch the current full PHOENIX configuration:

```bash
CFG=NIAF/continuous_trajectory_field/configs/phoenix_continuous_trajectory_full_smooth_q32_jreg1e2_w10_b32x8.yaml \
RUN_TAG=phoenix_continuous_trajectory_full_smooth_q32_jreg1e2_w10_b32x8 \
sbatch scripts/NIAF/train_continuous_trajectory_field_sbatch.sh
```

Inspect export and evaluation options:

```bash
"$PYTHON_ENV/bin/python" -m NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory --help
"$PYTHON_ENV/bin/python" -m NIAF.continuous_trajectory_field.scripts.evaluate_continuous_trajectory_field --help
```
