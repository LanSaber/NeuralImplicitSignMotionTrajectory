# Standalone Project Migration

## Snapshot

- Migration date: 2026-07-20
- Source: `/media/cvpr/haomian/SOKE`
- Destination: `/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory`
- Method: non-destructive copy followed by a destination-only project-root refactor
- Python environment: `/media/cvpr/haomian/python_envs/SOKE`

The source repository and its running jobs were not moved, stopped, or modified.

## Included

- All five NIAF implementation families: oracle latent, oracle SMPL-X,
  continuous sign field, retrieval-confidence field, and continuous trajectory
  field.
- The complete `flow` and `mGPT` runtime packages required by those models.
- NIAF and flow launchers, NIAF-focused tests, and the source documentation
  snapshot. Historical documents unrelated to the active continuous-trajectory
  pipeline were pruned from the destination after migration.
- The complete `experiments/NIAF` archive: 5,529 files and 5,186,060,759 bytes
  at migration time.
- All migrated NPZ exports, JSON/CSV metrics, checkpoints, plots, and rendered
  videos inside that archive.
- 194 NIAF-related Slurm output/error logs totaling 87,154,510 bytes.
- Local FLAN-T5 and SMPL-X assets under `deps/`.
- Exact PHOENIX and How2Sign flow dependencies referenced by the migrated NIAF
  configurations: three adapter checkpoints and two VAE checkpoints, plus their
  lightweight experiment metadata.
- The complete CSL-Daily SoftArranger adapter checkpoint archive added in the
  2026-07-21 follow-up, together with its required VAE `best.pt` and config.

## CSL-Daily Follow-Up

The initial migration selected `experiments/flow` artifacts by following direct
references from the migrated NIAF configurations. That excluded the broader
CSL-Daily adapter history. On 2026-07-21, the following stable artifacts were
added non-destructively:

```text
experiments/flow/adapter/csl_daily_soft_arranger_lw_2xb256/
  config.json
  checkpoints/                 # 18 PT checkpoints plus best_top_k.json

experiments/flow/VAE/csl_daily_rot6d_vae_jerk_b16x4_online/
  config.json
  checkpoints/best.pt
```

The adapter's historical `config.json` is preserved verbatim and records the
original job-local `/dev/shm/soke_16396446/SOKE_FLOW/...` training paths. For a
new run, override those paths with the persistent CSL-Daily datasets under
`/media/cvpr/haomian/data/SOKE_FLOW`; checkpoint loading itself resolves the
migrated relative VAE path without modification.

## Active Training Snapshot

Slurm job `140255` was still running during the copy. Its experiment directory is
present as a point-in-time snapshot:

```text
experiments/NIAF/continuous_trajectory_field/
  phoenix_continuous_trajectory_full_smooth_q32_jreg1e2_w10_b32x8/
```

At copy time, `checkpoints/last.pt` reported epoch 87 and global step 2,436.
`checkpoints/best.pt` reported epoch 30 and global step 840. The source job was
left running, so later source-side checkpoints may advance beyond this snapshot.

## Intentionally External

- Datasets and generated dataset caches remain under
  `/media/cvpr/haomian/data/SOKE_FLOW` and are referenced directly by configs.
- The existing SOKE conda environment remains the default runtime.
- Online W&B history remains in W&B. Local experiment outputs and Slurm logs are
  included, but the remote service was not mirrored.
- Unrelated SOKE models, tasks, experiments, and logs were not copied.

## Path Policy

Executable source, launchers, and usage docs now use
`/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory` as the project root.
Historical configs, result JSON, checkpoint metadata, and Slurm logs are retained
verbatim when they record the old SOKE path; those values are provenance, not
active defaults.

## Verification

Verification completed from the destination with the SOKE Python environment:

- A full source/destination content comparison found all 5,527 static NIAF
  experiment files byte-identical. Only the active run's `metrics.jsonl` and
  `checkpoints/last.pt` changed at the source after the epoch-87 snapshot.
- FLAN-T5 and SMPL-X dependency-tree comparisons found zero differences.
- All five selected adapter/VAE checkpoints match their source SHA-256 hashes.
- All 22 CSL-Daily follow-up files match their source SHA-256 hashes. Adapter
  `best.pt` loads at epoch 90/global step 3,150 with 77 model-state tensors, and
  the required VAE loads at epoch 345/global step 99,015 with 187 tensors.
- The migrated full-run `last.pt` and `best.pt` load successfully and each
  contains 71 model-state tensors.
- All 289 Python files compile, and all NIAF, flow, and mGPT package imports pass.
- All 58 focused NIAF/GUAVA test functions pass. The shared environment does not
  currently include `pytest`, so verification invoked these fixture-free test
  functions directly, supplying a temporary path for the one `tmp_path` case.
- All 15 copied shell launchers pass `bash -n`; the continuous-trajectory
  training launcher also passes a destination-root dry run.
- Train, export, and evaluation `--help` entry points initialize successfully.
- All 31 NIAF YAML configs parse, and all 209 referenced data, cache, model, and
  checkpoint paths exist.
- `ARTIFACT_MANIFEST.sha256` contains 5,909 per-file hashes for `deps/`,
  `experiments/`, and `logs/sbatch/`.

To recheck the settled artifact archive:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
sha256sum -c ARTIFACT_MANIFEST.sha256
```
