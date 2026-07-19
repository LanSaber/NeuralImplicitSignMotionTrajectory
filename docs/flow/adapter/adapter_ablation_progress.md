# Adapter Ablation Progress

Snapshot time: 2026-06-25 17:33 +04.

This note tracks the current adapter ablation runs for Phoenix, How2Sign, and
CSL Daily. The ablations use the existing `flow/train_adapter.py` switches:

| Ablation | Flags | Meaning |
| --- | --- | --- |
| Full adapter | `--prior-mode soft_arranger` | SoftWordArranger prior + ContentStyleAdapter. |
| Disable arranger | `--disable-softarranger` | Use concat/first-match word prior; keep adapter enabled. |
| Disable adapter | `--disable-adapter` | Train/use arranger prior directly; bypass adapter. |
| Disable both | `--disable-softarranger --disable-adapter` | Deterministic concat prior only; no trainable adapter/arranger path. |

## Current Queue

| Job | Run | State at snapshot | Nodes / reason |
| --- | --- | --- | --- |
| `139694` | Phoenix inner SWA `full_v2` | Running, `17:36` elapsed | `ADUAED21032WKLX20` |
| `139695` | Phoenix inner SWA `nogate` | Running, `17:35` elapsed | `ADUAED21043WKLX04` |
| `139696` | Phoenix inner SWA `nonull` | Running, `17:35` elapsed | `ADUAED21044WKLX03` |
| `139698` | Phoenix inner SWA `noattnvar` | Running, `17:35` elapsed | `ADUAED21046WKLX02` |
| `139699` | Phoenix inner SWA `noanticollapse` | Running, `5:17` elapsed | `ADUAED21045WKLX28` |
| `139700` | Phoenix inner SWA `nowordtext` | Pending | `(Resources)` |
| `139701` | Phoenix inner SWA `nowordmotion` | Pending | `(Priority)` |
| `139702` | Phoenix inner SWA `noneg` r2 | Pending | `(Priority)` |
| `139682` | CSL Daily disable arranger | Running, `4:36` elapsed | `ADUAED21018WKLX24,ADUAED21031WKLX22` |
| `139683` | CSL Daily disable adapter | Pending | `(Resources)` |
| `139684` | CSL Daily disable both | Pending | `(Priority)` |
| `139679` | Phoenix disable adapter | Running, `8:37` elapsed | `ADUAED21034WKLX25,ADUAED21035WKLX15,ADUAED21039WKLX09,ADUAED21042WKLX08` |
| `139678` | How2Sign disable adapter | Running, `8:49` elapsed | `ADUAED21019WKLX23,ADUAED21037WKLX10` |
| `139677` | How2Sign disable both | Running, `9:09` elapsed | `ADUAED21023WKLX21,ADUAED21027WKLX14` |
| `139478` | How2Sign disable arranger | Running, `1-13:03` elapsed | `ADUAED21020WKLX07,ADUAED21038WKLX19` |

## Phoenix Inner SoftWordArranger

These runs isolate SoftWordArranger internals on Phoenix with arranger-only
training: `--prior-mode soft_arranger --disable-adapter`. They are fresh runs,
not resumes of the older Phoenix disable-adapter ablation.

Implementation added:
- `SoftWordArranger` runtime switches for candidate gates, NULL memory,
  candidate word-text features, and candidate motion-latent features.
- The switches preserve model parameter shapes and are stored in
  `arranger_config` so checkpoints/evaluation can reproduce the ablation.
- `scripts/flow/train_adapter_sbatch.sh` now accepts matching CLI flags and
  environment variables.
- `encode_word_candidates` now zeroes invalid padded candidate latents with
  `torch.where`; this fixes the no-negatives edge case where all-masked padded
  word clips could produce NaNs from the VAE encoder.

Validation before full submission:
- `python -m py_compile flow/temporal_word_attention.py flow/train_adapter.py flow/evaluate_adapter.py`: pass.
- `bash -n scripts/flow/train_adapter_sbatch.sh`: pass.
- `pytest` is not installed in the SOKE venv, so the no-fixture test file was
  executed with a direct runner: 11 tests passed.
- Smoke training passed for `full_v2`, `nogate`, `nonull`, and `nowordmotion`
  with `--limit-train 16 --limit-val 8 --epochs 2`.

Shared full-run settings:
- Sentence data: `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx`
- Word data: `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc`
- Word split: `train.balanced`; condition field: `gloss`
- VAE: `experiments/flow/VAE/phoenix_rot6d_vae_jerk_b16x4_online_retry1/checkpoints/best.pt`
- One node, one GPU task per experiment; `batch_size=256` per GPU.
- W&B online, `epochs=1000`, `val_every=10`, `save_every=100`,
  `save_last_every=10`.
- Candidates/losses use the V2 defaults: `K=32`, negatives `16`,
  `round_robin`, max positive variants `2`, no attention smoothness, and the
  anti-collapse loss bundle unless disabled by the ablation.

| Suffix | Output folder | Job | Status | Extra flags |
| --- | --- | --- | --- | --- |
| `full_v2` | `experiments/flow/adapter/phoenix_inner_swa_full_v2_noadapter_b256x1` | `139694` | Running; `config.json` written; epoch 2 finite | none |
| `nogate` | `experiments/flow/adapter/phoenix_inner_swa_nogate_noadapter_b256x1` | `139695` | Running; `config.json` written; epoch 2 finite | `--disable-arranger-candidate-gates --gate-bce-loss-weight 0 --gate-sparsity-loss-weight 0` |
| `nonull` | `experiments/flow/adapter/phoenix_inner_swa_nonull_noadapter_b256x1` | `139696` | Running; `config.json` written; epoch 2 finite | `--disable-arranger-null-memory --null-usage-loss-weight 0` |
| `noneg` | `experiments/flow/adapter/phoenix_inner_swa_noneg_noadapter_b256x1_r2` | `139702` | Pending `(Priority)` after NaN fix | `--num-word-candidates 16 --num-negative-candidates 0 --negative-usage-loss-weight 0` |
| `noattnvar` | `experiments/flow/adapter/phoenix_inner_swa_noattnvar_noadapter_b256x1` | `139698` | Running; `config.json` written; epoch 2 finite | `--attention-variation-loss-weight 0` |
| `noanticollapse` | `experiments/flow/adapter/phoenix_inner_swa_noanticollapse_noadapter_b256x1` | `139699` | Running; computing latent stats, no config yet | coverage, attention variation, prior dynamics, variance floor, and negative-usage loss weights set to `0` |
| `nowordtext` | `experiments/flow/adapter/phoenix_inner_swa_nowordtext_noadapter_b256x1` | `139700` | Pending `(Resources)`; no config/log yet | `--disable-arranger-word-text-features` |
| `nowordmotion` | `experiments/flow/adapter/phoenix_inner_swa_nowordmotion_noadapter_b256x1` | `139701` | Pending `(Priority)`; no config/log yet | `--disable-arranger-word-motion-latents` |

Notes:
- Four-node jobs `139686`-`139693` were cancelled because they remained queued.
- One-node job `139697` (`noneg`) reached epoch 1 but produced NaNs because the
  no-negative setting leaves padded all-masked candidates. It is superseded by
  `139702`, launched after the padding-latent fix.

## Phoenix

Reference full run:
`experiments/flow/adapter/phoenix_soft_arranger_adapter_balanced_b64x4_gloss_online`

| Ablation | Output folder | Job | Status | Latest evidence |
| --- | --- | --- | --- | --- |
| Disable arranger | `experiments/flow/adapter/phoenix_ablate_disable_arranger_b256x4_gloss_online` | `139475` | Complete | `Training complete: epoch=1000 global_step=6000`; `epoch_1000.pt` and `last.pt` exist. |
| Disable adapter | `experiments/flow/adapter/phoenix_ablate_disable_adapter_b256x4_gloss_online` | `139679` | Running normally | Relaunched as `r2`; `last.pt` is `epoch=450 global_step=2700` and was updated at `02:29`. |
| Disable both | `experiments/flow/adapter/phoenix_ablate_disable_both_b256x4_gloss_online` | `139477` | Complete | `Training complete: epoch=1000 global_step=6000`; `epoch_1000.pt` and `last.pt` exist. |

Notes:
- Earlier Phoenix disable-adapter attempts failed due Slurm launch or NCCL issues.
  The active job is `139679`, after cancelling `139675` because rank 3 reported an
  NCCL watchdog hang.

## How2Sign

Reference full run:
`experiments/flow/adapter/how2sign_soft_arranger_signasl_all_b256x2_k64_rr4_online`

| Ablation | Output folder | Job | Status | Latest evidence |
| --- | --- | --- | --- | --- |
| Disable arranger | `experiments/flow/adapter/how2sign_ablate_disable_arranger_signasl_all_b256x2_k64_rr4_online` | `139478` | Running normally | `last.pt` is `epoch=220 global_step=12980` and was updated at `01:47`. |
| Disable adapter | `experiments/flow/adapter/how2sign_ablate_disable_adapter_signasl_all_b256x2_k64_rr4_online` | `139678` | Running normally | Resumed from `epoch=10 global_step=590`; `last.pt` is `epoch=20 global_step=1180` and was updated at `01:01`. |
| Disable both | `experiments/flow/adapter/how2sign_ablate_disable_both_signasl_all_b256x2_k64_rr4_online` | `139677` | Running normally | Resumed from `epoch=100 global_step=5900`; `last.pt` is `epoch=190 global_step=11210` and was updated at `02:26`. |

Notes:
- How2Sign disable-adapter job `139676` was cancelled after it sat at DDP
  wrapping for about 15 minutes. It was relaunched as `139678` on a different
  node pair.
- How2Sign disable-both has no trainable parameters, but it still iterates the
  dataset to report deterministic prior metrics and save checkpoints.

## CSL Daily

Reference full run from the other HPC:
`experiments/flow/adapter/csl_daily_soft_arranger_lw_2xb256`

The copied full run has checkpoints through `epoch_1000.pt` plus `last.pt`.
Its config used `condition_field=label_word`, `word_split=all`, batch size
`256`, two nodes, `num_word_candidates=32`, `num_negative_candidates=16`,
`candidate_selection=round_robin`, and `max_positive_variants_per_key=2`.

### Label-word data overlay

Uploaded manifests:
`/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/data_aug/csl_daily_upper_smplx/meta`

The uploaded folder contains only `meta/`, so a metadata-only overlay was made:
`/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/data_aug/csl_daily_upper_smplx_label_word_abs`

The overlay rewrites each `motion_path` to an absolute path under the original
local motion dataset:
`/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx`

Validation before launch:
- Train rows: `18399`, matching the original CSL train split.
- Val rows: `1077`, matching the original CSL val split.
- Test rows: `1176`, matching the original CSL test split.
- `label_word` is non-empty for `18398/18399` train rows and all val/test rows.
- A single-sample dataset load succeeded with `condition_field=label_word`.
- A 200-sample matcher sanity check with the CSL word prior gave average span
  coverage around `0.759`; a few labels in that sample had no word-prior hit.

Submitted ablations:

| Ablation | Output folder | Job | Status |
| --- | --- | --- | --- |
| Disable arranger | `experiments/flow/adapter/csl_daily_ablate_disable_arranger_lw_2xb256` | `139682` | Running; no `last.pt` yet |
| Disable adapter | `experiments/flow/adapter/csl_daily_ablate_disable_adapter_lw_2xb256` | `139683` | Pending |
| Disable both | `experiments/flow/adapter/csl_daily_ablate_disable_both_lw_2xb256` | `139684` | Pending |

Update at `02:31 +04`:
- `139682` is running on `ADUAED21018WKLX24,ADUAED21031WKLX22`; no checkpoint
  has been written yet.
- `139683` is pending on resources; `139684` is pending on priority.

Shared CSL launch settings:
- `--data-dir /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/data_aug/csl_daily_upper_smplx_label_word_abs`
- `--stats-data-dir /media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx`
- `--word-data-dir /media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx_word_ctc`
- `--word-split all`
- `--condition-field label_word`
- `--batch-size 256`
- `--nodes 2`, `--ntasks-per-node 1`
- `--num-word-candidates 32`
- `--num-negative-candidates 16`
- `--candidate-selection round_robin`
- `--max-positive-variants-per-key 2`
- Anti-collapse losses copied from the full CSL config.

## Monitoring Commands

```bash
squeue -u "$USER" -o '%.18i %.9P %.40j %.8T %.12M %.6D %R'
```

```bash
find experiments/flow/adapter -maxdepth 3 -path '*ablate*checkpoints*' -type f -name '*.pt' | sort
```

```bash
tail -f logs/sbatch/flow/adapter/<job-name>_<job-id>.out
tail -f logs/sbatch/flow/adapter/<job-name>_<job-id>.err
```

## Known Infrastructure Notes

Recent restarts excluded nodes that previously showed Slurm launch failures,
DDP wrapping stalls, or NCCL watchdog failures:

`ADUAED21044WKLX03,ADUAED21040WKLX29,ADUAED21041WKLX30,ADUAED21022WKLX06,ADUAED21032WKLX20,ADUAED21045WKLX28,ADUAED21046WKLX02,ADUAED21043WKLX04`
