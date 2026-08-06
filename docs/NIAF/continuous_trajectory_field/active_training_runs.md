# Active SignTrajField Training Registry

This file is the canonical identity registry for live and queued SignTrajField
runs. Read it before answering an unqualified question such as "What is the
current training progress?" The scheduler and logs remain the source of truth
for live state; this registry determines **which run** the question refers to.

Last observed: **2026-07-22 23:19 Asia/Dubai (UTC+04:00)**

## Default run resolution

`DEFAULT_RUN_ALIAS: csl-daily-stage2-joint-scratch-20260722`

Unless the user names a dataset, job ID, output directory, or another alias,
"the training", "current training", and "the active training" refer to that
CSL-Daily run. Do not select a run merely because it has the highest Slurm job
ID or is already in the `RUNNING` state.

## Default: CSL-Daily Stage 2, joint scratch training

| Field | Value |
|---|---|
| Alias | `csl-daily-stage2-joint-scratch-20260722` |
| Dataset | CSL-Daily sentence-level train/validation splits (18,399/1,077 samples) |
| Intended state | Resumed from the epoch-9 recovery checkpoint; training epoch 10 |
| Training job | Slurm `140721`, name `csl_stf_s2_resume9` (`RUNNING`) |
| Previous job | Slurm `140592`, name `csl_stf_s2_joint`; cancelled after it stalled following epoch 9 |
| Configuration | `NIAF/continuous_trajectory_field/configs/csl_daily_continuous_trajectory_stage2_part_experts_joint_full.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/csl_daily_continuous_trajectory_stage2_part_experts_joint_full` |
| Initialization | Originally fresh Stage-2 joint training; resumed model and optimizer from `checkpoints/last.pt` at epoch 9, `global_step=648` |
| Adapter | `experiments/flow/adapter/csl_daily_soft_arranger_lw_2xb256/checkpoints/best.pt` at adapter epoch 90 (frozen) |
| Conditioning | Sentence `text`; the checkpoint's `label_word` condition falls back to text because sentence manifests have no `label_word` |
| Retrieval bank | Train-only `manifest_train.jsonl`: 133,689 word segments and 2,000 lexicon keys |
| Epochs | 50 |
| Allocation | 4 nodes, 1 GPU per node, DDP/NCCL, five-day limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally |
| Data loader | Synchronous (`num_workers: 0`, `persistent_workers: false`) on resume to avoid the prior shared-filesystem worker stall |
| Learning rates | Global `2e-4`; local `1e-4` from epoch 1 |
| W&B | Project `soke-niaf-continuous-trajectory`; run name `signtrajfield-csl-daily-stage2-joint-scratch`; ID `csldstfs2j20260722` |

### Resume status

Job `140721` restored the rolling epoch-9 checkpoint and the original W&B run,
then advanced through epoch-10 logical batch `54/144` with no reported error.
Epoch 10 is the analytic-jerk activation boundary. The original job `140592` was
cancelled only after its four ranks remained inactive after epoch 9; its
completed checkpoint is preserved.

The prerequisite FK, scaffold, and retrieval caches are complete. The
five-epoch prelaunch smoke run is separate from this full training; its output
is `experiments/NIAF/continuous_trajectory_field/csl_daily_stage2_joint_prelaunch_smoke`.

### Live tracing commands

```bash
squeue -j 140721 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 140721
tail -F logs/sbatch/csl_stf_s2_resume9_140721.out
tail -F logs/sbatch/csl_stf_s2_resume9_140721.err
```

Epoch-level progress is written to:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_continuous_trajectory_stage2_part_experts_joint_full/metrics.jsonl
```

The resumed process appends to the same output directory and resumes the same
online W&B run rather than creating a replacement experiment identity.

## Secondary: How2Sign Stage 2, joint scratch training

| Field | Value |
|---|---|
| Alias | `how2sign-stage2-joint-scratch-20260722` |
| Dataset | How2Sign sentence-level train/validation splits |
| Intended state | Resumed from the epoch-9 recovery checkpoint; training epoch 10 |
| Training job | Slurm `140728`, name `h2s_stf_s2_resume9c` (`RUNNING`) |
| Previous jobs | Slurm `140587` failed after epoch 9; first resume `140722` was cancelled after two allocated nodes became `NOT_RESPONDING`; `140723` exposed the remaining epoch-10 memory stall on healthy nodes |
| Configuration | `NIAF/continuous_trajectory_field/configs/how2sign_continuous_trajectory_stage2_part_experts_joint_full.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/how2sign_continuous_trajectory_stage2_part_experts_joint_full` |
| Initialization | Originally fresh Stage-2 joint training; resumed model and optimizer from `checkpoints/last.pt` at epoch 9, `global_step=1080` |
| Adapter | `experiments/flow/adapter/how2sign_soft_arranger_signasl_all_b256x4_online_r3_gloo/checkpoints/best.pt` (frozen) |
| Conditioning | Sentence `text` |
| Epochs | 50 |
| Allocation | 4 nodes, 1 GPU per node, DDP/NCCL, five-day limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally |
| Memory split | At most 16 samples per in-memory microbatch; logical and optimizer batch sizes are unchanged |
| Data loader | Synchronous (`num_workers: 0`, `persistent_workers: false`) on resume to avoid persistent-worker waits on the shared cache filesystem |
| Learning rates | Global `2e-4`; local `1e-4` from epoch 1 |
| W&B | Project `soke-niaf-continuous-trajectory`; run name `signtrajfield-how2sign-stage2-joint-scratch`; ID `h2sstfs2j20260722` |

### Resume status

The first resume, job `140722`, restored the rolling epoch-9 checkpoint and the
original W&B run, but two allocated nodes became `NOT_RESPONDING`. Replaying on
healthy nodes showed a second, deterministic issue: the 32-sample third-order
analytic jerk JVP exhausted each GB10 node's 119--121 GiB unified memory and
swapped indefinitely. A bounded four-rank run (`140727`) completed three
epoch-10 batches after capping only the in-memory split at 16 samples. The
production resume `140728` uses that verified setting, excludes all four nodes
from the bad allocation, and has advanced through logical batch `6/240`,
including online W&B optimizer-step logs. It resumes the same epoch-9 model,
optimizer, output directory, and W&B identity. The prerequisite FK, scaffold,
and retrieval caches remain complete.

The five-epoch prelaunch smoke run is not the full training. Its output is
`experiments/NIAF/continuous_trajectory_field/how2sign_stage2_joint_prelaunch_smoke`.

### Live tracing commands

```bash
squeue -j 140728 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 140728
tail -F logs/sbatch/h2s_stf_s2_resume9c_140728.out
tail -F logs/sbatch/h2s_stf_s2_resume9c_140728.err
```

Epoch-level progress is written to:

```text
experiments/NIAF/continuous_trajectory_field/how2sign_continuous_trajectory_stage2_part_experts_joint_full/metrics.jsonl
```

The resumed process appends to the same output directory and resumes the same
online W&B run rather than creating a replacement experiment identity.

## Secondary: PHOENIX Stage-2 resumed training

This run is still active, but it is **not** the default for unqualified status
questions.

| Field | Value |
|---|---|
| Alias | `phoenix-stage2-resume-epoch4-20260721` |
| Dataset | PHOENIX |
| Training job | Slurm `140556`, name `stf_s2_resume_wb` |
| Last observed state | `RUNNING`; completed epoch 8, `global_step=888` |
| Output | `experiments/NIAF/continuous_trajectory_field/phoenix_continuous_trajectory_stage2_part_experts_full` |
| Configuration | `NIAF/continuous_trajectory_field/configs/phoenix_continuous_trajectory_stage2_part_experts_full.yaml` |
| Allocation | 1 GPU |
| Batch semantics | 32 samples/loader batch, 2 accumulation steps, effective batch 64 |
| W&B | Project `soke-niaf-continuous-trajectory`; run ID `stfs2e4r20260721` |

Use this run only when the user says PHOENIX, job `140556`, the PHOENIX output
directory, or its alias.

## Registry maintenance protocol

1. Add every new full training launch before ending the launch task. Record a
   stable alias, dataset, Slurm job IDs, config, output, initialization mode,
   batch semantics, and W&B identity.
2. Change `DEFAULT_RUN_ALIAS` when the user launches or explicitly designates a
   different run as the default.
3. For a progress request, resolve the target here first and then query Slurm,
   logs, `metrics.jsonl`, and W&B as appropriate. Live observations supersede
   the snapshot in this document.
4. Update the last-observed timestamp and state after material transitions such
   as cache completion, training start, resume, failure, cancellation, or final
   completion.
5. Keep completed or failed runs as named historical entries until a concise
   experiment summary records their final outcome; never silently reuse an
   alias for a different output directory.
