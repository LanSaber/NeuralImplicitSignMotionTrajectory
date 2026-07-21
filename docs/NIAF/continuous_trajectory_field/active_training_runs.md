# Active SignTrajField Training Registry

This file is the canonical identity registry for live and queued SignTrajField
runs. Read it before answering an unqualified question such as "What is the
current training progress?" The scheduler and logs remain the source of truth
for live state; this registry determines **which run** the question refers to.

Last observed: **2026-07-22 01:34 Asia/Dubai (UTC+04:00)**

## Default run resolution

`DEFAULT_RUN_ALIAS: how2sign-stage2-joint-scratch-20260722`

Unless the user names a dataset, job ID, output directory, or another alias,
"the training", "current training", and "the active training" refer to that
How2Sign run. Do not select a run merely because it has the highest Slurm job
ID or is already in the `RUNNING` state.

## Default: How2Sign Stage 2, joint scratch training

| Field | Value |
|---|---|
| Alias | `how2sign-stage2-joint-scratch-20260722` |
| Dataset | How2Sign sentence-level train/validation splits |
| Intended state | Preparing required caches; full training is dependency-gated |
| Training job | Slurm `140587`, name `h2s_stf_s2_joint` |
| Configuration | `NIAF/continuous_trajectory_field/configs/how2sign_continuous_trajectory_stage2_part_experts_joint_full.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/how2sign_continuous_trajectory_stage2_part_experts_joint_full` |
| Initialization | Fresh Stage-2 initialization; joint global/local optimization from epoch 1; no warm start and no resume |
| Adapter | `experiments/flow/adapter/how2sign_soft_arranger_signasl_all_b256x4_online_r3_gloo/checkpoints/best.pt` (frozen) |
| Conditioning | Sentence `text` |
| Epochs | 50 |
| Allocation | 4 nodes, 1 GPU per node, DDP/NCCL, five-day limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally |
| Learning rates | Global `2e-4`; local `1e-4` from epoch 1 |
| W&B | Project `soke-niaf-continuous-trajectory`; run name `signtrajfield-how2sign-stage2-joint-scratch`; ID `h2sstfs2j20260722` |

### Dependency jobs

| Job | Purpose | Last observed state |
|---|---|---|
| `140585` (`h2s_stf_fk_cache`) | Full training-split FK cache | `RUNNING`; training job dependency |
| `140586` (`h2s_stf_scaffold_cache`) | Full train/validation frozen-adapter scaffold and retrieval-evidence cache | `RUNNING`; training job dependency |
| `140588` (`h2s_stf_fk_val_cache`) | Full validation-split FK cache | Completed successfully: 1,713 written and 4 smoke-cache entries skipped |
| `140587` (`h2s_stf_s2_joint`) | Four-GPU full training | `PENDING (Dependency)` on `140585` and `140586` |

The five-epoch prelaunch smoke run is not the full training. Its output is
`experiments/NIAF/continuous_trajectory_field/how2sign_stage2_joint_prelaunch_smoke`.

### Live tracing commands

```bash
squeue -j 140585,140586,140587 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 140587
tail -c 20000 logs/sbatch/h2s_stf_fk_cache_140585.err | tr -d '\000'
tail -c 20000 logs/sbatch/h2s_stf_scaffold_cache_140586.err | tr -d '\000'
tail -F logs/sbatch/h2s_stf_s2_joint_140587.out
tail -F logs/sbatch/h2s_stf_s2_joint_140587.err
```

After job `140587` starts, epoch-level progress is written to:

```text
experiments/NIAF/continuous_trajectory_field/how2sign_continuous_trajectory_stage2_part_experts_joint_full/metrics.jsonl
```

W&B will not create the online run until the dependency-gated training process
actually starts.

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
