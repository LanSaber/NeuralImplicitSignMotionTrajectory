# Active SignTrajField Training Registry

This file is the canonical identity registry for live and queued SignTrajField
runs. Read it before answering an unqualified question such as "What is the
current training progress?" The scheduler and logs remain the source of truth
for live state; this registry determines **which run** the question refers to.

Last observed: **2026-08-10 15:07 Asia/Dubai (UTC+04:00)**

## Default run resolution

`DEFAULT_RUN_ALIAS: phoenix-signtrajfield-v2-full-syncfix-r2-20260806`

Unless the user names a dataset, job ID, output directory, or another alias,
"the training", "current training", and "the active training" refer to the
full SignTrajField-v2 PHOENIX run below. Do not select a run merely because it
has the highest Slurm job ID or is already in the `RUNNING` state.

Dataset-qualified requests override that default: "the How2Sign training"
refers to `how2sign-signtrajfield-v2-full-20260807`, and "the CSL-Daily
training" refers to `csl-daily-signtrajfield-v2-full-20260807`.

## Default: full PHOENIX-2014T SignTrajField-v2 training

| Field | Value |
|---|---|
| Alias | `phoenix-signtrajfield-v2-full-syncfix-r2-20260806` |
| Dataset | Full PHOENIX-2014T train/validation splits (7,092/519 samples) |
| Intended state | Fresh contract-v2 dual-mode full training, explicitly launched after reviewing the five-sequence gate |
| Training job | Slurm `141033`, name `phx_stfv2_r2` (`COMPLETED`) |
| Failed predecessor | Slurm `141032`, name `phx_stfv2_full`; terminated in epoch 1 before writing a checkpoint |
| Configuration | `NIAF/continuous_trajectory_field/configs/phoenix_signtrajfield_v2_full.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/phoenix_signtrajfield_v2_full_syncfix_r2` |
| Initialization | Fresh v2 model; the predecessor had no checkpoint to resume |
| Adapter | Frozen PHOENIX SoftArranger epoch-400 checkpoint used as the optional word prior |
| Conditioning | Sentence `text`; optional word prior is dropped for 50% of training samples |
| Epochs | 50; checkpoint every epoch and dual-mode validation every 5 epochs |
| Allocation | 4 nodes, 1 GPU per node, DDP/NCCL, three-day limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally |
| Memory split | At most 16 samples per in-memory microbatch; all ranks use the smallest locally safe frame-budget size, so their DDP backward schedules match |
| Data loader | Synchronous (`num_workers: 0`) |
| W&B | Online project `soke-niaf-continuous-trajectory`; run `signtrajfield-v2-phoenix-full-syncfix-r2-20260806`; ID `phxstfv2fullr2_20260806` |

The retry completed all 50 epochs/global step 1,400 and synced W&B. Its final
text-only validation total is `10.8316`, but the word-prior total is `11.4528`
and the 5.87% selection degradation exceeds the 2% feasibility limit;
`best.pt` therefore remains the feasible epoch-10 checkpoint. Job `141032`
could not be resumed because its rank-local frame budgets caused unequal
backward counts and it failed before its first checkpoint. The user explicitly
authorized full training despite the five-sequence PA-hand/path continuation
failure; keep that caveat attached to later model-selection and test reporting.

### Live tracing commands

```bash
squeue -j 141033 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 141033
tail -F logs/sbatch/phx_stfv2_r2_141033.out
tail -F logs/sbatch/phx_stfv2_r2_141033.err
```

Epoch metrics are written to:

```text
experiments/NIAF/continuous_trajectory_field/phoenix_signtrajfield_v2_full_syncfix_r2/metrics.jsonl
```

## Previous: PHOENIX SignTrajField-v2 five-sequence gate

| Field | Value |
|---|---|
| Alias | `phoenix-signtrajfield-v2-overfit5-20260806` |
| Dataset | PHOENIX train split; five train samples and the same five validation samples |
| Intended state | Fresh contract-v2 dual-mode overfit gate before pilot/full training |
| Training job | Slurm `141025`, name `phx_stfv2_o5` (`COMPLETED`) |
| Configuration | `NIAF/continuous_trajectory_field/configs/phoenix_signtrajfield_v2_overfit5.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/phoenix_signtrajfield_v2_overfit5` |
| Initialization | Fresh v2 model; no v1 warm start and no resume checkpoint |
| Adapter | `experiments/flow/adapter/phoenix_soft_arranger_adapter_ctc_all_text_noneg_k64_b128x4_v2_online/checkpoints/epoch_0400.pt` (frozen) |
| Conditioning | Sentence `text`; optional word prior is dropped for 50% of samples |
| Epochs | 200; checkpoint every epoch and dual-mode validation every 10 epochs |
| Allocation | 1 node, 1 GPU, non-DDP, three-day limit |
| Batch semantics | 5 samples/GPU, one optimizer step per epoch |
| Data loader | Synchronous (`num_workers: 0`) |
| W&B | Online project `soke-niaf-continuous-trajectory`; run `signtrajfield-v2-phoenix-overfit5-20260806`; ID `phxstfv2o5_20260806` |

The run completed epoch 200/global step 200 and synced W&B successfully. Its
final selection score is `4.3622`, with feasible status and zero prior-regression
constraint violation. `best.pt` and `last.pt` both select epoch 200.

Evaluation jobs `141029`--`141031` completed aligned, default-DTW, and PA-DTW
evaluation on the exact five training sequences. Default nDTW beats the paired
scaffold for every part, but PA left/right hands regress by 10.83%/13.26%, hand
path loss is `0.60006` versus scaffold `0.30831`, and jerk ratio is `1.106`.
Pilot-500 remains blocked.

This is the mandatory five-sequence continuation gate, not the full PHOENIX
experiment. Inspect its dual-mode validation and path-quality results before
launching pilot-500; launch the full split only if the pilot subsequently passes.

### Live tracing commands

```bash
squeue -j 141025 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 141025
tail -F logs/sbatch/phx_stfv2_o5_141025.out
tail -F logs/sbatch/phx_stfv2_o5_141025.err
```

Epoch metrics are written to:

```text
experiments/NIAF/continuous_trajectory_field/phoenix_signtrajfield_v2_overfit5/metrics.jsonl
```

## Active: How2Sign SignTrajField-v2 full training

| Field | Value |
|---|---|
| Alias | `how2sign-signtrajfield-v2-full-20260807` |
| Dataset | Full How2Sign train/validation splits (30,685/1,717 samples) |
| Training job | Slurm `141037`, name `h2s_stfv2_full` (`RUNNING`) |
| Configuration | `NIAF/continuous_trajectory_field/configs/how2sign_signtrajfield_v2_full.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/how2sign_signtrajfield_v2_full` |
| Initialization | Fresh v2 model; no v1 checkpoint, warm start, or resume checkpoint |
| Adapter | Frozen `how2sign_soft_arranger_signasl_all_b256x4_online_r3_gloo/checkpoints/best.pt` used as the optional word prior |
| Conditioning | Sentence `text`; optional word prior is dropped for 50% of training samples |
| Retrieval bank | External SignASL `manifest_all.jsonl`: 113,524 entries and 41,260 lexicon keys |
| Epochs | 50; checkpoint every epoch and dual-mode validation every 5 epochs |
| Allocation | 4 nodes, 1 GPU per node, DDP/NCCL, five-day limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally |
| Memory split | At most 16 samples and 4,096 padded frames per local memory batch; ranks synchronize to the smallest safe size |
| W&B | Online project `soke-niaf-continuous-trajectory`; run `signtrajfield-v2-how2sign-full-20260807`; ID `h2sstfv2full_20260807` |

At the observation time, all four ranks had joined and completed logical batch
`4/240` in epoch 1 (two optimizer steps) with no DDP/NCCL error.

### Live tracing commands

```bash
squeue -j 141037 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 141037
tail -F logs/sbatch/h2s_stfv2_full_141037.out
tail -F logs/sbatch/h2s_stfv2_full_141037.err
```

Epoch metrics are written to:

```text
experiments/NIAF/continuous_trajectory_field/how2sign_signtrajfield_v2_full/metrics.jsonl
```

## Active: CSL-Daily SignTrajField-v2 full training

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-v2-full-20260807` |
| Dataset | Full CSL-Daily train/validation splits (18,399/1,077 samples) |
| Training job | Slurm `141306`, name `csl_stfv2_full` (`RUNNING`) |
| Failed predecessor | Slurm `141038`; its distributed step was cancelled before Python launched because node `ADUAED21044WKLX03` had a Slurm communication failure |
| Configuration | `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v2_full.yaml` |
| Output | `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_full` |
| Initialization | Fresh v2 model; no v1 checkpoint, warm start, or resume checkpoint |
| Adapter | Frozen `csl_daily_soft_arranger_lw_2xb256/checkpoints/best.pt` used as the optional word prior |
| Conditioning | Sentence `text`; optional word prior is dropped for 50% of training samples |
| Retrieval bank | Train-only `manifest_train.jsonl`: 133,689 entries and 2,000 lexicon keys |
| Epochs | 50; checkpoint every epoch and dual-mode validation every 5 epochs |
| Allocation | Requests 4 nodes, 1 GPU per node, DDP/NCCL, five-day limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally |
| Memory split | At most 16 samples and 4,096 padded frames per local memory batch; ranks synchronize to the smallest safe size |
| W&B | Online project `soke-niaf-continuous-trajectory`; run `signtrajfield-v2-csl-daily-full-20260807`; ID `csldstfv2full_20260807` |

The fresh retry was submitted and allocated immediately at 2026-08-10 14:59
Asia/Dubai on four different Spark GPU nodes. The predecessor created no model
output, checkpoint, metrics, or W&B process, so there was nothing to resume and
the retry correctly retains fresh initialization and `WANDB_RESUME=never`. All
four ranks joined successfully, the online W&B run initialized, and epoch 1
advanced through logical batch `6/144` with no task-launch, DDP, NCCL, or OOM
error during the startup observation window.

### Live tracing commands

```bash
squeue -j 141306 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 141306
tail -F logs/sbatch/csl_stfv2_full_141306.out
tail -F logs/sbatch/csl_stfv2_full_141306.err
```

Epoch metrics will be written to:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v2_full/metrics.jsonl
```

## Previous: CSL-Daily Stage 2, joint scratch training

| Field | Value |
|---|---|
| Alias | `csl-daily-stage2-joint-scratch-20260722` |
| Dataset | CSL-Daily sentence-level train/validation splits (18,399/1,077 samples) |
| Intended state | Resumed from the epoch-9 recovery checkpoint; training epoch 10 |
| Training job | Slurm `140721`, name `csl_stf_s2_resume9` (historical; no longer in scheduler) |
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

## Previous: How2Sign Stage 2, joint scratch training

| Field | Value |
|---|---|
| Alias | `how2sign-stage2-joint-scratch-20260722` |
| Dataset | How2Sign sentence-level train/validation splits |
| Intended state | Resumed from the epoch-9 recovery checkpoint; training epoch 10 |
| Training job | Slurm `140728`, name `h2s_stf_s2_resume9c` (historical; no longer in scheduler) |
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

## Previous: PHOENIX Stage-2 resumed training

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
