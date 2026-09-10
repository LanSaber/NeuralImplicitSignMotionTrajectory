# CSL-Daily Stage-C generator-adaptation protocol v5 / run-r6

Status: **preregistered implementation under independent audit;
development-only, non-authorizing, and not submitted from this snapshot.**
Protocol-v5/run-r6 is a distinct no-replace recovery generation. It is not a
retry, resume, continuation, or publication recovery of protocol-v4/run-r5.

## Immutable predecessor and sole correction

Protocol-v4/run-r5 is terminal after its one-update smoke job 143609. CPU job
143607 and fresh train-only calibration job 143608 completed. Smoke 143609 ran
both matched arms for exactly one optimizer update each and completed the
centered 11-mode, one-batch development validation, but the decision step
rejected the arm-config equivalence. Pilot 143610 never ran: its dependency
became unsatisfiable and it was cancelled with zero runtime and allocation.
The r5 output, controls, logs, and source are immutable evidence and cannot be
deleted, moved, modified, resumed, or used as a training/calibration start.

The exact failure is implementation-only. After the already approved
top-level arm, train-mode, dropout, experiment-name, and output-name fields are
removed, the two real retained `config.resolved.json` files differ only at
`sentence_memory_safety.stage_c.resolved_source_provenance.declared_contract.arm`
and the provenance `digest` derived from it. The decision already compares
normalized provenance separately, but its full-config normalization omitted
those two identity fields. Protocol v5 removes only those two nested fields
from paired-config equivalence. Every other mismatch remains terminal.

The r5 incident is preserved by the no-replace archive at
`experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v4_run_r5_smoke.invalid_attempts/source_36d121dd361df2c08616031bd7c2076d85da78d0_smoke143609_pilot143610/ARCHIVE.json`.
It is exactly 39,609 bytes, schema
`signtrajfield_stage_c_generator_adaptation_terminal_incident_archive` v4, and
SHA256 `9c4e2213ca51941b41281f0b869fc66aa4e440df846ff57dee2731fca5f1c41a`.
The v5 recovery manifest and policy bind that identity, its three exact tree
inventories, six logs, 13 source files, 10 absences, and r5 source identity
before any CPU or scientific work.

## Frozen scientific contract

Both arms must start independently, without resume or reuse of any r5 model or
optimizer state, from the Stage-B epoch-5/global-step-360
`best_infeasible.pt` checkpoint, SHA256
`b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`.
The sequential arms remain `memory` (dropout mode, probability 0.25) and
`matched_off` (off mode, probability 1.0). The sentence memory is frozen: all
62 sentence-memory tensors remain frozen. Exactly 20 generator tensors and
1,109,395 parameters under the seven registered hypernetwork prefixes may
change; all other 213 model tensors remain frozen. Each arm resets optimizer,
epoch/global step, selection, and RNG state.

The allocation is exactly two nodes and two GPUs from the same explicit
`pair01`--`pair15`, with one rank/GPU/node. Before training, both ConnectX-7
RoCE rails must pass 200-Gb/s link, MTU 9000, GID 5, peer-connectivity and raw
counter checks. Single- and dual-rail tests use the exact 4,437,580-byte
gradient payload; the measured profile is selected and each rank must report
NCCL `NET/IB` with no Socket fallback. Batch size is 64 per rank with
accumulation two, for effective global batch 256.

The smoke is exactly two logical batches and one optimizer update per arm,
with one development-validation batch in each of the 11 centered modes. The
pilot, if and only if an atomic `smoke_ready` is published, is one uncapped
epoch/global step 72 per arm. Scientific configs, source/data identities,
metrics, thresholds, validation modes, initialization, trainability, network
contract, duration, and decision statuses are identical to r5.

## Ordered gates and access boundary

The only lawful order is: a fresh repository-wide CPU/compile/Ruff/test gate
on the final clean pushed v5 source; fresh source-bound train-only calibration;
the two-node network preflight and paired one-update smoke; then the one-epoch
development pilot only after exact smoke publication. No r5 CPU READY,
calibration, staged data, lease, checkpoint, validation, or decision artifact
may satisfy a v5 gate.

Only train data and the sealed development subset of validation may be staged;
only train/validation neighbor tables may be opened. Test and confirmation
inputs must not be referenced, staged, opened, or evaluated. The global
confirmation-spend marker must remain absent. W&B is disabled and no online
credential may propagate into any job.

Every failure status is `stop` with `next_permitted_action=none`. Smoke success
is only `smoke_ready` with permission for the one-epoch development pilot.
Pilot success is only `pilot_complete_development_signal` with
`none_requires_fresh_preregistration_without_pilot_outcome_access`. No status
authorizes a retry, resume, longer run, promotion, confirmation, or test.

## Fresh protocol-v5/run-r6 isolation

The intended immutable branch is
`codex/csl-daily-centered-generator-stage-c-v5-run-r6`; the intended standalone
clone is `/media/cvpr/haomian/SignTrajField_centered_stage_c_v5_run_source_r6`.
Fresh names are mandatory for both configs, decision policy, recovery
evidence, six launch/runtime wrappers, CPU/prerequisite controls, calibration,
smoke, pilot, both arm outputs, leases, publications, and `csl_stage_c_protocol_v5_run_r6_*`
logs. The canonical new files are:

- `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v5_run_r6.yaml`
- `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v5_run_r6.yaml`
- `NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_decision_policy_v1.json`
- `NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_recovery_evidence_v1.json`
- `scripts/NIAF/launch_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6.sh`
- `scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6.sh`
- `scripts/NIAF/test_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh`
- `scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh`
- `scripts/NIAF/smoke_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh`
- `scripts/NIAF/train_csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_sbatch.sh`
- `tests/test_niaf_stage_c_protocol_v5_run_r6_ops.py`

The output roots are
`...csl_daily_stage_c_generator_adaptation_protocol_v5_run_r6_prerequisites`,
`...csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_protocol_v5_run_r6`,
`...generator_adaptation_protocol_v5_run_r6_smoke`,
`...generator_adaptation_protocol_v5_run_r6_pilot`, and the two arm-specific
`...generator_adaptation_{memory,matched_off}_protocol_v5_run_r6` roots. This
document does not authorize creating the branch/clone, committing, pushing,
submitting jobs, or spending any scientific or held-out data budget.
