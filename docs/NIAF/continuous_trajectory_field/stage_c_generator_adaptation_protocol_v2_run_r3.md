# CSL-Daily Stage-C generator-adaptation protocol v2 / run-r3

Status: **fresh protocol generation preregistered; no protocol-v2/run-r3 job
has been submitted and no result has been observed.** This is a
development-only, non-authorizing experiment. It does not amend the terminal
Stage-B `valid_infeasible` decision, authorize an action that Stage B did not
authorize, or make any checkpoint deployable.

The v1/run-r2 protocol and all of its evidence are immutable history. Smoke
job `143541` completed one `memory`-arm optimizer update and began validation,
then stopped on an evaluator-dispatch `KeyError` before any validation value,
metrics file, completed checkpoint, smoke decision, or `smoke_ready` marker
was published. The `matched_off` arm never started, and dependent pilot job
`143542` had zero runtime/allocation before it was explicitly cancelled. That
partial scientific output makes run-r2 terminal under its own predeclared
retry policy; it cannot be resumed or rerun.

The immutable incident manifest is:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_smoke.invalid_attempts/source_f12b993b5de361423df3b8cbfb4e873f4a95ad1e_smoke143541_pilot143542/ARCHIVE.json
SHA256 82ce35cf3d7337f218bda189080a45b9e3faedec200750310d0958296f9d1855
```

Protocol v2 was defined from the operational failure class only, without a
pilot decision or validation outcome. The two printed run-r2 training-batch
losses are disclosed in the incident manifest but were not used to change an
arm, hyperparameter, threshold, data split, duration, or decision rule.

The experiment asks the same single question as v1: with the learned Stage-B
sentence memory held fixed, does allowing a small, exactly defined part of
the trajectory generator to adapt produce a better development score than
the same generator adaptation trained without sentence memory?

## Authorization and data boundary

Every protocol-v2/run-r3 prerequisite, checkpoint, execution record, and
decision must record all of the following:

```text
development_only=true
non_authorizing=true
promotion_eligible=false
authorized_purpose=null
confirmation_manifest_opened=false
test_data_accessed=false
```

Only the complete CSL-Daily training split and the sealed development subset
of validation may be used. Only the train and validation sentence-neighbor
tables may be staged. No protocol-v2/run-r3 process may stage, hash, reference,
or open `neighbors_test.npz`, a test manifest or test example, the sealed
confirmation manifest, a confirmation example, or a prior confirmation/test
export. The global confirmation-holdout spend marker must be absent before
and after every stage. W&B is disabled.

No status emitted by this protocol authorizes checkpoint promotion,
deployment, confirmation access, test evaluation, a longer generator run, a
retry after partial scientific output, or another adaptive stage. Any such
work requires a separate future protocol whose scientific choices are frozen
without access to this pilot's outcome.

## Exact source and matched arms

Both arms start independently from the same Stage-B audit-only checkpoint:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/checkpoints/best_infeasible.pt
SHA256 b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202
epoch 5, global step 360, selection status best_infeasible
```

The pinned Stage-B terminal decision is the exact legacy schema-v1 file with
SHA256
`8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5`
and decision identity
`7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`.
It records `stage=stage2`, `status=valid_infeasible`,
`integrity_valid=true`, `development_feasible=false`, and no
confirmation/test access. Its top-level schema intentionally omits
`authorized_purpose`; the protocol must validate that exact omission and must
not rewrite the decision.

The two arms run sequentially within one allocation and in separate trainer
processes. Each independently reloads the pinned Stage-B checkpoint and
resets optimizer, epoch, global step, selection, and RNG state:

| Arm | Training-time sentence memory | Dropout probability |
|---|---|---:|
| `memory` | `dropout` | `0.25` |
| `matched_off` | `off` | `1.0` |

All other resolved training, data, optimizer, evaluation, source, and network
settings must match exactly. Paired corruption and the Stage-B
association-training objective remain disabled. The frozen v2 teacher stays
pinned at SHA256
`06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`.
Neither arm may load the run-r2 `last.pt`, optimizer, calibration, RNG state,
or any other run-r2 execution output.

## Frozen memory, generator scope, and batch

All 62 sentence-memory tensors remain bitwise frozen. Of 233 model tensors,
exactly 20 generator tensors, containing 1,109,395 parameters, are trainable;
the other 213 tensors are frozen. Trainability is limited to seven prefixes:

```text
hypernetwork.global_context.
hypernetwork.coarse_head.
hypernetwork.residual_head.
hypernetwork.gate_head.
hypernetwork.local_context.
hypernetwork.local_head.
hypernetwork.local_gate_head.
```

The global and local generator learning rates are `1e-5`, weight decay is
`1e-4`, and the gradient-norm cap is `1.0`. Each rank uses logical batch 64
with gradient accumulation two. Exactly two ranks therefore give effective
global batch `2 * 64 * 2 = 256`.

The pilot is exactly one uncapped, length-bucketed epoch with
`drop_last=false`, one validation, and global step 72 per arm. It may publish
only `best_exploratory.pt` or `best_exploratory_infeasible.pt`, never
`best.pt`. The smoke caps each arm at two logical training batches and one
batch for each configured validation mode, producing exactly one optimizer
update per arm.

## Centered evaluation and sole operational change

Both arms retain the same centered 11-mode validation suite:

```text
off
on
motion_shuffled_n0
motion_shuffled_n1
motion_shuffled_n2
cross_query_motion
full_replacement
joint_tuple_permuted
uniform_final_mass
analytic_prior
association_disabled
```

Run-r2 correctly disabled the paired-corruption **training** objective, but
its evaluator dispatcher selected the centered paired evaluator only when
that training objective was enabled. It therefore entered the legacy
five-namespace evaluator and failed when it indexed `motion_shuffled_n0`.

Protocol v2 permits one operational code correction: select the centered
paired evaluator when either paired-corruption training is enabled **or**
centered sentence-memory evaluation is enabled. This does not enable paired
training, change a validation mode or metric, or modify the scientific
contract. Merely adding names to the legacy namespace map is not an allowed
substitute, because the centered evaluator computes `Rpair`, nonce-specific
diagnostics, association fields, and exact invariants required by the frozen
decision policy. A companion fail-fast validation rejects centered-only mode
names in an otherwise noncentered, paired-off configuration before evaluation;
it does not change any checked-in Stage-A, Stage-B, or Stage-C configuration.

## Two-node paired-RoCE execution

The smoke and pilot each require exactly two Spark nodes and two GPUs: one
rank and one GPU per node, selected through one exact Slurm feature `pair01`
through `pair15`. Both nodes must be members of the same physical pair.

Before either trainer starts, both ConnectX-7 f1 RoCE rails must be live at at
least 200 Gb/s, MTU 9000, and RoCE-v2 GID index 5 on their pair-specific `/30`
networks. Both nodes must pass topology, peer-connectivity, and clean raw
counter checks. NCCL is forced to `NET/IB`; any socket data-transport fallback
is fatal.

The preflight benchmarks each rail separately and both rails together with
the exact 4,437,580-byte Stage-C gradient payload plus 64-MiB and 256-MiB
collectives. Dual rail is selected only when its gradient median is within 5%
of the fastest verified single rail; otherwise the fastest verified single
rail is selected. Both arms must use that same selected profile in the same
allocation. Post-training raw counters, deltas, and per-rank NCCL `NET/IB`
with no `NET/Socket` evidence are mandatory before a smoke or pilot decision
can be published.

## Fresh ordered gates

The only permitted order is:

1. From the new clean standalone clone, verify the final pushed immutable
   protocol-v2/run-r3 HEAD and remote ref, reopen this archive, revalidate all
   bound hashes and absences, and run the complete CPU compile, Ruff, and
   repository test gate. The gate must use a fresh protocol-v2/run-r3
   source-commit namespace.
2. Build a fresh, source-bound **train-only** relevance calibration in the
   new calibration root. Its coefficients, calibration map SHA/content,
   held-out metrics, and minimum gates must be semantically identical to the
   Stage-B calibration; only artifact and source-provenance identities may
   differ. The run-r2 calibration must not be reused or replaced.
3. In a fresh protocol-v2/run-r3 smoke namespace on one verified two-node
   physical pair, run forced-IB single/dual-rail preflight and the `memory`
   then `matched_off` smoke arms. Each arm must independently reload Stage B,
   complete exactly one optimizer update, and finish all 11 one-batch
   validation modes. The audit must prove all 20 approved tensors changed and
   have finite nonzero Adam moments, while all 62 sentence-memory tensors and
   all 213 nonapproved model tensors remain bitwise frozen.
4. Atomically publish a smoke decision and release the smoke lease. Only the
   exact status `smoke_ready` may satisfy the pilot dependency. Any absent,
   incomplete, failed, or other status means stop.
5. In a fresh protocol-v2/run-r3 pilot namespace, run `memory` then
   `matched_off` for one uncapped epoch each from independent Stage-B warm
   starts. Atomically publish the terminal pilot decision and release its
   lease. No pilot status permits another run.

The CPU gate, calibration, smoke, and pilot each use a fresh globally
singleton no-replace namespace. The exact final source HEAD, remote head/ref,
config and script hashes, incident-manifest hash, Stage-B anchors, calibration
identity, pair, network evidence, arm order, and decision policy must be bound
before work begins. No v1/run-r1 or retry2-v1/run-r2 prerequisite, execution,
lease, output, or log path may be reused, deleted, moved, or modified.

Leases may be reconciled only before scientific work, or after termination to
preserve a terminal history record. Once an attempt has any optimizer update,
checkpoint, computed validation batch, metric, or arm output, any interruption
is a terminal fail-closed stop for this protocol generation. Partial outputs
must be preserved and may not be deleted to manufacture a retry. There is no
resume or retry within protocol v2/run-r3.

## Predeclared decisions

The new immutable policy is
`NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_decision_policy_v1.json`.
It must retain the v1 scientific gates exactly while binding the new protocol
generation and incident archive.

| Mode and result | Exact status | Sole next permitted action |
|---|---|---|
| Smoke passes every gate | `smoke_ready` | `run_one_epoch_development_pilot` |
| Smoke fails, is incomplete, or is interrupted | `stop` | `none` |
| Pilot passes every gate | `pilot_complete_development_signal` | `none_requires_fresh_preregistration_without_pilot_outcome_access` |
| Pilot fails, is incomplete, or is interrupted | `stop` | `none` |

The pilot gates remain unchanged: both arms' live memory-off score must stay
within 0.5% of the pinned source off score; the `memory` correct score must
improve by at least 0.1% over both `matched_off` and the Stage-B source;
correct-versus-off relative gain must be at least 0.1%; `Rpair >= 0.10` and
each of its three nonce values must be at least `0.05`; all required invariant
metrics must be exactly zero; and left/right hand negative degradation versus
off must each be at least `-0.02`.

Thus `smoke_ready` authorizes only the already-preregistered one-epoch pilot,
and every terminal pilot status authorizes no longer run, promotion,
confirmation, test, or adaptive follow-up.

## Canonical protocol-v2/run-r3 names

The intended immutable branch and standalone run clone are:

```text
codex/csl-daily-centered-generator-stage-c-v2-run-r3
/media/cvpr/haomian/SignTrajField_centered_stage_c_v2_run_source_r3
```

Launch is forbidden until the final clean pushed HEAD, matching remote head,
and every required file hash have been recorded in fresh source-binding
evidence. The new files are:

```text
NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v2_run_r3.yaml
NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v2_run_r3.yaml
NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_decision_policy_v1.json
NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_recovery_evidence_v1.json
scripts/NIAF/launch_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh
scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3.sh
scripts/NIAF/test_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh
scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh
scripts/NIAF/smoke_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh
scripts/NIAF/train_csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_sbatch.sh
```

All artifact and log namespaces are also distinct:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_prerequisites/source_<final-head>
experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_protocol_v2_run_r3
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v2_run_r3_smoke/sources/source_<final-head>
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v2_run_r3_pilot/sources/source_<final-head>
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v2_run_r3
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v2_run_r3
logs/sbatch/csl_stage_c_protocol_v2_run_r3_cpu_%j.{out,err}
logs/sbatch/csl_stage_c_protocol_v2_run_r3_calibration_%j.{out,err}
logs/sbatch/csl_stage_c_protocol_v2_run_r3_smoke_%j.{out,err}
logs/sbatch/csl_stage_c_protocol_v2_run_r3_pilot_%j.{out,err}
```

The v1/run-r1 and retry2-v1/run-r2 runbooks, policies, scripts, configs,
calibrations, prerequisites, attempts, leases, logs, and incident evidence are
historical aliases and must remain byte-unchanged.
