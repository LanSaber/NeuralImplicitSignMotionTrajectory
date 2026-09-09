# CSL-Daily Stage-C generator-adaptation pilot v1

Status: **preregistered development-only experiment; no result exists and no
job had been submitted when this protocol was recorded.** This Stage C is a
separate exploratory generator-adaptation pilot requested after the terminal
Stage-A/B experiment. It does not amend the Stage-B `valid_infeasible`
decision, create an authorization that Stage B did not issue, or make the
source checkpoint deployable.

The experiment asks one question: with the learned Stage-B sentence memory
held fixed, does allowing a small, exactly defined part of the trajectory
generator to adapt produce a better development score than the same generator
adaptation trained without sentence memory?

## Authorization boundary

Every prerequisite, checkpoint, execution record, and decision is marked
`development_only=true`, `non_authorizing=true`,
`promotion_eligible=false`, and `authorized_purpose=null`. Only train and
validation neighbor tables may be staged. Stage-C code must not stage,
reference, or open `neighbors_test.npz`, the test manifest, or the sealed
confirmation manifest. The global confirmation-spend marker must remain
absent, and W&B is disabled.

No Stage-C status authorizes checkpoint promotion, confirmation, test
evaluation, a longer generator run, or another adaptive stage. A longer run
would require a new protocol preregistered without consulting the pilot
outcome; this v1 protocol itself cannot issue that permission.

## Exact source and matched arms

Both arms start independently from the same Stage-B audit-only checkpoint:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/checkpoints/best_infeasible.pt
SHA256 b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202
epoch 5, global step 360, selection status best_infeasible
```

The pinned Stage-B terminal decision is
`valid_infeasible`, has `authorized_purpose=null`, decision identity
`7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`,
and file SHA256
`8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5`.

The two arms run sequentially inside one allocation and in separate trainer
processes. Each reloads the pinned checkpoint and resets optimizer, epoch,
global-step, selection, and RNG state:

| Arm | Training-time sentence memory | Dropout probability |
|---|---|---:|
| `memory` | `dropout` | `0.25` |
| `matched_off` | `off` | `1.0` |

All other resolved training, data, optimizer, evaluation, and source settings
must match exactly. Paired corruption and the Stage-B association-training
objective are disabled. The frozen v2 teacher remains pinned at SHA256
`06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`.

## Trainability and batch contract

All 62 sentence-memory tensors remain bitwise frozen. Of 233 model tensors,
exactly 20 generator tensors (1,109,395 parameters) are trainable; the other
213 tensors are frozen. Trainability is limited to these seven prefixes:

```text
hypernetwork.global_context.
hypernetwork.coarse_head.
hypernetwork.residual_head.
hypernetwork.gate_head.
hypernetwork.local_context.
hypernetwork.local_head.
hypernetwork.local_gate_head.
```

The global and local generator learning rates are both `1e-5`, weight decay is
`1e-4`, and the gradient norm cap is `1.0`. Each rank uses logical batch 64
with two-way gradient accumulation. With exactly two ranks, the effective
global batch is `2 * 64 * 2 = 256`. The pilot is exactly one uncapped epoch,
with length-bucketed batches, `drop_last=false`, one validation, and global
step 72 per arm. It may publish only `best_exploratory.pt` or
`best_exploratory_infeasible.pt`, never `best.pt`.

## Two-node paired-RoCE execution

The smoke and pilot each require exactly two Spark nodes, one GPU/rank per
node, selected through one exact Slurm feature `pair01` through `pair15`. Both
nodes must belong to that same physical pair. Before either trainer starts,
the allocation must prove both ConnectX-7 f1 rails are active at at least
200 Gb/s, MTU 9000, and RoCE v2 GID index 5 on their pair-specific `/30`
networks, including peer connectivity and clean error counters.

NCCL is forced to `NET/IB`; a socket data-transport fallback is fatal. The
preflight benchmarks each rail separately and both rails together using the
exact 4,437,580-byte Stage-C gradient payload plus 64 MiB and 256 MiB
collectives. Dual rail is selected when its gradient median is within 5% of
the fastest single rail; otherwise the fastest verified single rail is used.
Both arms use the same selected network profile in the same allocation.

## Ordered launch gates

The only permitted order is:

1. Run the complete CPU compile, Ruff, and repository test gate against one
   exact clean, pushed, standalone source clone.
2. Build a fresh, source-bound **train-only** relevance calibration. It must
   be semantically identical to the Stage-B calibration in coefficients, map
   SHA/content, held-out metrics, and minimum gates; only its artifact and
   calibration provenance identities may differ.
3. On one verified two-node physical pair, run the forced-IB network preflight
   and both matched smoke arms. Each arm is capped at two train batches and one
   validation batch, producing exactly one optimizer update. The audit must
   prove that all 20 trainable tensors changed, all have finite nonzero Adam
   moments, and all sentence-memory and nonapproved tensors stayed bitwise
   frozen.
4. Only a `smoke_ready` decision may start the paired one-epoch pilot. Run the
   `memory` arm and then the `matched_off` arm from independent exact Stage-B
   warm starts in that same allocation.

The source commit, remote branch, calibration, CPU gate, selected pair,
network evidence, two arms, and decision policy are bound into globally
singleton, no-replace execution records. Failed or interrupted attempts are
reconciled through their attested leases. A new attempt is permitted only when
the prior attempt produced zero scientific output; any partial scientific
output is terminal and cannot be retrained under this v1 protocol. Evidence
must not be deleted to force a retry.

## Predeclared decisions

The immutable policy is
`NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_decision_policy_v1.json`.
It recomputes selection diagnostics from the validation fields embedded in
each checkpoint and checks exact arm modes, epochs, steps, source identities,
finite state, frozen/trainable scope, and zero-valued invariants.

| Mode and result | Status | Sole next permitted action |
|---|---|---|
| Smoke passes | `smoke_ready` | `run_one_epoch_development_pilot` |
| Smoke fails | `stop` | `none` |
| Pilot gates pass | `pilot_complete_development_signal` | `none_requires_fresh_preregistration_without_pilot_outcome_access` |
| Pilot gates fail | `stop` | `none` |

Thus even the positive pilot status is only a development signal and
authorizes no subsequent or longer run. The pilot gates require both arms'
live memory-off score to remain within 0.5% of the pinned source off score;
the `memory`
correct score to improve by at least 0.1% over both `matched_off` and the
Stage-B source; correct-versus-off relative gain of at least 0.1%;
`Rpair >= 0.10` and each of its three nonce values at least `0.05`; exact-zero
invariant metrics; and left/right hand negative degradation versus off of at
least `-0.02`.

## Canonical files

```text
NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_pilot.yaml
NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_pilot.yaml
NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_decision_policy_v1.json
scripts/NIAF/launch_csl_daily_stage_c_generator_adaptation_pilot.sh
scripts/NIAF/test_csl_daily_stage_c_generator_adaptation_sbatch.sh
scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh
scripts/NIAF/smoke_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh
scripts/NIAF/train_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh
```

The launcher is preview-only unless passed `--submit`. It must be invoked from
the final immutable shared clone with exact `PROJECT_DIR`, `SOURCE_GIT_HEAD`,
and `SOURCE_REMOTE_BRANCH`; the selected `pairNN` is supplied explicitly. The
registry must be updated with the immutable source and job/evidence ledger
after submission rather than guessing those values in advance.
