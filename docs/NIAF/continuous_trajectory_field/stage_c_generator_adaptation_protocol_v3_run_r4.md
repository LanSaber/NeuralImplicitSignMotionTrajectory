# CSL-Daily Stage-C generator-adaptation protocol v3 / run-r4

Status: **preregistered, development-only, non-authorizing, and not
submitted.** Protocol-v2/run-r3 is terminal after CPU job 143574 failed
before compile, Ruff, pytest, data staging, calibration, allocation, or
science. Its dependent jobs 143575--143577 had zero runtime/allocation and
were cancelled. No r3 result is reused and no r4 action is authorized by this
document.

The immutable zero-science incident is
experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v2_run_r3_cpu.invalid_attempts/source_90cad3d3a7a09279b4a882e8c28a13b2a320ec1a_cpu143574_dependents143575_143577/ARCHIVE.json,
SHA256 66368adf6a2310732a4a2bcca87625d45c8dd51c0c4912b16ceba9b4c12b8dc2,
bytes 11442.

Its sole cause was a 30-second archived-clone local Git-metadata check that
timed out during transient CIFS latency. The archive records zero optimizer
updates, validation batches, trainer/calibration/CPU-gate work, data access,
leases, or scientific output; it also binds the six absent r3 runtime roots,
six dependent runtime logs, and the absent confirmation-holdout spend marker.

## Frozen science

Every scientific setting is byte-semantically identical to protocol-v2/run-r3:
two sequential independent Stage-B best_infeasible warm starts; memory
(dropout, 0.25) followed by matched_off (off, 1.0); 20 trainable generator
tensors/1,109,395 parameters; frozen sentence memory; world size two; logical
batch 64 and accumulation two; the centered 11-mode validation suite;
one-update smoke and one uncapped length-bucketed development epoch. The
development-only, no-test, no-confirmation, non-promotion policy, source
checkpoint, gates, thresholds, outputs, and terminal no-retry rules are
unchanged.

## Sole operational recovery

The intended immutable branch is
codex/csl-daily-centered-generator-stage-c-v3-run-r4 and the standalone clone
is /media/cvpr/haomian/SignTrajField_centered_stage_c_v3_run_source_r4.

Only the CPU gate runs the full historical chain and archived r3-clone local
Git metadata validation. Its local metadata bound is exactly 600 seconds; its
separate remote ls-remote bound is at most 120 seconds. This is an
increase-only execution-safety change under the archive's allowed change
class, not a scientific or recovery-policy change.

CPU READY is schema v2 and binds the recovery and config-audit identities,
archive/policy/manifest hashes, both r4 config hashes, and both timeout
bounds. Calibration, smoke, pilot, and their runtime revalidation authenticate
that READY and reopen only the immutable archive/policy/manifest and current
source/config hashes. They must never repeat the historical archived-clone
Git scan after science begins.

## Ordered gates and roots

1. Fresh CPU gate:
   experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_prerequisites/source_<HEAD>/cpu_gate.
2. Fresh train-only calibration:
   experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_protocol_v3_run_r4.
3. Fresh two-node forced-IB smoke:
   experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v3_run_r4_smoke.
4. Fresh one-epoch pilot only if exact smoke READY is authenticated:
   experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v3_run_r4_pilot.

The arm roots are memory_protocol_v3_run_r4 and matched_off_protocol_v3_run_r4;
Slurm logs use logs/sbatch/csl_stage_c_protocol_v3_run_r4_. Every root is
no-replace and distinct from all older generations.

Implementation:

- NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_protocol_v3_run_r4.yaml
- NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_protocol_v3_run_r4.yaml
- NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_decision_policy_v1.json
- NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_recovery_evidence_v1.json
- scripts/NIAF/launch_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh
- scripts/NIAF/test_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh
- scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh
- scripts/NIAF/smoke_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh
- scripts/NIAF/train_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_sbatch.sh
- scripts/NIAF/run_csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4.sh
