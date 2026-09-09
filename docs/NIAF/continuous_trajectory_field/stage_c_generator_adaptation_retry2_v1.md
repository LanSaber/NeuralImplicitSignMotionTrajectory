# CSL-Daily Stage-C generator-adaptation retry-2 pilot v1

Status: **preregistered operational recovery; no retry-2 job submitted.** This
contract preserves the scientific question, arms, thresholds, batch/data
scope, RoCE gates, and non-authorizing boundary of
`stage_c_generator_adaptation_v1.md`. It changes only the source-bound
operational generation needed after run-r1 smoke job `143525` stopped before
science on a legacy decision-schema compatibility error.

## Exact recovery boundary

The pinned Stage-B decision is the immutable file with SHA256
`8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5`
and identity
`7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`.
Its exact schema-v1 top-level fields intentionally omit
`authorized_purpose`. The retry-2 gate requires that omission, plus
`stage=stage2`, `status=valid_infeasible`, `integrity_valid=true`,
`development_feasible=false`, and both confirmation/test-access flags false.
It rejects any added `authorized_purpose` field, including explicit null, and
rejects every non-null purpose.

Run-r1 CPU/calibration evidence under source
`a883baf4a0a95b4ebb2007564837c4d9b4f817cc` and calibration artifact
`csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_v1`
is historical and must remain unchanged. Retry 2 requires a newly pushed,
clean source commit, a fresh CPU gate in that commit's existing
`source_<git-head>` prerequisite namespace, and a new train-only calibration:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_retry2_v1
```

The smoke and pilot outer roots remain unchanged because their singleton
controls and attempts are already source-commit namespaced and run-r1 created
no smoke/pilot publication or scientific output. The arm outputs and
experiment names are distinct `*_pilot_run_r2` paths. No evidence may be
deleted or replaced to recover.

## Retry-2 files

```text
NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_memory_pilot_run_r2.yaml
NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_matched_off_pilot_run_r2.yaml
NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_decision_policy_retry2_v1.json
NIAF/continuous_trajectory_field/configs/csl_daily_stage_c_generator_adaptation_retry2_recovery_evidence_v1.json
scripts/NIAF/launch_csl_daily_stage_c_generator_adaptation_pilot.sh
scripts/NIAF/test_csl_daily_stage_c_generator_adaptation_sbatch.sh
scripts/NIAF/calibrate_csl_daily_stage_c_generator_adaptation_sbatch.sh
scripts/NIAF/smoke_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh
scripts/NIAF/train_csl_daily_stage_c_generator_adaptation_pilot_sbatch.sh
```

The policy hash-binds a recovery-evidence manifest. Every retry-2 stage reopens
the exact run-r1 CPU/calibration records and smoke logs, revalidates their
hashes and artifact identity, and proves that the old source-specific
smoke/pilot/attempt/publication and arm-output paths remain absent. The policy
records job `143525`, the pre-science failure class, zero observed scientific
output, no prior execution lease, mandatory run-r1 evidence preservation, and
an exact operational retry exception for an attested pre-claim/pre-science
implementation failure. This exception authorizes only the fixed run-r2
contract; it is not model, promotion, confirmation, or test authorization.
The launcher also refuses to proceed while `squeue` reports any nonterminal
Stage-C job; it reports those jobs and never cancels them automatically.

All outputs remain `development_only=true`, `non_authorizing=true`, and
`promotion_eligible=false`, with no confirmation or test access and W&B off.
Even a successful one-epoch pilot authorizes no longer run, promotion,
confirmation, or test use.
