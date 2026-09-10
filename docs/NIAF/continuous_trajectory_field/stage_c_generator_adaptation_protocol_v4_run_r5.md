# CSL-Daily Stage-C generator-adaptation protocol v4 / run-r5

Status: **preregistered implementation; independently audited,
development-only, non-authorizing, and not submitted from this snapshot.**
This is a distinct recovery generation, not a retry or continuation of
protocol-v3/run-r4.

## Immutable predecessor

Protocol-v3/run-r4 is terminal after CPU job 143593 failed its repository test
gate. The no-replace incident is
`experiments/NIAF/continuous_trajectory_field/csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_cpu.invalid_attempts/source_740400412386d60a8d6f56b0f871bc16dee755b0_cpu143593_dependents143594_143596/ARCHIVE.json`
(SHA256 `a23682b584bcc05ccee67ccec7526d33efc924104a0ea34209740bb4131e5f71`,
11,168 bytes). It records the exact CPU source, 1 failed/461 passed test
result, zero-runtime dependent jobs 143594--143596, and the 13 absent r4
downstream roots/logs/spend marker.

The failure is confined to a standalone-clone test fixture passing its clone
as `evidence_root`, while the immutable recovery manifest correctly binds the
shared canonical evidence root. Runtime scripts already use that canonical
root. R4 produced no CPU READY, calibration, staged data, lease, GPU
allocation, optimizer update, validation batch, trainer, scientific artifact,
test/confirmation access, or W&B activity.

## Frozen science and allowed change

Run-r5 must preserve the exact Stage-B epoch-5/global-step-360
`best_infeasible.pt` source (SHA256
`b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`),
both matched arms, all generator/frozen-memory settings, the two-GPU paired
RoCE contract, batch/accumulation, centered 11-mode development evaluation,
gates, no-test/no-confirmation policy, and no-retry/no-promotion semantics.

The only intended implementation change is to make the CPU test fixture derive
`evidence_root` from the recovery manifest's `canonical_evidence_root`, rather
than assuming its standalone clone is the evidence root. Runtime surfaces may
change only to register the fresh protocol-v4 namespace and evidence bindings;
runtime behavior, scientific configuration, data paths, thresholds, and model
settings may not change.

## Planned isolation

The intended immutable branch is
`codex/csl-daily-centered-generator-stage-c-v4-run-r5`; the intended standalone
clone is `/media/cvpr/haomian/SignTrajField_centered_stage_c_v4_run_source_r5`.
All policy, recovery evidence, configs, gates, calibration, lease, smoke,
pilot, arm outputs, and logs must use fresh protocol-v4/run-r5 namespaces.
They must bind the r4 archive before any work. No r4 artifact, job, log,
calibration, or source identity may be reused or modified.

Any eventual run-r5 requires a new clean pushed source, an independently
reviewed CPU gate, then fresh train-only calibration, paired two-GPU smoke, and
the one-epoch development pilot. This document authorizes none of those steps.
