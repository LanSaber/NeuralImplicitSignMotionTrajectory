# CSL-Daily Stage-C generator-adaptation protocol v4 / run-r5

Status: **terminal invalid attempt after a post-arm smoke decision-validation
failure. Development-only, non-authorizing, and not promotion-eligible.**
Protocol-v4/run-r5 is not resumable or retryable. It remains a distinct
generation from protocol-v3/run-r4, and any correction requires another fresh
protocol/run namespace.

## Immutable run-r5 incident

The no-replace terminal archive is
`experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_stage_c_generator_adaptation_protocol_v4_run_r5_smoke.invalid_attempts/source_36d121dd361df2c08616031bd7c2076d85da78d0_smoke143609_pilot143610/ARCHIVE.json`
(schema `signtrajfield_stage_c_generator_adaptation_terminal_incident_archive`
v4, 39,609 bytes, SHA256
`9c4e2213ca51941b41281f0b869fc66aa4e440df846ff57dee2731fca5f1c41a`).
It binds the exact source, job states, logs, prerequisites, every file in the
retained smoke/calibration trees, required absences, authorization boundary,
and the only allowed recovery change. It must not be replaced, and none of the
bound evidence may be deleted or rewritten.

The immutable standalone source is
`/media/cvpr/haomian/SignTrajField_centered_stage_c_v4_run_source_r5`, with
clean local HEAD and pushed remote head
`36d121dd361df2c08616031bd7c2076d85da78d0`, Git tree
`4e3debc39c4ace5d11d61b464af3f36fd8832df4`, and 536 tracked files on
`origin/codex/csl-daily-centered-generator-stage-c-v4-run-r5`.

## Terminal execution ledger

| Job | Gate | Terminal state | Allocation and runtime |
| --- | --- | --- | --- |
| 143607 | repository CPU gate | `COMPLETED`, `0:0` | one node (`ADUAED21030WKLX17`), 8 CPUs, 32 GiB, no GPU; `00:31:37` |
| 143608 | train-only calibration | `COMPLETED`, `0:0` | one node (`ADUAED21018WKLX24`), 8 CPUs, 32 GiB, no GPU; `00:02:19` |
| 143609 | paired smoke | `FAILED`, `1:0`, `NonZeroExitCode` | two `pair02` nodes (`ADUAED21043WKLX04`, `ADUAED21044WKLX03`), 32 CPUs, 200 GiB, two GPUs; `00:16:21` |
| 143610 | one-epoch pilot | `CANCELLED`, `0:0`, `DependencyNeverSatisfied` | zero runtime, no node, no allocation; cancelled at 2026-09-10 10:18:31 +04:00 |

The controller later purged the live records, and `sacct` remained unavailable
because `slurmdbd` refused the connection. The archive binds the exact terminal
fields captured from live `scontrol` before that purge rather than attempting
to reconstruct accounting history afterward.

## Completed prerequisites

CPU job 143607 passed compileall and Ruff and ran the complete
`tests/test_*.py` repository suite: **468 passed, 13 warnings in 343.39 s**.
Its schema-v2 READY has SHA256
`2ef706f2bbc6b7be42d1b2c38106a00fff33445e6056bfcbdd4d3bc1087ee84a`
and binds the clean source/remote identity, both configs, decision policy,
recovery manifest, predecessor incident, and false test/confirmation flags.

Calibration job 143608 published a fresh immutable train-only artifact with
identity
`07f556796238f45931c1bce757f0a99818f55dfd85a207f9d888e87028913656`
and completion SHA256
`7d44638f06bf3802e44f057c4cf77541f3503623b9e976e9be17e2a0332f6e2e`.
It passed the preregistered gates with holdout AUROC
`0.9817280587266659` and probability gap `0.7923572566612873`. The
transition audit records exact coefficients, held-out metrics, map content and
SHA, minimum gates, and every non-provenance field. Its lease was released;
there is no active calibration lease.

## Smoke boundary and retained observations

Job 143609 passed the paired-RoCE preflight on both 200 Gb/s rails with links
up, MTU 9000, GID index 5, forced `NET/IB`, peer connectivity, no socket
fallback, and zero harmful counter increments. The Stage-C-gradient benchmark
measured 97.9037 Gbit/s on the primary rail, 73.0971 Gbit/s on the secondary,
and 55.8313 Gbit/s on dual rail, so the preregistered selector correctly chose
single-primary `rocep1s0f1:1`. The dual-vs-single artifact SHA256 is
`9d8948f3cfafa0fb39d647fd153aa33b14be21f0cac68524ba8ee796a0024b70`;
counter-health SHA256 is
`8a3f13c3175845887899b023b7cfc70d39f5f2e5005f9446e67a2e9a50b6f433`.

Both arms ran in order (`memory`, then `matched_off`). Each consumed two
logical training batches, made exactly one optimizer update, reached epoch 1 /
global step 1, and evaluated one logical batch in each of the 11 centered
development modes. The checkpoint audit (SHA256
`027ab37a397fdfce83514741a6ac27cad9e6825569b7f12e9fb7681eb12b9626`)
found all model/optimizer tensors finite, all 20 trainable tensors changed with
nonzero finite Adam moments, all 62 sentence-memory tensors and all 213 frozen
model tensors bitwise frozen, and the same source/provenance/objective except
for the approved arm switch.

The following are raw, one-batch-per-mode arm observations retained for
forensics. They are **not a smoke decision or final scientific result**:

| Arm | Correct score | Off score | Relative gain vs off | Rpair (n0 / n1 / n2) | L/R negative degradation vs off | Train loss |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| memory | 10.68098695 | 10.70996361 | 0.00270558 | 0.41873715 (0.44215038 / 0.41489978 / 0.39797438) | -0.09447512 / 0.43309148 | 11.83277357 |
| matched-off | 10.68488899 | 10.71273042 | 0.00259891 | 0.41891111 (0.44225456 / 0.41504635 / 0.39825495) | -0.09959678 / 0.43166760 | 11.90146622 |

Both text-only guards and the implementation-invariance checks passed, but the
scientific comparison gates rejected both arms (aggregate constraint violation
10.96375901 for memory and 11.30600426 for matched-off). Only
`best_exploratory_infeasible` checkpoints exist
(memory SHA256
`18d288309eef7f8430cda4b35f469cf5041a59382644440ef0cf74a43a1f119b`;
matched-off SHA256
`5753e7ae9a8c4ca284e1e3eb9620d5f2be60f4d686243c64fab8b43e9d1bee7a`).
No canonical best checkpoint was published. The attempt-level `COMPLETE.json`
SHA256 is
`a8b5732a6b2feee487b2fc951367d0b32ebb24c8e9fc7361330bb99c80ccdee8`,
but there is no `DECISION.json` and no smoke `PUBLICATION/READY`.

## Exact failure

After both arms and their checkpoint audit completed,
`stage_c_pilot_decision.py:1592` raised:

```text
StageCDecisionError: Stage-C arm configs differ beyond the approved switch
```

Independent normalization of the two retained `last.pt` configs left exactly
two recursive differences:

1. `sentence_memory_safety.stage_c.resolved_source_provenance.declared_contract.arm`
   is `memory` versus `matched_off`.
2. `sentence_memory_safety.stage_c.resolved_source_provenance.digest` is the
   derived arm-specific digest
   `c62f9f55811c22e87b35a7272b8e3fe5fe0f01f43d71dc7904659d1f01d2b155`
   versus
   `7a314a943e9c41d0288c6882986d32eb3df38cb6bb7ca26fac69cede03dd1675`.

The general config normalizer removes the approved top-level arm/output
switches but omits these nested provenance fields. A later dedicated provenance
comparison already normalizes both. The result is a redundant operational
rejection, not evidence of a scientific-config difference.

The failed smoke lease was archived with claim identity
`14d495eba98e7f9a8172e85f6700b71dc8b5155b161181c4341f6e4a262157fe`,
terminal identity
`115acb352b5b7a9719098672560523e2bc6617edfc1ad20e275929a7a6266ec0`,
and self-reported exit code 1. No active smoke lease remains.

## Data and authorization boundary

Only train and validation tables were staged. The CPU READY, calibration
completion, smoke launch, and attempt completion all record false
test/confirmation access; the global confirmation-spend marker remains absent.
W&B was explicitly disabled (`WANDB=0`, `WANDB_MODE=disabled`,
`WANDB_DISABLED=true`, with the API key unset), and no W&B artifact exists in
the smoke tree. The run remains development-only, non-authorizing, and
ineligible for promotion.

Pilot job 143610 never started, acquired no nodes/GPUs, produced no logs or
output root, and performed no optimizer update or validation. It cannot be
described as skipped successfully: it is terminally cancelled because its
`afterok:143609` dependency failed.

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

## Recovery boundary

Protocol-v4/run-r5 and jobs 143607--143610 are terminal. Do not resume job
143609, reuse either retained arm checkpoint for execution, requeue 143610, or
submit another run-r5 job. A recovery must use a new protocol/run generation,
new clean pushed source identity, new configs/policy/recovery bindings, new CPU
and calibration artifacts, new leases, and new jobs.

The only allowed implementation correction is to normalize
`resolved_source_provenance.declared_contract.arm` and the digest derived from
it in the *general arm config-equivalence check*. Scientific settings, source
checkpoint, arm semantics, evaluation, thresholds, batch contract, RoCE
contract, and data/authorization policy may not change. The run-r5 archive
reserves the names protocol-v5/run-r6 but leaves their implementation commit,
source/remote identities, config hashes, and all job IDs explicitly null. This
document does not bind or authorize that future generation or any submission.
