# Active SignTrajField Training Registry

This file is the canonical identity registry for live and queued SignTrajField
runs. Read it before answering an unqualified question such as "What is the
current training progress?" The scheduler and logs remain the source of truth
for live state; this registry determines **which run** the question refers to.

Last observed: **2026-09-10 10:38 Asia/Dubai (UTC+04:00)**

## Default run resolution

`DEFAULT_RUN_ALIAS: phoenix-signtrajfield-v2-full-syncfix-r2-20260806`

Unless the user names a dataset, job ID, output directory, or another alias,
"the training", "current training", and "the active training" refer to the
full SignTrajField-v2 PHOENIX run below. Do not select a run merely because it
has the highest Slurm job ID or is already in the `RUNNING` state.

Dataset-qualified requests override that default: "the How2Sign training"
refers to `how2sign-signtrajfield-v2-full-20260807`, and "the CSL-Daily
training" refers to
`csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-protocol-v5-run-r6-20260910`.

## Active prerequisite artifact jobs (not training)

### CSL-Daily SignTrajField-RAG train-only sentence bank

| Field | Value |
|---|---|
| Artifact alias | `csl-daily-signtrajfield-rag-bank-mt5-vae-mu-train-v1-20260905` |
| Run class | Offline preprocessing artifact; **not** a model-training run and does not change `DEFAULT_RUN_ALIAS` |
| Dataset | CSL-Daily train split only: 18,399 items and 6,578 normalized-text semantic groups |
| Build job | Slurm `142705`, name `csl_rag_bank` (`COMPLETED`, exit code 0, runtime 1:44:23, node 19) |
| Superseded launches | Slurm `142700` exited before Python because the original inline `/bin/sh` wrapper did not support `pipefail`; Slurm `142701` was cancelled before data creation so exact dirty-worktree source hashes could be added to bank provenance. Neither created target files |
| Launcher | `scripts/NIAF/build_sentence_motion_bank_sbatch.sh` |
| Configuration | `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a.yaml` |
| Target | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1` |
| Motion codec | Deterministic `mu` from `experiments/flow/VAE/csl_daily_rot6d_vae_jerk_b16x4_online/checkpoints/best.pt`; rot6d, latent dimension 256, temporal downsampling 4 |
| Text keys | Frozen `deps/mt5-base`, masked-mean pooled and L2-normalized sentence-text keys |
| Build settings | Batch 64, 8 spawn-context data workers, CUDA VAE and text encoder; no row limit and no overwrite |
| Allocation | 1 Spark node (`ADUAED21038WKLX19`), 1 GPU, 8 CPUs, 32 GiB, six-hour limit |
| W&B | Disabled/not applicable |
| Result | Complete 243 MiB bank: 18,399 items, 6,578 semantic groups, 430,364 ragged VAE tokens (`float16`, 256-D), bank ID `a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd` |
| Success contract | Satisfied: `bank.json` and `build_summary.json` passed the builder's content validation and `READY` was written last |
| Independent verification | CPU-only Slurm `142732` on node 19; strict `--verify_only --verify_hashes` completed with exit code 0 and reported `valid: true`, the same bank ID, and 18,399 rows |
| Follow-up | Slurm `142745` completed and strictly audited all top-64 train/validation/test neighbor tables; Phase-A job `142746` subsequently completed, and duration-weight pilot `142893` reuses the same verified artifacts |

Logs:

```text
logs/sbatch/csl_rag_bank_142705.out
logs/sbatch/csl_rag_bank_142705.err
```

Live tracing commands:

```bash
squeue -j 142705 -o '%.18i %.28j %.2t %.12M %.12l %.4D %R'
scontrol show job 142705
tail -F logs/sbatch/csl_rag_bank_142705.out
tail -F logs/sbatch/csl_rag_bank_142705.err
```

### CSL-Daily SignTrajField-RAG top-64 sentence neighbors

| Field | Value |
|---|---|
| Artifact alias | `csl-daily-signtrajfield-rag-neighbors-top64-20260906` |
| Run class | Offline preprocessing artifact; **not** a model-training run and does not change `DEFAULT_RUN_ALIAS` |
| Job | Slurm `142745`, name `csl_rag_neighbors` (`COMPLETED`, exit code 0, runtime 25:18, node 19) |
| Launcher | `scripts/NIAF/build_sentence_neighbors_sbatch.sh` |
| Inputs | Verified bank `a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd`; complete train/validation/test manifests with 18,399/1,077/1,176 queries |
| Outputs | `neighbors_train.npz`, `neighbors_val.npz`, and `neighbors_test.npz` in the sentence-bank directory |
| Retrieval | Exact cosine search over semantic-group mT5 keys; top-M 64; CUDA query/text encoding; no overwrite |
| Allocation | 1 Spark node, 1 GPU, 8 CPUs, 32 GiB, four-hour limit |
| Artifact identities | Train SHA256 `d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852`; validation `b5f5d0914e55ba9e9c79963bacb955b97d62f99f16f117381187e72c345eb199`; test `8c607439121a44665e32e638867aa8b1d488f4228c9f4cf73074a5702309952a`; filter-policy digest `537cfd70355d928f79ad20380261a1f0c276c303c6917b974badbc15ba136930` |
| Audit result | Strict `--verify_hashes --splits train val test` reported `valid: true`; all 1,321,728 candidate slots were checked, every train/validation/test query has 64 valid distinct candidates, and bank/config/mT5/VAE/statistics/source identities match |
| Source | Branch `codex/csl-daily-signtrajfield-rag-v3`, commit `39328f12e2b604a4335fd1f6ec1a81790331ff93` |

Logs:

```text
logs/sbatch/csl_rag_neighbors_142745.out
logs/sbatch/csl_rag_neighbors_142745.err
```

## Preregistered CSL-Daily: Stage-C protocol-v5/run-r6 generator-adaptation pilot

State: implemented under independent audit, development-only/non-authorizing
preregistration; no r6 job is submitted from this snapshot. The
dataset-qualified CSL-Daily alias is
`csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-protocol-v5-run-r6-20260910`.
R6 is a fresh protocol generation, not a retry or resume. Its only
implementation correction removes the nested declared provenance arm and its
derived digest from the general matched-arm config-equivalence comparison;
every other mismatch remains rejected. It starts both arms independently from
the exact Stage-B best-infeasible source, requires a fresh CPU gate and
train-only calibration, and preserves the complete r5 scientific, two-GPU
paired-RoCE, data-access, duration, decision, and terminal no-retry contracts.
See
docs/NIAF/continuous_trajectory_field/stage_c_generator_adaptation_protocol_v5_run_r6.md.

## Terminal CSL-Daily Stage-C protocol-v4/run-r5 post-arm smoke failure

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-protocol-v4-run-r5-20260910` |
| State | **Terminal post-arm smoke decision-validation failure.** CPU 143607 and fresh train-only calibration 143608 completed. Paired smoke 143609 failed after both arms completed; pilot 143610 was dependency-unsatisfied and cancelled with zero runtime/allocation. R5 cannot resume or retry |
| Exact source | Clean pushed commit/remote `36d121dd361df2c08616031bd7c2076d85da78d0`, branch `origin/codex/csl-daily-centered-generator-stage-c-v4-run-r5`, standalone clone `/media/cvpr/haomian/SignTrajField_centered_stage_c_v4_run_source_r5` |
| Scientific boundary | `memory` then `matched_off` each completed two logical batches, exactly one optimizer update/global step 1, and one batch in every centered 11-mode development validation. The attempt COMPLETE and per-arm invalid-attempt checkpoints exist, but no decision, smoke READY, canonical best checkpoint, or pilot output exists |
| Trainability | Audit passed: exactly 20 generator tensors/1,109,395 parameters changed with finite optimizer moments; all 62 sentence-memory tensors and all other 213 model tensors remained bitwise frozen |
| Allocation/network | Exactly two `pair02` nodes/GPUs and one rank/node. Both 200-Gb/s ConnectX-7 RoCE rails passed MTU 9000, GID 5, connectivity/counter and NCCL `NET/IB`/no-Socket checks; the gradient benchmark selected single-primary `rocep1s0f1:1` |
| Failure | The full-config normalizer omitted only `sentence_memory_safety.stage_c.resolved_source_provenance.declared_contract.arm` and its derived provenance digest, even though the later dedicated provenance comparison already normalizes both. It raised `StageCDecisionError` before publication; this is an implementation-validation defect, not a scientific-config, checkpoint-audit, data, or network failure |
| Data/W&B boundary | Only train/validation tables were staged. No test or confirmation access occurred, the global spend marker remains absent, W&B was disabled, and the retained smoke outputs are non-authorizing invalid-attempt evidence |
| Incident archive | No-replace `...generator_adaptation_protocol_v4_run_r5_smoke.invalid_attempts/source_36d121dd361df2c08616031bd7c2076d85da78d0_smoke143609_pilot143610/ARCHIVE.json`, schema v4, 39,609 bytes, SHA256 `9c4e2213ca51941b41281f0b869fc66aa4e440df846ff57dee2731fca5f1c41a` |
| Authorization | None. R5 checkpoints, calibration, controls, and source may not be reused for execution. Only separately preregistered protocol-v5/run-r6 may apply the two-field normalizer correction with fresh source/config/gate/calibration/output/control namespaces |
| Runbook | `docs/NIAF/continuous_trajectory_field/stage_c_generator_adaptation_protocol_v4_run_r5.md` |
| Historical alias | Preserved, but no longer the dataset-qualified CSL-Daily resolution |

## Terminal CSL-Daily Stage-C protocol-v3/run-r4 zero-science CPU-test failure

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-protocol-v3-run-r4-20260910` |
| State | **Terminal zero-science CPU-test failure.** CPU job 143593 failed after compileall, Ruff, and the repository test suite reported one fixture-only failure; it published no CPU READY. Calibration 143594 and two-GPU smoke/pilot 143595--143596 were dependency-cancelled with zero runtime/allocation |
| Exact source | Standalone clone `/media/cvpr/haomian/SignTrajField_centered_stage_c_v3_run_source_r4`, clean commit/remote `740400412386d60a8d6f56b0f871bc16dee755b0`, tree `23613b49d5ac9d9ac1c9ec0ef17320b6e80d5249` |
| Failure | `test_protocol_v3_archived_clone_timeout_is_split_and_runtime_skips_it` supplied the standalone clone as `evidence_root`; the manifest correctly requires the canonical shared evidence root. This does not implicate the runtime scripts, which pass the canonical root |
| Scientific/access boundary | No calibration, data/neighbor-table open, lease, GPU allocation, optimizer update, validation batch, trainer, output, confirmation/test access, or W&B activity. The source checkpoint and all scientific settings were not executed |
| Incident archive | `...csl_daily_stage_c_generator_adaptation_protocol_v3_run_r4_cpu.invalid_attempts/source_740400412386d60a8d6f56b0f871bc16dee755b0_cpu143593_dependents143594_143596/ARCHIVE.json`, SHA256 `a23682b584bcc05ccee67ccec7526d33efc924104a0ea34209740bb4131e5f71`, 11,168 bytes |
| Authorization | None. R4 cannot resume, retry, or seed an execution. Only a separately preregistered protocol-v4/run-r5 source can repair the CPU-test fixture, preserving the frozen science |
| Runbook | `docs/NIAF/continuous_trajectory_field/stage_c_generator_adaptation_protocol_v3_run_r4.md` |

## Terminal CSL-Daily Stage-C protocol-v2/run-r3 zero-science preflight failure

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-protocol-v2-run-r3-20260910` |
| State | **Terminal zero-science CPU preflight failure.** CPU job 143574 timed out in archived-clone local Git-status validation before compile, Ruff, pytest, data access, calibration, allocation, or science; dependents 143575--143577 were cancelled with zero runtime/allocation. It cannot resume or retry |
| Why a new generation | Run-r2 produced one optimizer update and began validation before its evaluator-dispatch failure. Its v1 contract therefore makes that attempt terminal and forbids a retry. Protocol v2 is a wholly fresh generation defined from the operational failure class, without a pilot decision or validation outcome and without changing scientific choices |
| Exact source | Both arms independently reload the Stage-B epoch-5/global-step-360 `best_infeasible.pt`, SHA256 `b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`; terminal decision SHA256 `8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5`, identity `7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`, status `valid_infeasible` |
| Matched arms | Sequential `memory` (`dropout`, probability `0.25`) and `matched_off` (`off`, probability `1.0`) arms in one allocation; each resets optimizer, epoch/global-step, selection, and RNG state. Paired-corruption training remains disabled |
| Trainability | All 62 sentence-memory tensors frozen; exactly 20 generator tensors/1,109,395 parameters trainable under seven pinned hypernetwork prefixes; all other 213 model tensors frozen |
| Batch and duration | Exactly two ranks/two GPUs; batch 64/rank and accumulation two, effective global batch 256. Smoke is one optimizer update per arm; pilot is one uncapped epoch/global step 72 per arm |
| Allocation | Exactly two nodes from one explicit `pair01`--`pair15`, one GPU/rank/node. Both ConnectX-7 f1 RoCE rails must pass 200-Gb/s link, MTU 9000, GID 5, peer, raw-counter, NCCL `NET/IB`, and no-socket checks; the exact 4,437,580-byte gradient benchmark chooses single or dual rail |
| Sole implementation change | Centered 11-mode evaluation dispatches to the paired evaluator when paired-corruption training **or centered sentence-memory evaluation** is enabled. A companion fail-fast guard rejects centered-only mode names in otherwise noncentered, paired-off custom configs. This fixes the run-r2 `motion_shuffled_n0` namespace failure without enabling paired training or changing any checked-in arm, metric, mode, hyperparameter, threshold, data split, or duration |
| Ordered gates | Fresh complete CPU/compile/Ruff/tests on the final clean pushed HEAD -> fresh source-bound train-only calibration -> fresh paired forced-IB network and one-update two-arm smoke -> one-epoch paired pilot only after atomic `smoke_ready` |
| Isolation | New `protocol_v2_run_r3` configs, scripts, policy/recovery manifest, calibration, prerequisite, smoke, pilot, arm-output, lease, and log namespaces. Run-r2 `last.pt`, calibration, outputs, leases, and logs cannot be reused, moved, deleted, or modified |
| Decision boundary | Smoke pass is only `smoke_ready` -> `run_one_epoch_development_pilot`; every smoke failure is `stop`/`none`. Pilot pass is only `pilot_complete_development_signal` -> `none_requires_fresh_preregistration_without_pilot_outcome_access`; every pilot failure is `stop`/`none`. No status authorizes a longer run, retry, promotion, confirmation, or test |
| Data boundary | Only train and the sealed development subset of validation; only train/validation neighbor tables may be staged. No test/confirmation input may be referenced or opened, the global confirmation-spend marker must remain absent, and W&B is disabled |
| Incident prerequisite | No-replace run-r2 archive `...generator_adaptation_smoke.invalid_attempts/source_f12b993b5de361423df3b8cbfb4e873f4a95ad1e_smoke143541_pilot143542/ARCHIVE.json`, SHA256 `82ce35cf3d7337f218bda189080a45b9e3faedec200750310d0958296f9d1855`, must be reopened and validated before each gate |
| Immutable source | Intended branch `codex/csl-daily-centered-generator-stage-c-v2-run-r3` and standalone clone `/media/cvpr/haomian/SignTrajField_centered_stage_c_v2_run_source_r3`; launch is forbidden until the final pushed HEAD, matching remote ref/head, and all source/config/script hashes are bound |
| Runbook | `docs/NIAF/continuous_trajectory_field/stage_c_generator_adaptation_protocol_v2_run_r3.md` |
| Default alias | Global PHOENIX default unchanged; this alias is only the dataset-qualified CSL-Daily resolution |

## Stopped CSL-Daily Stage-C retry2-v1/run-r2 after partial smoke

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-pilot-run-r2-20260910` |
| State | **Terminal stop after partial scientific output; not retryable under retry2-v1/run-r2.** CPU job `143539` and fresh calibration job `143540` completed. Smoke job `143541` failed after one optimizer update and partial validation. No pilot result exists |
| Immutable source | Clean pushed commit `f12b993b5de361423df3b8cbfb4e873f4a95ad1e`, `origin/codex/csl-daily-centered-generator-stage-c-v1-run-r2`, standalone clone `/media/cvpr/haomian/SignTrajField_centered_stage_c_run_source_r2` |
| Exact source | Stage-B epoch-5/global-step-360 `best_infeasible.pt`, SHA256 `b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`; terminal decision SHA256 `8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5`, identity `7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`. Its exact legacy schema omits top-level `authorized_purpose` and explicitly records no confirmation/test access |
| Matched arms | Sequential `memory` (`dropout`, probability `0.25`) and `matched_off` (`off`, probability `1.0`) arms in one allocation; each independently reloads the same source and resets optimizer, epoch, selection, and RNG state |
| Trainability | All 62 sentence-memory tensors frozen; exactly 20 generator tensors/1,109,395 parameters trainable under seven pinned hypernetwork prefixes; all other 213 model tensors frozen |
| Batch and duration | Exactly two ranks on two GPUs; batch 64/rank and accumulation two, effective global batch 256; one uncapped epoch and global step 72 per arm after an exact one-update smoke |
| Completed prerequisites | CPU gate `143539` completed compile, Ruff, and all 424 repository tests and published source-bound `READY`. Calibration `143540` published accepted train-only artifact `be670961a7ab1c3976ceba3427a4541c629517a08b46e65070e91699a85d4419`; no confirmation/test access |
| Verified allocation/network | Smoke `143541` used exactly nodes `ADUAED21043WKLX04` and `ADUAED21044WKLX03`, pair `pair02`, two ranks/two GPUs. Both 200-Gb/s RoCE rails, peer connectivity, raw pre-counters, forced-NCCL `NET/IB`, and no socket fallback passed. The gradient benchmark selected single-primary `rocep1s0f1:1` because dual was 1.7218 times the fastest-single gradient latency |
| Partial scientific output | Only `memory` started. It completed logical batches 1/2 with printed losses `12.083765`/`11.824976`, wrote `last.pt` at epoch 1/global step 1 with validation pending, and computed the first `off`, `on`, and `motion_shuffled_n0` validation batches. No validation value, metrics file, completed checkpoint, scope audit, or decision was published; all 20-changed/213-frozen post-update assertions therefore remain unaudited |
| Failure | Both ranks raised `KeyError: 'motion_shuffled_n0'` because the dispatcher tied the centered evaluator to the disabled paired-training flag and entered a legacy five-namespace evaluator. Rank 1's later TCPStore error was consequential, not primary |
| Unrun arm and pilot | `matched_off` never started. Pilot `143542` remained `DependencyNeverSatisfied` with zero runtime/allocation and no logs, then was explicitly cancelled at `2026-09-10T03:54:12` |
| Publication and access | No smoke `READY`, `COMPLETE`, publication, selection summary, or decision exists. No pilot source namespace/result exists. Only train/validation tables were staged; no confirmation/test access occurred and the global spend marker remains absent |
| Scheduler-history caveat | The terminal fields are operator-attested prior live-`scontrol` observations. The controller has since purged jobs `143539`--`143542`, current `squeue` returns no rows, and `sacct` is unavailable because `slurmdbd` refuses the connection; no historical scheduler file is fabricated |
| Incident archive | Evidence remains in place and is hash-bound by no-replace `...generator_adaptation_smoke.invalid_attempts/source_f12b993b5de361423df3b8cbfb4e873f4a95ad1e_smoke143541_pilot143542/ARCHIVE.json`, SHA256 `82ce35cf3d7337f218bda189080a45b9e3faedec200750310d0958296f9d1855` |
| Authorization | None. Run-r2 cannot resume or rerun and its checkpoint/calibration cannot seed a new execution. It authorizes no longer run, promotion, confirmation, or test. Only the separately preregistered protocol-v2/run-r3 generation can proceed from a fresh source-bound CPU gate/calibration/smoke namespace |
| Runbook | `docs/NIAF/continuous_trajectory_field/stage_c_generator_adaptation_retry2_v1.md` |
| Historical alias | Preserved, but no longer the dataset-qualified CSL-Daily resolution |

## Stopped CSL-Daily Stage-C run-r1 before science

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-stage-c-generator-adaptation-pilot-run-r1-20260910` |
| Source | Clean pushed run-r1 commit `a883baf4a0a95b4ebb2007564837c4d9b4f817cc`, branch `origin/codex/csl-daily-centered-generator-stage-c-v1-run-r1` |
| Completed prerequisites | CPU gate job `143523` completed 422 tests and published its source-bound gate. Calibration job `143524` published accepted train-only artifact identity `8999d9e81b2fccab6ed368d374e99a9f65f0dcdb4c8203f2d41d5e02f469cf49` and an exact semantic/map-equivalence transition audit |
| Failure | Smoke job `143525` stopped after 12 seconds in the pre-science config validator with `KeyError: 'authorized_purpose'`: the runner indexed a field which the exact pinned legacy decision schema intentionally omits |
| Scientific-output audit | **Zero scientific output.** No smoke execution lease, RoCE/network preflight, trainer process, optimizer update, checkpoint, decision, READY, or pilot publication was created |
| Dependent pilot | Job `143526` remained dependency-blocked, received no allocation/runtime, and was explicitly cancelled after the dependency became unsatisfiable |
| Preserved evidence | Run-r1 CPU/calibration controls remain below `.../csl_daily_stage_c_generator_adaptation_prerequisites/source_a883baf4a0a95b4ebb2007564837c4d9b4f817cc`; its calibration remains at `.../csl_daily_sentence_memory_relevance_calibration_stage_c_generator_adaptation_v1`. Neither may be overwritten, deleted, or reused as retry-2 source evidence |
| Authorization | None. The incident does not change Stage B, authorize a retry beyond the preregistered run-r2 contract, or permit confirmation/test/promotion |
| Runbook | `docs/NIAF/continuous_trajectory_field/stage_c_generator_adaptation_v1.md` |

## Latest completed CSL-Daily: Phase-A''' centered-evidence and absolute-binding experiment

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-phase-a-centered-absolute-binding-v1-20260909` |
| State | **Ordered protocol complete; stopped by the Stage-B development gate and not deployable.** Stage A and Stage B were integrity-valid but scientifically infeasible. Neither stage produced `best.pt`; no locked diagnostic, confirmation, checkpoint promotion, test evaluation, Phase B, or further adaptive stage was authorized |
| Ordered arms | Stage A used per-layer/head/slot/part centered candidate-motion covariance values plus a frozen absolute relevance gate. Stage B added absolute candidate text-motion association with balanced BCE and symmetric multi-positive InfoNCE, but started independently from v2 and never loaded Stage-A weights or optimizer state. Both stages removed the Gaussian temporal prior |
| Dataset and controls | Full 18,399-row training set, `K=8`, top-M 64, seed 1234, candidate dropout 0.10, 90% motion-only and 10% full corruption. Selection used 256 equal-weight normalized-text development clusters/347 signer rows. Partition digest `80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e`; fixed control-map digest `5142f61a2e21d407cdfd9516440b7048c732ab4b7d872f9c666db3d56f437c31` |
| Source | Clean standalone clone `/media/cvpr/haomian/SignTrajField_centered_run_source_r3`, immutable branch `origin/codex/csl-daily-centered-memory-v1-run-r3`, commit `191b5c28515bff446926185ac965bbe6e8701e93`. Local and live origin heads were equal before, during, and after all eligible jobs; the immutable branch was not advanced by this registry update |
| Relevance calibration | CPU-only Slurm `143376` completed and published accepted train-only retry3 artifact identity `668694ff012a5b71bedeab42a8139b7a95014b9b790ec8d44514cdd50cac933d`. Held-out AUROC `0.9817280587`; positive-minus-negative probability gap `0.7923572567`; `a=68.17342257577045`, `b=0.8278962850773125`. Calibration/map/READY SHA256 `aa98927325147956d68caa0a66979d8d64a746b8812899d15e90cf4f09755775` / `948a7c2f7a8ec8fdbcfa8a9c130e665adfdf7615b8908cc31324b77c15e61411` / `e9da4d8240092c54121a733fd466aa6f0822802fad639b1eed10d50e20e5c394` |
| Engineering gates | Export-identity focused Slurm `143374`: **22 passed**. Complete CPU Slurm `143375`: compileall and Ruff passed; all 30 `tests/test_*.py` files passed, **387 passed**, 13 warnings. Sequential one-GPU smoke `143377` completed both fresh-v2 arms for exactly one optimizer step, with 10/11 modes, finite state and Stage-B association gradients, frozen base, exact parity/invariants, train/validation-only staging, and W&B disabled |
| Initialization | Each full stage strictly imported frozen v2 SHA256 `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`, reset optimizer/epoch state, and preserved exact memory-off prediction and duration parity. Stage A exposed 59 trainable sentence-memory tensors; Stage B exposed 62; the inherited 171 base/planner tensors remained bitwise unchanged |
| Stage-A training | Slurm `143380`, four nodes/GPUs, 16 CPUs and 100 GiB per node, finished five development validations/global step 360 and early-stopped at patience two. Scheduler `FAILED/143` is the intentional fail-closed `require_feasible` exit, not a CUDA, NCCL, OOM, numerical, or staging failure. Launch identity `0b1b9a8c089f88d9128027016cd5bb1ea7b39d3a67d5b5ab76c7dda2a078d97e`; config SHA256 `c5497e6a912261a00c48dd1fa3174305bf66c83212e25e9fbc93cad629273763`; authenticated failed-training lease claim `e588e1801b42148074b90970b42e60ea2edbb5a9fefee97c39208f498595cbb8` remains retained |
| Stage-A selected checkpoint | `best_infeasible.pt`, epoch 3/global step 216, SHA256 `39a66ba0422ad5be1ccf2745ea7464d3c43894f3cc7c53c0a83e0dffcd979cb6`; metrics SHA256 `2aeaf86de44cac053f9c5bf2ac37364820b675cb266ba80809339d9733c5d210`; selection-summary SHA256 `3c518eb472ff857f04dd2e151731c235bbb85cd81004138774a9eee36612f679`. It is audit-only and non-promotable |
| Stage-A decision | Slurm `143395` completed the full selected-checkpoint replay and three 347-row predicted-duration integrity exports. Canonical path `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1/evaluation/ordered_development_decision`; status `valid_infeasible`; decision identity `12c4fd20233f218422b36d977829cca5957c7c35c4a4e6be4aceec178642e837`; decision/READY SHA256 `73bcec6eab500588875edaf0f74661d1b2aa8548b79e5d3928403f78f29f3104` / `cd0fe2a9c4e7db59b9fbabb9c331ae7c226f0157de5e78014c6995a6543af020` |
| Stage-A science | Correct/text-only/mean-derangement/cross-query/full composite `12.6233378644 / 12.7318543089 / 12.6255987817 / 12.6262668752 / 12.6917002527`. Correct improved text-only by 0.8523% and full replacement by 0.5386%, but mean derangements by only **0.0179%** and cross-query motion by **0.0232%**, below 0.25%; `Rpair=0.1215702175`; identity utility `0.0208347904` |
| Stage-B authorization | Stage A emitted only `authorize_stage2.json`, authorization identity `ecca886ee2252a0db74f8ca8c404100cb1ea2fa8690f5e1e36bb68cc5e654d15`, file SHA256 `81978bc51af76741f9330010036e4c55e93bc219c6d8a0bfeb7f4397901c4b3a`. Stage-B fresh-v2 ordered-input identity `7a196496561c32df8ba1a71257775c31b20c75cad2944f200f7baee22facab57` preserved that predecessor chain |
| Stage-B training | Slurm `143400`, same four-node envelope, completed all six validation events/global step 432, then returned the intentional no-feasible `FAILED/143`. Launch identity `59a0015755ed927f506b0e6785b779092f56578317365cac7212c99b76d8d029`; config SHA256 `7741da46d37a4b77f481663f25fa580281f6d29de6c2616baebcc30dac59b85e`; retained authenticated lease claim `3868b9018b8ddfb7f08ecf85dd0ac3e4cebba138ef4f29b0eb165a73aec66a7d` |
| Stage-B selected checkpoint | `best_infeasible.pt`, epoch 5/global step 360, SHA256 `b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`; metrics SHA256 `9171e1e65ceb57c6b291cf24825eb934bcfd4983f7d822fe01e56af33e4c9b10`; selection-summary SHA256 `41b5bcb5eeece4132f1c576399232010ac00e6c0fec168327d2c9d03095cd561`. It is audit-only and non-promotable |
| Stage-B decision | Slurm `143427` completed all integrity work at `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1/evaluation/ordered_development_decision` and wrote canonical `valid_infeasible`, `integrity_valid=true`, and `development_feasible=false`; the exact legacy v1 schema omits top-level `authorized_purpose` and explicitly records no confirmation/test access. Decision identity `7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`; decision/READY SHA256 `8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5` / `fdc8e0823ccd2b3f769b5fdf31cb21098cf719a542cd9556546ee91fa174cfde` |
| Stage-B science | Correct/text-only/mean-derangement/cross-query/full composite `12.6293070649 / 12.7318543089 / 12.6497460611 / 12.6726873684 / 12.6987959243`. `Rpair=0.4059177979`; all nonces favored correct; matching top-1 `0.4660644531`. The mean-derangement gain was only **0.1616%**, identity utility `0.1993129743`, and true-minus-best-negative association margin `-0.0091593001`, so all three required gates failed |
| Integrity identities | Stage-A/Stage-B architecture digests `a41da52ab3005f55f4c428926813cb6ff4244cb5c84f7764281d0e8e539c4c62` / `bc69fd35ac58e10bc894460c35175f13356b79416df40e8c57156b1929614236`; objective digests `8267d7875a277e9b9ffcc7fc4e4d04e0db3f11565e6a251f987a80db33680821` / `9180692d6585b0e4e508b0456e1d3562144db21742bab6b84c6570779300d484`; shared evaluation-control `359315546b1fe4a2542d66ee784d30d5ed222e01dcd832eb00402cd57f915743`; selection aggregation `cd5754faf3a45267452c0245589f1a57b9ab30cdb8773e40154be797cddca50f` |
| Exact invariants and isolation | Every selected-checkpoint audit reported exact zero for uniform-final, broadcast-complete, all-null, joint-tuple, and v2 prediction/duration deltas, with all-null gate/candidate mass zero and null mass one. Eligible jobs staged only train and validation neighbor tables, with test absent before provider construction; all decision flags say `test_data_accessed=false` and `confirmation_manifest_opened=false`; W&B was disabled. Calibration and ordered-decision leases were released into immutable history; the two no-feasible training leases remain intentionally retained |
| Recovery history | The initial source/smoke incident at `578cbb5`, parity-replay recovery archive `source_5a884505_job143304_decision143319`, and export-schema recovery archive `source_cacd362_job143353_decision143367` remain preserved and documented in the runbook. None authorized Stage B or confirmation; the final r3 protocol restarted calibration, smoke, and Stage A fresh |
| Final scientific decision | Centering successfully removed the uniform candidate-average carrier and absolute association substantially increased causal sensitivity (`Rpair` from 0.1216 to 0.4059) and matching accuracy. It still failed to turn enough of the generic memory benefit into correct identity-specific motion utility. Per the predeclared rule, **stop after Stage B**; do not weaken gates, promote either `best_infeasible.pt`, open confirmation/test, run the locked diagnostic, begin Phase B, or launch another adaptive stage |
| Outputs | Stage A: `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1`; Stage B: `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1`; calibration: `experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_memory_relevance_calibration_v1_retry3` |
| Confirmation/test/diagnostic status | No `authorize_diagnostic.json` or `authorize_confirmation.json` exists. The global confirmation-spend marker remains absent; no confirmation export/analysis, diagnostic, test evaluation, promotion, or Phase-B job was run |
| Runbook | `docs/NIAF/continuous_trajectory_field/phase_a_centered_relevance_v1.md` |
| Default alias | Unchanged |

Terminal logs:

```text
logs/sbatch/csl_r3_fix_143374.out
logs/sbatch/csl_centered_cpu_143375.out
logs/sbatch/csl-rel-cal-v1_143376.out
logs/sbatch/csl_centered_smoke_143377.out
logs/sbatch/csl_centered_a_143380.out
logs/sbatch/csl_centered_decide_143395.out
logs/sbatch/csl_centered_b_143400.out
logs/sbatch/csl_centered_decide_143427.out
```

## Previous completed CSL-Daily: Phase-A'' factorized-memory experiment

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-phase-a-factorized-memory-v1-20260908` |
| State | **Ordered protocol complete; stopped by the Stage-2 development gate and not deployable.** Stage 1 and Stage 2 were integrity-valid but scientifically infeasible. Neither stage produced `best.pt`, so no confirmation, locked temporal-slot diagnostic, test evaluation, checkpoint promotion, or Phase B was authorized |
| Ordered arms | Stage 1 factorized metadata-only keys and motion-only values with uniform valid-token support. Stage 2 added only the fixed Gaussian slot/token bias (`sigma=0.25`) after a provenance-bound Stage-1 authorization; it started fresh from v2 and did not import Stage-1 model or optimizer state |
| Dataset and controls | Full 18,399-row training set; epoch validation on 256 equally weighted normalized-text development clusters/347 signer rows. Partition digest `80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e`; fixed validation corruption-map digest `403b57d632ede59023cf8470beafc1c4c4616503a0d8c2b398931d65868cfc06`. The 540-cluster/728-row confirmation partition remained unopened |
| Shared protocol | Seed 1234; `K=8`, top-M 64, duration weight 0.05, temperature 0.10, candidate dropout 0.10; Phase-A' paired correct/corrupt objective; frozen generator; only 59 `hypernetwork.sentence_memory_*` tensors trainable; four epochs/four development validation events per stage; five modes including explanatory-only analytic prior |
| Initialization | Each eligible stage used a fresh strict import from frozen v2 SHA256 `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`. Initial and selected-checkpoint memory-off prediction/duration parity were exactly `0.0`; all-null predictions and durations were bit-exact to memory-off with zero gates/candidate mass, null mass one, and zero motion-payload reads |
| Source | Clean standalone shared clone at final executable commit `ec11f7aac47de13315ddc3123303bd77f7b32d44` (`Stage only train and val for factorized decisions`). Both `origin/codex/csl-daily-factorized-memory-v1` and the immutable `origin/codex/csl-daily-factorized-memory-v1-run` were at that commit before the eligible launch; the run branch remained fixed throughout the protocol |
| Engineering gates | Full CPU Slurm `143196` completed: **308 passed**, 12 warnings, in 238.49 seconds. Repaired one-GPU smoke `143197` completed and exercised Stage 1 then Stage 2 for exactly one optimizer step each, with finite outputs/state, frozen-base enforcement, exact v2 parity, all five validation modes, train/validation-only staging, and W&B disabled |
| Eligible Stage 1 | Slurm `143198`, four nodes/GPUs, completed all four epochs and global step 288, then returned the intentional fail-closed scheduler status (`FAILED`, exit `143`) because no feasible checkpoint existed. Launch identity `a041d80035cd1259f20b65e94fadaea76fcc8420ee472b14c3f26331bb54f794`; config SHA256 `99984578cf7ea8ed537b589fa576c3b639fe3f8b19c4ea3831686ccfb93e36de` |
| Stage-1 decision | Slurm `143259` completed the sealed 347-row development integrity audit and wrote `valid_infeasible`, decision identity `74b0784d261d8a9d291b28938b7cac37a28495a82c1e08f45dd73591e82610f6`. Correct composite `12.6261835001` versus text-only `12.7318543089`, motion-shuffled `12.6262922686`, and full-shuffled `12.6273469365`; corresponding improvements were 0.82997%, **0.000861%**, and **0.009214%**, with `Rmotion=0.00344204`. Hand paths improved 18.21%/33.88%, but the required 0.1% corruption margins and `Rmotion >= 0.05` failed |
| Stage-1 checkpoint | `best_infeasible.pt`, epoch 4/global step 288, SHA256 `af28da2d80099ff7194147002722c14d2198368febe16afff2aa21ac3220a77b`; metrics SHA256 `c798f08a52d01ddc5a41695c0382ce3da2e883649f81ef6f931214ba1d51988a`; selection-summary SHA256 `fbf8ee43ebbd1b49aaac8af155ac981967760a1c21876f9565988bf49f463fc5`. It was retained for audit only and must not be promoted |
| Stage-2 authorization | Stage 1 emitted only `authorize_stage2.json`, SHA256 `12fb2f225c751b425208c6c6e52d76293a9df95759a21dff1c5e7aa24d2b2406`, authorization identity `d15bb66e67baec1f505201288cc879d87430803d444fa82837cea09066a14075`. Stage-2 launch input identity `b44bfd8746b236ad85d7983b0662e47413f457aa5f810fb7b7a75b2258348ea0` preserved that chain |
| Eligible Stage 2 | Slurm `143261`, four nodes/GPUs, fresh from v2, completed all four epochs and global step 288, then returned the same intentional fail-closed scheduler status (`FAILED`, exit `143`). Launch identity `165de4b4e65edc12a954ca09cd5d019618d4a7e3c8eb98cf4a3d5c13c7f9ce09`; config SHA256 `f80ab48ed91adaa6929db9d872d6f1fe3eb9bbe968f8ccc742519d0fd7420a06` |
| Stage-2 decision | Slurm `143277` completed the sealed development integrity audit and wrote `valid_infeasible`, decision identity `b8dd760128d39f858d654668131621bae8cbd8114fe71455d0931cb1e17570b7`. Correct composite `12.6256599036` versus text-only `12.7318543089`, motion-shuffled `12.6257268809`, and full-shuffled `12.6263435677`; corresponding improvements were 0.83408%, **0.000530%**, and **0.005415%**, with `Rmotion=0.00253200`. Hand paths improved 18.23%/33.90%, but the corruption margins and motion-sensitivity gate again failed |
| Stage-2 checkpoint | `best_infeasible.pt`, epoch 4/global step 288, SHA256 `e7c1e625088b160afe830da90aaaa612c2b3bda06d4e81c00b3eeb8c28c3fd46`; metrics SHA256 `d3cd8e7d66befa01f3eff28c55748bcaae8e805fa6fb5dc2479332756a1cf637`; selection-summary SHA256 `a66eff3a3c034e2170ebce330fa0786125831f3573497110409a84db76ac2660`. It was retained for audit only and must not be promoted |
| Identity evidence | Stage-1/Stage-2 architecture digests `918c1aaf1b36abf803b3335f207a2d479313235cfe1fba2dc075b151101d4f8a` / `d1562e4bf8d23b59997881129160418f7fec65fc576a29a0bfb9cd1a01d07eb3`; shared objective `8267d7875a277e9b9ffcc7fc4e4d04e0db3f11565e6a251f987a80db33680821`; evaluation control `c18aa1206d659bf096248a5bd194ab2c0ccfc36cacfad1ee8191929fb2156a58`; cluster aggregation `5370fb44c8a505b0ec2a7087c3f5dd88c280f4734416108cef9bac1e26b0f7df`; bank ID `a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd` |
| Eligible isolation | Jobs `143198`, `143259`, `143261`, and `143277` staged/audited only the train and validation neighbor tables. Their ordered decisions attest `test_data_accessed=false` and `confirmation_manifest_opened=false`; W&B was disabled throughout |
| Excluded integrity incident | Initial Stage-1 training `143170` at source `50a3854a38a3f3c0de1be0c3cccb15c026a7e768` completed four epochs but was excluded. Its decision `143194` incorrectly opened the shared full bank instead of a train/validation-only local stage, SHA256-read `neighbors_test.npz` metadata, then failed before any trajectory export because provider identities `{train,val,test}` differed from the checkpoint's `{train,val}`. No test example, test manifest, prediction, metric, confirmation row, or authorization was produced. The staging defect was repaired in `ec11f7a`, and the entire ordered experiment was restarted fresh rather than reusing the affected checkpoint |
| Incident preservation | Excluded run: `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1.integrity_failed_train143170_decision143194_test_neighbor_access`; matching launch prerequisites use the same path plus `.prerequisites`. Both are retained unchanged for audit |
| Final scientific decision | Factorized K/V separation and the fixed temporal prior still yielded a generic memory-on improvement without meaningful dependence on the retrieved motion. Per the predeclared stop rule, stop after Stage 2; do not weaken gates, inspect confirmation/test outcomes, promote `best_infeasible.pt`, or begin Phase B |
| Outputs | Stage 1: `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1`; Stage 2: `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1` |
| Confirmation/test/diagnostic status | No `authorize_confirmation.json` exists. No confirmation spend marker, confirmation export/analysis, test-set evaluation, locked temporal-slot diagnostic, or Phase-B job was launched. The only test-related access was the explicitly excluded `143194` neighbor-table metadata incident described above |
| Runbook | `docs/NIAF/continuous_trajectory_field/phase_a_factorized_memory_v1.md` |
| Default alias | Unchanged |

Logs and terminal evidence:

```text
logs/sbatch/csl_v3_factorized_cpu_repair_full_143196.out
logs/sbatch/csl_v3_splitkv_smoke_143197.out
logs/sbatch/csl_v3_splitkv_s1_143198.out
logs/sbatch/csl_v3_splitkv_decide_143259.out
logs/sbatch/csl_v3_splitkv_s2_143261.out
logs/sbatch/csl_v3_splitkv_decide_143277.out
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1/evaluation/ordered_development_decision/decision.json
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1/evaluation/ordered_development_decision/decision.json
```

## Previous completed CSL-Daily: SignTrajField-RAG v3 Phase-A' motion-contrast v1

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-phase-a-motion-contrast-v1-20260907` |
| State | **Stopped by the scientific gate; not deployable.** The full engineering run completed all configured training and development-validation work, but no feasible checkpoint was produced |
| Dataset | Full CSL-Daily train split (18,399 rows) plus the validation-only novel-text partition. Epoch selection used 256 development texts/347 signer rows; 540 confirmation texts/728 rows stayed hidden. The two exact-seen validation rows were descriptive only |
| Training job | Slurm `143050`, name `csl_v3_motion_contrast` (`FAILED`, exit code `143:0`, runtime 1:09:03). This scheduler state records the intentional fail-closed `require_feasible` exception after epoch 4; it is not an OOM, NCCL, staging, or training failure |
| Configuration | `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml`; SHA256 `277592b4cebb5d07da4e0cb7f9c162501e6c2ac3fa4860c3459fc1d2ae6b71f5` |
| Output | `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1` |
| Source | Branch `codex/csl-daily-signtrajfield-rag-v3`, commit `16b627681ab96a1dc086cc334de815a9ae990800`, pushed to `origin` before submission |
| Initialization | Fresh strict import from the frozen CSL-Daily v2 text-only `best.pt`, SHA256 `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`; no resume or warm start. Prediction and duration parity were both exactly `0.0` against the v2 model (tolerance `1e-7`) |
| Controlled objective change | Every train sample retained its correct-memory task loss and received one auxiliary corrupt copy: deterministic motion-only payload shuffle 90% of the time and full candidate shuffle 10% of the time. Benefit, detached ranking, and fallback losses were enabled at unit weight; sentence sparsity remained `1e-4`. Architecture, frozen generator, bank, `K=8`, top-M 64, duration weight `0.05`, temperature `0.10`, candidate dropout `0.10`, and seed `1234` were unchanged |
| Frozen/trainable | Phase A' only: all inherited v2 parameters stayed frozen; only the 82 `hypernetwork.sentence_memory_*` tensors trained. Phase B was disabled |
| Validation partition | SHA256 ordering of normalized novel texts; artifact digest `80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e`. Durable artifact: `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition` |
| Validation protocol | Four paired development modes every epoch: text-only, correct retrieval, motion-only shuffle, and full shuffle. Four complete validation events were recorded; no validation batch cap was active |
| Allocation | Four nodes (`ADUAED21020WKLX07,ADUAED21032WKLX20,ADUAED21033WKLX27,ADUAED21036WKLX05`), one GPU/node, 16 CPUs/node, and 100 GiB/node; rank 0 on node 07 |
| Batch semantics | Batch 32/GPU, accumulation 2, global effective batch 256; input microbatch cap 8 and frame cap 2048 for the doubled trainable forward |
| Memory staging | The verified bank and only the train/validation neighbor tables were staged under job-local `/tmp` and strictly audited on every allocated node. No test neighbor table was staged |
| W&B | Disabled; no external run or online credential dependency |
| Engineering validation | CPU Slurm `143046` exposed one stale synthetic fixture after the unsalted partition correction (241 passed, 1 failed); targeted repair job `143047` passed 4/4; final CPU Slurm `143048` passed the complete suite (242 tests, 12 warnings). Mandatory one-GPU smoke `143049` completed with exit code 0, exactly one optimizer step (`global_step=1`), exact v2 parity, all four validation modes, and W&B disabled |
| Best infeasible checkpoint | Epoch 3/global step 216: correct score `12.618111`, text-only `12.714579`, motion-shuffled `12.618139`, full-shuffled `12.618706`, `Rmotion=0.00293078`, constraint violation `0.943335`. It was selected lexicographically for the lowest violation, not promoted |
| Terminal epoch | Epoch 4/global step 288: correct score `12.607652`, text-only `12.714579`, motion-shuffled `12.607613`, full-shuffled `12.603107`, `Rmotion=0.00267572`, constraint violation `0.948849`. Correct retrieval remained effectively tied with motion shuffle and was worse than full shuffle |
| Scientific outcome | Failed the correct-over-motion/full 0.1% development margins and the internal `Rmotion >= 0.05` requirement. The run therefore stopped as predeclared; it does not justify Phase B, weaker gates, or checkpoint promotion |
| Checkpoints | `epoch0001.pt` through `epoch0004.pt`, `last.pt`, and `best_infeasible.pt` exist. **No `best.pt` exists**, and `best_infeasible.pt` must not be promoted or deployed |
| Confirmation/test status | The 540-text confirmation partition was constructed but never evaluated or used for selection; no confirmation/default-DTW/PA-DTW export was run, and no test-set data was accessed. The selected-checkpoint temporal-slot audit was also not launched because the feasibility prerequisite failed |

Historical inspection commands:

```bash
scontrol show job 143050
tail logs/sbatch/csl_v3_motion_contrast_143050.out
tail logs/sbatch/csl_v3_motion_contrast_143050.err
```

Run metrics and selection state are stored in:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1/metrics.jsonl
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1/selection_summary.json
```

## Previous: CSL-Daily SignTrajField-RAG v3 Phase-A duration-weight 0.05 pilot

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-phase-a-dw005-pilot2-20260906` |
| Dataset | Full CSL-Daily train/validation splits (18,399/1,077 samples); this is a two-epoch controlled pilot, not a reduced-data run |
| Training job | Slurm `142893`, name `csl_rag_dw005_p2` (`COMPLETED`; both configured epochs and global step 144 were written) |
| Configuration | `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2.yaml`; SHA256 `80843eb89075d9829c9db829b16b5d0930a6b6bce91262e4f655a4b4e70febdb` |
| Output | `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2` (fresh output; the duration-weight 0.10 experiment is not resumed or overwritten) |
| Source | Branch `codex/csl-daily-signtrajfield-rag-v3`, commit `cfe41628efe99cad7f33666da26278029d4b2fac`, pushed to `origin` before submission |
| Initialization | Fresh strict v2-to-v3 base import from `csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt`; SHA256 `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`; no resume or warm start |
| Controlled change | Sentence duration-reranking weight is reduced from `0.10` to `0.05`, and the run length is reduced from 10 to 2 epochs; the train-only bank, raw top-64 neighbor tables, `K=8`, model, losses, seed, optimizer, validation modes, and feasibility rules are unchanged |
| Frozen/trainable | Phase A only: all inherited v2 parameters remain frozen and only the 82 `hypernetwork.sentence_memory_*` tensors train at learning rate `1e-4`; Phase B is disabled |
| Epochs | 2; full `text_only`, `sentence_memory`, and `shuffled_sentence_memory` validation and checkpointing every epoch |
| Allocation | Exact ordered nodes `ADUAED21025WKLX11,ADUAED21027WKLX14,ADUAED21028WKLX16,ADUAED21047WKLX01`; batch host/rank 0 is node 11; 1 GPU, 16 CPUs, and 100 GiB per node; four-hour limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally; physical memory microbatch cap 16 |
| Memory staging | Complete bank and train/validation neighbor tables are copied into job-specific `/tmp` storage and strictly hash-audited once per allocated node before training |
| Artifact identities | Bank ID `a65661333c0f60aa65dc68d896f439a698e37832f04bb8834a378a2d5f068bcd`; train-neighbor SHA256 `d61c0271e190c41d20e822dd4d4f6a2690daa5bfbcd62ef45b58a767377d3852`; validation-neighbor SHA256 `b5f5d0914e55ba9e9c79963bacb955b97d62f99f16f117381187e72c345eb199` |
| W&B | Disabled (`WANDB=0`); this launch has no W&B run name, ID, URL, resume policy, or online credential dependency |
| Comparison contract | Compare both epochs against the frozen text-only baseline, shuffled retrieval, and the duration-weight 0.10 Phase-A result; do not advance to Phase B unless correct retrieval passes the existing validation and hand-path feasibility gates |
| Final outcome | Epoch 2 correct-memory score `12.716377` versus text-only `12.838353` and shuffled-memory `12.716469`; the required 0.1% correct-over-shuffled margin failed. `best_infeasible.pt` selects epoch 2, no feasible `best.pt` was created, and the pilot did not advance to Phase B |

Historical inspection commands:

```bash
squeue -j 142893 -o '%.18i %.28j %.10T %.12M %.12l %.4D %R'
scontrol show job 142893
tail -F logs/sbatch/csl_rag_dw005_p2_142893.out
tail -F logs/sbatch/csl_rag_dw005_p2_142893.err
```

Epoch metrics will be written to:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_dw005_pilot2/metrics.jsonl
```

The pilot is complete. Do not promote its `best_infeasible.pt` or resume it as
the Phase-A' motion-contrast experiment.

## Previous: CSL-Daily SignTrajField-RAG v3 Phase A (duration weight 0.10)

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-rag-v3-phase-a-20260906` |
| Dataset | Full CSL-Daily train/validation splits (18,399/1,077 samples) |
| Training job | Slurm `142746`, name `csl_rag_phasea` (`COMPLETED`; all 10 epochs and global step 720 were durably written, and W&B synced cleanly) |
| Configuration | `NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a.yaml`; SHA256 `9cc4c8b7ed5a740b0788374ae604797fe692a953f8a6c7a1230868c85c8b71b1` |
| Output | `experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a` |
| Source | Branch `codex/csl-daily-signtrajfield-rag-v3`, commit `39328f12e2b604a4335fd1f6ec1a81790331ff93`, pushed to `origin` |
| Initialization | Strict v2-to-v3 base import from `csl_daily_signtrajfield_v2_mt5_text_only_full/checkpoints/best.pt`; SHA256 `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54` |
| Conditioning | Sentence text plus sentence-motion memory (`K=8`, top-M 64); word prior completely off |
| Frozen/trainable | All inherited v2 parameters frozen; only `hypernetwork.sentence_memory_*` trainable at learning rate `1e-4` |
| Epochs | 10; validation and checkpointing every epoch in text-only, sentence-memory, and shuffled-memory modes |
| Allocation | Exact ordered nodes `ADUAED21042WKLX08,ADUAED21045WKLX28,ADUAED21046WKLX02,ADUAED21047WKLX01`; batch host/rank 0 is node 08; 1 GPU, 16 CPUs, and 100 GiB per node; DDP/NCCL 2.27.5; 72-hour limit |
| Batch semantics | 32 samples/GPU/loader batch, 2 accumulation steps, effective 64/GPU and 256 globally; physical memory microbatch cap 16 |
| Memory staging | Complete bank, including neighbor tables, copied to job-specific `/tmp` and strictly audited once per node before training |
| W&B | Online entity `hh3443-new-york-university`, project `soke-niaf-continuous-trajectory`; run `csl_daily_signtrajfield_v3_sentence_memory_phase_a_142746`, ID `ax3fimep`, URL `https://wandb.ai/hh3443-new-york-university/soke-niaf-continuous-trajectory/runs/ax3fimep`; distinct `train/*` and `validation/*` namespaces were synced successfully |
| Auth gate | Read-only node-08 preflight Slurm `142744` passed for viewer `hh3443`; launcher rejects inline/shared credentials and fails closed if the rank-0 node-local check changes |
| Validation | Full repository suite passed in Slurm `142742`: 182 tests; Ruff, Python compilation, shell syntax, and Git whitespace checks passed |
| Deliberate fast path | The user explicitly chose to skip separate retrieval diagnostics, smoke training, and K-ablation pilots before this full run |
| Initialization checks | Imported v2 epoch 10 with only 82 new `hypernetwork.sentence_memory_*` tensors missing; those same 82 tensors are the only trainable parameters; memory-off prediction and duration parity both have maximum absolute error `0.0` (tolerance `1e-7`) |
| Epoch 1 training | Completed 144/144 logical batches and 72 optimizer steps in 38:00; batch-mean loss `11.777318`, residual RMS `0.113245`; no non-finite loss, OOM, traceback, or NCCL failure |
| Epoch 1 validation | All 34 batches completed in each of `text_only`, `sentence_memory`, and `shuffled_sentence_memory`; sentence score `12.717379` versus text-only `12.838353` (0.9423% better), but shuffled score `12.717333` is effectively tied/slightly better, so the epoch is infeasible under the required 0.1% correct-over-shuffled margin |
| Epoch 1 hand/duration diagnostics | Sentence memory changed left/right path from `0.359614/0.334957` to `0.287256/0.229979` (20.1%/31.3% lower); hand-relative loss rose only 0.67%, within the 2% guard; predicted duration is exactly unchanged at `4.731181` seconds |
| Epoch 1 memory diagnostics | Part gates are approximately `0.1196`--`0.1198`; null mass `0.04116`; real-candidate mass `0.95884` |
| Final outcome | Epoch 10 completed with selection score `12.807284`, sentence-memory score `12.807284`, shuffled-memory score `12.806838`, and text-only score `12.838353`; correct retrieval still failed the required 0.1% margin over shuffled retrieval |
| Checkpoints | `epoch0001.pt` through `epoch0010.pt` and resumable `last.pt` exist; `best_infeasible.pt` selects epoch 2 at score `12.715823`; no feasible `best.pt` was created, and `best_infeasible.pt` must not be promoted |
| Checkpoint identities | Bank and all neighbor identities are embedded; sentence behavior digest `880d9ca2a61c80d56a53a0609dec39d58eee17f50e1b41e171d9f24ba87609fd`; exact-resume digest `da581b2d5db2d3fd262df6bb889e4f20922f8dafab2abc90769ff832282a98e8`; optimizer, per-rank RNG, and historical selection state are present |

Historical inspection commands:

```bash
tail logs/sbatch/csl_rag_phasea_142746.out
tail logs/sbatch/csl_rag_phasea_142746.err
```

Epoch metrics are stored in:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/metrics.jsonl
```

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

## Previous: CSL-Daily SignTrajField-v2 full training

| Field | Value |
|---|---|
| Alias | `csl-daily-signtrajfield-v2-full-20260807` |
| Dataset | Full CSL-Daily train/validation splits (18,399/1,077 samples) |
| Training job | Slurm `141306`, name `csl_stfv2_full` (no longer active; partial output through epoch 26/global step 1,872) |
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

The job is absent from the live scheduler. Its durable output reaches epoch 26,
but the final launch log ends during a continuation attempt because
`WANDB_RESUME=never` was used with an existing W&B run ID, followed by DDP
teardown. Treat these checkpoints as a partial historical run, not an active or
completed baseline.

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
