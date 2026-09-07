# CSL-Daily Phase-A'' factorized-memory runbook

Status: implementation prepared; no smoke, training, decision, diagnostic, or
confirmation job has been submitted by this change.

## Ordered experiment

| Stage | Experiment | Temporal prior | Authorized next action |
|---|---|---|---|
| 1 | `csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1` | none; uniform positive-validity token support | feasible -> confirmation; integrity-valid infeasible -> Stage 2; integrity invalid -> repair/exact continuation |
| 2 | `csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1` | fixed Gaussian, sigma `0.25` | feasible -> confirmation; infeasible -> stop |

Both stages start strictly from v2 checkpoint SHA256
`06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`.
Stage 2 never imports Stage-1 weights or optimizer state. Every launcher
requires an explicit `PROJECT_DIR` pointing to a standalone clone on shared
`/media/cvpr` storage; a Codex worktree whose `.git` file points into
node-local `/home` is rejected. The clone must expose the durable shared
`experiments` tree and frozen `deps/mt5-base` (ignored symlinks are allowed).
The launch helper requires a clean worktree and records a live, exact
`origin/<branch>` head,
config and launcher hashes, Slurm job ID, v2 identity, W&B-disabled state, and
the train/validation-only staging scope before model or training-data setup.

The decision helper emits exactly one of:

- `integrity_invalid`: no authorization;
- `development_feasible`: `authorize_confirmation.json`;
- `valid_infeasible`: only Stage 1 emits `authorize_stage2.json`.

Stage-2 launch evidence embeds the Stage-1 authorization path, SHA256,
authorization identity, and decision identity. Its terminal decision verifies
and embeds that chain again. `best_infeasible.pt` is never accepted for
confirmation.

The selected checkpoint is audited on all 347 development rows with both
memory-off and all-null controls and a separate export of the pinned original
v2 checkpoint. Authorization requires exact all-null equality and selected
memory-off versus v2 prediction and predicted-duration maximum absolute error
at most `1e-7`.

## Validation isolation

The existing partition digest is
`80f9f5e9fe8414d66730ff0f19bd95fa9f8156922b7cd6f14f27b182cae7d74e`:
256 novel development text clusters (347 signer rows) and 540 confirmation
clusters (728 signer rows). Training and development jobs stage only `train`
and `val`; no launcher in this protocol stages or opens test data.
They consume the already sealed Phase-A' partition at
`csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.prerequisites/validation_text_partition`.
Before checkpoint lock, the process opens only its 347-row
`manifest_development.jsonl` (SHA256
`9a0fc32fad9bfc8168af31086a0bd8ac08cc7821e987578d51d3bd10bf70651f`)
and the small READY envelope (artifact identity
`2d915a1b5c709a799eb2973d20b5b2ce6d9b0ae2d11b2871bfb221fe1a516a0a`).
It never rebuilds the split or reads `partition.json`, the exact-seen manifest,
or the confirmation manifest. Resolved configs/checkpoints contain only the
pinned digests and numeric counts, not sentence membership or assignments.

Epoch validation uses `[off, on, motion_shuffled, shuffled, analytic_prior]`
with fixed corruption identity
`fixed_query_condition_v1 / 1234 / csl_daily_validation_corruption_v1` and
equal normalized-text-cluster aggregation. `analytic_prior` is emitted as
`analytic_prior_sentence_memory` and is explanatory only.

Before any confirmation manifest is opened, a global exclusive spend marker is
bound to the first feasible authorization. A preempted confirmation may be
continued only with that identical authorization and marker. Each completed
mode is validated and atomically finalized; only provenance-complete units are
reused. A changed or partial final artifact fails closed. The final analysis is
also atomic and idempotent for identical inputs. Scientific failure still
spends the holdout and prohibits adaptive Stage 2, threshold changes, Phase B,
or test access.

Training/resume, ordered-decision, locked-diagnostic, and confirmation jobs
use atomic, non-empty execution-lease directories. A second live Slurm owner
is rejected. A stale owner is replaced only after `scontrol` proves it terminal
or no longer registered, and every displaced claim is retained in immutable
history. Leases are released only by the matching attestation after a normal
exit. Confirmation mode directories are published with no-replace rename
semantics, so a continuation can never nest or clobber a completed unit.

## Launch sequence

Do not submit until the implementation commit is clean and pushed and the CPU
suite plus one-GPU smoke have passed through Slurm.

Create a dedicated shared standalone clone at the immutable run branch/commit,
then expose the ignored durable dependencies. `PROJECT_DIR` is mandatory; the
launchers intentionally have no fallback to the development worktree.

```bash
export PROJECT_DIR=/media/cvpr/haomian/SignTrajField_factorized_run_source
export SOURCE_REMOTE_BRANCH=codex/csl-daily-factorized-memory-v1-run
export SOURCE_GIT_HEAD="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
test -d "$PROJECT_DIR/.git"
test -d "$PROJECT_DIR/experiments"
test -d "$PROJECT_DIR/deps/mt5-base"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$PROJECT_DIR" ls-remote --heads origin "refs/heads/$SOURCE_REMOTE_BRANCH" | awk 'NR==1 {print $1}')" = "$SOURCE_GIT_HEAD"
```

Keep that remote run branch fixed at `SOURCE_GIT_HEAD` until every possible
resume, ordered decision, Stage 2 run, diagnostic, and confirmation is
terminal. Advancing it invalidates the live source gate. Do not put the later
registry/documentation commit on this immutable run branch.

```bash
sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/smoke_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_motion_contrast_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/train_csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,FACTORIZED_STAGE=stage1,SOURCE_GIT_HEAD \
  "$PROJECT_DIR/scripts/NIAF/decide_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
```

Read `ordered_development_decision/READY` and `decision.json`; never infer the
next action from scheduler status alone. Submit Stage 2 only when the verified
Stage-1 artifact is `authorize_stage2.json`. Run the locked development-only
temporal-slot diagnostic before confirmation for the first feasible stage.

For a feasible Stage 1, the mandatory diagnostic invocation is below. The one
job runs its deterministic eight-example smoke first and then the full
development-only audit; `CONFIRMATION_AUTHORIZATION` must be the newly emitted
Stage-1 authorization.

```bash
export CONFIRMATION_AUTHORIZATION="$PROJECT_DIR/experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_motion_contrast_v1/evaluation/ordered_development_decision/authorize_confirmation.json"
sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,DIAGNOSTIC_PROFILE=factorized_stage1_v1,CONFIRMATION_AUTHORIZATION \
  "$PROJECT_DIR/scripts/NIAF/diagnose_temporal_slots_sbatch.sh"
```

If Stage 2 is instead the first feasible stage, use
`DIAGNOSTIC_PROFILE=factorized_stage2_v1` and its Stage-2
`authorize_confirmation.json`. Do not run either factorized profile before its
own feasible authorization exists.

```bash
sbatch --export=ALL,PROJECT_DIR,FACTORIZED_STAGE=stage2,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/train_csl_daily_signtrajfield_v3_sentence_memory_phase_a_split_kv_temporal_bias_motion_contrast_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,FACTORIZED_STAGE=stage2,SOURCE_GIT_HEAD \
  "$PROJECT_DIR/scripts/NIAF/decide_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,FACTORIZED_STAGE=stage1,SOURCE_GIT_HEAD \
  "$PROJECT_DIR/scripts/NIAF/confirm_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
```

Use `FACTORIZED_STAGE=stage2` in the final command only when Stage 2 is the
first feasible arm. Exact training continuation uses the dedicated resume
launcher and the same source clone/commit and remote branch:

```bash
sbatch --export=ALL,PROJECT_DIR,FACTORIZED_STAGE=stage1,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/resume_csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_v1_sbatch.sh"
```

The resume preflight accepts a completed epoch or the trainer's
post-training/pending-validation checkpoint, repairs only an exactly
checkpoint-prefixed truncated/missing final metrics row, and performs an
attested no-op terminal-summary finalization without re-entering training.
Confirmation continuation uses the same command, authorization, output root,
source commit, and spend-marker location.

## Registry follow-up template

Only after the complete ordered protocol is terminal, make a
documentation-only follow-up commit on a different branch recording:

| Field | Value |
|---|---|
| Source branch/commit/live remote proof | `TBD` |
| CPU suite job/result | `TBD` |
| GPU smoke job/result | `TBD` |
| Stage-1 full job/launch identity | `TBD` |
| Stage-1 selected checkpoint/decision | `TBD` |
| Stage-2 authorization and full job, if any | `TBD` / not authorized |
| First feasible stage | `TBD` / none |
| Locked development diagnostic job/artifact | `TBD` / not authorized |
| Confirmation job(s), spend identity, artifact | `TBD` / not authorized |
| Promotion decision | `TBD` / not evaluated |
| Test access | none |
| W&B | disabled |

Do not change `DEFAULT_RUN_ALIAS` merely because these jobs are prepared or
launched.
