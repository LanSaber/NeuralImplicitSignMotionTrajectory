# CSL-Daily Phase-A''' centered/relevance runbook

Status: implementation prepared only. No calibration, test, smoke, training,
decision, diagnostic, confirmation, or test-set job is submitted by this
change.

## Immutable protocol

Stage A is
`csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1`.
It uses candidate-centered motion covariance values, a frozen absolute
adjusted-score relevance gate, no association objective, and no temporal
prior. Stage B is
`csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_absolute_binding_motion_contrast_v1`.
It adds only the `absolute_text_motion_v1` association mechanism and its fixed
`0.10` BCE, `0.05` InfoNCE, and `0.10` temperature settings. Stage B is never
warm-started: it starts afresh from v2 SHA256
`06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`
only after a verified Stage-A `valid_infeasible` authorization.

Both full stages use six epochs maximum, three epochs minimum, patience two,
four nodes, one GPU and 16 CPUs per node, 100 GiB per node, and six hours.
W&B is disabled. Only train/validation bank material is staged locally.
Stage A has the ordered ten-mode development control set ending in
`analytic_prior`; Stage B appends `association_disabled` as mode 11.

The pre-existing sealed split remains authoritative: development is 256
normalized-text clusters/347 signer rows; confirmation is 540 clusters/728
rows. Confirmation remains unopened until the first development-feasible
checkpoint is locked. All protocols share the one durable spend marker:

```text
experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_factorized_ordered_v1_control/confirmation_holdout_spent.json
```

The new control directory is not an independent holdout lock. Any existing
global marker prohibits Stage A, Stage B, and adaptive confirmation. No test
neighbor table or test manifest is staged, discovered, hashed, or opened.

## Train-only calibration

Calibration is a mandatory sealed prerequisite. For every train query it
materializes the exact deterministic provider realization for each stored
top-eight positive group and eight deterministic full-replacement train-bank
items. Alternatives exclude the query item, its semantic/source/sign group,
all positive groups, and duplicate selected groups. Exact concrete item IDs,
durations, cosine scores, duration log-gaps, SHA query partition, and the
versioned replacement map are stored in `calibration_map.npz` and bound by a
content digest.

The model is an unregularized class-balanced scalar logistic regression fit in
float64 on the SHA-ranked 90% query partition. The 10% train-only holdout must
reach AUROC at least `0.75` and mean positive-minus-negative calibrated
probability at least `0.20`. Publication flushes files/directories and uses an
atomic no-replace directory rename. An accepted artifact has `READY`; a failed
gate has only immutable `REJECTED` and is terminal. Existing accepted output
can only be revalidated, never republished. Every training, evaluator,
exporter, decision, diagnostic, and confirmation construction reopens the
sealed artifact and binds its artifact identity, map SHA/content digest, and
exact slope/intercept; checkpoint-copied scalars alone are not authority.

## Source and staging invariants

All jobs require `PROJECT_DIR` to be a clean, pushed standalone clone on
shared `/media/cvpr` storage with a real `.git` directory. The branch and
origin head stay fixed for calibration, tests, smoke, all possible resumes,
both ordered stages, decision, diagnostic, and confirmation. The clone must
expose durable `experiments` and `deps/mt5-base`. Jobs reject a Codex worktree
whose gitdir points to node-local storage.

The strict staging helper reads only explicit core filenames from `bank.json`
and explicit requested `neighbors_train.npz`/`neighbors_val.npz`. It never
globs the shared bank. `READY` is copied last, the local bank is audited, and
every launcher proves `neighbors_test.npz` is absent before importing a model
or provider. This rule prevents recurrence of the earlier incident in which a
development-only decision process hashed a shared test-neighbor table before
local isolation.

Execution uses atomic attested leases. A new Slurm attempt can replace an old
claim only after `scontrol` proves its owner terminal or absent; displaced and
released claims remain in history. Failed/requeued jobs retain their claims.
Do not delete a lease to force progress. A pre-checkpoint training failure is
archived and restarted fresh; an exact resume requires `last.pt`, unchanged
source/config/calibration identities, optimizer and four-rank RNG state, and
the persisted early-stopping/selection state.

## Launch order

Do not launch until the implementation is committed and pushed and the shared
run clone is clean.

```bash
export PROJECT_DIR=/media/cvpr/haomian/SignTrajField_centered_run_source
export SOURCE_REMOTE_BRANCH=codex/csl-daily-centered-memory-v1-run
export SOURCE_GIT_HEAD="$(git -C "$PROJECT_DIR" rev-parse HEAD)"

sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/test_csl_daily_centered_memory_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/calibrate_csl_daily_sentence_memory_relevance_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/smoke_csl_daily_centered_memory_v1_sbatch.sh"

sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/train_csl_daily_centered_relevance_stage_a_v1_sbatch.sh"
```

For an interrupted full run with a valid `last.pt`, resubmit the same stage
wrapper with `CENTERED_RESUME=1`; all launch identities and leases are checked
before trainer re-entry.

After training, submit the centered ordered-decision job for that stage. Read
its `READY` and `decision.json`; scheduler `COMPLETED`, `FAILED`, or exit 143
alone is not a scientific decision. The decision independently replays the
selected checkpoint on all sealed development rows and binds globally reduced
all-null, uniform-final-mass, broadcast-complete-motion-package, joint-tuple,
and selected-off/v2 integrity evidence. It authorizes Stage B only for an
integrity-valid Stage-A scientific failure. For the first
development-feasible stage it mints only `authorize_diagnostic.json`. The
locked diagnostic must publish a complete validated `READY`; a separate
fail-closed finalizer then mints `authorize_confirmation.json`. Neither step
can ever authorize `best_infeasible.pt`.

```bash
# Only with authorize_stage2.json from Stage A:
sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/train_csl_daily_centered_absolute_binding_stage_b_v1_sbatch.sh"
```

The first feasible stage runs one locked development diagnostic, then one
confirmation workflow. The confirmation launcher atomically spends the global
marker before it opens `manifest_confirmation.jsonl`; exact continuation is
allowed only for the same authorization/spend identity. A confirmation
failure spends the holdout and prohibits Stage B, threshold changes, Phase B,
or test access.

## Registry follow-up template

After the ordered protocol is terminal, use a separate documentation-only
commit (never advance the immutable run branch) to record:

| Field | Value |
|---|---|
| Source branch/commit/live origin proof | `TBD` |
| Calibration job/artifact/map identity/held-out gates | `TBD` |
| Static + complete CPU suite job/result | `TBD` |
| Sequential GPU smoke job/result | `TBD` |
| Stage-A full job/launch/lease identities | `TBD` |
| Stage-A selected checkpoint/decision | `TBD` |
| Stage-B authorization/job, if permitted | `TBD` / not authorized |
| First development-feasible stage | `TBD` / none |
| Locked diagnostic job/artifact | `TBD` / not authorized |
| Confirmation job/spend/bootstrap/promotion | `TBD` / not authorized |
| Test access | none |
| W&B | disabled |

Leave the unqualified default run alias unchanged unless a later explicit
promotion decision authorizes it.
