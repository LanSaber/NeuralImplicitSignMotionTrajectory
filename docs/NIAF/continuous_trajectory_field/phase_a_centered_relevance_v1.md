# CSL-Daily Phase-A''' centered/relevance runbook

Status: **ordered protocol terminal; stopped after an integrity-valid but
scientifically infeasible Stage B.** The third recovery source
`191b5c28515bff446926185ac965bbe6e8701e93` completed the full CPU gate,
train-only calibration, sequential Stage-A/Stage-B smoke, both fresh full
training stages, and both ordered decisions. Stage A authorized only Stage B;
Stage B authorized nothing. No development-feasible checkpoint, `best.pt`,
diagnostic authorization, confirmation authorization, promotion, Phase B, or
test evaluation exists. The global confirmation holdout remains unopened and
unspent. The first immutable source commit
`578cbb550a9e17ead6bb2111c717046f660f712c`, accepted calibration job `143299`,
and failed smoke job `143300` remain invalid-attempt evidence. The first
recovery commit `5a8845057d3e3e0ebb1daf6b3fe91d981ce44820` passed the complete
CPU gate (`143301`), train-only calibration (`143302`), and both retry1 smoke
arms (`143303`). Its full Stage-A job `143304` completed five finite
development validations and was scientifically infeasible, but canceled
decision job `143319` showed that the decision helper replayed those rows
without the trainer-injected v2 parity proof. It published no decision or
authorization. Because decision source must equal training source, none of
the 5a884505 artifacts may authorize Stage B even though the trained weights
were internally coherent. The run, retained lease, and launch records were
moved without deletion or overwrite to
`experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1.invalid_attempts/source_5a884505_job143304_decision143319/`;
its `ARCHIVE.json` pins the original paths and evidence hashes.

The second recovery commit
`cacd362ed90c080e0965d0e457a64c6b9d94f5ca` passed the complete CPU gate
(`143332`), source-bound train-only retry2 calibration (`143339`), and both
retry2 smoke arms (`143344`). Its fresh Stage-A job `143353` reproduced five
finite development validations and was scientifically infeasible. The fixed
selection replay then passed exactly in ordered decision job `143367`, and its
complete selected-checkpoint audit passed every parity and exact-invariant
gate. However, the exporter placed the correct relevance-calibration identity
only under `sentence_memory_checkpoint_identities.relevance_calibration`,
while the decision reader required the same proof at top level. The decision
therefore failed closed with `Centered export calibration identity changed`
and published only nonterminal integrity-invalid evidence `82933329...`; it
created no canonical decision or authorization. The complete r2 run,
development-only exports, and both retained failed-job leases were moved
without deletion or overwrite to
`experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_centered_relevance_motion_contrast_v1.invalid_attempts/source_cacd362_job143353_decision143367/`.

The third recovery commit
`191b5c28515bff446926185ac965bbe6e8701e93` repeated the complete CPU gate,
published the source-bound train-only `_retry3` calibration, ran both
`_smoke_retry3` arms, and trained Stage A afresh. Stage-A decision `143395`
was `valid_infeasible` and emitted the sole Stage-B authorization. Stage B
then started fresh from v2, completed all six epochs, and decision `143427`
was also `valid_infeasible`; its legacy schema omits top-level
`authorized_purpose` and records no confirmation/test access. This is the
predeclared terminal stop. No diagnostic, confirmation, test-set job, or
confirmation-spend operation was authorized by any attempt.

## Terminal r3 execution ledger

| Step | Terminal evidence |
|---|---|
| Immutable source | Standalone clone `/media/cvpr/haomian/SignTrajField_centered_run_source_r3`; branch `origin/codex/csl-daily-centered-memory-v1-run-r3`; local and live-origin head `191b5c28515bff446926185ac965bbe6e8701e93`; clean before and after every eligible job |
| Export-schema repair gate | Slurm `143374` completed: 22 focused centered-ops tests passed, including real exporter-summary loading, missing/tampered dual calibration-proof rejection, and v2 exemption |
| Complete CPU gate | Slurm `143375` completed: repository compileall and pinned Ruff passed; explicit all-30-file `tests/test_*.py` suite reported 387 passed and 13 warnings |
| Train-only calibration | Slurm `143376` completed. Artifact identity `668694ff012a5b71bedeab42a8139b7a95014b9b790ec8d44514cdd50cac933d`; calibration/map/READY SHA256 `aa98927325147956d68caa0a66979d8d64a746b8812899d15e90cf4f09755775` / `948a7c2f7a8ec8fdbcfa8a9c130e665adfdf7615b8908cc31324b77c15e61411` / `e9da4d8240092c54121a733fd466aa6f0822802fad639b1eed10d50e20e5c394`; held-out AUROC `0.9817280587`, probability gap `0.7923572567` |
| Sequential GPU smoke | Slurm `143377` completed. Stage A and Stage B each ran exactly one optimizer step from a separate fresh v2 initialization. Frozen-base, 10/11 modes, association gradients, finite state, exact parity/invariants, and train/validation-only staging all passed |
| Stage-A training | Slurm `143380`, four nodes/GPUs, finished five validations/global step 360 and intentionally returned `FAILED/143` after patience two because no feasible checkpoint existed. Launch identity `0b1b9a8c089f88d9128027016cd5bb1ea7b39d3a67d5b5ab76c7dda2a078d97e`; retained failed-training lease claim `e588e1801b42148074b90970b42e60ea2edbb5a9fefee97c39208f498595cbb8` |
| Stage-A selection | Epoch 3/global step 216 `best_infeasible.pt`, SHA256 `39a66ba0422ad5be1ccf2745ea7464d3c43894f3cc7c53c0a83e0dffcd979cb6`; metrics `2aeaf86de44cac053f9c5bf2ac37364820b675cb266ba80809339d9733c5d210`; selection summary `3c518eb472ff857f04dd2e151731c235bbb85cd81004138774a9eee36612f679` |
| Stage-A decision | Slurm `143395` completed. Canonical `valid_infeasible`; decision identity `12c4fd20233f218422b36d977829cca5957c7c35c4a4e6be4aceec178642e837`; decision/READY SHA256 `73bcec6eab500588875edaf0f74661d1b2aa8548b79e5d3928403f78f29f3104` / `cd0fe2a9c4e7db59b9fbabb9c331ae7c226f0157de5e78014c6995a6543af020` |
| Stage-B authorization | Authorization identity `ecca886ee2252a0db74f8ca8c404100cb1ea2fa8690f5e1e36bb68cc5e654d15`; file SHA256 `81978bc51af76741f9330010036e4c55e93bc219c6d8a0bfeb7f4397901c4b3a`; fresh-v2 Stage-B input identity `7a196496561c32df8ba1a71257775c31b20c75cad2944f200f7baee22facab57` |
| Stage-B training | Slurm `143400`, four nodes/GPUs, completed six validations/global step 432 and intentionally returned `FAILED/143` because no feasible checkpoint existed. Launch identity `59a0015755ed927f506b0e6785b779092f56578317365cac7212c99b76d8d029`; retained failed-training lease claim `3868b9018b8ddfb7f08ecf85dd0ac3e4cebba138ef4f29b0eb165a73aec66a7d` |
| Stage-B selection | Epoch 5/global step 360 `best_infeasible.pt`, SHA256 `b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`; metrics `9171e1e65ceb57c6b291cf24825eb934bcfd4983f7d822fe01e56af33e4c9b10`; selection summary `41b5bcb5eeece4132f1c576399232010ac00e6c0fec168327d2c9d03095cd561` |
| Stage-B decision | Slurm `143427` completed. Canonical `valid_infeasible`, integrity valid, development infeasible; exact legacy schema omits top-level `authorized_purpose` and records no confirmation/test access. Decision identity `7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`; decision/READY SHA256 `8993f4d7ae61d2ecd2bc41d523c45ab55073c63a061ce24629a564627724f8c5` / `fdc8e0823ccd2b3f769b5fdf31cb21098cf719a542cd9556546ee91fa174cfde` |
| Isolation and spend | Every eligible job staged only train/validation tables; decisions record `test_data_accessed=false` and `confirmation_manifest_opened=false`; W&B was disabled. The global confirmation-spend marker is absent |

The terminal r3 job-log SHA256 values are:

```text
143374 out 61f469cf282cb0fdf5f3d858c2b4364405708196c90ac80fabfec3e1880c6eb6
143374 err 4e66e2121c0fc5d9aec5a1514ce44b5ad7d94554f47b9995d739088baad9fdaf
143375 out 2d48bc99cbedf4428e0a29c1d6123aeaf8974d9c2573f585eb3ad338314d8ea4
143375 err 28785a69aab6d46bb9f3859e31124cd0af3a35951d217f3a981c03b0e6a82543
143376 out c9d17e5191e1433292cb493385fd701d435ae872ca86b112b4e5a22ff90703c1
143376 err e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
143377 out e91574a253a2d06557158d377f07c07df9e66bba7f7e332d8b8ebde4e0579bec
143377 err d8e2fe717f59ae10f65fd3f90e86d4e25e1055bb592578979224870528690419
143380 out 685c90413cefa042cd58706908c38678ed6b68507d34497813e4533a2c91185b
143380 err f0307f1e3b2e79cbff22a45c79dc2f3718eef2373bac10535342e30e779774a5
143395 out 62170c576d13ea5c235b5b8953caac759e3ecbbc3fbd2bbd8c57e707ccfa6237
143395 err 5bef5ae484932e065c17aa8d7e28f0c55cca2c11d707fa10a16310cc8ad1955a
143400 out a8bddf2f8352a522e6f4952e617100e00271b366de64a9ad87b3b70c23930af8
143400 err 4317d3d345d8010ee60fbb162ecabd4282055f2431d87be5bc0ccfdb68ecf4b7
143427 out c804d9702388e9a317f0e6591d0634f4bc694097aefcbd195cc8ce4bbbd5e09f
143427 err d9ac4f5be613c5e1e5b28bca33af8fc956688a856527851627d36e1d00224e52
```

Forensic checksum anchor for the first recovery (recorded before the r2
commit): the archive contains 16 files and the command
`find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum`, executed
at its root, returns
`858dd01738858431b1d4c11c0f706a788509c6e215719be31024155962848abe`.
The preserved job-log SHA256 values are:

```text
143301 out fc16d5da49e566f1b46f3e927c3f9148a7397e5f3bdbf71a3eb462b0c7697319
143301 err 12caa88629b079c6c7de75c8ff743946220dbb4045edb245f8bcac1c6f188c9f
143302 out 6c0190543b539fe8fd26ebd182d9dd628cd0b7fb65affbaa89d0b00afe30664a
143302 err e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
143303 out cfa214cbe631cf5264a15da790ef87a4b638736c2ac05897b8c83cbb339a5215
143303 err d8f47ad8c88519af0f4e1a8dd4a81120ac2fc08c842c5e692aad98e73598fb87
143304 out 69f85d440fcd163118cae56eace2c669fcf5a51af04a61bd6c1fbec4014e7434
143304 err ffa12ad16fac7b0e99728fe0e53564d3fa561646e04b9f959d06c8885c255e16
143319 out e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
143319 err 672a8effc696e04c6732138b6c0dc8e09b24fb9e3a58109c364424ac3ea9b6c5
```

Forensic checksum anchor for the second recovery (recorded before the r3
commit): its archive contains 2,113 files totaling 1,407,774,082 bytes. The
same full-tree checksum command, executed at that archive root, returns
`256c414c588f797e7207e6aacdf3aa394322ef445b37927c9a03d7525b02ee44`.
Its `ARCHIVE.json` pins the selected/last checkpoints, training and decision
attempts and leases, complete selected-checkpoint audit, all three dev export
summaries, retry2 calibration/smoke evidence, and the integrity-invalid
decision. The preserved r2 job-log SHA256 values are:

```text
143332 out f19db4aab7cc1368e87273010df84c047e2ab3585a27ffaa40e5dc32b7b6257b
143332 err b062a1de51e755800ae81453b9349e89dbd9e6fb226b193bfcd7f209d112290c
143339 out 762c1920d70230d73d7190a4be33e5c72ac1f3805fe8db2cd3859cdbf5589a57
143339 err e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
143344 out ed652f1bd007f71bfa29b2103779fd8a81fa30cb567c041db7106cd31f9ad4cf
143344 err e61b6a1efbf4144df309ee77fd4bc4ed51e0a02fa2cf37bc8ecd29ddcb14df35
143353 out ddc457fde38e5c5ed3de22b5cd4ce27f49b0b3e683a0a3a14e0076127ca8cf0b
143353 err 328e5c312da3374b9fa48c726a535a560e3dac2def6953b6729e47838bd6ee80
143367 out 5f59f3c374adc83ab6108469755a3e5f424d3e6dad8fee97346a16a625373ef3
143367 err d83ad2811a58b8350a9a8d6f7fb684a38c4e834a168604977b344fd373f1e321
```

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
rows. Confirmation remained unopened because neither stage was
development-feasible and no checkpoint was locked. All protocols share the
one durable spend marker:

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
shared `/media/cvpr` storage with a real `.git` directory. The retry branch and
origin head stay fixed for calibration, tests, smoke, all possible resumes,
both ordered stages, decision, diagnostic, and confirmation. The clone must
expose durable `experiments` and `deps/mt5-base`. Jobs reject a Codex worktree
whose gitdir points to node-local storage.

The retired immutable branch
`codex/csl-daily-centered-memory-v1-run` remains fixed at `578cbb5...` for
jobs `143298`--`143300`. The first recovery branch
`codex/csl-daily-centered-memory-v1-run-r1` remains fixed at `5a884505...`
for jobs `143301`--`143304` and canceled job `143319`. The second recovery
branch `codex/csl-daily-centered-memory-v1-run-r2` remains fixed at
`cacd362e...` for jobs `143332`, `143339`, `143344`, `143353`, and `143367`.
None of these branches may be advanced or reused. The third recovery used the
distinct clone and immutable branch recorded below; that branch is now equally
frozen and must not be advanced or reused.

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

## Launch order used (historical; do not relaunch)

The commands below record the exact order used for r3. They are retained as
protocol provenance, not as authorization to rerun it. The ordered experiment
is terminal, so do not submit any of these commands again. Before execution,
the implementation was committed and pushed and the shared run clone was
clean.

```bash
export PROJECT_DIR=/media/cvpr/haomian/SignTrajField_centered_run_source_r3
export SOURCE_REMOTE_BRANCH=codex/csl-daily-centered-memory-v1-run-r3
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$PROJECT_DIR"
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

Before terminalization, an interrupted full run with a valid `last.pt` could
be resubmitted through the same stage wrapper with `CENTERED_RESUME=1`; all
launch identities and leases would be checked before trainer re-entry. This
recovery path is no longer authorized for r3.

After each r3 training stage, the centered ordered-decision job was submitted
and its `READY` and `decision.json` were read; scheduler `COMPLETED`, `FAILED`,
or exit 143 alone was not treated as a scientific decision. The decision
independently replays the
selected checkpoint on all sealed development rows and binds globally reduced
all-null, uniform-final-mass, broadcast-complete-motion-package, joint-tuple,
and selected-off/v2 integrity evidence. It authorizes Stage B only for an
integrity-valid Stage-A scientific failure. For the first
development-feasible stage it mints only `authorize_diagnostic.json`. The
locked diagnostic must publish a complete validated `READY`; a separate
fail-closed finalizer then mints `authorize_confirmation.json`. Neither step
can ever authorize `best_infeasible.pt`.

```bash
# Historical Stage-B command, used only after Stage A authorized it:
sbatch --export=ALL,PROJECT_DIR,SOURCE_GIT_HEAD,SOURCE_REMOTE_BRANCH \
  "$PROJECT_DIR/scripts/NIAF/train_csl_daily_centered_absolute_binding_stage_b_v1_sbatch.sh"
```

Had a first feasible stage existed, it would have run one locked development
diagnostic, then one confirmation workflow. The confirmation launcher would
atomically spend the global
marker before opening `manifest_confirmation.jsonl`; exact continuation would
have been allowed only for the same authorization/spend identity. No such
authorization was created. Under the frozen protocol, a confirmation failure
would spend the holdout and prohibit Stage B, threshold changes, Phase B, or
test access.

## Registry follow-up record

The ordered protocol is terminal. This table is the documentation-only
registry record; the immutable run-r3 branch remains fixed at the executable
source commit.

| Field | Value |
|---|---|
| Source branch/commit/live origin proof | `origin/codex/csl-daily-centered-memory-v1-run-r3` at `191b5c28515bff446926185ac965bbe6e8701e93`; clean standalone clone and exact live-origin equality verified throughout |
| Calibration job/artifact/map identity/held-out gates | `143376`; artifact `668694ff012a5b71bedeab42a8139b7a95014b9b790ec8d44514cdd50cac933d`; map content digest `45bd87e6f0657772e97f66efb8415d009b74f883562762842550d8b6349d7b54`; AUROC `0.9817280587`; probability gap `0.7923572567` |
| Static + complete CPU suite job/result | `143375`; compileall/Ruff green; 387 passed, 13 warnings across all 30 `tests/test_*.py` files |
| Sequential GPU smoke job/result | `143377`; both fresh-v2 arms completed one optimizer step and all strict checks |
| Stage-A full job/launch/lease identities | `143380`; launch `0b1b9a8c089f88d9128027016cd5bb1ea7b39d3a67d5b5ab76c7dda2a078d97e`; retained failed-training lease claim `e588e1801b42148074b90970b42e60ea2edbb5a9fefee97c39208f498595cbb8` |
| Stage-A selected checkpoint/decision | Epoch 3/step 216 `best_infeasible.pt` SHA `39a66ba0422ad5be1ccf2745ea7464d3c43894f3cc7c53c0a83e0dffcd979cb6`; decision job `143395`, `valid_infeasible`, identity `12c4fd20233f218422b36d977829cca5957c7c35c4a4e6be4aceec178642e837` |
| Stage-B authorization/job | Authorization `ecca886ee2252a0db74f8ca8c404100cb1ea2fa8690f5e1e36bb68cc5e654d15`; fresh-v2 input `7a196496561c32df8ba1a71257775c31b20c75cad2944f200f7baee22facab57`; training `143400`; decision `143427` |
| Stage-B selected checkpoint/decision | Epoch 5/step 360 `best_infeasible.pt` SHA `b37f000ccaaa4d952c3afc5faf7d5f776c18fae21c7addd753c1f7d83bb2b202`; `valid_infeasible`, decision identity `7022e30cccac9c597a864dc2884bb35d7ae54894084fe2f24717092c56f03c69`; exact legacy schema omits top-level `authorized_purpose` and records no confirmation/test access |
| First development-feasible stage | None |
| Locked diagnostic job/artifact | Not authorized or run |
| Confirmation job/spend/bootstrap/promotion | Not authorized or run; global spend marker absent; holdout unopened; no promotion |
| Test access | None; both decisions attest `test_data_accessed=false` |
| W&B | Disabled for calibration, smoke, training, and decisions |
| Default run alias | Unchanged |

The final scientific interpretation is narrow: exact centering removed the
uniform candidate-average shortcut, and Stage-B association raised
`Rpair` from `0.1215702175` to `0.4059177979` while achieving 46.6% pair
top-1 accuracy. The selected Stage-B checkpoint still improved correct over
the mean derangement by only 0.1616%, produced identity utility 0.1993, and
had a negative true-minus-best-negative association margin. These miss the
predeclared 0.25%, 0.25, and positive-margin gates, respectively. Stop after
Stage B; no threshold weakening, retuning, diagnostic, confirmation, test
access, Phase B, or `best_infeasible.pt` promotion is permitted.

Leave the unqualified default run alias unchanged unless a later explicit
promotion decision authorizes it.
