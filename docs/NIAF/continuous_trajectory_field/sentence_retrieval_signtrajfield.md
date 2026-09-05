# Sentence-Retrieval SignTrajField (SignTrajField-RAG)

This document describes the version-3 sentence-memory pipeline. It is intended
to be sufficient for an agent with no prior task context to build the memory,
verify it, train the frozen-base phase, and evaluate a checkpoint.

## What the pipeline does

SignTrajField-RAG keeps the complete text-to-trajectory model from
SignTrajField v2 and adds a sentence-level retrieval residual. A query sentence
retrieves coherent motions from the **training split only**. The retrieved
motions are encoded as frozen temporal-VAE `mu` tokens and cross-attended by the
16 text trajectory slots. Four independent streams attend for body, left hand,
right hand, and face. A learned null candidate and part gates can reject an
unhelpful memory.

Retrieved rotations are never averaged, concatenated, or copied into the
output pose. Memory changes only the latent trajectory plan. Duration remains
predicted from text alone.

The model type and checkpoint contract are:

```text
model.type: sentence_memory_continuous_trajectory_field
trajectory_contract_version: 3
```

The initial CSL-Daily experiment deliberately disables the older word prior so
that any gain can be attributed to sentence retrieval.

## Runtime environment

Run from the repository root:

```bash
export PROJECT_DIR=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
export PYTHON_ENV=/media/cvpr/haomian/python_envs/SOKE
export PYTHON_BIN="$PYTHON_ENV/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
```

The first supported configuration is:

```bash
export CFG=NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a.yaml
export BANK_DIR=/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx/meta/niaf_sentence_memory/mt5_vae_mu_train_v1
```

Do not build a bank or run training in an ordinary login shell. Request a GPU
through the live `spark` partition. W&B is not needed for bank construction and
must remain disabled unless the user explicitly requests online logging.

## 1. Build the train-only memory

The builder reads the full CSL-Daily train manifest with deterministic length
processing (`random_crop=false`), encodes VAE `mu`, pools hand-validity flags to
the latent rate, and writes one normalized mT5 key per unique normalized text.
Multiple recordings of the same text remain motion variants in that semantic
group.

Example interactive allocation:

```bash
srun --partition=spark --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --gres=gpu:1 --time=04:00:00 --pty bash
```

Inside the allocation:

```bash
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.build_sentence_motion_bank \
  --config "$CFG" \
  --split train \
  --out_dir "$BANK_DIR" \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --text_device cuda
```

The completed bank contains memory-mappable arrays, metadata, `bank.json`, a
build summary, and `READY`. The VAE checkpoint is used only here; training does
not load the VAE.

The primary payload files are `group_keys.float32.npy`,
`item_motion_mu.float16.npy`, `item_motion_offsets.int64.npy`,
`item_motion_lengths.int16.npy`, `item_hand_valid.uint8.npy`, and
`item_durations.float32.npy`. Sentence variants are indexed through
`group_item_offsets.int64.npy` and `group_item_ids.int32.npy`; item metadata is
stored line-by-line in `items.jsonl`.

Do not use `--overwrite` until the current target directory has been resolved
and checked. A build without `READY` is incomplete and must never be used.

## 2. Build deterministic neighbor tables

Precompute top-64 train-bank semantic groups for the fixed train, validation,
and test manifests:

```bash
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.build_sentence_neighbors \
  --config "$CFG" \
  --bank_dir "$BANK_DIR" \
  --splits train val test \
  --top_m 64 \
  --batch_size 128 \
  --device cuda \
  --text_device cuda
```

For training queries, the table excludes the query item, its normalized-text
group, and its source recording. Validation/test may use a same-text training
realization, but never the query recording itself. Report novel-text and
seen-text results separately.

Custom or reordered manifests cannot silently index these tables. The provider
checks the manifest and row-order identities and falls back to exact online
cosine search when they do not match.

## 3. Audit before training

Run the standalone audit after copying, rebuilding, or changing a config:

```bash
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.audit_sentence_memory \
  --config "$CFG" \
  --bank_dir "$BANK_DIR" \
  --verify_hashes \
  --splits train val test
```

The audit must confirm:

- the bank source split is exactly `train`;
- the active train manifest, mT5 artifacts, VAE checkpoint, motion statistics,
  preprocessing contract, and stored hashes match;
- offsets, group membership, dtypes, finite values, and normalized keys are
  valid;
- neighbor IDs are in range and obey the recorded exclusion policy.

Never bypass an identity failure by editing `bank.json`. Rebuild the artifact
with the active inputs instead.

### Pre-training retrieval diagnostics

Before fitting the sentence branch, measure whether text retrieval contains a
useful motion signal. Submit the diagnostic wrapper so the bank is copied and
verified once on node-local storage before any random motion-span reads. The
frozen VAE is loaded only to encode each query as deterministic `mu`; every
candidate still comes from the train-only bank and is rechecked against the
bank's exclusion policy:

```bash
export RETRIEVAL_DIAG_DIR=experiments/NIAF/continuous_trajectory_field/csl_daily_sentence_retrieval_diagnostics
CFG="$CFG" SENTENCE_MEMORY_DIR="$BANK_DIR" SPLIT=val TOP_K=8 \
OUT_JSON="$RETRIEVAL_DIAG_DIR/val_summary.json" \
DETAILS_JSONL="$RETRIEVAL_DIAG_DIR/val_details.jsonl" \
sbatch scripts/NIAF/diagnose_sentence_retrieval_sbatch.sh
```

Run it after `neighbors_val.npz` exists. Repeat with `SPLIT=train` or
`SPLIT=test` when those populations are also needed. Set `LIMIT=50` for a first
end-to-end smoke test; it limits query encoding but still requires a neighbor
table covering the complete, correctly ordered manifest.

The report compares the direct cosine top-1 with a deterministic random
eligible train-bank group and a diagnostic-only oracle over eligible recordings
in the top K. Motion distance is RMSE between deterministic VAE `mu`
trajectories after each latent channel is standardized from the complete train
bank and both sequences are linearly resampled to 64 normalized-time points.
This proxy tests retrieval coherence, not generated-pose quality: do not report
it as DTW/PA-DTW or use the oracle to choose training candidates. Ground-truth
query duration is used only for duration-compatibility statistics, never for
normal candidate ranking or variant selection.

Inspect the `overall`, `novel_text`, and `exact_seen_text` subsets. In addition
to direct/random/oracle latent distance, the report includes key-space and
source diversity, absolute log-duration gap, left/right hand validity, cosine
similarity, candidate coverage, and exact-text retrieval rate. Preserve the
optional detail JSONL when investigating individual failures.

## 4. Smoke test Phase A

The Phase-A configuration imports the clean CSL-Daily text-only v2 checkpoint
through the dedicated v2-to-v3 base loader. It freezes every inherited
parameter and trains only names below `hypernetwork.sentence_memory_`.

Before a full run, use one allocated GPU:

```bash
WANDB=0 \
CFG="$CFG" \
SENTENCE_MEMORY_DIR="$BANK_DIR" \
STAGE_SENTENCE_MEMORY=1 \
OUT_DIR=experiments/NIAF/continuous_trajectory_field/csl_sentence_memory_smoke \
LIMIT_TRAIN=50 \
LIMIT_VAL=50 \
MAX_TRAIN_BATCHES=1 \
MAX_VAL_BATCHES=1 \
EPOCHS=1 \
sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G \
  --time=00:45:00 scripts/NIAF/train_continuous_trajectory_field_sbatch.sh
```

The smoke run is acceptable only if:

- v2-to-v3 import reports missing sentence-memory parameters only;
- the memory-off parity check passes;
- only sentence-memory parameters are trainable;
- retrieval uses the train bank and the matching query neighbor table;
- one optimizer step and all configured validation modes finish without NaNs.

### 500/100 pilot and K ablation

After the smoke test, launch separate fresh pilots for `K={1,4,8}`. Each pilot
uses only 500 training and 100 validation queries but still retrieves from the
complete train-only bank and the full top-64 neighbor tables. Never reuse an
output directory between K values:

```bash
for K in 1 4 8; do
  WANDB=0 \
  CFG="$CFG" \
  SENTENCE_MEMORY_DIR="$BANK_DIR" \
  STAGE_SENTENCE_MEMORY=1 \
  SENTENCE_MEMORY_K="$K" \
  LIMIT_TRAIN=500 \
  LIMIT_VAL=100 \
  EPOCHS=5 \
  RUN_TAG="csl_daily_signtrajfield_v3_sentence_memory_phase_a_pilot_k${K}" \
  OUT_DIR="experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a_pilot_k${K}" \
  sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
    --time=12:00:00 scripts/NIAF/train_continuous_trajectory_field_sbatch.sh
done
```

Compare the same validation modes and wall-clock/step time across the three
runs. Keep `K=8` for the full run unless the pilot shows worse
latency-adjusted sentence-memory validation performance or a safety-constraint
failure. `--sentence_memory_k` is a fresh-run ablation override; exact v3
resume rejects changing K.

## 5. Train Phase A

Launch the full frozen-base run with the Phase-A config. The default retrieval
policy is `K=8`, top candidate retained plus deterministic weighted sampling
from the top 64, 10% supplementary-candidate dropout, and 25% shuffled-memory
training examples.

```bash
WANDB=0 \
CFG="$CFG" \
SENTENCE_MEMORY_DIR="$BANK_DIR" \
STAGE_SENTENCE_MEMORY=1 \
RUN_TAG=csl_daily_signtrajfield_v3_sentence_memory_phase_a \
sbatch scripts/NIAF/train_continuous_trajectory_field_sbatch.sh
```

If the user explicitly asks for W&B online logging, set `WANDB=1` and
`WANDB_MODE=online`. Use the machine's secure W&B authentication; never put a
credential in a config, command, shared file, or log.

The trainer validates three namespaces independently:

```text
text_only/*
sentence_memory/*
shuffled_sentence_memory/*
```

It saves `last.pt`, the best feasible checkpoint, the best infeasible
diagnostic checkpoint, and configured per-validation snapshots. Every v3
checkpoint records the external bank identity but not its payload.

## 6. Evaluate a checkpoint

Aligned evaluation for all causal modes should be submitted through Slurm:

```bash
export CHECKPOINT=experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/checkpoints/best.pt
export RESULT_JSON=experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/evaluation/test_aligned_modes.json
CFG="$CFG" CHECKPOINT="$CHECKPOINT" OUT_JSON="$RESULT_JSON" SPLIT=test \
SENTENCE_MEMORY=auto WORD_PRIOR=off \
SENTENCE_MEMORY_DIR="$BANK_DIR" STAGE_SENTENCE_MEMORY=1 \
sbatch scripts/NIAF/evaluate_continuous_trajectory_field_sbatch.sh
```

For predicted-length exports, use the combined Slurm launcher below for full
evaluation. If you invoke the Python exporter manually inside an allocation,
first stage the bank with `stage_sentence_memory_node.sh`, export
`SIGNTRAJ_SENTENCE_MEMORY_DIR` to that verified node-local directory, and pass
`--num_samples 0` explicitly:

```bash
"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.export_continuous_trajectory \
  --config "$CFG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --num_samples 0 \
  --length_mode predicted \
  --sentence_memory on \
  --word_prior off \
  --out_dir experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/evaluation/test_predicted_sentence_memory \
  --device cuda \
  --text_device cuda
```

Do not point this direct command at the shared CIFS bank. Repeat with
`--sentence_memory off` and `--sentence_memory shuffled`. Feed each
export to the existing default-DTW and corrected partwise PA-DTW evaluator.
The evaluator reports raw DTW, path-normalized DTW, reference-length-normalized
DTW, and unwarped left/right-hand travel and finite-difference jerk ratios.
Confirm that predicted duration is identical between off and on modes. For
cluster runs, the combined export/evaluation launcher stages the bank once per
node:

```bash
export GATE_ROOT=experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/evaluation/phase_b_gate
CFG="$CFG" CHECKPOINT="$CHECKPOINT" SPLIT=val \
SENTENCE_MEMORY=on WORD_PRIOR=off \
SENTENCE_MEMORY_DIR="$BANK_DIR" STAGE_SENTENCE_MEMORY=1 \
OUT_DIR="$GATE_ROOT/sentence_memory" \
sbatch scripts/NIAF/export_and_evaluate_continuous_trajectory_predicted_sbatch.sh
```

Run the same command with `SENTENCE_MEMORY=shuffled` and its corresponding
output directory. For `SENTENCE_MEMORY=off`, omit bank staging; the checkpoint
identity is retained in `export_summary.json`, while no bank payload or metadata
is opened.

## 7. Phase-B gate

Do not start partial unfreezing merely because Phase A completed. Require:

- at least the configured minimum number of paired **novel-text** validation
  samples;
- lower novel-text validation whole-body PA-DTW for correct retrieval than
  text-only;
- a paired novel-text bootstrap 95% confidence interval entirely below zero;
- correct retrieval better than shuffled retrieval on novel text;
- no more than 2% novel-text path-loss regression for either hand;
- text-only predictions matching the source v2 model within `1e-7` maximum
  absolute error.

The schema-v2 gate uses `novel_text` as its primary acceptance subset. It still
reports `overall` and `exact_seen_text` DTW/PA-DTW and hand diagnostics, but
those secondary aggregates cannot turn an otherwise failing novel-text result
into an acceptance. The required hand-path bound is computed from paired
novel-text predicted-duration exports. The checkpoint's aggregate validation
hand-path loss is retained as an explicitly non-gating diagnostic because an
aggregate checkpoint metric cannot recover the novel/seen strata. Predicted-
duration hand jerk ratios are also diagnostic; lower jerk is not assumed to be
universally better.

After producing predicted-length `text_only`, `sentence_memory`, and
`shuffled_sentence_memory` validation exports and running both DTW evaluators
for each export, generate the machine-readable gate artifact with:

```bash
export GATE_ROOT=experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/evaluation/phase_b_gate

"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.analyze_sentence_memory_phase_b_gate \
  --purpose phase_b_gate \
  --text_export_summary "$GATE_ROOT/text_only/export_summary.json" \
  --memory_export_summary "$GATE_ROOT/sentence_memory/export_summary.json" \
  --shuffled_export_summary "$GATE_ROOT/shuffled_sentence_memory/export_summary.json" \
  --text_only_default_dtw "$GATE_ROOT/text_only/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --text_only_pa_dtw "$GATE_ROOT/text_only/dtw_mpjpe_t2m_pa_h2s_betas.json" \
  --sentence_memory_default_dtw "$GATE_ROOT/sentence_memory/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --sentence_memory_pa_dtw "$GATE_ROOT/sentence_memory/dtw_mpjpe_t2m_pa_h2s_betas.json" \
  --shuffled_sentence_memory_default_dtw "$GATE_ROOT/shuffled_sentence_memory/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --shuffled_sentence_memory_pa_dtw "$GATE_ROOT/shuffled_sentence_memory/dtw_mpjpe_t2m_pa_h2s_betas.json" \
  --phase_a_checkpoint "$CHECKPOINT" \
  --gate_metric ndtw \
  --bootstrap_samples 10000 \
  --bootstrap_seed 1234 \
  --confidence 0.95 \
  --min_pairs 2 \
  --duration_tolerance 1e-7 \
  --hand_path_max_relative_degradation 0.02 \
  --parity_tolerance 1e-7 \
  --out_json "$GATE_ROOT/phase_b_gate.json" \
  --fail_on_reject
```

The current exporter writes compact per-sample body/hand/face gate means, null
attention mass, real-candidate mass, and memory availability into each export
summary. The analyzer aggregates these for all three text subsets and modes.
When analyzing an older export without these fields, the non-gating
`export_gate_null_diagnostics_available` check is false and the artifact carries
an explicit warning; regenerate the exports before interpreting memory usage.

After selecting the Phase-A checkpoint, report the same paired metrics on test
without turning test data into a training decision. First produce all three
predicted-length test exports and their default/PA-DTW files under
`$TEST_REPORT_ROOT`, then run:

```bash
export TEST_REPORT_ROOT=experiments/NIAF/continuous_trajectory_field/csl_daily_signtrajfield_v3_sentence_memory_phase_a/evaluation/test_report

"$PYTHON_BIN" -m \
  NIAF.continuous_trajectory_field.scripts.analyze_sentence_memory_phase_b_gate \
  --purpose test_report \
  --text_export_summary "$TEST_REPORT_ROOT/text_only/export_summary.json" \
  --memory_export_summary "$TEST_REPORT_ROOT/sentence_memory/export_summary.json" \
  --shuffled_export_summary "$TEST_REPORT_ROOT/shuffled_sentence_memory/export_summary.json" \
  --text_only_default_dtw "$TEST_REPORT_ROOT/text_only/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --text_only_pa_dtw "$TEST_REPORT_ROOT/text_only/dtw_mpjpe_t2m_pa_h2s_betas.json" \
  --sentence_memory_default_dtw "$TEST_REPORT_ROOT/sentence_memory/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --sentence_memory_pa_dtw "$TEST_REPORT_ROOT/sentence_memory/dtw_mpjpe_t2m_pa_h2s_betas.json" \
  --shuffled_sentence_memory_default_dtw "$TEST_REPORT_ROOT/shuffled_sentence_memory/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --shuffled_sentence_memory_pa_dtw "$TEST_REPORT_ROOT/shuffled_sentence_memory/dtw_mpjpe_t2m_pa_h2s_betas.json" \
  --phase_a_checkpoint "$CHECKPOINT" \
  --gate_metric ndtw \
  --bootstrap_samples 10000 \
  --bootstrap_seed 1234 \
  --confidence 0.95 \
  --min_pairs 2 \
  --duration_tolerance 1e-7 \
  --hand_path_max_relative_degradation 0.02 \
  --parity_tolerance 1e-7 \
  --out_json "$TEST_REPORT_ROOT/test_paired_report.json"
```

A test report sets `accepted=false` and `phase_b_authorizing=false` by design,
even when `scientific_criteria_passed=true`. It still contains paired bootstrap
results and gate/null aggregates for novel text, exact-seen text, and overall.
Only a `purpose=phase_b_gate`, split=`val` artifact may authorize Phase B.

When the gate passes, start a new model-only run with the Phase-B config and the
selected Phase-A checkpoint:

```bash
export PHASE_B_CFG=NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_b.yaml
WARM_START="$CHECKPOINT" \
PHASE_B_GATE_REPORT="$GATE_ROOT/phase_b_gate.json" \
WANDB=0 \
CFG="$PHASE_B_CFG" \
SENTENCE_MEMORY_DIR="$BANK_DIR" \
STAGE_SENTENCE_MEMORY=1 \
RUN_TAG=csl_daily_signtrajfield_v3_sentence_memory_phase_b \
sbatch scripts/NIAF/train_continuous_trajectory_field_sbatch.sh
```

Phase B creates a fresh optimizer and epoch counter. It keeps the text planner,
duration head, word branch, and SIREN fields frozen; trains the sentence branch
at `1e-4`; trains only the approved shared output heads at `1e-5`; and distills
the memory-off output against the frozen original v2 teacher.

## Operational invariants

- Do not point `bank_dir` at validation or test data.
- Do not load all raw motions or the VAE during training.
- Do not use target duration, target pose, or gloss to choose candidates.
- Do not relax exclusions to fill K; pad with the null candidate.
- Do not resume a v3 checkpoint with a different `bank_id`.
- Use `--base_checkpoint` only for v2-to-v3 initialization, `--resume` only for
  exact continuation, and `--warm_start` only for a model-only v3 Phase-B run.
- Keep word-prior conditioning off in the first CSL experiment.
