# CSL-Daily Phase-A motion-contrast v1

This experiment asks one narrow question before unfreezing the trajectory
generator: does the frozen-base sentence branch use the retrieved **motion**, or
does it respond only to candidate keys/scores and a nearly average residual?

The run is validation-only for model development. It never loads a test query
or `neighbors_test.npz`, and W&B is disabled. The exact configuration is
`NIAF/continuous_trajectory_field/configs/csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1.yaml`.

## Data lock

The canonical 1,077-row validation manifest is grouped by normalized sentence
text. Exact train-bank texts are excluded from scientific analysis. The 796
novel texts are sorted by `SHA256(normalized_text)`:

- First 256 text clusters: development, used for every epoch validation,
  checkpoint decision, and early-stop decision.
- Remaining 540 text clusters: confirmation, inaccessible to training and
  evaluated once only after code, configuration, and checkpoint are locked.

All signer realizations of a text stay together. The READY partition artifact
is stored beside, not inside, the fresh experiment directory at
`<run>.prerequisites/validation_text_partition`. The trainer derives the same
development indices from the canonical validation dataset, preserving the
audited `neighbors_val.npz` manifest identity.

## Training contract

- Full CSL-Daily training set and full train-only sentence bank, `K=8`, `M=64`,
  duration weight `0.05`.
- Strict fresh initialization from the clean v2 text-only checkpoint, whose
  required SHA256 is
  `06ca0a2613005b6e3949bab0e5d7ded999b212723debd3e7685a58c077e44c54`.
- Phase A only: inherited generator, duration head, text planner/encoder,
  SIRENs, and word-prior branch remain frozen.
- At most four epochs, validation every epoch, minimum two epochs, and patience
  two validation events. A logical batch remains 32 per GPU with accumulation
  two; the paired branches cap the physical microbatch at eight samples and
  2,048 padded frames.
- Correct retrieval is paired with one corrupt copy: 90% of eligible rows use
  motion-only corruption, while deterministic full shuffle replaces that
  corruption for the remaining 10%. This is not a third simultaneous branch.
  The five explicit paired objectives each have weight 1.0; sentence
  residual/gate sparsity keeps weight `1e-4`.
- Development validation has four modes: `text_only`, `sentence_memory`,
  `shuffled_sentence_memory`, and `motion_shuffled_sentence_memory`.
  Motion-only permutations are deterministically keyed by immutable query
  identity, global seed, and the current checkpoint epoch; the standalone
  confirmation exporter uses that same saved epoch.

The development checkpoint key is lexicographic: feasibility first, normalized
constraint violation second, and selection score third. If no checkpoint is
feasible, the run preserves `best_infeasible.pt` but exits nonzero. That file is
never eligible for confirmation or Phase B.

First run the mandatory one-optimizer-step smoke. Its committed overlay keeps
the same model/partition contract but uses one epoch and disables terminal
scientific feasibility, because two train batches and one validation batch
cannot establish a promotion result:

```bash
sbatch scripts/NIAF/smoke_csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1_sbatch.sh
```

Only after the smoke passes should the dedicated full launcher be used. It
hard-binds the fresh checkpoint, four-GPU Phase-A settings, W&B-off mode, and
node-local staging of only train/validation neighbor tables:

```bash
sbatch scripts/NIAF/train_csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1_sbatch.sh
```

Do not set `RESUME`, `WARM_START`, or `BASE_CHECKPOINT` for the fresh launch.
Exact recovery after an interruption should use the generic launcher's strict
resume contract, never a fresh output directory.

## One-shot confirmation

After choosing one development-feasible checkpoint, lock its path and launch
the four-mode confirmation export:

```bash
CHECKPOINT=/absolute/path/to/locked/checkpoint.pt \
sbatch scripts/NIAF/confirm_csl_daily_signtrajfield_v3_sentence_memory_phase_a_motion_contrast_v1_sbatch.sh
```

`CHECKPOINT` must resolve exactly to `<run>/checkpoints/best.pt`. Before any
promotion decision, the analyzer verifies the persisted objective, behavior,
resume, four-rank RNG, and full-data config identities. It also requires the
terminal `selection_summary.json` and complete `metrics.jsonl` to prove that
the full run completed at least two uncapped 347-row development validations.
Thus an epoch-1 best checkpoint remains eligible only after the independent
run-level minimum-epoch contract has been satisfied; smoke and one-GPU
checkpoints are rejected.

The job exports only the READY confirmation manifest, uses predicted duration,
computes default DTW and corrected partwise PA-DTW, and then applies the fixed
analysis. Four scientific modes are accompanied by two prediction-only
integrity controls: fully masked all-null memory and the pinned original v2
text-only checkpoint. The all-null path does not instantiate the memory
provider or read motion payloads. Repeated signer rows are averaged within
normalized text before any inference. Confidence intervals use 10,000
text-cluster bootstrap resamples, seed 1234.

Phase B is allowed only if every gate passes:

1. Correct-retrieval whole-body PA-nDTW improves by at least 0.5% against
   text-only, full-shuffle, and motion-only-shuffle, and each paired 95% CI for
   the absolute difference is entirely below zero. The two corruption-control
   p-values form the exact Holm family; the text-only promotion decision uses
   its predeclared bootstrap CI separately (its randomization p-value is
   descriptive only).
2. `Rmotion = sqrt(MSE(correct, motion-shuffled) / MSE(correct, memory-off))`,
   computed with equal text-cluster weight in exported rot6D, is at least 0.10
   and its bootstrap lower bound is at least 0.05.
3. The upper bootstrap bound on relative hand-path-error degradation is below
   2% for each hand.
4. Predicted durations match across all four modes within `1e-7`.
5. On every confirmation row, all-null and memory-off predictions/durations
   are array-exact, all-null gate and candidate mass are zero with null mass
   one, and selected-checkpoint memory-off predictions/durations match the
   pinned original v2 checkpoint within `1e-7`. The original v2 SHA256 is
   checked again before export and analysis.

The analyzer writes a READY evidence directory even when a scientific gate
fails, then exits nonzero to block an `afterok` Phase-B dependency. A failure
means redesign or further Phase-A work—not partial unfreezing.
