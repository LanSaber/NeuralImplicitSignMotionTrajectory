# Soft Word Arranger Findings and Next Steps

Snapshot date: 2026-06-26.

This note summarizes what we learned from the Phoenix inner
SoftWordArranger ablations and the adapter diagnostics. It is intended as a
future-improvement checklist, not as a finished paper table.

## Current Context

The Phoenix inner ablations isolate the SoftWordArranger by training with:

```text
--prior-mode soft_arranger --disable-adapter
```

Therefore, these experiments measure the arranged word prior `z_prior`, not the
full content-style adapter output. In evaluation, `z_adapt` is equal to
`z_prior` for these checkpoints.

Relevant evaluation outputs:

```text
visualize/phoenix_inner_swa_eval_best_test_shuffle123_no_test_negatives/table.csv
visualize/phoenix_inner_swa_eval_best_test_shuffle123_no_test_negatives/summary.json

visualize/phoenix_inner_swa_eval_best_test_shuffle123_gate_null/table.csv
visualize/phoenix_inner_swa_eval_best_test_shuffle123_gate_null/summary.json

visualize/phoenix_inner_swa_eval_checkpoint_sweep/table.csv
visualize/phoenix_inner_swa_eval_checkpoint_sweep/summary.json
```

## Main Findings

### 1. Test-Time Random Negatives Were a Protocol Mistake

The first ablation table reused each checkpoint's candidate configuration during
evaluation. For the original `full_v2` checkpoint this meant:

```text
num_word_candidates = 32
num_negative_candidates = 16
```

So the evaluator injected random negative word clips at test time. This is
useful as a stress test, but it is not the natural inference protocol. At
inference, the candidate set should come from lexical matches to the input
sentence/gloss, not from artificial distractors.

The corrected main evaluation uses:

```text
eval_num_word_candidates = 16
eval_num_negative_candidates = 0
```

for every variant. Under this clean inference setting, the run originally named
`noneg` is better described as:

```text
Full arranger trained without random negatives
```

because random-negative discrimination is not part of the final method.

### 2. Random Negatives Should Be Treated as Diagnostic, Not Core

The valid no-negative checkpoint was truly trained without negatives:

```text
experiments/flow/adapter/phoenix_inner_swa_noneg_noadapter_b256x1_r2
```

Its configuration:

```text
num_word_candidates = 16
num_negative_candidates = 0
negative_usage_loss_weight = 0.0
```

In the clean no-test-negative evaluation, this model gave the strongest prior
latent and pose numbers among the current runs. This suggests that random
negative training may hurt the natural lexical-matching inference setting, even
if it is useful for stress-testing gate discrimination.

Recommended paper framing:

```text
Main method: no random-negative training.
Diagnostic stress test: optionally inject random negatives and report gate gap.
```

Do not present random negatives as a main contribution unless we retrain and
show a clear benefit under the intended inference protocol.

### 3. Gate Gap Is Only Meaningful With Negatives

Gate gap is:

```text
mean positive-candidate gate probability
- mean negative-candidate gate probability
```

When evaluation has no negative candidates, `gate_neg` and `gate_gap` are
undefined. Gate gap should only appear in a diagnostic or robustness table where
random distractors are intentionally injected.

In the stress table with 16 injected test negatives, removing word-text features
collapsed the gate gap from about `0.199` to about `0.026`, which is useful
evidence that `e_k` carries important semantic candidate information.

### 4. Attention Cosine Near 1 Is Not a Good Sign

The attention frame cosine values are still close to 1.0 in most evaluated
checkpoints. This means the attention distribution is nearly constant across
latent time steps.

Interpretation:

```text
High attention cosine = frozen or slowly changing attention.
```

This is not a strength. It means the arranger may learn a global candidate
mixture rather than a time-varying word schedule. This matches the earlier
collapse diagnosis on the How2Sign sample where the soft arranger produced a
nearly constant latent sequence.

Attention cosine should be treated as a diagnostic, not as a headline table
metric.

### 5. Dynamics and Std Ratios Are Diagnostic, Not Pure Quality Metrics

The dynamics ratio and latent-std ratio measure how much temporal variation the
prior has relative to the ground-truth sentence latent sequence. Higher values
mean less collapse, but not necessarily better motion.

Example:

```text
w/o word-text features e_k
```

can have higher dynamics/std ratios while also having worse latent error, worse
pose error, and much higher NULL usage. This means it moves more, but the motion
is less semantically grounded.

Use these metrics carefully:

```text
Good for diagnosing collapse.
Not sufficient as quality metrics.
```

Main tables should prioritize reconstruction and candidate-selection metrics:

```text
prior latent
prior pose
group coverage
NULL usage
```

Dynamics/std and attention cosine are better suited for diagnostics or appendix
analysis.

### 6. Best-by-Latent Checkpoints Favor Static Priors

The checkpoint sweep showed a tradeoff:

```text
early best.pt checkpoints: lower latent/pose error but very static priors
later checkpoints: more temporal dynamics but worse latent/pose error
```

This means selecting `best.pt` only by validation latent can choose a smooth or
partly collapsed prior. Future model selection should use a multi-metric score
instead of latent error alone.

Possible validation score:

```text
score =
  latent_prior
  + lambda_pose * pose
  + lambda_null * null_usage
  - lambda_cov * group_coverage
  + lambda_dyn * abs(prior_delta_ratio - target)
```

The exact weights should be tuned, but the key point is that checkpoint
selection should include both quality and anti-collapse behavior.

## Component-Level Lessons

### Word-Text Features `e_k`

Removing word-text features causes the clearest semantic failure:

```text
worse prior latent
worse prior pose
higher NULL usage
collapsed gate gap in stress evaluation
```

Conclusion: `e_k` is essential. It should remain part of the final arranger.

### Word-Motion Latents `u_k`

Removing word-motion latents causes a smaller but consistent degradation.

Conclusion: `u_k` is useful, but less dominant than `e_k` in the current
Phoenix setup.

### Candidate Gates

Disabling candidate gates removes the learned candidate-level relevance filter.
In stress evaluation, the gate gap becomes exactly zero by construction.

The clean no-negative evaluation gives mixed numbers, because candidate gates
are mainly designed for candidate discrimination and there are no explicit
negative candidates at inference. Gates should still be kept for semantic
selection and compatibility with robustness diagnostics, but their contribution
should be evaluated again after retraining all ablations under the final
no-negative protocol.

### NULL Memory

NULL usage is low in some settings but rises sharply when candidate semantics
are weakened, especially without word-text features.

Interpretation: NULL memory is a useful fallback, but high NULL usage is a
warning that the arranger is avoiding candidate selection. NULL usage should be
reported as a percentage:

```text
NULL usage % = 100 * null_usage
```

### Anti-Collapse Losses

The anti-collapse bundle includes:

```text
group coverage loss
group entropy peakiness loss
attention variation loss
prior velocity loss
prior acceleration loss
prior variance floor loss
negative usage loss
```

The current `w/o anti-collapse` ablation removes the whole bundle, not each
sub-loss individually.

The best-latent checkpoint table does not strongly highlight the benefit because
`best.pt` tends to be early and static. The checkpoint sweep is more informative:
later full checkpoints increase temporal dynamics more than early checkpoints,
but they pay a reconstruction penalty.

Conclusion: anti-collapse losses are directionally useful, but the current
weights and checkpoint selection do not yet solve time-varying scheduling.
The group entropy peakiness loss is the next unordered constraint to test: it
penalizes high per-frame entropy over matched occurrence groups, encouraging a
clear local group choice without assuming English word order.

## Recommended Main Paper Table

For the current paper story, use the natural inference protocol:

```text
No random distractors at test time.
Full arranger = trained without random negatives.
```

Recommended columns:

```text
Prior latent
Prior pose
Group coverage
NULL usage
```

Avoid putting `attention cosine`, `dynamics ratio`, and `std ratio` in the main
table because they are easy to misread and expose remaining temporal-scheduling
weaknesses. They can be discussed as diagnostics or limitations.

Important caveat:

The current rows are not a perfectly clean one-factor ablation table because the
new full row is trained without negatives, while several component ablations
were trained with negatives. For a fully rigorous table, rerun all component
ablations from the no-negative baseline.

## Recommended Future Experiments

### A. Rerun Clean Phoenix Inner Ablations

Use the final no-negative baseline for every variant:

```text
num_word_candidates = 16
num_negative_candidates = 0
negative_usage_loss_weight = 0.0
```

Rerun:

```text
full
w/o candidate gates
w/o NULL memory
w/o attention variation loss
w/o anti-collapse losses
w/o word-text features e_k
w/o word-motion latents u_k
```

This will remove the current confound between component ablations and
random-negative training.

### B. Add Evaluation Overrides

`evaluate_adapter.py` currently uses the checkpoint candidate config unless a
custom helper overrides it. Add official CLI flags:

```text
--eval-num-word-candidates
--eval-num-negative-candidates
--eval-candidate-selection
--eval-max-positive-variants-per-key
```

This prevents accidental reuse of training-time random negatives during natural
inference evaluation.

### C. Improve Checkpoint Selection

Validation should not select only by latent loss. Add a configurable validation
selection metric that combines:

```text
latent quality
pose quality
group coverage
NULL usage
temporal dynamics
```

This should reduce the chance of selecting an early static prior.

### D. Improve Temporal Scheduling Without English-Order Supervision

We cannot assume English/gloss order equals sign-language order. Future temporal
alignment should avoid monotonic English-order losses.

Possible directions:

```text
unordered coverage with stronger minimum mass
attention entropy schedules
diversity losses over time
latent dynamics matching with better weighting
duration predictor conditioned on candidate set
weak CTC/DTW-style latent alignment without fixed lexical order
sign-order latent variable or permutation module
```

The goal is to make attention move through useful candidates over time without
forcing English word order.

### E. Separate Robustness From Main Inference

If we still care about robustness to distractors, keep it as a separate
diagnostic:

```text
Natural inference table: no test negatives.
Robustness table: inject 16 random negatives and report gate gap / negative usage.
```

This avoids confusing reviewers and keeps the main method aligned with actual
deployment.

## Practical Rules Going Forward

1. Do not inject random negatives in the main test evaluation.
2. Do not report gate gap unless negative candidates are present.
3. Do not interpret high dynamics/std ratio as quality by itself.
4. Treat attention cosine near 1.0 as a collapse warning.
5. Report NULL usage as a percentage.
6. Use `best.pt` carefully; it may favor static priors.
7. Rerun clean no-negative ablations before making strong component claims.
