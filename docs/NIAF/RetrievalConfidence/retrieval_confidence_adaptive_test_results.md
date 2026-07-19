# Retrieval-Confidence Adaptive NIAF: Test Results

## 1. Evaluation protocol

The retrieval-confidence adaptive NIAF model was trained for 50 epochs on PHOENIX.
The checkpoint with the best validation endpoint loss was epoch 12:

```text
experiments/NIAF/retrieval_confidence_field/
  phoenix_retrieval_confidence_adaptive_trainbalanced/checkpoints/best.pt
```

The deployable test path was evaluated on 100 random PHOENIX test sentences with
seed 123. Inference used:

- raw sentence text;
- text-predicted sequence length;
- the frozen SoftArranger adapter scaffold; and
- the retrieval-confidence adaptive residual field.

No ground-truth pose, gloss, or duration was supplied to the generator. Retrieval
was restricted to `manifest_train.balanced.jsonl`, containing 29,323 training clips
and 1,085 lexical keys.

The comparison is paired: `adapter scaffold` and `adaptive field` use the same text,
predicted duration, retrieval candidates, and test sample. Lower DTW is better.

## 2. Validation checkpoint

At epoch 12, the field improved the validation endpoint and FK joint losses, but did
not improve the hand or temporal-dynamics losses.

| Metric | Adapter scaffold | Adaptive field | Observation |
|---|---:|---:|---|
| Endpoint loss | 14.04943 | **13.85638** | 1.37% better |
| FK joint loss | 0.99407 | **0.97618** | 1.80% better |
| Hand loss | **0.11479** | 0.11524 | 0.39% worse |
| FK velocity loss | **0.14210** | 0.14608 | 2.80% worse |
| FK acceleration loss | **0.16473** | 0.17629 | 7.02% worse |
| FK jerk loss | **0.25855** | 0.28378 | 9.76% worse |
| Hand path ratio | 0.77544 | **0.98823** | Field is closer to the ideal 1.0 |

## 3. Predicted duration

| Statistic | Frames |
|---|---:|
| Mean ground-truth length | 76.20 |
| Mean predicted length | 82.60 |
| Mean absolute error | 13.52 |
| Median absolute error | 10.00 |
| Mean bias, predicted minus GT | +6.40 |
| Maximum absolute error | 128 |

The duration predictor tends to overestimate sequence length and has one severe
failure in this subset. Both compared motion methods use the same predicted lengths,
so this does not bias the field-versus-scaffold comparison, but it does affect their
absolute text-to-sign quality.

## 4. Translated DTW-MPJPE

The evaluator labels the adaptive field as `flow` and the adapter scaffold as
`adapter_prior` in the raw JSON and CSV files.

| Part | Adapter scaffold | Adaptive field | Relative change | Field wins |
|---|---:|---:|---:|---:|
| Body | **4.16581** | 4.18669 | 0.50% worse | 42% |
| Left hand | 5.89904 | **5.87810** | 0.35% better | 55% |
| Right hand | 5.38688 | **5.38245** | 0.08% better | 51% |
| Whole body | 11.35843 | **11.23215** | **1.11% better** | **66%** |

The paired whole-body DTW gain is 0.12628 with a bootstrap 95% confidence interval
of [0.05884, 0.19240]. The paired Wilcoxon two-sided p-value is 0.00040. Normalized
DTW also improves by 1.17%, so the result is not only an effect of DTW path length.

## 5. PA-DTW-MPJPE

| Part | Adapter scaffold | Adaptive field | Relative change | Field wins |
|---|---:|---:|---:|---:|
| Body | **4.69259** | 4.70735 | 0.31% worse | 44% |
| Left hand | **1.06121** | 1.06719 | 0.56% worse | 31% |
| Right hand | **1.18068** | 1.19026 | 0.81% worse | 17% |
| Whole body | 7.14925 | **7.06384** | **1.19% better** | **70%** |

The paired whole-body PA-DTW gain is 0.08542 with a bootstrap 95% confidence
interval of [0.03415, 0.13178]. The paired Wilcoxon two-sided p-value is
0.00000395. Normalized PA-DTW improves by 1.29%.

## 6. Learned adaptive behavior

Frame-weighted means over 8,260 generated frames show that the model applies small
corrections, especially to the hands and face.

| Articulator | Predicted confidence | Effective correction gate | Stride-16 weight |
|---|---:|---:|---:|
| Body | 0.533 | 0.0466 | 48.2% |
| Left hand | 0.773 | 0.0107 | 67.4% |
| Right hand | 0.737 | 0.0162 | 64.2% |
| Face | 0.782 | 0.0160 | 67.9% |

The mean sequence-level field-to-scaffold rot6D RMS difference is 0.03167. High
confidence favors the coarsest local code and suppresses the correction gate, as
designed. On this test set that behavior produces a small, consistent whole-body
gain, but leaves little capacity for correcting fine hand articulation.

## 7. Conclusion

This experiment is a positive but modest result. The full adaptive field improves
whole-body DTW on most samples under a strict train-only retrieval bank and a fully
non-oracle inference path. However, body-only DTW and PA hand articulation are
slightly worse than the adapter scaffold. The method currently improves overall
trajectory placement more convincingly than local finger pose quality.

This is not yet enough to claim that retrieval-confidence adaptation is better than
the previous fixed stride-8 local field. The next controlled study should evaluate,
with the same train-only bank and samples:

1. adapter scaffold only;
2. fixed stride-8 local field;
3. multi-scale field without confidence gating; and
4. the full confidence-adaptive field.

Checkpoint selection should include PA hand error and temporal dynamics rather than
endpoint loss alone. The duration head also needs a robust-loss or outlier-focused
ablation before a publication benchmark.

## 8. Artifacts

```text
experiments/NIAF/retrieval_confidence_field/
  phoenix_retrieval_confidence_adaptive_trainbalanced/
    eval_test100_seed123_predicted_best/
      export_summary.json
      dtw_mpjpe_default.json
      dtw_mpjpe_default.csv
      dtw_mpjpe_pa.json
      dtw_mpjpe_pa.csv
      sample_0000.npz ... sample_0099.npz
      gt_0000.npz ... gt_0099.npz
```
