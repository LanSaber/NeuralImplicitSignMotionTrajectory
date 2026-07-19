# Retrieval-Uncertainty Adaptive Knot Field: Test Results

## 1. Evaluation protocol

The retrieval-uncertainty adaptive knot field was trained for 50 epochs on
PHOENIX. The checkpoint selected by the validation composite score is epoch 10:

```text
experiments/NIAF/retrieval_confidence_field/
  phoenix_retrieval_uncertainty_adaptive_knots_trainbalanced/checkpoints/best.pt
```

The deployable text-to-sign path was evaluated on the same 100 PHOENIX test
sentences and seed-123 manifest used for the previous retrieval-confidence model.
Inference used only:

- raw sentence text;
- text-predicted sequence length;
- the frozen SoftArranger adapter scaffold; and
- the trained uncertainty-adaptive residual field.

No ground-truth pose, gloss, or duration was supplied to the generator. Retrieval
was restricted to `manifest_train.balanced.jsonl`, with 29,323 training clips and
1,085 lexical keys.

The main comparison is paired: `adapter scaffold` and `adaptive field` use the
same text, predicted duration, retrieval candidates, and sample. Lower DTW is
better. The evaluator names these methods `adapter_prior` and `flow` in the raw
JSON and CSV files.

## 2. Training and checkpoint selection

The best composite validation score occurred at epoch 10, while the last epoch
was mildly worse:

| Checkpoint | Composite score | Endpoint loss | Hand-relative loss | Residual RMS |
|---|---:|---:|---:|---:|
| Epoch 10, best | **24.55116** | **21.41280** | **0.11451** | 0.03535 |
| Epoch 50, last | 24.78287 | 21.57101 | 0.11583 | 0.05596 |

At the selected checkpoint, the validation comparison was:

| Metric | Adapter scaffold | Adaptive field | Relative change |
|---|---:|---:|---:|
| Endpoint loss | 21.81228 | **21.41280** | 1.83% better |
| Hand-relative loss | 0.11496 | **0.11451** | 0.39% better |
| FK velocity loss | **0.20704** | 0.20733 | 0.14% worse |
| FK acceleration loss | **0.23684** | 0.24202 | 2.19% worse |
| FK jerk loss | **0.36980** | 0.38286 | 3.53% worse |
| Hand-path ratio | 0.77573 | **0.84150** | Closer to the ideal 1.0 |

The field recovers more hand travel and improves endpoint reconstruction, but the
extra motion is slightly less accurate in velocity, acceleration, and jerk space.
The epoch-10 checkpoint should be used instead of `last.pt`.

## 3. Predicted duration

| Statistic | New model | Previous model |
|---|---:|---:|
| Mean ground-truth length | 76.20 | 76.20 |
| Mean predicted length | 81.20 | 82.60 |
| Mean absolute error | **12.52** | 13.52 |
| Median absolute error | **8.00** | 10.00 |
| Mean bias, predicted minus GT | **+5.00** | +6.40 |
| Maximum absolute error | **124** | 128 |

The duration head improved on this fixed subset, but one severe outlier remains.
Both methods in each paired field-versus-scaffold comparison use the same duration,
so duration error does not explain the residual field's within-model gain.

## 4. Translated DTW-MPJPE

| Part | Adapter scaffold | Adaptive field | Relative change | Field wins |
|---|---:|---:|---:|---:|
| Body | **4.11067** | 4.12479 | 0.34% worse | 39% |
| Left hand | 5.80817 | **5.74274** | **1.13% better** | 65% |
| Right hand | 5.33827 | **5.27300** | **1.22% better** | 65% |
| Whole body | 11.13865 | **10.91885** | **1.97% better** | **67%** |

The paired whole-body gain is 0.21980, with bootstrap 95% confidence interval
[0.13875, 0.30347] and two-sided Wilcoxon p-value 0.00000185. Normalized DTW
improves by 2.04%.

The left- and right-hand gains are also significant: p=0.00164 and p=0.0000165,
respectively. The small body degradation is not significant (p=0.0801).

## 5. PA-DTW-MPJPE

| Part | Adapter scaffold | Adaptive field | Relative change | Field wins |
|---|---:|---:|---:|---:|
| Body | **4.61413** | 4.61562 | 0.03% worse | 46% |
| Left hand | **1.05513** | 1.06114 | 0.57% worse | 30% |
| Right hand | **1.16702** | 1.17384 | 0.58% worse | 24% |
| Whole body | 7.04700 | **6.81637** | **3.27% better** | **79%** |

The paired whole-body PA gain is 0.23063, with bootstrap 95% confidence interval
[0.17575, 0.29166] and two-sided Wilcoxon p-value 5.56e-12. Normalized PA-DTW
improves by 3.07%.

The isolated PA hand metrics still degrade significantly. This means the field
improves global hand trajectories and whole-body coordination, but not the local
finger configuration after rigid alignment.

## 6. Comparison with the previous adaptive field

On the same test manifest, the new full prediction has lower absolute DTW than the
previous retrieval-confidence field for every reported part:

| Metric | Change from previous full field |
|---|---:|
| Translated body | 1.48% better |
| Translated left hand | 2.30% better |
| Translated right hand | 2.03% better |
| Translated whole body | **2.79% better** |
| PA body | 1.95% better |
| PA left hand | 0.57% better |
| PA right hand | 1.38% better |
| PA whole body | **3.50% better** |

This direct cross-checkpoint comparison includes the improved duration predictor,
so it is not a pure architecture ablation. The cleaner within-model result is that
the new field improves translated whole-body DTW by 1.97% over its scaffold,
compared with 1.11% for the previous field, and improves PA whole-body DTW by 3.27%,
compared with 1.19% previously.

## 7. Learned adaptive behavior

Frame-weighted statistics over 8,120 generated frames show nontrivial, structured
corrections:

| Articulator | Trust | Correction need | Knot density | Gate | Mean angular correction | Dominant code scale |
|---|---:|---:|---:|---:|---:|---:|
| Body | 0.498 | 0.642 | 0.825 | 0.235 | 2.26 deg | stride 4, 52.6% |
| Left hand | 0.739 | 0.445 | 0.742 | 0.411 | 1.56 deg | stride 8, 52.8% |
| Right hand | 0.729 | 0.459 | 0.732 | 0.439 | 1.82 deg | stride 8, 51.4% |
| Face | 0.785 | 0.284 | 0.531 | 0.254 | 0.58 deg | stride 16, 65.8% |

Correction need is associated with actual correction magnitude at frame level,
most strongly for the body (Spearman rho 0.482), then the left hand (0.228) and
right hand (0.167). The model is therefore using the adaptive controls rather than
collapsing to its scaffold.

However, uncertainty calibration remains weak. At the best checkpoint, validation
confidence-to-target correlations were 0.029 for body, 0.088 for left hand, -0.015
for right hand, and 0.012 for face. Predicted correction need also systematically
underestimated its supervised target:

| Articulator | Predicted need | Target need |
|---|---:|---:|
| Body | 0.648 | 0.793 |
| Left hand | 0.445 | 0.788 |
| Right hand | 0.465 | 0.830 |
| Face | 0.287 | 0.412 |

Thus, the motion improvements demonstrate a useful uncertainty-conditioned,
multi-scale residual architecture, but they do not yet establish that retrieval
uncertainty is accurately calibrated to per-frame scaffold error. The preference
for stride-8 hand codes also suggests that the adaptive field has not learned to
use its finest hand scale as the primary source of finger detail.

## 8. Conclusion

This is a positive model-quality result. The new deployable field improves both
translated hands, translated whole-body DTW, PA whole-body DTW, duration error, and
hand-path coverage. Its whole-body gains are larger and more consistent than those
of the previous retrieval-confidence field.

It is not yet a complete validation of the uncertainty mechanism. PA-normalized
finger articulation and temporal dynamics remain worse than the adapter scaffold,
and the confidence calibrator has almost no validation correlation with its error
target. Before using calibration as the main paper claim, the next controlled
experiment should compare constant controls, confidence-only controls, adaptive
knots without confidence, and the full model under the same duration and scaffold.
Checkpoint selection should also give explicit weight to PA hand error.

## 9. Artifacts

```text
experiments/NIAF/retrieval_confidence_field/
  phoenix_retrieval_uncertainty_adaptive_knots_trainbalanced/
    metrics.jsonl
    checkpoints/best.pt
    eval_test100_seed123_predicted_best/
      export_summary.json
      dtw_mpjpe_default.json
      dtw_mpjpe_default.csv
      dtw_mpjpe_pa.json
      dtw_mpjpe_pa.csv
      sample_0000.npz ... sample_0099.npz
      gt_0000.npz ... gt_0099.npz
```

## Appendix A. Epoch-50 `last.pt` evaluation

The epoch-50 checkpoint was evaluated separately with the identical test manifest,
predicted-length protocol, adapter, retrieval bank, and metric implementation. Its
export directory is:

```text
experiments/NIAF/retrieval_confidence_field/
  phoenix_retrieval_uncertainty_adaptive_knots_trainbalanced/
    eval_test100_seed123_predicted_last/
```

### A.1 Predicted duration

| Statistic | Epoch 10, best | Epoch 50, last |
|---|---:|---:|
| Mean predicted length | 81.20 | **78.08** |
| Mean absolute error | **12.52** | 13.88 |
| Median absolute error | **8.00** | 10.00 |
| Mean bias | +5.00 | **+1.88** |
| Maximum absolute error | 124 | 124 |

Epoch 50 has less mean overprediction but worse per-sample duration accuracy.

### A.2 Translated DTW-MPJPE

| Part | Epoch-50 scaffold | Epoch-50 field | Relative change | Field wins |
|---|---:|---:|---:|---:|
| Body | **4.04616** | 4.05852 | 0.31% worse | 45% |
| Left hand | 5.56203 | **5.55323** | 0.16% better | 51% |
| Right hand | **5.22257** | 5.22935 | 0.13% worse | 53% |
| Whole body | 11.00534 | **10.80538** | **1.82% better** | **65%** |

Only the whole-body improvement is significant: paired bootstrap 95% confidence
interval [0.09811, 0.30232], Wilcoxon p=0.000681. The isolated body and hand
changes are not significant. Normalized whole-body DTW improves by 1.33%.

### A.3 PA-DTW-MPJPE

| Part | Epoch-50 scaffold | Epoch-50 field | Relative change | Field wins |
|---|---:|---:|---:|---:|
| Body | 4.53782 | **4.52814** | 0.21% better | 51% |
| Left hand | **1.03030** | 1.03974 | 0.92% worse | 33% |
| Right hand | **1.14211** | 1.15351 | 1.00% worse | 24% |
| Whole body | 6.85030 | **6.67144** | **2.61% better** | **66%** |

The whole-body gain is significant (95% confidence interval [0.10081, 0.25861],
p=0.0000249), but both PA hand degradations are also significant. Normalized
whole-body PA-DTW improves by 2.39%.

### A.4 Best versus last

The epoch-50 full output has lower absolute mean DTW than epoch 10 on this subset:

| Full-output metric | Epoch 10, best | Epoch 50, last | Last relative change |
|---|---:|---:|---:|
| Translated body | 4.12479 | **4.05852** | 1.61% better |
| Translated left hand | 5.74274 | **5.55323** | 3.30% better |
| Translated right hand | 5.27300 | **5.22935** | 0.83% better |
| Translated whole body | 10.91885 | **10.80538** | 1.04% better |
| PA body | 4.61562 | **4.52814** | 1.90% better |
| PA left hand | 1.06114 | **1.03974** | 2.02% better |
| PA right hand | 1.17384 | **1.15351** | 1.73% better |
| PA whole body | 6.81637 | **6.67144** | 2.13% better |

This is not a clean residual-field comparison because each checkpoint predicts a
different duration and therefore receives a different adapter scaffold. The
epoch-50 scaffold itself is 1.20% to 4.24% lower in translated DTW and 1.65% to
2.79% lower in PA-DTW than the epoch-10 scaffold. In paired direct comparisons,
most epoch-50 versus epoch-10 full-output differences are not significant; only
translated left hand reaches an uncorrected p=0.0394.

The controlled field-versus-own-scaffold comparison still favors epoch 10:

- translated left hand: 1.13% significant gain at epoch 10 versus 0.16%
  nonsignificant gain at epoch 50;
- translated right hand: 1.22% significant gain versus 0.13% degradation;
- translated whole body: 1.97% gain versus 1.82% gain;
- PA whole body: 3.27% gain versus 2.61% gain;
- PA hand degradation: 0.57%/0.58% versus 0.92%/1.00%.

Epoch 50 also makes larger residual changes: its mean sequence rot6D residual RMS
is 0.05562, compared with 0.03487 at epoch 10, and its mean body gate increases
from 0.235 to 0.397. The larger correction does not produce better isolated hand
articulation or validation dynamics. Therefore, `best.pt` remains the preferable
residual-field checkpoint, while `last.pt` has a lower raw end-to-end mean on this
particular predicted-duration subset largely because its scaffold differs.
