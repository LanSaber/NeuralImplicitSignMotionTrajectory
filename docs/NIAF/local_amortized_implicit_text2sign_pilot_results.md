# Adapter Scaffold + Local Amortized Implicit Field Pilot

Date: 2026-07-13

## 1. Objective

This experiment tests a deployable, non-oracle text-to-sign path:

```text
raw sentence text
  -> text-conditioned SoftArranger adapter scaffold
  -> local amortized implicit residual field
  -> predicted SMPL-X rot6D trajectory
```

The local field predicts a continuous residual over the adapter trajectory:

$$
\hat{x}(\tau) = s_{\mathrm{adapter}}(\tau)
  + D_\theta\!\left(\tau, s_{\mathrm{adapter}}(\tau), z(\tau), c_{\mathrm{text}}\right),
$$

where local codes (z(\tau)) are inferred from text and overlapping windows of the
adapter scaffold. No target pose, target gloss sequence, target duration, or test-time
gradient adaptation is supplied during generation. Ground-truth SMPL-X is used only
as the training target and for evaluation.

## 2. Implementation

- Model: `NIAF/continuous_sign_field/models/local_implicit.py`
- Trainer: `NIAF/continuous_sign_field/scripts/train_local_implicit_field.py`
- Scaffold cache: `NIAF/continuous_sign_field/scripts/cache_adapter_scaffolds.py`
- Exporter: `NIAF/continuous_sign_field/scripts/export_local_implicit_samples.py`
- Config: `NIAF/continuous_sign_field/configs/phoenix_local_amortized_adapter_text_pilot500_rawgt.yaml`
- Tests: `tests/test_niaf_local_implicit.py`

The model uses stride-8 local windows, 128-dimensional local codes, text cross-attention,
a temporal code Transformer, and a Fourier/SIREN residual decoder. Body, left hand,
right hand, and face residuals have separate learned gates. A text-only duration head
predicts the sequence length before scaffold construction.

Training includes rot6D, FK joint, hand, velocity, acceleration, path, residual,
local-code smoothness, duration, FK acceleration, and FK jerk terms. This first pilot
uses raw ground truth as its reconstruction target; it does not yet use the previously
fitted soft-reconstruction trajectories as teachers.

## 3. Setting

| Item | Value |
|---|---|
| Dataset | PHOENIX sentence SMPL-X |
| Adapter checkpoint | `epoch_0400.pt` supplied for this experiment |
| Text encoder | Frozen `flan-t5-base` |
| Training subset | 500 random training sentences, seed 1234 |
| Validation subset | 100 random validation sentences, seed 1234 |
| Training | 30 epochs, 7,500 optimizer steps |
| Test subset | 100 random test sentences, seed 123 |
| Test duration | Predicted from raw text |
| Best checkpoint | Epoch 30 |

Training completed in 1,644 seconds. The online W&B run is
[`whwuphtg`](https://wandb.ai/hh3443-new-york-university/soke-niaf-local-implicit/runs/whwuphtg).

## 4. Validation Result

Lower loss is better.

| Metric | Adapter scaffold | Local field | Relative change |
|---|---:|---:|---:|
| Endpoint loss | 14.33384 | **14.05828** | **1.92% better** |
| FK joint loss | 0.97325 | **0.95074** | **2.31% better** |
| Hand loss | 0.11370 | **0.11193** | **1.56% better** |
| rot6D loss | **0.66944** | 0.67033 | 0.13% worse |
| Hand path ratio | **0.73121** | 0.69788 | farther from 1.0 |

The field learns a nonzero correction (`residual RMS = 0.01768` on validation), but
the lower hand path ratio shows that the current regularization still smooths some motion.
Validation duration MAE is 13.90 frames.

## 5. Text-Only Test Result

The exporter first predicts duration from raw text and then runs the adapter and local
field at that predicted length. Across 100 samples, duration MAE is 12.64 frames; mean
ground-truth and predicted lengths are 76.20 and 78.04 frames.

### Translated DTW-MPJPE

Lower is better. The evaluator names the local-field output `flow` and the scaffold
`adapter_prior` in its JSON/CSV files.

| Part | Adapter scaffold | Local field | Relative change | Local wins |
|---|---:|---:|---:|---:|
| Body | 4.01597 | **4.01026** | **0.14% better** | 50% |
| Left hand | 5.70558 | **5.66233** | **0.76% better** | 67% |
| Right hand | 5.23455 | **5.15644** | **1.49% better** | 68% |
| Whole body | 10.89985 | **10.72162** | **1.64% better** | 73% |

### PA-DTW-MPJPE

| Part | Adapter scaffold | Local field | Relative change | Local wins |
|---|---:|---:|---:|---:|
| Body | 4.52343 | **4.51730** | **0.14% better** | 54% |
| Left hand | **1.03393** | 1.03852 | 0.44% worse | 27% |
| Right hand | **1.15329** | 1.16021 | 0.60% worse | 33% |
| Whole body | 6.85382 | **6.76536** | **1.29% better** | 67% |

The default hand and whole-body improvements indicate better wrist/global trajectory
placement. The slightly worse PA hand scores indicate that finger articulation itself
has not improved yet.

## 6. Split Caveat

The supplied adapter checkpoint and its runtime configuration use the
`all.balanced` PHOENIX word bank. That bank was built from train, validation, and test
word clips. The test generator does not read the target sentence motion, glosses, or
duration, but its lexical retrieval memory can contain word-level clips originating
from the test split. Therefore, this result validates the architecture and execution
path, but it is not a strict leakage-free test benchmark.

For a publication-quality comparison, retrain the adapter with a train-only balanced
word bank, regenerate adapter scaffolds, and train/evaluate the local field unchanged.

## 7. Conclusion

The pilot supports continuing with the local amortized residual formulation: it improves
the adapter baseline on whole-body DTW and on most test samples without sentence-level
ground-truth input. The gain is still small, and the next experiment should use a
train-only adapter, stronger hand-articulation supervision, and a lower smoothness/jerk
weight or a soft-reconstruction teacher to preserve hand path length.

## 8. Artifacts

- Checkpoint: `experiments/NIAF/continuous_sign_field/phoenix_local_amortized_adapter_text_pilot500_rawgt/checkpoints/best.pt`
- Training metrics: `experiments/NIAF/continuous_sign_field/phoenix_local_amortized_adapter_text_pilot500_rawgt/metrics.jsonl`
- Test exports: `visualize/NIAF/continuous_sign_field/phoenix_local_amortized_adapter_text_pilot500_rawgt_best_test100_seed123_predlen/`
- Default DTW result: `dtw_mpjpe_t2m_default_h2s_betas.json`
- PA-DTW result: `dtw_pa_mpjpe_t2m_h2s_betas.json`
