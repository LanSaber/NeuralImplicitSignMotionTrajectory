# Flow Matching Quantitative Evaluation

This document records the current quantitative evaluation protocol for flow
matching checkpoints and the recent 100-sequence results.

For translator-based semantic back-translation from generated flow poses to
German text, see `docs/flow/evaluation/flow_back_translation_evaluation.md`.

Run commands from the SOKE project root:

```bash
cd /media/cvpr/haomian/NeuralImplicitSignMotionTrajectory

export PYTHONNOUSERSITE=1
export PYTHONPATH=/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory
export SOKE_PY=/media/cvpr/haomian/python_envs/soke/bin/python
```

## Current Protocol

Use `flow.evaluate.dtw_mpjpe_t2m_default` for the main metric. It follows the
`mGPT/metrics/t2m.py` joint selection more closely than the older
`ndtw_smplx_keypoints.py` evaluator.

Evaluation settings used for the recent results:

| Setting | Value |
| --- | --- |
| Test subset | 100 random sequences from the corresponding test manifest |
| Random seed | `123` |
| Sequence length | Match each selected manifest row with `--match_manifest_lengths` |
| Negative word candidates | Disabled with `--no_negative_candidates` for adapter-residual checkpoints; not applicable to pure-flow checkpoints |
| Shape betas | Fixed H2S betas from `/media/cvpr/haomian/NeuralImplicitSignMotionTrajectory/mGPT/data/H2S.py` |
| Body joints | 12 upper-body joints: `12,16,17,18,19,20,21,59,58,57,56,55` |
| Hand joints | 42 total hand joints: 21 left hand + 21 right hand |
| Reported parts | `body`, `lhand`, `rhand`, `wholebody` |
| Main scalar | `dtw_mean` |
| Extra scalar | `ndtw_mean`, normalized by the DTW path length |

`DTW-MPJPE` means the default, non-Procrustes metric:

- body: mGPT-style translation alignment, then upper-body joint subset.
- hands: wrist-translation alignment for each hand.
- wholebody: upper body plus both hands, aligned by the first concatenated joint.

`DTW-PA-MPJPE` means Procrustes-aligned DTW-MPJPE:

- each frame-pair cost uses similarity Procrustes alignment.
- for body, the current implementation follows `mGPT/metrics/t2m.py`: align the
  full SMPL-X joint set first, then measure the selected upper-body joints.
- because of that full-joint PA step, body `DTW-PA-MPJPE` can be larger than
  body `DTW-MPJPE`.

## Evaluated Checkpoints

### Adapter-Residual Checkpoints

| Dataset | Checkpoint | Data dir | Word data dir | Word split | Condition field |
| --- | --- | --- | --- | --- | --- |
| Phoenix | `experiments/flow/phoenix_latent_adapter_residual_b64x4_all_balanced_online_retry3_ncclif/checkpoints/best.pt` | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx` | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc` | `all.balanced` | `text` |
| How2Sign | `experiments/flow/how2sign_latent_adapter_residual_signasl_all_b128x4_online_retry2/checkpoints/best.pt` | `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx` | `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_word_signasl` | `all` | `text` |
| CSL Daily | `experiments/flow/csl_daily_latent_adapter_residual_2xb64/checkpoints/best.pt` | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx` | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx_word_ctc` | `all` | `label_word`, falling back to `text` |

The CSL Daily checkpoint resolves `condition_field=label_word`, but the sentence
test manifest does not contain a `label_word` field. The sampler therefore falls
back to raw Chinese `text`, matching the current training/sampling fallback
logic. If a gloss-conditioned CSL evaluation is needed, rerun separately with
`--condition_field gloss` and record it as a different protocol.

### Text-Only Noise Checkpoints

These checkpoints are pure flow matching checkpoints with `source_mode=noise`.
They do not have adapter checkpoints, word-prior inputs, or `coarse_smplx`
outputs. The evaluator therefore reports `skipped_prior=100` and only `flow/*`
metrics.

| Dataset | Checkpoint | Data dir | Condition field |
| --- | --- | --- | --- |
| CSL Daily | `experiments/flow/csldaily_latent_text_only_noise_b128x2_online_retry1/checkpoints/best.pt` | `/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx` | `text` |
| Phoenix | `experiments/flow/phoenix_latent_text_only_noise_b256x2_online/checkpoints/best.pt` | `/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx` | `text` |
| How2Sign | `experiments/flow/how2sign_latent_text_only_noise_b256x2_online/checkpoints/best.pt` | `/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx` | `text` |

## Generate A Random-100 Manifest

Set dataset-specific paths first:

```bash
DATA_DIR=/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx
TEST_MANIFEST="$DATA_DIR/meta/manifest_test.jsonl"
SAMPLES_DIR=visualize/phoenix_latent_adapter_residual_all_balanced_retry3_best_test100_seed123_noneg_t2m_eval
SEED=123

mkdir -p "$SAMPLES_DIR"
```

Then sample 100 test rows:

```bash
TEST_MANIFEST="$TEST_MANIFEST" SAMPLES_DIR="$SAMPLES_DIR" SEED="$SEED" $SOKE_PY - <<'PY'
import json
import os
import random
from pathlib import Path

src = Path(os.environ["TEST_MANIFEST"])
out_dir = Path(os.environ["SAMPLES_DIR"])
seed = int(os.environ.get("SEED", "123"))

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
rng = random.Random(seed)
indices = rng.sample(range(len(rows)), 100)
selected = [rows[i] for i in indices]

manifest = out_dir / f"manifest_test_random100_seed{seed}.jsonl"
with manifest.open("w", encoding="utf-8") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "source_manifest": str(src),
    "output_manifest": str(manifest),
    "seed": seed,
    "source_count": len(rows),
    "sample_count": len(selected),
    "min_frames": min(int(row["num_frames"]) for row in selected),
    "max_frames": max(int(row["num_frames"]) for row in selected),
    "mean_frames": sum(int(row["num_frames"]) for row in selected) / len(selected),
    "indices": indices,
}
(out_dir / "sample_manifest_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps({k: summary[k] for k in ["source_count", "sample_count", "min_frames", "max_frames", "mean_frames"]}, indent=2))
PY
```

## Generate Predictions

Use the checkpoint and corresponding dataset paths. Passing `--data_dir` and
`--word_data_dir` is recommended when old checkpoints contain stale `/dev/shm`
paths.

```bash
CHECKPOINT=experiments/flow/phoenix_latent_adapter_residual_b64x4_all_balanced_online_retry3_ncclif/checkpoints/best.pt
DATA_DIR=/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx
WORD_DATA_DIR=/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word_ctc
SAMPLES_DIR=visualize/phoenix_latent_adapter_residual_all_balanced_retry3_best_test100_seed123_noneg_t2m_eval
MANIFEST="$SAMPLES_DIR/manifest_test_random100_seed123.jsonl"

$SOKE_PY -m flow.sample_text_conditional \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --word_data_dir "$WORD_DATA_DIR" \
  --manifest "$MANIFEST" \
  --num_prompts 100 \
  --match_manifest_lengths \
  --seed 123 \
  --device auto \
  --no_negative_candidates \
  --out_dir "$SAMPLES_DIR"
```

Expected files:

| File pattern | Meaning |
| --- | --- |
| `sample_*.npz` | Generated flow output. Contains `smplx`, `motion`, and usually `coarse_smplx`. |
| `gt_*.npz` | Ground-truth motion loaded from the dataset manifest. Contains `smplx` and `motion`. |

## Run Metrics

Default DTW-MPJPE:

```bash
$SOKE_PY -m flow.evaluate.dtw_mpjpe_t2m_default \
  --samples_dir "$SAMPLES_DIR" \
  --out_json "$SAMPLES_DIR/dtw_mpjpe_t2m_default_h2s_betas.json" \
  --out_csv "$SAMPLES_DIR/dtw_mpjpe_t2m_default_h2s_betas.csv" \
  --device cpu \
  --betas_mode h2s_fixed \
  --alignment_mode default \
  --parts body lhand rhand wholebody
```

Procrustes-aligned DTW-PA-MPJPE:

```bash
$SOKE_PY -m flow.evaluate.dtw_mpjpe_t2m_default \
  --samples_dir "$SAMPLES_DIR" \
  --out_json "$SAMPLES_DIR/dtw_pa_mpjpe_t2m_h2s_betas.json" \
  --out_csv "$SAMPLES_DIR/dtw_pa_mpjpe_t2m_h2s_betas.csv" \
  --device cpu \
  --betas_mode h2s_fixed \
  --alignment_mode pa \
  --parts body lhand rhand wholebody
```

For adapter-residual checkpoints, the output JSON contains both flow metrics and
adapter-prior baseline metrics:

| Prefix | Meaning |
| --- | --- |
| `flow/*` | Final flow output compared with GT. |
| `adapter_prior/*` | Coarse adapter prior, from `coarse_smplx`, compared with GT. |

For pure flow checkpoints, `sample_*.npz` files do not contain `coarse_smplx`, so
the evaluator skips adapter-prior rows and reports `skipped_prior=100`.

## Recent Adapter-Residual Flow Results

These are `dtw_mean` values. Lower is better.

### DTW-MPJPE

| Dataset | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Phoenix | 3.762903 | 5.290709 | 5.125202 | 10.573446 |
| How2Sign | 6.537785 | 8.929884 | 8.777009 | 18.424476 |
| CSL Daily | 6.460884 | 8.772336 | 8.255499 | 19.987975 |

### DTW-PA-MPJPE

| Dataset | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Phoenix | 4.295069 | 1.016887 | 1.130047 | 6.437334 |
| How2Sign | 7.007232 | 1.942267 | 2.076322 | 9.934184 |
| CSL Daily | 7.282794 | 1.419269 | 1.943996 | 9.924511 |

### Path-Normalized `ndtw_mean`

| Dataset | Metric | Body | LHand | RHand | Wholebody |
| --- | --- | ---: | ---: | ---: | ---: |
| Phoenix | DTW-MPJPE | 0.047264 | 0.057735 | 0.058070 | 0.130060 |
| Phoenix | DTW-PA-MPJPE | 0.053346 | 0.011298 | 0.012468 | 0.074658 |
| How2Sign | DTW-MPJPE | 0.060960 | 0.071509 | 0.071667 | 0.171673 |
| How2Sign | DTW-PA-MPJPE | 0.064883 | 0.015605 | 0.016785 | 0.082769 |
| CSL Daily | DTW-MPJPE | 0.055511 | 0.063130 | 0.065454 | 0.163832 |
| CSL Daily | DTW-PA-MPJPE | 0.063070 | 0.010270 | 0.014899 | 0.078995 |

## Adapter-Prior Baseline

These are `adapter_prior/*` `dtw_mean` values from the same JSON files.

### DTW-MPJPE

| Dataset | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Phoenix | 4.016800 | 5.860343 | 5.452655 | 11.281457 |
| How2Sign | 4.985624 | 8.071643 | 7.821709 | 13.870090 |
| CSL Daily | 6.539116 | 9.174479 | 8.288540 | 20.513151 |

### DTW-PA-MPJPE

| Dataset | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Phoenix | 4.558876 | 1.148458 | 1.230236 | 7.140536 |
| How2Sign | 5.495663 | 1.684972 | 1.786483 | 9.887035 |
| CSL Daily | 7.206164 | 1.399082 | 1.896487 | 9.918965 |

## CSL Daily Gloss-Conditioned Ablation

These ablations use the same CSL Daily seed-123 random-100 test manifest as the
main CSL Daily run, but force `--condition_field gloss`. This is an
oracle/gloss-conditioned ablation protocol, not the same as raw-text inference.

All runs use `--word_split all`, `--no_negative_candidates`, fixed H2S betas, and
checkpoint:

```text
experiments/flow/csl_daily_latent_adapter_residual_2xb64/checkpoints/best.pt
```

### DTW-MPJPE

| Ablation | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | 6.199517 | 7.460638 | 7.672053 | 17.799752 |
| Concat + adapter + flow | 6.505312 | 7.925778 | 8.037072 | 18.602105 |
| Soft arranger + adapter + flow | 6.172673 | 8.344274 | 8.239434 | 18.647644 |
| Soft arranger only + flow | 6.848242 | 9.697560 | 8.329158 | 22.328524 |

### DTW-PA-MPJPE

| Ablation | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | 6.896856 | 1.160611 | 1.768073 | 8.018505 |
| Concat + adapter + flow | 7.242357 | 1.262106 | 1.885471 | 8.727239 |
| Soft arranger + adapter + flow | 7.003438 | 1.326088 | 1.818410 | 9.763865 |
| Soft arranger only + flow | 7.815885 | 1.253761 | 1.913717 | 9.341911 |

### Path-Normalized `ndtw_mean`

| Ablation | Metric | Body | LHand | RHand | Wholebody |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | DTW-MPJPE | 0.053696 | 0.055932 | 0.061249 | 0.149012 |
| Raw concat residual + flow | DTW-PA-MPJPE | 0.059256 | 0.008993 | 0.014647 | 0.066378 |
| Concat + adapter + flow | DTW-MPJPE | 0.056894 | 0.060029 | 0.064630 | 0.156456 |
| Concat + adapter + flow | DTW-PA-MPJPE | 0.063150 | 0.009864 | 0.015713 | 0.072300 |
| Soft arranger + adapter + flow | DTW-MPJPE | 0.051476 | 0.060856 | 0.065041 | 0.150108 |
| Soft arranger + adapter + flow | DTW-PA-MPJPE | 0.059274 | 0.009713 | 0.014141 | 0.075806 |
| Soft arranger only + flow | DTW-MPJPE | 0.059721 | 0.071164 | 0.065537 | 0.186146 |
| Soft arranger only + flow | DTW-PA-MPJPE | 0.067978 | 0.009558 | 0.014994 | 0.078003 |

The JSONs also contain `adapter_prior/*` rows. For this ablation suite, read that
prefix as the coarse/source prior before flow, including for the raw concat
residual variant.

## How2Sign Text-Conditioned Ablation

These ablations use the same How2Sign seed-123 random-100 test manifest as the
main How2Sign run and force `--condition_field text`. This is the appropriate
condition field for the SignASL word-prior checkpoint because the test manifest
has no usable `gloss` entries for this split.

All runs use `--word_split all`, `--no_negative_candidates`, fixed H2S betas, and
checkpoint:

```text
experiments/flow/how2sign_latent_adapter_residual_signasl_all_b128x4_online_retry2/checkpoints/best.pt
```

The subset includes one 462-frame sequence, which the sampler capped to the
model maximum of 400 frames.

### DTW-MPJPE

| Ablation | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | 8.952571 | 10.592602 | 9.817921 | 26.420543 |
| Concat + adapter + flow | 9.198504 | 10.940287 | 9.855151 | 27.456154 |
| Soft arranger + adapter + flow | 6.537785 | 8.929884 | 8.777009 | 18.424476 |
| Soft arranger only + flow | 7.436477 | 10.339678 | 9.771170 | 23.648839 |

### DTW-PA-MPJPE

| Ablation | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | 8.543501 | 2.139205 | 2.430977 | 10.215384 |
| Concat + adapter + flow | 8.617959 | 2.112620 | 2.396821 | 10.158051 |
| Soft arranger + adapter + flow | 7.007232 | 1.942267 | 2.076322 | 9.934184 |
| Soft arranger only + flow | 7.623868 | 2.057511 | 2.086441 | 10.300506 |

### Path-Normalized `ndtw_mean`

| Ablation | Metric | Body | LHand | RHand | Wholebody |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | DTW-MPJPE | 0.076470 | 0.080487 | 0.077575 | 0.223581 |
| Raw concat residual + flow | DTW-PA-MPJPE | 0.074531 | 0.017385 | 0.019833 | 0.085454 |
| Concat + adapter + flow | DTW-MPJPE | 0.078479 | 0.082567 | 0.077225 | 0.231615 |
| Concat + adapter + flow | DTW-PA-MPJPE | 0.075416 | 0.017214 | 0.019707 | 0.085024 |
| Soft arranger + adapter + flow | DTW-MPJPE | 0.060960 | 0.071509 | 0.071667 | 0.171673 |
| Soft arranger + adapter + flow | DTW-PA-MPJPE | 0.064883 | 0.015605 | 0.016785 | 0.082769 |
| Soft arranger only + flow | DTW-MPJPE | 0.070607 | 0.091555 | 0.080915 | 0.224969 |
| Soft arranger only + flow | DTW-PA-MPJPE | 0.071038 | 0.016678 | 0.017177 | 0.086202 |

The metric JSONs for this ablation suite were written with
`--prior_key __skip_prior__`, so they contain only `flow/*` rows and report
`skipped_prior=100`. This does not change the flow metric values.

## Phoenix Gloss-Conditioned Ablation

These ablations use the same Phoenix seed-123 random-100 test manifest as the
main Phoenix run, but force `--condition_field gloss`. The `gloss` field has much
higher word-prior coverage for this checkpoint than raw `text`, so this is an
oracle/gloss-conditioned structural ablation rather than the raw-text inference
protocol.

All runs use `--word_split all.balanced`, `--no_negative_candidates`, fixed H2S
betas, and checkpoint:

```text
experiments/flow/phoenix_latent_adapter_residual_b64x4_all_balanced_online_retry3_ncclif/checkpoints/best.pt
```

### DTW-MPJPE

| Ablation | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | 4.018829 | 5.496028 | 4.993900 | 11.120770 |
| Concat + adapter + flow | 4.267050 | 5.547054 | 5.036461 | 11.503290 |
| Soft arranger + adapter + flow | 4.017944 | 5.745918 | 5.180745 | 11.386463 |
| Soft arranger only + flow | 4.007014 | 6.081220 | 5.288359 | 11.200061 |

### DTW-PA-MPJPE

| Ablation | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | 4.473294 | 1.044575 | 1.119184 | 6.150276 |
| Concat + adapter + flow | 4.784164 | 1.101030 | 1.274487 | 6.252452 |
| Soft arranger + adapter + flow | 4.489006 | 1.019926 | 1.192460 | 6.136673 |
| Soft arranger only + flow | 4.426797 | 1.135700 | 1.255772 | 6.274162 |

### Path-Normalized `ndtw_mean`

| Ablation | Metric | Body | LHand | RHand | Wholebody |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw concat residual + flow | DTW-MPJPE | 0.049840 | 0.057801 | 0.055981 | 0.131275 |
| Raw concat residual + flow | DTW-PA-MPJPE | 0.054921 | 0.011405 | 0.012284 | 0.071050 |
| Concat + adapter + flow | DTW-MPJPE | 0.052546 | 0.059001 | 0.055808 | 0.134863 |
| Concat + adapter + flow | DTW-PA-MPJPE | 0.058224 | 0.012069 | 0.013668 | 0.072694 |
| Soft arranger + adapter + flow | DTW-MPJPE | 0.050306 | 0.064121 | 0.059072 | 0.138500 |
| Soft arranger + adapter + flow | DTW-PA-MPJPE | 0.056290 | 0.011243 | 0.012932 | 0.072647 |
| Soft arranger only + flow | DTW-MPJPE | 0.052280 | 0.075522 | 0.064724 | 0.145805 |
| Soft arranger only + flow | DTW-PA-MPJPE | 0.057525 | 0.014074 | 0.015241 | 0.079718 |

The metric JSONs for this ablation suite were written with
`--prior_key __skip_prior__`, so they contain only `flow/*` rows and report
`skipped_prior=100`. This does not change the flow metric values.

## Recent Text-Only Noise Flow Results

These are pure-flow `source_mode=noise` checkpoint results. There is no
adapter-prior baseline for this group.

### DTW-MPJPE

| Dataset | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| CSL Daily | 6.416356 | 8.817539 | 8.570226 | 19.504741 |
| Phoenix | 3.939570 | 5.501277 | 5.079305 | 10.946931 |
| How2Sign | 6.178267 | 8.838400 | 8.545245 | 17.114922 |

### DTW-PA-MPJPE

| Dataset | Body | LHand | RHand | Wholebody |
| --- | ---: | ---: | ---: | ---: |
| CSL Daily | 7.188969 | 1.498566 | 1.929790 | 10.057995 |
| Phoenix | 4.422935 | 1.028857 | 1.175715 | 6.374540 |
| How2Sign | 6.807621 | 1.968151 | 2.083296 | 9.836296 |

### Path-Normalized `ndtw_mean`

| Dataset | Metric | Body | LHand | RHand | Wholebody |
| --- | --- | ---: | ---: | ---: | ---: |
| CSL Daily | DTW-MPJPE | 0.056200 | 0.067937 | 0.068803 | 0.163543 |
| CSL Daily | DTW-PA-MPJPE | 0.064233 | 0.011672 | 0.015005 | 0.080677 |
| Phoenix | DTW-MPJPE | 0.049535 | 0.062984 | 0.057312 | 0.133472 |
| Phoenix | DTW-PA-MPJPE | 0.054837 | 0.011359 | 0.012681 | 0.076125 |
| How2Sign | DTW-MPJPE | 0.055381 | 0.070139 | 0.070163 | 0.153744 |
| How2Sign | DTW-PA-MPJPE | 0.061576 | 0.016341 | 0.016846 | 0.081985 |

The How2Sign text-only noise subset includes one 462-frame sequence, which the
sampler capped to the model maximum of 400 frames.

## Text-Only Noise Latency

These latency runs use the same seed-123 random-100 test manifests as the
text-only noise metric runs above. The first 5 prompts are warmup and excluded
from the summary. Runs use CUDA-synchronized timers, `--profile_latency`, and
`--skip_save_outputs`, so `sample_*.npz` and `gt_*.npz` pose sequences are not
written. Model/checkpoint loading and text/word-index construction are startup
costs and are not included in per-prompt rows.

### Mean Latency, ms

| Dataset | Total | Text encoder | Flow sampler | VAE decode | Postprocess |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSL Daily | 336.837 | 3.670 | 329.259 | 1.736 | 2.050 |
| Phoenix | 353.784 | 5.165 | 345.853 | 1.454 | 1.194 |
| How2Sign | 364.803 | 4.891 | 353.796 | 3.370 | 2.633 |

### Median Latency, ms

| Dataset | Total | Text encoder | Flow sampler | VAE decode | Postprocess |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSL Daily | 335.358 | 3.899 | 328.584 | 1.566 | 0.845 |
| Phoenix | 351.624 | 5.113 | 344.216 | 1.535 | 0.736 |
| How2Sign | 349.196 | 4.787 | 341.560 | 1.517 | 0.642 |

### Latency Profile Details

| Dataset | Included prompts | Mean frames | Frame range | Mean latent frames | Latent frame range | Flow sampler share |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| CSL Daily | 95 | 105.095 | 44-240 | 26.684 | 11-60 | 97.75% |
| Phoenix | 95 | 75.863 | 22-170 | 19.337 | 6-43 | 97.76% |
| How2Sign | 95 | 113.379 | 4-400 | 28.653 | 1-100 | 96.98% |

Latency outputs:

| Dataset | Profile directory |
| --- | --- |
| CSL Daily | `visualize/csldaily_latent_text_only_noise_retry1_best_test100_seed123_latency_nosave` |
| Phoenix | `visualize/phoenix_latent_text_only_noise_best_test100_seed123_latency_nosave` |
| How2Sign | `visualize/how2sign_latent_text_only_noise_best_test100_seed123_latency_nosave` |

Each latency profile directory contains only `latency_profile.json` and
`latency_profile.csv` because `--skip_save_outputs` was enabled.

## Saved Result Locations

| Group | Dataset | Sample/result directory |
| --- | --- | --- |
| Adapter residual | Phoenix | `visualize/phoenix_latent_adapter_residual_all_balanced_retry3_best_test100_seed123_noneg_t2m_eval` |
| Adapter residual | How2Sign | `visualize/how2sign_latent_adapter_residual_signasl_retry2_best_test100_seed123_noneg_t2m_eval` |
| Adapter residual | CSL Daily | `visualize/csl_daily_latent_adapter_residual_2xb64_best_test100_seed123_noneg_t2m_eval` |
| CSL ablation | Raw concat residual + flow | `visualize/csl_daily_ablation_raw_concat_residual_flow_best_test100_seed123_gloss_t2m_eval` |
| CSL ablation | Concat + adapter + flow | `visualize/csl_daily_ablation_concat_adapter_flow_best_test100_seed123_gloss_t2m_eval` |
| CSL ablation | Soft arranger + adapter + flow | `visualize/csl_daily_ablation_soft_arranger_adapter_flow_best_test100_seed123_gloss_t2m_eval` |
| CSL ablation | Soft arranger only + flow | `visualize/csl_daily_ablation_soft_arranger_only_flow_best_test100_seed123_gloss_t2m_eval` |
| How2Sign ablation | Raw concat residual + flow | `visualize/how2sign_ablation_raw_concat_residual_flow_best_test100_seed123_text_t2m_eval` |
| How2Sign ablation | Concat + adapter + flow | `visualize/how2sign_ablation_concat_adapter_flow_best_test100_seed123_text_t2m_eval` |
| How2Sign ablation | Soft arranger + adapter + flow | `visualize/how2sign_ablation_soft_arranger_adapter_flow_best_test100_seed123_text_t2m_eval` |
| How2Sign ablation | Soft arranger only + flow | `visualize/how2sign_ablation_soft_arranger_only_flow_best_test100_seed123_text_t2m_eval` |
| Phoenix ablation | Raw concat residual + flow | `visualize/phoenix_ablation_raw_concat_residual_flow_best_test100_seed123_gloss_t2m_eval` |
| Phoenix ablation | Concat + adapter + flow | `visualize/phoenix_ablation_concat_adapter_flow_best_test100_seed123_gloss_t2m_eval` |
| Phoenix ablation | Soft arranger + adapter + flow | `visualize/phoenix_ablation_soft_arranger_adapter_flow_best_test100_seed123_gloss_t2m_eval` |
| Phoenix ablation | Soft arranger only + flow | `visualize/phoenix_ablation_soft_arranger_only_flow_best_test100_seed123_gloss_t2m_eval` |
| Text-only noise | CSL Daily | `visualize/csldaily_latent_text_only_noise_retry1_best_test100_seed123_t2m_eval` |
| Text-only noise | Phoenix | `visualize/phoenix_latent_text_only_noise_best_test100_seed123_t2m_eval` |
| Text-only noise | How2Sign | `visualize/how2sign_latent_text_only_noise_best_test100_seed123_t2m_eval` |

Each directory contains:

- `manifest_test_random100_seed123.jsonl`
- `sample_manifest_summary.json`
- `sample_*.npz`
- `gt_*.npz`
- `dtw_mpjpe_t2m_default_h2s_betas.json`
- `dtw_mpjpe_t2m_default_h2s_betas.csv`
- `dtw_pa_mpjpe_t2m_h2s_betas.json`
- `dtw_pa_mpjpe_t2m_h2s_betas.csv`

## Legacy Evaluator

`flow.evaluate.ndtw_smplx_keypoints` is still useful for older experiments, RMS
frame costs, and custom body/hands combinations. It is not the default metric for
the recent tables above.

Example legacy command:

```bash
$SOKE_PY -m flow.evaluate.ndtw_smplx_keypoints \
  --samples_dir "$SAMPLES_DIR" \
  --out_json "$SAMPLES_DIR/keypoint_ndtw_body_hands_mpjpe.json" \
  --out_csv "$SAMPLES_DIR/keypoint_ndtw_body_hands_mpjpe.csv" \
  --device cpu \
  --betas_mode h2s_fixed \
  --parts upper_body_hands \
  --frame_metric mpjpe
```
