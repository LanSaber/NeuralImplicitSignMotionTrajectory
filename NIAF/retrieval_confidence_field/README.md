# Retrieval-Confidence Field

This package owns the retrieval-confidence adaptive neural sign fields. It contains
the fixed multi-scale baseline, the uncertainty-adaptive knot successor, and the
hierarchical segmental rollout field.

Shared infrastructure remains in `NIAF/continuous_sign_field`:

- PHOENIX dataset and collation
- adapter scaffold construction and cache format
- normalized time grids
- SMPL-X FK losses and metrics
- text encoding helpers

Canonical commands:

```bash
sbatch scripts/NIAF/cache_retrieval_confidence_scaffolds_sbatch.sh
sbatch scripts/NIAF/train_retrieval_confidence_field_sbatch.sh
sbatch scripts/NIAF/train_retrieval_uncertainty_adaptive_knots_sbatch.sh
sbatch scripts/NIAF/train_retrieval_uncertainty_segmental_sbatch.sh
```

Method documents:

- `docs/NIAF/RetrievalConfidence/retrieval_confidence_adaptive_neural_sign_field.md`
- `docs/NIAF/RetrievalConfidence/retrieval_confidence_adaptive_test_results.md`
- `docs/NIAF/RetrievalConfidence/retrieval_uncertainty_adaptive_knot_field.md`
- `docs/NIAF/RetrievalConfidence/hierarchical_segmental_niaf.md`

Legacy Python entry points under `NIAF.continuous_sign_field` forward to this package for compatibility. New code should import from `NIAF.retrieval_confidence_field`.
