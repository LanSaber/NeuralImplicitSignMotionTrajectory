# Current Pipeline Dependency Docs

The continuous trajectory field reuses a frozen adapter stack and the prepared
SMPL-X dataset format from `flow/`. Only documentation for those active
dependencies is retained here:

- `adapter/soke_content_style_adapter.md`: frozen content-style adapter used to
  produce the motion scaffold;
- `adapter/soke_soft_word_arranger.md`: SoftArranger retrieval and alignment used
  by that adapter; and
- `dataset/flow_training_dataset_format.md`: compact SMPL-X files and manifests
  consumed by the PHOENIX loaders.

Historical latent-flow training, ablation, and flow-only evaluation notes were
removed because they do not describe the current neural implicit trajectory
pipeline.
