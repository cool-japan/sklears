# TODO - v0.2.0

## Current Status
This crate is part of the sklears v0.2.0 release.

## Completed in v0.1.1
- [x] Re-enable `cross_validation` — HRTB eliminated via `FitCV`/`PredictCV` adapter traits;
  all 9 cross_validation tests pass, 0 clippy warnings. Achieved by:
  - Introducing `FitCV` trait with owned-array signature (no lifetime params, no HRTB)
  - Implementing `FitCV` for `Pipeline<Untrained>` in `pipeline.rs` using concrete local lifetimes
  - Implementing `PredictCV` for `Pipeline<PipelineTrained>` in `pipeline.rs`
  - Adding `Clone` impls for `Pipeline<Untrained>` and `Pipeline<PipelineTrained>` using
    `clone_step()` and `clone_predictor()` trait methods
  - Updating all CV function signatures to use `FitCV`/`PredictCV` bounds instead of HRTB

## Completed (2026-07-04 session)
- [x] `model_fusion`: base models now genuinely trained via real `fit()` calls (previously forwarded unfitted models unchanged, silently faking training). `FusionStrategy::WeightedLinear` now solves a real OLS/ridge-regularized least-squares problem for fusion weights (previously hard-coded uniform weights while claiming a least-squares solve); `FusionStrategy::NeuralNetwork` now trains a small MLP via real backprop/gradient descent (verified to beat a mean-prediction baseline). Every other `FusionStrategy` variant now honestly returns `SklearsError::NotImplemented` instead of fabricating a "trained" result.
- [x] `hierarchical_composition`: every level now genuinely trains (previously one fake trainer shared by all strategies, `model.clone() // Simplified`, that never really fit anything). `HierarchicalStrategy::Stacked` now builds meta-features via genuine out-of-fold k-fold cross-validation (previously returned all-zero meta-features); `Cascaded` trains each level as an independent, self-sufficient predictor on the full training data.
- [x] New `stacking` module — real k-fold out-of-fold stacking ensemble (`StackingEnsemble`) with a pluggable meta-learner (defaults to OLS), exposing `oof_meta_features()` for leakage verification.
- [x] `pipeline_visualization`: `DefaultRenderingEngine::render()` and node/edge extraction (`extract_nodes`/`extract_edges`) now return honest `Err(SklearsError::NotImplemented(_))` instead of silently returning fake/empty output (previously a hardcoded empty `<svg></svg>` and always-empty node/edge vectors regardless of the pipeline). The full feature — real graph extraction, rendering, metrics, interactive output — is still **not implemented**; see the workspace TODO entry below before relying on this module.
- 782 tests passing.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples
- Blanket `FitCV` impl for non-pipeline estimators (currently only `Pipeline<Untrained>` covered)
- `pipeline_visualization`: real graph node/edge extraction, an actual SVG/PNG/HTML rendering backend, per-node metrics, and interactive output — tracked in detail in the [workspace TODO](../../TODO.md).

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
