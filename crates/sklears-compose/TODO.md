# TODO - v0.2.0

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.0 release. 782 tests passing (`cargo nextest run -p sklears-compose --all-features`, re-verified 2026-07-11 — matches the count already recorded below).

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

## OxiCUDA Migration (v0.2.0)
Status: real device discovery wired, remaining GPU-flavored types documented as scheduling metadata. Part of workspace Phase 4 (honesty pass); not blocking the scirs2-GPU removal goal.

- [x] (M) Wire GPU device discovery and counting to oxicuda-driver behind a new `gpu` feature (2026-07-06)
  - Added `gpu = ["sklears-core/gpu_support"]` to `Cargo.toml`.
  - `GpuResourceManager::allocate_gpu` (`src/resource_management/gpu_manager.rs`) now has an `#[cfg(feature = "gpu")]` impl that enumerates real devices via `sklears_core::gpu::GpuUtils::device_count`/`device_properties` (oxicuda-driver-backed); the `#[cfg(not(feature = "gpu"))]` impl keeps returning an honest empty `GpuAllocation` for default builds.
  - `SystemResources::get_gpu_count` (`src/comprehensive_benchmarking/execution_engine.rs`) now has a `gpu`-feature-gated variant returning the real oxicuda device count and a default variant returning `Ok(0)`.
  - `execution::resources::detect_gpu_info` similarly gained a `gpu`-feature-gated real-device-enumeration path (previously an unconditional empty-`Vec` placeholder) alongside an honest empty-list default path; `GpuMetrics`/`collect_gpu_metrics` stay empty always (documented) since no oxicuda API here exposes live utilization/temperature sampling yet.
  - Reworded `src/execution_metrics.rs` GPU-usage/GPU-memory placeholders to explain there is no oxicuda live-sampling API for these two fields yet (as opposed to the now-real device count/discovery), rather than naming "CUDA/OpenCL APIs" generically.
  - Verified: `cargo check -p sklears-compose` and `--features gpu` both pass warning-free; `cargo clippy -p sklears-compose --features gpu` clean; relevant `execution`/`resource_management` nextest suites (98 + 5 tests) pass in both default and `gpu`-feature builds.
- [x] (S) Document `GpuExecutionStrategy` and GPU telemetry fields as scheduling metadata, not GPU compute (2026-07-06)
  - `src/execution_strategies.rs`: `GpuExecutionStrategy`, `GpuContext`, `GpuDevice`, `GpuKernel` all gained doc comments stating they are scheduler-input descriptors, not live device handles/measurements.
  - `gpu_utilization`/`gpu_usage`/`gpu_temperature` `Option`/plain fields in `src/cv_pipelines/metrics_statistics.rs` (`PerformanceMetrics`, `ResourceUtilization`, `MetricsSummary`, `ThermalMetrics`), `src/resource_context/types.rs` (`ResourceUsage`, `ResourceUtilization`), and `src/execution/resources.rs` (`GpuInfo::utilization`, `GpuMetrics`) now document that they are scheduling metadata / not live-sampled readings.
  - Backing `GpuExecutionStrategy` with oxicuda-launch kernels remains a separate future (L) item, not required for scirs2 removal.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples
- Blanket `FitCV` impl for non-pipeline estimators (currently only `Pipeline<Untrained>` covered)
- `pipeline_visualization`: real graph node/edge extraction, an actual SVG/PNG/HTML rendering backend, per-node metrics, and interactive output — tracked in detail in the [workspace TODO](../../TODO.md).
- [ ] `ColumnTransformer` (`src/column_transformer.rs`): the builder's `.transformer(name, columns)`
  only supports name-based dispatch to `"passthrough"`/`"drop"` (`build_transformer`); any other
  name silently falls back to a passthrough transformer at `fit()` time. `add_transformer_step`
  takes a real `Box<dyn PipelineStep>` parameter but currently discards it (only `name`/`columns`
  are stored) — the error message pointing callers at `add_transformer_step` as the escape hatch
  ("For other transformers use `add_transformer_step` and supply the concrete `Box<dyn
  PipelineStep>` directly") is therefore not actually honored yet. Needs the fitted-transformer
  list to be wired to the caller-supplied transformer instances instead of the name-keyed lookup.
  Verified 2026-07-11 (`fit()` in `column_transformer.rs` calls `build_transformer(name)`, never
  consulting any transformer instance the caller passed in). `FeatureUnion::transformer(name, Box<dyn
  PipelineStep>)` does not have this bug — it genuinely stores and uses the supplied transformer.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
