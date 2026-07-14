# TODO - v0.2.1

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.1 release. 426 tests passing (`cargo nextest run -p sklears-metrics --all-features`; previously recorded as 411).

## README accuracy pass (2026-07-11)
- [x] Verified README code examples against real source: fixed several fabricated module
  paths/function names/signatures (`vision`→`computer_vision`, `iou_score`→`iou_boxes`/`iou_masks`,
  `rouge_scores`→`rouge_n_score`/`rouge_l_score`, a fictional `timeseries` module→
  `regression::{mean_absolute_scaled_error, mean_directional_accuracy,
  symmetric_mean_absolute_percentage_error}`, `privacy_preserving_metrics`→
  `federated_learning::privacy_preserving_aggregation`, a fictional `calibration` module→
  `probabilistic_metrics::{reliability_diagram, expected_calibration_error}`, `gpu_accuracy`→
  `GpuMetricsContext::compute_metric(GpuMetricType::Accuracy, ...)`, `StreamingMetrics::update`→
  `add_samples`/`finalize`, and `MetricsBuilder::with_confidence_intervals` argument count/order).
  None of the underlying functionality was missing — only the documented paths/signatures were
  wrong; see `README.md` for the corrected examples.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: done (2026-07-06). The GPU acceleration layer now wires onto the
oxicuda-backed `sklears_core::gpu` module (Phase 4 of the workspace migration plan —
honesty pass).

- [x] (S) Fix gpu/cuda feature wiring and module gating — `Cargo.toml`, `src/lib.rs`
  - Changed `gpu = []` to `gpu = ["dep:oxicuda-blas", "dep:oxicuda-memory", "sklears-core/gpu_support"]`; `cuda = ["gpu"]` kept as a back-compat alias.
  - Regated `src/gpu_acceleration.rs` in `src/lib.rs` from `cfg(feature = "cuda")` to `cfg(feature = "gpu")`, so `full` (which enables `gpu` but not the `cuda` alias) now actually compiles the module.
  - Defaults stay GPU-free per Pure Rust policy; verified `cargo check -p sklears-metrics` (default), `--features gpu`, and `--features full` all compile warning-free.
- [x] (L) Rewrite `src/gpu_acceleration.rs` onto sklears-core gpu_support / oxicuda
  - Replaced the null-pointer `CudaStream`/`GpuBuffer`/`GpuMemoryPool` types and the hardcoded-`false` `is_cuda_available` with a real `sklears_core::gpu::GpuBackend` (oxicuda-driver `Context` + oxicuda-blas `BlasHandle`); detection via `GpuBackend::with_device_id`/`is_available`. Metric kernels upload/download via `oxicuda_memory::DeviceBuffer` directly (not `GpuArray`, whose `GpuMatrixOps` trait only covers matmul/add/mul/scale/transpose — not enough surface for elementwise sub/abs/cmp_eq or dot/nrm2/reductions).
  - Implemented on device: Accuracy (`elementwise::cmp_eq` + `reduction::mean`), MSE/MAE (`elementwise::sub` + `mul`/`abs_val` + `reduction::mean`), Euclidean distance (`elementwise::sub` + `level1::nrm2`), cosine distance (`level1::dot` + two `nrm2`s), `compute_distance_matrix` for both metrics via GEMM Gram-matrix expansion (`level3::gemm` for `X X^T` + `elementwise::mul` + `reduction::reduce_axis` for row squared-norms, combined into `dist2 = ||x_i||^2+||x_j||^2-2*gram[i,j]` on the host since the full matrix must be downloaded anyway), and `parallel_reduction` Sum/Mean/Max/Min via `oxicuda_blas::reduction::{sum,mean,max,min}`.
  - `GpuMetricsError::CudaError`/`UnsupportedReduction` now wrap real `oxicuda-blas`/`oxicuda-memory` errors instead of unconditional `GpuNotAvailable`. `get_memory_stats` reports real device memory via `GpuBackend::memory_info` (`cuMemGetInfo`) instead of `/proc/meminfo` host RAM. `GpuDeviceProperties` is now a type alias for `sklears_core::gpu::GpuDeviceProperties` (dropped four fields — `multiprocessor_count`, `max_threads_per_block`, `max_blocks_per_grid`, `warp_size` — that were previously hardcoded to `0` and never queried; downscoped rather than fabricated).
  - `supports_mixed_precision` now returns `true` honestly (`oxicuda-blas`'s `GpuFloat` trait genuinely implements both `f32`/`f64`, the same capability `sklears-svm`'s GPU kernels use), but this crate's own metric kernels still always compute in `f64` regardless of the flag — **(deferred 2026-07-06: threading an actual f32 compute path through `compute_metric`/`compute_distance_matrix` is a follow-up, not done in this pass)**.
  - Unimplemented `GpuMetricType` entries (confusion matrix, ROC-AUC, etc.) still return `UnsupportedMetric` with rustdoc saying so.
  - Also fixed a pre-existing cache-key bug in `generate_cache_key`: it previously hashed only `y_true.len()` (not the array contents), so distinct same-length inputs could collide on a stale cached value once caching was actually exercised (previously moot, since every path always returned `Err(GpuNotAvailable)`).
  - Removed `GpuMetricsConfig`'s `memory_pool_size`/`num_streams`/`block_size`/`grid_size` and `ParallelReductionConfig`'s `block_size`/`shared_memory_size` fields: none of them mapped to anything `oxicuda-blas` actually lets a caller configure (it manages its own kernel block sizing and a single stream per `BlasHandle`), so they were decorative dead configuration rather than real knobs.
  - Verified with `cargo test -p sklears-metrics --features gpu gpu_acceleration` (12/12 pass on this no-GPU macOS host, via the `DeviceNotAvailable`-style skip-on-`GpuNotAvailable` pattern from `sklears-svm`).

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
