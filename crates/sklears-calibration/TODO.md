# TODO - v0.2.0

## Current Status (updated 2026-07-14)
This crate is part of the sklears v0.2.0 release. 406 tests passing (`cargo nextest run -p sklears-calibration --all-features`).

## Fixed in 0.2.0 (2026-07-14)
- [x] `GpuTemperatureScalingCalibrator`: the device-accelerated prediction path previously ran the
  wrapper's native f64 logits through a device sigmoid kernel that only exists in f32 PTX (built
  from the `ex2.approx` special-function unit, which has no faithful f64 form in the oxicuda
  stack) — a latent precision mismatch. The device fast path is now taken only under an explicit
  `use_mixed_precision` opt-in (`GpuCalibrationConfig::use_mixed_precision`, default `false`) and
  runs genuine f32 kernels (`elementwise::scale`/`elementwise::sigmoid`); without that opt-in — or
  with no device/small batch — prediction correctly falls back to the exact CPU f64 path. This
  supersedes the looser framing of the same code path in the 2026-07-06 entry below (which did not
  yet require the explicit opt-in). Verified in `src/gpu_calibration.rs`
  (`GpuTemperatureScalingCalibrator::try_gpu_predict`/`GpuCalibrationConfig`).

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: stub-gpu — `src/gpu_calibration.rs` is a CPU-delegating shim; its module doc (lines 8-9) already anticipates a real backend behind a feature gate. Wire it to the oxicuda-backed `sklears_core::gpu` module (Wave A2) so GPU claims are honest. Part of workspace Phase 4 (parallelizable per crate, does not block the 0.2.0 scirs2-GPU excision goal).

- [x] (M) Slot oxicuda in behind a `gpu` feature as the real backend for the `gpu_calibration` wrappers:
  - [x] Added `gpu = ["dep:oxicuda-blas", "dep:oxicuda-memory", "sklears-core/gpu_support"]` to `Cargo.toml` (plus `oxicuda-blas`/`oxicuda-memory` as optional workspace deps); `src/gpu_calibration.rs` cfg-gates the real backend internally rather than needing a separate re-export in `src/lib.rs` (the module itself is unconditional; only its internals branch on `gpu`).
  - [x] `GpuUtils::init_devices` / `get_device` / `get_best_device` now route through `sklears_core::gpu::GpuBackend::detect`/`device_id` under the `gpu` feature (real oxicuda-driver device enumeration); without the feature (or with it but no device detected) they still honestly report no device.
  - [x] `get_memory_stats` now queries live device memory via `GpuBackend::memory_info` (`cuMemGetInfo`) under the `gpu` feature when a device is present, falling back to the `/proc/meminfo` host-RAM path (or zeros) otherwise.
  - [x] `get_utilization` (deferred 2026-07-06: oxicuda-driver 0.4.0 wraps the CUDA *driver* API only; SM/compute-occupancy sampling is an NVML API (`nvmlDeviceGetUtilizationRates`) with no oxicuda-driver binding, so there is no real occupancy figure available. Left honestly at `0.0` with a doc comment explaining why, rather than fabricating a number or silently redefining "utilization" as device-memory occupancy.).
  - [x] `GpuTemperatureScalingCalibrator::predict_proba` now takes a real device fast path under `gpu`: `sigmoid(logits / T)` runs as two genuine `oxicuda-blas` kernels (`elementwise::scale`, `elementwise::sigmoid`) once a device is detected and the batch is at/above `gpu_threshold`; verified bit-for-bit equivalent (within 1e-9) to the CPU path via `test_gpu_temperature_wrapper_matches_plain_cpu_calibrator`. Fitting (grid + line search over a handful of scalar temperature candidates) stays CPU-only as it isn't a GPU-shaped workload.
  - [x] Default (non-`gpu`) build unchanged: `IsotonicCalibrator` / `TemperatureScalingCalibrator` wrappers continue to delegate to CPU paths; `cargo check -p sklears-calibration` and `--features gpu` both pass warning-free, and all 402 tests (8 gpu_calibration-specific) pass in both configurations on this GPU-less macOS host.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
