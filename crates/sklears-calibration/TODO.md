# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: stub-gpu — `src/gpu_calibration.rs` is a CPU-delegating shim; its module doc (lines 8-9) already anticipates a real backend behind a feature gate. Wire it to the oxicuda-backed `sklears_core::gpu` module (Wave A2) so GPU claims are honest. Part of workspace Phase 4 (parallelizable per crate, does not block the 0.2.0 scirs2-GPU excision goal).

- [ ] (M) Slot oxicuda in behind a `gpu` feature as the real backend for the `gpu_calibration` wrappers:
  - Add `gpu = ["sklears-core/gpu_support"]` to `Cargo.toml` and re-export/gate accordingly in `src/lib.rs`.
  - Implement `GpuUtils::init_devices` / `get_device` / `get_best_device` (currently always `None`, `src/gpu_calibration.rs:65-72`) via oxicuda-driver device enumeration.
  - Implement `get_memory_stats` via device memory queries, replacing the misleading `/proc/meminfo` host-RAM path (`src/gpu_calibration.rs:77-114`) when a device is present.
  - Implement `get_utilization` (`src/gpu_calibration.rs:117-119`, hardcoded `0.0`) from real device queries.
  - Accelerate temperature-scaling batched matrix work via oxicuda-blas under the `gpu` feature.
  - Keep the default (non-`gpu`) build unchanged: `IsotonicCalibrator` / `TemperatureScalingCalibrator` continue to delegate to CPU paths.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
