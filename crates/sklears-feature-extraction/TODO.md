# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Completed (2026-07-04 session)
- [x] `image::simd_accelerated`: 9 public functions carried "SIMD-accelerated"/"vectorized" doc comments (some claiming specific speedups up to "8.7x") that didn't match their bodies — e.g. `simd_array_subtraction` was literally `Ok(a - b)` (an ndarray operator, not SIMD), and `simd_compute_kurtosis` (claimed "6.2x speedup") simply delegated to a scalar fallback. All are now real vector code via `scirs2_core::simd_ops::SimdUnifiedOps`, with new numerical-equivalence regression tests (1e-9 tolerance) guarding against a silent vectorization correctness bug.
- 416 tests passing.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
