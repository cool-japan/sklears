# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Completed This Session
- [x] Re-enable `deep_gp` module — DONE: deep Gaussian Process layers were already fully implemented, just gated behind a stale TODO; now compiled and exported (`DeepGaussianProcessRegressor`, `DeepGPLayer`, `DeepGPConfig`).
- [x] Add `convolution_processes` module — DONE: from-scratch Convolution Process / dependent multi-output GP implementation (Álvarez & Lawrence style); verified to collapse exactly to a standard single-output GP in the degenerate case and to demonstrably share information across correlated outputs.
- [x] 182 tests passing in this crate.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
