# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

- [x] Make `DeepLearningConfig.device` honest: validate or wire to oxicuda (S) — `src/fluent_api.rs`
  - Fixed 2026-07-06: `device: String` (src/fluent_api.rs:174) was accepted but
    never read for dispatch, so `"cuda"` silently did nothing. Added
    `DeepLearningConfig::validate_device()` (case-insensitive, only `"cpu"`
    accepted) and wired it into `ImputationPipeline::new()` so
    `ImputationBuilder::build()` now returns `SklearsError::InvalidParameter`
    for any other value instead of ignoring it. Covered by
    `test_deep_learning_device_cpu_is_accepted`,
    `test_deep_learning_device_is_case_insensitive`, and
    `test_deep_learning_device_cuda_is_rejected` in `src/fluent_api.rs`.
  - Later: route `device == "cuda"` through `sklears-core/gpu_support` once a
    real oxicuda-backed deep-learning imputer exists (today `DeepLearning` is
    itself an unimplemented placeholder that falls back to mean imputation in
    `ImputationPipeline::fit_transform`'s wildcard match arm — out of scope
    for this device-honesty fix).

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
