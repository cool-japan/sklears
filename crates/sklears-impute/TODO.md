# TODO - v0.1.0

## Current Status
This crate is part of the sklears v0.1.0 initial release.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

## OxiCUDA Migration (v0.2.0)

Migration status: simulated-gpu. The `DeepLearningConfig.device` setting is
accepted but never used for dispatch, so GPU claims must either be made honest
or wired to the oxicuda-backed `sklears_core::gpu` module (Wave A2).

- [ ] Make `DeepLearningConfig.device` honest: validate or wire to oxicuda (S) — `src/fluent_api.rs`
  - `device: String` (src/fluent_api.rs:174, comment `"cpu", "cuda"`; default `"cpu"` at :761; setter at :801-802) is never read for dispatch — accepting `"cuda"` silently does nothing.
  - Minimum viable 0.2.0 fix: reject or warn on values other than `"cpu"` and document CPU-only status.
  - Later: route `device == "cuda"` through `sklears-core/gpu_support` once an oxicuda-backed deep-learning imputer exists.

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
