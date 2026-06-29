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

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples
- Blanket `FitCV` impl for non-pipeline estimators (currently only `Pipeline<Untrained>` covered)

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
