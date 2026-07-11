# TODO - v0.2.0

## Current Status (updated 2026-07-11)
This crate is part of the sklears v0.2.0 release. The compiled public API (`src/lib.rs`) currently
exposes only: `generators` (basic + adversarial/causal/domain_specific/experimental/manifold/
multimodal/privacy/simd/spatial/statistical/time_series/type_safe/performance submodules),
`memory_pool`, `parallel_rng`, `simd_gen`, `traits`, `validation`, `versioning`, and `viz`.
190 tests passing (`cargo nextest run -p sklears-datasets --all-features`).

## Known gap: disabled modules not wired into lib.rs

`MIGRATION_STATUS.md` (dated 2026-03-20) documents ~400K lines of previously-implemented code that
is present in `src/` but not declared as a `mod` anywhere reachable from the current `lib.rs`, so it
does not compile into the crate at all. Verified still true as of 2026-07-11 (`src/lib.rs` is 48
lines; none of these files are referenced by any `mod` declaration reachable from it):

- [ ] `src/loaders.rs` — classic dataset loaders (`load_iris`, `load_wine`, `load_digits`,
  `load_breast_cancer`, `load_diabetes`, `load_boston`, `load_california_housing`, `load_linnerud`,
  `load_mnist`, `load_fashion_mnist`, `load_cifar10`, `load_newsgroups`, `load_reuters`,
  `load_olivetti_faces`). Note: even if re-wired, every one of these is itself marked
  `#[deprecated]` and documented as returning **synthetic generated data**, not the real dataset —
  re-enabling the module alone would not restore real scikit-learn-compatible loaders.
- [ ] `src/format/` — CSV/Parquet/Arrow/HDF5/cloud-storage IO (`format_support`-equivalent; only
  declared in the dead `lib_backup.rs`).
- [ ] `src/benchmarks.rs`, `src/streaming.rs`, `src/memory.rs`, `src/zero_copy.rs`,
  `src/plugins.rs`, `src/composable.rs`, `src/config.rs`, `src/config_templates.rs`,
  `src/classification_regression.rs`, `src/graphs.rs`, `src/manifolds_spatial.rs`, `src/matrix.rs`,
  `src/missing_data.rs`, `src/samples.rs`, `src/specialized.rs`, `src/timeseries.rs`,
  `src/visualization.rs` (root-level), `src/domain_specific.rs` (root-level, distinct from
  `generators::domain_specific`), `src/distributions.rs` (root-level, distinct from
  `validation::distributions`) — all orphaned, not part of the compiled crate.
- [ ] `src/lib_backup.rs` / `src/lib_composable.rs` — alternate `lib.rs` drafts that declare many of
  the above modules; neither is the crate root, so neither is compiled either.
- The one active example, `examples/basic_demo.rs`, explicitly documents itself as showing
  "the basic dataset generation functions available in the current minimal implementation" —
  consistent with the above.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage (contingent on resolving the disabled-modules gap above)
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
