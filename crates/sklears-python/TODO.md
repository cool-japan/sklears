# TODO - sklears-python

## Current Status
This crate provides PyO3-based Python bindings for the sklears workspace.

## Recently Completed
- [x] Enabled the `StandardScaler` / `MinMaxScaler` / `LabelEncoder` Python bindings — the
      transformers themselves were already fully implemented but their `m.add_class::<...>()`
      registrations were left out of the `#[pymodule]` init function, so they were unreachable
      from Python.
- [x] Registered `get_hardware_info()` and `benchmark_basic_operations()` as Python functions —
      both were implemented but never wired into the `#[pymodule]` init function. Also fixed a bug
      in `get_hardware_info()` where the `num_cpus` entry collapsed the real CPU core count into a
      boolean (`num_cpus::get() > 1`); it now reports the actual core count.
- [x] `KMeans` is now genuinely wired to `sklears_clustering::KMeans` (K-means++ init). Previously
      `fit()` silently discarded its input and `predict()` always returned `[0]`.
- [x] `DBSCAN` is now genuinely wired to `sklears_clustering::DBSCAN`. Since the underlying
      algorithm has no `predict()` for unseen data (matching scikit-learn's own transductive
      behavior), it exposes `fit_predict()` + `labels_` rather than separate `fit()`/`predict()`.
- [x] `train_test_split()` is now genuinely wired to `sklears_model_selection::train_test_split`.
      Previously it was a stub that always returned 1x1 zero arrays regardless of input. Known
      limitation: the underlying split always shuffles and does not yet support scikit-learn's
      `train_size`/`stratify` parameters (see "Remaining" below).
- [x] `KFold` is now genuinely wired to `sklears_model_selection::{KFold, CrossValidator}`, with
      real `shuffle`/`random_state` support. Previously `split()` always returned one hardcoded
      fold (`[(vec![0], vec![1])]`) regardless of `n_splits` or input size.
- [x] `__version__` now tracks `env!("CARGO_PKG_VERSION")` instead of a hardcoded stale string.
- [x] Reduced `unwrap()` calls in `src/metrics/classification.rs` (array-contiguity checks in
      `accuracy_score`/`precision_score`/`recall_score`/`f1_score`/`confusion_matrix`/
      `classification_report` now return a `PyValueError` instead of potentially panicking), in
      line with the workspace no-`unwrap()` policy.
- [x] Fixed `examples/python_demo.py` and `examples/comprehensive_benchmarks.py`: the demo now
      casts labels to `float64` before calling `train_test_split` (required by its exact-dtype
      `PyReadonlyArray1<f64>` signature), and the benchmark harness now tracks fit/predict
      success explicitly instead of via `result is not None` — sklears' `.fit()` returns `None`
      on success (unlike scikit-learn's `.fit()`, which returns `self`), so the old check
      silently misclassified every successful sklears fit as a failure.

## Infrastructure
- [x] Fixed a crate-wide bug where pyo3's `extension-module` feature was baked into the shared
      workspace dependency (`pyo3 = { features = ["extension-module", ...] }` in the root
      `Cargo.toml`). That feature defers linking against libpython to the Python interpreter that
      `dlopen()`s the built `cdylib`, which is correct for the distributable wheel but breaks a
      standalone `cargo test`/`cargo nextest` binary at the link step (no Python process ever
      loads it), unconditionally, for every test in this crate. The workspace dependency no longer
      enables the feature; `sklears-python/pyproject.toml`'s `[tool.maturin] features =
      ["pyo3/extension-module"]` already activates it independently for real `maturin`
      build/develop invocations, so the wheel build is unaffected. Test coverage that needs
      `#[pymethods]` logic is exercised through `Python`-free `*_core` helper functions (see
      `clustering.rs`, `model_selection.rs`) since `Python::with_gil` still cannot be used from a
      plain `cargo test` binary. Unblocked 44 previously-dead property tests (22 proptest cases,
      each present in both the `tests/lib.rs` aggregate target and its own standalone integration
      binary) plus 11 new unit tests across `clustering.rs`/`model_selection.rs`.

## OxiCUDA Migration (v0.2.0)
Part of the workspace-wide scirs2-core GPU removal: all GPU claims must either route through
the oxicuda-backed `sklears_core::gpu` module (behind `gpu_support`) or honestly report absence.
This crate currently hardcodes GPU availability in its system-info reporting.

- [x] (S) Report real CUDA availability via oxicuda when the bindings are built with GPU
      support. `src/utils.rs` previously hardcoded `cuda_available = false` /
      `opencl_available = false` in the system-info dict ("placeholder - would need actual
      detection"). Added a `gpu` feature in `Cargo.toml` forwarding to
      `sklears-core/gpu_support`; behind it, `cuda_available` is now set from sklears-core's
      oxicuda-driver detection (`sklears_core::gpu::GpuUtils::is_gpu_available()`). The
      hardcoded `false` remains for the default Pure-Rust build (no `gpu` feature), and
      `opencl_available` always stays honestly `false` since OpenCL is not supported by the
      oxicuda stack (CUDA-only). Verified on this macOS host (no GPU): both `cargo check -p
      sklears-python` and `--features gpu` are warning-free, and `GpuUtils::is_gpu_available()`
      honestly reports `false` (all 55 tests pass in both configurations).
      Files: `src/utils.rs`, `Cargo.toml`

## Remaining / Future Enhancements
- Tree-based models (`RandomForestClassifier`, `DecisionTreeClassifier`, etc.) are implemented in
  `src/tree.rs` but still commented out of the `#[pymodule]` init function ("Temporarily disabled
  to test ensemble" in `src/lib.rs`) — same class of bug as the preprocessing classes had; needs
  re-enabling and verification.
- `train_test_split`: add `train_size` and `stratify` support, and a way to opt out of shuffling
  (the underlying `sklears_model_selection::train_test_split` always shuffles).
- `StratifiedKFold` and `cross_val_score` are not implemented yet.
- `LabelEncoder.fit()`/`transform()` currently require a Python list of strings (`Vec<String>`);
  there's no direct path for numpy numeric/object arrays without converting to a list first.
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
