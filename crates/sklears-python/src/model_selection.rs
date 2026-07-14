//! Python bindings for model selection utilities
//!
//! This module provides Python bindings for sklears model selection,
//! offering scikit-learn compatible cross-validation and data splitting utilities.

use crate::linear::common::{
    core_array1_to_py, core_array2_to_py, pyarray_to_core_array1, pyarray_to_core_array2,
    PyValueError,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_model_selection::{train_test_split as core_train_test_split, CrossValidator, KFold};

/// Train-test split result: (X_train, X_test, y_train, y_test)
type TrainTestSplitResult = (
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
);

/// Train-test split result using plain `ndarray` types, i.e. the
/// Python-free counterpart of [`TrainTestSplitResult`].
type CoreTrainTestSplitResult = (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>);

/// Core (Python-free) train/test split logic, directly unit-testable
/// without a live Python interpreter -- this crate builds with pyo3's
/// `extension-module` feature (required so the compiled `cdylib` can be
/// imported from Python), which means `Python::with_gil` cannot be used
/// from a standalone `cargo test` binary.
///
/// Defaults `test_size` to `0.25` (scikit-learn's default) when `None`.
///
/// # Known limitations
/// The underlying `sklears_model_selection::train_test_split` always
/// shuffles before splitting and has no `train_size`/`stratify` support
/// (unlike scikit-learn's version). Extending the core splitting algorithm
/// to support those is out of scope for this fix.
///
/// `pub` (rather than crate-private) so that
/// `benches/core_helpers_benchmarks.rs` -- which compiles as a separate
/// crate -- can call it directly for the same reason; this is a minimal
/// exposed surface for benchmarking, not part of the stable Python-facing
/// API.
pub fn train_test_split_core(
    x: &Array2<f64>,
    y: &Array1<f64>,
    test_size: Option<f64>,
    random_state: Option<u64>,
) -> PyResult<CoreTrainTestSplitResult> {
    core_train_test_split(x, y, test_size.unwrap_or(0.25), random_state)
        .map_err(|e| PyValueError::new_err(format!("train_test_split failed: {e}")))
}

/// Split arrays into random train and test subsets.
///
/// Notes
/// -----
/// The underlying implementation always shuffles before splitting and does
/// not yet support scikit-learn's `train_size` or `stratify` parameters;
/// only `test_size` and `random_state` are real, working parameters.
#[pyfunction]
#[pyo3(signature = (x, y, test_size=None, random_state=None))]
pub fn train_test_split(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    test_size: Option<f64>,
    random_state: Option<u64>,
) -> PyResult<TrainTestSplitResult> {
    let x_arr = pyarray_to_core_array2(x)?;
    let y_arr = pyarray_to_core_array1(y)?;

    let (x_train, x_test, y_train, y_test) =
        train_test_split_core(&x_arr, &y_arr, test_size, random_state)?;

    Ok((
        core_array2_to_py(py, &x_train)?,
        core_array2_to_py(py, &x_test)?,
        core_array1_to_py(py, &y_train),
        core_array1_to_py(py, &y_test),
    ))
}

/// K-Fold cross-validator.
///
/// Splits data into `n_splits` consecutive (or shuffled) folds; each fold
/// is used once as the validation set while the remaining folds form the
/// training set.
#[pyclass(name = "KFold")]
pub struct PyKFold {
    inner: KFold,
}

impl PyKFold {
    /// Core split logic, directly unit-testable without a live Python
    /// interpreter (see `train_test_split_core` for why) and `pub`
    /// (exposed for benchmarking from `benches/`, not part of the stable
    /// Python-facing API).
    pub fn split_core(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let n_splits = self.inner.n_splits();
        if n_splits > n_samples {
            return Err(PyValueError::new_err(format!(
                "Cannot have number of splits n_splits={n_splits} greater than the \
                 number of samples n_samples={n_samples}"
            )));
        }
        Ok(self.inner.split(n_samples, None))
    }
}

#[pymethods]
impl PyKFold {
    /// `pub` in addition to being reachable from Python via `#[new]`, so
    /// `benches/core_helpers_benchmarks.rs` (a separate crate) can
    /// construct instances to call `split_core` on; not part of the
    /// stable Python-facing API surface.
    #[new]
    #[pyo3(signature = (n_splits=5, shuffle=false, random_state=None))]
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> PyResult<Self> {
        // The real `KFold::new` asserts `n_splits >= 2` and panics
        // otherwise; guard here so bad input raises a normal Python
        // `ValueError` instead of an uncatchable Rust panic.
        if n_splits < 2 {
            return Err(PyValueError::new_err(format!(
                "n_splits must be at least 2, got {n_splits}"
            )));
        }

        let mut inner = KFold::new(n_splits).shuffle(shuffle);
        if let Some(seed) = random_state {
            inner = inner.random_state(seed);
        }

        Ok(Self { inner })
    }

    fn get_n_splits(&self) -> usize {
        self.inner.n_splits()
    }

    /// Generate train/test indices for each fold.
    ///
    /// `y` is accepted (but ignored) purely for scikit-learn API
    /// compatibility: plain `KFold` does not need target labels to split
    /// (unlike `StratifiedKFold`), matching scikit-learn's own
    /// `KFold.split(X, y=None, groups=None)` signature. It is typed as an
    /// untyped `PyAny` rather than a specific numpy dtype so that passing
    /// e.g. integer class labels here (as scikit-learn users routinely do)
    /// never fails with a dtype mismatch.
    #[pyo3(signature = (x, y=None))]
    fn split(
        &self,
        x: PyReadonlyArray2<f64>,
        y: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let _ = y;
        let n_samples = x.shape()[0];
        self.split_core(n_samples)
    }

    fn __repr__(&self) -> String {
        format!("KFold(n_splits={})", self.inner.n_splits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_indexed_dataset(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
        let x_data: Vec<f64> = (0..n_samples * n_features).map(|v| v as f64).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), x_data)
            .expect("shape matches data length");
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());
        (x, y)
    }

    #[test]
    fn train_test_split_respects_requested_test_size() {
        let (x, y) = make_indexed_dataset(100, 2);

        let (x_train, x_test, y_train, y_test) =
            train_test_split_core(&x, &y, Some(0.3), Some(7)).expect("split should succeed");

        assert_eq!(x_test.nrows(), 30);
        assert_eq!(x_train.nrows(), 70);
        assert_eq!(y_test.len(), 30);
        assert_eq!(y_train.len(), 70);
    }

    #[test]
    fn train_test_split_train_and_test_are_disjoint_and_cover_everything() {
        let (x, y) = make_indexed_dataset(100, 2);

        let (_, _, y_train, y_test) =
            train_test_split_core(&x, &y, Some(0.3), Some(7)).expect("split should succeed");

        let mut seen: Vec<usize> = y_train
            .iter()
            .chain(y_test.iter())
            .map(|&v| v as usize)
            .collect();
        seen.sort_unstable();

        // If train/test overlapped, `seen` would contain duplicate indices
        // and this equality would fail; if any index were dropped instead,
        // the comparison against the full range would fail too.
        assert_eq!(seen, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn train_test_split_same_random_state_is_deterministic() {
        let (x, y) = make_indexed_dataset(50, 3);

        let (_, _, y_train1, y_test1) =
            train_test_split_core(&x, &y, Some(0.25), Some(123)).expect("split should succeed");
        let (_, _, y_train2, y_test2) =
            train_test_split_core(&x, &y, Some(0.25), Some(123)).expect("split should succeed");

        assert_eq!(y_train1, y_train2);
        assert_eq!(y_test1, y_test2);
    }

    #[test]
    fn train_test_split_defaults_test_size_to_a_quarter() {
        let (x, y) = make_indexed_dataset(40, 2);

        let (_, x_test, _, _) =
            train_test_split_core(&x, &y, None, Some(1)).expect("split should succeed");

        assert_eq!(x_test.nrows(), 10);
    }

    #[test]
    fn kfold_produces_n_splits_folds_covering_every_index_exactly_once() {
        let kfold = PyKFold::new(5, false, None).expect("n_splits=5 is valid");
        let folds = kfold.split_core(100).expect("split should succeed");

        assert_eq!(folds.len(), 5);

        let mut all_test_indices: Vec<usize> = folds
            .iter()
            .flat_map(|(_, test)| test.iter().copied())
            .collect();
        all_test_indices.sort_unstable();
        assert_eq!(all_test_indices, (0..100).collect::<Vec<_>>());

        // Every fold's train/test split should also partition n_samples.
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn kfold_new_rejects_n_splits_below_two_with_value_error_not_panic() {
        assert!(PyKFold::new(1, false, None).is_err());
        assert!(PyKFold::new(0, false, None).is_err());
    }
}
