//! Python bindings for tree-based algorithms
//!
//! This module provides PyO3-based Python bindings for sklears tree algorithms,
//! including Decision Trees, Random Forest, and Extra Trees.

use crate::utils::{numpy_to_ndarray1, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict, Trained};
use sklears_tree::random_forest::RandomForestRegressor;
use sklears_tree::{DecisionTree, MaxFeatures, RandomForestClassifier, SplitCriterion};

/// Python wrapper for Decision Tree Classifier
#[pyclass(name = "DecisionTreeClassifier")]
pub struct PyDecisionTreeClassifier {
    inner: Option<DecisionTree>,
    trained: Option<DecisionTree<Trained>>,
}

#[pymethods]
impl PyDecisionTreeClassifier {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        criterion="gini",
        _splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        _min_weight_fraction_leaf=0.0,
        _max_features=None,
        random_state=None,
        _max_leaf_nodes=None,
        _min_impurity_decrease=0.0,
        _class_weight=None,
        _ccp_alpha=0.0
    ))]
    fn new(
        criterion: &str,
        _splitter: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        _min_weight_fraction_leaf: f64,
        _max_features: Option<&str>,
        random_state: Option<u64>,
        _max_leaf_nodes: Option<usize>,
        _min_impurity_decrease: f64,
        _class_weight: Option<&str>,
        _ccp_alpha: f64,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            "log_loss" => SplitCriterion::LogLoss,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let mut tree = DecisionTree::new()
            .criterion(split_criterion)
            .min_samples_split(min_samples_split)
            .min_samples_leaf(min_samples_leaf);

        if let Some(depth) = max_depth {
            tree = tree.max_depth(depth);
        }

        if let Some(seed) = random_state {
            tree = tree.random_state(Some(seed));
        }

        Ok(Self {
            inner: Some(tree),
            trained: None,
        })
    }

    /// Fit the decision tree classifier
    fn fit<'py>(
        &mut self,
        x: &Bound<'py, PyArray2<f64>>,
        y: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Some(importances) => Ok(importances.clone().into_pyarray(py).unbind()),
            None => Err(PyRuntimeError::new_err("Feature importances not available")),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "DecisionTreeClassifier(fitted=True)".to_string()
        } else {
            "DecisionTreeClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Decision Tree Regressor
#[pyclass(name = "DecisionTreeRegressor")]
pub struct PyDecisionTreeRegressor {
    inner: Option<DecisionTree>,
    trained: Option<DecisionTree<Trained>>,
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        criterion="squared_error",
        _splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        _min_weight_fraction_leaf=0.0,
        _max_features=None,
        random_state=None,
        _max_leaf_nodes=None,
        _min_impurity_decrease=0.0,
        _ccp_alpha=0.0
    ))]
    fn new(
        criterion: &str,
        _splitter: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        _min_weight_fraction_leaf: f64,
        _max_features: Option<&str>,
        random_state: Option<u64>,
        _max_leaf_nodes: Option<usize>,
        _min_impurity_decrease: f64,
        _ccp_alpha: f64,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "squared_error" | "mse" => SplitCriterion::MSE,
            "mae" | "absolute_error" => SplitCriterion::MAE,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let mut tree = DecisionTree::new()
            .criterion(split_criterion)
            .min_samples_split(min_samples_split)
            .min_samples_leaf(min_samples_leaf);

        if let Some(depth) = max_depth {
            tree = tree.max_depth(depth);
        }

        if let Some(seed) = random_state {
            tree = tree.random_state(Some(seed));
        }

        Ok(Self {
            inner: Some(tree),
            trained: None,
        })
    }

    /// Fit the decision tree regressor
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Some(importances) => Ok(importances.clone().into_pyarray(py).unbind()),
            None => Err(PyRuntimeError::new_err("Feature importances not available")),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "DecisionTreeRegressor(fitted=True)".to_string()
        } else {
            "DecisionTreeRegressor(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Random Forest Classifier
#[pyclass(name = "RandomForestClassifier")]
pub struct PyRandomForestClassifier {
    inner: Option<RandomForestClassifier>,
    trained: Option<RandomForestClassifier<Trained>>,
}

#[pymethods]
impl PyRandomForestClassifier {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        _min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        _max_leaf_nodes=None,
        _min_impurity_decrease=0.0,
        bootstrap=true,
        _oob_score=false,
        n_jobs=None,
        random_state=None,
        _verbose=0,
        _warm_start=false,
        _class_weight=None,
        _ccp_alpha=0.0,
        _max_samples=None
    ))]
    fn new(
        n_estimators: usize,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        _min_weight_fraction_leaf: f64,
        max_features: &str,
        _max_leaf_nodes: Option<usize>,
        _min_impurity_decrease: f64,
        bootstrap: bool,
        _oob_score: bool,
        n_jobs: Option<i32>,
        random_state: Option<u64>,
        _verbose: i32,
        _warm_start: bool,
        _class_weight: Option<&str>,
        _ccp_alpha: f64,
        _max_samples: Option<f64>,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            "log_loss" => SplitCriterion::LogLoss,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let max_features_strategy = match max_features {
            "auto" | "sqrt" => MaxFeatures::Sqrt,
            "log2" => MaxFeatures::Log2,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown max_features: {}",
                    max_features
                )))
            }
        };

        let mut forest = RandomForestClassifier::new()
            .n_estimators(n_estimators)
            .criterion(split_criterion)
            .min_samples_split(min_samples_split)
            .min_samples_leaf(min_samples_leaf)
            .max_features(max_features_strategy)
            .bootstrap(bootstrap);

        if let Some(depth) = max_depth {
            forest = forest.max_depth(depth);
        }

        if let Some(seed) = random_state {
            forest = forest.random_state(seed);
        }

        if let Some(jobs) = n_jobs {
            forest = forest.n_jobs(jobs);
        }

        Ok(Self {
            inner: Some(forest),
            trained: None,
        })
    }

    /// Fit the random forest classifier
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let y_int: Array1<i32> = y_array.mapv(|val| val as i32);

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_int) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<i32> =
            Predict::<Array2<f64>, Array1<i32>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        let predictions_f64: Vec<f64> = predictions.iter().map(|&v| v as f64).collect();
        Ok(PyArray1::from_vec(py, predictions_f64).unbind())
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Ok(importances) => Ok(importances.into_pyarray(py).unbind()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to compute feature importances: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "RandomForestClassifier(fitted=True)".to_string()
        } else {
            "RandomForestClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Random Forest Regressor
#[pyclass(name = "RandomForestRegressor")]
pub struct PyRandomForestRegressor {
    inner: Option<RandomForestRegressor>,
    trained: Option<RandomForestRegressor<Trained>>,
}

#[pymethods]
impl PyRandomForestRegressor {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        _min_weight_fraction_leaf=0.0,
        max_features=1.0,
        _max_leaf_nodes=None,
        _min_impurity_decrease=0.0,
        bootstrap=true,
        _oob_score=false,
        n_jobs=None,
        random_state=None,
        _verbose=0,
        _warm_start=false,
        _ccp_alpha=0.0,
        _max_samples=None
    ))]
    fn new(
        n_estimators: usize,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        _min_weight_fraction_leaf: f64,
        max_features: f64,
        _max_leaf_nodes: Option<usize>,
        _min_impurity_decrease: f64,
        bootstrap: bool,
        _oob_score: bool,
        n_jobs: Option<i32>,
        random_state: Option<u64>,
        _verbose: i32,
        _warm_start: bool,
        _ccp_alpha: f64,
        _max_samples: Option<f64>,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "squared_error" | "mse" => SplitCriterion::MSE,
            "mae" | "absolute_error" => SplitCriterion::MAE,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let max_features_strategy = if (max_features - 1.0).abs() < f64::EPSILON {
            MaxFeatures::All
        } else {
            MaxFeatures::Fraction(max_features)
        };

        let mut forest = RandomForestRegressor::new()
            .n_estimators(n_estimators)
            .criterion(split_criterion)
            .min_samples_split(min_samples_split)
            .min_samples_leaf(min_samples_leaf)
            .max_features(max_features_strategy)
            .bootstrap(bootstrap);

        if let Some(depth) = max_depth {
            forest = forest.max_depth(depth);
        }

        if let Some(seed) = random_state {
            forest = forest.random_state(seed);
        }

        if let Some(jobs) = n_jobs {
            forest = forest.n_jobs(jobs);
        }

        Ok(Self {
            inner: Some(forest),
            trained: None,
        })
    }

    /// Fit the random forest regressor
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Ok(importances) => Ok(importances.into_pyarray(py).unbind()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to compute feature importances: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "RandomForestRegressor(fitted=True)".to_string()
        } else {
            "RandomForestRegressor(fitted=False)".to_string()
        }
    }
}
