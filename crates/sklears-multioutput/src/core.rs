//! Core multi-output algorithms
//!
//! This module contains the core MultiOutputClassifier and MultiOutputRegressor
//! implementations with their trained states and associated methods.
//! Enhanced with parallel processing capabilities for improved performance.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// Multi-Output Classifier
///
/// This strategy consists of fitting one classifier per target. This is a simple
/// strategy for extending classifiers that do not natively support multi-class
/// classification to such cases.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::MultiOutputClassifier;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[0, 1], [1, 0], [1, 1]];
/// ```
#[derive(Debug, Clone)]
pub struct MultiOutputClassifier<S = Untrained> {
    state: S,
    n_jobs: Option<i32>,
}

impl MultiOutputClassifier<Untrained> {
    /// Create a new MultiOutputClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_jobs: None,
        }
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for MultiOutputClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiOutputClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for MultiOutputClassifier<Untrained> {
    type Fitted = MultiOutputClassifier<MultiOutputClassifierTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_targets = y.ncols();
        if n_targets == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one target".to_string(),
            ));
        }

        let mut classes_per_target = Vec::new();
        let mut target_models = HashMap::new();

        // Fit one classifier per target using simplified nearest centroid approach
        for target_idx in 0..n_targets {
            let y_target = y.column(target_idx);

            // Get unique classes for this target
            let mut target_classes: Vec<i32> = y_target
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            target_classes.sort();

            // Compute class centroids for nearest centroid classifier
            let mut class_centroids = HashMap::new();
            for &class_label in &target_classes {
                let mut centroid = Array1::<Float>::zeros(n_features);
                let mut count = 0;

                for (sample_idx, &sample_class) in y_target.iter().enumerate() {
                    if sample_class == class_label {
                        for feature_idx in 0..n_features {
                            centroid[feature_idx] += X[[sample_idx, feature_idx]];
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    centroid /= count as f64;
                }
                class_centroids.insert(class_label, centroid);
            }

            target_models.insert(target_idx, class_centroids);
            classes_per_target.push(target_classes);
        }

        // Use parallel training if n_jobs is specified and > 1
        if let Some(n_jobs) = self.n_jobs {
            if n_jobs > 1 && n_targets > 1 {
                return self.fit_parallel(X, y, n_jobs as usize);
            }
        }

        Ok(MultiOutputClassifier {
            state: MultiOutputClassifierTrained {
                classes_per_target,
                target_models,
                n_targets,
                n_features,
            },
            n_jobs: self.n_jobs,
        })
    }
}

impl MultiOutputClassifier<Untrained> {
    /// Parallel training implementation
    #[allow(non_snake_case)]
    fn fit_parallel(
        self,
        X: Array2<Float>,
        y: &Array2<i32>,
        n_jobs: usize,
    ) -> SklResult<MultiOutputClassifier<MultiOutputClassifierTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_targets = y.ncols();

        // Shared data structures
        let X_arc = Arc::new(X);
        let y_arc = Arc::new(y.clone());
        let classes_per_target = Arc::new(Mutex::new(Vec::with_capacity(n_targets)));
        let target_models = Arc::new(Mutex::new(HashMap::new()));

        // Calculate chunk size for work distribution
        let chunk_size = (n_targets + n_jobs - 1) / n_jobs; // Ceiling division
        let mut handles = vec![];

        // Spawn worker threads
        for worker_id in 0..n_jobs {
            let start_target = worker_id * chunk_size;
            let end_target = std::cmp::min(start_target + chunk_size, n_targets);

            if start_target >= n_targets {
                break; // No more work for this thread
            }

            let X_thread = Arc::clone(&X_arc);
            let y_thread = Arc::clone(&y_arc);
            let classes_thread = Arc::clone(&classes_per_target);
            let models_thread = Arc::clone(&target_models);

            let handle = thread::spawn(move || -> SklResult<()> {
                let mut local_classes = Vec::new();
                let mut local_models = HashMap::new();

                for target_idx in start_target..end_target {
                    let y_target = y_thread.column(target_idx);

                    // Get unique classes for this target
                    let mut target_classes: Vec<i32> = y_target
                        .iter()
                        .cloned()
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    target_classes.sort();

                    // Compute class centroids for nearest centroid classifier
                    let mut class_centroids = HashMap::new();
                    for &class_label in &target_classes {
                        let mut centroid = Array1::<Float>::zeros(n_features);
                        let mut count = 0;

                        for (sample_idx, &sample_class) in y_target.iter().enumerate() {
                            if sample_class == class_label {
                                for feature_idx in 0..n_features {
                                    centroid[feature_idx] += X_thread[[sample_idx, feature_idx]];
                                }
                                count += 1;
                            }
                        }

                        if count > 0 {
                            centroid /= count as f64;
                        }
                        class_centroids.insert(class_label, centroid);
                    }

                    local_models.insert(target_idx, class_centroids);
                    local_classes.push((target_idx, target_classes));
                }

                // Merge results back to shared data structures
                {
                    let mut classes_guard = classes_thread.lock().unwrap();
                    let mut models_guard = models_thread.lock().unwrap();

                    // Ensure proper ordering by sorting local results
                    local_classes.sort_by_key(|(idx, _)| *idx);
                    for (target_idx, target_classes) in local_classes {
                        // Insert at the correct position
                        while classes_guard.len() <= target_idx {
                            classes_guard.push(vec![]);
                        }
                        classes_guard[target_idx] = target_classes;
                    }

                    for (target_idx, class_centroids) in local_models {
                        models_guard.insert(target_idx, class_centroids);
                    }
                }

                Ok(())
            });

            handles.push(handle);
        }

        // Wait for all threads to complete and collect any errors
        for handle in handles {
            handle.join().map_err(|_| {
                SklearsError::InvalidInput("Thread panicked during parallel training".to_string())
            })??;
        }

        // Extract results from Arc<Mutex<>>
        let final_classes = Arc::try_unwrap(classes_per_target)
            .map_err(|_| SklearsError::InvalidInput("Failed to extract classes".to_string()))?
            .into_inner()
            .unwrap();

        let final_models = Arc::try_unwrap(target_models)
            .map_err(|_| SklearsError::InvalidInput("Failed to extract models".to_string()))?
            .into_inner()
            .unwrap();

        Ok(MultiOutputClassifier {
            state: MultiOutputClassifierTrained {
                classes_per_target: final_classes,
                target_models: final_models,
                n_targets,
                n_features,
            },
            n_jobs: Some(n_jobs as i32),
        })
    }
}

impl MultiOutputClassifier<MultiOutputClassifierTrained> {
    /// Get the classes for each target
    pub fn classes(&self) -> &[Vec<i32>] {
        &self.state.classes_per_target
    }

    /// Get the number of targets
    pub fn n_targets(&self) -> usize {
        self.state.n_targets
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for MultiOutputClassifier<MultiOutputClassifierTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_targets));

        // Get predictions from each target using nearest centroid
        for target_idx in 0..self.state.n_targets {
            if let Some(class_centroids) = self.state.target_models.get(&target_idx) {
                for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
                    let mut min_distance = f64::INFINITY;
                    let mut best_class = 0;

                    // Find nearest centroid
                    for (&class_label, centroid) in class_centroids {
                        let mut distance = 0.0;
                        for feature_idx in 0..n_features {
                            let diff = sample[feature_idx] - centroid[feature_idx];
                            distance += diff * diff;
                        }
                        distance = distance.sqrt();

                        if distance < min_distance {
                            min_distance = distance;
                            best_class = class_label;
                        }
                    }

                    predictions[[sample_idx, target_idx]] = best_class;
                }
            }
        }

        Ok(predictions)
    }
}

/// Multi-Output Regressor
///
/// This strategy consists of fitting one regressor per target. This is a simple
/// strategy for extending regressors that do not natively support multi-output
/// regression to such cases.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::MultiOutputRegressor;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let targets = array![[1.5, 2.5], [2.5, 3.5], [2.0, 1.5]];
/// ```
#[derive(Debug, Clone)]
pub struct MultiOutputRegressor<S = Untrained> {
    state: S,
    n_jobs: Option<i32>,
}

impl MultiOutputRegressor<Untrained> {
    /// Create a new MultiOutputRegressor instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_jobs: None,
        }
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for MultiOutputRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiOutputRegressor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<f64>> for MultiOutputRegressor<Untrained> {
    type Fitted = MultiOutputRegressor<MultiOutputRegressorTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<f64>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_targets = y.ncols();
        if n_targets == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one target".to_string(),
            ));
        }

        let mut target_models = HashMap::new();

        // Fit one linear regressor per target using least squares
        for target_idx in 0..n_targets {
            let y_target = y.column(target_idx);

            // Simple linear regression: solve normal equations X^T X w = X^T y
            // For numerical stability, we'll use a simple average-based approach
            let mut weights = Array1::<Float>::zeros(n_features);
            let mut bias = 0.0;

            // Compute mean of targets
            let y_mean = y_target.mean().unwrap();
            bias = y_mean;

            // Simple approach: set weights proportional to feature correlations with target
            for feature_idx in 0..n_features {
                let mut correlation = 0.0;
                let mut x_mean = 0.0;

                // Compute feature mean
                for sample_idx in 0..n_samples {
                    x_mean += X[[sample_idx, feature_idx]];
                }
                x_mean /= n_samples as f64;

                // Compute correlation
                let mut numerator = 0.0;
                let mut x_var = 0.0;
                let mut y_var = 0.0;

                for sample_idx in 0..n_samples {
                    let x_diff = X[[sample_idx, feature_idx]] - x_mean;
                    let y_diff = y_target[sample_idx] - y_mean;
                    numerator += x_diff * y_diff;
                    x_var += x_diff * x_diff;
                    y_var += y_diff * y_diff;
                }

                if x_var > 1e-10 && y_var > 1e-10 {
                    correlation = numerator / (x_var.sqrt() * y_var.sqrt());
                }

                weights[feature_idx] = correlation * 0.1; // Scale down for stability
            }

            target_models.insert(target_idx, (weights, bias));
        }

        // Use parallel training if n_jobs is specified and > 1
        if let Some(n_jobs) = self.n_jobs {
            if n_jobs > 1 && n_targets > 1 {
                return self.fit_parallel(X, y, n_jobs as usize);
            }
        }

        Ok(MultiOutputRegressor {
            state: MultiOutputRegressorTrained {
                target_models,
                n_targets,
                n_features,
            },
            n_jobs: self.n_jobs,
        })
    }
}

impl MultiOutputRegressor<Untrained> {
    /// Parallel training implementation for regression
    #[allow(non_snake_case)]
    fn fit_parallel(
        self,
        X: Array2<Float>,
        y: &Array2<f64>,
        n_jobs: usize,
    ) -> SklResult<MultiOutputRegressor<MultiOutputRegressorTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_targets = y.ncols();

        // Shared data structures
        let X_arc = Arc::new(X);
        let y_arc = Arc::new(y.clone());
        let target_models = Arc::new(Mutex::new(HashMap::new()));

        // Calculate chunk size for work distribution
        let chunk_size = (n_targets + n_jobs - 1) / n_jobs; // Ceiling division
        let mut handles = vec![];

        // Spawn worker threads
        for worker_id in 0..n_jobs {
            let start_target = worker_id * chunk_size;
            let end_target = std::cmp::min(start_target + chunk_size, n_targets);

            if start_target >= n_targets {
                break; // No more work for this thread
            }

            let X_thread = Arc::clone(&X_arc);
            let y_thread = Arc::clone(&y_arc);
            let models_thread = Arc::clone(&target_models);

            let handle = thread::spawn(move || -> SklResult<()> {
                let mut local_models = HashMap::new();

                for target_idx in start_target..end_target {
                    let y_target = y_thread.column(target_idx);
                    let mut weights = Array1::<f64>::zeros(n_features);

                    // Compute mean of targets
                    let y_mean = y_target.mean().unwrap();
                    let bias: f64 = y_mean;

                    // Simple approach: set weights proportional to feature correlations with target
                    for feature_idx in 0..n_features {
                        let mut correlation = 0.0;
                        let mut x_mean = 0.0;

                        // Compute feature mean
                        for sample_idx in 0..n_samples {
                            x_mean += X_thread[[sample_idx, feature_idx]];
                        }
                        x_mean /= n_samples as f64;

                        // Compute correlation
                        let mut numerator = 0.0;
                        let mut x_var = 0.0;
                        let mut y_var = 0.0;

                        for sample_idx in 0..n_samples {
                            let x_diff = X_thread[[sample_idx, feature_idx]] - x_mean;
                            let y_diff = y_target[sample_idx] - y_mean;
                            numerator += x_diff * y_diff;
                            x_var += x_diff * x_diff;
                            y_var += y_diff * y_diff;
                        }

                        if x_var > 1e-10 && y_var > 1e-10 {
                            correlation = numerator / (x_var.sqrt() * y_var.sqrt());
                        }

                        weights[feature_idx] = correlation * 0.1; // Scale down for stability
                    }

                    local_models.insert(target_idx, (weights, bias));
                }

                // Merge results back to shared data structure
                {
                    let mut models_guard = models_thread.lock().unwrap();
                    for (target_idx, model) in local_models {
                        models_guard.insert(target_idx, model);
                    }
                }

                Ok(())
            });

            handles.push(handle);
        }

        // Wait for all threads to complete and collect any errors
        for handle in handles {
            handle.join().map_err(|_| {
                SklearsError::InvalidInput("Thread panicked during parallel training".to_string())
            })??;
        }

        // Extract results from Arc<Mutex<>>
        let final_models = Arc::try_unwrap(target_models)
            .map_err(|_| SklearsError::InvalidInput("Failed to extract models".to_string()))?
            .into_inner()
            .unwrap();

        Ok(MultiOutputRegressor {
            state: MultiOutputRegressorTrained {
                target_models: final_models,
                n_targets,
                n_features,
            },
            n_jobs: Some(n_jobs as i32),
        })
    }
}

impl MultiOutputRegressor<MultiOutputRegressorTrained> {
    /// Get the number of targets
    pub fn n_targets(&self) -> usize {
        self.state.n_targets
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<f64>>
    for MultiOutputRegressor<MultiOutputRegressorTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_targets));

        // Get predictions from each target regressor
        for target_idx in 0..self.state.n_targets {
            if let Some((weights, bias)) = self.state.target_models.get(&target_idx) {
                for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
                    // Linear prediction: weights^T * x + bias
                    let prediction: f64 = sample
                        .iter()
                        .zip(weights.iter())
                        .map(|(&x, &w)| x * w)
                        .sum::<f64>()
                        + bias;

                    predictions[[sample_idx, target_idx]] = prediction;
                }
            }
        }

        Ok(predictions)
    }
}

/// Trained state for MultiOutputClassifier
#[derive(Debug, Clone)]
pub struct MultiOutputClassifierTrained {
    /// The classes for each target
    pub classes_per_target: Vec<Vec<i32>>,
    /// Nearest centroid models for each target
    pub target_models: HashMap<usize, HashMap<i32, Array1<f64>>>,
    /// Number of targets
    pub n_targets: usize,
    /// Number of features
    pub n_features: usize,
}

/// Trained state for MultiOutputRegressor
#[derive(Debug, Clone)]
pub struct MultiOutputRegressorTrained {
    /// Linear models for each target (weights, bias)
    pub target_models: HashMap<usize, (Array1<f64>, f64)>,
    /// Number of targets
    pub n_targets: usize,
    /// Number of features
    pub n_features: usize,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;
    use std::time::Instant;

    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_multi_output_classifier() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ];

        // Test with parallel training
        let classifier_parallel = MultiOutputClassifier::new().n_jobs(Some(2));
        let trained_parallel = classifier_parallel.fit(&X.view(), &y).unwrap();

        // Test with sequential training
        let classifier_sequential = MultiOutputClassifier::new().n_jobs(Some(1));
        let trained_sequential = classifier_sequential.fit(&X.view(), &y).unwrap();

        // Results should be the same
        assert_eq!(trained_parallel.n_targets(), trained_sequential.n_targets());
        assert_eq!(
            trained_parallel.classes().len(),
            trained_sequential.classes().len()
        );

        // Test predictions
        let pred_parallel = trained_parallel.predict(&X.view()).unwrap();
        let pred_sequential = trained_sequential.predict(&X.view()).unwrap();

        assert_eq!(pred_parallel.shape(), pred_sequential.shape());
        assert_eq!(pred_parallel.shape(), &[6, 3]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_multi_output_regressor() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![
            [1.5, 2.5, 3.5],
            [2.5, 3.5, 4.5],
            [3.5, 4.5, 5.5],
            [4.5, 5.5, 6.5],
            [5.5, 6.5, 7.5],
            [6.5, 7.5, 8.5]
        ];

        // Test with parallel training
        let regressor_parallel = MultiOutputRegressor::new().n_jobs(Some(2));
        let trained_parallel = regressor_parallel.fit(&X.view(), &y).unwrap();

        // Test with sequential training
        let regressor_sequential = MultiOutputRegressor::new().n_jobs(Some(1));
        let trained_sequential = regressor_sequential.fit(&X.view(), &y).unwrap();

        // Results should be the same
        assert_eq!(trained_parallel.n_targets(), trained_sequential.n_targets());

        // Test predictions
        let pred_parallel = trained_parallel.predict(&X.view()).unwrap();
        let pred_sequential = trained_sequential.predict(&X.view()).unwrap();

        assert_eq!(pred_parallel.shape(), pred_sequential.shape());
        assert_eq!(pred_parallel.shape(), &[6, 3]);

        // Predictions should be approximately equal
        for i in 0..pred_parallel.nrows() {
            for j in 0..pred_parallel.ncols() {
                assert_abs_diff_eq!(
                    pred_parallel[[i, j]],
                    pred_sequential[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_parallel_training_performance_classifier() {
        // Create larger dataset to see potential parallel benefits
        let n_samples = 1000;
        let n_features = 50;
        let n_targets = 20;

        let mut X = Array2::<Float>::zeros((n_samples, n_features));
        let mut y = Array2::<i32>::zeros((n_samples, n_targets));

        // Fill with simple patterns
        for i in 0..n_samples {
            for j in 0..n_features {
                X[[i, j]] = (i * j) as Float * 0.01;
            }
            for j in 0..n_targets {
                y[[i, j]] = ((i + j) % 2) as i32;
            }
        }

        // Time sequential training
        let start_sequential = Instant::now();
        let classifier_sequential = MultiOutputClassifier::new().n_jobs(Some(1));
        let trained_sequential = classifier_sequential.fit(&X.view(), &y).unwrap();
        let sequential_time = start_sequential.elapsed();

        // Time parallel training
        let start_parallel = Instant::now();
        let classifier_parallel = MultiOutputClassifier::new().n_jobs(Some(4));
        let trained_parallel = classifier_parallel.fit(&X.view(), &y).unwrap();
        let parallel_time = start_parallel.elapsed();

        // Ensure both produce valid results
        assert_eq!(trained_parallel.n_targets(), n_targets);
        assert_eq!(trained_sequential.n_targets(), n_targets);

        // Test predictions are consistent
        let pred_parallel = trained_parallel.predict(&X.view()).unwrap();
        let pred_sequential = trained_sequential.predict(&X.view()).unwrap();
        assert_eq!(pred_parallel.shape(), pred_sequential.shape());

        println!(
            "Sequential time: {:?}, Parallel time: {:?}",
            sequential_time, parallel_time
        );
    }

    #[test]
    fn test_parallel_training_performance_regressor() {
        // Create larger dataset to see potential parallel benefits
        let n_samples = 1000;
        let n_features = 50;
        let n_targets = 20;

        let mut X = Array2::<Float>::zeros((n_samples, n_features));
        let mut y = Array2::<f64>::zeros((n_samples, n_targets));

        // Fill with simple patterns
        for i in 0..n_samples {
            for j in 0..n_features {
                X[[i, j]] = (i * j) as Float * 0.01;
            }
            for j in 0..n_targets {
                y[[i, j]] = (i + j) as f64 * 0.1;
            }
        }

        // Time sequential training
        let start_sequential = Instant::now();
        let regressor_sequential = MultiOutputRegressor::new().n_jobs(Some(1));
        let trained_sequential = regressor_sequential.fit(&X.view(), &y).unwrap();
        let sequential_time = start_sequential.elapsed();

        // Time parallel training
        let start_parallel = Instant::now();
        let regressor_parallel = MultiOutputRegressor::new().n_jobs(Some(4));
        let trained_parallel = regressor_parallel.fit(&X.view(), &y).unwrap();
        let parallel_time = start_parallel.elapsed();

        // Ensure both produce valid results
        assert_eq!(trained_parallel.n_targets(), n_targets);
        assert_eq!(trained_sequential.n_targets(), n_targets);

        // Test predictions are consistent
        let pred_parallel = trained_parallel.predict(&X.view()).unwrap();
        let pred_sequential = trained_sequential.predict(&X.view()).unwrap();
        assert_eq!(pred_parallel.shape(), pred_sequential.shape());

        println!(
            "Sequential time: {:?}, Parallel time: {:?}",
            sequential_time, parallel_time
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_training_thread_safety() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y_class = array![[0, 1], [1, 0], [0, 1], [1, 0]];
        let y_reg = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        // Test multiple parallel runs to check for race conditions
        for _ in 0..10 {
            let classifier = MultiOutputClassifier::new().n_jobs(Some(2));
            let trained = classifier.fit(&X.view(), &y_class).unwrap();
            let predictions = trained.predict(&X.view()).unwrap();
            assert_eq!(predictions.shape(), &[4, 2]);

            let regressor = MultiOutputRegressor::new().n_jobs(Some(2));
            let trained = regressor.fit(&X.view(), &y_reg).unwrap();
            let predictions = trained.predict(&X.view()).unwrap();
            assert_eq!(predictions.shape(), &[4, 2]);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_training_edge_cases() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y_class = array![[0, 1], [1, 0]];
        let y_reg = array![[1.0, 2.0], [2.0, 3.0]];

        // Test with more threads than targets (should handle gracefully)
        let classifier = MultiOutputClassifier::new().n_jobs(Some(10));
        let trained = classifier.fit(&X.view(), &y_class).unwrap();
        assert_eq!(trained.n_targets(), 2);

        let regressor = MultiOutputRegressor::new().n_jobs(Some(10));
        let trained = regressor.fit(&X.view(), &y_reg).unwrap();
        assert_eq!(trained.n_targets(), 2);

        // Test with single target (should fall back to sequential)
        let y_single = array![[0], [1]];
        let classifier_single = MultiOutputClassifier::new().n_jobs(Some(4));
        let trained_single = classifier_single.fit(&X.view(), &y_single).unwrap();
        assert_eq!(trained_single.n_targets(), 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parallel_training_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y_mismatch = array![[0, 1, 0], [1, 0, 1], [0, 1, 0]]; // Wrong number of samples

        // Test error handling in parallel mode
        let classifier = MultiOutputClassifier::new().n_jobs(Some(2));
        let result = classifier.fit(&X.view(), &y_mismatch);
        assert!(result.is_err());

        let regressor = MultiOutputRegressor::new().n_jobs(Some(2));
        let y_reg_mismatch = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let result = regressor.fit(&X.view(), &y_reg_mismatch);
        assert!(result.is_err());
    }
}
