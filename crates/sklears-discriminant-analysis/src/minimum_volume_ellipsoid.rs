//! Minimum Volume Ellipsoid (MVE) Methods
//!
//! This module implements minimum volume ellipsoid methods for robust estimation
//! in discriminant analysis. MVE is a robust estimator that finds the ellipsoid
//! of minimum volume that contains at least h observations, where h is chosen
//! to ensure high breakdown point.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct MinimumVolumeEllipsoidConfig {
    /// Support fraction (fraction of data used for estimation)
    pub support_fraction: Float,
    /// Number of random subsamples to try
    pub n_trials: usize,
    /// Maximum number of iterations for refinement
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use reweighting step
    pub reweight: bool,
    /// Reweighting threshold
    pub reweight_threshold: Float,
}

impl Default for MinimumVolumeEllipsoidConfig {
    fn default() -> Self {
        Self {
            support_fraction: 0.5,
            n_trials: 500,
            max_iter: 30,
            tol: 1e-4,
            random_state: None,
            reweight: true,
            reweight_threshold: 2.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinimumVolumeEllipsoidDiscriminantAnalysis<State = Untrained> {
    config: MinimumVolumeEllipsoidConfig,
    data: Option<TrainedData>,
    _state: PhantomData<State>,
}

#[derive(Debug, Clone)]
struct TrainedData {
    location: Array1<Float>,
    covariance: Array2<Float>,
    precision: Array2<Float>,
    support: Array1<bool>,
    classes: Array1<i32>,
    class_locations: Array2<Float>,
    class_covariances: Vec<Array2<Float>>,
    class_precisions: Vec<Array2<Float>>,
    class_supports: Vec<Array1<bool>>,
    n_features: usize,
    determinant: Float,
}

impl MinimumVolumeEllipsoidDiscriminantAnalysis<Untrained> {
    pub fn new() -> Self {
        Self {
            config: MinimumVolumeEllipsoidConfig::default(),
            data: None,
            _state: PhantomData,
        }
    }

    pub fn with_config(config: MinimumVolumeEllipsoidConfig) -> Self {
        Self {
            config,
            data: None,
            _state: PhantomData,
        }
    }

    pub fn support_fraction(mut self, support_fraction: Float) -> Self {
        self.config.support_fraction = support_fraction.max(0.0).min(1.0);
        self
    }

    pub fn n_trials(mut self, n_trials: usize) -> Self {
        self.config.n_trials = n_trials;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn reweight(mut self, reweight: bool) -> Self {
        self.config.reweight = reweight;
        self
    }

    pub fn reweight_threshold(mut self, threshold: Float) -> Self {
        self.config.reweight_threshold = threshold;
        self
    }
}

impl Default for MinimumVolumeEllipsoidDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

pub type TrainedMinimumVolumeEllipsoidDiscriminantAnalysis =
    MinimumVolumeEllipsoidDiscriminantAnalysis<Trained>;

impl MinimumVolumeEllipsoidDiscriminantAnalysis<Trained> {
    pub fn location(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().location
    }

    pub fn covariance(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().covariance
    }

    pub fn precision(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().precision
    }

    pub fn support(&self) -> &Array1<bool> {
        &self.data.as_ref().unwrap().support
    }

    pub fn classes(&self) -> &Array1<i32> {
        &self.data.as_ref().unwrap().classes
    }

    pub fn class_locations(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().class_locations
    }

    pub fn class_covariances(&self) -> &Vec<Array2<Float>> {
        &self.data.as_ref().unwrap().class_covariances
    }

    pub fn n_features(&self) -> usize {
        self.data.as_ref().unwrap().n_features
    }

    pub fn determinant(&self) -> Float {
        self.data.as_ref().unwrap().determinant
    }

    /// Compute Mahalanobis distance to the center
    pub fn mahalanobis_distance(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let data = self.data.as_ref().unwrap();
        let mut distances = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let diff = &sample - &data.location;
            let distance = diff.dot(&data.precision.dot(&diff)).sqrt();
            distances[i] = distance;
        }

        Ok(distances)
    }

    /// Identify outliers based on Mahalanobis distance
    pub fn is_outlier(&self, x: &Array2<Float>) -> Result<Array1<bool>> {
        let distances = self.mahalanobis_distance(x)?;
        let threshold = self.config.reweight_threshold;
        Ok(distances.mapv(|d| d > threshold))
    }
}

impl Estimator<Untrained> for MinimumVolumeEllipsoidDiscriminantAnalysis<Untrained> {
    type Config = MinimumVolumeEllipsoidConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for MinimumVolumeEllipsoidDiscriminantAnalysis<Untrained> {
    type Fitted = MinimumVolumeEllipsoidDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Input data and targets have different lengths".to_string(),
            ));
        }

        // Extract unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Compute global MVE estimate
        let (global_location, global_covariance, global_support) = self.compute_mve(x)?;
        let global_precision = self.invert_matrix(&global_covariance)?;
        let global_determinant = self.determinant(&global_covariance)?;

        // Compute per-class MVE estimates
        let mut class_locations = Array2::zeros((n_classes, n_features));
        let mut class_covariances = Vec::new();
        let mut class_precisions = Vec::new();
        let mut class_supports = Vec::new();

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            if class_mask.len() < n_features + 1 {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has too few samples for MVE estimation",
                    class_label
                )));
            }

            let class_data = x.select(Axis(0), &class_mask);
            let (class_location, class_covariance, class_support) =
                self.compute_mve(&class_data)?;
            let class_precision = self.invert_matrix(&class_covariance)?;

            class_locations.row_mut(class_idx).assign(&class_location);
            class_covariances.push(class_covariance);
            class_precisions.push(class_precision);

            // Map support back to global indices
            let mut global_class_support = Array1::from_elem(n_samples, false);
            for (local_idx, &global_idx) in class_mask.iter().enumerate() {
                global_class_support[global_idx] = class_support[local_idx];
            }
            class_supports.push(global_class_support);
        }

        let trained_data = TrainedData {
            location: global_location,
            covariance: global_covariance,
            precision: global_precision,
            support: global_support,
            classes,
            class_locations,
            class_covariances,
            class_precisions,
            class_supports,
            n_features,
            determinant: global_determinant,
        };

        Ok(MinimumVolumeEllipsoidDiscriminantAnalysis {
            config: self.config,
            data: Some(trained_data),
            _state: PhantomData,
        })
    }
}

impl MinimumVolumeEllipsoidDiscriminantAnalysis<Untrained> {
    /// Compute the Minimum Volume Ellipsoid estimate
    fn compute_mve(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>, Array1<bool>)> {
        let (n_samples, n_features) = x.dim();
        let h = ((n_samples as Float * self.config.support_fraction).ceil() as usize)
            .max(n_features + 1)
            .min(n_samples);

        let mut best_volume = Float::INFINITY;
        let mut best_location = Array1::zeros(n_features);
        let mut best_covariance = Array2::eye(n_features);
        let mut best_support = Array1::from_elem(n_samples, false);

        // Initialize random number generator
        let mut hasher = DefaultHasher::new();
        self.config.random_state.unwrap_or(42).hash(&mut hasher);
        let mut seed = hasher.finish();

        let next_random = |seed: &mut u64| -> Float {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((*seed / 65536) % 32768) as Float / 32768.0
        };

        // Random subsample trials
        for _trial in 0..self.config.n_trials {
            // Generate random subsample of size h
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = (next_random(&mut seed) * (i + 1) as Float) as usize;
                indices.swap(i, j);
            }
            indices.truncate(h);

            let subsample = x.select(Axis(0), &indices);

            // Compute initial estimates from subsample
            if let Ok((location, covariance)) = self.compute_sample_statistics(&subsample) {
                if let Ok(_precision) = self.invert_matrix(&covariance) {
                    // C-step iterations to improve the estimate
                    let (refined_location, refined_covariance, support) =
                        self.c_step_iterations(x, &location, &covariance, h)?;

                    // Compute volume (determinant)
                    if let Ok(det) = self.determinant(&refined_covariance) {
                        if det > 0.0 && det < best_volume {
                            best_volume = det;
                            best_location = refined_location;
                            best_covariance = refined_covariance;
                            best_support = support;
                        }
                    }
                }
            }
        }

        if best_volume == Float::INFINITY {
            return Err(SklearsError::InvalidInput(
                "Failed to find valid MVE estimate".to_string(),
            ));
        }

        // Reweighting step if enabled
        if self.config.reweight {
            let (reweighted_location, reweighted_covariance, reweighted_support) =
                self.reweighting_step(x, &best_location, &best_covariance)?;
            Ok((
                reweighted_location,
                reweighted_covariance,
                reweighted_support,
            ))
        } else {
            Ok((best_location, best_covariance, best_support))
        }
    }

    /// C-step iterations to refine the MVE estimate
    fn c_step_iterations(
        &self,
        x: &Array2<Float>,
        initial_location: &Array1<Float>,
        initial_covariance: &Array2<Float>,
        h: usize,
    ) -> Result<(Array1<Float>, Array2<Float>, Array1<bool>)> {
        let n_samples = x.nrows();
        let mut location = initial_location.clone();
        let mut covariance = initial_covariance.clone();
        let mut prev_det = self.determinant(&covariance)?;

        for _iter in 0..self.config.max_iter {
            // Compute Mahalanobis distances
            let precision = self.invert_matrix(&covariance)?;
            let mut distances: Vec<(Float, usize)> = Vec::new();

            for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
                let diff = &sample - &location;
                let distance = diff.dot(&precision.dot(&diff));
                distances.push((distance, i));
            }

            // Sort by distance and take h closest points
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let subset_indices: Vec<usize> =
                distances.iter().take(h).map(|(_, idx)| *idx).collect();
            let subset = x.select(Axis(0), &subset_indices);

            // Recompute location and covariance
            let (new_location, new_covariance) = self.compute_sample_statistics(&subset)?;
            let new_det = self.determinant(&new_covariance)?;

            // Check for convergence
            if (new_det - prev_det).abs() / prev_det < self.config.tol {
                location = new_location;
                covariance = new_covariance;
                break;
            }

            location = new_location;
            covariance = new_covariance;
            prev_det = new_det;
        }

        // Create support indicator
        let precision = self.invert_matrix(&covariance)?;
        let mut distances: Vec<(Float, usize)> = Vec::new();
        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let diff = &sample - &location;
            let distance = diff.dot(&precision.dot(&diff));
            distances.push((distance, i));
        }
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut support = Array1::from_elem(n_samples, false);
        for (_, idx) in distances.iter().take(h) {
            support[*idx] = true;
        }

        Ok((location, covariance, support))
    }

    /// Reweighting step to improve efficiency
    fn reweighting_step(
        &self,
        x: &Array2<Float>,
        location: &Array1<Float>,
        covariance: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>, Array1<bool>)> {
        let n_samples = x.nrows();
        let precision = self.invert_matrix(covariance)?;
        let threshold = self.config.reweight_threshold * self.config.reweight_threshold;

        // Identify inliers based on Mahalanobis distance
        let mut inlier_indices = Vec::new();
        let mut support = Array1::from_elem(n_samples, false);

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let diff = &sample - location;
            let distance_sq = diff.dot(&precision.dot(&diff));

            if distance_sq <= threshold {
                inlier_indices.push(i);
                support[i] = true;
            }
        }

        if inlier_indices.len() < x.ncols() + 1 {
            // Fall back to original estimate if too few inliers
            return Ok((location.clone(), covariance.clone(), support));
        }

        // Recompute statistics using inliers
        let inlier_data = x.select(Axis(0), &inlier_indices);
        let (reweighted_location, reweighted_covariance) =
            self.compute_sample_statistics(&inlier_data)?;

        Ok((reweighted_location, reweighted_covariance, support))
    }

    /// Compute sample mean and covariance matrix
    fn compute_sample_statistics(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let (n_samples, n_features) = x.dim();

        if n_samples < n_features + 1 {
            return Err(SklearsError::InvalidInput(
                "Too few samples for covariance estimation".to_string(),
            ));
        }

        // Compute mean
        let mean = x.mean_axis(Axis(0)).unwrap();

        // Compute covariance matrix
        let mut covariance = Array2::zeros((n_features, n_features));
        for sample in x.axis_iter(Axis(0)) {
            let diff = &sample - &mean;
            for i in 0..n_features {
                for j in 0..n_features {
                    covariance[[i, j]] += diff[i] * diff[j];
                }
            }
        }
        covariance /= (n_samples - 1) as Float;

        // Add regularization for numerical stability
        for i in 0..n_features {
            covariance[[i, i]] += 1e-8;
        }

        Ok((mean, covariance))
    }

    /// Compute matrix determinant
    fn determinant(&self, matrix: &Array2<Float>) -> Result<Float> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        let mut lu = matrix.clone();
        let mut det = 1.0;

        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            for k in (i + 1)..n {
                if lu[[k, i]].abs() > lu[[max_idx, i]].abs() {
                    max_idx = k;
                }
            }

            if lu[[max_idx, i]].abs() < 1e-12 {
                return Ok(0.0); // Singular matrix
            }

            if max_idx != i {
                // Swap rows
                for j in 0..n {
                    let temp = lu[[i, j]];
                    lu[[i, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;
                }
                det = -det;
            }

            det *= lu[[i, i]];

            // Eliminate column
            for k in (i + 1)..n {
                let factor = lu[[k, i]] / lu[[i, i]];
                for j in (i + 1)..n {
                    lu[[k, j]] -= factor * lu[[i, j]];
                }
            }
        }

        Ok(det)
    }

    /// Invert a symmetric positive definite matrix
    fn invert_matrix(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        let mut lu = matrix.clone();
        let mut inv = Array2::eye(n);

        // LU decomposition with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            for k in (i + 1)..n {
                if lu[[k, i]].abs() > lu[[max_idx, i]].abs() {
                    max_idx = k;
                }
            }

            if lu[[max_idx, i]].abs() < 1e-12 {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }

            if max_idx != i {
                // Swap rows in both matrices
                for j in 0..n {
                    let temp = lu[[i, j]];
                    lu[[i, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;

                    let temp = inv[[i, j]];
                    inv[[i, j]] = inv[[max_idx, j]];
                    inv[[max_idx, j]] = temp;
                }
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = lu[[k, i]] / lu[[i, i]];
                for j in (i + 1)..n {
                    lu[[k, j]] -= factor * lu[[i, j]];
                }
                for j in 0..n {
                    inv[[k, j]] -= factor * inv[[i, j]];
                }
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in 0..n {
                for k in (i + 1)..n {
                    inv[[i, j]] -= lu[[i, k]] * inv[[k, j]];
                }
                inv[[i, j]] /= lu[[i, i]];
            }
        }

        Ok(inv)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MinimumVolumeEllipsoidDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probas.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes()[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>>
    for MinimumVolumeEllipsoidDiscriminantAnalysis<Trained>
{
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_features
            )));
        }

        let data = self.data.as_ref().unwrap();
        let n_classes = data.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let mut log_probas = Array1::zeros(n_classes);

            for (class_idx, _) in data.classes.iter().enumerate() {
                let class_location = data.class_locations.row(class_idx);
                let class_precision = &data.class_precisions[class_idx];

                // Compute log probability using multivariate normal distribution
                let diff = &sample - &class_location;
                let mahalanobis_sq = diff.dot(&class_precision.dot(&diff));

                // Log probability (without normalization constant)
                log_probas[class_idx] = -0.5 * mahalanobis_sq;
            }

            // Convert to probabilities using softmax
            let max_log_proba = log_probas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_log_probas: Array1<Float> = log_probas.mapv(|x| (x - max_log_proba).exp());
            let sum_exp: Float = exp_log_probas.sum();

            if sum_exp > 1e-10 {
                probas.row_mut(i).assign(&(exp_log_probas / sum_exp));
            } else {
                probas.row_mut(i).fill(1.0 / n_classes as Float);
            }
        }

        Ok(probas)
    }
}

impl Transform<Array2<Float>, Array2<Float>>
    for MinimumVolumeEllipsoidDiscriminantAnalysis<Trained>
{
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Transform data using the global precision matrix (Mahalanobis transformation)
        let data = self.data.as_ref().unwrap();
        let mut transformed = Array2::zeros(x.dim());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let diff = &sample - &data.location;
            let transformed_sample = data.precision.dot(&diff);
            transformed.row_mut(i).assign(&transformed_sample);
        }

        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_minimum_volume_ellipsoid_basic() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],   // Class 0 - clean data
            [10.0, 10.0], // Outlier
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2],   // Class 1 - clean data
            [20.0, 20.0]  // Outlier
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new()
            .support_fraction(0.6)
            .n_trials(100);

        let fitted = mve.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_mve_predict_proba() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new()
            .support_fraction(0.8)
            .n_trials(50);

        let fitted = mve.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mve_outlier_detection() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],   // Class 0 normal points
            [10.0, 10.0], // Class 0 outlier
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2],   // Class 1 normal points
            [20.0, 20.0]  // Class 1 outlier
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new()
            .support_fraction(0.7)
            .reweight_threshold(2.0)
            .n_trials(50);

        let fitted = mve.fit(&x, &y).unwrap();
        let outliers = fitted.is_outlier(&x).unwrap();

        // The outliers at (10, 10) and (20, 20) should be detected
        assert!(outliers[3] || outliers[7]); // At least one outlier should be detected
        assert!(!outliers[0]); // Normal point should not be outlier
    }

    #[test]
    fn test_mve_mahalanobis_distance() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new().n_trials(50);

        let fitted = mve.fit(&x, &y).unwrap();
        let distances = fitted.mahalanobis_distance(&x).unwrap();

        assert_eq!(distances.len(), 6);
        assert!(distances.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_mve_transform() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new().n_trials(50);

        let fitted = mve.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (6, 2));
    }

    #[test]
    fn test_mve_different_support_fractions() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let support_fractions = vec![0.5, 0.7, 0.9];

        for support_fraction in support_fractions {
            let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new()
                .support_fraction(support_fraction)
                .n_trials(30);

            let fitted = mve.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 9);
            assert_eq!(fitted.classes().len(), 3);
        }
    }

    #[test]
    fn test_mve_reweighting() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        // Test with and without reweighting
        let mve_reweight = MinimumVolumeEllipsoidDiscriminantAnalysis::new()
            .reweight(true)
            .n_trials(50);

        let mve_no_reweight = MinimumVolumeEllipsoidDiscriminantAnalysis::new()
            .reweight(false)
            .n_trials(50);

        let fitted_reweight = mve_reweight.fit(&x, &y).unwrap();
        let fitted_no_reweight = mve_no_reweight.fit(&x, &y).unwrap();

        let predictions_reweight = fitted_reweight.predict(&x).unwrap();
        let predictions_no_reweight = fitted_no_reweight.predict(&x).unwrap();

        assert_eq!(predictions_reweight.len(), 6);
        assert_eq!(predictions_no_reweight.len(), 6);
    }

    #[test]
    fn test_mve_accessor_methods() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mve = MinimumVolumeEllipsoidDiscriminantAnalysis::new().n_trials(50);

        let fitted = mve.fit(&x, &y).unwrap();

        // Test accessor methods
        assert_eq!(fitted.location().len(), 2);
        assert_eq!(fitted.covariance().dim(), (2, 2));
        assert_eq!(fitted.precision().dim(), (2, 2));
        assert_eq!(fitted.support().len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.class_locations().dim(), (2, 2));
        assert_eq!(fitted.class_covariances().len(), 2);
        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.determinant() > 0.0);
    }
}
