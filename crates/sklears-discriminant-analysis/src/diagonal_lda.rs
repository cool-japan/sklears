//! Diagonal Linear Discriminant Analysis
//!
//! This module implements diagonal linear discriminant analysis, a variant of LDA
//! that assumes diagonal covariance matrices. This is particularly useful for
//! high-dimensional data where the full covariance matrix would be computationally
//! expensive or poorly estimated.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct DiagonalLinearDiscriminantAnalysisConfig {
    /// Regularization parameter for diagonal elements
    pub reg_param: Float,
    /// Whether to store class priors
    pub store_covariance: bool,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum number of iterations for iterative algorithms
    pub max_iter: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to normalize features
    pub normalize_features: bool,
    /// Feature selection threshold
    pub feature_threshold: Option<Float>,
}

impl Default for DiagonalLinearDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            reg_param: 1e-4,
            store_covariance: true,
            tol: 1e-4,
            max_iter: 100,
            random_state: None,
            normalize_features: false,
            feature_threshold: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagonalLinearDiscriminantAnalysis<State = Untrained> {
    config: DiagonalLinearDiscriminantAnalysisConfig,
    data: Option<TrainedData>,
    _state: PhantomData<State>,
}

#[derive(Debug, Clone)]
struct TrainedData {
    /// Class means [n_classes x n_features]
    class_means: Array2<Float>,
    /// Diagonal covariance elements [n_features]
    diagonal_covariance: Array1<Float>,
    /// Class priors [n_classes]
    class_priors: Array1<Float>,
    /// Unique class labels
    classes: Array1<i32>,
    /// Number of features
    n_features: usize,
    /// Feature scaling factors (if normalization is enabled)
    feature_scales: Option<Array1<Float>>,
    /// Feature means (if normalization is enabled)
    feature_means: Option<Array1<Float>>,
    /// Selected features (if feature selection is enabled)
    selected_features: Option<Array1<bool>>,
    /// Log of diagonal variances (cached for efficiency)
    log_variances: Array1<Float>,
}

impl DiagonalLinearDiscriminantAnalysis<Untrained> {
    pub fn new() -> Self {
        Self {
            config: DiagonalLinearDiscriminantAnalysisConfig::default(),
            data: None,
            _state: PhantomData,
        }
    }

    pub fn with_config(config: DiagonalLinearDiscriminantAnalysisConfig) -> Self {
        Self {
            config,
            data: None,
            _state: PhantomData,
        }
    }

    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    pub fn store_covariance(mut self, store_covariance: bool) -> Self {
        self.config.store_covariance = store_covariance;
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn normalize_features(mut self, normalize_features: bool) -> Self {
        self.config.normalize_features = normalize_features;
        self
    }

    pub fn feature_threshold(mut self, threshold: Option<Float>) -> Self {
        self.config.feature_threshold = threshold;
        self
    }
}

impl Default for DiagonalLinearDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

pub type TrainedDiagonalLinearDiscriminantAnalysis = DiagonalLinearDiscriminantAnalysis<Trained>;

impl DiagonalLinearDiscriminantAnalysis<Trained> {
    pub fn class_means(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().class_means
    }

    pub fn diagonal_covariance(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().diagonal_covariance
    }

    pub fn class_priors(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().class_priors
    }

    pub fn classes(&self) -> &Array1<i32> {
        &self.data.as_ref().unwrap().classes
    }

    pub fn n_features(&self) -> usize {
        self.data.as_ref().unwrap().n_features
    }

    pub fn feature_scales(&self) -> Option<&Array1<Float>> {
        self.data.as_ref().unwrap().feature_scales.as_ref()
    }

    pub fn feature_means(&self) -> Option<&Array1<Float>> {
        self.data.as_ref().unwrap().feature_means.as_ref()
    }

    pub fn selected_features(&self) -> Option<&Array1<bool>> {
        self.data.as_ref().unwrap().selected_features.as_ref()
    }

    /// Get the discriminant function coefficients for each class
    pub fn discriminant_coefficients(&self) -> Array2<Float> {
        let data = self.data.as_ref().unwrap();
        let n_classes = data.classes.len();
        let n_features = data.n_features;
        let mut coefficients = Array2::zeros((n_classes, n_features));

        for (class_idx, _) in data.classes.iter().enumerate() {
            for j in 0..n_features {
                // Linear discriminant coefficient: mean / variance
                coefficients[[class_idx, j]] =
                    data.class_means[[class_idx, j]] / data.diagonal_covariance[j];
            }
        }

        coefficients
    }

    /// Get the discriminant intercepts for each class
    pub fn discriminant_intercepts(&self) -> Array1<Float> {
        let data = self.data.as_ref().unwrap();
        let n_classes = data.classes.len();
        let mut intercepts = Array1::zeros(n_classes);

        for (class_idx, _) in data.classes.iter().enumerate() {
            let mut intercept = data.class_priors[class_idx].ln();

            for j in 0..data.n_features {
                let mean = data.class_means[[class_idx, j]];
                let var = data.diagonal_covariance[j];
                intercept -= 0.5 * mean * mean / var;
            }

            intercepts[class_idx] = intercept;
        }

        intercepts
    }

    /// Compute feature importance scores
    pub fn feature_importance(&self) -> Array1<Float> {
        let data = self.data.as_ref().unwrap();
        let n_features = data.n_features;
        let mut importance = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut feature_score = 0.0;

            // Compute variance of class means for this feature
            let means_j: Array1<Float> = data.class_means.column(j).to_owned();
            let mean_of_means = means_j.mean().unwrap();
            let between_class_var = means_j
                .mapv(|x| (x - mean_of_means).powi(2))
                .mean()
                .unwrap();

            // Feature importance is between-class variance / within-class variance
            feature_score = between_class_var / data.diagonal_covariance[j];
            importance[j] = feature_score;
        }

        importance
    }
}

impl Estimator<Untrained> for DiagonalLinearDiscriminantAnalysis<Untrained> {
    type Config = DiagonalLinearDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for DiagonalLinearDiscriminantAnalysis<Untrained> {
    type Fitted = DiagonalLinearDiscriminantAnalysis<Trained>;

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

        let mut x_processed = x.clone();
        let mut feature_means = None;
        let mut feature_scales = None;

        // Feature normalization
        if self.config.normalize_features {
            let means = x.mean_axis(Axis(0)).unwrap();
            let stds = self.compute_feature_stds(x, &means);

            for (i, mut sample) in x_processed.axis_iter_mut(Axis(0)).enumerate() {
                for j in 0..n_features {
                    sample[j] = (sample[j] - means[j]) / stds[j];
                }
            }

            feature_means = Some(means);
            feature_scales = Some(stds);
        }

        // Compute class means and priors
        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut class_counts = Array1::<Float>::zeros(n_classes);

        for (sample_idx, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                let sample = x_processed.row(sample_idx);
                for j in 0..n_features {
                    class_means[[class_idx, j]] += sample[j];
                }
                class_counts[class_idx] += 1.0;
            }
        }

        // Normalize means by class counts
        for (class_idx, &count) in class_counts.iter().enumerate() {
            if count > 0.0 {
                for j in 0..n_features {
                    class_means[[class_idx, j]] /= count;
                }
            }
        }

        // Compute class priors
        let total_samples = class_counts.sum();
        let class_priors = class_counts.mapv(|count| count / total_samples);

        // Compute pooled diagonal covariance
        let mut diagonal_covariance = Array1::zeros(n_features);

        for (sample_idx, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                let sample = x_processed.row(sample_idx);
                let class_mean = class_means.row(class_idx);

                for j in 0..n_features {
                    let diff = sample[j] - class_mean[j];
                    diagonal_covariance[j] += diff * diff;
                }
            }
        }

        // Normalize by degrees of freedom and add regularization
        for j in 0..n_features {
            diagonal_covariance[j] =
                diagonal_covariance[j] / (n_samples - n_classes) as Float + self.config.reg_param;
        }

        // Feature selection based on variance threshold
        let mut selected_features = None;
        if let Some(threshold) = self.config.feature_threshold {
            let mut feature_mask = Array1::from_elem(n_features, true);

            for j in 0..n_features {
                if diagonal_covariance[j] < threshold {
                    feature_mask[j] = false;
                }
            }

            selected_features = Some(feature_mask);
        }

        // Compute log variances for efficiency
        let log_variances = diagonal_covariance.mapv(|var: Float| var.ln());

        let trained_data = TrainedData {
            class_means,
            diagonal_covariance,
            class_priors,
            classes,
            n_features,
            feature_scales,
            feature_means,
            selected_features,
            log_variances,
        };

        Ok(DiagonalLinearDiscriminantAnalysis {
            config: self.config,
            data: Some(trained_data),
            _state: PhantomData,
        })
    }
}

impl DiagonalLinearDiscriminantAnalysis<Untrained> {
    /// Compute feature standard deviations
    fn compute_feature_stds(&self, x: &Array2<Float>, means: &Array1<Float>) -> Array1<Float> {
        let (n_samples, n_features) = x.dim();
        let mut stds = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut var = 0.0;
            for i in 0..n_samples {
                let diff = x[[i, j]] - means[j];
                var += diff * diff;
            }
            var /= (n_samples - 1) as Float;
            stds[j] = (var + 1e-8).sqrt(); // Add small epsilon to avoid division by zero
        }

        stds
    }
}

impl DiagonalLinearDiscriminantAnalysis<Trained> {
    /// Preprocess input data according to training normalization
    fn preprocess_input(&self, x: &Array2<Float>) -> Array2<Float> {
        let data = self.data.as_ref().unwrap();
        let mut x_processed = x.clone();

        if let (Some(means), Some(scales)) = (&data.feature_means, &data.feature_scales) {
            for (i, mut sample) in x_processed.axis_iter_mut(Axis(0)).enumerate() {
                for j in 0..data.n_features {
                    sample[j] = (sample[j] - means[j]) / scales[j];
                }
            }
        }

        x_processed
    }

    /// Apply feature selection if enabled
    fn apply_feature_selection(&self, x: &Array2<Float>) -> Array2<Float> {
        let data = self.data.as_ref().unwrap();

        if let Some(feature_mask) = &data.selected_features {
            let selected_indices: Vec<usize> = feature_mask
                .iter()
                .enumerate()
                .filter(|(_, &selected)| selected)
                .map(|(idx, _)| idx)
                .collect();

            if !selected_indices.is_empty() {
                return x.select(Axis(1), &selected_indices);
            }
        }

        x.clone()
    }
}

impl Predict<Array2<Float>, Array1<i32>> for DiagonalLinearDiscriminantAnalysis<Trained> {
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

impl PredictProba<Array2<Float>, Array2<Float>> for DiagonalLinearDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features_input) = x.dim();
        let data = self.data.as_ref().unwrap();

        // Check input dimensions before preprocessing
        if n_features_input != data.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                data.n_features, n_features_input
            )));
        }

        // Preprocess input data
        let x_processed = self.preprocess_input(x);
        let x_selected = self.apply_feature_selection(&x_processed);
        let (_, n_features_effective) = x_selected.dim();

        let n_classes = data.classes.len();
        let mut log_probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x_selected.axis_iter(Axis(0)).enumerate() {
            for (class_idx, _) in data.classes.iter().enumerate() {
                let mut log_prob = data.class_priors[class_idx].ln();

                // Compute log likelihood for diagonal covariance
                for j in 0..n_features_effective {
                    let mean = data.class_means[[class_idx, j]];
                    let var = data.diagonal_covariance[j];
                    let diff = sample[j] - mean;

                    // Log likelihood: -0.5 * log(2π) - 0.5 * log(σ²) - 0.5 * (x-μ)²/σ²
                    log_prob += -0.5 * data.log_variances[j] - 0.5 * diff * diff / var;
                }

                log_probas[[i, class_idx]] = log_prob;
            }
        }

        // Convert log probabilities to probabilities using softmax
        let mut probas = Array2::zeros((n_samples, n_classes));
        for (i, log_row) in log_probas.axis_iter(Axis(0)).enumerate() {
            let max_log_prob = log_row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_log_probas: Array1<Float> = log_row.mapv(|x| (x - max_log_prob).exp());
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

impl Transform<Array2<Float>, Array2<Float>> for DiagonalLinearDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let data = self.data.as_ref().unwrap();
        let (n_samples, n_features_input) = x.dim();

        if n_features_input != data.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                data.n_features, n_features_input
            )));
        }

        // Preprocess input data
        let x_processed = self.preprocess_input(x);
        let x_selected = self.apply_feature_selection(&x_processed);
        let (_, n_features_effective) = x_selected.dim();

        // Transform using diagonal LDA: scale by inverse square root of variances
        let mut x_transformed = Array2::zeros((n_samples, n_features_effective));

        for (i, sample) in x_selected.axis_iter(Axis(0)).enumerate() {
            for j in 0..n_features_effective {
                // Scale by inverse square root of variance (whitening transformation)
                x_transformed[[i, j]] = sample[j] / data.diagonal_covariance[j].sqrt();
            }
        }

        Ok(x_transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_diagonal_lda_basic() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2] // Class 1
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_diagonal_lda_predict_proba() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_diagonal_lda_with_regularization() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new().reg_param(0.1);
        let fitted = dlda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_diagonal_lda_with_normalization() {
        let x = array![[10.0, 20.0], [11.0, 21.0], [30.0, 40.0], [31.0, 41.0]];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new().normalize_features(true);
        let fitted = dlda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
        assert!(fitted.feature_means().is_some());
        assert!(fitted.feature_scales().is_some());
    }

    #[test]
    fn test_diagonal_lda_transform() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 2));
    }

    #[test]
    fn test_diagonal_lda_discriminant_coefficients() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let coefficients = fitted.discriminant_coefficients();
        let intercepts = fitted.discriminant_intercepts();

        assert_eq!(coefficients.dim(), (2, 2));
        assert_eq!(intercepts.len(), 2);
    }

    #[test]
    fn test_diagonal_lda_feature_importance() {
        let x = array![
            [1.0, 2.0, 5.0],
            [1.1, 2.1, 5.1],
            [3.0, 4.0, 5.0],
            [3.1, 4.1, 5.1]
        ];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let importance = fitted.feature_importance();

        assert_eq!(importance.len(), 3);

        // Features 0 and 1 should be more important than feature 2
        // (which has very little class separation)
        assert!(importance[0] > importance[2]);
        assert!(importance[1] > importance[2]);
    }

    #[test]
    fn test_diagonal_lda_with_feature_threshold() {
        let x = array![
            [1.0, 2.0, 5.0],
            [1.1, 2.1, 5.000001],
            [3.0, 4.0, 5.0],
            [3.1, 4.1, 5.000001]
        ];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new().feature_threshold(Some(1e-5));
        let fitted = dlda.fit(&x, &y).unwrap();

        // Feature selection should remove the low-variance feature
        assert!(fitted.selected_features().is_some());

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_diagonal_lda_diagonal_covariance() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let diag_cov = fitted.diagonal_covariance();

        assert_eq!(diag_cov.len(), 2);
        assert!(diag_cov.iter().all(|&var| var > 0.0));
    }

    #[test]
    fn test_diagonal_lda_multiclass() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let dlda = DiagonalLinearDiscriminantAnalysis::new();
        let fitted = dlda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);
        assert_eq!(probas.dim(), (6, 3));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }
}
