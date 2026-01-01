//! Canonical Discriminant Analysis
//!
//! This module implements Canonical Discriminant Analysis (CDA), which is a multivariate
//! statistical analysis technique that finds the linear combinations of quantitative
//! variables that best separate two or more groups of observations.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_linalg::compat::{Eig, Inverse};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};

/// Configuration for Canonical Discriminant Analysis
#[derive(Debug, Clone)]
pub struct CanonicalDiscriminantAnalysisConfig {
    /// Number of canonical variables to compute
    pub n_components: Option<usize>,
    /// Regularization parameter for numerical stability
    pub reg_param: Float,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to standardize the data
    pub standardize: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for CanonicalDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            reg_param: 1e-6,
            tol: 1e-6,
            max_iter: 1000,
            standardize: true,
            random_state: None,
        }
    }
}

/// Canonical Discriminant Analysis estimator
#[derive(Debug, Clone)]
pub struct CanonicalDiscriminantAnalysis {
    config: CanonicalDiscriminantAnalysisConfig,
}

impl CanonicalDiscriminantAnalysis {
    /// Create a new Canonical Discriminant Analysis estimator
    pub fn new() -> Self {
        Self {
            config: CanonicalDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the number of canonical variables to compute
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to standardize the data
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.config.standardize = standardize;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

/// Trained Canonical Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedCanonicalDiscriminantAnalysis {
    /// Configuration used for training
    config: CanonicalDiscriminantAnalysisConfig,
    /// Unique classes found during training
    classes: Array1<i32>,
    /// Canonical coefficients (discriminant functions)
    coefficients: Array2<Float>,
    /// Canonical eigenvalues
    eigenvalues: Array1<Float>,
    /// Group means in original space
    group_means: Array2<Float>,
    /// Overall mean
    overall_mean: Array1<Float>,
    /// Standardization parameters
    means: Array1<Float>,
    stds: Array1<Float>,
    /// Number of features
    n_features: usize,
    /// Number of samples seen during training
    n_samples_seen: usize,
}

impl TrainedCanonicalDiscriminantAnalysis {
    /// Get the classes found during training
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the canonical coefficients
    pub fn coefficients(&self) -> &Array2<Float> {
        &self.coefficients
    }

    /// Get the canonical eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        &self.eigenvalues
    }

    /// Get the group means in original space
    pub fn group_means(&self) -> &Array2<Float> {
        &self.group_means
    }

    /// Get the overall mean
    pub fn overall_mean(&self) -> &Array1<Float> {
        &self.overall_mean
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the number of samples seen during training
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Compute canonical correlations
    pub fn canonical_correlations(&self) -> Array1<Float> {
        self.eigenvalues.mapv(|x| x.sqrt() / (1.0 + x.sqrt()))
    }

    /// Standardize data using training statistics
    fn standardize_data<D: Data<Elem = Float>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<Float> {
        if self.config.standardize {
            let mut standardized = x.to_owned();
            for (i, mut column) in standardized.columns_mut().into_iter().enumerate() {
                column -= self.means[i];
                if self.stds[i] != 0.0 {
                    column /= self.stds[i];
                }
            }
            standardized
        } else {
            x.to_owned()
        }
    }

    /// Compute discriminant scores for samples
    fn compute_discriminant_scores<D: Data<Elem = Float>>(
        &self,
        x: &ArrayBase<D, Ix2>,
    ) -> Array2<Float> {
        let standardized = self.standardize_data(x);
        standardized.dot(&self.coefficients)
    }

    /// Compute Mahalanobis distances to group centroids
    fn compute_mahalanobis_distances<D: Data<Elem = Float>>(
        &self,
        x: &ArrayBase<D, Ix2>,
    ) -> Array2<Float> {
        let standardized = self.standardize_data(x);
        let mut distances = Array2::zeros((x.nrows(), self.classes.len()));

        for (i, &_class) in self.classes.iter().enumerate() {
            let group_mean = self.group_means.row(i);
            for (j, sample) in standardized.axis_iter(Axis(0)).enumerate() {
                let diff = &sample - &group_mean;
                distances[[j, i]] = diff.dot(&diff);
            }
        }

        distances
    }
}

impl Estimator for CanonicalDiscriminantAnalysis {
    type Config = CanonicalDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<D: Data<Elem = Float>> Fit<ArrayBase<D, Ix2>, Array1<i32>> for CanonicalDiscriminantAnalysis {
    type Fitted = TrainedCanonicalDiscriminantAnalysis;

    fn fit(self, x: &ArrayBase<D, Ix2>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", x.nrows(), y.len()),
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 samples are required".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 classes are required".to_string(),
            ));
        }

        // Standardize data if requested
        let (standardized_x, means, stds) = if self.config.standardize {
            let means = x.mean_axis(Axis(0)).unwrap();
            let stds = x.std_axis(Axis(0), 0.0);
            let mut standardized = x.to_owned();
            for (i, mut column) in standardized.columns_mut().into_iter().enumerate() {
                column -= means[i];
                if stds[i] != 0.0 {
                    column /= stds[i];
                }
            }
            (standardized, means, stds)
        } else {
            let means = Array1::zeros(n_features);
            let stds = Array1::ones(n_features);
            (x.to_owned(), means, stds)
        };

        // Compute overall mean
        let overall_mean = standardized_x.mean_axis(Axis(0)).unwrap();

        // Compute group means and sizes
        let mut group_means = Array2::zeros((n_classes, n_features));
        let mut group_sizes = Array1::zeros(n_classes);

        for (i, &class) in classes.iter().enumerate() {
            let mask: Array1<bool> = y.mapv(|c| c == class);
            let class_samples = standardized_x.select(
                Axis(0),
                &mask
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &b)| if b { Some(idx) } else { None })
                    .collect::<Vec<_>>(),
            );

            if class_samples.nrows() > 0 {
                group_means
                    .row_mut(i)
                    .assign(&class_samples.mean_axis(Axis(0)).unwrap());
                group_sizes[i] = class_samples.nrows() as Float;
            }
        }

        // Compute within-class scatter matrix (W)
        let mut w_matrix = Array2::zeros((n_features, n_features));
        for (i, &class) in classes.iter().enumerate() {
            let mask: Array1<bool> = y.mapv(|c| c == class);
            let class_samples = standardized_x.select(
                Axis(0),
                &mask
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &b)| if b { Some(idx) } else { None })
                    .collect::<Vec<_>>(),
            );

            if class_samples.nrows() > 0 {
                let group_mean = group_means.row(i);
                for sample in class_samples.axis_iter(Axis(0)) {
                    let diff = &sample - &group_mean;
                    let outer = diff
                        .clone()
                        .into_shape((n_features, 1))
                        .unwrap()
                        .dot(&diff.clone().into_shape((1, n_features)).unwrap());
                    w_matrix += &outer;
                }
            }
        }

        // Compute between-class scatter matrix (B)
        let mut b_matrix = Array2::zeros((n_features, n_features));
        for (i, &_class) in classes.iter().enumerate() {
            let diff = &group_means.row(i) - &overall_mean;
            let outer = diff
                .clone()
                .into_shape((n_features, 1))
                .unwrap()
                .dot(&diff.clone().into_shape((1, n_features)).unwrap());
            b_matrix += &(outer * group_sizes[i]);
        }

        // Add regularization to within-class scatter matrix
        for i in 0..n_features {
            w_matrix[[i, i]] += self.config.reg_param;
        }

        // Solve generalized eigenvalue problem: B * v = λ * W * v
        // This is equivalent to: W^(-1) * B * v = λ * v
        let w_inv = w_matrix.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to invert within-class scatter matrix".to_string())
        })?;

        let matrix = w_inv.dot(&b_matrix);
        let (eigenvalues, eigenvectors) = matrix.eig().map_err(|_| {
            SklearsError::NumericalError("Failed to compute eigenvalues".to_string())
        })?;

        // Sort eigenvalues and eigenvectors in descending order
        // Extract real parts from complex eigenvalues
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .zip(eigenvectors.columns())
            .map(|(val, vec)| {
                let real_val = val.re;
                let real_vec = vec.mapv(|c| c.re);
                (real_val, real_vec)
            })
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Determine number of components
        let n_components = self
            .config
            .n_components
            .unwrap_or(std::cmp::min(n_classes - 1, n_features));
        let n_components = std::cmp::min(n_components, eigen_pairs.len());

        // Extract top components
        let selected_eigenvalues = Array1::from_vec(
            eigen_pairs
                .iter()
                .take(n_components)
                .map(|(val, _)| *val)
                .collect(),
        );

        let mut coefficients = Array2::zeros((n_features, n_components));
        for (i, (_, eigenvector)) in eigen_pairs.iter().take(n_components).enumerate() {
            coefficients.column_mut(i).assign(eigenvector);
        }

        Ok(TrainedCanonicalDiscriminantAnalysis {
            config: self.config,
            classes,
            coefficients,
            eigenvalues: selected_eigenvalues,
            group_means,
            overall_mean,
            means,
            stds,
            n_features,
            n_samples_seen: n_samples,
        })
    }
}

impl<D: Data<Elem = Float>> Predict<ArrayBase<D, Ix2>, Array1<i32>>
    for TrainedCanonicalDiscriminantAnalysis
{
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        let distances = self.compute_mahalanobis_distances(x);
        let mut predictions = Array1::zeros(x.nrows());

        for (i, distance_row) in distances.axis_iter(Axis(0)).enumerate() {
            let min_idx = distance_row
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.classes[min_idx];
        }

        Ok(predictions)
    }
}

impl<D: Data<Elem = Float>> PredictProba<ArrayBase<D, Ix2>, Array2<Float>>
    for TrainedCanonicalDiscriminantAnalysis
{
    fn predict_proba(&self, x: &ArrayBase<D, Ix2>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        let distances = self.compute_mahalanobis_distances(x);
        let mut probabilities = Array2::zeros((x.nrows(), self.classes.len()));

        for (i, distance_row) in distances.axis_iter(Axis(0)).enumerate() {
            // Convert distances to probabilities using softmax
            let max_dist = distance_row
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_distances: Array1<Float> = distance_row.mapv(|d| (-(d - max_dist)).exp());
            let sum_exp = exp_distances.sum();

            for (j, &exp_dist) in exp_distances.iter().enumerate() {
                probabilities[[i, j]] = exp_dist / sum_exp;
            }
        }

        Ok(probabilities)
    }
}

impl<D: Data<Elem = Float>> Transform<ArrayBase<D, Ix2>, Array2<Float>>
    for TrainedCanonicalDiscriminantAnalysis
{
    fn transform(&self, x: &ArrayBase<D, Ix2>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        Ok(self.compute_discriminant_scores(x))
    }
}

impl Default for CanonicalDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use scirs2_core::ndarray::Axis;

    #[test]
    fn test_canonical_discriminant_analysis_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [4.0, 5.0, 6.0],
            [4.1, 5.1, 6.1],
            [4.2, 5.2, 6.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let cda = CanonicalDiscriminantAnalysis::new();
        let fitted = cda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.coefficients().nrows(), 3);
    }

    #[test]
    fn test_canonical_discriminant_predict_proba() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let cda = CanonicalDiscriminantAnalysis::new();
        let fitted = cda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_canonical_discriminant_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [4.0, 5.0, 6.0],
            [4.1, 5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1];

        let cda = CanonicalDiscriminantAnalysis::new().n_components(Some(1));
        let fitted = cda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 1));
    }

    #[test]
    fn test_canonical_discriminant_multiclass() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let cda = CanonicalDiscriminantAnalysis::new();
        let fitted = cda.fit(&x, &y).unwrap();
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

    #[test]
    fn test_canonical_discriminant_no_standardization() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let cda = CanonicalDiscriminantAnalysis::new().standardize(false);
        let fitted = cda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_canonical_discriminant_with_regularization() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let cda = CanonicalDiscriminantAnalysis::new().reg_param(0.1);
        let fitted = cda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_canonical_correlations() {
        let x = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [4.0, 5.0, 6.0],
            [4.1, 5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1];

        let cda = CanonicalDiscriminantAnalysis::new();
        let fitted = cda.fit(&x, &y).unwrap();
        let correlations = fitted.canonical_correlations();

        assert_eq!(correlations.len(), fitted.eigenvalues().len());
        for &corr in correlations.iter() {
            assert!(corr >= 0.0 && corr <= 1.0);
        }
    }
}
