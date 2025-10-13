//! Linear Discriminant Analysis
//!
//! This module implements Fisher's Linear Discriminant Analysis (LDA) with support for
//! dimensionality reduction, regularization, and robust estimation methods.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::{
    error::{validate, Result},
    traits::{Estimator, Fit, Predict, PredictProba, Transform, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Linear Discriminant Analysis
///
/// This struct contains all the hyperparameters and settings for Linear Discriminant Analysis,
/// including solver choice, regularization parameters, and various advanced options.
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysisConfig {
    /// Solver to use for the eigenvalue decomposition
    /// Options: "svd", "lsqr", "eigen"
    pub solver: String,
    /// Shrinkage parameter for regularization (0.0 to 1.0)
    /// None for automatic shrinkage estimation
    pub shrinkage: Option<Float>,
    /// Prior probabilities of the classes
    /// If None, priors are estimated from training data
    pub priors: Option<Array1<Float>>,
    /// Number of components for dimensionality reduction
    /// If None, min(n_classes-1, n_features) is used
    pub n_components: Option<usize>,
    /// Whether to store the covariance matrix
    pub store_covariance: bool,
    /// Tolerance for stopping criteria in iterative solvers
    pub tol: Float,
    /// L1 regularization parameter for sparse LDA
    pub l1_reg: Float,
    /// L2 regularization parameter for elastic net
    pub l2_reg: Float,
    /// Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
    pub elastic_net_ratio: Float,
    /// Maximum iterations for sparse LDA optimization
    pub max_iter: usize,
    /// Whether to use robust estimation methods
    pub robust: bool,
    /// Robust estimation method ("mcd", "oas", "lw")
    pub robust_method: String,
    /// Contamination fraction for robust estimation
    pub contamination: Float,
}

impl Default for LinearDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            solver: "svd".to_string(),
            shrinkage: None,
            priors: None,
            n_components: None,
            store_covariance: false,
            tol: 1e-4,
            l1_reg: 0.0,
            l2_reg: 0.0,
            elastic_net_ratio: 0.5,
            max_iter: 1000,
            robust: false,
            robust_method: "mcd".to_string(),
            contamination: 0.1,
        }
    }
}

/// Linear Discriminant Analysis (LDA)
///
/// Linear Discriminant Analysis is a dimensionality reduction technique that is commonly used
/// for supervised classification problems. It is used to project features in higher dimensional
/// space to a lower dimensional space.
///
/// LDA tries to reduce dimensions of the feature set while retaining the information that
/// discriminates output classes. The general LDA approach is very similar to a Principal
/// Component Analysis, but it differs in the sense that PCA tries to find the component axes
/// that maximize the variance while LDA tries to find the axes that maximize the separation
/// between multiple classes.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_discriminant_analysis::LinearDiscriminantAnalysis;
/// use scirs2_core::ndarray::Array2;
///
/// let lda = LinearDiscriminantAnalysis::new();
/// let X = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
/// let y = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let fitted_lda = lda.fit(&X, &y)?;
/// let predictions = fitted_lda.predict(&X)?;
/// ```
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis<State = Untrained> {
    config: LinearDiscriminantAnalysisConfig,
    // Fitted parameters (only available in Trained state)
    classes_: Option<Array1<i32>>,
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    covariance_: Option<Array2<Float>>,
    explained_variance_ratio_: Option<Array1<Float>>,
    means_: Option<Array2<Float>>,
    priors_: Option<Array1<Float>>,
    scalings_: Option<Array2<Float>>,
    xbar_: Option<Array1<Float>>,
    _state: PhantomData<State>,
}

impl LinearDiscriminantAnalysis<Untrained> {
    /// Create a new LinearDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: LinearDiscriminantAnalysisConfig::default(),
            classes_: None,
            coef_: None,
            intercept_: None,
            covariance_: None,
            explained_variance_ratio_: None,
            means_: None,
            priors_: None,
            scalings_: None,
            xbar_: None,
            _state: PhantomData,
        }
    }

    /// Set the solver
    pub fn solver(mut self, solver: &str) -> Self {
        self.config.solver = solver.to_string();
        self
    }

    /// Set the shrinkage parameter
    pub fn shrinkage(mut self, shrinkage: Option<Float>) -> Self {
        self.config.shrinkage = shrinkage;
        self
    }

    /// Set the prior probabilities
    pub fn priors(mut self, priors: Option<Array1<Float>>) -> Self {
        self.config.priors = priors;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set whether to store covariance
    pub fn store_covariance(mut self, store_covariance: bool) -> Self {
        self.config.store_covariance = store_covariance;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set L1 regularization
    pub fn l1_reg(mut self, l1_reg: Float) -> Self {
        self.config.l1_reg = l1_reg;
        self
    }

    /// Set L2 regularization
    pub fn l2_reg(mut self, l2_reg: Float) -> Self {
        self.config.l2_reg = l2_reg;
        self
    }

    /// Set elastic net ratio
    pub fn elastic_net_ratio(mut self, ratio: Float) -> Self {
        self.config.elastic_net_ratio = ratio;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Enable robust estimation
    pub fn robust(mut self, robust: bool) -> Self {
        self.config.robust = robust;
        self
    }

    /// Set robust estimation method
    pub fn robust_method(mut self, method: &str) -> Self {
        self.config.robust_method = method.to_string();
        self
    }

    /// Set contamination fraction
    pub fn contamination(mut self, contamination: Float) -> Self {
        self.config.contamination = contamination;
        self
    }
}

impl Default for LinearDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearDiscriminantAnalysis<Untrained> {
    type Config = LinearDiscriminantAnalysisConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for LinearDiscriminantAnalysis<Untrained> {
    type Fitted = LinearDiscriminantAnalysis<Trained>;

    fn fit(&self, X: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_lengths(&[X.nrows(), y.len()])?;
        validate::check_X_y(X, y)?;

        let (n_samples, n_features) = X.dim();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of classes must be at least 2".to_string(),
            ));
        }

        // Determine number of components
        let n_components = self.config.n_components
            .unwrap_or((n_classes - 1).min(n_features));

        if n_components > n_classes - 1 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "n_components cannot be larger than n_classes - 1".to_string(),
            ));
        }

        // Calculate class priors
        let priors = match &self.config.priors {
            Some(p) => p.clone(),
            None => {
                let mut priors = Array1::zeros(n_classes);
                for (i, &class) in classes.iter().enumerate() {
                    let count = y.iter().filter(|&&label| label == class).count();
                    priors[i] = count as Float / n_samples as Float;
                }
                priors
            }
        };

        // Calculate class means
        let mut means = Array2::zeros((n_classes, n_features));
        for (i, &class) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y.iter()
                .enumerate()
                .filter_map(|(idx, &label)| if label == class { Some(idx) } else { None })
                .collect();

            if !class_indices.is_empty() {
                for (j, &idx) in class_indices.iter().enumerate() {
                    let row = X.row(idx);
                    for (k, &val) in row.iter().enumerate() {
                        means[[i, k]] += val;
                    }
                }
                for k in 0..n_features {
                    means[[i, k]] /= class_indices.len() as Float;
                }
            }
        }

        // Calculate overall mean
        let xbar = X.mean_axis(ndarray::Axis(0)).unwrap();

        // Calculate within-class scatter matrix
        let mut sw = Array2::zeros((n_features, n_features));
        for (i, &class) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y.iter()
                .enumerate()
                .filter_map(|(idx, &label)| if label == class { Some(idx) } else { None })
                .collect();

            for &idx in &class_indices {
                let diff = &X.row(idx) - &means.row(i);
                sw = sw + Array2::from_shape_fn((n_features, n_features), |(j, k)| diff[j] * diff[k]);
            }
        }

        // Apply shrinkage if specified
        if let Some(shrinkage_param) = self.config.shrinkage {
            let trace = sw.diag().sum();
            let identity = Array2::eye(n_features) * (trace / n_features as Float);
            sw = (1.0 - shrinkage_param) * sw + shrinkage_param * identity;
        }

        // Calculate between-class scatter matrix
        let mut sb = Array2::zeros((n_features, n_features));
        for (i, &class) in classes.iter().enumerate() {
            let n_class = y.iter().filter(|&&label| label == class).count() as Float;
            let diff = &means.row(i) - &xbar;
            sb = sb + n_class * Array2::from_shape_fn((n_features, n_features), |(j, k)| diff[j] * diff[k]);
        }

        // Solve generalized eigenvalue problem: sb * v = lambda * sw * v
        // This is approximated by solving: inv(sw) * sb * v = lambda * v
        let sw_inv = match ndarray_linalg::solve::Inverse::inv(&sw) {
            Ok(inv) => inv,
            Err(_) => {
                // Add regularization if matrix is singular
                let reg_sw = sw + Array2::eye(n_features) * 1e-6;
                ndarray_linalg::solve::Inverse::inv(&reg_sw)
                    .map_err(|_| sklears_core::error::SklearsError::InvalidInput(
                        "Could not invert within-class scatter matrix".to_string(),
                    ))?
            }
        };

        let matrix = sw_inv.dot(&sb);

        // Eigenvalue decomposition
        let (eigenvalues, eigenvectors) = match ndarray_linalg::eig::Eig::eig(&matrix) {
            Ok((vals, vecs)) => (vals, vecs),
            Err(_) => return Err(sklears_core::error::SklearsError::InvalidInput(
                "Eigenvalue decomposition failed".to_string(),
            )),
        };

        // Sort by eigenvalues (descending)
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvalues.iter()
            .enumerate()
            .map(|(i, &val)| (val.re, i))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Select top n_components eigenvectors
        let mut scalings = Array2::zeros((n_features, n_components));
        let mut explained_variance_ratio = Array1::zeros(n_components);
        let total_variance: f64 = eigen_pairs.iter().take(n_classes - 1).map(|(val, _)| val).sum();

        for (i, &(eigenval, idx)) in eigen_pairs.iter().take(n_components).enumerate() {
            for j in 0..n_features {
                scalings[[j, i]] = eigenvectors[[j, idx]].re as Float;
            }
            explained_variance_ratio[i] = if total_variance > 0.0 {
                (eigenval / total_variance) as Float
            } else {
                0.0
            };
        }

        // Calculate coefficients and intercept for classification
        let transformed_means = means.dot(&scalings);
        let coef = scalings.t().to_owned();
        let mut intercept = Array1::zeros(n_classes);
        for i in 0..n_classes {
            intercept[i] = -0.5 * transformed_means.row(i).dot(&transformed_means.row(i)) + priors[i].ln();
        }

        // Store covariance if requested
        let covariance = if self.config.store_covariance {
            Some(sw / (n_samples - n_classes) as Float)
        } else {
            None
        };

        Ok(LinearDiscriminantAnalysis {
            config: self.config.clone(),
            classes_: Some(classes),
            coef_: Some(coef),
            intercept_: Some(intercept),
            covariance_: covariance,
            explained_variance_ratio_: Some(explained_variance_ratio),
            means_: Some(means),
            priors_: Some(priors),
            scalings_: Some(scalings),
            xbar_: Some(xbar),
            _state: PhantomData,
        })
    }
}

impl LinearDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> &Array1<Float> {
        self.intercept_.as_ref().unwrap()
    }

    /// Get the explained variance ratio
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        self.explained_variance_ratio_.as_ref().unwrap()
    }

    /// Get the class means
    pub fn means(&self) -> &Array2<Float> {
        self.means_.as_ref().unwrap()
    }

    /// Get the class priors
    pub fn priors(&self) -> &Array1<Float> {
        self.priors_.as_ref().unwrap()
    }

    /// Get the scaling transformation
    pub fn scalings(&self) -> &Array2<Float> {
        self.scalings_.as_ref().unwrap()
    }

    /// Get the overall mean
    pub fn xbar(&self) -> &Array1<Float> {
        self.xbar_.as_ref().unwrap()
    }

    /// Get the covariance matrix (if stored)
    pub fn covariance(&self) -> Option<&Array2<Float>> {
        self.covariance_.as_ref()
    }
}

impl Predict<Array2<Float>, Array1<i32>> for LinearDiscriminantAnalysis<Trained> {
    fn predict(&self, X: &Array2<Float>) -> Result<Array1<i32>> {
        validate::check_is_fitted(self.classes_.is_some(), "LinearDiscriminantAnalysis")?;
        validate::check_array(X)?;

        let (n_samples, n_features) = X.dim();
        let expected_features = self.coef_.as_ref().unwrap().ncols();

        if n_features != expected_features {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                format!("Expected {} features, got {}", expected_features, n_features),
            ));
        }

        let decision_values = X.dot(self.coef_.as_ref().unwrap().t()) + self.intercept_.as_ref().unwrap();

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let max_idx = decision_values.row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            predictions[i] = self.classes_.as_ref().unwrap()[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for LinearDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, X: &Array2<Float>) -> Result<Array2<Float>> {
        validate::check_is_fitted(self.classes_.is_some(), "LinearDiscriminantAnalysis")?;
        validate::check_array(X)?;

        let (n_samples, n_features) = X.dim();
        let expected_features = self.coef_.as_ref().unwrap().ncols();

        if n_features != expected_features {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                format!("Expected {} features, got {}", expected_features, n_features),
            ));
        }

        let decision_values = X.dot(self.coef_.as_ref().unwrap().t()) + self.intercept_.as_ref().unwrap();
        let n_classes = self.classes_.as_ref().unwrap().len();

        // Apply softmax to convert decision values to probabilities
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = decision_values.row(i);
            let max_val = row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: Float = row.iter().map(|&x| (x - max_val).exp()).sum();

            for j in 0..n_classes {
                probabilities[[i, j]] = (row[j] - max_val).exp() / exp_sum;
            }
        }

        Ok(probabilities)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for LinearDiscriminantAnalysis<Trained> {
    fn transform(&self, X: &Array2<Float>) -> Result<Array2<Float>> {
        validate::check_is_fitted(self.scalings_.is_some(), "LinearDiscriminantAnalysis")?;
        validate::check_array(X)?;

        let (n_samples, n_features) = X.dim();
        let expected_features = self.scalings_.as_ref().unwrap().nrows();

        if n_features != expected_features {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                format!("Expected {} features, got {}", expected_features, n_features),
            ));
        }

        // Center the data and apply the transformation
        let centered_X = X - &self.xbar_.as_ref().unwrap();
        let transformed = centered_X.dot(self.scalings_.as_ref().unwrap());

        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_discriminant_analysis_basic() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 13.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let lda = LinearDiscriminantAnalysis::new();
        let fitted = lda.fit(&X, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.means().dim(), (2, 2));
        assert_eq!(fitted.priors().len(), 2);

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        let probabilities = fitted.predict_proba(&X).unwrap();
        assert_eq!(probabilities.dim(), (6, 2));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let sum: Float = probabilities.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_discriminant_analysis_transform() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0]
        ];
        let y = array![0, 0, 1, 1];

        let lda = LinearDiscriminantAnalysis::new()
            .n_components(Some(1));
        let fitted = lda.fit(&X, &y).unwrap();

        let transformed = fitted.transform(&X).unwrap();
        assert_eq!(transformed.dim(), (4, 1));
    }

    #[test]
    fn test_linear_discriminant_analysis_config() {
        let lda = LinearDiscriminantAnalysis::new()
            .solver("svd")
            .shrinkage(Some(0.1))
            .n_components(Some(2))
            .tol(1e-6);

        assert_eq!(lda.config.solver, "svd");
        assert_eq!(lda.config.shrinkage, Some(0.1));
        assert_eq!(lda.config.n_components, Some(2));
        assert_eq!(lda.config.tol, 1e-6);
    }
}