//! Heteroscedastic Discriminant Analysis implementation

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Heteroscedastic Discriminant Analysis
#[derive(Debug, Clone)]
pub struct HeteroscedasticDiscriminantAnalysisConfig {
    /// Prior probabilities of the classes
    pub priors: Option<Array1<Float>>,
    /// Regularization parameter for covariance matrices
    pub reg_param: Float,
    /// Whether to store the covariance matrices
    pub store_covariance: bool,
    /// Tolerance for stopping criteria
    pub tol: Float,
    /// Covariance structure type
    pub covariance_type: String,
    /// Whether to use adaptive regularization
    pub adaptive_regularization: bool,
    /// Adaptive regularization method
    pub adaptive_method: String,
    /// Shrinkage parameter for covariance estimation
    pub shrinkage: Option<Float>,
}

impl Default for HeteroscedasticDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            priors: None,
            reg_param: 0.01,
            store_covariance: false,
            tol: 1e-4,
            covariance_type: "full".to_string(), // "full", "tied", "diag", "spherical"
            adaptive_regularization: false,
            adaptive_method: "ledoit_wolf".to_string(),
            shrinkage: None,
        }
    }
}

/// Heteroscedastic Discriminant Analysis
///
/// A generalization of LDA and QDA that allows for flexible covariance structures.
/// This method can handle different assumptions about the covariance matrices:
/// - "full": Each class has its own full covariance matrix (like QDA)
/// - "tied": All classes share the same covariance matrix (like LDA)
/// - "diag": Each class has its own diagonal covariance matrix
/// - "spherical": Each class has its own spherical covariance matrix (σ²I)
#[derive(Debug, Clone)]
pub struct HeteroscedasticDiscriminantAnalysis<State = Untrained> {
    config: HeteroscedasticDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    means_: Option<Array2<Float>>,
    covariances_: Option<Vec<Array2<Float>>>,
    shared_covariance_: Option<Array2<Float>>, // For tied covariance
    priors_: Option<Array1<Float>>,
    n_features_: Option<usize>,
}

impl HeteroscedasticDiscriminantAnalysis<Untrained> {
    /// Create a new HeteroscedasticDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: HeteroscedasticDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            means_: None,
            covariances_: None,
            shared_covariance_: None,
            priors_: None,
            n_features_: None,
        }
    }

    /// Set the prior probabilities
    pub fn priors(mut self, priors: Option<Array1<Float>>) -> Self {
        self.config.priors = priors;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set whether to store covariance matrices
    pub fn store_covariance(mut self, store_covariance: bool) -> Self {
        self.config.store_covariance = store_covariance;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the covariance structure type
    pub fn covariance_type(mut self, covariance_type: &str) -> Self {
        self.config.covariance_type = covariance_type.to_string();
        self
    }

    /// Set whether to use adaptive regularization
    pub fn adaptive_regularization(mut self, adaptive_regularization: bool) -> Self {
        self.config.adaptive_regularization = adaptive_regularization;
        self
    }

    /// Set the adaptive regularization method
    pub fn adaptive_method(mut self, adaptive_method: &str) -> Self {
        self.config.adaptive_method = adaptive_method.to_string();
        self
    }

    /// Set the shrinkage parameter
    pub fn shrinkage(mut self, shrinkage: Option<Float>) -> Self {
        self.config.shrinkage = shrinkage;
        self
    }

    /// Compute class-specific covariance matrix
    fn compute_class_covariance(
        &self,
        class_data: &Array2<Float>,
        class_mean: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = class_data.dim();

        if n_samples <= 1 {
            return Ok(Array2::eye(n_features) * self.config.reg_param);
        }

        match self.config.covariance_type.as_str() {
            "full" => self.compute_full_covariance(class_data, class_mean),
            "diag" => self.compute_diagonal_covariance(class_data, class_mean),
            "spherical" => self.compute_spherical_covariance(class_data, class_mean),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown covariance type: {}",
                self.config.covariance_type
            ))),
        }
    }

    /// Compute full covariance matrix
    fn compute_full_covariance(
        &self,
        class_data: &Array2<Float>,
        class_mean: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = class_data.dim();
        let mut cov = Array2::zeros((n_features, n_features));

        // Compute sample covariance
        for i in 0..n_samples {
            let sample = class_data.row(i);
            let diff = &sample - class_mean;
            for j in 0..n_features {
                for k in 0..n_features {
                    cov[[j, k]] += diff[j] * diff[k];
                }
            }
        }

        if n_samples > 1 {
            cov /= (n_samples - 1) as Float;
        }

        // Apply regularization or shrinkage
        if let Some(shrinkage) = self.config.shrinkage {
            let identity = Array2::eye(n_features);
            let trace = cov.diag().sum() / n_features as Float;
            cov = (1.0 - shrinkage) * cov + shrinkage * trace * identity;
        } else if self.config.adaptive_regularization {
            let adaptive_param = self.compute_adaptive_regularization(class_data, &cov)?;
            match self.config.adaptive_method.as_str() {
                "ledoit_wolf" | "oas" => {
                    let identity = Array2::eye(n_features);
                    let trace = cov.diag().sum() / n_features as Float;
                    cov = (1.0 - adaptive_param) * cov + adaptive_param * trace * identity;
                }
                _ => {
                    for i in 0..n_features {
                        cov[[i, i]] += adaptive_param;
                    }
                }
            }
        } else {
            // Add regularization to diagonal
            for i in 0..n_features {
                cov[[i, i]] += self.config.reg_param;
            }
        }

        Ok(cov)
    }

    /// Compute diagonal covariance matrix
    fn compute_diagonal_covariance(
        &self,
        class_data: &Array2<Float>,
        class_mean: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = class_data.dim();
        let mut cov = Array2::zeros((n_features, n_features));

        // Compute diagonal elements only
        for i in 0..n_samples {
            let sample = class_data.row(i);
            let diff = &sample - class_mean;
            for j in 0..n_features {
                cov[[j, j]] += diff[j] * diff[j];
            }
        }

        if n_samples > 1 {
            for j in 0..n_features {
                cov[[j, j]] /= (n_samples - 1) as Float;
            }
        }

        // Add regularization to diagonal
        for i in 0..n_features {
            cov[[i, i]] += self.config.reg_param;
        }

        Ok(cov)
    }

    /// Compute spherical covariance matrix (σ²I)
    fn compute_spherical_covariance(
        &self,
        class_data: &Array2<Float>,
        class_mean: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = class_data.dim();
        let mut variance_sum = 0.0;

        // Compute pooled variance across all features
        for i in 0..n_samples {
            let sample = class_data.row(i);
            let diff = &sample - class_mean;
            for j in 0..n_features {
                variance_sum += diff[j] * diff[j];
            }
        }

        let pooled_variance = if n_samples > 1 {
            variance_sum / ((n_samples - 1) * n_features) as Float
        } else {
            1.0
        };

        let variance_with_reg = pooled_variance + self.config.reg_param;
        Ok(Array2::eye(n_features) * variance_with_reg)
    }

    /// Compute tied (shared) covariance matrix across all classes
    fn compute_tied_covariance(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        means: &Array2<Float>,
        classes: &[i32],
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let mut pooled_cov = Array2::zeros((n_features, n_features));
        let mut total_samples = 0;

        // Pool covariance across all classes
        for (i, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&yi| yi == class).collect();
            let class_samples: Vec<ArrayView1<Float>> = x
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(j, _)| class_mask[*j])
                .map(|(_, sample)| sample)
                .collect();

            let class_mean = means.row(i);
            let n_class_samples = class_samples.len();

            for sample in class_samples {
                let diff = &sample - &class_mean;
                for j in 0..n_features {
                    for k in 0..n_features {
                        pooled_cov[[j, k]] += diff[j] * diff[k];
                    }
                }
            }
            total_samples += n_class_samples;
        }

        if total_samples > classes.len() {
            pooled_cov /= (total_samples - classes.len()) as Float;
        }

        // Apply regularization or shrinkage
        if let Some(shrinkage) = self.config.shrinkage {
            let identity = Array2::eye(n_features);
            let trace = pooled_cov.diag().sum() / n_features as Float;
            pooled_cov = (1.0 - shrinkage) * pooled_cov + shrinkage * trace * identity;
        } else if self.config.adaptive_regularization {
            let adaptive_param = self.compute_adaptive_regularization(x, &pooled_cov)?;
            match self.config.adaptive_method.as_str() {
                "ledoit_wolf" | "oas" => {
                    let identity = Array2::eye(n_features);
                    let trace = pooled_cov.diag().sum() / n_features as Float;
                    pooled_cov =
                        (1.0 - adaptive_param) * pooled_cov + adaptive_param * trace * identity;
                }
                _ => {
                    for i in 0..n_features {
                        pooled_cov[[i, i]] += adaptive_param;
                    }
                }
            }
        } else {
            // Add regularization to diagonal
            for i in 0..n_features {
                pooled_cov[[i, i]] += self.config.reg_param;
            }
        }

        Ok(pooled_cov)
    }

    /// Compute adaptive regularization parameter (simplified version)
    fn compute_adaptive_regularization(
        &self,
        data: &Array2<Float>,
        sample_cov: &Array2<Float>,
    ) -> Result<Float> {
        match self.config.adaptive_method.as_str() {
            "ledoit_wolf" => self.ledoit_wolf_shrinkage(data, sample_cov),
            "oas" => self.oas_shrinkage(data, sample_cov),
            _ => Ok(0.1), // Default adaptive parameter
        }
    }

    /// Simplified Ledoit-Wolf shrinkage estimator
    fn ledoit_wolf_shrinkage(
        &self,
        data: &Array2<Float>,
        sample_cov: &Array2<Float>,
    ) -> Result<Float> {
        let (n_samples, n_features) = data.dim();

        if n_samples <= n_features {
            return Ok(1.0);
        }

        let trace = sample_cov.diag().sum();
        let mu = trace / n_features as Float;

        // Simplified shrinkage estimation
        let shrinkage = ((n_features as Float) / (n_samples as Float))
            .min(1.0)
            .max(0.0);

        Ok(shrinkage)
    }

    /// Simplified OAS shrinkage estimator
    fn oas_shrinkage(&self, data: &Array2<Float>, sample_cov: &Array2<Float>) -> Result<Float> {
        let (n_samples, n_features) = data.dim();

        if n_samples <= n_features {
            return Ok(1.0);
        }

        // OAS shrinkage formula (simplified)
        let rho = ((1.0 - 2.0 / n_features as Float)
            / (n_samples as Float + 1.0 - 2.0 / n_features as Float))
            .min(1.0)
            .max(0.0);

        Ok(rho)
    }
}

impl Default for HeteroscedasticDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> HeteroscedasticDiscriminantAnalysis<State> {
    /// Compute matrix determinant (simplified - product of diagonal)
    fn compute_log_determinant(&self, matrix: &Array2<Float>) -> Float {
        matrix.diag().iter().map(|&x| x.ln()).sum()
    }

    /// Compute inverse using pseudo-inverse approach
    fn compute_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();

        match self.config.covariance_type.as_str() {
            "diag" | "spherical" => {
                // For diagonal matrices, inverse is easy
                let mut inv = Array2::zeros((n, n));
                for i in 0..n {
                    inv[[i, i]] = 1.0 / matrix[[i, i]];
                }
                Ok(inv)
            }
            _ => {
                // For full matrices, use simplified inverse
                let mut inv = matrix.clone();
                for i in 0..n {
                    if inv[[i, i]].abs() < self.config.tol {
                        inv[[i, i]] += self.config.reg_param;
                    }
                }

                // Simple diagonal approximation for inverse
                let mut result = Array2::zeros((n, n));
                for i in 0..n {
                    result[[i, i]] = 1.0 / inv[[i, i]];
                }
                Ok(result)
            }
        }
    }
}

impl Estimator for HeteroscedasticDiscriminantAnalysis<Untrained> {
    type Config = HeteroscedasticDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for HeteroscedasticDiscriminantAnalysis<Untrained> {
    type Fitted = HeteroscedasticDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        // Basic validation
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Calculate class means
        let mut means = Array2::zeros((n_classes, n_features));
        let mut class_counts = Array1::zeros(n_classes);

        for (i, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&yi| yi == class).collect();
            let class_samples: Vec<ArrayView1<Float>> = x
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(j, _)| class_mask[*j])
                .map(|(_, sample)| sample)
                .collect();

            let count = class_samples.len();
            class_counts[i] = count as Float;

            if count == 0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has no samples",
                    class
                )));
            }

            // Calculate class mean
            let mut class_mean = Array1::zeros(n_features);
            for sample in &class_samples {
                for j in 0..n_features {
                    class_mean[j] += sample[j];
                }
            }
            class_mean /= count as Float;
            means.row_mut(i).assign(&class_mean);
        }

        // Calculate covariances based on covariance type
        let (covariances, shared_covariance) = if self.config.covariance_type == "tied" {
            // Compute shared covariance matrix
            let shared_cov = self.compute_tied_covariance(x, y, &means, &classes)?;
            (None, Some(shared_cov))
        } else {
            // Compute class-specific covariances
            let mut covariances = Vec::with_capacity(n_classes);

            for (i, &class) in classes.iter().enumerate() {
                let class_mask: Vec<bool> = y.iter().map(|&yi| yi == class).collect();
                let class_samples: Vec<ArrayView1<Float>> = x
                    .axis_iter(Axis(0))
                    .enumerate()
                    .filter(|(j, _)| class_mask[*j])
                    .map(|(_, sample)| sample)
                    .collect();

                let class_data =
                    Array2::from_shape_fn((class_samples.len(), n_features), |(i, j)| {
                        class_samples[i][j]
                    });

                let class_mean = means.row(i);
                let class_cov =
                    self.compute_class_covariance(&class_data, &class_mean.to_owned())?;
                covariances.push(class_cov);
            }

            (Some(covariances), None)
        };

        // Calculate priors
        let priors = if let Some(ref p) = self.config.priors {
            p.clone()
        } else {
            &class_counts / n_samples as Float
        };

        let store_covariance = self.config.store_covariance;
        let covariance_type = self.config.covariance_type.clone();
        Ok(HeteroscedasticDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(Array1::from(classes)),
            means_: Some(means),
            covariances_: if store_covariance || covariance_type != "tied" {
                covariances
            } else {
                None
            },
            shared_covariance_: shared_covariance,
            priors_: Some(priors),
            n_features_: Some(n_features),
        })
    }
}

impl HeteroscedasticDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().expect("Model is trained")
    }

    /// Get the means
    pub fn means(&self) -> &Array2<Float> {
        self.means_.as_ref().expect("Model is trained")
    }

    /// Get the priors
    pub fn priors(&self) -> &Array1<Float> {
        self.priors_.as_ref().expect("Model is trained")
    }

    /// Get the covariance matrices (if not tied)
    pub fn covariances(&self) -> Option<&Vec<Array2<Float>>> {
        self.covariances_.as_ref()
    }

    /// Get the shared covariance matrix (if tied)
    pub fn shared_covariance(&self) -> Option<&Array2<Float>> {
        self.shared_covariance_.as_ref()
    }
}

impl Predict<Array2<Float>, Array1<i32>> for HeteroscedasticDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().expect("Model is trained");

        let predictions: Vec<i32> = probas
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from(predictions))
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for HeteroscedasticDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let means = self.means_.as_ref().expect("Model is trained");
        let priors = self.priors_.as_ref().expect("Model is trained");
        let n_classes = means.nrows();
        let n_samples = x.nrows();

        let mut log_likelihoods = Array2::zeros((n_samples, n_classes));

        for i in 0..n_classes {
            let class_mean = means.row(i);

            // Get covariance matrix for this class
            let class_cov = if let Some(ref shared_cov) = self.shared_covariance_ {
                shared_cov
            } else if let Some(ref covariances) = self.covariances_ {
                &covariances[i]
            } else {
                return Err(SklearsError::InvalidInput(
                    "No covariance matrices available".to_string(),
                ));
            };

            let log_det = self.compute_log_determinant(class_cov);
            let cov_inv = self.compute_inverse(class_cov)?;

            for j in 0..n_samples {
                let sample = x.row(j);
                let diff = &sample - &class_mean;

                // Compute quadratic form
                let quad_form = match self.config.covariance_type.as_str() {
                    "diag" | "spherical" => {
                        // For diagonal matrices, quadratic form is simplified
                        let mut sum = 0.0;
                        for k in 0..n_features {
                            sum += diff[k] * diff[k] / class_cov[[k, k]];
                        }
                        sum
                    }
                    _ => {
                        // For full matrices, use simplified diagonal approximation
                        let mut sum = 0.0;
                        for k in 0..n_features {
                            sum += diff[k] * diff[k] * cov_inv[[k, k]];
                        }
                        sum
                    }
                };

                log_likelihoods[[j, i]] = priors[i].ln() - 0.5 * (log_det + quad_form);
            }
        }

        // Convert log-likelihoods to probabilities using softmax
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = log_likelihoods.row(i);
            let max_log_like = row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let exp_likes: Vec<Float> = row.iter().map(|&x| (x - max_log_like).exp()).collect();
            let sum_exp: Float = exp_likes.iter().sum();

            for j in 0..n_classes {
                probabilities[[i, j]] = exp_likes[j] / sum_exp;
            }
        }

        Ok(probabilities)
    }
}
