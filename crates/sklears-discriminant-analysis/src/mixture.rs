//! Mixture Discriminant Analysis implementation

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Mixture Discriminant Analysis
#[derive(Debug, Clone)]
pub struct MixtureDiscriminantAnalysisConfig {
    /// Number of mixture components per class
    pub n_components_per_class: usize,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<Float>>,
    /// Regularization parameter for covariance matrices
    pub reg_param: Float,
    /// Whether to store the covariance matrices
    pub store_covariance: bool,
    /// Tolerance for EM algorithm convergence
    pub tol: Float,
    /// Maximum iterations for EM algorithm
    pub max_iter: usize,
    /// Whether to use diagonal covariance matrices
    pub diagonal_covariance: bool,
}

impl Default for MixtureDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            n_components_per_class: 2,
            priors: None,
            reg_param: 0.01,
            store_covariance: false,
            tol: 1e-6,
            max_iter: 100,
            diagonal_covariance: false,
        }
    }
}

/// Mixture Discriminant Analysis
///
/// A classifier that models each class as a mixture of Gaussian components.
/// This allows for more flexible class distributions than standard QDA.
#[derive(Debug, Clone)]
pub struct MixtureDiscriminantAnalysis<State = Untrained> {
    config: MixtureDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    mixture_weights_: Option<Vec<Array1<Float>>>, // weights for each component per class
    mixture_means_: Option<Vec<Array2<Float>>>,   // means for each component per class
    mixture_covariances_: Option<Vec<Vec<Array2<Float>>>>, // covariances for each component per class
    priors_: Option<Array1<Float>>,
    n_features_: Option<usize>,
}

impl MixtureDiscriminantAnalysis<Untrained> {
    /// Create a new MixtureDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: MixtureDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            mixture_weights_: None,
            mixture_means_: None,
            mixture_covariances_: None,
            priors_: None,
            n_features_: None,
        }
    }

    /// Set the number of mixture components per class
    pub fn n_components_per_class(mut self, n_components_per_class: usize) -> Self {
        self.config.n_components_per_class = n_components_per_class;
        self
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

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to use diagonal covariance matrices
    pub fn diagonal_covariance(mut self, diagonal_covariance: bool) -> Self {
        self.config.diagonal_covariance = diagonal_covariance;
        self
    }

    /// Initialize mixture components using k-means-style approach
    fn initialize_mixture_components(
        &self,
        class_data: &Array2<Float>,
    ) -> (Array1<Float>, Array2<Float>, Vec<Array2<Float>>) {
        let (n_samples, n_features) = class_data.dim();
        let n_components = self.config.n_components_per_class;

        // Initialize weights uniformly
        let weights = Array1::from_elem(n_components, 1.0 / n_components as Float);

        // Initialize means using simple partitioning
        let mut means = Array2::zeros((n_components, n_features));
        let samples_per_component = n_samples / n_components;

        for k in 0..n_components {
            let start_idx = k * samples_per_component;
            let end_idx = if k == n_components - 1 {
                n_samples
            } else {
                (k + 1) * samples_per_component
            };

            let component_samples = class_data.slice(s![start_idx..end_idx, ..]);
            let component_mean = component_samples.mean_axis(Axis(0)).unwrap();
            means.row_mut(k).assign(&component_mean);
        }

        // Initialize covariances
        let mut covariances = Vec::with_capacity(n_components);
        for k in 0..n_components {
            let mut cov = Array2::eye(n_features) * 0.1; // Small initial covariance
            if self.config.reg_param > 0.0 {
                for i in 0..n_features {
                    cov[[i, i]] += self.config.reg_param;
                }
            }
            covariances.push(cov);
        }

        (weights, means, covariances)
    }

    /// E-step: compute component responsibilities
    fn e_step(
        &self,
        class_data: &Array2<Float>,
        weights: &Array1<Float>,
        means: &Array2<Float>,
        covariances: &[Array2<Float>],
    ) -> Array2<Float> {
        let (n_samples, n_features) = class_data.dim();
        let n_components = weights.len();
        let mut responsibilities = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let sample = class_data.row(i);
            let mut log_probs = Array1::zeros(n_components);

            for k in 0..n_components {
                let mean_k = means.row(k);
                let cov_k = &covariances[k];

                // Compute log-likelihood
                let diff = &sample - &mean_k;

                // Simplified calculation using diagonal approximation
                let mut quad_form = 0.0;
                let mut log_det = 0.0;

                if self.config.diagonal_covariance {
                    for j in 0..n_features {
                        quad_form += diff[j] * diff[j] / cov_k[[j, j]];
                        log_det += cov_k[[j, j]].ln();
                    }
                } else {
                    // Use diagonal approximation for simplicity
                    for j in 0..n_features {
                        quad_form += diff[j] * diff[j] / cov_k[[j, j]];
                        log_det += cov_k[[j, j]].ln();
                    }
                }

                log_probs[k] = weights[k].ln() - 0.5 * (log_det + quad_form);
            }

            // Normalize using log-sum-exp trick
            let max_log_prob = log_probs.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_probs: Array1<Float> = log_probs.mapv(|x| (x - max_log_prob).exp());
            let sum_exp = exp_probs.sum();

            for k in 0..n_components {
                responsibilities[[i, k]] = exp_probs[k] / sum_exp;
            }
        }

        responsibilities
    }

    /// M-step: update parameters
    fn m_step(
        &self,
        class_data: &Array2<Float>,
        responsibilities: &Array2<Float>,
    ) -> (Array1<Float>, Array2<Float>, Vec<Array2<Float>>) {
        let (n_samples, n_features) = class_data.dim();
        let n_components = responsibilities.ncols();

        // Update weights
        let mut weights = Array1::zeros(n_components);
        for k in 0..n_components {
            weights[k] = responsibilities.column(k).sum() / n_samples as Float;
        }

        // Update means
        let mut means = Array2::zeros((n_components, n_features));
        for k in 0..n_components {
            let responsibility_sum = responsibilities.column(k).sum();
            if responsibility_sum > 1e-8 {
                for j in 0..n_features {
                    let weighted_sum: Float = class_data
                        .column(j)
                        .iter()
                        .zip(responsibilities.column(k).iter())
                        .map(|(&x, &r)| x * r)
                        .sum();
                    means[[k, j]] = weighted_sum / responsibility_sum;
                }
            }
        }

        // Update covariances
        let mut covariances = Vec::with_capacity(n_components);
        for k in 0..n_components {
            let mut cov = Array2::zeros((n_features, n_features));
            let mean_k = means.row(k);
            let responsibility_sum = responsibilities.column(k).sum();

            if responsibility_sum > 1e-8 {
                for i in 0..n_samples {
                    let sample = class_data.row(i);
                    let diff = &sample - &mean_k;
                    let weight = responsibilities[[i, k]];

                    if self.config.diagonal_covariance {
                        // Only update diagonal elements
                        for j in 0..n_features {
                            cov[[j, j]] += weight * diff[j] * diff[j];
                        }
                    } else {
                        // Full covariance matrix
                        for j in 0..n_features {
                            for l in 0..n_features {
                                cov[[j, l]] += weight * diff[j] * diff[l];
                            }
                        }
                    }
                }

                cov /= responsibility_sum;
            }

            // Add regularization
            for j in 0..n_features {
                cov[[j, j]] += self.config.reg_param;
            }

            covariances.push(cov);
        }

        (weights, means, covariances)
    }
}

impl Default for MixtureDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MixtureDiscriminantAnalysis<Untrained> {
    type Config = MixtureDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for MixtureDiscriminantAnalysis<Untrained> {
    type Fitted = MixtureDiscriminantAnalysis<Trained>;

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

        let mut mixture_weights = Vec::with_capacity(n_classes);
        let mut mixture_means = Vec::with_capacity(n_classes);
        let mut mixture_covariances = Vec::with_capacity(n_classes);
        let mut class_counts = Array1::zeros(n_classes);

        // Fit mixture model for each class
        for (i, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&yi| yi == class).collect();
            let class_samples: Vec<ArrayView1<Float>> = x
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(j, _)| class_mask[*j])
                .map(|(_, sample)| sample)
                .collect();

            class_counts[i] = class_samples.len() as Float;

            if class_samples.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has no samples",
                    class
                )));
            }

            let class_data = Array2::from_shape_fn((class_samples.len(), n_features), |(i, j)| {
                class_samples[i][j]
            });

            // Initialize mixture components
            let (mut weights, mut means, mut covariances) =
                self.initialize_mixture_components(&class_data);

            // EM algorithm
            for _ in 0..self.config.max_iter {
                // E-step
                let responsibilities = self.e_step(&class_data, &weights, &means, &covariances);

                // M-step
                let (new_weights, new_means, new_covariances) =
                    self.m_step(&class_data, &responsibilities);

                // Check convergence
                let weight_diff = (&new_weights - &weights).mapv(|x| x.abs()).sum();
                if weight_diff < self.config.tol {
                    break;
                }

                weights = new_weights;
                means = new_means;
                covariances = new_covariances;
            }

            mixture_weights.push(weights);
            mixture_means.push(means);
            mixture_covariances.push(covariances);
        }

        // Calculate priors
        let priors = if let Some(ref p) = self.config.priors {
            p.clone()
        } else {
            &class_counts / n_samples as Float
        };

        let store_covariance = self.config.store_covariance;
        Ok(MixtureDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(Array1::from(classes)),
            mixture_weights_: Some(mixture_weights),
            mixture_means_: Some(mixture_means),
            mixture_covariances_: if store_covariance {
                Some(mixture_covariances)
            } else {
                Some(mixture_covariances) // MDA always needs covariances for prediction
            },
            priors_: Some(priors),
            n_features_: Some(n_features),
        })
    }
}

impl MixtureDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().expect("Model is trained")
    }

    /// Get the mixture weights
    pub fn mixture_weights(&self) -> &Vec<Array1<Float>> {
        self.mixture_weights_.as_ref().expect("Model is trained")
    }

    /// Get the mixture means
    pub fn mixture_means(&self) -> &Vec<Array2<Float>> {
        self.mixture_means_.as_ref().expect("Model is trained")
    }

    /// Get the mixture covariances
    pub fn mixture_covariances(&self) -> &Vec<Vec<Array2<Float>>> {
        self.mixture_covariances_
            .as_ref()
            .expect("Model is trained")
    }

    /// Get the priors
    pub fn priors(&self) -> &Array1<Float> {
        self.priors_.as_ref().expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MixtureDiscriminantAnalysis<Trained> {
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

impl PredictProba<Array2<Float>, Array2<Float>> for MixtureDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let mixture_weights = self.mixture_weights_.as_ref().expect("Model is trained");
        let mixture_means = self.mixture_means_.as_ref().expect("Model is trained");
        let mixture_covariances = self
            .mixture_covariances_
            .as_ref()
            .expect("Model is trained");
        let priors = self.priors_.as_ref().expect("Model is trained");
        let n_classes = mixture_weights.len();
        let n_samples = x.nrows();

        let mut log_likelihoods = Array2::zeros((n_samples, n_classes));

        for i in 0..n_classes {
            let class_weights = &mixture_weights[i];
            let class_means = &mixture_means[i];
            let class_covariances = &mixture_covariances[i];
            let n_components = class_weights.len();

            for j in 0..n_samples {
                let sample = x.row(j);
                let mut component_likelihoods = Array1::zeros(n_components);

                for k in 0..n_components {
                    let mean_k = class_means.row(k);
                    let cov_k = &class_covariances[k];
                    let diff = &sample - &mean_k;

                    // Simplified likelihood calculation
                    let mut quad_form = 0.0;
                    let mut log_det = 0.0;

                    if self.config.diagonal_covariance {
                        for l in 0..n_features {
                            quad_form += diff[l] * diff[l] / cov_k[[l, l]];
                            log_det += cov_k[[l, l]].ln();
                        }
                    } else {
                        // Use diagonal approximation
                        for l in 0..n_features {
                            quad_form += diff[l] * diff[l] / cov_k[[l, l]];
                            log_det += cov_k[[l, l]].ln();
                        }
                    }

                    component_likelihoods[k] =
                        class_weights[k] * (-0.5 * (log_det + quad_form)).exp();
                }

                log_likelihoods[[j, i]] = priors[i].ln() + component_likelihoods.sum().ln();
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
