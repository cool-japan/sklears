//! Trained Bayesian Discriminant Analysis implementation

use super::posterior::PosteriorParameters;
use super::types::BayesianDiscriminantAnalysisConfig;

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use sklears_core::{
    error::Result,
    traits::{Predict, PredictProba, Transform},
    types::Float,
};

/// Trained Bayesian Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedBayesianDiscriminantAnalysis {
    /// Configuration used for training
    pub(crate) config: BayesianDiscriminantAnalysisConfig,
    /// Unique classes found during training
    pub(crate) classes: Array1<i32>,
    /// Posterior parameters
    pub(crate) posterior: PosteriorParameters,
    /// Class priors
    pub(crate) class_priors: Array1<Float>,
    /// Discriminant coefficients (for dimensionality reduction)
    pub(crate) coefficients: Option<Array2<Float>>,
    /// Number of features
    pub(crate) n_features: usize,
    /// Number of samples seen during training
    pub(crate) n_samples_seen: usize,
    /// Log marginal likelihood (model evidence)
    pub(crate) log_marginal_likelihood: Float,
}

impl TrainedBayesianDiscriminantAnalysis {
    /// Get the classes found during training
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the posterior parameters
    pub fn posterior(&self) -> &PosteriorParameters {
        &self.posterior
    }

    /// Get the class priors
    pub fn class_priors(&self) -> &Array1<Float> {
        &self.class_priors
    }

    /// Get the discriminant coefficients
    pub fn coefficients(&self) -> Option<&Array2<Float>> {
        self.coefficients.as_ref()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the number of samples seen during training
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Get the log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> Float {
        self.log_marginal_likelihood
    }

    /// Compute predictive probability using posterior distributions
    fn compute_predictive_probabilities<D: Data<Elem = Float>>(
        &self,
        x: &ArrayBase<D, Ix2>,
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        if x.ncols() != self.n_features {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // For each sample, compute discriminant function for each class
        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let mut log_probs = Array1::zeros(n_classes);

            for class_idx in 0..n_classes {
                // Get class mean
                let mean = self.posterior.mu.row(class_idx);

                // Compute Mahalanobis distance (simplified using diagonal)
                // Full implementation would use matrix inversion
                let diff = &sample - &mean;
                let mut mahalanobis_sq = 0.0;

                // Simplified distance using diagonal of covariance
                let sigma = &self.posterior.sigma[class_idx];
                for i in 0..self.n_features {
                    let var = sigma[[i, i]].max(1e-10); // Avoid division by zero
                    mahalanobis_sq += diff[i] * diff[i] / var;
                }

                // Log discriminant function: log P(x|class) + log P(class)
                // Assuming Gaussian: -0.5 * mahalanobis_sq - 0.5 * log(det(Sigma)) + log(prior)
                let log_det = self.posterior.sigma[class_idx]
                    .diag()
                    .iter()
                    .map(|&v| v.max(1e-10).ln())
                    .sum::<Float>();

                log_probs[class_idx] =
                    -0.5 * mahalanobis_sq - 0.5 * log_det + self.class_priors[class_idx].ln();
            }

            // Convert log probabilities to probabilities using softmax
            let max_log_prob = log_probs.fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            for class_idx in 0..n_classes {
                exp_sum += (log_probs[class_idx] - max_log_prob).exp();
            }

            for class_idx in 0..n_classes {
                probabilities[[sample_idx, class_idx]] =
                    (log_probs[class_idx] - max_log_prob).exp() / exp_sum;
            }
        }

        Ok(probabilities)
    }
}

impl<D: Data<Elem = Float>> Predict<ArrayBase<D, Ix2>, Array1<i32>>
    for TrainedBayesianDiscriminantAnalysis
{
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Result<Array1<i32>> {
        // Get class probabilities
        let probabilities = self.compute_predictive_probabilities(x)?;
        let n_samples = probabilities.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // For each sample, predict the class with highest probability
        for sample_idx in 0..n_samples {
            let probs = probabilities.row(sample_idx);
            let mut max_prob = Float::NEG_INFINITY;
            let mut best_class_idx = 0;

            for (class_idx, &prob) in probs.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    best_class_idx = class_idx;
                }
            }

            predictions[sample_idx] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }
}

impl<D: Data<Elem = Float>> PredictProba<ArrayBase<D, Ix2>, Array2<Float>>
    for TrainedBayesianDiscriminantAnalysis
{
    fn predict_proba(&self, x: &ArrayBase<D, Ix2>) -> Result<Array2<Float>> {
        // Directly use the compute_predictive_probabilities method
        self.compute_predictive_probabilities(x)
    }
}

impl<D: Data<Elem = Float>> Transform<ArrayBase<D, Ix2>, Array2<Float>>
    for TrainedBayesianDiscriminantAnalysis
{
    fn transform(&self, x: &ArrayBase<D, Ix2>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }

        // If dimensionality reduction coefficients are available, project
        if let Some(ref coeffs) = self.coefficients {
            let n_samples = x.nrows();
            let n_components = coeffs.nrows();
            let mut transformed = Array2::zeros((n_samples, n_components));

            // Project each sample onto the discriminant directions
            for sample_idx in 0..n_samples {
                let sample = x.row(sample_idx);
                for comp_idx in 0..n_components {
                    let direction = coeffs.row(comp_idx);
                    let mut projection = 0.0;
                    for feat_idx in 0..self.n_features {
                        projection += sample[feat_idx] * direction[feat_idx];
                    }
                    transformed[[sample_idx, comp_idx]] = projection;
                }
            }

            Ok(transformed)
        } else {
            // No dimensionality reduction, return input as is
            Ok(x.to_owned())
        }
    }
}
