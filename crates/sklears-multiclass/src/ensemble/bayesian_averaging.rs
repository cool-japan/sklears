//! Bayesian Model Averaging for Ensemble Combination
//!
//! Implements Bayesian Model Averaging (BMA) to combine predictions from multiple models
//! using posterior probabilities as weights.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Bayesian Model Averaging configuration
#[derive(Debug, Clone)]
pub struct BMAConfig {
    /// Prior distribution type
    pub prior_type: PriorType,
    /// Prior concentration parameter (for Dirichlet)
    pub alpha: f64,
    /// Whether to use log probabilities for numerical stability
    pub use_log_probs: bool,
    /// Minimum probability threshold
    pub min_prob: f64,
}

impl Default for BMAConfig {
    fn default() -> Self {
        Self {
            prior_type: PriorType::Uniform,
            alpha: 1.0,
            use_log_probs: true,
            min_prob: 1e-10,
        }
    }
}

/// Type of prior distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorType {
    /// Uniform prior (all models equally likely)
    Uniform,
    /// Dirichlet prior (with concentration parameter)
    Dirichlet,
    /// Empirical prior (based on validation performance)
    Empirical,
}

/// Bayesian Model Averaging ensemble
#[derive(Debug, Clone)]
pub struct BayesianModelAveraging {
    config: BMAConfig,
    /// Posterior model probabilities
    model_posteriors: Option<Array1<f64>>,
    /// Number of models in ensemble
    n_models: usize,
    /// Number of classes
    n_classes: usize,
}

impl BayesianModelAveraging {
    /// Create a new BMA ensemble
    pub fn new(config: BMAConfig) -> Self {
        Self {
            config,
            model_posteriors: None,
            n_models: 0,
            n_classes: 0,
        }
    }

    /// Fit BMA using validation data
    ///
    /// # Arguments
    /// * `predictions` - Predictions from each model [n_models, n_samples, n_classes]
    /// * `y_true` - True labels [n_samples]
    pub fn fit(&mut self, predictions: &[Array2<f64>], y_true: &Array1<i32>) -> SklResult<()> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions provided".to_string(),
            ));
        }

        self.n_models = predictions.len();
        let (n_samples, n_classes) = predictions[0].dim();
        self.n_classes = n_classes;

        if y_true.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of labels doesn't match predictions".to_string(),
            ));
        }

        // Compute log-likelihood for each model
        let log_likelihoods = self.compute_log_likelihoods(predictions, y_true)?;

        // Compute posterior probabilities
        self.model_posteriors = Some(self.compute_posteriors(&log_likelihoods)?);

        Ok(())
    }

    /// Compute log-likelihood for each model
    fn compute_log_likelihoods(
        &self,
        predictions: &[Array2<f64>],
        y_true: &Array1<i32>,
    ) -> SklResult<Array1<f64>> {
        let n_samples = y_true.len();
        let mut log_likelihoods = Array1::zeros(self.n_models);

        for (model_idx, preds) in predictions.iter().enumerate() {
            let mut log_likelihood = 0.0;

            for i in 0..n_samples {
                let true_class = y_true[i] as usize;
                if true_class >= self.n_classes {
                    return Err(SklearsError::InvalidInput(format!(
                        "Class label {} out of range [0, {})",
                        true_class, self.n_classes
                    )));
                }

                let prob = preds[[i, true_class]].max(self.config.min_prob);
                log_likelihood += prob.ln();
            }

            log_likelihoods[model_idx] = log_likelihood;
        }

        Ok(log_likelihoods)
    }

    /// Compute posterior model probabilities
    fn compute_posteriors(&self, log_likelihoods: &Array1<f64>) -> SklResult<Array1<f64>> {
        let mut posteriors = match self.config.prior_type {
            PriorType::Uniform => {
                // Uniform prior
                Array1::from_elem(self.n_models, 1.0 / self.n_models as f64)
            }
            PriorType::Dirichlet => {
                // Dirichlet prior
                Array1::from_elem(self.n_models, self.config.alpha)
            }
            PriorType::Empirical => {
                // Will be computed from likelihood
                Array1::from_elem(self.n_models, 1.0)
            }
        };

        // Compute unnormalized posteriors (prior * likelihood)
        for i in 0..self.n_models {
            if self.config.use_log_probs {
                posteriors[i] = (posteriors[i].ln() + log_likelihoods[i]).exp();
            } else {
                posteriors[i] *= log_likelihoods[i].exp();
            }
        }

        // Normalize
        let total: f64 = posteriors.sum();
        if total > 0.0 {
            posteriors /= total;
        } else {
            // Fallback to uniform if normalization fails
            posteriors.fill(1.0 / self.n_models as f64);
        }

        Ok(posteriors)
    }

    /// Predict class probabilities using BMA
    pub fn predict_proba(&self, predictions: &[Array2<f64>]) -> SklResult<Array2<f64>> {
        let posteriors = self
            .model_posteriors
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("BMA not fitted yet".to_string()))?;

        if predictions.len() != self.n_models {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} models, got {}",
                self.n_models,
                predictions.len()
            )));
        }

        let (n_samples, n_classes) = predictions[0].dim();
        let mut combined = Array2::zeros((n_samples, n_classes));

        // Weighted average of predictions
        for (model_idx, preds) in predictions.iter().enumerate() {
            let weight = posteriors[model_idx];
            combined = combined + preds * weight;
        }

        // Normalize probabilities
        for i in 0..n_samples {
            let row_sum: f64 = combined.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_classes {
                    combined[[i, j]] /= row_sum;
                }
            }
        }

        Ok(combined)
    }

    /// Predict class labels using BMA
    pub fn predict(&self, predictions: &[Array2<f64>]) -> SklResult<Array1<i32>> {
        let proba = self.predict_proba(predictions)?;
        let n_samples = proba.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = proba.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = max_idx as i32;
        }

        Ok(predictions)
    }

    /// Get model posterior probabilities
    pub fn model_weights(&self) -> Option<&Array1<f64>> {
        self.model_posteriors.as_ref()
    }

    /// Compute model evidence (marginal likelihood)
    pub fn model_evidence(
        &self,
        predictions: &[Array2<f64>],
        y_true: &Array1<i32>,
    ) -> SklResult<f64> {
        let log_likelihoods = self.compute_log_likelihoods(predictions, y_true)?;

        // Compute log marginal likelihood
        let mut log_evidence = f64::NEG_INFINITY;

        for &log_likelihood in log_likelihoods.iter() {
            let log_prior = match self.config.prior_type {
                PriorType::Uniform => -(self.n_models as f64).ln(),
                PriorType::Dirichlet => {
                    // Simplified Dirichlet log prior
                    (self.config.alpha - 1.0) * (1.0 / self.n_models as f64).ln()
                }
                PriorType::Empirical => 0.0,
            };

            let log_joint = log_prior + log_likelihood;
            log_evidence = log_sum_exp(log_evidence, log_joint);
        }

        Ok(log_evidence.exp())
    }

    /// Get number of models
    pub fn n_models(&self) -> usize {
        self.n_models
    }

    /// Get effective number of models (based on entropy of posteriors)
    pub fn effective_n_models(&self) -> Option<f64> {
        self.model_posteriors.as_ref().map(|posteriors| {
            let entropy: f64 = posteriors
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();
            entropy.exp()
        })
    }
}

/// Log-sum-exp trick for numerical stability
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a < 0.0 {
        return b;
    }
    if b.is_infinite() && b < 0.0 {
        return a;
    }

    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_bma_config_default() {
        let config = BMAConfig::default();
        assert_eq!(config.prior_type, PriorType::Uniform);
        assert_eq!(config.alpha, 1.0);
        assert!(config.use_log_probs);
    }

    #[test]
    fn test_bma_creation() {
        let bma = BayesianModelAveraging::new(BMAConfig::default());
        assert_eq!(bma.n_models, 0);
        assert!(bma.model_posteriors.is_none());
    }

    #[test]
    fn test_bma_fit() {
        let mut bma = BayesianModelAveraging::new(BMAConfig::default());

        // Create mock predictions from 2 models
        let pred1 = array![[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]];
        let pred2 = array![[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]];
        let predictions = vec![pred1, pred2];

        let y_true = array![0, 0, 0];

        assert!(bma.fit(&predictions, &y_true).is_ok());
        assert_eq!(bma.n_models(), 2);
        assert!(bma.model_weights().is_some());
    }

    #[test]
    fn test_bma_predict_proba() {
        let mut bma = BayesianModelAveraging::new(BMAConfig::default());

        let pred1 = array![[0.9, 0.1], [0.2, 0.8]];
        let pred2 = array![[0.8, 0.2], [0.3, 0.7]];
        let predictions = vec![pred1.clone(), pred2.clone()];

        let y_true = array![0, 1];
        bma.fit(&predictions, &y_true).unwrap();

        let combined = bma.predict_proba(&predictions).unwrap();
        assert_eq!(combined.dim(), (2, 2));

        // Check probabilities sum to 1
        for i in 0..2 {
            let row_sum: f64 = combined.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bma_predict() {
        let mut bma = BayesianModelAveraging::new(BMAConfig::default());

        let pred1 = array![[0.9, 0.1], [0.2, 0.8]];
        let pred2 = array![[0.8, 0.2], [0.3, 0.7]];
        let predictions = vec![pred1.clone(), pred2.clone()];

        let y_true = array![0, 1];
        bma.fit(&predictions, &y_true).unwrap();

        let labels = bma.predict(&predictions).unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_bma_uniform_prior() {
        let config = BMAConfig {
            prior_type: PriorType::Uniform,
            ..Default::default()
        };
        let mut bma = BayesianModelAveraging::new(config);

        let pred1 = array![[0.9, 0.1], [0.8, 0.2]];
        let pred2 = array![[0.7, 0.3], [0.6, 0.4]];
        let predictions = vec![pred1, pred2];

        let y_true = array![0, 0];
        bma.fit(&predictions, &y_true).unwrap();

        let weights = bma.model_weights().unwrap();
        // Both models should have non-zero weight
        assert!(weights[0] > 0.0);
        assert!(weights[1] > 0.0);
        // Weights should sum to 1
        assert!((weights.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bma_effective_models() {
        let mut bma = BayesianModelAveraging::new(BMAConfig::default());

        let pred1 = array![[0.9, 0.1]];
        let pred2 = array![[0.8, 0.2]];
        let pred3 = array![[0.7, 0.3]];
        let predictions = vec![pred1, pred2, pred3];

        let y_true = array![0];
        bma.fit(&predictions, &y_true).unwrap();

        let eff = bma.effective_n_models().unwrap();
        // Effective number should be between 1 and 3
        assert!(eff >= 1.0 && eff <= 3.0);
    }

    #[test]
    fn test_log_sum_exp() {
        let a = 1.0_f64.ln();
        let b = 2.0_f64.ln();
        let result = log_sum_exp(a, b);
        let expected = 3.0_f64.ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bma_model_evidence() {
        let mut bma = BayesianModelAveraging::new(BMAConfig::default());

        let pred1 = array![[0.9, 0.1], [0.8, 0.2]];
        let pred2 = array![[0.7, 0.3], [0.6, 0.4]];
        let predictions = vec![pred1.clone(), pred2.clone()];

        let y_true = array![0, 0];
        bma.fit(&predictions, &y_true).unwrap();

        let evidence = bma.model_evidence(&predictions, &y_true).unwrap();
        assert!(evidence > 0.0);
    }
}
