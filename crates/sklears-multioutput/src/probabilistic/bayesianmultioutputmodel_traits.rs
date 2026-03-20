//! # BayesianMultiOutputModel - Trait Implementations
//!
//! This module contains trait implementations for `BayesianMultiOutputModel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Estimator`
//! - `Fit`
//! - `Predict`
//! - `Estimator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BayesianMultiOutputModel<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BayesianMultiOutputModel<Untrained> {
    type Config = BayesianMultiOutputConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for BayesianMultiOutputModel<Untrained> {
    type Fitted = BayesianMultiOutputModel<BayesianMultiOutputModelTrained>;
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let (y_samples, n_outputs) = y.dim();
        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        let mut rng = thread_rng();
        let (
            weight_posterior,
            bias_posterior,
            noise_posterior,
            log_marginal_likelihood,
            elbo_history,
        ) = match self.config.inference_method {
            InferenceMethod::Variational => self.variational_inference(X, y, &mut rng)?,
            InferenceMethod::MCMC => self.mcmc_inference(X, y, &mut rng)?,
            InferenceMethod::EM => self.em_inference(X, y, &mut rng)?,
            InferenceMethod::Laplace => self.laplace_inference(X, y, &mut rng)?,
            InferenceMethod::Exact => self.exact_inference(X, y, &mut rng)?,
        };
        Ok(BayesianMultiOutputModel {
            state: BayesianMultiOutputModelTrained {
                weight_posterior,
                bias_posterior,
                noise_posterior,
                n_features,
                n_outputs,
                log_marginal_likelihood,
                elbo_history,
                config: self.config.clone(),
            },
            config: self.config,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for BayesianMultiOutputModel<BayesianMultiOutputModelTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features, n_features
            )));
        }
        let bias_1d = self.state.bias_posterior.mean.slice(s![.., 0]);
        let predictions = X.dot(&self.state.weight_posterior.mean) + bias_1d;
        Ok(predictions)
    }
}

impl Estimator for BayesianMultiOutputModel<BayesianMultiOutputModelTrained> {
    type Config = BayesianMultiOutputConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}
