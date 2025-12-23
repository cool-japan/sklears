//! # EnsembleBayesianModel - Trait Implementations
//!
//! This module contains trait implementations for `EnsembleBayesianModel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Fit`
//! - `Predict`
//! - `Estimator`
//! - `Estimator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for EnsembleBayesianModel<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for EnsembleBayesianModel<Untrained> {
    type Fitted = EnsembleBayesianModel<EnsembleBayesianModelTrained>;
    fn fit(self, X: &ArrayView2<Float>, y: &ArrayView2<Float>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outputs = y.ncols();
        let mut rng = match self.config.random_state {
            Some(seed) => scirs2_core::random::seeded_rng(seed),
            None => scirs2_core::random::seeded_rng(42),
        };
        let mut models = Vec::new();
        let mut log_likelihoods = Vec::new();
        for i in 0..self.config.n_models {
            let bootstrap_size = (n_samples as Float * self.config.bootstrap_ratio) as usize;
            let mut bootstrap_indices = Vec::new();
            for _ in 0..bootstrap_size {
                bootstrap_indices.push(rng.gen_range(0..n_samples));
            }
            let mut X_boot = Array2::zeros((bootstrap_size, n_features));
            let mut y_boot = Array2::zeros((bootstrap_size, n_outputs));
            for (j, &idx) in bootstrap_indices.iter().enumerate() {
                X_boot.row_mut(j).assign(&X.row(idx));
                y_boot.row_mut(j).assign(&y.row(idx));
            }
            let model_config = BayesianMultiOutputConfig {
                random_state: self.config.random_state.map(|s| s + i as u64),
                ..self.config.base_config.clone()
            };
            let base_model = BayesianMultiOutputModel::new().config(model_config);
            let trained_result = base_model.fit(&X_boot.view(), &y_boot.view())?;
            let trained_state = trained_result.state;
            log_likelihoods.push(trained_state.log_marginal_likelihood);
            models.push(trained_state);
        }
        let model_weights = self.compute_model_weights(&log_likelihoods)?;
        let ensemble_log_likelihood =
            self.compute_ensemble_log_likelihood(&log_likelihoods, &model_weights);
        Ok(EnsembleBayesianModel {
            state: EnsembleBayesianModelTrained {
                models,
                model_weights,
                ensemble_log_likelihood,
                n_features,
                n_outputs,
                config: self.config,
            },
            config: EnsembleBayesianConfig::default(),
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for EnsembleBayesianModel<EnsembleBayesianModelTrained>
{
    fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                X.ncols()
            )));
        }
        let n_samples = X.nrows();
        let n_outputs = self.state.n_outputs;
        match self.state.config.strategy {
            EnsembleStrategy::BayesianAveraging | EnsembleStrategy::EqualWeight => {
                let mut ensemble_pred = Array2::zeros((n_samples, n_outputs));
                for (model, &weight) in self
                    .state
                    .models
                    .iter()
                    .zip(self.state.model_weights.iter())
                {
                    let model_wrapper = BayesianMultiOutputModel {
                        state: model.clone(),
                        config: self.state.config.base_config.clone(),
                    };
                    let pred = model_wrapper.predict(X)?;
                    ensemble_pred = ensemble_pred + &(pred * weight);
                }
                Ok(ensemble_pred)
            }
            EnsembleStrategy::ProductOfExperts => {
                let mut log_ensemble_pred = Array2::zeros((n_samples, n_outputs));
                for model in &self.state.models {
                    let model_wrapper = BayesianMultiOutputModel {
                        state: model.clone(),
                        config: self.state.config.base_config.clone(),
                    };
                    let pred = model_wrapper.predict(X)?;
                    log_ensemble_pred = log_ensemble_pred + &pred.mapv(|x| x.abs().max(1e-10).ln());
                }
                let n_models_f = self.state.models.len() as Float;
                let ensemble_pred = log_ensemble_pred.mapv(|x: Float| (x / n_models_f).exp());
                Ok(ensemble_pred)
            }
            EnsembleStrategy::CommitteeMachine => {
                let mut all_predictions = Vec::new();
                for model in &self.state.models {
                    let model_wrapper = BayesianMultiOutputModel {
                        state: model.clone(),
                        config: self.state.config.base_config.clone(),
                    };
                    all_predictions.push(model_wrapper.predict(X)?);
                }
                let mut ensemble_pred = Array2::zeros((n_samples, n_outputs));
                for i in 0..n_samples {
                    for j in 0..n_outputs {
                        let mut values: Vec<Float> =
                            all_predictions.iter().map(|pred| pred[[i, j]]).collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = if values.len() % 2 == 0 {
                            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                        } else {
                            values[values.len() / 2]
                        };
                        ensemble_pred[[i, j]] = median;
                    }
                }
                Ok(ensemble_pred)
            }
            EnsembleStrategy::MixtureOfExperts => {
                let mut ensemble_pred = Array2::zeros((n_samples, n_outputs));
                for (model, &weight) in self
                    .state
                    .models
                    .iter()
                    .zip(self.state.model_weights.iter())
                {
                    let model_wrapper = BayesianMultiOutputModel {
                        state: model.clone(),
                        config: self.state.config.base_config.clone(),
                    };
                    let pred = model_wrapper.predict(X)?;
                    ensemble_pred = ensemble_pred + &(pred * weight);
                }
                Ok(ensemble_pred)
            }
        }
    }
}

impl Estimator for EnsembleBayesianModel<Untrained> {
    type Config = EnsembleBayesianConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for EnsembleBayesianModel<EnsembleBayesianModelTrained> {
    type Config = EnsembleBayesianConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}
