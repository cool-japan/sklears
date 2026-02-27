//! Risk Assessment Naive Bayes

use super::{DMatrix, FinanceError};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Risk Assessment Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentNB<T: Float> {
    /// Class priors (risk levels)
    priors: HashMap<RiskLevel, T>,
    /// Feature statistics for each risk level
    feature_stats: HashMap<RiskLevel, RiskFeatureStats<T>>,
    /// Risk assessment parameters
    params: RiskAssessmentParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low

    Low,
    /// Medium

    Medium,
    /// High

    High,
    /// VeryHigh

    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFeatureStats<T: Float> {
    /// Volatility statistics
    volatility_stats: (T, T), // (mean, variance)
    /// Value at Risk statistics
    var_stats: (T, T),
    /// Sharpe ratio statistics
    sharpe_stats: (T, T),
    /// Maximum drawdown statistics
    max_drawdown_stats: (T, T),
    /// Beta statistics
    beta_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentParams<T: Float> {
    /// Confidence level for VaR calculation
    var_confidence: T,
    /// Risk-free rate for Sharpe ratio
    risk_free_rate: T,
    /// Benchmark correlation threshold
    benchmark_threshold: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    RiskAssessmentNB<T>
{
    /// Create a new risk assessment classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: RiskAssessmentParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the risk assessment model
    pub fn fit(
        &mut self,
        returns: &DMatrix<T>,
        risk_labels: &[RiskLevel],
    ) -> Result<(), FinanceError> {
        if returns.nrows() != risk_labels.len() {
            return Err(FinanceError::RiskAssessment(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &risk_level in risk_labels {
            *class_counts.entry(risk_level).or_insert(0) += 1;
        }

        let total_samples = T::from(risk_labels.len()).unwrap();
        for (&risk_level, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(risk_level, prior);
        }

        // Calculate feature statistics for each risk level
        for (&risk_level, _) in class_counts.iter() {
            let class_indices: Vec<usize> = risk_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == risk_level)
                .map(|(i, _)| i)
                .collect();

            let feature_stats = self.calculate_risk_features(returns, &class_indices)?;
            self.feature_stats.insert(risk_level, feature_stats);
        }

        Ok(())
    }

    /// Calculate risk features for a given class
    fn calculate_risk_features(
        &self,
        returns: &DMatrix<T>,
        indices: &[usize],
    ) -> Result<RiskFeatureStats<T>, FinanceError> {
        let mut volatilities = Vec::new();
        let mut vars = Vec::new();
        let mut sharpe_ratios = Vec::new();
        let mut max_drawdowns = Vec::new();
        let mut betas = Vec::new();

        for &idx in indices {
            let return_series = returns.row(idx);
            let return_vec: Vec<T> = return_series.iter().cloned().collect();

            // Calculate volatility
            let volatility = self.calculate_volatility_from_returns(&return_vec);
            volatilities.push(volatility);

            // Calculate VaR
            let var = self.calculate_var(&return_vec)?;
            vars.push(var);

            // Calculate Sharpe ratio
            let sharpe = self.calculate_sharpe_ratio(&return_vec);
            sharpe_ratios.push(sharpe);

            // Calculate maximum drawdown
            let max_dd = self.calculate_max_drawdown(&return_vec);
            max_drawdowns.push(max_dd);

            // Calculate beta (simplified)
            let beta = self.calculate_beta(&return_vec);
            betas.push(beta);
        }

        Ok(RiskFeatureStats {
            volatility_stats: (
                self.calculate_mean(&volatilities),
                self.calculate_variance(&volatilities),
            ),
            var_stats: (self.calculate_mean(&vars), self.calculate_variance(&vars)),
            sharpe_stats: (
                self.calculate_mean(&sharpe_ratios),
                self.calculate_variance(&sharpe_ratios),
            ),
            max_drawdown_stats: (
                self.calculate_mean(&max_drawdowns),
                self.calculate_variance(&max_drawdowns),
            ),
            beta_stats: (self.calculate_mean(&betas), self.calculate_variance(&betas)),
        })
    }

    /// Calculate volatility from returns
    fn calculate_volatility_from_returns(&self, returns: &[T]) -> T {
        let variance = self.calculate_variance(returns);
        variance.sqrt() * T::from(252.0).unwrap().sqrt() // Annualized
    }

    /// Calculate Value at Risk
    fn calculate_var(&self, returns: &[T]) -> Result<T, FinanceError> {
        if returns.is_empty() {
            return Err(FinanceError::RiskAssessment("No returns data".to_string()));
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_index = ((T::one() - self.params.var_confidence)
            * T::from(sorted_returns.len()).unwrap())
        .floor()
        .to_usize()
        .unwrap();
        let percentile_index = percentile_index.min(sorted_returns.len() - 1);

        Ok(-sorted_returns[percentile_index]) // VaR is typically positive
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, returns: &[T]) -> T {
        let mean_return = self.calculate_mean(returns);
        let volatility = self.calculate_volatility_from_returns(returns);

        if volatility == T::zero() {
            T::zero()
        } else {
            (mean_return - self.params.risk_free_rate) / volatility
        }
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[T]) -> T {
        if returns.is_empty() {
            return T::zero();
        }

        let mut cumulative_returns = Vec::new();
        let mut cumulative = T::one();

        for &ret in returns {
            cumulative = cumulative * (T::one() + ret);
            cumulative_returns.push(cumulative);
        }

        let mut max_drawdown = T::zero();
        let mut peak = cumulative_returns[0];

        for &value in cumulative_returns.iter().skip(1) {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Calculate beta (simplified market beta)
    fn calculate_beta(&self, _returns: &[T]) -> T {
        // Simplified beta calculation (placeholder)
        T::one()
    }

    /// Predict risk level
    pub fn predict_risk(&self, returns: &[T]) -> Result<RiskLevel, FinanceError> {
        let probabilities = self.predict_risk_proba(returns)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| {
                // Handle NaN values by treating them as smaller than any finite number
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                }
            })
            .map(|(risk_level, _)| risk_level)
            .ok_or_else(|| FinanceError::RiskAssessment("No predictions available".to_string()))
    }

    /// Predict risk level probabilities
    pub fn predict_risk_proba(&self, returns: &[T]) -> Result<HashMap<RiskLevel, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        // Calculate risk features
        let volatility = self.calculate_volatility_from_returns(returns);
        let var = self.calculate_var(returns)?;
        let sharpe = self.calculate_sharpe_ratio(returns);
        let max_dd = self.calculate_max_drawdown(returns);
        let beta = self.calculate_beta(returns);

        for (&risk_level, &prior) in self.priors.iter() {
            let feature_stats = self
                .feature_stats
                .get(&risk_level)
                .ok_or_else(|| FinanceError::RiskAssessment("Risk level not found".to_string()))?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    volatility,
                    feature_stats.volatility_stats.0,
                    feature_stats.volatility_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(var, feature_stats.var_stats.0, feature_stats.var_stats.1);
            likelihood = likelihood
                * self.gaussian_pdf(
                    sharpe,
                    feature_stats.sharpe_stats.0,
                    feature_stats.sharpe_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    max_dd,
                    feature_stats.max_drawdown_stats.0,
                    feature_stats.max_drawdown_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(beta, feature_stats.beta_stats.0, feature_stats.beta_stats.1);

            let posterior = prior * likelihood;
            class_probs.insert(risk_level, posterior);
        }

        // Normalize probabilities
        let total_prob: T = class_probs.values().cloned().sum();
        if total_prob > T::zero() {
            for prob in class_probs.values_mut() {
                *prob = *prob / total_prob;
            }
        }

        Ok(class_probs)
    }

    /// Gaussian probability density function
    fn gaussian_pdf(&self, x: T, mean: T, variance: T) -> T {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let coefficient = T::one() / (two_pi * variance).sqrt();
        let exponent = -((x - mean).powi(2)) / (T::from(2.0).unwrap() * variance);
        coefficient * exponent.exp()
    }

    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum: T = values.iter().sum();
        sum / T::from(values.len()).unwrap()
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::from(0.01).unwrap(); // Small default variance
        }
        let mean = self.calculate_mean(values);
        let sum_squared_diff: T = values.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_squared_diff / T::from(values.len() - 1).unwrap()
    }
}

impl<T: Float> Default for RiskAssessmentParams<T> {
    fn default() -> Self {
        Self {
            var_confidence: T::from(0.95).unwrap(),
            risk_free_rate: T::from(0.02).unwrap(),
            benchmark_threshold: T::from(0.7).unwrap(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_risk_assessment_nb() {
        let mut classifier = RiskAssessmentNB::<f64>::new();

        // Create sample return data
        let returns =
            Array2::from_shape_vec((10, 5), (0..50).map(|i| (i as f64 - 25.0) * 0.01).collect())
                .unwrap();
        let risk_labels = vec![
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::VeryHigh,
        ];

        // Fit the model
        assert!(classifier.fit(&returns, &risk_labels).is_ok());

        // Test prediction
        let test_returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];
        let risk_prediction = classifier.predict_risk(&test_returns);
        assert!(risk_prediction.is_ok());
    }
}
