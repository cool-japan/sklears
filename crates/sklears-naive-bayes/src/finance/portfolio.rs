//! Portfolio Classification Naive Bayes

use super::FinanceError;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Portfolio Classification Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioClassificationNB<T: Float> {
    /// Class priors (portfolio categories)
    priors: HashMap<PortfolioCategory, T>,
    /// Feature statistics for each portfolio category
    feature_stats: HashMap<PortfolioCategory, PortfolioFeatureStats<T>>,
    /// Portfolio classification parameters
    params: PortfolioClassificationParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortfolioCategory {
    /// Conservative

    Conservative,
    /// Moderate

    Moderate,
    /// Aggressive

    Aggressive,
    /// Growth

    Growth,
    /// Income

    Income,
    /// Balanced

    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioFeatureStats<T: Float> {
    /// Asset allocation statistics
    stock_allocation_stats: (T, T), // (mean, variance)
    bond_allocation_stats: (T, T),
    cash_allocation_stats: (T, T),
    /// Risk-return statistics
    expected_return_stats: (T, T),
    volatility_stats: (T, T),
    /// Diversification statistics
    sector_diversification_stats: (T, T),
    geographic_diversification_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioClassificationParams<T: Float> {
    /// Minimum diversification threshold
    min_diversification: T,
    /// Maximum concentration threshold
    max_concentration: T,
    /// Risk tolerance parameters
    risk_tolerance: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    PortfolioClassificationNB<T>
{
    /// Create a new portfolio classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: PortfolioClassificationParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the portfolio classification model
    pub fn fit(
        &mut self,
        portfolios: &[PortfolioData<T>],
        categories: &[PortfolioCategory],
    ) -> Result<(), FinanceError> {
        if portfolios.len() != categories.len() {
            return Err(FinanceError::PortfolioClassification(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &category in categories {
            *class_counts.entry(category).or_insert(0) += 1;
        }

        let total_samples = T::from(categories.len()).unwrap();
        for (&category, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(category, prior);
        }

        // Calculate feature statistics for each category
        for (&category, _) in class_counts.iter() {
            let class_portfolios: Vec<&PortfolioData<T>> = categories
                .iter()
                .enumerate()
                .filter(|(_, &cat)| cat == category)
                .map(|(i, _)| &portfolios[i])
                .collect();

            let feature_stats = self.calculate_portfolio_features(&class_portfolios)?;
            self.feature_stats.insert(category, feature_stats);
        }

        Ok(())
    }

    /// Calculate portfolio features for a given category
    fn calculate_portfolio_features(
        &self,
        portfolios: &[&PortfolioData<T>],
    ) -> Result<PortfolioFeatureStats<T>, FinanceError> {
        let mut stock_allocations = Vec::new();
        let mut bond_allocations = Vec::new();
        let mut cash_allocations = Vec::new();
        let mut expected_returns = Vec::new();
        let mut volatilities = Vec::new();
        let mut sector_diversifications = Vec::new();
        let mut geographic_diversifications = Vec::new();

        for portfolio in portfolios {
            stock_allocations.push(portfolio.stock_allocation);
            bond_allocations.push(portfolio.bond_allocation);
            cash_allocations.push(portfolio.cash_allocation);
            expected_returns.push(portfolio.expected_return);
            volatilities.push(portfolio.volatility);
            sector_diversifications.push(portfolio.sector_diversification);
            geographic_diversifications.push(portfolio.geographic_diversification);
        }

        Ok(PortfolioFeatureStats {
            stock_allocation_stats: (
                self.calculate_mean(&stock_allocations),
                self.calculate_variance(&stock_allocations),
            ),
            bond_allocation_stats: (
                self.calculate_mean(&bond_allocations),
                self.calculate_variance(&bond_allocations),
            ),
            cash_allocation_stats: (
                self.calculate_mean(&cash_allocations),
                self.calculate_variance(&cash_allocations),
            ),
            expected_return_stats: (
                self.calculate_mean(&expected_returns),
                self.calculate_variance(&expected_returns),
            ),
            volatility_stats: (
                self.calculate_mean(&volatilities),
                self.calculate_variance(&volatilities),
            ),
            sector_diversification_stats: (
                self.calculate_mean(&sector_diversifications),
                self.calculate_variance(&sector_diversifications),
            ),
            geographic_diversification_stats: (
                self.calculate_mean(&geographic_diversifications),
                self.calculate_variance(&geographic_diversifications),
            ),
        })
    }

    /// Predict portfolio category
    pub fn predict_category(
        &self,
        portfolio: &PortfolioData<T>,
    ) -> Result<PortfolioCategory, FinanceError> {
        let probabilities = self.predict_category_proba(portfolio)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(category, _)| category)
            .ok_or_else(|| {
                FinanceError::PortfolioClassification("No predictions available".to_string())
            })
    }

    /// Predict portfolio category probabilities
    pub fn predict_category_proba(
        &self,
        portfolio: &PortfolioData<T>,
    ) -> Result<HashMap<PortfolioCategory, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        for (&category, &prior) in self.priors.iter() {
            let feature_stats = self.feature_stats.get(&category).ok_or_else(|| {
                FinanceError::PortfolioClassification("Category not found".to_string())
            })?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.stock_allocation,
                    feature_stats.stock_allocation_stats.0,
                    feature_stats.stock_allocation_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.bond_allocation,
                    feature_stats.bond_allocation_stats.0,
                    feature_stats.bond_allocation_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.cash_allocation,
                    feature_stats.cash_allocation_stats.0,
                    feature_stats.cash_allocation_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.expected_return,
                    feature_stats.expected_return_stats.0,
                    feature_stats.expected_return_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.volatility,
                    feature_stats.volatility_stats.0,
                    feature_stats.volatility_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.sector_diversification,
                    feature_stats.sector_diversification_stats.0,
                    feature_stats.sector_diversification_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    portfolio.geographic_diversification,
                    feature_stats.geographic_diversification_stats.0,
                    feature_stats.geographic_diversification_stats.1,
                );

            let posterior = prior * likelihood;
            class_probs.insert(category, posterior);
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioData<T: Float> {
    pub stock_allocation: T,
    pub bond_allocation: T,
    pub cash_allocation: T,
    pub expected_return: T,
    pub volatility: T,
    pub sector_diversification: T,
    pub geographic_diversification: T,
}

impl<T: Float> Default for PortfolioClassificationParams<T> {
    fn default() -> Self {
        Self {
            min_diversification: T::from(0.1).unwrap(),
            max_concentration: T::from(0.3).unwrap(),
            risk_tolerance: T::from(0.15).unwrap(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_classification_nb() {
        let mut classifier = PortfolioClassificationNB::<f64>::new();

        // Create sample portfolio data
        let portfolios = vec![
            PortfolioData {
                stock_allocation: 0.8,
                bond_allocation: 0.2,
                cash_allocation: 0.0,
                expected_return: 0.08,
                volatility: 0.15,
                sector_diversification: 0.7,
                geographic_diversification: 0.6,
            },
            PortfolioData {
                stock_allocation: 0.4,
                bond_allocation: 0.5,
                cash_allocation: 0.1,
                expected_return: 0.05,
                volatility: 0.08,
                sector_diversification: 0.8,
                geographic_diversification: 0.7,
            },
        ];

        let categories = vec![
            PortfolioCategory::Aggressive,
            PortfolioCategory::Conservative,
        ];

        // Fit the model
        assert!(classifier.fit(&portfolios, &categories).is_ok());

        // Test prediction
        let test_portfolio = PortfolioData {
            stock_allocation: 0.6,
            bond_allocation: 0.3,
            cash_allocation: 0.1,
            expected_return: 0.06,
            volatility: 0.12,
            sector_diversification: 0.75,
            geographic_diversification: 0.65,
        };

        let category_prediction = classifier.predict_category(&test_portfolio);
        assert!(category_prediction.is_ok());
    }
}
