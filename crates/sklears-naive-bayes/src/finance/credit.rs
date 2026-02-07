//! Credit Scoring Naive Bayes

use super::FinanceError;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Credit Scoring Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditScoringNB<T: Float> {
    /// Class priors (credit risk categories)
    priors: HashMap<CreditRisk, T>,
    /// Feature statistics for each credit risk category
    feature_stats: HashMap<CreditRisk, CreditFeatureStats<T>>,
    /// Credit scoring parameters
    params: CreditScoringParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreditRisk {
    /// Low

    Low,
    /// Medium

    Medium,
    /// High

    High,
    /// Default

    Default,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditFeatureStats<T: Float> {
    /// Financial statistics
    credit_score_stats: (T, T), // (mean, variance)
    income_stats: (T, T),
    debt_to_income_stats: (T, T),
    payment_history_stats: (T, T),
    credit_utilization_stats: (T, T),
    /// Behavioral statistics
    account_age_stats: (T, T),
    num_accounts_stats: (T, T),
    recent_inquiries_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditScoringParams<T: Float> {
    /// Minimum credit score threshold
    min_credit_score: T,
    /// Maximum debt-to-income ratio
    max_debt_to_income: T,
    /// Payment history weight
    payment_history_weight: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    CreditScoringNB<T>
{
    /// Create a new credit scoring classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: CreditScoringParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the credit scoring model
    pub fn fit(
        &mut self,
        credit_data: &[CreditData<T>],
        risk_labels: &[CreditRisk],
    ) -> Result<(), FinanceError> {
        if credit_data.len() != risk_labels.len() {
            return Err(FinanceError::CreditScoring(
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
            let class_data: Vec<&CreditData<T>> = risk_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == risk_level)
                .map(|(i, _)| &credit_data[i])
                .collect();

            let feature_stats = self.calculate_credit_features(&class_data)?;
            self.feature_stats.insert(risk_level, feature_stats);
        }

        Ok(())
    }

    /// Calculate credit features for a given risk level
    fn calculate_credit_features(
        &self,
        credit_data: &[&CreditData<T>],
    ) -> Result<CreditFeatureStats<T>, FinanceError> {
        let mut credit_scores = Vec::new();
        let mut incomes = Vec::new();
        let mut debt_to_incomes = Vec::new();
        let mut payment_histories = Vec::new();
        let mut credit_utilizations = Vec::new();
        let mut account_ages = Vec::new();
        let mut num_accounts = Vec::new();
        let mut recent_inquiries = Vec::new();

        for data in credit_data {
            credit_scores.push(data.credit_score);
            incomes.push(data.income);
            debt_to_incomes.push(data.debt_to_income);
            payment_histories.push(data.payment_history);
            credit_utilizations.push(data.credit_utilization);
            account_ages.push(data.account_age);
            num_accounts.push(data.num_accounts);
            recent_inquiries.push(data.recent_inquiries);
        }

        Ok(CreditFeatureStats {
            credit_score_stats: (
                self.calculate_mean(&credit_scores),
                self.calculate_variance(&credit_scores),
            ),
            income_stats: (
                self.calculate_mean(&incomes),
                self.calculate_variance(&incomes),
            ),
            debt_to_income_stats: (
                self.calculate_mean(&debt_to_incomes),
                self.calculate_variance(&debt_to_incomes),
            ),
            payment_history_stats: (
                self.calculate_mean(&payment_histories),
                self.calculate_variance(&payment_histories),
            ),
            credit_utilization_stats: (
                self.calculate_mean(&credit_utilizations),
                self.calculate_variance(&credit_utilizations),
            ),
            account_age_stats: (
                self.calculate_mean(&account_ages),
                self.calculate_variance(&account_ages),
            ),
            num_accounts_stats: (
                self.calculate_mean(&num_accounts),
                self.calculate_variance(&num_accounts),
            ),
            recent_inquiries_stats: (
                self.calculate_mean(&recent_inquiries),
                self.calculate_variance(&recent_inquiries),
            ),
        })
    }

    /// Predict credit risk
    pub fn predict_risk(&self, credit_data: &CreditData<T>) -> Result<CreditRisk, FinanceError> {
        let probabilities = self.predict_risk_proba(credit_data)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(risk_level, _)| risk_level)
            .ok_or_else(|| FinanceError::CreditScoring("No predictions available".to_string()))
    }

    /// Predict credit risk probabilities
    pub fn predict_risk_proba(
        &self,
        credit_data: &CreditData<T>,
    ) -> Result<HashMap<CreditRisk, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        for (&risk_level, &prior) in self.priors.iter() {
            let feature_stats = self
                .feature_stats
                .get(&risk_level)
                .ok_or_else(|| FinanceError::CreditScoring("Risk level not found".to_string()))?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.credit_score,
                    feature_stats.credit_score_stats.0,
                    feature_stats.credit_score_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.income,
                    feature_stats.income_stats.0,
                    feature_stats.income_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.debt_to_income,
                    feature_stats.debt_to_income_stats.0,
                    feature_stats.debt_to_income_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.payment_history,
                    feature_stats.payment_history_stats.0,
                    feature_stats.payment_history_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.credit_utilization,
                    feature_stats.credit_utilization_stats.0,
                    feature_stats.credit_utilization_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.account_age,
                    feature_stats.account_age_stats.0,
                    feature_stats.account_age_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.num_accounts,
                    feature_stats.num_accounts_stats.0,
                    feature_stats.num_accounts_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    credit_data.recent_inquiries,
                    feature_stats.recent_inquiries_stats.0,
                    feature_stats.recent_inquiries_stats.1,
                );

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditData<T: Float> {
    pub credit_score: T,
    pub income: T,
    pub debt_to_income: T,
    pub payment_history: T,
    pub credit_utilization: T,
    pub account_age: T,
    pub num_accounts: T,
    pub recent_inquiries: T,
}

impl<T: Float> Default for CreditScoringParams<T> {
    fn default() -> Self {
        Self {
            min_credit_score: T::from(600.0).unwrap(),
            max_debt_to_income: T::from(0.4).unwrap(),
            payment_history_weight: T::from(0.35).unwrap(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_scoring_nb() {
        let mut classifier = CreditScoringNB::<f64>::new();

        // Create sample credit data
        let credit_data = vec![
            CreditData {
                credit_score: 750.0,
                income: 80000.0,
                debt_to_income: 0.2,
                payment_history: 0.95,
                credit_utilization: 0.15,
                account_age: 120.0,
                num_accounts: 5.0,
                recent_inquiries: 1.0,
            },
            CreditData {
                credit_score: 550.0,
                income: 40000.0,
                debt_to_income: 0.6,
                payment_history: 0.7,
                credit_utilization: 0.8,
                account_age: 24.0,
                num_accounts: 12.0,
                recent_inquiries: 5.0,
            },
        ];

        let risk_labels = vec![CreditRisk::Low, CreditRisk::High];

        // Fit the model
        assert!(classifier.fit(&credit_data, &risk_labels).is_ok());

        // Test prediction
        let test_credit = CreditData {
            credit_score: 650.0,
            income: 60000.0,
            debt_to_income: 0.3,
            payment_history: 0.85,
            credit_utilization: 0.25,
            account_age: 72.0,
            num_accounts: 8.0,
            recent_inquiries: 2.0,
        };

        let risk_prediction = classifier.predict_risk(&test_credit);
        assert!(risk_prediction.is_ok());
    }
}
