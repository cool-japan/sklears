//! Fraud Detection Naive Bayes

use super::FinanceError;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// Fraud Detection Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudDetectionNB<T: Float> {
    /// Class priors (fraud vs legitimate)
    priors: HashMap<FraudLabel, T>,
    /// Feature statistics for each class
    feature_stats: HashMap<FraudLabel, FraudFeatureStats<T>>,
    /// Fraud detection parameters
    params: FraudDetectionParams<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FraudLabel {
    /// Legitimate

    Legitimate,
    /// Fraud

    Fraud,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudFeatureStats<T: Float> {
    /// Transaction statistics
    amount_stats: (T, T), // (mean, variance)

    frequency_stats: (T, T),

    time_stats: (T, T),
    /// Behavioral statistics
    merchant_category_stats: (T, T),
    location_stats: (T, T),
    device_stats: (T, T),
    /// Risk statistics
    velocity_stats: (T, T),
    pattern_deviation_stats: (T, T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudDetectionParams<T: Float> {
    /// Fraud detection threshold
    fraud_threshold: T,
    /// Velocity check window (hours)
    velocity_window: T,
    /// Maximum transaction amount
    max_transaction_amount: T,
}

impl<T: Float + Default + Display + Debug + for<'a> std::iter::Sum<&'a T> + std::iter::Sum>
    FraudDetectionNB<T>
{
    /// Create a new fraud detection classifier
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            params: FraudDetectionParams::default(),
            _phantom: PhantomData,
        }
    }

    /// Fit the fraud detection model
    pub fn fit(
        &mut self,
        transaction_data: &[TransactionData<T>],
        fraud_labels: &[FraudLabel],
    ) -> Result<(), FinanceError> {
        if transaction_data.len() != fraud_labels.len() {
            return Err(FinanceError::FraudDetection(
                "Dimension mismatch".to_string(),
            ));
        }

        // Calculate class priors
        let mut class_counts = HashMap::new();
        for &fraud_label in fraud_labels {
            *class_counts.entry(fraud_label).or_insert(0) += 1;
        }

        let total_samples = T::from(fraud_labels.len()).unwrap();
        for (&fraud_label, &count) in class_counts.iter() {
            let prior = T::from(count).unwrap() / total_samples;
            self.priors.insert(fraud_label, prior);
        }

        // Calculate feature statistics for each class
        for (&fraud_label, _) in class_counts.iter() {
            let class_data: Vec<&TransactionData<T>> = fraud_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == fraud_label)
                .map(|(i, _)| &transaction_data[i])
                .collect();

            let feature_stats = self.calculate_fraud_features(&class_data)?;
            self.feature_stats.insert(fraud_label, feature_stats);
        }

        Ok(())
    }

    /// Calculate fraud features for a given class
    fn calculate_fraud_features(
        &self,
        transaction_data: &[&TransactionData<T>],
    ) -> Result<FraudFeatureStats<T>, FinanceError> {
        let mut amounts = Vec::new();
        let mut frequencies = Vec::new();
        let mut times = Vec::new();
        let mut merchant_categories = Vec::new();
        let mut locations = Vec::new();
        let mut devices = Vec::new();
        let mut velocities = Vec::new();
        let mut pattern_deviations = Vec::new();

        for data in transaction_data {
            amounts.push(data.amount);
            frequencies.push(data.frequency);
            times.push(data.time);
            merchant_categories.push(data.merchant_category);
            locations.push(data.location);
            devices.push(data.device);
            velocities.push(data.velocity);
            pattern_deviations.push(data.pattern_deviation);
        }

        Ok(FraudFeatureStats {
            amount_stats: (
                self.calculate_mean(&amounts),
                self.calculate_variance(&amounts),
            ),
            frequency_stats: (
                self.calculate_mean(&frequencies),
                self.calculate_variance(&frequencies),
            ),
            time_stats: (self.calculate_mean(&times), self.calculate_variance(&times)),
            merchant_category_stats: (
                self.calculate_mean(&merchant_categories),
                self.calculate_variance(&merchant_categories),
            ),
            location_stats: (
                self.calculate_mean(&locations),
                self.calculate_variance(&locations),
            ),
            device_stats: (
                self.calculate_mean(&devices),
                self.calculate_variance(&devices),
            ),
            velocity_stats: (
                self.calculate_mean(&velocities),
                self.calculate_variance(&velocities),
            ),
            pattern_deviation_stats: (
                self.calculate_mean(&pattern_deviations),
                self.calculate_variance(&pattern_deviations),
            ),
        })
    }

    /// Predict fraud probability
    pub fn predict_fraud(
        &self,
        transaction_data: &TransactionData<T>,
    ) -> Result<FraudLabel, FinanceError> {
        let probabilities = self.predict_fraud_proba(transaction_data)?;

        probabilities
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(fraud_label, _)| fraud_label)
            .ok_or_else(|| FinanceError::FraudDetection("No predictions available".to_string()))
    }

    /// Predict fraud probabilities
    pub fn predict_fraud_proba(
        &self,
        transaction_data: &TransactionData<T>,
    ) -> Result<HashMap<FraudLabel, T>, FinanceError> {
        let mut class_probs = HashMap::new();

        for (&fraud_label, &prior) in self.priors.iter() {
            let feature_stats = self
                .feature_stats
                .get(&fraud_label)
                .ok_or_else(|| FinanceError::FraudDetection("Fraud label not found".to_string()))?;

            let mut likelihood = T::one();

            // Calculate likelihood for each feature
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.amount,
                    feature_stats.amount_stats.0,
                    feature_stats.amount_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.frequency,
                    feature_stats.frequency_stats.0,
                    feature_stats.frequency_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.time,
                    feature_stats.time_stats.0,
                    feature_stats.time_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.merchant_category,
                    feature_stats.merchant_category_stats.0,
                    feature_stats.merchant_category_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.location,
                    feature_stats.location_stats.0,
                    feature_stats.location_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.device,
                    feature_stats.device_stats.0,
                    feature_stats.device_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.velocity,
                    feature_stats.velocity_stats.0,
                    feature_stats.velocity_stats.1,
                );
            likelihood = likelihood
                * self.gaussian_pdf(
                    transaction_data.pattern_deviation,
                    feature_stats.pattern_deviation_stats.0,
                    feature_stats.pattern_deviation_stats.1,
                );

            let posterior = prior * likelihood;
            class_probs.insert(fraud_label, posterior);
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
pub struct TransactionData<T: Float> {
    pub amount: T,
    pub frequency: T,
    pub time: T,
    pub merchant_category: T,
    pub location: T,
    pub device: T,
    pub velocity: T,
    pub pattern_deviation: T,
}

impl<T: Float> Default for FraudDetectionParams<T> {
    fn default() -> Self {
        Self {
            fraud_threshold: T::from(0.5).unwrap(),
            velocity_window: T::from(24.0).unwrap(),
            max_transaction_amount: T::from(10000.0).unwrap(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraud_detection_nb() {
        let mut classifier = FraudDetectionNB::<f64>::new();

        // Create sample transaction data
        let transaction_data = vec![
            TransactionData {
                amount: 50.0,
                frequency: 5.0,
                time: 14.0,
                merchant_category: 1.0,
                location: 1.0,
                device: 1.0,
                velocity: 2.0,
                pattern_deviation: 0.1,
            },
            TransactionData {
                amount: 5000.0,
                frequency: 1.0,
                time: 3.0,
                merchant_category: 5.0,
                location: 10.0,
                device: 3.0,
                velocity: 10.0,
                pattern_deviation: 0.9,
            },
        ];

        let fraud_labels = vec![FraudLabel::Legitimate, FraudLabel::Fraud];

        // Fit the model
        assert!(classifier.fit(&transaction_data, &fraud_labels).is_ok());

        // Test prediction
        let test_transaction = TransactionData {
            amount: 100.0,
            frequency: 3.0,
            time: 12.0,
            merchant_category: 2.0,
            location: 2.0,
            device: 1.0,
            velocity: 3.0,
            pattern_deviation: 0.2,
        };

        let fraud_prediction = classifier.predict_fraud(&test_transaction);
        assert!(fraud_prediction.is_ok());
    }
}
