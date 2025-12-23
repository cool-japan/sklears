//! Adversarial validation for robustness testing and data leakage detection
//!
//! This module provides methods to detect potential issues in training/test splits
//! by training discriminators to distinguish between training and test data.

use scirs2_core::ndarray::Array2;
use scirs2_core::SliceRandomExt;
use sklears_core::error::{Result, SklearsError};

/// Configuration for adversarial validation
#[derive(Debug, Clone)]
pub struct AdversarialValidationConfig {
    /// Number of cross-validation folds for discriminator training
    pub cv_folds: usize,
    /// Test size for discriminator evaluation
    pub test_size: f64,
    /// Threshold for considering distributions significantly different
    pub significance_threshold: f64,
    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap: usize,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Whether to perform feature importance analysis
    pub analyze_features: bool,
    /// Maximum number of discriminator iterations
    pub max_iterations: usize,
}

impl Default for AdversarialValidationConfig {
    fn default() -> Self {
        Self {
            cv_folds: 5,
            test_size: 0.2,
            significance_threshold: 0.6, // AUC threshold for detecting issues
            n_bootstrap: 1000,
            random_state: None,
            analyze_features: true,
            max_iterations: 100,
        }
    }
}

/// Results from adversarial validation
#[derive(Debug, Clone)]
pub struct AdversarialValidationResult {
    /// AUC score of the discriminator (0.5 = no difference, 1.0 = perfect discrimination)
    pub discriminator_auc: f64,
    /// Confidence interval for the AUC score
    pub auc_confidence_interval: (f64, f64),
    /// P-value for statistical significance test
    pub p_value: f64,
    /// Whether the distributions are significantly different
    pub is_significantly_different: bool,
    /// Feature importance scores (if enabled)
    pub feature_importance: Option<Vec<f64>>,
    /// Suspicious feature indices (features that help discriminate)
    pub suspicious_features: Vec<usize>,
    /// Cross-validation scores for the discriminator
    pub cv_scores: Vec<f64>,
    /// Detailed statistics about the validation
    pub statistics: AdversarialStatistics,
}

/// Detailed statistics from adversarial validation
#[derive(Debug, Clone)]
pub struct AdversarialStatistics {
    /// Number of training samples used
    pub n_train_samples: usize,
    /// Number of test samples used
    pub n_test_samples: usize,
    /// Number of features analyzed
    pub n_features: usize,
    /// Mean AUC across cross-validation folds
    pub mean_cv_auc: f64,
    /// Standard deviation of AUC across folds
    pub std_cv_auc: f64,
    /// Best single fold AUC
    pub best_cv_auc: f64,
    /// Worst single fold AUC
    pub worst_cv_auc: f64,
}

/// Adversarial validator for detecting data leakage and distribution shifts
#[derive(Debug, Clone)]
pub struct AdversarialValidator {
    config: AdversarialValidationConfig,
}

impl AdversarialValidator {
    pub fn new(config: AdversarialValidationConfig) -> Self {
        Self { config }
    }

    /// Perform adversarial validation on training and test sets
    pub fn validate(
        &self,
        train_data: &Array2<f64>,
        test_data: &Array2<f64>,
    ) -> Result<AdversarialValidationResult> {
        if train_data.ncols() != test_data.ncols() {
            return Err(SklearsError::InvalidInput(
                "Training and test data must have the same number of features".to_string(),
            ));
        }

        // Create combined dataset with labels (0 = train, 1 = test)
        let (combined_data, labels) = self.prepare_adversarial_data(train_data, test_data)?;

        // Train discriminator using cross-validation
        let cv_scores = self.cross_validate_discriminator(&combined_data, &labels)?;

        // Calculate main discriminator performance
        let discriminator_auc = self.train_discriminator(&combined_data, &labels)?;

        // Bootstrap confidence intervals
        let auc_ci = self.bootstrap_confidence_interval(&combined_data, &labels)?;

        // Feature importance analysis
        let (feature_importance, suspicious_features) = if self.config.analyze_features {
            self.analyze_feature_importance(&combined_data, &labels)?
        } else {
            (None, Vec::new())
        };

        // Statistical significance test
        let p_value = self.calculate_p_value(&cv_scores);
        let is_significantly_different = discriminator_auc > self.config.significance_threshold;

        // Calculate statistics
        let statistics = AdversarialStatistics {
            n_train_samples: train_data.nrows(),
            n_test_samples: test_data.nrows(),
            n_features: train_data.ncols(),
            mean_cv_auc: cv_scores.iter().sum::<f64>() / cv_scores.len() as f64,
            std_cv_auc: self.calculate_std(&cv_scores),
            best_cv_auc: cv_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            worst_cv_auc: cv_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        };

        Ok(AdversarialValidationResult {
            discriminator_auc,
            auc_confidence_interval: auc_ci,
            p_value,
            is_significantly_different,
            feature_importance,
            suspicious_features,
            cv_scores,
            statistics,
        })
    }

    /// Prepare adversarial dataset by combining train and test data
    fn prepare_adversarial_data(
        &self,
        train_data: &Array2<f64>,
        test_data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Vec<usize>)> {
        let n_train = train_data.nrows();
        let n_test = test_data.nrows();
        let n_features = train_data.ncols();

        // Combine data
        let mut combined_data = Array2::zeros((n_train + n_test, n_features));

        // Copy training data
        for i in 0..n_train {
            for j in 0..n_features {
                combined_data[[i, j]] = train_data[[i, j]];
            }
        }

        // Copy test data
        for i in 0..n_test {
            for j in 0..n_features {
                combined_data[[n_train + i, j]] = test_data[[i, j]];
            }
        }

        // Create labels (0 = train, 1 = test)
        let mut labels = Vec::with_capacity(n_train + n_test);
        labels.extend(vec![0; n_train]);
        labels.extend(vec![1; n_test]);

        Ok((combined_data, labels))
    }

    /// Cross-validate discriminator performance
    fn cross_validate_discriminator(
        &self,
        data: &Array2<f64>,
        labels: &[usize],
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let fold_size = n_samples / self.config.cv_folds;
        let mut cv_scores = Vec::new();

        for fold in 0..self.config.cv_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == self.config.cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits for this fold
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= test_start && i < test_end {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            // Extract fold data
            let train_fold_data = self.extract_rows(data, &train_indices);
            let test_fold_data = self.extract_rows(data, &test_indices);
            let train_fold_labels: Vec<usize> = train_indices.iter().map(|&i| labels[i]).collect();
            let test_fold_labels: Vec<usize> = test_indices.iter().map(|&i| labels[i]).collect();

            // Train discriminator on this fold
            let fold_auc = self.train_simple_discriminator(
                &train_fold_data,
                &train_fold_labels,
                &test_fold_data,
                &test_fold_labels,
            )?;
            cv_scores.push(fold_auc);
        }

        Ok(cv_scores)
    }

    /// Train a discriminator and return AUC score
    fn train_discriminator(&self, data: &Array2<f64>, labels: &[usize]) -> Result<f64> {
        // Split data for training discriminator
        let n_samples = data.nrows();
        let test_size = (n_samples as f64 * self.config.test_size) as usize;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        self.shuffle_indices(&mut indices);

        let train_indices = &indices[test_size..];
        let test_indices = &indices[..test_size];

        let train_data = self.extract_rows(data, train_indices);
        let test_data = self.extract_rows(data, test_indices);
        let train_labels: Vec<usize> = train_indices.iter().map(|&i| labels[i]).collect();
        let test_labels: Vec<usize> = test_indices.iter().map(|&i| labels[i]).collect();

        self.train_simple_discriminator(&train_data, &train_labels, &test_data, &test_labels)
    }

    /// Train a simple logistic regression discriminator
    fn train_simple_discriminator(
        &self,
        train_data: &Array2<f64>,
        train_labels: &[usize],
        test_data: &Array2<f64>,
        test_labels: &[usize],
    ) -> Result<f64> {
        let n_features = train_data.ncols();
        let mut weights = vec![0.0; n_features + 1]; // +1 for bias
        let learning_rate = 0.01;

        // Convert labels to -1/1 for logistic regression
        let train_y: Vec<f64> = train_labels
            .iter()
            .map(|&label| if label == 1 { 1.0 } else { -1.0 })
            .collect();

        // Gradient descent training
        for _iteration in 0..self.config.max_iterations {
            let mut gradients = vec![0.0; n_features + 1];

            for (i, &y) in train_y.iter().enumerate() {
                // Compute prediction
                let mut prediction = weights[0]; // bias
                for j in 0..n_features {
                    prediction += weights[j + 1] * train_data[[i, j]];
                }

                // Sigmoid activation
                let prob = 1.0 / (1.0 + (-prediction).exp());
                let error = prob - (y + 1.0) / 2.0; // Convert back to 0/1

                // Update gradients
                gradients[0] += error; // bias gradient
                for j in 0..n_features {
                    gradients[j + 1] += error * train_data[[i, j]];
                }
            }

            // Update weights
            for j in 0..weights.len() {
                weights[j] -= learning_rate * gradients[j] / train_y.len() as f64;
            }
        }

        // Evaluate on test set and calculate AUC
        self.calculate_auc(&weights, test_data, test_labels)
    }

    /// Calculate AUC score
    fn calculate_auc(
        &self,
        weights: &[f64],
        test_data: &Array2<f64>,
        test_labels: &[usize],
    ) -> Result<f64> {
        let n_features = test_data.ncols();
        let mut predictions = Vec::new();

        for i in 0..test_data.nrows() {
            let mut prediction = weights[0]; // bias
            for j in 0..n_features {
                prediction += weights[j + 1] * test_data[[i, j]];
            }
            let prob = 1.0 / (1.0 + (-prediction).exp());
            predictions.push(prob);
        }

        // Calculate AUC using trapezoidal rule
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        for (i, &score) in predictions.iter().enumerate() {
            if test_labels[i] == 1 {
                positive_scores.push(score);
            } else {
                negative_scores.push(score);
            }
        }

        if positive_scores.is_empty() || negative_scores.is_empty() {
            return Ok(0.5); // No discrimination possible
        }

        // Count concordant pairs
        let mut concordant = 0;
        let mut total = 0;

        for &pos_score in &positive_scores {
            for &neg_score in &negative_scores {
                total += 1;
                if pos_score > neg_score {
                    concordant += 1;
                }
            }
        }

        Ok(concordant as f64 / total as f64)
    }

    /// Bootstrap confidence intervals for AUC
    fn bootstrap_confidence_interval(
        &self,
        data: &Array2<f64>,
        labels: &[usize],
    ) -> Result<(f64, f64)> {
        let mut bootstrap_aucs = Vec::new();
        let n_samples = data.nrows();

        for _ in 0..self.config.n_bootstrap {
            // Bootstrap sample
            let mut boot_indices = Vec::new();
            for _ in 0..n_samples {
                boot_indices.push(self.random_index(n_samples));
            }

            let boot_data = self.extract_rows(data, &boot_indices);
            let boot_labels: Vec<usize> = boot_indices.iter().map(|&i| labels[i]).collect();

            // Train discriminator on bootstrap sample
            if let Ok(auc) = self.train_discriminator(&boot_data, &boot_labels) {
                bootstrap_aucs.push(auc);
            }
        }

        if bootstrap_aucs.is_empty() {
            return Ok((0.5, 0.5));
        }

        bootstrap_aucs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = (self.config.n_bootstrap as f64 * 0.025) as usize;
        let upper_idx = (self.config.n_bootstrap as f64 * 0.975) as usize;

        let lower_bound = bootstrap_aucs[lower_idx.min(bootstrap_aucs.len() - 1)];
        let upper_bound = bootstrap_aucs[upper_idx.min(bootstrap_aucs.len() - 1)];

        Ok((lower_bound, upper_bound))
    }

    /// Analyze feature importance for discrimination
    fn analyze_feature_importance(
        &self,
        data: &Array2<f64>,
        labels: &[usize],
    ) -> Result<(Option<Vec<f64>>, Vec<usize>)> {
        let n_features = data.ncols();
        let mut feature_importance = vec![0.0; n_features];

        // Calculate baseline AUC
        let baseline_auc = self.train_discriminator(data, labels)?;

        // Permute each feature and measure AUC drop
        for feature_idx in 0..n_features {
            let mut permuted_data = data.clone();

            // Permute this feature
            let mut feature_values: Vec<f64> =
                (0..data.nrows()).map(|i| data[[i, feature_idx]]).collect();
            self.shuffle_f64(&mut feature_values);

            for (i, &value) in feature_values.iter().enumerate() {
                permuted_data[[i, feature_idx]] = value;
            }

            // Calculate AUC with permuted feature
            let permuted_auc = self.train_discriminator(&permuted_data, labels)?;

            // Feature importance is the drop in AUC
            feature_importance[feature_idx] = baseline_auc - permuted_auc;
        }

        // Identify suspicious features (those that help discrimination)
        let mut suspicious_features = Vec::new();
        let importance_threshold = 0.01; // 1% AUC drop threshold

        for (i, &importance) in feature_importance.iter().enumerate() {
            if importance > importance_threshold {
                suspicious_features.push(i);
            }
        }

        Ok((Some(feature_importance), suspicious_features))
    }

    /// Calculate p-value for statistical significance
    fn calculate_p_value(&self, cv_scores: &[f64]) -> f64 {
        // One-sample t-test against null hypothesis (AUC = 0.5)
        let mean_auc = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
        let std_auc = self.calculate_std(cv_scores);
        let n = cv_scores.len() as f64;

        if std_auc == 0.0 {
            return if mean_auc > 0.5 { 0.0 } else { 1.0 };
        }

        let t_stat = (mean_auc - 0.5) * n.sqrt() / std_auc;

        // Approximate p-value using normal distribution (for large samples)
        let p_value = 2.0 * (1.0 - self.normal_cdf(t_stat.abs()));
        p_value.clamp(0.0, 1.0)
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Approximate normal CDF
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Approximate error function
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Extract rows from array by indices
    fn extract_rows(&self, data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_rows = indices.len();
        let n_cols = data.ncols();
        let mut result = Array2::zeros((n_rows, n_cols));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..n_cols {
                result[[i, j]] = data[[idx, j]];
            }
        }

        result
    }

    /// Shuffle indices randomly
    fn shuffle_indices(&self, indices: &mut [usize]) {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };
        indices.shuffle(&mut rng);
    }

    /// Shuffle f64 values randomly
    fn shuffle_f64(&self, values: &mut [f64]) {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };
        values.shuffle(&mut rng);
    }

    /// Generate random index
    fn random_index(&self, max: usize) -> usize {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::Rng;
        use scirs2_core::random::SeedableRng;
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };
        rng.gen_range(0..max)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_validation_same_distribution() {
        let config = AdversarialValidationConfig::default();
        let validator = AdversarialValidator::new(config);

        // Create identical distributions
        let train_data = Array2::zeros((100, 5));
        let test_data = Array2::zeros((50, 5));

        let result = validator.validate(&train_data, &test_data).unwrap();

        // AUC should be close to 0.5 for identical distributions
        assert!(
            result.discriminator_auc < 0.6,
            "AUC should be close to 0.5 for identical distributions"
        );
        assert!(
            !result.is_significantly_different,
            "Identical distributions should not be significantly different"
        );
    }

    #[test]
    fn test_adversarial_validation_different_distributions() {
        let config = AdversarialValidationConfig {
            significance_threshold: 0.6,
            ..Default::default()
        };
        let validator = AdversarialValidator::new(config);

        // Create different distributions
        let mut train_data = Array2::zeros((100, 5));
        let mut test_data = Array2::ones((50, 5));

        // Make them clearly different
        for i in 0..train_data.nrows() {
            for j in 0..train_data.ncols() {
                train_data[[i, j]] = 0.0;
            }
        }

        for i in 0..test_data.nrows() {
            for j in 0..test_data.ncols() {
                test_data[[i, j]] = 1.0;
            }
        }

        let result = validator.validate(&train_data, &test_data).unwrap();

        // AUC should be high for clearly different distributions
        assert!(
            result.discriminator_auc > 0.7,
            "AUC should be high for different distributions"
        );
    }
}
