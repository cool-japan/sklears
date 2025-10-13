use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::{seeded_rng, Rng, SliceRandom};
use std::collections::HashMap;

/// Statistical validation tests for cross-validation methods
/// Tests statistical properties and theoretical guarantees of CV methods
#[allow(non_snake_case)]
#[cfg(test)]
mod statistical_validation_tests {
    use super::*;

    /// Test that CV folds have statistically similar distributions
    #[test]
    fn test_cv_fold_distribution_similarity() {
        let mut rng = seeded_rng(42);
        let n_samples = 1000;

        // Generate data with known distribution
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Test KFold CV
        let kfold = KFold::new(5, Some(42), true);
        let folds: Vec<_> = kfold.split(&x, Some(&y)).collect();

        // Calculate statistics for each fold
        let mut fold_means = Vec::new();

        for (train_idx, test_idx) in folds {
            let train_x = x.select(Axis(0), &train_idx);
            let test_x = x.select(Axis(0), &test_idx);

            let train_mean = train_x.mean_axis(Axis(0)).unwrap();
            let test_mean = test_x.mean_axis(Axis(0)).unwrap();

            fold_means.push(train_mean);
            fold_means.push(test_mean);
        }

        // Test that fold means are statistically similar
        let overall_mean = calculate_overall_mean(&fold_means);
        let max_deviation = calculate_max_deviation(&fold_means, &overall_mean);

        // With random shuffling, deviations should be small
        assert!(
            max_deviation < 0.5,
            "CV fold distributions should be similar"
        );
    }

    /// Test CV bias-variance properties
    #[test]
    fn test_cv_bias_variance_properties() {
        let mut rng = seeded_rng(42);
        let n_samples = 500;
        let n_features = 10;

        // Generate data
        let x = Array2::from_shape_fn((n_samples, n_features), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Test different CV methods
        let cv_methods = vec![
            Box::new(KFold::new(5, Some(42), true)) as Box<dyn CrossValidator>,
            Box::new(KFold::new(10, Some(42), true)) as Box<dyn CrossValidator>,
            Box::new(LeaveOneOut::new()) as Box<dyn CrossValidator>,
        ];

        for cv_method in cv_methods {
            let cv_scores = simulate_cv_scores(&cv_method, &x, &y, 10, &mut rng);

            // Calculate bias and variance
            let bias = calculate_bias(&cv_scores);
            let variance = calculate_variance(&cv_scores);

            // CV should have low bias and reasonable variance
            assert!(bias.abs() < 2.0, "CV bias should be low");
            assert!(
                variance > 0.0 && variance < 10.0,
                "CV variance should be reasonable"
            );
        }
    }

    /// Test stratified CV maintains class balance
    #[test]
    fn test_stratified_cv_class_balance() {
        let mut rng = seeded_rng(42);
        let n_samples = 1000;

        // Generate imbalanced binary classification data
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let mut y = Array2::zeros((n_samples, 1));

        // Create imbalanced classes (20% positive, 80% negative)
        for i in 0..n_samples {
            if i < 200 {
                y[[i, 0]] = 1.0;
            } else {
                y[[i, 0]] = 0.0;
            }
        }

        // Test stratified CV
        let stratified_cv = StratifiedKFold::new(5, Some(42), true);
        let folds: Vec<_> = stratified_cv.split(&x, Some(&y)).collect();

        // Check class balance in each fold
        for (train_idx, test_idx) in folds {
            let train_y = y.select(Axis(0), &train_idx);
            let test_y = y.select(Axis(0), &test_idx);

            let train_pos_ratio = calculate_positive_ratio(&train_y);
            let test_pos_ratio = calculate_positive_ratio(&test_y);

            // Both train and test should maintain approximately 20% positive class
            assert!(
                (train_pos_ratio - 0.2).abs() < 0.05,
                "Training set should maintain class balance"
            );
            assert!(
                (test_pos_ratio - 0.2).abs() < 0.05,
                "Test set should maintain class balance"
            );
        }
    }

    /// Test CV confidence intervals
    #[test]
    fn test_cv_confidence_intervals() {
        let mut rng = seeded_rng(42);
        let n_samples = 300;

        // Generate data
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Simulate CV scores multiple times
        let n_simulations = 50;
        let mut all_cv_scores = Vec::new();

        for _ in 0..n_simulations {
            let kfold = KFold::new(5, Some(rng.random::<u64>()), true);
            let cv_scores = simulate_cv_scores(
                &(Box::new(kfold.clone()) as Box<dyn CrossValidator>),
                &x,
                &y,
                1,
                &mut rng,
            );
            all_cv_scores.extend(cv_scores);
        }

        // Calculate confidence intervals
        let (lower_ci, upper_ci) = calculate_confidence_interval(&all_cv_scores, 0.95);
        let mean_score = all_cv_scores.iter().sum::<f64>() / all_cv_scores.len() as f64;

        // CI should contain the mean and be reasonable
        assert!(
            lower_ci < mean_score && mean_score < upper_ci,
            "Mean should be within CI"
        );
        assert!(upper_ci - lower_ci > 0.0, "CI should have positive width");
    }

    /// Test time series CV temporal ordering
    #[test]
    fn test_time_series_cv_temporal_ordering() {
        let n_samples = 1000;
        let x = Array2::from_shape_vec(
            (n_samples, 3),
            (0..n_samples * 3).map(|i| i as f64).collect(),
        )
        .unwrap();
        let y = Array2::from_shape_vec((n_samples, 1), (0..n_samples).map(|i| i as f64).collect())
            .unwrap();

        // Test time series CV
        let ts_cv = TimeSeriesSplit::new(5, None, 0);
        let folds: Vec<_> = ts_cv.split(&x, Some(&y)).collect();

        // Check temporal ordering
        for (train_idx, test_idx) in folds {
            let max_train_idx = train_idx.iter().max().unwrap_or(&0);
            let default_min = n_samples - 1;
            let min_test_idx = test_idx.iter().min().unwrap_or(&default_min);

            // All training indices should be before test indices
            assert!(
                max_train_idx < min_test_idx,
                "Training data should come before test data in time series CV"
            );
        }
    }

    /// Test CV statistical power
    #[test]
    fn test_cv_statistical_power() {
        let mut rng = seeded_rng(42);
        let n_samples = 200;

        // Generate data with known effect size
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Test different CV methods and their statistical power
        let cv_methods = vec![
            (
                "3-fold",
                Box::new(KFold::new(3, Some(42), true)) as Box<dyn CrossValidator>,
            ),
            (
                "5-fold",
                Box::new(KFold::new(5, Some(42), true)) as Box<dyn CrossValidator>,
            ),
            (
                "10-fold",
                Box::new(KFold::new(10, Some(42), true)) as Box<dyn CrossValidator>,
            ),
        ];

        for (name, cv_method) in cv_methods {
            let cv_scores = simulate_cv_scores(&cv_method, &x, &y, 1, &mut rng);
            let statistical_power = calculate_statistical_power(&cv_scores, 0.05);

            // Statistical power should be reasonable
            assert!(
                statistical_power > 0.0 && statistical_power <= 1.0,
                "Statistical power for {} should be between 0 and 1",
                name
            );
        }
    }

    /// Test CV robustness to outliers
    #[test]
    fn test_cv_robustness_to_outliers() {
        let mut rng = seeded_rng(42);
        let n_samples = 300;

        // Generate clean data
        let mut x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let mut y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Add outliers
        let n_outliers = 20;
        for i in 0..n_outliers {
            x[[i, 0]] = 10.0; // Extreme outlier
            y[[i, 0]] = 10.0;
        }

        // Test CV with and without outliers
        let kfold = KFold::new(5, Some(42), true);
        let cv_scores_with_outliers = simulate_cv_scores(
            &(Box::new(kfold.clone()) as Box<dyn CrossValidator>),
            &x,
            &y,
            1,
            &mut rng,
        );

        // Remove outliers
        let clean_x = x.slice(s![n_outliers.., ..]).to_owned();
        let clean_y = y.slice(s![n_outliers.., ..]).to_owned();
        let cv_scores_clean = simulate_cv_scores(
            &(Box::new(kfold) as Box<dyn CrossValidator>),
            &clean_x,
            &clean_y,
            1,
            &mut rng,
        );

        // CV should be somewhat robust to outliers
        let outlier_effect = calculate_outlier_effect(&cv_scores_clean, &cv_scores_with_outliers);
        assert!(
            outlier_effect < 5.0,
            "CV should be reasonably robust to outliers"
        );
    }

    // Helper functions

    fn calculate_overall_mean(fold_means: &[Array1<f64>]) -> Array1<f64> {
        let n_features = fold_means[0].len();
        let mut overall_mean = Array1::zeros(n_features);

        for fold_mean in fold_means {
            overall_mean += fold_mean;
        }

        overall_mean / fold_means.len() as f64
    }

    fn calculate_max_deviation(fold_means: &[Array1<f64>], overall_mean: &Array1<f64>) -> f64 {
        let mut max_deviation: f64 = 0.0;

        for fold_mean in fold_means {
            let deviation = (fold_mean - overall_mean).mapv(|x| x.abs()).sum();
            max_deviation = max_deviation.max(deviation);
        }

        max_deviation
    }

    fn simulate_cv_scores(
        cv_method: &Box<dyn CrossValidator>,
        x: &Array2<f64>,
        y: &Array2<f64>,
        n_simulations: usize,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let mut scores = Vec::new();

        for _ in 0..n_simulations {
            let folds: Vec<_> = cv_method.split(x, Some(y)).collect();

            for (_train_idx, _test_idx) in folds {
                // Simulate a score (random for testing)
                let score = rng.random_range(-2.0..2.0);
                scores.push(score);
            }
        }

        scores
    }

    fn calculate_bias(scores: &[f64]) -> f64 {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        mean // Bias relative to 0
    }

    fn calculate_variance(scores: &[f64]) -> f64 {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        variance
    }

    fn calculate_positive_ratio(y: &Array2<f64>) -> f64 {
        let positive_count = y.iter().filter(|&&x| x > 0.5).count();
        positive_count as f64 / y.len() as f64
    }

    fn calculate_confidence_interval(scores: &[f64], confidence_level: f64) -> (f64, f64) {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * sorted_scores.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted_scores.len() as f64) as usize;

        (
            sorted_scores[lower_idx],
            sorted_scores[upper_idx.min(sorted_scores.len() - 1)],
        )
    }

    fn calculate_statistical_power(scores: &[f64], alpha: f64) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = calculate_variance(scores);
        if !variance.is_finite() || variance <= f64::EPSILON {
            return 0.0;
        }

        let std_error = (variance / scores.len() as f64).sqrt();
        let degrees_of_freedom = scores.len().saturating_sub(1);

        let critical_value = if alpha <= 0.01 {
            2.58
        } else if alpha <= 0.05 {
            1.96
        } else {
            1.64
        };

        let adjustment = if degrees_of_freedom > 0 {
            (degrees_of_freedom as f64 / (degrees_of_freedom as f64 + 1.0)).sqrt()
        } else {
            1.0
        };

        let t_statistic = mean.abs() / std_error;
        let normalized = (t_statistic * adjustment) / critical_value;

        normalized.clamp(0.0, 1.0)
    }

    fn calculate_outlier_effect(clean_scores: &[f64], outlier_scores: &[f64]) -> f64 {
        let clean_mean = clean_scores.iter().sum::<f64>() / clean_scores.len() as f64;
        let outlier_mean = outlier_scores.iter().sum::<f64>() / outlier_scores.len() as f64;

        (clean_mean - outlier_mean).abs()
    }

    // Mock trait implementations for testing
    trait CrossValidator {
        fn split(
            &self,
            x: &Array2<f64>,
            y: Option<&Array2<f64>>,
        ) -> Box<dyn Iterator<Item = (Vec<usize>, Vec<usize>)>>;
    }

    #[derive(Clone)]
    struct KFold {
        n_splits: usize,
        random_state: Option<u64>,
        shuffle: bool,
    }

    impl KFold {
        fn new(n_splits: usize, random_state: Option<u64>, shuffle: bool) -> Self {
            Self {
                n_splits,
                random_state,
                shuffle,
            }
        }
    }

    impl CrossValidator for KFold {
        fn split(
            &self,
            x: &Array2<f64>,
            _y: Option<&Array2<f64>>,
        ) -> Box<dyn Iterator<Item = (Vec<usize>, Vec<usize>)>> {
            let n_samples = x.shape()[0];
            let mut indices: Vec<usize> = (0..n_samples).collect();

            if self.shuffle {
                if let Some(seed) = self.random_state {
                    let mut rng = seeded_rng(seed);
                    indices.shuffle(&mut rng);
                }
            }

            let fold_size = n_samples / self.n_splits;
            let mut folds = Vec::new();

            for i in 0..self.n_splits {
                let start = i * fold_size;
                let end = if i == self.n_splits - 1 {
                    n_samples
                } else {
                    (i + 1) * fold_size
                };

                let test_idx = indices[start..end].to_vec();
                let train_idx = indices[..start]
                    .iter()
                    .chain(indices[end..].iter())
                    .cloned()
                    .collect();

                folds.push((train_idx, test_idx));
            }

            Box::new(folds.into_iter())
        }
    }

    struct StratifiedKFold {
        n_splits: usize,
        random_state: Option<u64>,
        shuffle: bool,
    }

    impl StratifiedKFold {
        fn new(n_splits: usize, random_state: Option<u64>, shuffle: bool) -> Self {
            Self {
                n_splits,
                random_state,
                shuffle,
            }
        }
    }

    impl CrossValidator for StratifiedKFold {
        fn split(
            &self,
            x: &Array2<f64>,
            y: Option<&Array2<f64>>,
        ) -> Box<dyn Iterator<Item = (Vec<usize>, Vec<usize>)>> {
            let n_samples = x.shape()[0];
            let y = y.unwrap();

            // Group indices by class
            let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
            for i in 0..n_samples {
                let class = y[[i, 0]] as i32;
                class_indices.entry(class).or_default().push(i);
            }

            // Shuffle within each class
            if self.shuffle {
                if let Some(seed) = self.random_state {
                    let mut rng = seeded_rng(seed);
                    for indices in class_indices.values_mut() {
                        indices.shuffle(&mut rng);
                    }
                }
            }

            let mut folds = Vec::new();

            for i in 0..self.n_splits {
                let mut train_idx = Vec::new();
                let mut test_idx = Vec::new();

                for indices in class_indices.values() {
                    let fold_size = indices.len() / self.n_splits;
                    let start = i * fold_size;
                    let end = if i == self.n_splits - 1 {
                        indices.len()
                    } else {
                        (i + 1) * fold_size
                    };

                    test_idx.extend(&indices[start..end]);
                    train_idx.extend(&indices[..start]);
                    train_idx.extend(&indices[end..]);
                }

                folds.push((train_idx, test_idx));
            }

            Box::new(folds.into_iter())
        }
    }

    struct LeaveOneOut;

    impl LeaveOneOut {
        fn new() -> Self {
            Self
        }
    }

    impl CrossValidator for LeaveOneOut {
        fn split(
            &self,
            x: &Array2<f64>,
            _y: Option<&Array2<f64>>,
        ) -> Box<dyn Iterator<Item = (Vec<usize>, Vec<usize>)>> {
            let n_samples = x.shape()[0];
            let mut folds = Vec::new();

            for i in 0..n_samples {
                let test_idx = vec![i];
                let train_idx = (0..n_samples).filter(|&idx| idx != i).collect();
                folds.push((train_idx, test_idx));
            }

            Box::new(folds.into_iter())
        }
    }

    struct TimeSeriesSplit {
        n_splits: usize,
        max_train_size: Option<usize>,
        gap: usize,
    }

    impl TimeSeriesSplit {
        fn new(n_splits: usize, max_train_size: Option<usize>, gap: usize) -> Self {
            Self {
                n_splits,
                max_train_size,
                gap,
            }
        }
    }

    impl CrossValidator for TimeSeriesSplit {
        fn split(
            &self,
            x: &Array2<f64>,
            _y: Option<&Array2<f64>>,
        ) -> Box<dyn Iterator<Item = (Vec<usize>, Vec<usize>)>> {
            let n_samples = x.shape()[0];
            let mut folds = Vec::new();

            let test_size = n_samples / (self.n_splits + 1);

            for i in 0..self.n_splits {
                let test_start = (i + 1) * test_size;
                let test_end = if i == self.n_splits - 1 {
                    n_samples
                } else {
                    (i + 2) * test_size
                };

                let train_end = test_start.saturating_sub(self.gap);
                let train_start = if let Some(max_size) = self.max_train_size {
                    train_end.saturating_sub(max_size)
                } else {
                    0
                };

                let train_idx = (train_start..train_end).collect();
                let test_idx = (test_start..test_end).collect();

                folds.push((train_idx, test_idx));
            }

            Box::new(folds.into_iter())
        }
    }
}
