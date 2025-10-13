#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::{DummyClassifier, DummyRegressor};
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use sklears_core::types::{Features, Float, Int};

    fn create_classification_data() -> (Features, Array1<Int>) {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1, 0, 1];
        (x, y)
    }

    fn create_regression_data() -> (Features, Array1<Float>) {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        (x, y)
    }

    mod validation_core_tests {
        use super::*;

        #[test]
        fn test_dummy_validation_result_creation() {
            let fold_scores = vec![0.8, 0.9, 0.7, 0.85];
            let result =
                DummyValidationResult::new(0.8125, 0.075, fold_scores, "MostFrequent".to_string());

            assert_eq!(result.strategy, "MostFrequent");
            assert_abs_diff_eq!(result.mean_score, 0.8125, epsilon = 1e-6);
            assert_abs_diff_eq!(result.std_score, 0.075, epsilon = 1e-6);
            assert_eq!(result.cv_scores.len(), 4);
        }

        #[test]
        fn test_confidence_interval() {
            let fold_scores = vec![0.8, 0.9, 0.7, 0.85];
            let result =
                DummyValidationResult::new(0.8125, 0.075, fold_scores, "MostFrequent".to_string());

            let (lower, upper) = result.confidence_interval;
            assert!(lower < result.mean_score);
            assert!(upper > result.mean_score);
            assert!(upper - lower > 0.0);
        }

        #[test]
        fn test_validation_config_builder() {
            let config = ValidationConfig::new()
                .cv_folds(10)
                .random_state(42)
                .shuffle(true)
                .scoring_metric("accuracy".to_string())
                .bootstrap_samples(500);

            assert_eq!(config.cv_folds, 10);
            assert_eq!(config.random_state, Some(42));
            assert!(config.shuffle);
            assert_eq!(config.scoring_metric, "accuracy");
            assert_eq!(config.bootstrap_samples, 500);
        }

        #[test]
        fn test_statistical_summary_from_scores() {
            let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let summary = validation_core::StatisticalSummary::from_scores(&scores);

            assert_abs_diff_eq!(summary.mean, 3.0, epsilon = 1e-6);
            assert_abs_diff_eq!(summary.median, 3.0, epsilon = 1e-6);
            assert_abs_diff_eq!(summary.min, 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(summary.max, 5.0, epsilon = 1e-6);
        }

        #[test]
        fn test_is_classification_task() {
            let classification_y = array![0.0, 1.0, 2.0, 1.0, 0.0];
            let regression_y = array![0.5, 1.2, 2.7, 1.8, 0.3];

            assert!(is_classification_task(&classification_y));
            assert!(!is_classification_task(&regression_y));
        }
    }

    mod cross_validation_tests {
        use super::*;

        #[test]
        fn test_cross_validate_dummy_classifier() {
            let (x, y) = create_classification_data();
            let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);

            let result = cross_validate_dummy_classifier(classifier, &x, &y, 3);
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
            assert!(result.std_score >= 0.0);
            assert_eq!(result.fold_scores.len(), 3);
        }

        #[test]
        fn test_cross_validate_dummy_regressor() {
            let (x, y) = create_regression_data();
            let regressor = DummyRegressor::new(RegressorStrategy::Mean);

            let result = cross_validate_dummy_regressor(regressor, &x, &y, 3);
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.mean_score <= 0.0); // Negative MSE
            assert!(result.std_score >= 0.0);
            assert_eq!(result.fold_scores.len(), 3);
        }

        #[test]
        fn test_stratified_cross_validation() {
            let (x, y) = create_classification_data();
            let classifier = DummyClassifier::new(ClassifierStrategy::Stratified);

            let result = stratified_cross_validate_classifier(classifier, &x, &y, 2, Some(42));
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
        }

        #[test]
        fn test_comprehensive_cross_validation() {
            let (x, y) = create_classification_data();
            let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);
            let config = ValidationConfig::new().cv_folds(2);

            let result = comprehensive_cross_validate_classifier(classifier, &x, &y, &config);
            assert!(result.is_ok());

            let result = result.unwrap();
            assert_eq!(result.fold_details.len(), 2);
            assert!(result.statistical_summary.mean >= 0.0);
        }
    }

    mod bootstrap_validation_tests {
        use super::*;

        #[test]
        fn test_bootstrap_validate_classifier() {
            let (x, y) = create_classification_data();
            let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);

            let result = bootstrap_validate_classifier(classifier, &x, &y, 10, Some(42));
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.bootstrap_scores.len() <= 10); // May be fewer due to OOB filtering
            assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
        }

        #[test]
        fn test_bootstrap_validate_regressor() {
            let (x, y) = create_regression_data();
            let regressor = DummyRegressor::new(RegressorStrategy::Mean);

            let result = bootstrap_validate_regressor(regressor, &x, &y, 10, Some(42));
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.bootstrap_scores.len() <= 10);
            assert!(result.mean_score <= 0.0); // Negative MSE
        }

        #[test]
        fn test_bootstrap_hypothesis_test() {
            let (x, y) = create_classification_data();
            let strategy1 = DummyClassifier::new(ClassifierStrategy::MostFrequent);
            let strategy2 = DummyClassifier::new(ClassifierStrategy::Uniform);

            let result = bootstrap_hypothesis_test(strategy1, strategy2, &x, &y, 5, Some(42));
            assert!(result.is_ok());

            let test_result = result.unwrap();
            assert!(test_result.p_value >= 0.0 && test_result.p_value <= 1.0);
            assert!(!test_result.differences.is_empty());
        }
    }

    mod validation_metrics_tests {
        use super::*;

        #[test]
        fn test_accuracy_score() {
            let predictions = array![0, 1, 0, 1];
            let y_true = array![0, 1, 1, 1];
            let accuracy = accuracy_score(&predictions, &y_true);
            assert_abs_diff_eq!(accuracy, 0.75, epsilon = 1e-6);
        }

        #[test]
        fn test_precision_score() {
            let predictions = array![0, 1, 0, 1];
            let y_true = array![0, 1, 1, 1];
            let precision = precision_score(&predictions, &y_true);
            assert!(precision.is_ok());
            let precision = precision.unwrap();
            assert!(precision >= 0.0 && precision <= 1.0);
        }

        #[test]
        fn test_recall_score() {
            let predictions = array![0, 1, 0, 1];
            let y_true = array![0, 1, 1, 1];
            let recall = recall_score(&predictions, &y_true);
            assert!(recall.is_ok());
            let recall = recall.unwrap();
            assert!(recall >= 0.0 && recall <= 1.0);
        }

        #[test]
        fn test_f1_score() {
            let predictions = array![0, 1, 0, 1];
            let y_true = array![0, 1, 1, 1];
            let f1 = f1_score(&predictions, &y_true);
            assert!(f1.is_ok());
            let f1 = f1.unwrap();
            assert!(f1 >= 0.0 && f1 <= 1.0);
        }

        #[test]
        fn test_mean_squared_error() {
            let predictions = array![1.0, 2.0, 3.0];
            let y_true = array![1.5, 2.5, 2.5];
            let mse = mean_squared_error(&predictions, &y_true);
            assert!(mse >= 0.0);
        }

        #[test]
        fn test_r2_score() {
            let predictions = array![1.0, 2.0, 3.0];
            let y_true = array![1.0, 2.0, 3.0];
            let r2 = r2_score(&predictions, &y_true);
            assert!(r2.is_ok());
            let r2 = r2.unwrap();
            assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-6);
        }

        #[test]
        fn test_classification_metrics_compute() {
            let predictions = array![0, 1, 0, 1];
            let y_true = array![0, 1, 1, 1];

            let metrics = ClassificationMetrics::compute(&predictions, &y_true);
            assert!(metrics.is_ok());

            let metrics = metrics.unwrap();
            assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
            assert!(metrics.precision >= 0.0 && metrics.precision <= 1.0);
            assert!(metrics.recall >= 0.0 && metrics.recall <= 1.0);
            assert!(metrics.f1 >= 0.0 && metrics.f1 <= 1.0);
        }
    }

    mod data_splitting_tests {
        use super::*;

        #[test]
        fn test_create_cv_folds() {
            let indices = vec![0, 1, 2, 3, 4, 5];
            let folds = create_cv_folds(&indices, 3);

            assert_eq!(folds.len(), 3);
            for (train_indices, test_indices) in &folds {
                assert!(!train_indices.is_empty());
                assert!(!test_indices.is_empty());
                assert_eq!(train_indices.len() + test_indices.len(), indices.len());
            }
        }

        #[test]
        fn test_create_stratified_folds() {
            let y = array![0, 0, 1, 1, 0, 1];
            let folds = create_stratified_folds(&y, 2, Some(42));
            assert!(folds.is_ok());

            let folds = folds.unwrap();
            assert_eq!(folds.len(), 2);
        }

        #[test]
        fn test_create_shuffled_indices() {
            let indices = create_shuffled_indices(10, Some(42));
            assert_eq!(indices.len(), 10);

            // Check all indices are present
            let mut sorted_indices = indices.clone();
            sorted_indices.sort();
            assert_eq!(sorted_indices, (0..10).collect::<Vec<_>>());
        }

        #[test]
        fn test_train_test_split() {
            let split = train_test_split(100, Some(0.2), None, Some(42), true, None);
            assert!(split.is_ok());

            let split = split.unwrap();
            assert_eq!(split.train_size + split.test_size, 100);
            assert!(split.test_size >= 15 && split.test_size <= 25); // Approximately 20%
        }

        #[test]
        fn test_cv_strategy_kfold() {
            let strategy = CVStrategy::KFold {
                n_splits: 3,
                shuffle: true,
                random_state: Some(42),
            };

            let folds = strategy.split(12, None, None);
            assert!(folds.is_ok());

            let folds = folds.unwrap();
            assert_eq!(folds.len(), 3);
        }

        #[test]
        fn test_cv_strategy_leave_one_out() {
            let strategy = CVStrategy::LeaveOneOut;
            let folds = strategy.split(5, None, None);
            assert!(folds.is_ok());

            let folds = folds.unwrap();
            assert_eq!(folds.len(), 5);
            for (train_indices, test_indices) in folds {
                assert_eq!(train_indices.len(), 4);
                assert_eq!(test_indices.len(), 1);
            }
        }
    }

    mod validation_utils_tests {
        use super::*;

        #[test]
        fn test_validation_timer() {
            let mut timer = ValidationTimer::new();
            timer.start_stage("training");
            std::thread::sleep(std::time::Duration::from_millis(10));
            let duration = timer.end_stage("training");

            assert!(duration > 0.0);
            assert!(timer.get_stage_time("training").is_some());
        }

        #[test]
        fn test_memory_tracker() {
            let mut tracker = MemoryTracker::new();
            tracker.record_usage(1000);
            tracker.record_usage(1500);
            tracker.record_usage(800);

            assert_eq!(tracker.get_peak_memory(), 1500);
            assert_abs_diff_eq!(tracker.get_average_memory(), 1100.0, epsilon = 1e-6);
        }

        #[test]
        fn test_progress_tracker() {
            let mut tracker = ProgressTracker::new(10);
            tracker.update(5, "halfway");

            assert_abs_diff_eq!(tracker.progress_percentage(), 50.0, epsilon = 1e-6);
            assert!(!tracker.is_complete());

            tracker.update(10, "complete");
            assert!(tracker.is_complete());
        }

        #[test]
        fn test_data_validator_finite_check() {
            let valid_data = array![1.0, 2.0, 3.0];
            assert!(DataValidator::check_finite(&valid_data).is_ok());

            let nan_data = array![1.0, Float::NAN, 3.0];
            assert!(DataValidator::check_finite(&nan_data).is_err());

            let inf_data = array![1.0, Float::INFINITY, 3.0];
            assert!(DataValidator::check_finite(&inf_data).is_err());
        }

        #[test]
        fn test_data_validator_consistent_length() {
            let array1 = array![1.0, 2.0, 3.0];
            let array2 = array![4.0, 5.0, 6.0];
            let array3 = array![7.0, 8.0];

            assert!(DataValidator::check_consistent_length(&[&array1, &array2]).is_ok());
            assert!(DataValidator::check_consistent_length(&[&array1, &array3]).is_err());
        }

        #[test]
        fn test_data_validator_classification_targets() {
            let valid_targets = array![0, 1, 2, 1, 0];
            assert!(DataValidator::check_classification_targets(&valid_targets).is_ok());

            let negative_targets = array![0, -1, 2, 1, 0];
            assert!(DataValidator::check_classification_targets(&negative_targets).is_err());

            let single_class = array![0, 0, 0, 0];
            assert!(DataValidator::check_classification_targets(&single_class).is_err());
        }

        #[test]
        fn test_score_utils_confidence_interval() {
            let scores = vec![0.8, 0.9, 0.7, 0.85, 0.75];
            let (lower, upper) = ScoreUtils::confidence_interval(&scores, 0.95);

            let mean = scores.iter().sum::<Float>() / scores.len() as Float;
            assert!(lower < mean);
            assert!(upper > mean);
        }

        #[test]
        fn test_score_utils_cohens_d() {
            let scores1 = vec![0.8, 0.9, 0.7];
            let scores2 = vec![0.6, 0.7, 0.5];
            let effect_size = ScoreUtils::cohens_d(&scores1, &scores2);

            assert!(effect_size > 0.0); // scores1 should be higher
        }
    }

    mod strategy_comparison_tests {
        use super::*;

        #[test]
        fn test_compare_dummy_strategies() {
            let (x, y_int) = create_classification_data();
            let y = y_int.mapv(|x| x as f64); // Convert to Float for generic function
            let strategies = vec!["most_frequent".to_string(), "uniform".to_string()];

            let results = compare_dummy_strategies(&strategies, &x, &y, 2);
            assert!(results.is_ok());

            let results = results.unwrap();
            assert_eq!(results.len(), 2);
            for result in results {
                assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
            }
        }

        #[test]
        fn test_find_best_strategy() {
            let result1 =
                DummyValidationResult::new(0.7, 0.1, vec![0.6, 0.8], "Strategy1".to_string());
            let result2 =
                DummyValidationResult::new(0.9, 0.05, vec![0.85, 0.95], "Strategy2".to_string());
            let results = vec![result1, result2];

            // Convert to the expected type for find_best_strategy
            let core_results: Vec<validation_core::DummyValidationResult> = results
                .into_iter()
                .map(|r| validation_core::DummyValidationResult {
                    mean_score: r.mean_score,
                    std_score: r.std_score,
                    fold_scores: r.cv_scores,
                    strategy: r.strategy,
                })
                .collect();
            let best = find_best_strategy(&core_results[..]);
            assert!(best.is_some());
            assert_eq!(best.unwrap().strategy, "Strategy2");
        }

        #[test]
        fn test_paired_t_test() {
            let scores1 = vec![0.8, 0.9, 0.7, 0.85];
            let scores2 = vec![0.75, 0.85, 0.65, 0.8];

            let test = perform_paired_t_test(&scores1, &scores2, "Strategy1", "Strategy2");
            assert!(test.is_ok());

            let test = test.unwrap();
            assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
            assert!(!test.test_statistic.is_nan());
        }
    }

    mod statistical_analysis_tests {
        use super::*;

        #[test]
        fn test_kolmogorov_smirnov_test() {
            let sample1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
            let sample2 = array![1.5, 2.5, 3.5, 4.5, 5.5];

            let result = kolmogorov_smirnov_test(&sample1, &sample2);
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        }

        #[test]
        fn test_shapiro_wilk_test() {
            let sample = array![1.0, 2.0, 3.0, 4.0, 5.0];
            let result = shapiro_wilk_test(&sample);
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        }

        #[test]
        fn test_anderson_darling_test() {
            let sample = array![1.0, 2.0, 3.0, 4.0, 5.0];
            let result = anderson_darling_test(&sample);
            assert!(result.is_ok());

            let result = result.unwrap();
            assert!(result.statistic >= 0.0);
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        }

        #[test]
        fn test_summary_statistics_from_sample() {
            let sample = array![1.0, 2.0, 3.0, 4.0, 5.0];
            let stats = SummaryStatistics::from_sample(&sample);

            assert_abs_diff_eq!(stats.mean, 3.0, epsilon = 1e-6);
            assert_abs_diff_eq!(stats.median, 3.0, epsilon = 1e-6);
            assert_abs_diff_eq!(stats.min, 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(stats.max, 5.0, epsilon = 1e-6);
            assert_abs_diff_eq!(stats.range, 4.0, epsilon = 1e-6);
        }
    }

    mod integration_tests {
        use super::*;

        #[test]
        fn test_full_validation_pipeline() {
            let (x, y_int) = create_classification_data();
            let y_float = y_int.mapv(|x| x as f64); // Convert to Float for generic function
            let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);

            // Test cross-validation
            let cv_result = cross_validate_dummy_classifier(classifier.clone(), &x, &y_int, 3);
            assert!(cv_result.is_ok());

            // Test bootstrap validation
            let bootstrap_result =
                bootstrap_validate_classifier(classifier, &x, &y_int, 5, Some(42));
            assert!(bootstrap_result.is_ok());

            // Test strategy comparison
            let strategies = vec!["most_frequent".to_string(), "uniform".to_string()];
            let comparison_result = compare_dummy_strategies(&strategies, &x, &y_float, 2);
            assert!(comparison_result.is_ok());
        }

        #[test]
        fn test_comprehensive_validation_workflow() {
            let (x, y) = create_classification_data();
            let config = ValidationConfig::new()
                .cv_folds(2)
                .random_state(42)
                .scoring_metric("accuracy".to_string());

            let classifier = DummyClassifier::new(ClassifierStrategy::MostFrequent);
            let result = comprehensive_cross_validate_classifier(classifier, &x, &y, &config);

            assert!(result.is_ok());
            let result = result.unwrap();

            // Check that all components are present
            assert!(!result.fold_details.is_empty());
            assert!(result.statistical_summary.mean >= 0.0);
            assert_eq!(result.config.cv_folds, 2);
        }
    }
}
