//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{BiasVarianceAnalyzer, BiasVarianceConfig, BiasVarianceDecomposition, BiasVarianceEnsembleSizeAnalysis, DiversityAnalyzer, DiversityMetrics, EnsembleCVStrategy, EnsembleConstructionConfig, EnsembleCrossValidator, InterraterReliability, ModelSelectionLossFunction, SampleBiasVariance};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    #[test]
    fn test_ensemble_cv_config_creation() {
        let config = EnsembleConstructionConfig::default();
        let cv = EnsembleCrossValidator::new(config);
        assert!(true);
    }
    #[test]
    fn test_kfold_creation() {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::KFold {
                n_splits: 3,
                shuffle: false,
            },
            ..Default::default()
        };
        let cv = EnsembleCrossValidator::new(config);
        let x = array![
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]
        ];
        let y = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let folds = cv.create_folds(&x, &y).unwrap();
        assert_eq!(folds.len(), 3);
        for (train_indices, val_indices) in &folds {
            assert!(! train_indices.is_empty());
            assert!(! val_indices.is_empty());
            assert_eq!(train_indices.len() + val_indices.len(), 6);
        }
    }
    #[test]
    fn test_stratified_kfold() {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::StratifiedKFold {
                n_splits: 2,
                shuffle: false,
            },
            ..Default::default()
        };
        let cv = EnsembleCrossValidator::new(config);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let folds = cv.create_folds(&x, &y).unwrap();
        assert_eq!(folds.len(), 2);
    }
    #[test]
    fn test_scoring_metrics() {
        let config = EnsembleConstructionConfig::default();
        let cv = EnsembleCrossValidator::new(config);
        let predictions = array![0.0, 1.0, 1.0, 0.0];
        let y_true = array![0.0, 1.0, 0.0, 0.0];
        let accuracy = cv.compute_score(&predictions, &y_true);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
    #[test]
    fn test_parameter_combinations() {
        let config = EnsembleConstructionConfig::default();
        let cv = EnsembleCrossValidator::new(config);
        let mut param_grid = HashMap::new();
        param_grid.insert("learning_rate".to_string(), vec![0.01, 0.1, 1.0]);
        param_grid.insert("n_estimators".to_string(), vec![10.0, 50.0]);
        let combinations = cv.generate_parameter_combinations(&param_grid);
        assert_eq!(combinations.len(), 6);
        for combo in &combinations {
            assert!(combo.contains_key("learning_rate"));
            assert!(combo.contains_key("n_estimators"));
        }
    }
    #[test]
    fn test_convenience_constructors() {
        let _classifier_cv = EnsembleCrossValidator::for_classification(5);
        let _regressor_cv = EnsembleCrossValidator::for_regression(3);
        let _ts_cv = EnsembleCrossValidator::for_time_series(4, Some(100));
        let _mo_cv = EnsembleCrossValidator::multi_objective(5, 0.7, 0.3);
    }
    #[test]
    fn test_time_series_split() {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::TimeSeriesSplit {
                n_splits: 3,
                max_train_size: None,
            },
            ..Default::default()
        };
        let cv = EnsembleCrossValidator::new(config);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let folds = cv.create_folds(&x, &y).unwrap();
        let mut prev_train_size = 0;
        for (train_indices, _) in &folds {
            assert!(train_indices.len() >= prev_train_size);
            prev_train_size = train_indices.len();
        }
    }
    #[test]
    fn test_bias_variance_config() {
        let config = BiasVarianceConfig::default();
        assert_eq!(config.n_bootstrap_samples, 100);
        assert_eq!(config.bootstrap_size, 1.0);
        assert!(! config.compute_sample_level);
        let analyzer = BiasVarianceAnalyzer::new(config);
        assert!(true);
    }
    #[test]
    fn test_bias_variance_convenience_constructors() {
        let _regression_analyzer = BiasVarianceAnalyzer::for_regression(50);
        let _classification_analyzer = BiasVarianceAnalyzer::for_classification(30);
        let _sample_analyzer = BiasVarianceAnalyzer::with_sample_analysis(20);
    }
    #[test]
    fn test_loss_functions() {
        let config = BiasVarianceConfig {
            loss_function: ModelSelectionLossFunction::SquaredLoss,
            ..Default::default()
        };
        let analyzer = BiasVarianceAnalyzer::new(config);
        let squared_loss = analyzer.compute_loss(2.0, 1.0);
        assert_eq!(squared_loss, 1.0);
        let config_01 = BiasVarianceConfig {
            loss_function: ModelSelectionLossFunction::ZeroOneLoss,
            ..Default::default()
        };
        let analyzer_01 = BiasVarianceAnalyzer::new(config_01);
        let correct_pred = analyzer_01.compute_loss(1.0, 1.0);
        assert_eq!(correct_pred, 0.0);
        let wrong_pred = analyzer_01.compute_loss(1.0, 0.0);
        assert_eq!(wrong_pred, 1.0);
    }
    #[test]
    fn test_bootstrap_sample_generation() {
        let config = BiasVarianceConfig::default();
        let analyzer = BiasVarianceAnalyzer::new(config);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let (x_bootstrap, y_bootstrap) = analyzer
            .generate_bootstrap_sample(&x, &y, 0)
            .unwrap();
        assert_eq!(x_bootstrap.ncols(), x.ncols());
        assert_eq!(y_bootstrap.len(), x_bootstrap.nrows());
    }
    #[test]
    fn test_bias_variance_decomposition_structure() {
        let decomp = BiasVarianceDecomposition {
            bias_squared: 0.1,
            variance: 0.2,
            noise: 0.05,
            total_loss: 0.35,
            n_bootstrap_samples: 100,
            sample_decompositions: None,
        };
        assert_eq!(decomp.bias_squared, 0.1);
        assert_eq!(decomp.variance, 0.2);
        assert_eq!(decomp.noise, 0.05);
        assert_eq!(decomp.total_loss, 0.35);
        assert_eq!(decomp.n_bootstrap_samples, 100);
        assert!(decomp.sample_decompositions.is_none());
    }
    #[test]
    fn test_sample_bias_variance_structure() {
        let sample_bv = SampleBiasVariance {
            sample_idx: 0,
            true_value: 1.0,
            mean_prediction: 0.9,
            bias_squared: 0.01,
            variance: 0.05,
            bootstrap_predictions: vec![0.8, 0.9, 1.0, 1.1],
        };
        assert_eq!(sample_bv.sample_idx, 0);
        assert_eq!(sample_bv.true_value, 1.0);
        assert_eq!(sample_bv.mean_prediction, 0.9);
        assert_eq!(sample_bv.bias_squared, 0.01);
        assert_eq!(sample_bv.variance, 0.05);
        assert_eq!(sample_bv.bootstrap_predictions.len(), 4);
    }
    #[test]
    fn test_ensemble_size_analysis_structure() {
        let analysis = BiasVarianceEnsembleSizeAnalysis {
            ensemble_sizes: vec![1, 2, 3, 4, 5],
            bias_curve: vec![0.5, 0.45, 0.42, 0.41, 0.4],
            variance_curve: vec![0.3, 0.2, 0.15, 0.12, 0.1],
            total_error_curve: vec![0.8, 0.65, 0.57, 0.53, 0.5],
            optimal_ensemble_size: 5,
            bias_reduction: 0.1,
            variance_reduction: 0.2,
        };
        assert_eq!(analysis.ensemble_sizes.len(), 5);
        assert_eq!(analysis.optimal_ensemble_size, 5);
        assert_eq!(analysis.bias_reduction, 0.1);
        assert_eq!(analysis.variance_reduction, 0.2);
        assert_eq!(analysis.bias_curve.len(), analysis.ensemble_sizes.len());
        assert_eq!(analysis.variance_curve.len(), analysis.ensemble_sizes.len());
        assert_eq!(analysis.total_error_curve.len(), analysis.ensemble_sizes.len());
    }
    #[test]
    fn test_diversity_analyzer_creation() {
        let _analyzer = DiversityAnalyzer;
    }
    #[test]
    fn test_cohens_kappa_calculation() {
        let pred1 = vec![0, 1, 0, 1, 1, 0];
        let pred2 = vec![0, 1, 1, 1, 0, 0];
        let kappa = DiversityAnalyzer::compute_pairwise_kappa(&pred1, &pred2).unwrap();
        assert!(kappa >= - 1.0 && kappa <= 1.0);
    }
    #[test]
    fn test_fleiss_kappa_calculation() {
        let predictions = vec![vec![0, 1, 0, 1], vec![0, 1, 1, 1], vec![1, 1, 0, 0]];
        let fleiss_kappa = DiversityAnalyzer::compute_fleiss_kappa(&predictions)
            .unwrap();
        assert!(fleiss_kappa >= - 1.0 && fleiss_kappa <= 1.0);
    }
    #[test]
    fn test_disagreement_calculation() {
        let predictions = vec![vec![0, 1, 0, 1], vec![0, 1, 1, 0], vec![1, 0, 0, 1]];
        let disagreement = DiversityAnalyzer::compute_disagreement(&predictions);
        assert!(disagreement >= 0.0 && disagreement <= 1.0);
    }
    #[test]
    fn test_pearson_correlation() {
        let pred1 = vec![1, 2, 3, 4, 5];
        let pred2 = vec![2, 4, 6, 8, 10];
        let correlation = DiversityAnalyzer::compute_pearson_correlation(&pred1, &pred2)
            .unwrap();
        assert!((correlation - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_diversity_metrics_structure() {
        let metrics = DiversityMetrics {
            disagreement: 0.3,
            double_fault: 0.1,
            q_statistic: 0.2,
            entropy_diversity: 0.4,
            kw_variance: 0.25,
            kappa: 0.6,
            fleiss_kappa: 0.65,
            interrater_reliability: InterraterReliability {
                overall_agreement: 0.7,
                chance_agreement: 0.4,
                krippendorff_alpha: 0.55,
                pearson_correlation: 0.8,
                weighted_kappa: 0.6,
                kappa_std_error: 0.05,
            },
        };
        assert_eq!(metrics.disagreement, 0.3);
        assert_eq!(metrics.kappa, 0.6);
        assert_eq!(metrics.fleiss_kappa, 0.65);
        assert_eq!(metrics.interrater_reliability.overall_agreement, 0.7);
    }
    #[test]
    fn test_comprehensive_diversity_metrics() {
        let predictions = vec![
            array![0.0, 1.0, 0.0, 1.0], array![0.0, 1.0, 1.0, 1.0], array![1.0, 0.0, 0.0,
            1.0],
        ];
        let ground_truth = array![0.0, 1.0, 0.0, 1.0];
        let diversity_metrics = DiversityAnalyzer::compute_diversity_metrics(
                &predictions,
                &ground_truth,
            )
            .unwrap();
        assert!(
            diversity_metrics.disagreement >= 0.0 && diversity_metrics.disagreement <=
            1.0
        );
        assert!(
            diversity_metrics.double_fault >= 0.0 && diversity_metrics.double_fault <=
            1.0
        );
        assert!(diversity_metrics.kappa >= - 1.0 && diversity_metrics.kappa <= 1.0);
        assert!(
            diversity_metrics.fleiss_kappa >= - 1.0 && diversity_metrics.fleiss_kappa <=
            1.0
        );
        assert!(
            diversity_metrics.interrater_reliability.overall_agreement >= 0.0 &&
            diversity_metrics.interrater_reliability.overall_agreement <= 1.0
        );
    }
    #[test]
    fn test_perfect_agreement_kappa() {
        let predictions = vec![vec![0, 1, 0, 1], vec![0, 1, 0, 1], vec![0, 1, 0, 1]];
        let cohens_kappa = DiversityAnalyzer::compute_cohens_kappa(&predictions)
            .unwrap();
        let fleiss_kappa = DiversityAnalyzer::compute_fleiss_kappa(&predictions)
            .unwrap();
        assert!((cohens_kappa - 1.0).abs() < 0.001);
        assert!((fleiss_kappa - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_zero_agreement_kappa() {
        let predictions = vec![vec![0, 0, 0, 0], vec![1, 1, 1, 1]];
        let cohens_kappa = DiversityAnalyzer::compute_cohens_kappa(&predictions)
            .unwrap();
        assert!(cohens_kappa <= 0.0);
    }
}
