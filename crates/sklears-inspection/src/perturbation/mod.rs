//! Perturbation Strategies for Model Analysis
//!
//! This module provides various perturbation strategies for model inspection and
//! robustness analysis, including noise-based, adversarial, synthetic data,
//! distribution-preserving, and structured perturbations.
//!
//! ## Modules
//!
//! - [`core`] - Core types and configuration structures
//! - [`strategies`] - Individual perturbation strategy implementations
//! - [`pipeline`] - Flexible perturbation pipeline for chaining operations
//! - [`analysis`] - Main analysis functions and robustness testing
//! - [`helpers`] - Utility functions for statistics and metrics

pub mod analysis;
pub mod core;
pub mod helpers;
pub mod pipeline;
pub mod strategies;

// Re-export core types for backward compatibility
pub use core::{
    DependencyType, ExecutionCondition, ExecutionEdge, ExecutionGraph, ExecutionMode,
    ExecutionNode, ExecutionStatus, NoiseDistribution, PerturbationConfig, PerturbationResult,
    PerturbationStage, PerturbationStats, PerturbationStrategy, PipelineConfig, PipelineMetadata,
    PipelineResult, RobustnessMetrics, StageQualityMetrics, StageResult,
};

// Re-export main analysis functions
pub use analysis::{
    analyze_robustness, generate_counterfactuals, sensitivity_analysis, stability_analysis,
    CounterfactualResult, FeatureSensitivity, MagnitudeStabilityResult, SensitivityResult,
    StabilityResult,
};

// Re-export pipeline functionality
pub use pipeline::{PerturbationPipeline, PipelineBuilder};

// Re-export strategy function
pub use strategies::generate_perturbations;

// Re-export helper functions and types
pub use helpers::{
    calculate_confidence_intervals, calculate_consistency_metrics, calculate_diversity_metrics,
    calculate_feature_importance, calculate_perturbation_stats, calculate_robustness_metrics,
    detect_prediction_outliers, ConsistencyMetrics, DiversityMetrics, OutlierAnalysis,
};

// ✅ SciRS2 Policy Compliant Import

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;
    use sklears_core::prelude::{ArrayView2, Float};

    fn simple_model(x: &ArrayView2<Float>) -> Vec<Float> {
        x.rows()
            .into_iter()
            .map(|row| row[0] * 2.0 + row[1] * 0.5)
            .collect()
    }

    #[test]
    fn test_gaussian_perturbation() {
        let X = array![[0.5, 0.7], [0.3, 0.9]];
        let config = PerturbationConfig {
            strategy: PerturbationStrategy::Gaussian,
            magnitude: 0.1,
            n_samples: 5,
            random_state: Some(42),
            ..Default::default()
        };

        let perturbed = generate_perturbations(&X.view(), &config).unwrap();
        assert_eq!(perturbed.len(), 5);
        assert_eq!(perturbed[0].dim(), X.dim());

        // Check that perturbations are different from original
        let original_sum: Float = X.sum();
        let perturbed_sum: Float = perturbed[0].sum();
        assert!((original_sum - perturbed_sum).abs() > 1e-10);
    }

    #[test]
    fn test_uniform_perturbation() {
        let X = array![[0.5, 0.7]];
        let config = PerturbationConfig {
            strategy: PerturbationStrategy::Uniform,
            magnitude: 0.1,
            n_samples: 3,
            random_state: Some(42),
            ..Default::default()
        };

        let perturbed = generate_perturbations(&X.view(), &config).unwrap();
        assert_eq!(perturbed.len(), 3);
    }

    #[test]
    fn test_robustness_analysis() {
        let X = array![[0.5, 0.7], [0.3, 0.9]];
        let config = PerturbationConfig {
            strategy: PerturbationStrategy::Gaussian,
            magnitude: 0.05,
            n_samples: 10,
            random_state: Some(42),
            ..Default::default()
        };

        let result = analyze_robustness(&simple_model, &X.view(), &config).unwrap();

        assert_eq!(result.original_predictions.len(), 2);
        assert_eq!(result.perturbed_predictions.len(), 10);
        assert!(result.robustness_metrics.prediction_stability >= 0.0);
        assert!(result.robustness_metrics.max_prediction_change >= 0.0);
    }

    #[test]
    fn test_salt_pepper_perturbation() {
        let X = array![[0.5, 0.7], [0.3, 0.9]];
        let config = PerturbationConfig {
            strategy: PerturbationStrategy::SaltPepper,
            magnitude: 0.2, // 20% chance of noise
            n_samples: 5,
            random_state: Some(42),
            ..Default::default()
        };

        let perturbed = generate_perturbations(&X.view(), &config).unwrap();
        assert_eq!(perturbed.len(), 5);
    }

    #[test]
    fn test_dropout_perturbation() {
        let X = array![[0.5, 0.7, 0.3]];
        let config = PerturbationConfig {
            strategy: PerturbationStrategy::Dropout,
            magnitude: 0.3, // 30% dropout rate
            n_samples: 5,
            random_state: Some(42),
            ..Default::default()
        };

        let perturbed = generate_perturbations(&X.view(), &config).unwrap();
        assert_eq!(perturbed.len(), 5);

        // Check that some values are zero (dropped)
        let has_zeros = perturbed.iter().any(|p| p.iter().any(|&x| x == 0.0));
        assert!(has_zeros);
    }

    #[test]
    fn test_structured_perturbation() {
        let X = array![[0.5, 0.7, 0.3, 0.8, 0.2, 0.9]]; // 6 features
        let config = PerturbationConfig {
            strategy: PerturbationStrategy::Structured,
            magnitude: 0.1,
            n_samples: 3,
            random_state: Some(42),
            ..Default::default()
        };

        let perturbed = generate_perturbations(&X.view(), &config).unwrap();
        assert_eq!(perturbed.len(), 3);
        assert_eq!(perturbed[0].dim(), X.dim());
    }

    #[test]
    fn test_perturbation_stats() {
        let original = array![[1.0, 2.0], [3.0, 4.0]];
        let perturbed1 = array![[1.1, 2.1], [3.1, 4.1]];
        let perturbed2 = array![[0.9, 1.9], [2.9, 3.9]];
        let perturbed_data = vec![perturbed1, perturbed2];

        let stats = calculate_perturbation_stats(&original.view(), &perturbed_data).unwrap();

        assert_eq!(stats.mean_magnitude.len(), 2);
        assert_eq!(stats.std_magnitude.len(), 2);
        assert_eq!(stats.max_magnitude.len(), 2);
        assert_eq!(stats.correlation_matrix.dim(), (2, 2));
    }

    #[test]
    fn test_pipeline_sequential_execution() {
        let X = array![[0.5, 0.7], [0.3, 0.9]];

        let mut pipeline = PerturbationPipeline::builder()
            .execution_mode(ExecutionMode::Sequential)
            .add_perturbation_stage(
                "gaussian".to_string(),
                "Gaussian Noise".to_string(),
                PerturbationStrategy::Gaussian,
                0.1,
                5,
            )
            .add_perturbation_stage(
                "uniform".to_string(),
                "Uniform Noise".to_string(),
                PerturbationStrategy::Uniform,
                0.05,
                3,
            )
            .build();

        let result = pipeline.execute(&X.view()).unwrap();

        assert_eq!(result.stage_results.len(), 2);
        assert!(result.stage_results.contains_key("gaussian"));
        assert!(result.stage_results.contains_key("uniform"));
        assert_eq!(result.final_perturbed_data.len(), 8); // 5 + 3 samples
        assert!(result.metadata.success_rate > 0.0);
    }

    #[test]
    fn test_pipeline_conditional_execution() {
        let X = array![[0.5, 0.7], [0.3, 0.9]]; // 2 samples, 2 features

        let stage_with_condition = PerturbationStage {
            id: "conditional".to_string(),
            name: "Conditional Stage".to_string(),
            config: PerturbationConfig {
                strategy: PerturbationStrategy::Gaussian,
                magnitude: 0.1,
                n_samples: 5,
                ..Default::default()
            },
            condition: Some(ExecutionCondition::DataCharacteristics {
                min_samples: Some(3), // This should fail since we only have 2 samples
                max_samples: None,
                min_features: None,
                max_features: None,
                sparsity_threshold: None,
            }),
            dependencies: Vec::new(),
            enabled: true,
            priority: 0,
            max_retries: 3,
        };

        let mut pipeline = PerturbationPipeline::new(PipelineConfig {
            execution_mode: ExecutionMode::Conditional,
            ..Default::default()
        });
        pipeline.add_stage(stage_with_condition);

        let result = pipeline.execute(&X.view()).unwrap();

        // Stage should be skipped due to condition
        assert_eq!(result.stage_results.len(), 0);
        assert_eq!(result.final_perturbed_data.len(), 0);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PerturbationPipeline::builder()
            .execution_mode(ExecutionMode::Sequential)
            .max_parallel_stages(2)
            .enable_caching(true)
            .memory_limit_mb(256)
            .add_perturbation_stage(
                "test".to_string(),
                "Test Stage".to_string(),
                PerturbationStrategy::Gaussian,
                0.1,
                5,
            )
            .build();

        assert_eq!(pipeline.config.execution_mode, ExecutionMode::Sequential);
        assert_eq!(pipeline.config.max_parallel_stages, 2);
        assert!(pipeline.config.enable_caching);
        assert_eq!(pipeline.config.memory_limit_mb, 256);
        assert_eq!(pipeline.stages.len(), 1);
        assert_eq!(pipeline.stages[0].id, "test");
    }

    #[test]
    fn test_diversity_metrics() {
        let perturbed1 = array![[1.0, 2.0], [3.0, 4.0]];
        let perturbed2 = array![[1.5, 2.5], [3.5, 4.5]];
        let perturbed_data = vec![perturbed1, perturbed2];

        let metrics = calculate_diversity_metrics(&perturbed_data);

        assert!(metrics.coverage_score >= 0.0);
        assert!(metrics.uniqueness_score >= 0.0);
        assert!(metrics.variance_score >= 0.0);
    }

    #[test]
    fn test_confidence_intervals() {
        let perturbed_preds = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 1.9, 3.1],
            vec![0.9, 2.1, 2.9],
        ];

        let (lower_bounds, upper_bounds) = calculate_confidence_intervals(&perturbed_preds, 0.95);

        assert_eq!(lower_bounds.len(), 3);
        assert_eq!(upper_bounds.len(), 3);

        // Lower bounds should be <= upper bounds
        for (lower, upper) in lower_bounds.iter().zip(upper_bounds.iter()) {
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_outlier_detection() {
        let original_preds = vec![1.0, 2.0, 3.0];
        let perturbed_preds = vec![
            vec![1.1, 2.1, 3.1],  // Small changes
            vec![1.0, 2.0, 10.0], // Large change for third sample (outlier)
            vec![0.9, 1.9, 2.9],  // Small changes
        ];

        let outlier_analysis = detect_prediction_outliers(&original_preds, &perturbed_preds, 2.0);

        assert!(outlier_analysis.outlier_fraction > 0.0);
        assert!(outlier_analysis.sample_outlier_counts[2] > 0); // Third sample should have outliers
    }
}
