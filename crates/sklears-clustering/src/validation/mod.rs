//! Comprehensive Clustering Validation Framework
//!
//! This module provides a complete framework for evaluating clustering quality through
//! multiple complementary validation approaches. The framework is organized into
//! specialized submodules, each focused on specific aspects of cluster validation.
//!
//! # Module Organization
//!
//! ## Core Types and Configuration
//! - [`validation_types`]: Core types, metrics, and configuration structures
//!
//! ## Internal Validation (No Ground Truth Required)
//! - [`internal_validation`]: Silhouette analysis, Calinski-Harabasz, Davies-Bouldin indices
//! - [`gap_statistic`]: Gap statistic for optimal cluster number selection
//! - [`coherence_separation`]: Cluster coherence and separation analysis
//!
//! ## External Validation (Requires Ground Truth)
//! - [`external_validation`]: Adjusted Rand Index, Normalized Mutual Information, V-measure
//!
//! ## Stability Analysis
//! - [`stability_analysis`]: Bootstrap, perturbation, consensus clustering stability
//!
//! ## Automated Framework
//! - [`automated_validation`]: Comprehensive automated validation with recommendations
//!
//! ## Testing
//! - [`validation_tests`]: Comprehensive test suite for all validation components
//!
//! # Quick Start
//!
//! ## Basic Internal Validation
//! ```rust
//! use sklears_clustering::validation::*;
//! use sklears_core::prelude::*;
//!
//! let data = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 5.2, 7.8],
//! )
//! .unwrap();
//! let labels = vec![0, 0, 1, 1];
//!
//! // Quick internal validation
//! let validator = InternalValidationMethods::euclidean();
//! let silhouette = validator.silhouette_analysis(&data, &labels).unwrap();
//! let ch_index = validator.calinski_harabasz_index(&data, &labels).unwrap();
//!
//! println!("Silhouette score: {:.3}", silhouette.mean_silhouette);
//! println!("Calinski-Harabasz index: {:.3}", ch_index);
//! ```
//!
//! ## Comprehensive Automated Validation
//! ```rust
//! use sklears_clustering::validation::*;
//! use sklears_core::prelude::*;
//!
//! let data = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 5.2, 7.8],
//! )
//! .unwrap();
//! let labels = vec![0, 0, 1, 1];
//!
//! // Automated validation with recommendations
//! let validator = AutomatedValidator::euclidean();
//! let config = AutomatedValidationConfig::default();
//! let result = validator
//!     .automated_cluster_validation(&data, &labels, &config)
//!     .unwrap();
//!
//! println!("Overall quality: {:?}", result.overall_quality);
//! println!("Quality score: {:.3}", result.internal_quality_score);
//! for recommendation in &result.recommendations {
//!     println!("Recommendation: {}", recommendation);
//! }
//! ```
//!
//! ## External Validation with Ground Truth
//! ```rust
//! use sklears_clustering::validation::*;
//! use sklears_core::prelude::*;
//!
//! let predicted_labels = vec![0, 0, 1, 1, 2, 2];
//! let true_labels = vec![0, 0, 0, 1, 1, 1];
//!
//! let validator = InternalValidationMethods::euclidean();
//! let metrics = validator
//!     .external_validation(&true_labels, &predicted_labels)
//!     .unwrap();
//!
//! println!("Adjusted Rand Index: {:.3}", metrics.adjusted_rand_index);
//! println!("Normalized Mutual Information: {:.3}", metrics.normalized_mutual_info);
//! ```
//!
//! ## Stability Analysis
//! ```rust
//! use sklears_clustering::validation::*;
//! use sklears_core::prelude::*;
//!
//! let data = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 5.2, 7.8],
//! )
//! .unwrap();
//!
//! let analyzer = StabilityAnalyzer::euclidean();
//!
//! // Define a simple clustering function
//! let clustering_fn = |data: &Array2<f64>| -> Result<Vec<i32>> {
//!     // Your clustering algorithm here
//!     Ok(vec![0, 0, 1, 1]) // Placeholder
//! };
//!
//! let stability = analyzer
//!     .subsample_stability(&data, clustering_fn, 0.8, 10)
//!     .unwrap();
//! println!("Mean stability: {:.3}", stability.mean_stability);
//! ```
//!
//! # Validation Metrics Overview
//!
//! ## Internal Validation Metrics
//! - **Silhouette Coefficient**: Measures how similar points are to their own cluster vs. other clusters
//! - **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
//! - **Davies-Bouldin Index**: Average similarity between clusters (lower is better)
//! - **Dunn Index**: Ratio of minimum inter-cluster to maximum intra-cluster distance
//!
//! ## External Validation Metrics
//! - **Adjusted Rand Index (ARI)**: Similarity between clusterings, adjusted for chance
//! - **Normalized Mutual Information (NMI)**: Information-theoretic similarity measure
//! - **V-measure**: Harmonic mean of homogeneity and completeness
//! - **Fowlkes-Mallows Index**: Geometric mean of precision and recall
//!
//! ## Stability Metrics
//! - **Subsample Stability**: Consistency across random subsamples
//! - **Consensus Stability**: Agreement across multiple runs with different seeds
//! - **Perturbation Stability**: Robustness to data perturbation (noise injection)
//! - **Parameter Sensitivity**: Stability across parameter value ranges
//!
//! # Advanced Features
//!
//! ## Gap Statistic for Optimal K Selection
//! ```rust
//! use sklears_clustering::validation::*;
//! use sklears_core::prelude::*;
//!
//! let data = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 5.2, 7.8],
//! )
//! .unwrap();
//! let analyzer = ClusteringValidator::euclidean();
//!
//! let clustering_fn = |data: &Array2<f64>, k: usize| -> Result<Vec<i32>> {
//!     // Your clustering algorithm with k clusters
//!     Ok((0..data.nrows()).map(|i| (i % k) as i32).collect())
//! };
//!
//! let gap_result = analyzer
//!     .gap_statistic(&data, 1..4, Some(50), clustering_fn)
//!     .unwrap();
//! println!("Optimal k: {}", gap_result.optimal_k);
//! ```
//!
//! ## Coherence and Separation Analysis
//! ```rust
//! use sklears_clustering::validation::*;
//! use sklears_core::prelude::*;
//!
//! let data = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 5.2, 7.8],
//! )
//! .unwrap();
//! let labels = vec![0, 0, 1, 1];
//!
//! let analyzer = CoherenceSeparationAnalyzer::euclidean();
//! let coherence = analyzer.cluster_coherence(&data, &labels).unwrap();
//! let separation = analyzer.cluster_separation(&data, &labels).unwrap();
//!
//! println!("Overall coherence: {:.3}", coherence.overall_coherence);
//! println!("Gap ratio: {:.3}", separation.gap_ratio);
//! ```

// Module declarations
pub mod automated_validation;
pub mod coherence_separation;
pub mod external_validation;
pub mod gap_statistic;
pub mod internal_validation;
pub mod stability_analysis;
pub mod validation_types;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod validation_tests;

// Import external validation implementation for methods to be available

// Re-export core types for convenience
pub use validation_types::{
    ClusterQuality as ValidationQuality, ExternalValidationMetrics, GapStatisticResult,
    SilhouetteQuality, SilhouetteResult, ValidationConfig, ValidationMetric,
};

// Re-export main validator classes
pub use internal_validation::ClusteringValidator as InternalValidationMethods;
// TODO: GapStatisticAnalyzer - functionality is in ClusteringValidator
// pub use gap_statistic::GapStatisticAnalyzer;
pub use automated_validation::{
    AutomatedValidationConfig, AutomatedValidationResult, AutomatedValidator, ClusterQuality,
};
pub use coherence_separation::CoherenceSeparationAnalyzer;
pub use stability_analysis::StabilityAnalyzer;

// Re-export result types
pub use coherence_separation::{ClusterCoherenceResult, ClusterSeparationResult};
pub use stability_analysis::{
    BootstrapStabilityResult, ConsensusStabilityResult, CrossValidationStabilityResult,
    NoiseStabilityResult, ParameterSensitivityResult, PerturbationStabilityResult,
    SubsampleStabilityResult,
};

// Re-export test utilities for external testing
#[allow(non_snake_case)]
#[cfg(test)]
pub use validation_tests::{
    generate_clusters_with_noise, generate_elongated_clusters, generate_overlapping_clusters,
    generate_random_clusters, generate_well_separated_clusters,
};

/// Unified validation interface combining all validation approaches
///
/// This struct provides a single entry point for all clustering validation needs,
/// combining internal validation, external validation, stability analysis, and
/// automated recommendations in one convenient interface.
pub struct ClusteringValidator {
    metric: ValidationMetric,
    internal_validator: InternalValidationMethods,
    external_validator: ExternalValidationMetrics,
    stability_analyzer: StabilityAnalyzer,
    coherence_separation_analyzer: CoherenceSeparationAnalyzer,
    automated_validator: AutomatedValidator,
}

impl ClusteringValidator {
    /// Create a new comprehensive clustering validator
    pub fn new(metric: ValidationMetric) -> Self {
        Self {
            metric,
            internal_validator: InternalValidationMethods::new(metric),
            external_validator: ExternalValidationMetrics::default(),
            stability_analyzer: StabilityAnalyzer::new(metric),
            coherence_separation_analyzer: CoherenceSeparationAnalyzer::new(metric),
            automated_validator: AutomatedValidator::new(metric),
        }
    }

    /// Create validator with Euclidean distance
    pub fn euclidean() -> Self {
        Self::new(ValidationMetric::Euclidean)
    }

    /// Create validator with Manhattan distance
    pub fn manhattan() -> Self {
        Self::new(ValidationMetric::Manhattan)
    }

    /// Create validator with Cosine distance
    pub fn cosine() -> Self {
        Self::new(ValidationMetric::Cosine)
    }

    /// Get the distance metric used by this validator
    pub fn metric(&self) -> ValidationMetric {
        self.metric
    }

    // Delegate to internal validation methods

    /// Compute silhouette analysis
    pub fn silhouette_analysis(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<SilhouetteResult> {
        self.internal_validator.silhouette_analysis(X, labels)
    }

    /// Compute Calinski-Harabasz index
    pub fn calinski_harabasz_index(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<f64> {
        self.internal_validator.calinski_harabasz_index(X, labels)
    }

    /// Compute Davies-Bouldin index
    pub fn davies_bouldin_index(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<f64> {
        self.internal_validator.davies_bouldin_index(X, labels)
    }

    /// Compute Dunn index
    pub fn dunn_index(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<f64> {
        self.internal_validator.dunn_index(X, labels)
    }

    // Note: External validation methods (adjusted_rand_index, normalized_mutual_information,
    // v_measure, fowlkes_mallows_index) are implemented in external_validation.rs

    // Delegate to gap statistic analyzer

    /// Compute gap statistic for optimal cluster selection
    pub fn gap_statistic<F>(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        k_range: std::ops::Range<usize>,
        n_refs: Option<usize>,
        clustering_fn: F,
    ) -> sklears_core::error::Result<GapStatisticResult>
    where
        F: Fn(&scirs2_core::ndarray::Array2<f64>, usize) -> sklears_core::error::Result<Vec<i32>>,
    {
        self.internal_validator
            .gap_statistic(X, k_range, n_refs, clustering_fn)
    }

    // Delegate to stability analyzer

    /// Compute subsample stability
    pub fn subsample_stability<F>(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        clustering_fn: F,
        subsample_ratio: f64,
        n_trials: usize,
    ) -> sklears_core::error::Result<SubsampleStabilityResult>
    where
        F: Fn(&scirs2_core::ndarray::Array2<f64>) -> sklears_core::error::Result<Vec<i32>>,
    {
        self.stability_analyzer
            .subsample_stability(X, clustering_fn, subsample_ratio, n_trials)
    }

    /// Compute consensus stability
    pub fn consensus_stability<F>(
        &self,
        clusterer: F,
        X: &scirs2_core::ndarray::Array2<f64>,
        n_runs: usize,
        seeds: Option<Vec<u64>>,
    ) -> sklears_core::error::Result<ConsensusStabilityResult>
    where
        F: Fn(&scirs2_core::ndarray::Array2<f64>, u64) -> sklears_core::error::Result<Vec<i32>>,
    {
        self.stability_analyzer
            .consensus_stability(clusterer, X, n_runs, seeds)
    }

    // Delegate to coherence and separation analyzer

    /// Compute cluster coherence analysis
    pub fn cluster_coherence(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<ClusterCoherenceResult> {
        self.coherence_separation_analyzer
            .cluster_coherence(X, labels)
    }

    /// Compute cluster separation analysis
    pub fn cluster_separation(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<ClusterSeparationResult> {
        self.coherence_separation_analyzer
            .cluster_separation(X, labels)
    }

    // Delegate to automated validator

    /// Perform comprehensive automated validation
    pub fn automated_cluster_validation(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
        validation_config: &AutomatedValidationConfig,
    ) -> sklears_core::error::Result<AutomatedValidationResult> {
        self.automated_validator
            .automated_cluster_validation(X, labels, validation_config)
    }

    /// Quick validation with default settings
    pub fn quick_validation(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<AutomatedValidationResult> {
        let config = AutomatedValidationConfig::fast();
        self.automated_cluster_validation(X, labels, &config)
    }

    /// Comprehensive validation with all metrics
    pub fn comprehensive_validation(
        &self,
        X: &scirs2_core::ndarray::Array2<f64>,
        labels: &[i32],
    ) -> sklears_core::error::Result<AutomatedValidationResult> {
        let config = AutomatedValidationConfig::comprehensive();
        self.automated_cluster_validation(X, labels, &config)
    }
}

/// Convenience function for quick silhouette analysis
pub fn silhouette_score(
    X: &scirs2_core::ndarray::Array2<f64>,
    labels: &[i32],
) -> sklears_core::error::Result<f64> {
    let validator = ClusteringValidator::euclidean();
    let result = validator.silhouette_analysis(X, labels)?;
    Ok(result.mean_silhouette)
}

/// Convenience function for quick Calinski-Harabasz index
pub fn calinski_harabasz_score(
    X: &scirs2_core::ndarray::Array2<f64>,
    labels: &[i32],
) -> sklears_core::error::Result<f64> {
    let validator = ClusteringValidator::euclidean();
    validator.calinski_harabasz_index(X, labels)
}

/// Convenience function for quick Davies-Bouldin index
pub fn davies_bouldin_score(
    X: &scirs2_core::ndarray::Array2<f64>,
    labels: &[i32],
) -> sklears_core::error::Result<f64> {
    let validator = ClusteringValidator::euclidean();
    validator.davies_bouldin_index(X, labels)
}

/// Convenience function for quick Adjusted Rand Index
pub fn adjusted_rand_score(
    true_labels: &[i32],
    pred_labels: &[i32],
) -> sklears_core::error::Result<f64> {
    let validator = InternalValidationMethods::euclidean();
    validator.adjusted_rand_index(true_labels, pred_labels)
}

/// Convenience function for quick Normalized Mutual Information
pub fn normalized_mutual_info_score(
    true_labels: &[i32],
    pred_labels: &[i32],
) -> sklears_core::error::Result<f64> {
    let validator = InternalValidationMethods::euclidean();
    validator.normalized_mutual_information(true_labels, pred_labels)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_unified_validator() {
        let data = array![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [5.2, 7.8],];
        let labels = vec![0, 0, 1, 1];

        let validator = ClusteringValidator::euclidean();

        // Test internal validation
        let silhouette = validator.silhouette_analysis(&data, &labels).unwrap();
        assert!(silhouette.mean_silhouette >= -1.0 && silhouette.mean_silhouette <= 1.0);

        let ch_index = validator.calinski_harabasz_index(&data, &labels).unwrap();
        assert!(ch_index >= 0.0);

        let db_index = validator.davies_bouldin_index(&data, &labels).unwrap();
        assert!(db_index >= 0.0);

        // Test external validation
        let ari = validator
            .internal_validator
            .adjusted_rand_index(&labels, &labels)
            .unwrap();
        assert!((ari - 1.0).abs() < 1e-10); // Perfect match

        // Test coherence and separation
        let coherence = validator.cluster_coherence(&data, &labels).unwrap();
        assert!(coherence.overall_coherence >= 0.0);

        let separation = validator.cluster_separation(&data, &labels).unwrap();
        assert!(separation.gap_ratio >= 0.0);

        // Test automated validation
        let result = validator.quick_validation(&data, &labels).unwrap();
        assert!(result.internal_quality_score >= 0.0 && result.internal_quality_score <= 1.0);
    }

    #[test]
    fn test_convenience_functions() {
        let data = array![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [5.2, 7.8],];
        let labels = vec![0, 0, 1, 1];

        let sil_score = silhouette_score(&data, &labels).unwrap();
        assert!(sil_score >= -1.0 && sil_score <= 1.0);

        let ch_score = calinski_harabasz_score(&data, &labels).unwrap();
        assert!(ch_score >= 0.0);

        let db_score = davies_bouldin_score(&data, &labels).unwrap();
        assert!(db_score >= 0.0);

        let ari_score = adjusted_rand_score(&labels, &labels).unwrap();
        assert!((ari_score - 1.0).abs() < 1e-10);

        let nmi_score = normalized_mutual_info_score(&labels, &labels).unwrap();
        assert!((nmi_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_different_metrics() {
        let data = array![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [5.2, 7.8],];
        let labels = vec![0, 0, 1, 1];

        let euclidean_validator = ClusteringValidator::euclidean();
        let manhattan_validator = ClusteringValidator::manhattan();
        let cosine_validator = ClusteringValidator::cosine();

        let euc_sil = euclidean_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let man_sil = manhattan_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let cos_sil = cosine_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();

        // All should produce valid results
        assert!(euc_sil.mean_silhouette >= -1.0 && euc_sil.mean_silhouette <= 1.0);
        assert!(man_sil.mean_silhouette >= -1.0 && man_sil.mean_silhouette <= 1.0);
        assert!(cos_sil.mean_silhouette >= -1.0 && cos_sil.mean_silhouette <= 1.0);

        // Results should be different for different metrics
        assert_ne!(euc_sil.mean_silhouette, man_sil.mean_silhouette);
    }

    #[test]
    fn test_validator_properties() {
        let validator = ClusteringValidator::manhattan();
        assert_eq!(validator.metric(), ValidationMetric::Manhattan);
    }
}
