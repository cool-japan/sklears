#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrow)]
//! Evaluation metrics for machine learning models
//!
//! This crate provides a comprehensive set of metrics for evaluating machine learning models,
//! including classification, regression, and clustering metrics.
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
//! use scirs2_core::ndarray::array;
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 1];
//!
//! let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
//! println!("Accuracy: {:.3}", accuracy);
//! ```
//!
//! # Classification Metrics
//!
//! - `accuracy_score`: Classification accuracy
//! - `precision_score`, `recall_score`, `f1_score`: Precision, recall, and F1 score
//! - `fbeta_score`: F-beta score with configurable beta
//! - `balanced_accuracy_score`: Accuracy adjusted for imbalanced datasets
//! - `cohen_kappa_score`: Cohen's kappa coefficient
//! - `matthews_corrcoef`: Matthews correlation coefficient
//! - `confusion_matrix`: Confusion matrix
//! - `log_loss`: Logarithmic loss
//! - `hamming_loss`: Hamming loss
//! - `jaccard_score`: Jaccard similarity coefficient
//! - `zero_one_loss`: Zero-one classification loss
//! - `hinge_loss`: Hinge loss for SVM
//! - `brier_score_loss`: Brier score for probabilistic predictions
//! - `top_k_accuracy_score`: Top-k accuracy for multi-class classification
//! - `top_2_accuracy_score`, `top_3_accuracy_score`, `top_5_accuracy_score`: Convenience functions
//! - `multilabel_exact_match_ratio`: Exact match ratio for multi-label classification
//! - `multilabel_accuracy_score`: Jaccard similarity for multi-label classification
//! - `multilabel_hamming_loss`: Hamming loss for multi-label classification
//! - `multilabel_ranking_loss`: Ranking loss for multi-label classification
//! - `multilabel_average_precision_score`: Average precision for multi-label classification
//! - `demographic_parity_difference`: Fairness metric measuring difference in positive prediction rates
//! - `demographic_parity_ratio`: Fairness metric measuring ratio of positive prediction rates
//! - `equalized_odds_difference`: Fairness metric measuring difference in true positive rates
//! - `equal_opportunity_difference`: Alias for equalized_odds_difference
//! - `expected_calibration_error`: Expected Calibration Error (ECE) for model calibration assessment
//! - `reliability_diagram`: Data for creating reliability diagrams (calibration plots)
//! - `cost_sensitive_accuracy`: Cost-sensitive accuracy for imbalanced classification
//! - `cost_sensitive_loss`: Cost-sensitive loss function
//! - `expected_cost`: Expected cost given prediction probabilities
//! - `cost_weighted_f1_score`: Cost-weighted F1 score
//! - `spherical_score`: Spherical scoring rule for probabilistic classification
//! - `quadratic_score`: Quadratic scoring rule for probabilistic classification
//! - `cross_entropy`: Cross-entropy loss for multi-class classification
//! - `kl_divergence`: Kullback-Leibler divergence between probability distributions
//! - `js_divergence`: Jensen-Shannon divergence (symmetric version of KL divergence)
//! - `hierarchical_precision`: Hierarchical precision with partial credit for ancestor matches
//! - `hierarchical_recall`: Hierarchical recall with partial credit for ancestor matches
//! - `hierarchical_f1_score`: Hierarchical F1-score for tree-structured labels
//! - `tree_distance_loss`: Tree distance-based loss for hierarchical classification
//! - `hierarchical_accuracy`: Hierarchical accuracy with optional partial credit
//!
//! # Regression Metrics
//!
//! - `mean_absolute_error`: Mean absolute error (MAE)
//! - `mean_squared_error`: Mean squared error (MSE)
//! - `root_mean_squared_error`: Root mean squared error (RMSE)
//! - `r2_score`: Coefficient of determination (R²)
//! - `mean_absolute_percentage_error`: Mean absolute percentage error (MAPE)
//! - `explained_variance_score`: Explained variance score
//! - `max_error`: Maximum error
//! - `median_absolute_error`: Median absolute error
//! - `mean_squared_log_error`: Mean squared logarithmic error
//! - `root_mean_squared_log_error`: Root mean squared logarithmic error
//! - `mean_gamma_deviance`: Mean Gamma deviance
//! - `mean_poisson_deviance`: Mean Poisson deviance
//! - `mean_tweedie_deviance`: Mean Tweedie deviance with power parameter
//! - `mean_pinball_loss`: Pinball loss for quantile regression
//! - `d2_absolute_error_score`: D² score based on absolute error
//! - `d2_pinball_score`: D² score based on pinball loss
//! - `d2_tweedie_score`: D² score based on Tweedie deviance
//! - `huber_loss`: Huber loss for robust regression
//! - `quantile_loss`: Quantile loss (alias for pinball loss)
//! - `median_absolute_percentage_error`: Median absolute percentage error (MdAPE)
//! - `theil_u_statistic`: Theil's U statistic for forecast accuracy
//! - `symmetric_mean_absolute_percentage_error`: Symmetric MAPE (sMAPE)
//! - `mean_absolute_scaled_error`: Mean Absolute Scaled Error (MASE) for time series
//! - `robust_r2_score`: Robust R-squared variants (median, trimmed, Huber, MAD-based)
//! - `median_r2_score`: Median-based robust R-squared
//! - `trimmed_r2_score`: Trimmed R-squared (excluding outliers)
//! - `huber_r2_score`: Huber loss-based robust R-squared
//! - `mad_r2_score`: Median Absolute Deviation-based R-squared
//! - `continuous_ranked_probability_score`: CRPS for distributional forecasts
//! - `crps_ensemble`: CRPS for ensemble forecasts (convenience function)
//! - `crps_gaussian`: CRPS for Gaussian predictions with analytical formula
//! - `energy_score`: Energy score for multivariate distributional forecasts
//! - `direction_accuracy`: Directional accuracy percentage for time series forecasting
//! - `directional_symmetry`: Balance between upward and downward predictions
//! - `hit_rate`: Hit rate for directional predictions above a threshold
//! - `directional_theil_u`: Directional adaptation of Theil's U statistic
//! - `trend_accuracy`: Trend accuracy using sliding window comparison
//! - `prediction_interval_coverage`: Proportion of true values within prediction intervals
//! - `mean_interval_width`: Average width of prediction intervals  
//! - `interval_score`: Proper scoring rule combining coverage and width
//! - `probability_integral_transform`: PIT for distributional forecast evaluation
//! - `pit_uniformity_test`: Test PIT values for uniform distribution
//! - `diebold_mariano_test`: Statistical test for comparing forecast accuracy
//! - `seasonal_mase`: Seasonal Mean Absolute Scaled Error using seasonal naive forecast
//! - `seasonal_naive_forecast_error`: Relative error compared to seasonal naive forecast
//! - `seasonal_autocorrelation`: Autocorrelation coefficient at seasonal lag
//! - `seasonal_strength`: Strength of seasonal component (0 to 1)
//! - `persistence_model_comparison`: Compare against simple, seasonal, and trend persistence models
//!
//! # Ranking Metrics
//!
//! - `auc`: Area Under the Curve
//! - `average_precision_score`: Average precision from precision-recall curve
//! - `coverage_error`: Coverage error for multi-label ranking
//! - `dcg_score`: Discounted Cumulative Gain
//! - `ndcg_score`: Normalized Discounted Cumulative Gain
//! - `roc_auc_score`: Area Under the ROC Curve (binary)
//! - `roc_auc_score_multiclass`: Multi-class ROC AUC with OvR/OvO strategies
//! - `precision_recall_auc_score`: Area Under the Precision-Recall Curve
//! - `roc_curve`: Compute ROC curve coordinates
//! - `precision_recall_curve`: Compute precision-recall curve coordinates
//! - `mean_average_precision`: Mean Average Precision (MAP) for information retrieval
//! - `mean_reciprocal_rank`: Mean Reciprocal Rank (MRR) for ranking evaluation
//!
//! # Clustering Metrics
//!
//! - `adjusted_rand_score`: Adjusted Rand index
//! - `adjusted_mutual_info_score`: Adjusted mutual information
//! - `calinski_harabasz_score`: Calinski-Harabasz index (variance ratio criterion)
//! - `completeness_score`: Completeness metric
//! - `davies_bouldin_score`: Davies-Bouldin index
//! - `fowlkes_mallows_score`: Fowlkes-Mallows index
//! - `homogeneity_score`: Homogeneity metric
//! - `homogeneity_completeness_v_measure`: All three metrics in one call
//! - `mutual_info_score`: Mutual information
//! - `normalized_mutual_info_score`: Normalized mutual information
//! - `rand_score`: Rand index
//! - `silhouette_score`: Silhouette coefficient
//! - `v_measure_score`: V-measure (harmonic mean of homogeneity and completeness)
//! - `dunn_index`: Dunn index for cluster separation assessment
//! - `gap_statistic`: Gap statistic for optimal number of clusters
//! - `within_cluster_sum_of_squares`: Within-cluster sum of squares for cluster compactness
//! - `between_cluster_sum_of_squares`: Between-cluster sum of squares for cluster separation
//! - `bootstrap_stability`: Bootstrap stability for clustering robustness assessment
//! - `jaccard_stability`: Jaccard stability coefficient between clusterings
//! - `consensus_clustering_stability`: Consensus clustering stability metric
//! - `perturbation_stability`: Clustering stability under data perturbations
//! - `parameter_stability`: Clustering stability across parameter variations
//! - `entropy`: Shannon entropy for discrete distributions
//! - `conditional_entropy`: Conditional entropy H(Y|X)
//! - `mutual_information`: Mutual information I(X; Y) between variables
//! - `normalized_mutual_information_symmetric`: Normalized mutual information (symmetric)
//! - `joint_entropy`: Joint entropy H(X, Y) of two variables
//! - `variation_of_information`: Variation of information distance between clusterings
//! - `information_gain`: Information gain (entropy reduction)
//! - `information_gain_ratio`: Normalized information gain
//! - `intra_cluster_coherence`: Average pairwise similarity within clusters
//! - `inter_cluster_separation`: Average dissimilarity between clusters
//! - `cluster_coherence_score`: Combined coherence score with configurable weighting
//! - `semantic_coherence`: Semantic coherence for text clustering using word co-occurrence
//! - `xie_beni_index`: Validity index for fuzzy clustering (compactness to separation ratio)
//! - `ball_hall_index`: Internal validity measure (average within-cluster distance)
//! - `hartigan_index`: Stopping rule for k-means clustering
//! - `krzanowski_lai_index`: Stopping rule using rate of change in WCSS
//! - `bic_clustering`: Bayesian Information Criterion for clustering
//! - `aic_clustering`: Akaike Information Criterion for clustering
//! - `sugar_james_index`: Model selection criterion based on distortion and degrees of freedom
//!
//! # Computer Vision Metrics
//!
//! - `psnr`: Peak Signal-to-Noise Ratio for image quality assessment
//! - `ssim`: Structural Similarity Index for perceptual image quality
//! - `iou_boxes`: Intersection over Union for bounding boxes
//! - `iou_masks`: Intersection over Union for segmentation masks
//! - `mean_average_precision`: Mean Average Precision (mAP) for object detection
//! - `mean_iou`: Mean Intersection over Union for semantic segmentation
//! - `pixel_accuracy`: Pixel-wise accuracy for segmentation tasks
//! - `Detection`: Structure for object detection results
//! - `GroundTruth`: Structure for ground truth annotations
//!
//! # Natural Language Processing Metrics
//!
//! - `bleu_score`: BLEU score for machine translation evaluation
//! - `rouge_n_score`: ROUGE-N score for summarization evaluation
//! - `rouge_l_score`: ROUGE-L score using longest common subsequence
//! - `perplexity`: Perplexity for language model evaluation
//! - `jaccard_similarity`: Jaccard similarity coefficient for text
//! - `cosine_similarity_tfidf`: Cosine similarity using TF-IDF vectors
//! - `edit_distance`: Levenshtein distance between strings
//! - `normalized_edit_distance`: Normalized edit distance between strings
//! - `SmoothingFunction`: Enumeration of smoothing methods for BLEU
//!
//! # Survival Analysis Metrics
//!
//! - `concordance_index`: C-index for survival analysis
//! - `time_dependent_auc`: Time-dependent AUC for survival predictions
//! - `brier_score_survival`: Brier score for survival analysis
//! - `integrated_brier_score`: Integrated Brier score over time
//! - `kaplan_meier_survival`: Kaplan-Meier survival function estimation
//! - `log_rank_test`: Log-rank test for comparing survival curves
//!
//! # Pairwise Metrics
//!
//! - `euclidean_distances`: Compute euclidean distances between samples
//! - `nan_euclidean_distances`: Euclidean distances ignoring NaN values
//! - `pairwise_distances`: Compute distances with various metrics (Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, Hamming)
//! - `pairwise_distances_argmin`: Find minimum distances and indices
//! - `pairwise_distances_argmin_min`: Find both argmin and min values
//! - `pairwise_kernels`: Compute kernel matrix (Linear, Polynomial, RBF, Sigmoid, Cosine)
//! - `wasserstein_distance`: Earth Mover's Distance between 1D distributions
//! - `mahalanobis_distances`: Mahalanobis distance accounting for correlations
//! - `cosine_similarity`: Cosine similarity matrix between samples
//! - `normalized_compression_distance`: Universal metric based on compression algorithms
//! - `normalized_compression_distance_matrix`: NCD between all pairs of sequences
//! - `approximate_kolmogorov_complexity`: Estimate Kolmogorov complexity using compression
//! - `information_distance`: Information distance between byte sequences
//! - `string_kernel_similarity`: Subsequence-based similarity for strings
//! - `string_kernel_matrix`: Pairwise string kernel similarities
//!
//! # Statistical Tests
//!
//! - `mcnemar_test`: Compare two binary classifiers using McNemar's test
//! - `friedman_test`: Compare multiple algorithms across datasets
//! - `wilcoxon_signed_rank_test`: Non-parametric test for paired samples
//! - `permutation_test`: General permutation test framework
//! - `transfer_entropy`: Directed information transfer between time series
//! - `bidirectional_transfer_entropy`: Transfer entropy in both directions
//! - `net_transfer_entropy`: Net information flow between time series
//! - `multi_lag_transfer_entropy`: Transfer entropy across multiple lags
//! - `partial_transfer_entropy`: Transfer entropy conditioning on third variable
//!
//! # Temporal and Dynamic Metrics
//!
//! - `TemporalMetricsAnalyzer`: Main analyzer for temporal metric patterns
//! - `detect_concept_drift`: Detect concept drift using multiple methods (KS test, Page-Hinkley, ADWIN, SPC, PSI)
//! - `analyze_temporal_trend`: Analyze trends with seasonal decomposition and change point detection
//! - `calculate_adaptive_weights`: Calculate time-decaying weights for temporal data
//! - `temporal_stability`: Measure temporal stability of metrics over time
//! - `track_metric_evolution`: Track how metrics evolve with rolling statistics
//! - `ConceptDriftResult`: Results of drift detection with confidence and magnitude
//! - `TemporalTrendAnalysis`: Comprehensive trend analysis with seasonality detection
//! - `MetricEvolution`: Evolution tracking with change point detection
//! - `WindowConfig`: Configuration for sliding windows and drift detection parameters
//!
//! # Interpretability Metrics
//!
//! - `calculate_faithfulness_removal`: Faithfulness using feature removal/occlusion
//! - `calculate_faithfulness_permutation`: Faithfulness using feature permutation
//! - `calculate_explanation_stability`: Stability analysis using correlation, cosine similarity, rank correlation
//! - `calculate_comprehensibility`: Comprehensibility assessment with sparsity and complexity measures
//! - `calculate_trustworthiness`: Comprehensive trustworthiness combining multiple metrics
//! - `evaluate_ranking_quality`: Quality assessment for feature importance rankings
//! - `FaithfulnessResult`: Detailed faithfulness evaluation with confidence intervals
//! - `StabilityResult`: Pairwise stability analysis across explanations
//! - `ComprehensibilityResult`: Comprehensibility assessment with entropy and consistency
//! - `TrustworthinessResult`: Combined trustworthiness score with individual components
//! - `RankingQualityResult`: Feature importance ranking validation and consistency
//!
//! # Multi-Objective Evaluation
//!
//! - `pareto_frontier`: Find Pareto optimal solutions (non-dominated models)
//! - `topsis_ranking`: TOPSIS multi-criteria decision analysis
//! - `weighted_sum_ranking`: Weighted sum approach for model ranking
//! - `trade_off_analysis`: Analyze trade-offs between competing metrics
//! - `utility_optimization`: Optimize custom utility functions
//! - `multi_objective_evaluation`: Comprehensive multi-objective evaluation
//! - `MultiObjectiveResult`: Structure containing complete evaluation results
//! - `MultiObjectiveConfig`: Configuration for multi-objective evaluation
//!
//! # Uncertainty Quantification
//!
//! - `bootstrap_confidence_interval`: Bootstrap confidence intervals for any metric
//! - `bca_bootstrap_confidence_interval`: Bias-corrected and accelerated bootstrap
//! - `bayesian_accuracy_credible_interval`: Bayesian credible intervals for accuracy
//! - `correlation_confidence_interval`: Analytical confidence intervals for correlation
//! - `mse_confidence_interval`: Confidence intervals for mean squared error
//! - `uncertainty_propagation`: Uncertainty propagation for composite metrics
//! - `bootstrap_metric_comparison`: Bootstrap hypothesis testing for metric comparison
//! - `UncertaintyResult`: Structure containing uncertainty quantification results
//! - `UncertaintyConfig`: Configuration for uncertainty quantification
//!
//! # Type Safety and Compile-Time Validation
//!
//! - `TypedMetric`: Type-safe metric wrapper with phantom types
//! - `MetricCategory`: Trait for defining metric categories
//! - `Classification`, `Regression`, `Clustering`: Phantom types for metric categories
//! - `Metric`: Trait for computable metrics with type safety
//! - `MetricSuite`: Type-safe collection of metrics from the same category
//! - `CompositeMetric`: Type-safe composition of metrics from different categories
//! - `MetricTransform`: Trait for type-safe metric transformations
//! - `MetricBuilder`: Builder pattern for type-safe metric construction
//! - `ValidatedMetric`: Compile-time validation using const generics
//! - `ZeroCostMetric`: Zero-cost abstraction for metric computation
//!
//! # Performance Enhancements
//!
//! - `HighPerformanceMetricsComputer`: All-in-one high-performance metrics computation
//! - `AdaptiveMetricsComputer`: Adaptive algorithm selection based on data characteristics
//! - `CacheFriendlyAccumulator`: Cache-aligned data structures for metric accumulation
//! - `LockFreeMetricsAccumulator`: Lock-free concurrent metrics accumulation
//! - `PrefetchingMetricsComputer`: Memory prefetching for improved cache performance
//! - `CacheOptimizedMatrixOps`: Cache-conscious matrix operations
//! - `ProfileGuidedOptimizer`: Profile-guided optimization and performance analysis
//! - `MemoryPrefetcher`: Memory prefetching utilities for cache optimization
//!
//! # Validation Framework
//!
//! - `MetricValidator`: Comprehensive validation framework for metric correctness
//! - `SyntheticDataGenerator`: Generate synthetic data for testing metric implementations
//! - `ReferenceTestCase`: Reference test cases with known expected results
//! - `ValidationResult`: Results of metric validation with error analysis
//! - `ComprehensiveValidationReport`: Complete validation report with multiple test types
//! - `StabilityAnalysis`: Bootstrap-based stability analysis for metric robustness
//! - `MetamericAnalysis`: Parameter sensitivity analysis for understanding metric behavior
//! - `StandardReferenceDatasets`: Standard test cases for common metrics
//!
//! # Automated Benchmarking
//!
//! - `BenchmarkSuite`: Comprehensive benchmarking suite for metrics validation
//! - `add_classification_benchmark`: Add classification metric benchmarks with datasets
//! - `add_regression_benchmark`: Add regression metric benchmarks with datasets  
//! - `add_clustering_benchmark`: Add clustering metric benchmarks with datasets
//! - `run_all`: Execute all benchmarks with performance and accuracy testing
//! - `run_scalability_test`: Test metric performance across different data sizes
//! - `BenchmarkResult`: Detailed benchmark results with timing and accuracy
//! - `BenchmarkReport`: Comprehensive benchmarking report with statistics
//! - `Dataset`: Standard datasets (Iris, Wine, BreastCancer, BostonHousing, etc.)
//! - `ReferenceImplementations`: Reference metric implementations for validation
//!
//! # Interactive Visualization
//!
//! - `create_roc_curve_data`: Generate ROC curve data with AUC calculation
//! - `create_precision_recall_data`: Generate PR curve data with average precision
//! - `create_calibration_plot_data`: Generate calibration plots for probability assessment
//! - `RocCurveData`: ROC curve visualization with HTML plot generation
//! - `PrecisionRecallData`: PR curve visualization with interactive features
//! - `ConfusionMatrixVisualization`: Confusion matrix heatmaps with normalization
//! - `CalibrationPlot`: Calibration plots with Brier score and ECE metrics
//! - `LearningCurve`: Learning curve visualization for training progress
//! - `FeatureImportanceViz`: Feature importance bar charts with ranking
//! - `MetricDashboard`: Comprehensive metric comparison dashboard
//! - `PlotConfig`: Configuration for plot styling and interactivity
//!
//! # Federated Learning Metrics
//!
//! - `privacy_preserving_aggregation`: Differentially private metric aggregation
//! - `communication_efficient_aggregation`: Weighted aggregation with compression effects
//! - `demographic_parity_across_clients`: Fairness evaluation using coefficient of variation
//! - `equalized_odds_across_clients`: Equalized odds difference across federated clients
//! - `communication_efficiency`: Measure efficiency as improvement per unit communication cost
//! - `client_contribution_score`: Assess individual client contributions using Shapley-like values
//! - `shapley_client_contributions`: Calculate exact Shapley values for client coalitions
//! - `privacy_budget_allocation`: Track differential privacy budget across federated rounds
//! - `analyze_convergence`: Convergence analysis for federated training with stability measures
//! - `comprehensive_federated_evaluation`: Complete federated learning evaluation framework
//! - `secure_aggregation`: Secure multiparty computation for metric aggregation
//! - `FederatedConfig`: Configuration for federated learning evaluation parameters
//! - `FederatedEvaluationResult`: Comprehensive results including global metrics, fairness, efficiency
//! - `ConvergenceMetrics`: Convergence rate, stability, and rounds to convergence analysis
//! - `PrivacyComposition`: Privacy composition methods (Basic, Advanced, RDP)
//!
//! # Adversarial Robustness Metrics
//!
//! - `adversarial_accuracy`: Accuracy on adversarial examples compared to true labels
//! - `attack_success_rate`: Fraction of examples where predictions changed due to perturbations
//! - `robust_accuracy`: Accuracy within a specific perturbation budget constraint
//! - `certified_accuracy`: Accuracy with formal certification guarantees against perturbations
//! - `average_perturbation_magnitude`: Average L-p norm of adversarial perturbations
//! - `robustness_score`: Weighted combination of clean and adversarial accuracy
//! - `adversarial_transferability`: Success rate of adversarial examples across different models
//! - `gradient_based_robustness`: Local intrinsic dimensionality using gradient information
//! - `adaptive_attack_resistance`: Resistance to adaptive attacks accounting for gradient masking
//! - `empirical_robustness`: Robustness estimation using random noise perturbations
//! - `area_under_robustness_curve`: AURC metric for robustness across perturbation budgets
//! - `comprehensive_adversarial_evaluation`: Complete adversarial evaluation framework
//! - `AdversarialConfig`: Configuration for perturbation budgets, norms, and attack parameters
//! - `AdversarialResult`: Comprehensive results including multiple attack types and metrics
//! - `AttackResult`: Individual attack results with success rates and perturbation statistics
//! - `NormType`: Perturbation norm types (L-infinity, L2, L1, L0)
//! - `AttackType`: Supported attack methods (FGSM, PGD, C&W, AutoAttack, etc.)
//!
//! # Automated Reporting
//!
//! - `generate_metric_report`: Create comprehensive metric reports with statistical analysis
//! - `generate_model_comparison_report`: Automated comparison reports with significance testing
//! - `MetricReport`: Complete report structure with metadata, summaries, and recommendations
//! - `MetricSummary`: Individual metric summaries with interpretations and confidence intervals
//! - `ExecutiveSummary`: High-level summary for stakeholders with key findings and business impact
//! - `ModelComparison`: Pairwise model comparison with statistical and practical significance
//! - `StatisticalAnalysis`: Sample size analysis, confidence intervals, and power analysis
//! - `Recommendation`: Automated recommendations with priority levels and implementation guidance
//! - `PerformanceTrends`: Trend analysis and performance regression detection
//! - `ReportConfig`: Configuration for report generation including formats and significance thresholds
//! - `ReportFormat`: Output formats (HTML, Markdown, Text, JSON)
//! - `PerformanceGrade`: Performance grading system (Excellent, Good, Fair, Poor, Critical)
//! - `SignificanceTest`: Statistical significance test results with p-values and effect sizes
//! - `PracticalSignificance`: Practical significance assessment beyond statistical significance
//!
//! # Fluent API and Builder Patterns
//!
//! - `MetricsBuilder`: Fluent API for metric computation with method chaining
//! - `MetricPreset`: Configuration presets for common use cases (ClassificationBasic, RegressionBasic, etc.)
//! - `MetricConfig`: Configuration for metric computation with confidence intervals and averaging
//! - `MetricResults`: Serializable metric results with metadata and confidence intervals
//! - `ConfigBuilder`: Builder pattern for advanced metric configuration
//! - `quick_classification_metrics`: Convenience function for rapid classification evaluation
//! - `quick_regression_metrics`: Convenience function for rapid regression evaluation
//! - `AveragingStrategy`: Strategies for multi-class averaging (Macro, Micro, Weighted, Binary)
//! - `ZeroDivisionStrategy`: Handling strategies for division by zero in metrics
//!
//! # Async Streaming Metrics (feature = "async")
//!
//! - `StreamingMetricsComputer`: Async streaming metric computation with backpressure handling
//! - `MetricStream`: Stream wrapper for real-time metric computation with reactive updates
//! - `ChannelMetricsComputer`: Channel-based async metric computation for producer-consumer patterns
//! - `StreamingConfig`: Configuration for async streaming operations (chunk size, concurrency, windowing)
//! - `MetricAccumulator`: Incremental metric accumulation for streaming data with sliding windows
//! - `streaming_accuracy`: Convenience function for streaming accuracy computation
//! - `streaming_classification_metrics`: Convenience function for comprehensive streaming classification evaluation
//!
//! # Modular Framework
//!
//! - `Metric`: Core trait for all metrics with type-safe input/output
//! - `ComposableMetric`: Trait for metrics that can be combined and transformed
//! - `MetricAggregator`: Trait for aggregating multiple metric results
//! - `MetricPipeline`: Pipeline for composing multiple metrics and aggregators
//! - `MetricMiddleware`: Middleware system for metric processing pipelines
//! - `MetricRegistry`: Dynamic registration and discovery of metrics
//! - `ScoringFunction`: Extensible scoring function system
//! - `MetricPlugin`: Plugin architecture for extending framework capabilities
//! - `PluginManager`: Manager for loading and organizing metric plugins

pub mod advanced_metrics;
pub mod adversarial_robustness;
#[cfg(feature = "async")]
pub mod async_streaming;
pub mod automated_reporting;
pub mod basic_metrics;
pub mod benchmarking;
pub mod classification;
pub mod clustering;
pub mod computer_vision;
pub mod display_utils;
pub mod mathematical_foundations;
// Distributed metrics require full MPI with libffi, which doesn't work on macOS ARM64
#[cfg(all(feature = "distributed", not(target_os = "macos")))]
pub mod distributed_metrics;
pub mod fairness_metrics;
pub mod federated_learning;
pub mod fluent_api;
#[cfg(feature = "cuda")]
pub mod gpu_acceleration;
pub mod interpretability;
#[cfg(feature = "latex")]
pub mod latex_export;
pub mod memory_efficiency;
pub mod modular_framework;
pub mod multi_objective;
pub mod multilabel_metrics;
pub mod nlp;
pub mod optimized;
pub mod pairwise;
pub mod performance_enhancements;
pub mod probabilistic_metrics;
pub mod ranking;
pub mod regression;
pub mod scoring;
pub mod statistical_tests;
pub mod survival;
pub mod temporal;
pub mod thread_local_optimization;
pub mod type_safety;
pub mod uncertainty;
mod utils;
pub mod validation;
pub mod visualization;

// Re-export scoring utilities
pub use scoring::{
    check_multimetric_scoring, check_scoring, get_scorer, get_scorer_names, make_scorer, Scorer,
    ScorerConfig, ScoringMetric,
};

// Re-export display classes for enhanced user experience
pub use classification::{
    ClassificationReport, ConfusionMatrixDisplay, DetCurveDisplay, MetricsDisplay,
    PrecisionRecallDisplay, RocCurveDisplay,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests;

#[allow(non_snake_case)]
#[cfg(test)]
mod comparative_tests;

/// Common error type for metrics
#[derive(thiserror::Error, Debug)]
pub enum MetricsError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,

        actual: Vec<usize>,
    },
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Empty input")]
    EmptyInput,
    #[error("Invalid labels: {0}")]
    InvalidLabels(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<MetricsError> for sklears_core::error::SklearsError {
    fn from(err: MetricsError) -> Self {
        sklears_core::error::SklearsError::InvalidInput(err.to_string())
    }
}

/// Type alias for metrics results
pub type MetricsResult<T> = std::result::Result<T, MetricsError>;
