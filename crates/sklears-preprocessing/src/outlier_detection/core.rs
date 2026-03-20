//! Core types and configurations for outlier detection

use sklears_core::types::Float;

/// Methods for outlier detection (both univariate and multivariate)
#[derive(Debug, Clone, Copy, Default)]
pub enum OutlierDetectionMethod {
    /// Z-score based outlier detection (univariate)
    #[default]
    ZScore,
    /// Modified Z-score using median absolute deviation (univariate)
    ModifiedZScore,
    /// Interquartile Range (IQR) based detection (univariate)
    IQR,
    /// Percentile-based outlier detection (univariate)
    Percentile,
    /// Mahalanobis distance based detection (multivariate)
    MahalanobisDistance,
    /// Isolation Forest for anomaly detection (multivariate)
    IsolationForest,
    /// Local Outlier Factor (multivariate)
    LocalOutlierFactor,
    /// One-Class SVM for novelty detection (multivariate)
    OneClassSVM,
    /// Ensemble of multiple outlier detection methods
    Ensemble,
}

/// Configuration for outlier detector (supports both univariate and multivariate methods)
#[derive(Debug, Clone)]
pub struct OutlierDetectorConfig {
    /// Method for outlier detection
    pub method: OutlierDetectionMethod,
    /// Threshold for Z-score and Modified Z-score methods (default: 3.0)
    pub threshold: Float,
    /// Multiplier for IQR method (default: 1.5, commonly used value)
    pub iqr_multiplier: Float,
    /// Lower percentile for percentile method (default: 5.0)
    pub lower_percentile: Float,
    /// Upper percentile for percentile method (default: 95.0)
    pub upper_percentile: Float,
    /// Chi-squared threshold for Mahalanobis distance (default: based on degrees of freedom)
    pub mahalanobis_threshold: Option<Float>,
    /// Confidence level for automatic Mahalanobis threshold (default: 0.95)
    pub confidence_level: Float,
    /// Number of trees for Isolation Forest (default: 100)
    pub n_estimators: usize,
    /// Subsampling size for Isolation Forest (default: 256)
    pub max_samples: usize,
    /// Number of neighbors for LOF (default: 20)
    pub n_neighbors: usize,
    /// Contamination rate - expected proportion of outliers (default: 0.1)
    pub contamination: Float,
    /// Kernel for One-Class SVM (default: RBF)
    pub svm_kernel: String,
    /// Nu parameter for One-Class SVM (default: 0.05)
    pub nu: Float,
    /// Gamma parameter for RBF kernel (default: scale)
    pub gamma: Float,
    /// Ensemble methods to combine (for Ensemble method)
    pub ensemble_methods: Vec<OutlierDetectionMethod>,
    /// Voting strategy for ensemble: "majority" or "average" (default: "majority")
    pub voting_strategy: String,
}

impl Default for OutlierDetectorConfig {
    fn default() -> Self {
        Self {
            method: OutlierDetectionMethod::ZScore,
            threshold: 3.0,
            iqr_multiplier: 1.5,
            lower_percentile: 5.0,
            upper_percentile: 95.0,
            mahalanobis_threshold: None,
            confidence_level: 0.95,
            n_estimators: 100,
            max_samples: 256,
            n_neighbors: 20,
            contamination: 0.1,
            svm_kernel: "rbf".to_string(),
            nu: 0.05,
            gamma: 1.0, // Will be set to 1/n_features by default
            ensemble_methods: vec![
                OutlierDetectionMethod::ZScore,
                OutlierDetectionMethod::MahalanobisDistance,
            ],
            voting_strategy: "majority".to_string(),
        }
    }
}

/// Statistics computed for outlier detection
#[derive(Debug, Clone)]
pub struct OutlierStatistics {
    pub n_outliers: usize,
    pub outlier_fraction: Float,
    pub feature_outlier_counts: Vec<usize>,
}

/// Parameters for multivariate outlier detection
#[derive(Debug, Clone, Default)]
pub struct MultivariateOutlierParams {
    pub mean: Vec<Float>,
    pub covariance: Vec<Vec<Float>>,
    pub inv_covariance: Vec<Vec<Float>>,
    pub threshold: Float,
}

/// Parameters for outlier detection on a single feature
#[derive(Debug, Clone, Default)]
pub struct FeatureOutlierParams {
    pub mean: Option<Float>,
    pub std: Option<Float>,
    pub median: Option<Float>,
    pub mad: Option<Float>, // Median Absolute Deviation
    pub q1: Option<Float>,  // First quartile
    pub q3: Option<Float>,  // Third quartile
    pub iqr: Option<Float>, // Interquartile Range
    pub lower_bound: Option<Float>,
    pub upper_bound: Option<Float>,
    pub lower_percentile_value: Option<Float>,
    pub upper_percentile_value: Option<Float>,
}

/// Outlier detection results providing detailed information
#[derive(Debug, Clone)]
pub struct OutlierDetectionResult {
    /// Boolean array indicating which samples are outliers
    pub outliers: Vec<bool>,
    /// Outlier scores for each sample (higher = more likely outlier)
    pub scores: Vec<Float>,
    /// Summary statistics
    pub summary: OutlierSummary,
}

/// Summary of outlier detection results
#[derive(Debug, Clone)]
pub struct OutlierSummary {
    /// Total number of samples
    pub n_samples: usize,
    /// Number of outliers detected
    pub n_outliers: usize,
    /// Fraction of samples that are outliers
    pub outlier_fraction: Float,
    /// Method used for detection
    pub method: OutlierDetectionMethod,
}
