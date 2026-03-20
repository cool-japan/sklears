use scirs2_core::ndarray::Array1;
use scirs2_core::random::{thread_rng, RngCore, SeedableRng};
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::Float;
use std::cmp::Ordering;

/// Result of dummy estimator validation
#[derive(Debug, Clone)]
pub struct DummyValidationResult {
    /// Mean score across folds
    pub mean_score: Float,
    /// Standard deviation of scores across folds
    pub std_score: Float,
    /// Individual fold scores
    pub fold_scores: Vec<Float>,
    /// Strategy that was evaluated
    pub strategy: String,
}

impl DummyValidationResult {
    pub fn new(
        mean_score: Float,
        std_score: Float,
        fold_scores: Vec<Float>,
        strategy: String,
    ) -> Self {
        Self {
            mean_score,
            std_score,
            fold_scores,
            strategy,
        }
    }

    pub fn confidence_interval(&self, confidence_level: Float) -> (Float, Float) {
        let n = self.fold_scores.len() as Float;
        let sem = self.std_score / n.sqrt();

        // Approximate t-value for common confidence levels
        let t_value = match confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        };

        let margin = t_value * sem;
        (self.mean_score - margin, self.mean_score + margin)
    }

    pub fn is_significantly_better_than(
        &self,
        other: &DummyValidationResult,
        alpha: Float,
    ) -> bool {
        // Simple t-test approximation
        let pooled_std = ((self.std_score.powi(2) + other.std_score.powi(2)) / 2.0).sqrt();
        let n1 = self.fold_scores.len() as Float;
        let n2 = other.fold_scores.len() as Float;
        let se_diff = pooled_std * ((1.0 / n1) + (1.0 / n2)).sqrt();

        if se_diff == 0.0 {
            return self.mean_score > other.mean_score;
        }

        let t_stat = (self.mean_score - other.mean_score) / se_diff;
        let t_critical = match alpha {
            0.01 => 2.576,
            0.05 => 1.96,
            0.10 => 1.645,
            _ => 1.96,
        };

        t_stat > t_critical
    }
}

/// Configuration for validation procedures
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// cv_folds
    pub cv_folds: usize,
    /// random_state
    pub random_state: Option<u64>,
    /// shuffle
    pub shuffle: bool,
    /// stratify
    pub stratify: bool,
    /// scoring_metric
    pub scoring_metric: String,
    /// bootstrap_samples
    pub bootstrap_samples: usize,
    /// confidence_level
    pub confidence_level: Float,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            cv_folds: 5,
            random_state: None,
            shuffle: true,
            stratify: false,
            scoring_metric: "accuracy".to_string(),
            bootstrap_samples: 1000,
            confidence_level: 0.95,
        }
    }
}

impl ValidationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn stratify(mut self, stratify: bool) -> Self {
        self.stratify = stratify;
        self
    }

    pub fn scoring_metric(mut self, metric: String) -> Self {
        self.scoring_metric = metric;
        self
    }

    pub fn bootstrap_samples(mut self, samples: usize) -> Self {
        self.bootstrap_samples = samples;
        self
    }

    pub fn confidence_level(mut self, level: Float) -> Self {
        self.confidence_level = level;
        self
    }
}

/// Comprehensive validation result with additional statistics
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationResult {
    /// validation_result
    pub validation_result: DummyValidationResult,
    /// fold_details
    pub fold_details: Vec<FoldResult>,
    /// statistical_summary
    pub statistical_summary: StatisticalSummary,
    /// config
    pub config: ValidationConfig,
}

/// Result for individual fold
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// fold_index
    pub fold_index: usize,
    /// train_size
    pub train_size: usize,
    /// test_size
    pub test_size: usize,
    /// score
    pub score: Float,
    /// fit_time
    pub fit_time: Float,
    /// predict_time
    pub predict_time: Float,
}

/// Statistical summary of validation results
#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    /// mean
    pub mean: Float,
    /// std
    pub std: Float,
    /// min
    pub min: Float,
    /// max
    pub max: Float,
    /// median
    pub median: Float,
    /// q25
    pub q25: Float,
    /// q75
    pub q75: Float,
    /// skewness
    pub skewness: Float,
    /// kurtosis
    pub kurtosis: Float,
}

impl StatisticalSummary {
    pub fn from_scores(scores: &[Float]) -> Self {
        if scores.is_empty() {
            return Self::default();
        }

        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let n = scores.len() as Float;
        let mean = scores.iter().sum::<Float>() / n;
        let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n;
        let std = variance.sqrt();
        let min = sorted_scores[0];
        let max = sorted_scores[sorted_scores.len() - 1];

        let median = if sorted_scores.len() % 2 == 0 {
            let mid = sorted_scores.len() / 2;
            (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
        } else {
            sorted_scores[sorted_scores.len() / 2]
        };

        let q25_idx = (sorted_scores.len() as Float * 0.25) as usize;
        let q75_idx = (sorted_scores.len() as Float * 0.75) as usize;
        let q25 = sorted_scores[q25_idx.min(sorted_scores.len() - 1)];
        let q75 = sorted_scores[q75_idx.min(sorted_scores.len() - 1)];

        // Calculate skewness and kurtosis
        let m3 = scores.iter().map(|&x| (x - mean).powi(3)).sum::<Float>() / n;
        let m4 = scores.iter().map(|&x| (x - mean).powi(4)).sum::<Float>() / n;
        let skewness = if std > 0.0 { m3 / std.powi(3) } else { 0.0 };
        let kurtosis = if std > 0.0 {
            m4 / std.powi(4) - 3.0
        } else {
            0.0
        };

        Self {
            mean,
            std,
            min,
            max,
            median,
            q25,
            q75,
            skewness,
            kurtosis,
        }
    }
}

impl Default for StatisticalSummary {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            q25: 0.0,
            q75: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

/// Validation error types specific to dummy estimators
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// InsufficientData
    InsufficientData(String),
    /// InvalidFolds
    InvalidFolds(String),
    /// StratificationError
    StratificationError(String),
    /// ScoringError
    ScoringError(String),
    /// ConfigurationError
    ConfigurationError(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            ValidationError::InvalidFolds(msg) => write!(f, "Invalid fold configuration: {}", msg),
            ValidationError::StratificationError(msg) => write!(f, "Stratification error: {}", msg),
            ValidationError::ScoringError(msg) => write!(f, "Scoring error: {}", msg),
            ValidationError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Utility function to validate common validation parameters
pub fn validate_cv_params(n_samples: usize, cv_folds: usize) -> Result<()> {
    if cv_folds < 2 {
        return Err(SklearsError::InvalidInput(
            "Cross-validation folds must be at least 2".to_string(),
        ));
    }

    if n_samples < cv_folds {
        return Err(SklearsError::InvalidInput(
            "Number of samples must be at least equal to cv folds".to_string(),
        ));
    }

    Ok(())
}

/// Determine if the target variable represents a classification task
pub fn is_classification_task(y: &Array1<Float>) -> bool {
    if y.is_empty() {
        return false;
    }

    // Check for NaN or infinite values
    if y.iter().any(|&val| val.is_nan() || val.is_infinite()) {
        return false;
    }

    // Check if all values are integers
    let all_integers = y.iter().all(|&val| val.fract() == 0.0);
    if !all_integers {
        return false;
    }

    // Check the number of unique values
    let mut unique_values: Vec<Float> = y.iter().copied().collect();
    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    unique_values.dedup();

    // Typically classification tasks have fewer than 50 unique classes
    // and more than 1 class
    unique_values.len() > 1 && unique_values.len() < 50
}

/// Create a random number generator with optional seed
pub fn create_rng(random_state: Option<u64>) -> Box<dyn RngCore> {
    match random_state {
        Some(seed) => Box::new(scirs2_core::random::rngs::StdRng::seed_from_u64(seed)),
        None => Box::new(thread_rng()),
    }
}
