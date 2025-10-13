use scirs2_core::ndarray::Array1;
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;
use std::time::Instant;

/// Validation timing utilities
#[derive(Debug, Clone)]
pub struct ValidationTimer {
    start_time: Instant,
    stage_times: HashMap<String, Float>,
}

impl ValidationTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            stage_times: HashMap::new(),
        }
    }

    pub fn start_stage(&mut self, stage: &str) {
        self.stage_times
            .insert(format!("{}_start", stage), self.elapsed());
    }

    pub fn end_stage(&mut self, stage: &str) -> Float {
        let elapsed = self.elapsed();
        let start_key = format!("{}_start", stage);
        let start_time = self.stage_times.get(&start_key).copied().unwrap_or(0.0);
        let duration = elapsed - start_time;
        self.stage_times.insert(stage.to_string(), duration);
        duration
    }

    pub fn elapsed(&self) -> Float {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn get_stage_time(&self, stage: &str) -> Option<Float> {
        self.stage_times.get(stage).copied()
    }

    pub fn get_all_stage_times(&self) -> &HashMap<String, Float> {
        &self.stage_times
    }
}

impl Default for ValidationTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage tracking for validation
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    peak_memory: usize,
    current_memory: usize,
    memory_samples: Vec<usize>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            current_memory: 0,
            memory_samples: Vec::new(),
        }
    }

    pub fn record_usage(&mut self, usage: usize) {
        self.current_memory = usage;
        self.peak_memory = self.peak_memory.max(usage);
        self.memory_samples.push(usage);
    }

    pub fn get_peak_memory(&self) -> usize {
        self.peak_memory
    }

    pub fn get_average_memory(&self) -> Float {
        if self.memory_samples.is_empty() {
            0.0
        } else {
            self.memory_samples.iter().sum::<usize>() as Float / self.memory_samples.len() as Float
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Progress tracker for validation operations
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    total_steps: usize,
    completed_steps: usize,
    current_stage: String,
    start_time: Instant,
}

impl ProgressTracker {
    pub fn new(total_steps: usize) -> Self {
        Self {
            total_steps,
            completed_steps: 0,
            current_stage: String::new(),
            start_time: Instant::now(),
        }
    }

    pub fn update(&mut self, completed: usize, stage: &str) {
        self.completed_steps = completed;
        self.current_stage = stage.to_string();
    }

    pub fn increment(&mut self, stage: &str) {
        self.completed_steps += 1;
        self.current_stage = stage.to_string();
    }

    pub fn progress_percentage(&self) -> Float {
        if self.total_steps == 0 {
            100.0
        } else {
            (self.completed_steps as Float / self.total_steps as Float) * 100.0
        }
    }

    pub fn estimated_time_remaining(&self) -> Float {
        if self.completed_steps == 0 {
            return Float::INFINITY;
        }

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.completed_steps as Float / elapsed;
        let remaining_steps = self.total_steps - self.completed_steps;

        if rate > 0.0 {
            remaining_steps as Float / rate
        } else {
            Float::INFINITY
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed_steps >= self.total_steps
    }
}

/// Validation result formatter
pub struct ResultFormatter;

impl ResultFormatter {
    /// Format validation results as a table string
    pub fn format_results_table(results: &[ValidationResult]) -> String {
        if results.is_empty() {
            return "No results to display".to_string();
        }

        let mut table = String::new();

        // Header
        table.push_str(&format!(
            "{:<20} {:<12} {:<12} {:<10}\n",
            "Strategy", "Mean Score", "Std Score", "CV Folds"
        ));
        table.push_str(&"-".repeat(60));
        table.push('\n');

        // Data rows
        for result in results {
            table.push_str(&format!(
                "{:<20} {:<12.6} {:<12.6} {:<10}\n",
                result.strategy,
                result.mean_score,
                result.std_score,
                result.fold_scores.len()
            ));
        }

        table
    }

    /// Format statistical summary
    pub fn format_statistical_summary(summary: &StatisticalSummary) -> String {
        format!(
            "Statistical Summary:\n\
             Mean:     {:.6}\n\
             Std:      {:.6}\n\
             Min:      {:.6}\n\
             Max:      {:.6}\n\
             Median:   {:.6}\n\
             Q25:      {:.6}\n\
             Q75:      {:.6}\n\
             Skew:     {:.6}\n\
             Kurt:     {:.6}",
            summary.mean,
            summary.std,
            summary.min,
            summary.max,
            summary.median,
            summary.q25,
            summary.q75,
            summary.skewness,
            summary.kurtosis
        )
    }

    /// Format timing information
    pub fn format_timing_info(timer: &ValidationTimer) -> String {
        let mut info = String::new();
        info.push_str(&format!("Total elapsed time: {:.3}s\n", timer.elapsed()));

        for (stage, duration) in timer.get_all_stage_times() {
            if !stage.ends_with("_start") {
                info.push_str(&format!("  {}: {:.3}s\n", stage, duration));
            }
        }

        info
    }
}

/// Validation result for formatting
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// strategy
    pub strategy: String,
    /// mean_score
    pub mean_score: Float,
    /// std_score
    pub std_score: Float,
    /// fold_scores
    pub fold_scores: Vec<Float>,
}

/// Statistical summary for formatting
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

/// Data validation utilities
pub struct DataValidator;

impl DataValidator {
    /// Check if data contains NaN or infinite values
    pub fn check_finite(data: &Array1<Float>) -> Result<()> {
        for &value in data.iter() {
            if value.is_nan() {
                return Err(SklearsError::InvalidInput(
                    "Data contains NaN values".to_string(),
                ));
            }
            if value.is_infinite() {
                return Err(SklearsError::InvalidInput(
                    "Data contains infinite values".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Check if arrays have consistent lengths
    pub fn check_consistent_length(arrays: &[&Array1<Float>]) -> Result<()> {
        if arrays.is_empty() {
            return Ok(());
        }

        let expected_length = arrays[0].len();
        for (i, array) in arrays.iter().enumerate().skip(1) {
            if array.len() != expected_length {
                return Err(SklearsError::InvalidInput(format!(
                    "Array {} has length {} but expected {}",
                    i,
                    array.len(),
                    expected_length
                )));
            }
        }
        Ok(())
    }

    /// Check if target values are valid for classification
    pub fn check_classification_targets(y: &Array1<Int>) -> Result<()> {
        if y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Target array is empty".to_string(),
            ));
        }

        // Check for negative values
        for &value in y.iter() {
            if value < 0 {
                return Err(SklearsError::InvalidInput(
                    "Classification targets must be non-negative".to_string(),
                ));
            }
        }

        // Check number of unique classes
        let mut unique_values: Vec<Int> = y.iter().copied().collect();
        unique_values.sort_unstable();
        unique_values.dedup();

        if unique_values.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Classification requires at least 2 classes".to_string(),
            ));
        }

        if unique_values.len() > 1000 {
            return Err(SklearsError::InvalidInput(
                "Too many classes for classification (>1000)".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if sample sizes are sufficient for validation
    pub fn check_sample_sizes(
        n_samples: usize,
        n_folds: usize,
        min_samples_per_fold: usize,
    ) -> Result<()> {
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        if n_folds > n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of folds cannot exceed number of samples".to_string(),
            ));
        }

        let samples_per_fold = n_samples / n_folds;
        if samples_per_fold < min_samples_per_fold {
            return Err(SklearsError::InvalidInput(format!(
                "Each fold would have {} samples, but minimum {} required",
                samples_per_fold, min_samples_per_fold
            )));
        }

        Ok(())
    }
}

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate cross-validation configuration
    pub fn validate_cv_config(
        n_samples: usize,
        n_folds: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<()> {
        if n_folds < 2 {
            return Err(SklearsError::InvalidInput(
                "Number of folds must be at least 2".to_string(),
            ));
        }

        if n_folds > n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of folds cannot exceed number of samples".to_string(),
            ));
        }

        if shuffle && random_state.is_none() {
            // Warning: shuffling without fixed random state affects reproducibility
        }

        Ok(())
    }

    /// Validate bootstrap configuration
    pub fn validate_bootstrap_config(
        n_samples: usize,
        n_bootstrap: usize,
        confidence_level: Float,
    ) -> Result<()> {
        if n_bootstrap < 10 {
            return Err(SklearsError::InvalidInput(
                "Number of bootstrap samples should be at least 10".to_string(),
            ));
        }

        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        if n_samples < 5 {
            return Err(SklearsError::InvalidInput(
                "Bootstrap validation requires at least 5 samples".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate scoring metric
    pub fn validate_scoring_metric(metric: &str, is_classification: bool) -> Result<()> {
        let classification_metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"];
        let regression_metrics = [
            "neg_mean_squared_error",
            "mse",
            "neg_mean_absolute_error",
            "mae",
            "r2",
            "neg_root_mean_squared_error",
            "rmse",
            "explained_variance",
        ];

        let metric_lower = metric.to_lowercase();

        if is_classification {
            if !classification_metrics.contains(&metric_lower.as_str()) {
                return Err(SklearsError::InvalidInput(format!(
                    "'{}' is not a valid classification metric",
                    metric
                )));
            }
        } else if !regression_metrics.contains(&metric_lower.as_str()) {
            return Err(SklearsError::InvalidInput(format!(
                "'{}' is not a valid regression metric",
                metric
            )));
        }

        Ok(())
    }
}

/// Score utilities
pub struct ScoreUtils;

impl ScoreUtils {
    /// Calculate confidence interval for scores
    pub fn confidence_interval(scores: &[Float], confidence_level: Float) -> (Float, Float) {
        if scores.is_empty() {
            return (0.0, 0.0);
        }

        let n = scores.len() as Float;
        let mean = scores.iter().sum::<Float>() / n;
        let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n;
        let std_error = (variance / n).sqrt();

        let t_value = match confidence_level {
            x if x >= 0.99 => 2.576,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            _ => 1.96,
        };

        let margin = t_value * std_error;
        (mean - margin, mean + margin)
    }

    /// Calculate effect size (Cohen's d) between two score arrays
    pub fn cohens_d(scores1: &[Float], scores2: &[Float]) -> Float {
        if scores1.is_empty() || scores2.is_empty() {
            return 0.0;
        }

        let n1 = scores1.len() as Float;
        let n2 = scores2.len() as Float;
        let mean1 = scores1.iter().sum::<Float>() / n1;
        let mean2 = scores2.iter().sum::<Float>() / n2;

        let var1 = scores1.iter().map(|&x| (x - mean1).powi(2)).sum::<Float>() / (n1 - 1.0);
        let var2 = scores2.iter().map(|&x| (x - mean2).powi(2)).sum::<Float>() / (n2 - 1.0);
        let pooled_std = ((var1 + var2) / 2.0).sqrt();

        if pooled_std > 0.0 {
            (mean1 - mean2) / pooled_std
        } else {
            0.0
        }
    }

    /// Check if score difference is practically significant
    pub fn is_practically_significant(
        scores1: &[Float],
        scores2: &[Float],
        min_effect_size: Float,
    ) -> bool {
        let effect_size = Self::cohens_d(scores1, scores2).abs();
        effect_size >= min_effect_size
    }

    /// Calculate percentile of scores
    pub fn percentile(scores: &[Float], percentile: Float) -> Float {
        if scores.is_empty() {
            return 0.0;
        }

        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (percentile / 100.0 * (sorted_scores.len() - 1) as Float) as usize;
        sorted_scores[idx.min(sorted_scores.len() - 1)]
    }
}

/// Validate cross-validation parameters
pub fn validate_cv_params(n_samples: usize, cv: usize) -> Result<()> {
    if cv == 0 {
        return Err(SklearsError::InvalidParameter {
            name: "cv".to_string(),
            reason: "must be > 0".to_string(),
        });
    }
    if cv > n_samples {
        return Err(SklearsError::InvalidParameter {
            name: "cv".to_string(),
            reason: "cannot be larger than number of samples".to_string(),
        });
    }
    Ok(())
}

/// Calculate classification score (accuracy)
pub fn calculate_classification_score(y_true: &Array1<Int>, y_pred: &Array1<Int>) -> Float {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return 0.0;
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| if t == p { 1.0 } else { 0.0 })
        .sum::<Float>();

    correct / y_true.len() as Float
}

/// Calculate regression score (R² coefficient of determination)
pub fn calculate_regression_score(y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Float {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return 0.0;
    }

    let y_mean = y_true.mean().unwrap_or(0.0);

    let ss_tot: Float = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let ss_res: Float = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum();

    if ss_tot == 0.0 {
        return 1.0; // Perfect score when all y_true values are identical
    }

    1.0 - (ss_res / ss_tot)
}
