//! Conformal prediction methods for uncertainty quantification
//!
//! This module provides conformal prediction methods that create prediction sets
//! with distribution-free validity guarantees. Unlike traditional calibration methods
//! that focus on probability calibration, conformal prediction provides prediction
//! intervals or sets that contain the true outcome with a specified probability.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Configuration for conformal prediction methods
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Miscoverage level (1 - coverage)
    pub alpha: Float,
    /// Whether to use adaptive conformal prediction
    pub adaptive: bool,
    /// Number of calibration samples to use (if None, use all)
    pub n_calibration: Option<usize>,
    /// Random seed for calibration/validation split
    pub random_seed: Option<u64>,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1, // 90% coverage
            adaptive: false,
            n_calibration: None,
            random_seed: Some(42),
        }
    }
}

/// Type of conformal prediction method
#[derive(Debug, Clone)]
pub enum ConformalMethod {
    /// Split conformal prediction (train/calibration split)
    Split,
    /// Cross-conformal prediction (cross-validation)
    CrossConformal { n_folds: usize },
    /// Jackknife+ conformal prediction
    JackknifeePlus,
    /// Adaptive conformal prediction with online update
    Adaptive { gamma: Float },
    /// Conformalized quantile regression
    QuantileRegression { quantiles: Vec<Float> },
}

/// Conformal prediction result
#[derive(Debug, Clone)]
pub struct ConformalPrediction {
    /// Prediction intervals (lower, upper bounds)
    pub intervals: Array2<Float>,
    /// Point predictions
    pub predictions: Array1<Float>,
    /// Conformity scores used
    pub conformity_scores: Array1<Float>,
    /// Coverage achieved on calibration set
    pub empirical_coverage: Float,
    /// Conformal quantile threshold
    pub threshold: Float,
}

/// Trait for conformity score functions
pub trait ConformityScore: Send + Sync + std::fmt::Debug {
    /// Compute conformity scores for predictions and targets
    fn compute(
        &self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Result<Array1<Float>>;

    /// Get prediction intervals given conformity scores and threshold
    fn prediction_intervals(
        &self,
        predictions: &Array1<Float>,
        threshold: Float,
    ) -> Result<Array2<Float>>;

    /// Clone the conformity score function
    fn clone_box(&self) -> Box<dyn ConformityScore>;
}

impl Clone for Box<dyn ConformityScore> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Absolute residual conformity score: |y - ŷ|
#[derive(Debug, Clone)]
pub struct AbsoluteResidualScore;

impl ConformityScore for AbsoluteResidualScore {
    fn compute(
        &self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if predictions.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        let scores = (predictions - targets).mapv(|x| x.abs());
        Ok(scores)
    }

    fn prediction_intervals(
        &self,
        predictions: &Array1<Float>,
        threshold: Float,
    ) -> Result<Array2<Float>> {
        let n = predictions.len();
        let mut intervals = Array2::zeros((n, 2));

        for (i, &pred) in predictions.iter().enumerate() {
            intervals[[i, 0]] = pred - threshold; // lower bound
            intervals[[i, 1]] = pred + threshold; // upper bound
        }

        Ok(intervals)
    }

    fn clone_box(&self) -> Box<dyn ConformityScore> {
        Box::new(self.clone())
    }
}

/// Normalized absolute residual score: |y - ŷ| / σ(x)
#[derive(Debug, Clone)]
pub struct NormalizedResidualScore {
    /// Estimated standard deviations for normalization
    pub sigmas: Array1<Float>,
}

impl NormalizedResidualScore {
    /// Create a new normalized residual score with estimated sigmas
    pub fn new(sigmas: Array1<Float>) -> Self {
        Self { sigmas }
    }
}

impl ConformityScore for NormalizedResidualScore {
    fn compute(
        &self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if predictions.len() != targets.len() || predictions.len() != self.sigmas.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions, targets, and sigmas must have same length".to_string(),
            ));
        }

        let scores = (predictions - targets).mapv(|x| x.abs()) / &self.sigmas;
        Ok(scores)
    }

    fn prediction_intervals(
        &self,
        predictions: &Array1<Float>,
        threshold: Float,
    ) -> Result<Array2<Float>> {
        let n = predictions.len();
        if n != self.sigmas.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and sigmas must have same length".to_string(),
            ));
        }

        let mut intervals = Array2::zeros((n, 2));

        for (i, (&pred, &sigma)) in predictions.iter().zip(self.sigmas.iter()).enumerate() {
            let margin = threshold * sigma;
            intervals[[i, 0]] = pred - margin; // lower bound
            intervals[[i, 1]] = pred + margin; // upper bound
        }

        Ok(intervals)
    }

    fn clone_box(&self) -> Box<dyn ConformityScore> {
        Box::new(self.clone())
    }
}

/// Quantile-based conformity score for conformalized quantile regression
#[derive(Debug, Clone)]
pub struct QuantileScore {
    /// Lower quantile predictions
    pub lower_quantiles: Array1<Float>,
    /// Upper quantile predictions  
    pub upper_quantiles: Array1<Float>,
}

impl QuantileScore {
    /// Create a new quantile score
    pub fn new(lower_quantiles: Array1<Float>, upper_quantiles: Array1<Float>) -> Self {
        Self {
            lower_quantiles,
            upper_quantiles,
        }
    }
}

impl ConformityScore for QuantileScore {
    fn compute(
        &self,
        _predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        if targets.len() != self.lower_quantiles.len()
            || targets.len() != self.upper_quantiles.len()
        {
            return Err(SklearsError::InvalidInput(
                "Targets and quantiles must have same length".to_string(),
            ));
        }

        // Conformity score is max of distances to quantile bounds
        let scores = targets
            .iter()
            .zip(self.lower_quantiles.iter())
            .zip(self.upper_quantiles.iter())
            .map(|((&y, &q_low), &q_high)| Float::max(q_low - y, y - q_high))
            .collect::<Vec<_>>();

        Ok(Array1::from(scores))
    }

    fn prediction_intervals(
        &self,
        _predictions: &Array1<Float>,
        threshold: Float,
    ) -> Result<Array2<Float>> {
        let n = self.lower_quantiles.len();
        let mut intervals = Array2::zeros((n, 2));

        for (i, (&q_low, &q_high)) in self
            .lower_quantiles
            .iter()
            .zip(self.upper_quantiles.iter())
            .enumerate()
        {
            intervals[[i, 0]] = q_low - threshold; // adjusted lower bound
            intervals[[i, 1]] = q_high + threshold; // adjusted upper bound
        }

        Ok(intervals)
    }

    fn clone_box(&self) -> Box<dyn ConformityScore> {
        Box::new(self.clone())
    }
}

/// Main conformal predictor
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    config: ConformalConfig,
    method: ConformalMethod,
    conformity_score: Box<dyn ConformityScore>,
    /// Fitted conformal quantile
    threshold_: Option<Float>,
    /// Calibration conformity scores
    calibration_scores_: Option<Array1<Float>>,
}

impl ConformalPredictor {
    /// Create a new conformal predictor
    pub fn new(method: ConformalMethod, conformity_score: Box<dyn ConformityScore>) -> Self {
        Self {
            config: ConformalConfig::default(),
            method,
            conformity_score,
            threshold_: None,
            calibration_scores_: None,
        }
    }

    /// Set the miscoverage level (1 - coverage)
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to use adaptive conformal prediction
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.config.adaptive = adaptive;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Fit the conformal predictor using calibration data
    pub fn fit(
        &mut self,
        calibration_predictions: &Array1<Float>,
        calibration_targets: &Array1<Float>,
    ) -> Result<()> {
        // Compute conformity scores on calibration set
        let conformity_scores = self
            .conformity_score
            .compute(calibration_predictions, calibration_targets)?;

        // Compute conformal quantile
        let threshold = self.compute_threshold(&conformity_scores)?;

        self.threshold_ = Some(threshold);
        self.calibration_scores_ = Some(conformity_scores);

        Ok(())
    }

    /// Make conformal predictions
    pub fn predict(&self, predictions: &Array1<Float>) -> Result<ConformalPrediction> {
        let threshold = self.threshold_.ok_or_else(|| SklearsError::InvalidData {
            reason: "Conformal predictor not fitted".to_string(),
        })?;

        let intervals = self
            .conformity_score
            .prediction_intervals(predictions, threshold)?;

        // Compute empirical coverage on calibration set
        let empirical_coverage = if let Some(ref cal_scores) = self.calibration_scores_ {
            let coverage = cal_scores
                .iter()
                .filter(|&&score| score <= threshold)
                .count() as Float
                / cal_scores.len() as Float;
            coverage
        } else {
            1.0 - self.config.alpha // target coverage
        };

        Ok(ConformalPrediction {
            intervals,
            predictions: predictions.clone(),
            conformity_scores: self
                .calibration_scores_
                .as_ref()
                .unwrap_or(&Array1::zeros(0))
                .clone(),
            empirical_coverage,
            threshold,
        })
    }

    /// Compute the conformal threshold (quantile)
    fn compute_threshold(&self, conformity_scores: &Array1<Float>) -> Result<Float> {
        let n = conformity_scores.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "No conformity scores provided".to_string(),
            ));
        }

        // Sort conformity scores
        let mut scores = conformity_scores.to_vec();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute conformal quantile
        let quantile_level = match self.method {
            ConformalMethod::Split => {
                // Standard split conformal: (n+1)(1-α)/n quantile
                ((n as Float + 1.0) * (1.0 - self.config.alpha) / n as Float).ceil() / n as Float
            }
            ConformalMethod::CrossConformal { .. } => {
                // Cross-conformal: standard (1-α) quantile
                1.0 - self.config.alpha
            }
            ConformalMethod::JackknifeePlus => {
                // Jackknife+: (1-α)(1+1/n) quantile
                (1.0 - self.config.alpha) * (1.0 + 1.0 / n as Float)
            }
            ConformalMethod::Adaptive { .. } => {
                // Adaptive: standard (1-α) quantile, will be updated online
                1.0 - self.config.alpha
            }
            ConformalMethod::QuantileRegression { .. } => {
                // CQR: standard (1-α) quantile
                1.0 - self.config.alpha
            }
        };

        let index = (quantile_level * (n - 1) as Float).floor() as usize;
        let index = index.min(n - 1);

        Ok(scores[index])
    }

    /// Update conformal predictor for adaptive methods
    pub fn update(
        &mut self,
        new_predictions: &Array1<Float>,
        new_targets: &Array1<Float>,
    ) -> Result<()> {
        if !self.config.adaptive {
            return Ok(()); // No update for non-adaptive methods
        }

        if let ConformalMethod::Adaptive { gamma } = self.method {
            // Compute new conformity scores
            let new_scores = self
                .conformity_score
                .compute(new_predictions, new_targets)?;

            // Update threshold using exponential moving average
            if let Some(current_threshold) = self.threshold_ {
                let new_threshold = new_scores.mean().unwrap_or(current_threshold);
                self.threshold_ = Some(gamma * new_threshold + (1.0 - gamma) * current_threshold);
            } else {
                self.threshold_ = Some(new_scores.mean().unwrap_or(0.0));
            }
        }

        Ok(())
    }
}

/// Split conformal prediction implementation
pub fn split_conformal_prediction(
    train_predictions: &Array1<Float>,
    train_targets: &Array1<Float>,
    test_predictions: &Array1<Float>,
    alpha: Float,
    conformity_score: Box<dyn ConformityScore>,
    split_ratio: Float,
) -> Result<ConformalPrediction> {
    let n_train = train_predictions.len();
    let n_calibration = ((1.0 - split_ratio) * n_train as Float) as usize;

    if n_calibration == 0 {
        return Err(SklearsError::InvalidInput(
            "Calibration set is empty".to_string(),
        ));
    }

    // Split into training and calibration sets
    let calibration_start = n_train - n_calibration;
    let cal_predictions = train_predictions.slice(s![calibration_start..]).to_owned();
    let cal_targets = train_targets.slice(s![calibration_start..]).to_owned();

    // Create and fit conformal predictor
    let mut predictor =
        ConformalPredictor::new(ConformalMethod::Split, conformity_score).alpha(alpha);

    predictor.fit(&cal_predictions, &cal_targets)?;
    predictor.predict(test_predictions)
}

/// Cross-conformal prediction using K-fold cross-validation
pub fn cross_conformal_prediction(
    predictions: &Array1<Float>,
    targets: &Array1<Float>,
    test_predictions: &Array1<Float>,
    alpha: Float,
    conformity_score: Box<dyn ConformityScore>,
    n_folds: usize,
) -> Result<ConformalPrediction> {
    let n = predictions.len();
    if n_folds > n {
        return Err(SklearsError::InvalidInput(
            "Number of folds cannot exceed number of samples".to_string(),
        ));
    }

    let fold_size = n / n_folds;
    let mut all_conformity_scores = Vec::new();

    // Perform K-fold cross-validation to collect conformity scores
    for fold in 0..n_folds {
        let start = fold * fold_size;
        let end = if fold == n_folds - 1 {
            n
        } else {
            (fold + 1) * fold_size
        };

        // Use fold as calibration set
        let cal_predictions = predictions.slice(s![start..end]).to_owned();
        let cal_targets = targets.slice(s![start..end]).to_owned();

        // Compute conformity scores for this fold
        let scores = conformity_score.compute(&cal_predictions, &cal_targets)?;
        all_conformity_scores.extend(scores.iter());
    }

    let all_scores = Array1::from(all_conformity_scores);

    // Create and fit conformal predictor
    let mut predictor = ConformalPredictor::new(
        ConformalMethod::CrossConformal { n_folds },
        conformity_score,
    )
    .alpha(alpha);

    // Manually set the calibration scores and compute threshold
    predictor.calibration_scores_ = Some(all_scores.clone());
    let threshold = predictor.compute_threshold(&all_scores)?;
    predictor.threshold_ = Some(threshold);

    predictor.predict(test_predictions)
}

/// Jackknife+ conformal prediction
pub fn jackknife_plus_prediction(
    predictions: &Array1<Float>,
    targets: &Array1<Float>,
    test_predictions: &Array1<Float>,
    alpha: Float,
    conformity_score: Box<dyn ConformityScore>,
) -> Result<ConformalPrediction> {
    let n = predictions.len();
    let mut leave_one_out_scores = Vec::new();

    // Perform leave-one-out to collect conformity scores
    for i in 0..n {
        // Create LOO predictions and targets (exclude sample i)
        let mut loo_pred = Vec::new();
        let mut loo_targ = Vec::new();

        for j in 0..n {
            if i != j {
                loo_pred.push(predictions[j]);
                loo_targ.push(targets[j]);
            }
        }

        let loo_predictions = Array1::from(loo_pred);
        let loo_targets = Array1::from(loo_targ);

        // Compute conformity score for excluded sample
        let pred_i = Array1::from(vec![predictions[i]]);
        let targ_i = Array1::from(vec![targets[i]]);
        let score = conformity_score.compute(&pred_i, &targ_i)?;
        leave_one_out_scores.extend(score.iter());
    }

    let all_scores = Array1::from(leave_one_out_scores);

    // Create and fit conformal predictor
    let mut predictor =
        ConformalPredictor::new(ConformalMethod::JackknifeePlus, conformity_score).alpha(alpha);

    predictor.calibration_scores_ = Some(all_scores.clone());
    let threshold = predictor.compute_threshold(&all_scores)?;
    predictor.threshold_ = Some(threshold);

    predictor.predict(test_predictions)
}

/// Conformalized quantile regression
pub fn conformalized_quantile_regression(
    lower_quantile_preds: &Array1<Float>,
    upper_quantile_preds: &Array1<Float>,
    targets: &Array1<Float>,
    test_lower_preds: &Array1<Float>,
    test_upper_preds: &Array1<Float>,
    alpha: Float,
) -> Result<ConformalPrediction> {
    // Create quantile conformity score
    let conformity_score = Box::new(QuantileScore::new(
        lower_quantile_preds.clone(),
        upper_quantile_preds.clone(),
    ));

    // Dummy predictions (not used for quantile score)
    let dummy_preds = Array1::zeros(targets.len());
    let dummy_test_preds = Array1::zeros(test_lower_preds.len());

    // Create and fit conformal predictor
    let mut predictor = ConformalPredictor::new(
        ConformalMethod::QuantileRegression {
            quantiles: vec![alpha / 2.0, 1.0 - alpha / 2.0],
        },
        conformity_score,
    )
    .alpha(alpha);

    predictor.fit(&dummy_preds, targets)?;

    // Create test quantile score for prediction
    let test_conformity_score = Box::new(QuantileScore::new(
        test_lower_preds.clone(),
        test_upper_preds.clone(),
    ));
    predictor.conformity_score = test_conformity_score;

    predictor.predict(&dummy_test_preds)
}

/// Compute marginal coverage empirically
pub fn compute_marginal_coverage(
    intervals: &Array2<Float>,
    targets: &Array1<Float>,
) -> Result<Float> {
    if intervals.nrows() != targets.len() {
        return Err(SklearsError::InvalidInput(
            "Number of intervals must match number of targets".to_string(),
        ));
    }

    let mut covered = 0;
    for (i, &target) in targets.iter().enumerate() {
        let lower = intervals[[i, 0]];
        let upper = intervals[[i, 1]];
        if target >= lower && target <= upper {
            covered += 1;
        }
    }

    Ok(covered as Float / targets.len() as Float)
}

/// Compute average interval width
pub fn compute_average_width(intervals: &Array2<Float>) -> Float {
    let widths: Vec<Float> = (0..intervals.nrows())
        .map(|i| intervals[[i, 1]] - intervals[[i, 0]])
        .collect();

    widths.iter().sum::<Float>() / widths.len() as Float
}

/// Compute conditional coverage by grouping
pub fn compute_conditional_coverage(
    intervals: &Array2<Float>,
    targets: &Array1<Float>,
    groups: &Array1<i32>,
) -> Result<HashMap<i32, Float>> {
    if intervals.nrows() != targets.len() || targets.len() != groups.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have same length".to_string(),
        ));
    }

    let mut group_coverage = HashMap::new();
    let mut group_counts = HashMap::new();
    let mut group_covered = HashMap::new();

    for (i, (&target, &group)) in targets.iter().zip(groups.iter()).enumerate() {
        let lower = intervals[[i, 0]];
        let upper = intervals[[i, 1]];
        let is_covered = target >= lower && target <= upper;

        *group_counts.entry(group).or_insert(0) += 1;
        if is_covered {
            *group_covered.entry(group).or_insert(0) += 1;
        }
    }

    for (&group, &count) in group_counts.iter() {
        let covered = group_covered.get(&group).unwrap_or(&0);
        group_coverage.insert(group, *covered as Float / count as Float);
    }

    Ok(group_coverage)
}

// Import necessary for slicing
use scirs2_core::ndarray::s;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_absolute_residual_score() {
        let predictions = array![1.0, 2.0, 3.0, 4.0];
        let targets = array![1.1, 1.8, 3.2, 3.9];

        let score = AbsoluteResidualScore;
        let conformity_scores = score.compute(&predictions, &targets).unwrap();

        assert_eq!(conformity_scores.len(), 4);
        assert!((conformity_scores[0] - 0.1).abs() < 1e-10);
        assert!((conformity_scores[1] - 0.2).abs() < 1e-10);
        assert!((conformity_scores[2] - 0.2).abs() < 1e-10);
        assert!((conformity_scores[3] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_residual_score() {
        let predictions = array![1.0, 2.0, 3.0, 4.0];
        let targets = array![1.1, 1.8, 3.2, 3.9];
        let sigmas = array![0.1, 0.2, 0.1, 0.1];

        let score = NormalizedResidualScore::new(sigmas);
        let conformity_scores = score.compute(&predictions, &targets).unwrap();

        assert_eq!(conformity_scores.len(), 4);
        assert!((conformity_scores[0] - 1.0).abs() < 1e-10); // 0.1 / 0.1
        assert!((conformity_scores[1] - 1.0).abs() < 1e-10); // 0.2 / 0.2
        assert!((conformity_scores[2] - 2.0).abs() < 1e-10); // 0.2 / 0.1
        assert!((conformity_scores[3] - 1.0).abs() < 1e-10); // 0.1 / 0.1
    }

    #[test]
    fn test_conformal_predictor_basic() {
        let cal_predictions = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let cal_targets = array![1.1, 1.9, 3.1, 4.1, 4.9];
        let test_predictions = array![2.5, 3.5];

        let conformity_score = Box::new(AbsoluteResidualScore);
        let mut predictor =
            ConformalPredictor::new(ConformalMethod::Split, conformity_score).alpha(0.2); // 80% coverage

        predictor.fit(&cal_predictions, &cal_targets).unwrap();
        let result = predictor.predict(&test_predictions).unwrap();

        assert_eq!(result.intervals.nrows(), 2);
        assert_eq!(result.intervals.ncols(), 2);
        assert_eq!(result.predictions.len(), 2);
        assert!(result.empirical_coverage >= 0.0 && result.empirical_coverage <= 1.0);
        assert!(result.threshold > 0.0);
    }

    #[test]
    fn test_split_conformal_prediction() {
        let train_predictions = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let train_targets = array![1.1, 1.9, 3.1, 4.1, 4.9, 6.1];
        let test_predictions = array![2.5, 3.5];

        let conformity_score = Box::new(AbsoluteResidualScore);
        let result = split_conformal_prediction(
            &train_predictions,
            &train_targets,
            &test_predictions,
            0.1, // 90% coverage
            conformity_score,
            0.5, // 50% train, 50% calibration
        )
        .unwrap();

        assert_eq!(result.intervals.nrows(), 2);
        assert_eq!(result.intervals.ncols(), 2);

        // Check that intervals contain some reasonable bounds
        for i in 0..result.intervals.nrows() {
            let lower = result.intervals[[i, 0]];
            let upper = result.intervals[[i, 1]];
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_cross_conformal_prediction() {
        let predictions = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets = array![1.1, 1.9, 3.1, 4.1, 4.9, 6.1];
        let test_predictions = array![2.5, 3.5];

        let conformity_score = Box::new(AbsoluteResidualScore);
        let result = cross_conformal_prediction(
            &predictions,
            &targets,
            &test_predictions,
            0.1, // 90% coverage
            conformity_score,
            3, // 3-fold CV
        )
        .unwrap();

        assert_eq!(result.intervals.nrows(), 2);
        assert_eq!(result.intervals.ncols(), 2);

        // Check that intervals are reasonable
        for i in 0..result.intervals.nrows() {
            let lower = result.intervals[[i, 0]];
            let upper = result.intervals[[i, 1]];
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_marginal_coverage() {
        let intervals = array![[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]];
        let targets = array![2.0, 3.0, 4.0];

        let coverage = compute_marginal_coverage(&intervals, &targets).unwrap();
        assert!((coverage - 1.0).abs() < 1e-10); // All targets should be covered

        let targets_outside = array![0.5, 1.5, 6.0];
        let coverage_outside = compute_marginal_coverage(&intervals, &targets_outside).unwrap();
        assert!(coverage_outside < 1.0); // Not all targets should be covered
    }

    #[test]
    fn test_average_width() {
        let intervals = array![[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]];
        let avg_width = compute_average_width(&intervals);
        assert!((avg_width - 2.0).abs() < 1e-10); // All intervals have width 2.0
    }

    #[test]
    fn test_conditional_coverage() {
        let intervals = array![[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]];
        let targets = array![2.0, 3.0, 4.0, 7.0];
        let groups = array![0, 0, 1, 1];

        let coverage = compute_conditional_coverage(&intervals, &targets, &groups).unwrap();

        assert_eq!(coverage.len(), 2);
        assert!((coverage[&0] - 1.0).abs() < 1e-10); // Group 0: both covered
        assert!((coverage[&1] - 0.5).abs() < 1e-10); // Group 1: one covered, one not
    }
}
