//! Domain-specific calibration methods
//!
//! This module implements calibration methods tailored for specific domains and data types,
//! including time series calibration, regression calibration, and ranking calibration.

use scirs2_core::ndarray::{s, Array1};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

use crate::{isotonic::IsotonicCalibrator, CalibrationEstimator, SigmoidCalibrator};

/// Time Series Calibration Method
///
/// Calibration method that accounts for temporal dependencies in time series data.
/// Uses windowed calibration and time-aware cross-validation.
#[derive(Debug, Clone)]
pub struct TimeSeriesCalibrator {
    /// Window size for temporal calibration
    window_size: usize,
    /// Base calibration method to use within each window
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Temporal weights (exponential decay)
    temporal_decay: Float,
    /// Whether to use seasonal adjustment
    seasonal_adjustment: bool,
    /// Seasonal period (if seasonal adjustment is enabled)
    seasonal_period: Option<usize>,
    /// Trained calibrators for each time window
    window_calibrators: Vec<Box<dyn CalibrationEstimator>>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl TimeSeriesCalibrator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            base_calibrator: Box::new(SigmoidCalibrator::new()),
            temporal_decay: 0.95,
            seasonal_adjustment: false,
            seasonal_period: None,
            window_calibrators: Vec::new(),
            is_fitted: false,
        }
    }

    /// Configure temporal decay parameter
    pub fn with_temporal_decay(mut self, decay: Float) -> Self {
        self.temporal_decay = decay;
        self
    }

    /// Enable seasonal adjustment with specified period
    pub fn with_seasonal_adjustment(mut self, period: usize) -> Self {
        self.seasonal_adjustment = true;
        self.seasonal_period = Some(period);
        self
    }

    /// Set base calibration method
    pub fn with_base_calibrator(mut self, calibrator: Box<dyn CalibrationEstimator>) -> Self {
        self.base_calibrator = calibrator;
        self
    }

    /// Compute temporal weights for samples
    fn compute_temporal_weights(&self, n_samples: usize) -> Array1<Float> {
        let mut weights = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let age = (n_samples - 1 - i) as Float;
            weights[i] = self.temporal_decay.powf(age);
        }

        // Normalize weights
        let sum: Float = weights.sum();
        if sum > 0.0 {
            weights /= sum;
        }

        weights
    }

    /// Apply seasonal adjustment if enabled
    fn apply_seasonal_adjustment(&self, probabilities: &Array1<Float>) -> Array1<Float> {
        if !self.seasonal_adjustment || self.seasonal_period.is_none() {
            return probabilities.clone();
        }

        let period = self.seasonal_period.unwrap();
        let n = probabilities.len();
        let mut adjusted = probabilities.clone();

        // Simple seasonal adjustment using moving averages
        for i in 0..n {
            let start = if i >= period { i - period } else { 0 };
            let end = (i + period + 1).min(n);

            let seasonal_mean = probabilities.slice(s![start..end]).mean().unwrap_or(0.5);
            let global_mean = probabilities.mean().unwrap_or(0.5);

            if global_mean > 0.0 {
                let seasonal_factor = seasonal_mean / global_mean;
                adjusted[i] = (probabilities[i] / seasonal_factor).clamp(0.0, 1.0);
            }
        }

        adjusted
    }

    /// Fit time series calibrator using windowed approach
    fn fit_windowed(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        let n = probabilities.len();
        if n < self.window_size {
            return Err(SklearsError::InvalidInput(
                "Time series too short for specified window size".to_string(),
            ));
        }

        // Apply seasonal adjustment if enabled
        let adjusted_probs = self.apply_seasonal_adjustment(probabilities);

        // Create windows and fit calibrators
        let n_windows = (n - self.window_size) / (self.window_size / 2) + 1;
        self.window_calibrators.clear();

        for i in 0..n_windows {
            let start = i * (self.window_size / 2);
            let end = (start + self.window_size).min(n);

            if end - start < 3 {
                break; // Need minimum samples
            }

            let window_probs = adjusted_probs.slice(s![start..end]).to_owned();
            let window_targets = y_true.slice(s![start..end]).to_owned();

            // Compute temporal weights for this window
            let weights = self.compute_temporal_weights(window_probs.len());

            // Fit calibrator for this window
            let mut calibrator = self.base_calibrator.clone_box();
            calibrator.fit(&window_probs, &window_targets)?;

            self.window_calibrators.push(calibrator);
        }

        Ok(())
    }
}

impl CalibrationEstimator for TimeSeriesCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        self.fit_windowed(probabilities, y_true)?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted || self.window_calibrators.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on TimeSeriesCalibrator".to_string(),
            });
        }

        // For prediction, use the most recent calibrator or ensemble of recent calibrators
        let recent_calibrator = self.window_calibrators.last().unwrap();
        let calibrated = recent_calibrator.predict_proba(probabilities)?;

        // Ensure probabilities are in valid range [0, 1]
        let normalized = calibrated.mapv(|p| p.clamp(0.0, 1.0));
        Ok(normalized)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Regression Calibration Method
///
/// Calibration method specifically designed for regression problems,
/// using quantile regression and distributional calibration.
#[derive(Debug, Clone)]
pub struct RegressionCalibrator {
    /// Quantile levels for calibration
    quantile_levels: Vec<Float>,
    /// Calibrators for each quantile
    quantile_calibrators: HashMap<String, Box<dyn CalibrationEstimator>>,
    /// Variance calibrator for distributional calibration
    variance_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Whether to use distributional calibration
    distributional: bool,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl RegressionCalibrator {
    pub fn new() -> Self {
        Self {
            quantile_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
            quantile_calibrators: HashMap::new(),
            variance_calibrator: None,
            distributional: false,
            is_fitted: false,
        }
    }

    /// Set custom quantile levels
    pub fn with_quantiles(mut self, quantiles: Vec<Float>) -> Self {
        self.quantile_levels = quantiles;
        self
    }

    /// Enable distributional calibration (calibrates both mean and variance)
    pub fn with_distributional_calibration(mut self) -> Self {
        self.distributional = true;
        self.variance_calibrator = Some(Box::new(SigmoidCalibrator::new()));
        self
    }

    /// Convert regression residuals to calibration targets
    fn residuals_to_calibration_targets(
        &self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
        quantile: Float,
    ) -> Array1<i32> {
        let residuals = targets - predictions;
        let threshold = self.compute_quantile_threshold(&residuals, quantile);

        residuals.mapv(|r| if r <= threshold { 1 } else { 0 })
    }

    /// Compute quantile threshold from residuals
    fn compute_quantile_threshold(&self, residuals: &Array1<Float>, quantile: Float) -> Float {
        let mut sorted_residuals: Vec<Float> = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (quantile * (sorted_residuals.len() - 1) as Float).round() as usize;
        sorted_residuals[index.min(sorted_residuals.len() - 1)]
    }

    /// Convert regression predictions to probabilities for calibration
    fn predictions_to_probabilities(&self, predictions: &Array1<Float>) -> Array1<Float> {
        // Normalize predictions to [0, 1] range using sigmoid-like transformation
        let mean = predictions.mean().unwrap_or(0.0);
        let std = predictions.std(0.0);

        if std > 0.0 {
            predictions.mapv(|p| {
                let normalized = (p - mean) / std;
                1.0 / (1.0 + (-normalized).exp())
            })
        } else {
            Array1::from(vec![0.5; predictions.len()])
        }
    }
}

impl CalibrationEstimator for RegressionCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // For regression calibration, we expect continuous targets
        // Convert integer targets to float for this implementation
        let targets = y_true.mapv(|y| y as Float);

        // Train quantile calibrators
        for &quantile in &self.quantile_levels {
            let calibration_targets =
                self.residuals_to_calibration_targets(probabilities, &targets, quantile);
            let calibration_probs = self.predictions_to_probabilities(probabilities);

            let calibrator =
                IsotonicCalibrator::new().fit(&calibration_probs, &calibration_targets)?;

            self.quantile_calibrators
                .insert(format!("q{:.2}", quantile), Box::new(calibrator));
        }

        // Train variance calibrator if distributional calibration is enabled
        if self.distributional {
            let residuals = &targets - probabilities;
            let squared_residuals = residuals.mapv(|r| r * r);

            // Convert squared residuals to binary targets (high/low variance)
            let median_var = self.compute_quantile_threshold(&squared_residuals, 0.5);
            let var_targets = squared_residuals.mapv(|r| if r > median_var { 1 } else { 0 });

            let var_probs = self.predictions_to_probabilities(&squared_residuals);

            if let Some(ref mut var_calibrator) = self.variance_calibrator {
                var_calibrator.fit(&var_probs, &var_targets)?;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on RegressionCalibrator".to_string(),
            });
        }

        // For simplicity, return calibrated probabilities using median quantile calibrator
        let result = if let Some(median_calibrator) = self.quantile_calibrators.get("q0.50") {
            let calib_probs = self.predictions_to_probabilities(probabilities);
            median_calibrator.predict_proba(&calib_probs)?
        } else {
            // Fallback to first available quantile calibrator
            if let Some(calibrator) = self.quantile_calibrators.values().next() {
                let calib_probs = self.predictions_to_probabilities(probabilities);
                calibrator.predict_proba(&calib_probs)?
            } else {
                probabilities.clone()
            }
        };

        // Ensure probabilities are in valid range [0, 1]
        let normalized = result.mapv(|p| p.clamp(0.0, 1.0));
        Ok(normalized)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for RegressionCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Ranking Calibration Method
///
/// Calibration method for ranking and learning-to-rank problems,
/// focusing on preserving ranking order while improving probability calibration.
#[derive(Debug, Clone)]
pub struct RankingCalibrator {
    /// Base calibration method
    base_calibrator: Box<dyn CalibrationEstimator>,
    /// Ranking preservation weight
    ranking_weight: Float,
    /// Pairwise ranking constraints
    pairwise_constraints: Vec<(usize, usize)>, // (higher_ranked, lower_ranked) indices
    /// Whether to use listwise calibration
    listwise: bool,
    /// Group size for listwise calibration
    group_size: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl RankingCalibrator {
    pub fn new() -> Self {
        Self {
            base_calibrator: Box::new(IsotonicCalibrator::new()),
            ranking_weight: 0.5,
            pairwise_constraints: Vec::new(),
            listwise: false,
            group_size: 10,
            is_fitted: false,
        }
    }

    /// Set ranking preservation weight (0.0 = only calibration, 1.0 = only ranking)
    pub fn with_ranking_weight(mut self, weight: Float) -> Self {
        self.ranking_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Enable listwise calibration with specified group size
    pub fn with_listwise_calibration(mut self, group_size: usize) -> Self {
        self.listwise = true;
        self.group_size = group_size;
        self
    }

    /// Add pairwise ranking constraint
    pub fn add_pairwise_constraint(&mut self, higher_idx: usize, lower_idx: usize) {
        self.pairwise_constraints.push((higher_idx, lower_idx));
    }

    /// Generate pairwise constraints from relevance scores
    fn generate_pairwise_constraints(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) {
        self.pairwise_constraints.clear();

        for i in 0..probabilities.len() {
            for j in (i + 1)..probabilities.len() {
                // If true relevance differs, add constraint
                if y_true[i] > y_true[j] {
                    self.pairwise_constraints.push((i, j));
                } else if y_true[i] < y_true[j] {
                    self.pairwise_constraints.push((j, i));
                }
            }
        }
    }

    /// Compute ranking loss for current probabilities
    fn compute_ranking_loss(&self, probabilities: &Array1<Float>) -> Float {
        let mut loss = 0.0;
        let mut count = 0;

        for &(higher_idx, lower_idx) in &self.pairwise_constraints {
            if higher_idx < probabilities.len() && lower_idx < probabilities.len() {
                // Ranking is violated if lower ranked item has higher probability
                if probabilities[lower_idx] > probabilities[higher_idx] {
                    loss += probabilities[lower_idx] - probabilities[higher_idx];
                }
                count += 1;
            }
        }

        if count > 0 {
            loss / count as Float
        } else {
            0.0
        }
    }

    /// Apply ranking-aware calibration adjustment
    fn apply_ranking_adjustment(
        &self,
        calibrated_probs: &Array1<Float>,
        original_probs: &Array1<Float>,
    ) -> Array1<Float> {
        let ranking_loss = self.compute_ranking_loss(calibrated_probs);

        if ranking_loss > 0.01 && self.ranking_weight > 0.0 {
            // Blend calibrated probabilities with original to preserve ranking
            let blend_weight = self.ranking_weight * ranking_loss.min(1.0);
            calibrated_probs * (1.0 - blend_weight) + original_probs * blend_weight
        } else {
            calibrated_probs.clone()
        }
    }
}

impl CalibrationEstimator for RankingCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        // Generate pairwise ranking constraints
        self.generate_pairwise_constraints(probabilities, y_true);

        // Train base calibrator
        self.base_calibrator.fit(probabilities, y_true)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on RankingCalibrator".to_string(),
            });
        }

        // Get base calibrated probabilities
        let calibrated_probs = self.base_calibrator.predict_proba(probabilities)?;

        // Apply ranking-aware adjustment
        let adjusted_probs = self.apply_ranking_adjustment(&calibrated_probs, probabilities);

        // Ensure probabilities are in valid range [0, 1]
        let normalized = adjusted_probs.mapv(|p| p.clamp(0.0, 1.0));
        Ok(normalized)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for RankingCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Survival Analysis Calibrator
///
/// Calibration method for survival analysis and time-to-event data,
/// accounting for censoring and hazard functions.
#[derive(Debug, Clone)]
pub struct SurvivalCalibrator {
    /// Time points for survival estimation
    time_points: Vec<Float>,
    /// Calibrators for each time point
    time_calibrators: HashMap<String, Box<dyn CalibrationEstimator>>,
    /// Whether to account for censoring
    handle_censoring: bool,
    /// Censoring indicators (if available)
    censoring_indicators: Option<Array1<i32>>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl SurvivalCalibrator {
    /// Create a new survival calibrator
    pub fn new(time_points: Vec<Float>) -> Self {
        Self {
            time_points,
            time_calibrators: HashMap::new(),
            handle_censoring: false,
            censoring_indicators: None,
            is_fitted: false,
        }
    }

    /// Enable censoring handling
    pub fn with_censoring_handling(mut self) -> Self {
        self.handle_censoring = true;
        self
    }

    /// Set censoring indicators (1 = event observed, 0 = censored)
    pub fn set_censoring_indicators(&mut self, indicators: Array1<i32>) {
        self.censoring_indicators = Some(indicators);
    }

    /// Convert survival probabilities to binary outcomes for each time point
    fn survival_to_binary_outcomes(
        &self,
        survival_times: &Array1<Float>,
        time_point: Float,
    ) -> Array1<i32> {
        survival_times.mapv(|t| if t > time_point { 1 } else { 0 })
    }
}

impl CalibrationEstimator for SurvivalCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Convert integer targets to survival times
        let survival_times = y_true.mapv(|y| y as Float);

        // Train calibrators for each time point
        for &time_point in &self.time_points {
            let binary_outcomes = self.survival_to_binary_outcomes(&survival_times, time_point);

            // Account for censoring if enabled
            let adjusted_probs = if self.handle_censoring && self.censoring_indicators.is_some() {
                let censoring = self.censoring_indicators.as_ref().unwrap();
                // Adjust probabilities based on censoring (simplified approach)
                Array1::from_iter(probabilities.iter().zip(censoring.iter()).map(|(&p, &c)| {
                    if c == 0 {
                        p * 0.5
                    } else {
                        p
                    } // Reduce confidence for censored observations
                }))
            } else {
                probabilities.clone()
            };

            let calibrator = IsotonicCalibrator::new().fit(&adjusted_probs, &binary_outcomes)?;

            self.time_calibrators
                .insert(format!("t{:.2}", time_point), Box::new(calibrator));
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on SurvivalCalibrator".to_string(),
            });
        }

        // For simplicity, return calibrated probabilities for the first time point
        if let Some(calibrator) = self.time_calibrators.values().next() {
            let calibrated = calibrator.predict_proba(probabilities)?;

            // Ensure probabilities are in valid range [0, 1]
            let normalized = calibrated.mapv(|p| p.clamp(0.0, 1.0));
            Ok(normalized)
        } else {
            // Ensure input probabilities are in valid range
            let normalized = probabilities.mapv(|p| p.clamp(0.0, 1.0));
            Ok(normalized)
        }
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Structured Prediction Calibration Method
///
/// Calibration method for structured outputs like sequences, trees, or graphs.
/// Calibrates the confidence in the entire structured prediction by considering
/// dependencies between output components.
#[derive(Debug, Clone)]
pub struct StructuredPredictionCalibrator {
    /// Structure type for the prediction
    structure_type: StructureType,
    /// Component calibrators for individual elements
    component_calibrators: HashMap<String, Box<dyn CalibrationEstimator>>,
    /// Dependency graph between components
    dependency_graph: Option<Vec<(usize, usize)>>,
    /// Joint calibrator for structured dependencies
    joint_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Temperature parameter for structured softmax
    temperature: Float,
    /// Whether to use Markov Random Field (MRF) modeling
    use_mrf: bool,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

/// Types of structures supported by structured prediction calibration
#[derive(Debug, Clone)]
pub enum StructureType {
    Sequence,
    Tree,
    Graph,
    Grid { height: usize, width: usize },
}

impl StructuredPredictionCalibrator {
    /// Create a new structured prediction calibrator
    pub fn new(structure_type: StructureType) -> Self {
        Self {
            structure_type,
            component_calibrators: HashMap::new(),
            dependency_graph: None,
            joint_calibrator: None,
            temperature: 1.0,
            use_mrf: false,
            is_fitted: false,
        }
    }

    /// Set dependency graph between components
    pub fn with_dependencies(mut self, dependencies: Vec<(usize, usize)>) -> Self {
        self.dependency_graph = Some(dependencies);
        self
    }

    /// Enable Markov Random Field modeling for dependencies
    pub fn with_mrf_modeling(mut self) -> Self {
        self.use_mrf = true;
        self
    }

    /// Set temperature parameter for structured softmax
    pub fn with_temperature(mut self, temperature: Float) -> Self {
        self.temperature = temperature;
        self
    }

    /// Decompose structured probabilities into component probabilities
    fn decompose_structured_probabilities(
        &self,
        probabilities: &Array1<Float>,
    ) -> HashMap<String, Array1<Float>> {
        let mut components = HashMap::new();

        // For structured prediction calibration, we create multiple views of the same probability data
        // Each component represents a different aspect or decomposition of the structured prediction
        match &self.structure_type {
            StructureType::Sequence => {
                // For sequences, create overlapping windows of the probability sequence
                let window_size = (probabilities.len() / 2).max(1);

                // First component: first half
                if probabilities.len() >= window_size {
                    let component_probs = probabilities.slice(s![0..window_size]).to_owned();
                    components.insert("seq_first".to_string(), component_probs);
                }

                // Second component: second half (overlap allowed)
                let start = probabilities.len().saturating_sub(window_size);
                let component_probs = probabilities.slice(s![start..]).to_owned();
                components.insert("seq_second".to_string(), component_probs);
            }
            StructureType::Tree => {
                // For trees, create different levels of the tree representation
                let half = probabilities.len() / 2;

                // Root level: first half
                if half > 0 {
                    let component_probs = probabilities.slice(s![0..half]).to_owned();
                    components.insert("tree_root".to_string(), component_probs);
                }

                // Leaf level: second half
                let component_probs = probabilities.slice(s![half..]).to_owned();
                components.insert("tree_leaves".to_string(), component_probs);
            }
            StructureType::Graph => {
                // For graphs, create node-based and edge-based views
                let third = probabilities.len() / 3;

                // Node features: first third
                if third > 0 {
                    let component_probs = probabilities.slice(s![0..third]).to_owned();
                    components.insert("graph_nodes".to_string(), component_probs);
                }

                // Edge features: middle third
                if probabilities.len() > third {
                    let start = third;
                    let end = (2 * third).min(probabilities.len());
                    let component_probs = probabilities.slice(s![start..end]).to_owned();
                    components.insert("graph_edges".to_string(), component_probs);
                }

                // Global features: last third
                let start = (2 * third).min(probabilities.len());
                if start < probabilities.len() {
                    let component_probs = probabilities.slice(s![start..]).to_owned();
                    components.insert("graph_global".to_string(), component_probs);
                }
            }
            StructureType::Grid {
                height: _,
                width: _,
            } => {
                // For grids, create spatial decompositions
                let quarter = probabilities.len() / 4;

                // Four spatial regions
                for region in 0..4 {
                    let start = region * quarter;
                    let end = if region == 3 {
                        probabilities.len()
                    } else {
                        (region + 1) * quarter
                    };

                    if start < probabilities.len() {
                        let component_probs = probabilities.slice(s![start..end]).to_owned();
                        components.insert(format!("grid_region_{}", region), component_probs);
                    }
                }
            }
        }

        // If no components were created, use the full probability array as a single component
        if components.is_empty() {
            components.insert("full".to_string(), probabilities.clone());
        }

        components
    }

    /// Compute structured dependencies between components
    fn compute_structured_dependencies(
        &self,
        components: &HashMap<String, Array1<Float>>,
    ) -> Array1<Float> {
        if !self.use_mrf || components.is_empty() {
            return Array1::zeros(components.values().next().map(|v| v.len()).unwrap_or(1));
        }

        // Simple pairwise dependency modeling
        let mut dependency_scores = Vec::new();
        let component_names: Vec<_> = components.keys().collect();

        for i in 0..component_names.len() {
            for j in (i + 1)..component_names.len() {
                let comp1 = &components[component_names[i]];
                let comp2 = &components[component_names[j]];

                // Compute correlation-based dependency score
                let min_len = comp1.len().min(comp2.len());
                if min_len > 0 {
                    let mut correlation = 0.0;
                    for k in 0..min_len {
                        correlation += comp1[k] * comp2[k];
                    }
                    correlation /= min_len as Float;
                    dependency_scores.push(correlation);
                }
            }
        }

        if dependency_scores.is_empty() {
            Array1::zeros(1)
        } else {
            Array1::from(dependency_scores)
        }
    }

    /// Apply structured temperature scaling
    fn apply_structured_temperature(&self, probabilities: &Array1<Float>) -> Array1<Float> {
        if (self.temperature - 1.0).abs() < 1e-6 {
            return probabilities.clone();
        }

        // Apply temperature scaling with structured normalization
        let scaled = probabilities.mapv(|p| {
            let logit = if p > 0.0 && p < 1.0 {
                (p / (1.0 - p)).ln()
            } else {
                0.0
            };
            let scaled_logit = logit / self.temperature;
            1.0 / (1.0 + (-scaled_logit).exp())
        });

        // Normalize to ensure valid probabilities
        let sum = scaled.sum();
        if sum > 0.0 {
            scaled / sum
        } else {
            Array1::from(vec![1.0 / scaled.len() as Float; scaled.len()])
        }
    }
}

impl CalibrationEstimator for StructuredPredictionCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        // Decompose structured probabilities into components
        let components = self.decompose_structured_probabilities(probabilities);

        // Train component calibrators
        for (component_name, component_probs) in &components {
            // Create corresponding targets for this component by taking the appropriate slice
            let component_targets = if component_probs.len() == y_true.len() {
                y_true.clone()
            } else if component_probs.len() < y_true.len() {
                // Take a slice of targets that matches the component length
                y_true.slice(s![0..component_probs.len()]).to_owned()
            } else {
                // If component is larger, repeat targets to match
                let mut extended_targets = Vec::new();
                for i in 0..component_probs.len() {
                    extended_targets.push(y_true[i % y_true.len()]);
                }
                Array1::from(extended_targets)
            };

            let component_calibrator =
                SigmoidCalibrator::new().fit(component_probs, &component_targets)?;
            self.component_calibrators
                .insert(component_name.clone(), Box::new(component_calibrator));
        }

        // Train joint calibrator for structured dependencies if MRF is enabled
        if self.use_mrf {
            let dependency_features = self.compute_structured_dependencies(&components);

            // Create binary targets for dependency calibration that match the feature length
            let dependency_targets = if dependency_features.len() == y_true.len() {
                y_true.clone()
            } else if dependency_features.len() < y_true.len() {
                y_true.slice(s![0..dependency_features.len()]).to_owned()
            } else {
                // If features are longer, repeat targets to match
                let mut extended_targets = Vec::new();
                for i in 0..dependency_features.len() {
                    extended_targets.push(y_true[i % y_true.len()]);
                }
                Array1::from(extended_targets)
            };

            let joint_calibrator =
                IsotonicCalibrator::new().fit(&dependency_features, &dependency_targets)?;
            self.joint_calibrator = Some(Box::new(joint_calibrator));
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            });
        }

        // Decompose structured probabilities
        let components = self.decompose_structured_probabilities(probabilities);

        // Calibrate each component
        let mut calibrated_components = HashMap::new();
        for (component_name, component_probs) in &components {
            if let Some(calibrator) = self.component_calibrators.get(component_name) {
                let calibrated = calibrator.predict_proba(component_probs)?;
                calibrated_components.insert(component_name.clone(), calibrated);
            }
        }

        // Combine calibrated components back into structured prediction
        let mut combined_probabilities = probabilities.clone();

        // Apply component calibrations
        let mut offset = 0;
        for (_, calibrated_comp) in calibrated_components {
            let end_offset = (offset + calibrated_comp.len()).min(combined_probabilities.len());
            for (i, &val) in calibrated_comp.iter().enumerate() {
                if offset + i < end_offset {
                    combined_probabilities[offset + i] = val;
                }
            }
            offset = end_offset;
        }

        // Apply joint calibration if MRF is enabled
        if self.use_mrf && self.joint_calibrator.is_some() {
            let dependency_features = self.compute_structured_dependencies(&components);
            if let Some(ref joint_calibrator) = self.joint_calibrator {
                let joint_calibrated = joint_calibrator.predict_proba(&dependency_features)?;

                // Weight the component and joint calibrations
                let joint_weight = 0.3; // Hyperparameter for balancing component vs joint calibration
                let component_weight = 1.0 - joint_weight;

                // For simplicity, apply joint calibration as a multiplicative factor
                if !joint_calibrated.is_empty() {
                    let joint_factor = joint_calibrated[0];
                    combined_probabilities = combined_probabilities
                        .mapv(|p| component_weight * p + joint_weight * joint_factor * p);
                }
            }
        }

        // Apply structured temperature scaling
        let temperature_calibrated = self.apply_structured_temperature(&combined_probabilities);

        // Ensure probabilities are in valid range [0, 1]
        let final_probabilities = temperature_calibrated.mapv(|p| p.clamp(0.0, 1.0));

        Ok(final_probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn create_time_series_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
            0.65, 0.75, 0.85, 0.9,
        ]);
        let targets = Array1::from(vec![
            0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        ]);
        (probabilities, targets)
    }

    #[test]
    fn test_time_series_calibrator() {
        let (probabilities, targets) = create_time_series_data();

        let mut ts_calibrator = TimeSeriesCalibrator::new(5)
            .with_temporal_decay(0.9)
            .with_seasonal_adjustment(4);

        ts_calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = ts_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_regression_calibrator() {
        let (probabilities, targets) = create_time_series_data();

        let mut reg_calibrator = RegressionCalibrator::new()
            .with_quantiles(vec![0.25, 0.5, 0.75])
            .with_distributional_calibration();

        reg_calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = reg_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_ranking_calibrator() {
        let (probabilities, targets) = create_time_series_data();

        let mut ranking_calibrator = RankingCalibrator::new()
            .with_ranking_weight(0.3)
            .with_listwise_calibration(5);

        ranking_calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = ranking_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_survival_calibrator() {
        let (probabilities, targets) = create_time_series_data();

        let time_points = vec![5.0, 10.0, 15.0];
        let mut survival_calibrator =
            SurvivalCalibrator::new(time_points).with_censoring_handling();

        // Set censoring indicators
        let censoring = Array1::from(vec![1; probabilities.len()]);
        survival_calibrator.set_censoring_indicators(censoring);

        survival_calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = survival_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_structured_prediction_calibrator() {
        let (probabilities, targets) = create_time_series_data();

        // Test sequence structure calibration
        let mut seq_calibrator = StructuredPredictionCalibrator::new(StructureType::Sequence)
            .with_temperature(1.2)
            .with_mrf_modeling();

        seq_calibrator.fit(&probabilities, &targets).unwrap();
        let seq_predictions = seq_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(seq_predictions.len(), probabilities.len());
        assert!(seq_predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        // Test tree structure calibration
        let mut tree_calibrator = StructuredPredictionCalibrator::new(StructureType::Tree)
            .with_dependencies(vec![(0, 1), (1, 2), (2, 3)]);

        tree_calibrator.fit(&probabilities, &targets).unwrap();
        let tree_predictions = tree_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(tree_predictions.len(), probabilities.len());
        assert!(tree_predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        // Test graph structure calibration
        let mut graph_calibrator = StructuredPredictionCalibrator::new(StructureType::Graph)
            .with_mrf_modeling()
            .with_temperature(0.8);

        graph_calibrator.fit(&probabilities, &targets).unwrap();
        let graph_predictions = graph_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(graph_predictions.len(), probabilities.len());
        assert!(graph_predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));

        // Test grid structure calibration
        let mut grid_calibrator = StructuredPredictionCalibrator::new(StructureType::Grid {
            height: 4,
            width: 5,
        });

        grid_calibrator.fit(&probabilities, &targets).unwrap();
        let grid_predictions = grid_calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(grid_predictions.len(), probabilities.len());
        assert!(grid_predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_domain_specific_calibrators_basic_properties() {
        let (probabilities, targets) = create_time_series_data();

        let calibrators: Vec<Box<dyn CalibrationEstimator>> = vec![
            Box::new(TimeSeriesCalibrator::new(3)),
            Box::new(RegressionCalibrator::new()),
            Box::new(RankingCalibrator::new()),
            Box::new(SurvivalCalibrator::new(vec![10.0])),
            Box::new(StructuredPredictionCalibrator::new(StructureType::Sequence)),
        ];

        for mut calibrator in calibrators {
            calibrator.fit(&probabilities, &targets).unwrap();
            let predictions = calibrator.predict_proba(&probabilities).unwrap();

            assert_eq!(predictions.len(), probabilities.len());
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }
}
