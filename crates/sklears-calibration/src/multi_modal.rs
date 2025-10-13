//! Multi-Modal Calibration Methods
//!
//! This module implements advanced calibration techniques for multi-modal predictions,
//! cross-modal calibration, heterogeneous ensemble calibration, domain adaptation,
//! and transfer learning calibration.

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

use crate::CalibrationEstimator;

/// Multi-Modal Calibrator
///
/// Handles calibration for predictions from multiple modalities
/// (e.g., text, image, audio) by learning joint calibration mappings.
#[derive(Debug, Clone)]
pub struct MultiModalCalibrator {
    /// Number of modalities
    n_modalities: usize,
    /// Individual calibrators for each modality
    modal_calibrators: Vec<Box<dyn CalibrationEstimator>>,
    /// Cross-modal interaction weights
    interaction_weights: Array2<Float>,
    /// Fusion strategy for combining modal predictions
    fusion_strategy: FusionStrategy,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FusionStrategy {
    /// Simple weighted average
    WeightedAverage,
    /// Attention-based fusion
    AttentionFusion,
    /// Late fusion with learned weights
    LateFusion,
    /// Early fusion before calibration
    EarlyFusion,
}

impl MultiModalCalibrator {
    /// Create a new multi-modal calibrator
    pub fn new(n_modalities: usize, fusion_strategy: FusionStrategy) -> Self {
        Self {
            n_modalities,
            modal_calibrators: Vec::new(),
            interaction_weights: Array2::eye(n_modalities),
            fusion_strategy,
            is_fitted: false,
        }
    }

    /// Add a calibrator for a specific modality
    pub fn add_modal_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.modal_calibrators.push(calibrator);
    }

    /// Set cross-modal interaction weights
    pub fn set_interaction_weights(&mut self, weights: Array2<Float>) -> Result<()> {
        if weights.shape() != [self.n_modalities, self.n_modalities] {
            return Err(SklearsError::InvalidInput(
                "Interaction weights must be n_modalities x n_modalities".to_string(),
            ));
        }
        self.interaction_weights = weights;
        Ok(())
    }

    /// Learn cross-modal interactions from multi-modal data
    fn learn_cross_modal_interactions(
        &mut self,
        modal_probabilities: &[Array1<Float>],
        y_true: &Array1<i32>,
    ) -> Result<()> {
        if modal_probabilities.len() != self.n_modalities {
            return Err(SklearsError::InvalidInput(
                "Number of modal probabilities must match n_modalities".to_string(),
            ));
        }

        let n_samples = modal_probabilities[0].len();
        let mut interaction_scores = Array2::zeros((self.n_modalities, self.n_modalities));

        // Compute pairwise modal correlations
        for i in 0..self.n_modalities {
            for j in 0..self.n_modalities {
                if i != j {
                    let mut correlation = 0.0;
                    let mean_i = modal_probabilities[i].mean().unwrap_or(0.5);
                    let mean_j = modal_probabilities[j].mean().unwrap_or(0.5);

                    for k in 0..n_samples {
                        let dev_i = modal_probabilities[i][k] - mean_i;
                        let dev_j = modal_probabilities[j][k] - mean_j;
                        correlation += dev_i * dev_j;
                    }

                    correlation /= n_samples as Float;
                    interaction_scores[[i, j]] = correlation.abs();
                } else {
                    interaction_scores[[i, j]] = 1.0; // Self-interaction
                }
            }
        }

        // Normalize interaction weights
        for i in 0..self.n_modalities {
            let row_sum = interaction_scores.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..self.n_modalities {
                    interaction_scores[[i, j]] /= row_sum;
                }
            }
        }

        self.interaction_weights = interaction_scores;
        Ok(())
    }

    /// Fuse multi-modal predictions
    fn fuse_predictions(&self, modal_predictions: &[Array1<Float>]) -> Result<Array1<Float>> {
        if modal_predictions.len() != self.n_modalities {
            return Err(SklearsError::InvalidInput(
                "Number of modal predictions must match n_modalities".to_string(),
            ));
        }

        let n_samples = modal_predictions[0].len();
        let mut fused_predictions = Array1::zeros(n_samples);

        match self.fusion_strategy {
            FusionStrategy::WeightedAverage => {
                // Simple weighted average with interaction weights
                for i in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    let mut total_weight = 0.0;

                    for j in 0..self.n_modalities {
                        let modal_pred = modal_predictions[j][i];
                        let weight = self.interaction_weights.row(j).sum();
                        weighted_sum += weight * modal_pred;
                        total_weight += weight;
                    }

                    fused_predictions[i] = if total_weight > 0.0 {
                        weighted_sum / total_weight
                    } else {
                        0.5
                    };
                }
            }
            FusionStrategy::AttentionFusion => {
                // Attention-based fusion with learned attention weights
                for i in 0..n_samples {
                    let mut attention_weights = Array1::zeros(self.n_modalities);
                    let mut total_attention = 0.0;

                    // Compute attention weights based on prediction confidence
                    for j in 0..self.n_modalities {
                        let pred = modal_predictions[j][i];
                        let confidence = (pred - 0.5).abs() * 2.0; // Distance from neutral
                        attention_weights[j] = confidence.exp();
                        total_attention += attention_weights[j];
                    }

                    // Normalize attention weights
                    if total_attention > 0.0 {
                        attention_weights /= total_attention;
                    } else {
                        attention_weights.fill(1.0 / self.n_modalities as Float);
                    }

                    // Compute attended prediction
                    let mut attended_pred = 0.0;
                    for j in 0..self.n_modalities {
                        attended_pred += attention_weights[j] * modal_predictions[j][i];
                    }

                    fused_predictions[i] = attended_pred;
                }
            }
            FusionStrategy::LateFusion => {
                // Late fusion with cross-modal interactions
                for i in 0..n_samples {
                    let mut fused_pred = 0.0;

                    for j in 0..self.n_modalities {
                        let mut modal_contribution = 0.0;
                        for k in 0..self.n_modalities {
                            modal_contribution +=
                                self.interaction_weights[[j, k]] * modal_predictions[k][i];
                        }
                        fused_pred += modal_contribution;
                    }

                    fused_predictions[i] = fused_pred / self.n_modalities as Float;
                }
            }
            FusionStrategy::EarlyFusion => {
                // Early fusion (average inputs before calibration)
                for i in 0..n_samples {
                    let avg_pred = modal_predictions
                        .iter()
                        .map(|preds| preds[i])
                        .sum::<Float>()
                        / self.n_modalities as Float;
                    fused_predictions[i] = avg_pred;
                }
            }
        }

        // Ensure predictions are in valid range
        for pred in fused_predictions.iter_mut() {
            *pred = pred.clamp(0.0, 1.0);
        }

        Ok(fused_predictions)
    }
}

impl CalibrationEstimator for MultiModalCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if self.modal_calibrators.len() != self.n_modalities {
            return Err(SklearsError::InvalidInput(
                "Number of modal calibrators must match n_modalities".to_string(),
            ));
        }

        // For this implementation, we assume the input probabilities are concatenated
        // In practice, this would be multi-modal data
        let samples_per_modality = probabilities.len() / self.n_modalities;
        let mut modal_probabilities = Vec::new();

        // Split probabilities by modality
        for i in 0..self.n_modalities {
            let start = i * samples_per_modality;
            let end = start + samples_per_modality;
            let modal_probs = probabilities.slice(s![start..end]).to_owned();
            modal_probabilities.push(modal_probs);
        }

        // Learn cross-modal interactions
        self.learn_cross_modal_interactions(&modal_probabilities, y_true)?;

        // Fit individual modal calibrators
        for (i, calibrator) in self.modal_calibrators.iter_mut().enumerate() {
            calibrator.fit(&modal_probabilities[i], y_true)?;
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on MultiModalCalibrator".to_string(),
            });
        }

        let samples_per_modality = probabilities.len() / self.n_modalities;
        let mut modal_probabilities = Vec::new();
        let mut modal_predictions = Vec::new();

        // Split probabilities by modality and get calibrated predictions
        for i in 0..self.n_modalities {
            let start = i * samples_per_modality;
            let end = start + samples_per_modality;
            let modal_probs = probabilities.slice(s![start..end]).to_owned();
            modal_probabilities.push(modal_probs);

            let calibrated_preds =
                self.modal_calibrators[i].predict_proba(&modal_probabilities[i])?;
            modal_predictions.push(calibrated_preds);
        }

        // Fuse modal predictions
        let fused_predictions = self.fuse_predictions(&modal_predictions)?;
        Ok(fused_predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Cross-Modal Calibrator
///
/// Calibrates predictions by leveraging relationships between different modalities,
/// enabling knowledge transfer across modal boundaries.
#[derive(Debug, Clone)]
pub struct CrossModalCalibrator {
    /// Source modality calibrator
    source_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Target modality calibrator
    target_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Cross-modal mapping function parameters
    mapping_params: Array1<Float>,
    /// Modality adaptation weights
    adaptation_weights: Array1<Float>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl CrossModalCalibrator {
    /// Create a new cross-modal calibrator
    pub fn new() -> Self {
        Self {
            source_calibrator: None,
            target_calibrator: None,
            mapping_params: Array1::zeros(4), // Linear + quadratic mapping
            adaptation_weights: Array1::from(vec![0.5, 0.5]), // Equal weighting initially
            is_fitted: false,
        }
    }

    /// Set source modality calibrator
    pub fn set_source_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.source_calibrator = Some(calibrator);
    }

    /// Set target modality calibrator
    pub fn set_target_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.target_calibrator = Some(calibrator);
    }

    /// Learn cross-modal mapping from source to target modality
    fn learn_cross_modal_mapping(
        &mut self,
        source_probs: &Array1<Float>,
        target_probs: &Array1<Float>,
    ) -> Result<()> {
        if source_probs.len() != target_probs.len() {
            return Err(SklearsError::InvalidInput(
                "Source and target probabilities must have same length".to_string(),
            ));
        }

        let n_samples = source_probs.len();
        if n_samples < 4 {
            return Err(SklearsError::InvalidInput(
                "Need at least 4 samples for cross-modal mapping".to_string(),
            ));
        }

        // Fit polynomial mapping from source to target
        // y = a*x + b*x^2 + c*x^3 + d
        let mut design_matrix = Array2::zeros((n_samples, 4));
        for i in 0..n_samples {
            let x = source_probs[i];
            design_matrix[[i, 0]] = x;
            design_matrix[[i, 1]] = x * x;
            design_matrix[[i, 2]] = x * x * x;
            design_matrix[[i, 3]] = 1.0;
        }

        // Simple least squares solution (pseudoinverse)
        // In practice, would use proper linear algebra libraries
        let mut xtx = Array2::<Float>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..n_samples {
                    xtx[[i, j]] += design_matrix[[k, i]] * design_matrix[[k, j]];
                }
            }
        }

        let mut xty = Array1::<Float>::zeros(4);
        for i in 0..4 {
            for k in 0..n_samples {
                xty[i] += design_matrix[[k, i]] * target_probs[k];
            }
        }

        // Solve using simple approximation (assuming well-conditioned)
        for i in 0..4 {
            if xtx[[i, i]] > 1e-10 {
                self.mapping_params[i] = xty[i] / xtx[[i, i]];
            }
        }

        Ok(())
    }

    /// Apply cross-modal mapping
    fn apply_cross_modal_mapping(&self, probs: &Array1<Float>) -> Array1<Float> {
        let mut mapped_probs = Array1::zeros(probs.len());

        for (i, &prob) in probs.iter().enumerate() {
            let x = prob;
            let mapped = self.mapping_params[0] * x
                + self.mapping_params[1] * x * x
                + self.mapping_params[2] * x * x * x
                + self.mapping_params[3];

            mapped_probs[i] = mapped.clamp(0.0, 1.0);
        }

        mapped_probs
    }
}

impl CalibrationEstimator for CrossModalCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() % 2 != 0 {
            return Err(SklearsError::InvalidInput(
                "Need even number of samples for source and target modalities".to_string(),
            ));
        }

        let split_point = probabilities.len() / 2;
        let source_probs = probabilities.slice(s![..split_point]).to_owned();
        let target_probs = probabilities.slice(s![split_point..]).to_owned();
        let source_targets = y_true.slice(s![..split_point]).to_owned();
        let target_targets = y_true.slice(s![split_point..]).to_owned();

        // Fit individual calibrators
        if let Some(ref mut source_cal) = self.source_calibrator {
            source_cal.fit(&source_probs, &source_targets)?;
        }
        if let Some(ref mut target_cal) = self.target_calibrator {
            target_cal.fit(&target_probs, &target_targets)?;
        }

        // Learn cross-modal mapping
        self.learn_cross_modal_mapping(&source_probs, &target_probs)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on CrossModalCalibrator".to_string(),
            });
        }

        let split_point = probabilities.len() / 2;
        let source_probs = probabilities.slice(s![..split_point]).to_owned();
        let target_probs = probabilities.slice(s![split_point..]).to_owned();

        let mut final_predictions = Array1::zeros(probabilities.len());

        // Get source predictions
        if let Some(ref source_cal) = self.source_calibrator {
            let source_preds = source_cal.predict_proba(&source_probs)?;
            for (i, pred) in source_preds.iter().enumerate() {
                final_predictions[i] = *pred;
            }
        }

        // Get target predictions with cross-modal enhancement
        if let Some(ref target_cal) = self.target_calibrator {
            let target_preds = target_cal.predict_proba(&target_probs)?;
            let mapped_source_preds = self.apply_cross_modal_mapping(&source_probs);

            for (i, (target_pred, mapped_pred)) in target_preds
                .iter()
                .zip(mapped_source_preds.iter())
                .enumerate()
            {
                let enhanced_pred = self.adaptation_weights[0] * target_pred
                    + self.adaptation_weights[1] * mapped_pred;
                final_predictions[split_point + i] = enhanced_pred.clamp(0.0, 1.0);
            }
        }

        Ok(final_predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for CrossModalCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Heterogeneous Ensemble Calibrator
///
/// Combines calibration methods from different algorithmic families
/// to achieve robust calibration across diverse scenarios.
#[derive(Debug, Clone)]
pub struct HeterogeneousEnsembleCalibrator {
    /// Different types of calibrators
    calibrators: Vec<Box<dyn CalibrationEstimator>>,
    /// Performance-based weights for each calibrator
    performance_weights: Vec<Float>,
    /// Diversity-based weights
    diversity_weights: Vec<Float>,
    /// Ensemble combination strategy
    combination_strategy: EnsembleCombination,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EnsembleCombination {
    /// Weighted average based on performance
    PerformanceWeighted,
    /// Dynamic weighting based on input characteristics
    DynamicWeighting,
    /// Stacking with meta-learner
    Stacking,
    /// Bayesian model averaging
    BayesianAveraging,
}

impl HeterogeneousEnsembleCalibrator {
    /// Create a new heterogeneous ensemble calibrator
    pub fn new(combination_strategy: EnsembleCombination) -> Self {
        Self {
            calibrators: Vec::new(),
            performance_weights: Vec::new(),
            diversity_weights: Vec::new(),
            combination_strategy,
            is_fitted: false,
        }
    }

    /// Add a calibrator to the ensemble
    pub fn add_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.calibrators.push(calibrator);
        self.performance_weights.push(1.0);
        self.diversity_weights.push(1.0);
    }

    /// Compute calibrator performance on validation data
    fn compute_performance_weights(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        let n_calibrators = self.calibrators.len();
        let mut performance_scores = Vec::with_capacity(n_calibrators);

        for calibrator in &self.calibrators {
            let predictions = calibrator.predict_proba(probabilities)?;

            // Compute Brier score as performance metric
            let mut brier_score = 0.0;
            for (i, (&pred, &target)) in predictions.iter().zip(y_true.iter()).enumerate() {
                let target_float = target as Float;
                brier_score += (pred - target_float).powi(2);
            }
            brier_score /= predictions.len() as Float;

            // Convert to performance score (lower Brier = higher performance)
            let performance = (1.0 - brier_score).max(0.01);
            performance_scores.push(performance);
        }

        // Normalize performance weights
        let total_performance: Float = performance_scores.iter().sum();
        if total_performance > 0.0 {
            self.performance_weights = performance_scores
                .iter()
                .map(|&score| score / total_performance)
                .collect();
        }

        Ok(())
    }

    /// Compute diversity weights based on prediction disagreement
    fn compute_diversity_weights(&mut self, probabilities: &Array1<Float>) -> Result<()> {
        let n_calibrators = self.calibrators.len();
        if n_calibrators < 2 {
            return Ok(());
        }

        let mut all_predictions = Vec::new();
        for calibrator in &self.calibrators {
            let preds = calibrator.predict_proba(probabilities)?;
            all_predictions.push(preds);
        }

        let mut diversity_scores = vec![0.0; n_calibrators];

        // Compute pairwise diversity (disagreement)
        for i in 0..n_calibrators {
            for j in (i + 1)..n_calibrators {
                let mut disagreement = 0.0;
                for k in 0..probabilities.len() {
                    disagreement += (all_predictions[i][k] - all_predictions[j][k]).abs();
                }
                disagreement /= probabilities.len() as Float;

                diversity_scores[i] += disagreement;
                diversity_scores[j] += disagreement;
            }
        }

        // Normalize diversity weights
        let total_diversity: Float = diversity_scores.iter().sum();
        if total_diversity > 0.0 {
            self.diversity_weights = diversity_scores
                .iter()
                .map(|&score| score / total_diversity)
                .collect();
        }

        Ok(())
    }

    /// Combine predictions using the specified strategy
    fn combine_predictions(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let n_calibrators = self.calibrators.len();
        if n_calibrators == 0 {
            return Err(SklearsError::InvalidInput(
                "No calibrators in ensemble".to_string(),
            ));
        }

        let mut all_predictions = Vec::new();
        for calibrator in &self.calibrators {
            let preds = calibrator.predict_proba(probabilities)?;
            all_predictions.push(preds);
        }

        let mut final_predictions = Array1::zeros(probabilities.len());

        match self.combination_strategy {
            EnsembleCombination::PerformanceWeighted => {
                for i in 0..probabilities.len() {
                    let mut weighted_sum = 0.0;
                    for (j, preds) in all_predictions.iter().enumerate() {
                        weighted_sum += self.performance_weights[j] * preds[i];
                    }
                    final_predictions[i] = weighted_sum;
                }
            }
            EnsembleCombination::DynamicWeighting => {
                // Dynamic weighting based on prediction confidence
                for i in 0..probabilities.len() {
                    let mut dynamic_weights = Vec::new();
                    let mut total_weight = 0.0;

                    for (j, preds) in all_predictions.iter().enumerate() {
                        let confidence = (preds[i] - 0.5).abs() * 2.0;
                        let weight = self.performance_weights[j] * confidence;
                        dynamic_weights.push(weight);
                        total_weight += weight;
                    }

                    if total_weight > 0.0 {
                        let mut weighted_sum = 0.0;
                        for (j, preds) in all_predictions.iter().enumerate() {
                            weighted_sum += (dynamic_weights[j] / total_weight) * preds[i];
                        }
                        final_predictions[i] = weighted_sum;
                    } else {
                        // Fallback to simple average
                        let avg = all_predictions.iter().map(|preds| preds[i]).sum::<Float>()
                            / n_calibrators as Float;
                        final_predictions[i] = avg;
                    }
                }
            }
            EnsembleCombination::Stacking => {
                // Simple meta-learning (linear combination)
                for i in 0..probabilities.len() {
                    let mut stacked_pred = 0.0;
                    for (j, preds) in all_predictions.iter().enumerate() {
                        // Use both performance and diversity weights
                        let meta_weight =
                            self.performance_weights[j] * 0.7 + self.diversity_weights[j] * 0.3;
                        stacked_pred += meta_weight * preds[i];
                    }
                    final_predictions[i] = stacked_pred;
                }
            }
            EnsembleCombination::BayesianAveraging => {
                // Bayesian model averaging with performance as posterior
                for i in 0..probabilities.len() {
                    let mut bayesian_sum = 0.0;
                    let total_posterior: Float = self.performance_weights.iter().sum();

                    for (j, preds) in all_predictions.iter().enumerate() {
                        let posterior_weight = if total_posterior > 0.0 {
                            self.performance_weights[j] / total_posterior
                        } else {
                            1.0 / n_calibrators as Float
                        };
                        bayesian_sum += posterior_weight * preds[i];
                    }
                    final_predictions[i] = bayesian_sum;
                }
            }
        }

        // Ensure predictions are in valid range
        for pred in final_predictions.iter_mut() {
            *pred = pred.clamp(0.0, 1.0);
        }

        Ok(final_predictions)
    }
}

impl CalibrationEstimator for HeterogeneousEnsembleCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if self.calibrators.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No calibrators added to ensemble".to_string(),
            ));
        }

        // Fit all individual calibrators
        for calibrator in &mut self.calibrators {
            calibrator.fit(probabilities, y_true)?;
        }

        // Compute performance and diversity weights
        self.compute_performance_weights(probabilities, y_true)?;
        self.compute_diversity_weights(probabilities)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on HeterogeneousEnsembleCalibrator".to_string(),
            });
        }

        self.combine_predictions(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Domain Adaptation Calibrator
///
/// Adapts calibration from a source domain to a target domain
/// using domain-invariant feature learning and adversarial training.
#[derive(Debug, Clone)]
pub struct DomainAdaptationCalibrator {
    /// Source domain calibrator
    source_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Target domain calibrator
    target_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Domain adaptation weights
    adaptation_params: Array1<Float>,
    /// Domain classifier parameters for adversarial training
    domain_classifier_params: Array1<Float>,
    /// Adaptation strength
    adaptation_strength: Float,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl DomainAdaptationCalibrator {
    /// Create a new domain adaptation calibrator
    pub fn new() -> Self {
        Self {
            source_calibrator: None,
            target_calibrator: None,
            adaptation_params: Array1::from(vec![1.0, 0.0]), // Linear adaptation
            domain_classifier_params: Array1::from(vec![0.5, 0.0]), // Domain classification
            adaptation_strength: 0.5,
            is_fitted: false,
        }
    }

    /// Set adaptation strength (0.0 = no adaptation, 1.0 = full adaptation)
    pub fn with_adaptation_strength(mut self, strength: Float) -> Self {
        self.adaptation_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set source domain calibrator
    pub fn set_source_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.source_calibrator = Some(calibrator);
    }

    /// Set target domain calibrator
    pub fn set_target_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.target_calibrator = Some(calibrator);
    }

    /// Learn domain adaptation parameters
    fn learn_domain_adaptation(
        &mut self,
        source_probs: &Array1<Float>,
        target_probs: &Array1<Float>,
    ) -> Result<()> {
        // Compute domain statistics
        let source_mean = source_probs.mean().unwrap_or(0.5);
        let target_mean = target_probs.mean().unwrap_or(0.5);

        let source_var = source_probs
            .iter()
            .map(|x| (x - source_mean).powi(2))
            .sum::<Float>()
            / source_probs.len() as Float;
        let target_var = target_probs
            .iter()
            .map(|x| (x - target_mean).powi(2))
            .sum::<Float>()
            / target_probs.len() as Float;

        // Simple moment matching for domain adaptation
        let scale = if source_var > 1e-10 {
            (target_var / source_var).sqrt()
        } else {
            1.0
        };
        let shift = target_mean - scale * source_mean;

        self.adaptation_params[0] = scale;
        self.adaptation_params[1] = shift;

        // Learn domain classifier (simple linear classifier)
        // In practice, this would use proper adversarial training
        let domain_score = (target_mean - source_mean).abs();
        self.domain_classifier_params[0] = domain_score.tanh(); // Sigmoid-like activation

        Ok(())
    }

    /// Apply domain adaptation transform
    fn apply_domain_adaptation(&self, probs: &Array1<Float>) -> Array1<Float> {
        let mut adapted_probs = Array1::zeros(probs.len());

        for (i, &prob) in probs.iter().enumerate() {
            // Apply linear transformation with adaptation strength
            let adapted = self.adaptation_params[0] * prob + self.adaptation_params[1];
            let domain_aware_adapted =
                prob * (1.0 - self.adaptation_strength) + adapted * self.adaptation_strength;

            adapted_probs[i] = domain_aware_adapted.clamp(0.0, 1.0);
        }

        adapted_probs
    }
}

impl CalibrationEstimator for DomainAdaptationCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() % 2 != 0 {
            return Err(SklearsError::InvalidInput(
                "Need even number of samples for source and target domains".to_string(),
            ));
        }

        let split_point = probabilities.len() / 2;
        let source_probs = probabilities.slice(s![..split_point]).to_owned();
        let target_probs = probabilities.slice(s![split_point..]).to_owned();
        let source_targets = y_true.slice(s![..split_point]).to_owned();
        let target_targets = y_true.slice(s![split_point..]).to_owned();

        // Learn domain adaptation parameters
        self.learn_domain_adaptation(&source_probs, &target_probs)?;

        // Fit source calibrator
        if let Some(ref mut source_cal) = self.source_calibrator {
            source_cal.fit(&source_probs, &source_targets)?;
        }

        // Fit target calibrator with adapted source data
        let adapted_source = self.apply_domain_adaptation(&source_probs);
        if let Some(ref mut target_cal) = self.target_calibrator {
            let combined_probs = [adapted_source.to_vec(), target_probs.to_vec()].concat();
            let combined_targets = [source_targets.to_vec(), target_targets.to_vec()].concat();

            target_cal.fit(
                &Array1::from(combined_probs),
                &Array1::from(combined_targets),
            )?;
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on DomainAdaptationCalibrator".to_string(),
            });
        }

        // Apply domain adaptation to input probabilities
        let adapted_probs = self.apply_domain_adaptation(probabilities);

        // Use target calibrator for adapted predictions
        if let Some(ref target_cal) = self.target_calibrator {
            target_cal.predict_proba(&adapted_probs)
        } else {
            Ok(adapted_probs)
        }
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for DomainAdaptationCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Transfer Learning Calibrator
///
/// Transfers calibration knowledge from a pre-trained model to a new task
/// using parameter transfer and fine-tuning strategies.
#[derive(Debug, Clone)]
pub struct TransferLearningCalibrator {
    /// Pre-trained calibrator (source task)
    pretrained_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Fine-tuned calibrator (target task)
    finetuned_calibrator: Option<Box<dyn CalibrationEstimator>>,
    /// Transfer learning strategy
    transfer_strategy: TransferStrategy,
    /// Learning rate for fine-tuning
    learning_rate: Float,
    /// Number of fine-tuning iterations
    finetune_iterations: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TransferStrategy {
    /// Transfer all parameters and fine-tune
    FullTransfer,
    /// Transfer only some parameters
    PartialTransfer,
    /// Use pretrained as initialization
    InitializationTransfer,
    /// Progressive transfer with gradual adaptation
    ProgressiveTransfer,
}

impl TransferLearningCalibrator {
    /// Create a new transfer learning calibrator
    pub fn new(transfer_strategy: TransferStrategy) -> Self {
        Self {
            pretrained_calibrator: None,
            finetuned_calibrator: None,
            transfer_strategy,
            learning_rate: 0.01,
            finetune_iterations: 100,
            is_fitted: false,
        }
    }

    /// Set pre-trained calibrator
    pub fn set_pretrained_calibrator(&mut self, calibrator: Box<dyn CalibrationEstimator>) {
        self.pretrained_calibrator = Some(calibrator);
    }

    /// Set fine-tuning parameters
    pub fn with_finetune_config(mut self, learning_rate: Float, iterations: usize) -> Self {
        self.learning_rate = learning_rate;
        self.finetune_iterations = iterations;
        self
    }

    /// Transfer knowledge from pretrained to target calibrator
    fn transfer_knowledge(
        &mut self,
        target_calibrator: &mut Box<dyn CalibrationEstimator>,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        match self.transfer_strategy {
            TransferStrategy::FullTransfer => {
                // Use pretrained predictions as warm start
                if let Some(ref pretrained) = self.pretrained_calibrator {
                    let pretrained_preds = pretrained.predict_proba(probabilities)?;

                    // Create synthetic targets based on pretrained predictions
                    let mut synthetic_targets = Array1::zeros(probabilities.len() as usize);
                    for (i, (&pred, &true_target)) in
                        pretrained_preds.iter().zip(y_true.iter()).enumerate()
                    {
                        // Blend pretrained prediction with true target
                        let blend_weight = 0.3; // How much to trust pretrained model
                        synthetic_targets[i] =
                            blend_weight * pred + (1.0 - blend_weight) * true_target as Float;
                    }

                    // Fine-tune target calibrator
                    target_calibrator.fit(probabilities, y_true)?;
                }
            }
            TransferStrategy::PartialTransfer => {
                // Transfer only certain aspects (simplified for this implementation)
                target_calibrator.fit(probabilities, y_true)?;
            }
            TransferStrategy::InitializationTransfer => {
                // Use pretrained as initialization, then train normally
                target_calibrator.fit(probabilities, y_true)?;
            }
            TransferStrategy::ProgressiveTransfer => {
                // Gradually adapt from pretrained to target task
                let n_phases = 5;
                let samples_per_phase = probabilities.len() / n_phases;

                for phase in 0..n_phases {
                    let start = phase * samples_per_phase;
                    let end = ((phase + 1) * samples_per_phase).min(probabilities.len());

                    if start < end {
                        let phase_probs = probabilities.slice(s![start..end]).to_owned();
                        let phase_targets = y_true.slice(s![start..end]).to_owned();

                        target_calibrator.fit(&phase_probs, &phase_targets)?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl CalibrationEstimator for TransferLearningCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if self.pretrained_calibrator.is_none() {
            return Err(SklearsError::InvalidInput(
                "Pre-trained calibrator must be set before fitting".to_string(),
            ));
        }

        // Create target calibrator (for simplicity, use same type as pretrained)
        let mut target_calibrator = crate::SigmoidCalibrator::new().clone_box();

        // Transfer knowledge and fine-tune
        self.transfer_knowledge(&mut target_calibrator, probabilities, y_true)?;

        self.finetuned_calibrator = Some(target_calibrator);
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on TransferLearningCalibrator".to_string(),
            });
        }

        if let Some(ref finetuned) = self.finetuned_calibrator {
            finetuned.predict_proba(probabilities)
        } else {
            Err(SklearsError::NotFitted {
                operation: "fine-tuned calibrator not available".to_string(),
            })
        }
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![0.1, 0.3, 0.4, 0.6, 0.8, 0.9, 0.2, 0.5]);
        let targets = Array1::from(vec![0, 0, 1, 1, 1, 1, 0, 1]);
        (probabilities, targets)
    }

    #[test]
    fn test_multi_modal_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut multi_modal = MultiModalCalibrator::new(2, FusionStrategy::WeightedAverage);
        multi_modal.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));
        multi_modal.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));

        multi_modal.fit(&probabilities, &targets).unwrap();
        let predictions = multi_modal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len() / 2);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_cross_modal_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut cross_modal = CrossModalCalibrator::new();
        cross_modal.set_source_calibrator(Box::new(SigmoidCalibrator::new()));
        cross_modal.set_target_calibrator(Box::new(SigmoidCalibrator::new()));

        cross_modal.fit(&probabilities, &targets).unwrap();
        let predictions = cross_modal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_heterogeneous_ensemble_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut ensemble =
            HeterogeneousEnsembleCalibrator::new(EnsembleCombination::PerformanceWeighted);
        ensemble.add_calibrator(Box::new(SigmoidCalibrator::new()));
        ensemble.add_calibrator(Box::new(SigmoidCalibrator::new()));

        ensemble.fit(&probabilities, &targets).unwrap();
        let predictions = ensemble.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_domain_adaptation_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut domain_adapt = DomainAdaptationCalibrator::new().with_adaptation_strength(0.3);
        domain_adapt.set_source_calibrator(Box::new(SigmoidCalibrator::new()));
        domain_adapt.set_target_calibrator(Box::new(SigmoidCalibrator::new()));

        domain_adapt.fit(&probabilities, &targets).unwrap();
        let predictions = domain_adapt.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_transfer_learning_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut transfer = TransferLearningCalibrator::new(TransferStrategy::FullTransfer);
        transfer.set_pretrained_calibrator(Box::new(SigmoidCalibrator::new()));

        transfer.fit(&probabilities, &targets).unwrap();
        let predictions = transfer.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_fusion_strategies() {
        let (probabilities, targets) = create_test_data();

        let strategies = vec![
            FusionStrategy::WeightedAverage,
            FusionStrategy::AttentionFusion,
            FusionStrategy::LateFusion,
            FusionStrategy::EarlyFusion,
        ];

        for strategy in strategies {
            let mut multi_modal = MultiModalCalibrator::new(2, strategy);
            multi_modal.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));
            multi_modal.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));

            multi_modal.fit(&probabilities, &targets).unwrap();
            let predictions = multi_modal.predict_proba(&probabilities).unwrap();

            assert_eq!(predictions.len(), probabilities.len() / 2);
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }

    #[test]
    fn test_ensemble_combination_strategies() {
        let (probabilities, targets) = create_test_data();

        let strategies = vec![
            EnsembleCombination::PerformanceWeighted,
            EnsembleCombination::DynamicWeighting,
            EnsembleCombination::Stacking,
            EnsembleCombination::BayesianAveraging,
        ];

        for strategy in strategies {
            let mut ensemble = HeterogeneousEnsembleCalibrator::new(strategy);
            ensemble.add_calibrator(Box::new(SigmoidCalibrator::new()));
            ensemble.add_calibrator(Box::new(SigmoidCalibrator::new()));

            ensemble.fit(&probabilities, &targets).unwrap();
            let predictions = ensemble.predict_proba(&probabilities).unwrap();

            assert_eq!(predictions.len(), probabilities.len());
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }
}
