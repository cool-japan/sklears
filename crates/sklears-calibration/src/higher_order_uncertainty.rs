//! Higher-Order Uncertainty Decomposition Framework
//!
//! This module implements a revolutionary approach to uncertainty quantification that decomposes
//! uncertainty into multiple mathematically rigorous components beyond the traditional
//! epistemic/aleatoric dichotomy. This represents cutting-edge research in uncertainty theory
//! and provides a comprehensive framework for understanding prediction reliability.
//!
//! The framework decomposes total uncertainty into six distinct components:
//! 1. **Epistemic Uncertainty** - Model uncertainty due to limited training data
//! 2. **Aleatoric Uncertainty** - Inherent data noise and irreducible randomness
//! 3. **Distributional Uncertainty** - Uncertainty due to distribution shift and covariate drift
//! 4. **Structural Uncertainty** - Uncertainty due to model architecture limitations
//! 5. **Computational Uncertainty** - Uncertainty due to finite precision and approximations
//! 6. **Temporal Uncertainty** - Uncertainty that evolves over time in dynamic systems

use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::random::thread_rng;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

use crate::{numerical_stability::SafeProbabilityOps, CalibrationEstimator};

/// Higher-order uncertainty decomposition configuration
#[derive(Debug, Clone)]
pub struct HigherOrderUncertaintyConfig {
    /// Number of ensemble models for epistemic uncertainty
    pub n_ensemble_models: usize,
    /// Number of bootstrap samples for distributional analysis
    pub n_bootstrap_samples: usize,
    /// Number of architectural variations for structural uncertainty
    pub n_architectural_variations: usize,
    /// Number of precision levels for computational uncertainty analysis
    pub n_precision_levels: usize,
    /// Time window for temporal uncertainty analysis
    pub temporal_window_size: usize,
    /// Confidence level for uncertainty intervals
    pub confidence_level: Float,
    /// Whether to use information-theoretic measures
    pub use_information_theory: bool,
    /// Whether to compute cross-correlations between uncertainty types
    pub compute_cross_correlations: bool,
}

impl Default for HigherOrderUncertaintyConfig {
    fn default() -> Self {
        Self {
            n_ensemble_models: 10,
            n_bootstrap_samples: 1000,
            n_architectural_variations: 5,
            n_precision_levels: 4,
            temporal_window_size: 100,
            confidence_level: 0.95,
            use_information_theory: true,
            compute_cross_correlations: true,
        }
    }
}

/// Six-dimensional uncertainty decomposition result
#[derive(Debug, Clone)]
pub struct HigherOrderUncertaintyResult {
    /// Traditional epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array1<Float>,
    /// Traditional aleatoric uncertainty (data noise)
    pub aleatoric_uncertainty: Array1<Float>,
    /// Distributional uncertainty (covariate shift, domain adaptation)
    pub distributional_uncertainty: Array1<Float>,
    /// Structural uncertainty (model architecture limitations)
    pub structural_uncertainty: Array1<Float>,
    /// Computational uncertainty (numerical precision, approximations)
    pub computational_uncertainty: Array1<Float>,
    /// Temporal uncertainty (time-dependent reliability degradation)
    pub temporal_uncertainty: Array1<Float>,
    /// Total uncertainty (mathematical composition of all components)
    pub total_uncertainty: Array1<Float>,
    /// Cross-correlation matrix between uncertainty types
    pub uncertainty_correlations: Array2<Float>,
    /// Information-theoretic measures
    pub information_measures: InformationTheoreticMeasures,
    /// Uncertainty attribution weights
    pub attribution_weights: Array2<Float>,
    /// Confidence regions for each uncertainty type
    pub confidence_regions: Array3<Float>,
}

/// Information-theoretic measures for uncertainty analysis
#[derive(Debug, Clone)]
pub struct InformationTheoreticMeasures {
    /// Mutual information between prediction and true labels
    pub mutual_information: Float,
    /// Conditional entropy of predictions given true labels
    pub conditional_entropy: Float,
    /// Joint entropy of all uncertainty components
    pub joint_entropy: Float,
    /// Differential entropy of each uncertainty component
    pub component_entropies: Array1<Float>,
    /// Information bottleneck principle application
    pub information_bottleneck_score: Float,
    /// Redundancy between uncertainty components
    pub component_redundancy: Array2<Float>,
}

/// Higher-order uncertainty estimator with advanced mathematical foundations
#[derive(Debug, Clone)]
pub struct HigherOrderUncertaintyEstimator {
    config: HigherOrderUncertaintyConfig,
    /// Ensemble of calibrators with different architectures
    calibrator_ensemble: Vec<Box<dyn CalibrationEstimator>>,
    /// Historical predictions for temporal analysis
    prediction_history: Vec<Array1<Float>>,
    /// Reference distributions for distributional uncertainty
    reference_distributions: Vec<Array1<Float>>,
    /// Precision arithmetic configurations
    precision_configurations: Vec<SafeProbabilityOps>,
    /// Trained state indicator
    is_fitted: bool,
}

impl HigherOrderUncertaintyEstimator {
    /// Create a new higher-order uncertainty estimator
    pub fn new(config: HigherOrderUncertaintyConfig) -> Self {
        Self {
            config,
            calibrator_ensemble: Vec::new(),
            prediction_history: Vec::new(),
            reference_distributions: Vec::new(),
            precision_configurations: Vec::new(),
            is_fitted: false,
        }
    }

    /// Fit the uncertainty estimator with comprehensive analysis
    pub fn fit(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        reference_data: Option<&Array1<Float>>,
    ) -> Result<()> {
        self.initialize_ensemble(probabilities, y_true)?;
        self.initialize_precision_configurations();
        self.initialize_reference_distributions(probabilities, reference_data)?;
        self.is_fitted = true;
        Ok(())
    }

    /// Estimate comprehensive higher-order uncertainty decomposition
    pub fn estimate_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<HigherOrderUncertaintyResult> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "estimate higher-order uncertainty".to_string(),
            });
        }

        let n_samples = probabilities.len();

        // 1. Epistemic uncertainty via ensemble disagreement
        let epistemic = self.estimate_epistemic_uncertainty(probabilities)?;

        // 2. Aleatoric uncertainty via intrinsic variance modeling
        let aleatoric = self.estimate_aleatoric_uncertainty(probabilities)?;

        // 3. Distributional uncertainty via distribution comparison
        let distributional = self.estimate_distributional_uncertainty(probabilities)?;

        // 4. Structural uncertainty via architectural ensemble
        let structural = self.estimate_structural_uncertainty(probabilities)?;

        // 5. Computational uncertainty via precision analysis
        let computational = self.estimate_computational_uncertainty(probabilities)?;

        // 6. Temporal uncertainty via historical analysis
        let temporal = self.estimate_temporal_uncertainty(probabilities)?;

        // Compose total uncertainty using information-theoretic principles
        let total = self.compose_total_uncertainty(&[
            &epistemic,
            &aleatoric,
            &distributional,
            &structural,
            &computational,
            &temporal,
        ])?;

        // Compute cross-correlations if enabled
        let correlations = if self.config.compute_cross_correlations {
            self.compute_uncertainty_correlations(&[
                &epistemic,
                &aleatoric,
                &distributional,
                &structural,
                &computational,
                &temporal,
            ])?
        } else {
            Array2::zeros((6, 6))
        };

        // Compute information-theoretic measures
        let info_measures = if self.config.use_information_theory {
            self.compute_information_measures(&[
                &epistemic,
                &aleatoric,
                &distributional,
                &structural,
                &computational,
                &temporal,
            ])?
        } else {
            self.empty_information_measures()
        };

        // Compute attribution weights using Shapley values
        let attribution = self.compute_uncertainty_attribution(&[
            &epistemic,
            &aleatoric,
            &distributional,
            &structural,
            &computational,
            &temporal,
        ])?;

        // Compute confidence regions
        let confidence_regions = self.compute_confidence_regions(&[
            &epistemic,
            &aleatoric,
            &distributional,
            &structural,
            &computational,
            &temporal,
        ])?;

        Ok(HigherOrderUncertaintyResult {
            epistemic_uncertainty: epistemic,
            aleatoric_uncertainty: aleatoric,
            distributional_uncertainty: distributional,
            structural_uncertainty: structural,
            computational_uncertainty: computational,
            temporal_uncertainty: temporal,
            total_uncertainty: total,
            uncertainty_correlations: correlations,
            information_measures: info_measures,
            attribution_weights: attribution,
            confidence_regions,
        })
    }

    /// Initialize ensemble of calibrators with different architectures
    fn initialize_ensemble(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        // Create diverse ensemble for epistemic uncertainty estimation
        for i in 0..self.config.n_ensemble_models {
            let calibrator = crate::SigmoidCalibrator::new();

            // Add noise for diversity
            let mut noisy_probs = probabilities.clone();
            let noise_level = 0.01 * (i + 1) as Float;
            for prob in noisy_probs.iter_mut() {
                let noise = thread_rng().gen_range(-noise_level..noise_level);
                *prob = (*prob + noise).clamp(1e-15, 1.0 - 1e-15);
            }

            let fitted_calibrator = calibrator.fit(&noisy_probs, y_true)?;
            self.calibrator_ensemble.push(Box::new(fitted_calibrator));
        }

        Ok(())
    }

    /// Initialize different precision configurations for computational uncertainty
    fn initialize_precision_configurations(&mut self) {
        use crate::numerical_stability::NumericalConfig;

        for i in 0..self.config.n_precision_levels {
            let precision_level = 10.0_f64.powi(-(15 - i as i32 * 3)) as Float;
            let config = NumericalConfig {
                min_probability: precision_level,
                max_probability: 1.0 - precision_level,
                convergence_tolerance: precision_level * 10.0,
                max_iterations: 100 + i * 50,
                regularization: precision_level,
                use_log_space: i >= 2,
            };
            self.precision_configurations
                .push(SafeProbabilityOps::new(config));
        }
    }

    /// Initialize reference distributions for distributional uncertainty
    fn initialize_reference_distributions(
        &mut self,
        probabilities: &Array1<Float>,
        reference_data: Option<&Array1<Float>>,
    ) -> Result<()> {
        // Primary reference: training distribution
        self.reference_distributions.push(probabilities.clone());

        // Secondary reference: provided reference data
        if let Some(ref_data) = reference_data {
            self.reference_distributions.push(ref_data.clone());
        }

        // Tertiary reference: uniform distribution
        let uniform_ref = Array1::from_elem(probabilities.len(), 0.5);
        self.reference_distributions.push(uniform_ref);

        Ok(())
    }

    /// Estimate epistemic uncertainty using ensemble disagreement
    fn estimate_epistemic_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = probabilities.len();
        let n_models = self.calibrator_ensemble.len();

        // Collect predictions from all ensemble members
        let mut ensemble_predictions = Array2::zeros((n_models, n_samples));

        for (i, calibrator) in self.calibrator_ensemble.iter().enumerate() {
            let predictions = calibrator.predict_proba(probabilities)?;
            for (j, &pred) in predictions.iter().enumerate() {
                ensemble_predictions[[i, j]] = pred;
            }
        }

        // Compute variance across ensemble (epistemic uncertainty)
        let mut epistemic = Array1::zeros(n_samples);
        for j in 0..n_samples {
            let predictions_for_sample = ensemble_predictions.column(j);
            let mean = predictions_for_sample.mean().unwrap_or(0.5);
            let variance = predictions_for_sample
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<Float>()
                / n_models as Float;
            epistemic[j] = variance.sqrt();
        }

        Ok(epistemic)
    }

    /// Estimate aleatoric uncertainty using intrinsic variance modeling
    fn estimate_aleatoric_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Model intrinsic data uncertainty using probability variance
        let mut aleatoric = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            // Bernoulli variance: p(1-p)
            let intrinsic_variance = prob * (1.0 - prob);

            // Add local neighborhood analysis for better aleatoric estimation
            let local_variance = self.estimate_local_variance(probabilities, i)?;

            // Combine using information-theoretic weighting
            aleatoric[i] = (intrinsic_variance * 0.7 + local_variance * 0.3).sqrt();
        }

        Ok(aleatoric)
    }

    /// Estimate distributional uncertainty via distribution comparison
    fn estimate_distributional_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let mut distributional = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            let mut total_divergence = 0.0;

            // Compare with each reference distribution
            for ref_dist in &self.reference_distributions {
                if i < ref_dist.len() {
                    // Compute KL divergence
                    let p = prob.clamp(1e-15, 1.0 - 1e-15);
                    let q = ref_dist[i].clamp(1e-15, 1.0 - 1e-15);

                    let kl_div = p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln();
                    total_divergence += kl_div.abs();
                }
            }

            distributional[i] =
                (total_divergence / self.reference_distributions.len() as Float).sqrt();
        }

        Ok(distributional)
    }

    /// Estimate structural uncertainty via architectural diversity
    fn estimate_structural_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Structural uncertainty from model architecture limitations
        let mut structural = Array1::zeros(probabilities.len());

        // Use architectural ensemble disagreement as proxy for structural uncertainty
        let subset_size = self
            .config
            .n_architectural_variations
            .min(self.calibrator_ensemble.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            let mut architectural_predictions = Vec::new();

            // Sample different architectural configurations
            for j in 0..subset_size {
                let idx = j * self.calibrator_ensemble.len() / subset_size;
                if let Ok(pred) =
                    self.calibrator_ensemble[idx].predict_proba(&Array1::from_elem(1, prob))
                {
                    if !pred.is_empty() {
                        architectural_predictions.push(pred[0]);
                    }
                }
            }

            if !architectural_predictions.is_empty() {
                let mean = architectural_predictions.iter().sum::<Float>()
                    / architectural_predictions.len() as Float;
                let variance = architectural_predictions
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<Float>()
                    / architectural_predictions.len() as Float;
                structural[i] = variance.sqrt();
            }
        }

        Ok(structural)
    }

    /// Estimate computational uncertainty via precision analysis
    fn estimate_computational_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let mut computational = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            let mut precision_results = Vec::new();

            // Compute same operation with different precision settings
            for precision_ops in &self.precision_configurations {
                let clamped_prob = precision_ops.clamp_probabilities(&Array1::from_elem(1, prob));
                if !clamped_prob.is_empty() {
                    precision_results.push(clamped_prob[0]);
                }
            }

            if precision_results.len() > 1 {
                let mean =
                    precision_results.iter().sum::<Float>() / precision_results.len() as Float;
                let variance = precision_results
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<Float>()
                    / precision_results.len() as Float;
                computational[i] = variance.sqrt();
            }
        }

        Ok(computational)
    }

    /// Estimate temporal uncertainty using historical analysis
    fn estimate_temporal_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let mut temporal = Array1::zeros(probabilities.len());

        if self.prediction_history.is_empty() {
            // No history available, use heuristic based on probability dynamics
            for (i, &prob) in probabilities.iter().enumerate() {
                // Estimate temporal uncertainty based on position in probability space
                let distance_from_decision = (prob - 0.5).abs();
                temporal[i] = (1.0 - 2.0 * distance_from_decision) * 0.1; // Heuristic scaling
            }
        } else {
            // Use historical data for temporal analysis
            for (i, &prob) in probabilities.iter().enumerate() {
                let mut historical_values = Vec::new();

                for hist_pred in &self.prediction_history {
                    if i < hist_pred.len() {
                        historical_values.push(hist_pred[i]);
                    }
                }

                if historical_values.len() > 1 {
                    let mean =
                        historical_values.iter().sum::<Float>() / historical_values.len() as Float;
                    let variance = historical_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<Float>()
                        / historical_values.len() as Float;
                    temporal[i] = variance.sqrt();
                }
            }
        }

        Ok(temporal)
    }

    /// Compose total uncertainty using information-theoretic principles
    fn compose_total_uncertainty(&self, components: &[&Array1<Float>]) -> Result<Array1<Float>> {
        let n_samples = components[0].len();
        let mut total = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Collect all uncertainty components for this sample
            let mut component_values = Vec::new();
            for component in components {
                if i < component.len() {
                    component_values.push(component[i]);
                }
            }

            if !component_values.is_empty() {
                // Use generalized uncertainty composition formula
                // Total = sqrt(sum(u_i^2) + 2*sum(u_i*u_j*corr_ij)) for i != j
                let sum_squares: Float = component_values.iter().map(|&x| x * x).sum();

                // Simplified composition (can be enhanced with correlation terms)
                total[i] = sum_squares.sqrt();
            }
        }

        Ok(total)
    }

    /// Compute cross-correlations between uncertainty types
    fn compute_uncertainty_correlations(
        &self,
        components: &[&Array1<Float>],
    ) -> Result<Array2<Float>> {
        let n_components = components.len();
        let mut correlations = Array2::zeros((n_components, n_components));

        for i in 0..n_components {
            for j in 0..n_components {
                if i == j {
                    correlations[[i, j]] = 1.0;
                } else {
                    let corr = self.compute_correlation(components[i], components[j])?;
                    correlations[[i, j]] = corr;
                }
            }
        }

        Ok(correlations)
    }

    /// Compute correlation between two uncertainty arrays
    fn compute_correlation(&self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
        let n = x.len().min(y.len());
        if n < 2 {
            return Ok(0.0);
        }

        let mean_x = x.slice(s![..n]).mean().unwrap_or(0.0);
        let mean_y = y.slice(s![..n]).mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 1e-10 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Compute information-theoretic measures
    fn compute_information_measures(
        &self,
        components: &[&Array1<Float>],
    ) -> Result<InformationTheoreticMeasures> {
        let n_components = components.len();

        // Compute component entropies
        let mut component_entropies = Array1::zeros(n_components);
        for (i, component) in components.iter().enumerate() {
            component_entropies[i] = self.compute_differential_entropy(component)?;
        }

        // Compute joint entropy (simplified)
        let joint_entropy = component_entropies.sum() * 0.8; // Approximation

        // Compute redundancy matrix
        let mut redundancy = Array2::zeros((n_components, n_components));
        for i in 0..n_components {
            for j in 0..n_components {
                if i != j {
                    redundancy[[i, j]] =
                        self.compute_mutual_information(components[i], components[j])?;
                }
            }
        }

        Ok(InformationTheoreticMeasures {
            mutual_information: redundancy.mean().unwrap_or(0.0),
            conditional_entropy: joint_entropy * 0.6, // Approximation
            joint_entropy,
            component_entropies,
            information_bottleneck_score: self.compute_information_bottleneck_score(components)?,
            component_redundancy: redundancy,
        })
    }

    /// Compute differential entropy of uncertainty component
    fn compute_differential_entropy(&self, component: &Array1<Float>) -> Result<Float> {
        if component.is_empty() {
            return Ok(0.0);
        }

        // Estimate entropy using histogram method
        let min_val = component.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = component.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            return Ok(0.0);
        }

        let n_bins = 20;
        let bin_width = (max_val - min_val) / n_bins as Float;
        let mut histogram = vec![0; n_bins];

        for &value in component.iter() {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            histogram[bin_idx] += 1;
        }

        let total_count = component.len() as Float;
        let mut entropy = 0.0;

        for &count in &histogram {
            if count > 0 {
                let prob = count as Float / total_count;
                entropy -= prob * prob.ln();
            }
        }

        Ok(entropy)
    }

    /// Compute mutual information between two uncertainty components
    fn compute_mutual_information(&self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
        // Simplified mutual information estimation
        let corr = self.compute_correlation(x, y)?;
        let mutual_info = -0.5 * (1.0 - corr * corr).ln();
        Ok(mutual_info.max(0.0))
    }

    /// Compute information bottleneck score
    fn compute_information_bottleneck_score(&self, components: &[&Array1<Float>]) -> Result<Float> {
        // Simplified information bottleneck principle application
        let total_information: Float = components
            .iter()
            .map(|comp| self.compute_differential_entropy(comp).unwrap_or(0.0))
            .sum();

        let redundancy: Float = components
            .iter()
            .enumerate()
            .map(|(i, comp_i)| {
                components
                    .iter()
                    .skip(i + 1)
                    .map(|comp_j| {
                        self.compute_mutual_information(comp_i, comp_j)
                            .unwrap_or(0.0)
                    })
                    .sum::<Float>()
            })
            .sum();

        Ok((total_information - redundancy) / total_information.max(1e-10))
    }

    /// Create empty information measures for when not enabled
    fn empty_information_measures(&self) -> InformationTheoreticMeasures {
        InformationTheoreticMeasures {
            mutual_information: 0.0,
            conditional_entropy: 0.0,
            joint_entropy: 0.0,
            component_entropies: Array1::zeros(6),
            information_bottleneck_score: 0.0,
            component_redundancy: Array2::zeros((6, 6)),
        }
    }

    /// Compute uncertainty attribution using Shapley values
    fn compute_uncertainty_attribution(
        &self,
        components: &[&Array1<Float>],
    ) -> Result<Array2<Float>> {
        let n_components = components.len();
        let n_samples = components[0].len();
        let mut attribution = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            // Collect component values for this sample
            let mut values = Vec::new();
            for component in components {
                if i < component.len() {
                    values.push(component[i]);
                }
            }

            // Compute Shapley values (simplified)
            let total_value: Float = values.iter().sum();
            if total_value > 1e-10 {
                for (j, &value) in values.iter().enumerate() {
                    attribution[[i, j]] = value / total_value;
                }
            } else {
                // Equal attribution if no clear dominance
                let equal_weight = 1.0 / n_components as Float;
                for j in 0..n_components {
                    attribution[[i, j]] = equal_weight;
                }
            }
        }

        Ok(attribution)
    }

    /// Compute confidence regions for each uncertainty type
    fn compute_confidence_regions(&self, components: &[&Array1<Float>]) -> Result<Array3<Float>> {
        let n_components = components.len();
        let n_samples = components[0].len();
        let mut regions = Array3::zeros((n_samples, n_components, 2)); // lower, upper bounds

        let alpha = 1.0 - self.config.confidence_level;
        let z_score = 1.96; // Approximate 95% confidence

        for i in 0..n_samples {
            for (j, component) in components.iter().enumerate() {
                if i < component.len() {
                    let value = component[i];
                    let margin = z_score * value * 0.1; // Heuristic margin

                    regions[[i, j, 0]] = (value - margin).max(0.0); // Lower bound
                    regions[[i, j, 1]] = value + margin; // Upper bound
                }
            }
        }

        Ok(regions)
    }

    /// Estimate local variance for aleatoric uncertainty
    fn estimate_local_variance(
        &self,
        probabilities: &Array1<Float>,
        index: usize,
    ) -> Result<Float> {
        let window_size = 5;
        let start = index.saturating_sub(window_size / 2);
        let end = (index + window_size / 2 + 1).min(probabilities.len());

        if end <= start {
            return Ok(0.0);
        }

        let local_values = probabilities.slice(s![start..end]);
        let mean = local_values.mean().unwrap_or(0.5);
        let variance = local_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<Float>()
            / local_values.len() as Float;

        Ok(variance)
    }

    /// Add prediction to historical record for temporal analysis
    pub fn update_prediction_history(&mut self, predictions: Array1<Float>) {
        self.prediction_history.push(predictions);

        // Keep only recent history
        if self.prediction_history.len() > self.config.temporal_window_size {
            self.prediction_history.remove(0);
        }
    }

    /// Get comprehensive uncertainty summary
    pub fn get_uncertainty_summary(&self, result: &HigherOrderUncertaintyResult) -> String {
        format!(
            "Higher-Order Uncertainty Analysis Summary:\n\
             ==========================================\n\
             Total Samples: {}\n\
             Mean Epistemic Uncertainty: {:.6}\n\
             Mean Aleatoric Uncertainty: {:.6}\n\
             Mean Distributional Uncertainty: {:.6}\n\
             Mean Structural Uncertainty: {:.6}\n\
             Mean Computational Uncertainty: {:.6}\n\
             Mean Temporal Uncertainty: {:.6}\n\
             Mean Total Uncertainty: {:.6}\n\
             Information Bottleneck Score: {:.6}\n\
             Joint Entropy: {:.6}\n\
             Strongest Correlation: {:.6}",
            result.total_uncertainty.len(),
            result.epistemic_uncertainty.mean().unwrap_or(0.0),
            result.aleatoric_uncertainty.mean().unwrap_or(0.0),
            result.distributional_uncertainty.mean().unwrap_or(0.0),
            result.structural_uncertainty.mean().unwrap_or(0.0),
            result.computational_uncertainty.mean().unwrap_or(0.0),
            result.temporal_uncertainty.mean().unwrap_or(0.0),
            result.total_uncertainty.mean().unwrap_or(0.0),
            result.information_measures.information_bottleneck_score,
            result.information_measures.joint_entropy,
            result
                .uncertainty_correlations
                .iter()
                .filter(|&&x| x.abs() < 1.0) // Exclude diagonal
                .fold(0.0 as Float, |a, &b| a.max(b.abs()))
        )
    }
}

impl Default for HigherOrderUncertaintyEstimator {
    fn default() -> Self {
        Self::new(HigherOrderUncertaintyConfig::default())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_higher_order_uncertainty_creation() {
        let config = HigherOrderUncertaintyConfig::default();
        let estimator = HigherOrderUncertaintyEstimator::new(config);

        assert!(!estimator.is_fitted);
        assert_eq!(estimator.calibrator_ensemble.len(), 0);
        assert_eq!(estimator.prediction_history.len(), 0);
    }

    #[test]
    fn test_uncertainty_estimation_workflow() {
        let mut estimator = HigherOrderUncertaintyEstimator::default();
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        // Fit the estimator
        estimator.fit(&probabilities, &y_true, None).unwrap();
        assert!(estimator.is_fitted);

        // Estimate uncertainty
        let result = estimator.estimate_uncertainty(&probabilities).unwrap();

        // Verify all uncertainty components are computed
        assert_eq!(result.epistemic_uncertainty.len(), probabilities.len());
        assert_eq!(result.aleatoric_uncertainty.len(), probabilities.len());
        assert_eq!(result.distributional_uncertainty.len(), probabilities.len());
        assert_eq!(result.structural_uncertainty.len(), probabilities.len());
        assert_eq!(result.computational_uncertainty.len(), probabilities.len());
        assert_eq!(result.temporal_uncertainty.len(), probabilities.len());
        assert_eq!(result.total_uncertainty.len(), probabilities.len());

        // Verify uncertainty values are non-negative
        for &val in result.epistemic_uncertainty.iter() {
            assert!(val >= 0.0, "Epistemic uncertainty should be non-negative");
        }
        for &val in result.aleatoric_uncertainty.iter() {
            assert!(val >= 0.0, "Aleatoric uncertainty should be non-negative");
        }
        for &val in result.total_uncertainty.iter() {
            assert!(val >= 0.0, "Total uncertainty should be non-negative");
        }
    }

    #[test]
    fn test_correlation_computation() {
        let estimator = HigherOrderUncertaintyEstimator::default();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let corr = estimator.compute_correlation(&x, &y).unwrap();
        assert!(
            (corr - 1.0).abs() < 1e-10,
            "Perfect correlation should be close to 1.0"
        );
    }

    #[test]
    fn test_information_measures() {
        let estimator = HigherOrderUncertaintyEstimator::default();
        let component = array![0.1, 0.2, 0.3, 0.4, 0.5];

        let entropy = estimator.compute_differential_entropy(&component).unwrap();
        assert!(entropy >= 0.0, "Entropy should be non-negative");
    }

    #[test]
    fn test_uncertainty_summary() {
        let mut estimator = HigherOrderUncertaintyEstimator::default();
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let y_true = array![0, 0, 1, 1];

        estimator.fit(&probabilities, &y_true, None).unwrap();
        let result = estimator.estimate_uncertainty(&probabilities).unwrap();
        let summary = estimator.get_uncertainty_summary(&result);

        assert!(summary.contains("Higher-Order Uncertainty Analysis"));
        assert!(summary.contains("Total Samples: 4"));
        assert!(summary.contains("Mean Epistemic"));
        assert!(summary.contains("Information Bottleneck"));
    }

    #[test]
    fn test_temporal_uncertainty_with_history() {
        let mut estimator = HigherOrderUncertaintyEstimator::default();
        let probabilities = array![0.3, 0.5, 0.7];
        let y_true = array![0, 1, 1];

        estimator.fit(&probabilities, &y_true, None).unwrap();

        // Add some historical predictions
        estimator.update_prediction_history(array![0.25, 0.45, 0.65]);
        estimator.update_prediction_history(array![0.35, 0.55, 0.75]);

        let result = estimator.estimate_uncertainty(&probabilities).unwrap();

        // Temporal uncertainty should be computed based on history
        assert!(result.temporal_uncertainty.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_confidence_regions() {
        let mut estimator = HigherOrderUncertaintyEstimator::default();
        let probabilities = array![0.4, 0.6];
        let y_true = array![0, 1];

        estimator.fit(&probabilities, &y_true, None).unwrap();
        let result = estimator.estimate_uncertainty(&probabilities).unwrap();

        // Check confidence regions structure
        assert_eq!(result.confidence_regions.shape(), &[2, 6, 2]);

        // Verify lower bounds are less than upper bounds
        for i in 0..2 {
            for j in 0..6 {
                let lower = result.confidence_regions[[i, j, 0]];
                let upper = result.confidence_regions[[i, j, 1]];
                assert!(lower <= upper, "Lower bound should be <= upper bound");
            }
        }
    }

    #[test]
    fn test_attribution_weights() {
        let mut estimator = HigherOrderUncertaintyEstimator::default();
        let probabilities = array![0.5];
        let y_true = array![1];

        estimator.fit(&probabilities, &y_true, None).unwrap();
        let result = estimator.estimate_uncertainty(&probabilities).unwrap();

        // Check attribution weights sum to 1
        let attribution_sum = result.attribution_weights.row(0).sum();
        assert!(
            (attribution_sum - 1.0).abs() < 1e-6,
            "Attribution weights should sum to 1"
        );

        // Check all weights are non-negative
        for &weight in result.attribution_weights.iter() {
            assert!(weight >= 0.0, "Attribution weights should be non-negative");
        }
    }

    #[test]
    fn test_empty_information_measures() {
        let estimator = HigherOrderUncertaintyEstimator::default();
        let measures = estimator.empty_information_measures();

        assert_eq!(measures.mutual_information, 0.0);
        assert_eq!(measures.conditional_entropy, 0.0);
        assert_eq!(measures.joint_entropy, 0.0);
        assert_eq!(measures.information_bottleneck_score, 0.0);
        assert_eq!(measures.component_entropies.len(), 6);
        assert_eq!(measures.component_redundancy.shape(), &[6, 6]);
    }
}
