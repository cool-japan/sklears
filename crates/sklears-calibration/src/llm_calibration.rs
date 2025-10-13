//! Large Language Model (LLM) Specific Calibration Methods
//!
//! This module implements state-of-the-art calibration techniques specifically designed
//! for large language models, including token-level calibration, sequence-level calibration,
//! verbalized confidence extraction, and attention-based calibration methods.
//!
//! These methods address the unique challenges of calibrating autoregressive language models
//! where uncertainty varies across tokens and sequences.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Token-level calibration for language models
///
/// Calibrates probability distributions at the token level, accounting for
/// positional dependencies and contextual information.
#[derive(Debug, Clone)]
pub struct TokenLevelCalibrator {
    /// Position-specific calibration parameters
    position_calibrators: Vec<TemperatureScaling>,
    /// Context-aware calibration weights
    context_weights: Array2<Float>,
    /// Maximum sequence length supported
    max_seq_length: usize,
    /// Whether to use positional encoding in calibration
    use_positional_encoding: bool,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl TokenLevelCalibrator {
    pub fn new(max_seq_length: usize) -> Self {
        Self {
            position_calibrators: Vec::new(),
            context_weights: Array2::zeros((0, 0)),
            max_seq_length,
            use_positional_encoding: true,
            is_fitted: false,
        }
    }

    /// Set whether to use positional encoding
    pub fn use_positional_encoding(mut self, use_pos: bool) -> Self {
        self.use_positional_encoding = use_pos;
        self
    }

    /// Fit token-level calibration parameters
    pub fn fit(
        &mut self,
        token_probabilities: &Array3<Float>, // [batch, sequence, vocab]
        true_tokens: &Array2<i32>,           // [batch, sequence]
        attention_weights: Option<&Array3<Float>>, // [batch, sequence, sequence]
    ) -> Result<()> {
        let (batch_size, seq_length, vocab_size) = token_probabilities.dim();

        if seq_length > self.max_seq_length {
            return Err(SklearsError::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seq_length, self.max_seq_length
            )));
        }

        // Initialize position-specific calibrators
        self.position_calibrators = Vec::with_capacity(seq_length);

        for pos in 0..seq_length {
            let mut calibrator = TemperatureScaling::new();

            // Extract probabilities and true labels for this position
            let pos_probs = token_probabilities.slice(s![.., pos, ..]);
            let pos_labels = true_tokens.slice(s![.., pos]);

            // Convert to probability format expected by calibrator
            let max_probs = pos_probs.map_axis(Axis(1), |row| {
                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max)
            });

            let binary_labels = pos_labels.mapv(|label| {
                // Convert to binary based on whether prediction was correct
                if label >= 0 && (label as usize) < vocab_size {
                    let pred_class = pos_probs
                        .slice(s![.., label as usize])
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    if pred_class == label as usize {
                        1
                    } else {
                        0
                    }
                } else {
                    0
                }
            });

            calibrator.fit(&max_probs, &binary_labels)?;
            self.position_calibrators.push(calibrator);
        }

        // Initialize context weights based on attention if provided
        if let Some(attn) = attention_weights {
            self.context_weights = attn.mean_axis(Axis(0)).unwrap().to_owned();
        } else {
            // Use uniform weights if no attention provided
            self.context_weights = Array2::ones((seq_length, seq_length)) / (seq_length as Float);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Calibrate token probabilities using position-specific calibrators
    pub fn calibrate(
        &self,
        token_probabilities: &Array3<Float>,
        attention_weights: Option<&Array3<Float>>,
    ) -> Result<Array3<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "calibrate token probabilities".to_string(),
            });
        }

        let (batch_size, seq_length, vocab_size) = token_probabilities.dim();
        let mut calibrated_probs = token_probabilities.clone();

        for pos in 0..seq_length.min(self.position_calibrators.len()) {
            let calibrator = &self.position_calibrators[pos];

            // Extract probabilities for this position
            let pos_probs = token_probabilities.slice(s![.., pos, ..]);

            // Apply position-specific calibration
            for batch_idx in 0..batch_size {
                let token_probs = pos_probs.slice(s![batch_idx, ..]);
                let max_prob = token_probs
                    .iter()
                    .cloned()
                    .fold(Float::NEG_INFINITY, Float::max);

                // Calibrate using temperature scaling
                let calibrated_max_prob = calibrator.calibrate_single(max_prob)?;

                // Apply scaling to all probabilities for this token
                let scale_factor = if max_prob > 0.0 {
                    calibrated_max_prob / max_prob
                } else {
                    1.0
                };

                let mut calibrated_token_probs = token_probs.to_owned() * scale_factor;

                // Renormalize to ensure probabilities sum to 1
                let sum = calibrated_token_probs.sum();
                if sum > 0.0 {
                    calibrated_token_probs /= sum;
                }

                calibrated_probs
                    .slice_mut(s![batch_idx, pos, ..])
                    .assign(&calibrated_token_probs);
            }
        }

        Ok(calibrated_probs)
    }
}

/// Sequence-level calibration for entire generated sequences
#[derive(Debug, Clone)]
pub struct SequenceLevelCalibrator {
    /// Sequence-level temperature parameter
    sequence_temperature: Float,
    /// Length-dependent calibration parameters
    length_calibrators: HashMap<usize, Float>,
    /// Aggregation method for sequence probabilities
    aggregation_method: SequenceAggregation,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub enum SequenceAggregation {
    /// Average log probabilities across sequence
    Mean,
    /// Geometric mean of probabilities
    GeometricMean,
    /// Minimum probability in sequence
    Min,
    /// Product of probabilities (joint probability)
    Product,
    /// Length-normalized log probability
    LengthNormalized,
}

impl SequenceLevelCalibrator {
    /// Create a new sequence-level calibrator
    pub fn new(aggregation_method: SequenceAggregation) -> Self {
        Self {
            sequence_temperature: 1.0,
            length_calibrators: HashMap::new(),
            aggregation_method,
            is_fitted: false,
        }
    }

    /// Fit sequence-level calibration parameters
    pub fn fit(
        &mut self,
        sequence_probabilities: &Array3<Float>, // [batch, sequence, vocab]
        sequence_rewards: &Array1<Float>,       // [batch] - quality scores for sequences
        sequence_lengths: &Array1<usize>,       // [batch] - actual sequence lengths
    ) -> Result<()> {
        let batch_size = sequence_probabilities.shape()[0];

        if sequence_rewards.len() != batch_size || sequence_lengths.len() != batch_size {
            return Err(SklearsError::InvalidInput(
                "Inconsistent batch sizes".to_string(),
            ));
        }

        // Compute sequence-level probabilities using aggregation method
        let mut sequence_probs = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let seq_len = sequence_lengths[batch_idx];
            let seq_data = sequence_probabilities.slice(s![batch_idx, ..seq_len, ..]);

            let aggregated_prob = match self.aggregation_method {
                SequenceAggregation::Mean => {
                    let log_probs: Float = seq_data
                        .map_axis(Axis(1), |row| {
                            let max_prob =
                                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                            if max_prob > 0.0 {
                                max_prob.ln()
                            } else {
                                Float::NEG_INFINITY
                            }
                        })
                        .mean()
                        .unwrap_or(0.0);
                    log_probs.exp()
                }
                SequenceAggregation::GeometricMean => {
                    let log_sum: Float = seq_data
                        .map_axis(Axis(1), |row| {
                            let max_prob =
                                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                            if max_prob > 0.0 {
                                max_prob.ln()
                            } else {
                                Float::NEG_INFINITY
                            }
                        })
                        .sum();
                    (log_sum / seq_len as Float).exp()
                }
                SequenceAggregation::Min => seq_data
                    .map_axis(Axis(1), |row| {
                        row.iter().cloned().fold(Float::INFINITY, Float::min)
                    })
                    .iter()
                    .cloned()
                    .fold(Float::INFINITY, Float::min),
                SequenceAggregation::Product => seq_data
                    .map_axis(Axis(1), |row| {
                        row.iter().cloned().fold(Float::NEG_INFINITY, Float::max)
                    })
                    .product(),
                SequenceAggregation::LengthNormalized => {
                    let log_sum: Float = seq_data
                        .map_axis(Axis(1), |row| {
                            let max_prob =
                                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                            if max_prob > 0.0 {
                                max_prob.ln()
                            } else {
                                Float::NEG_INFINITY
                            }
                        })
                        .sum();
                    (log_sum / seq_len as Float).exp()
                }
            };

            sequence_probs.push(aggregated_prob);
        }

        // Fit temperature parameter using sequence probabilities and rewards
        self.sequence_temperature =
            self.optimize_temperature(&sequence_probs, sequence_rewards.as_slice().unwrap())?;

        // Fit length-specific calibrators
        for &seq_len in sequence_lengths.iter() {
            if !self.length_calibrators.contains_key(&seq_len) {
                // Collect data for this sequence length
                let length_indices: Vec<usize> = sequence_lengths
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &len)| if len == seq_len { Some(idx) } else { None })
                    .collect();

                if length_indices.len() >= 2 {
                    let length_probs: Vec<Float> = length_indices
                        .iter()
                        .map(|&idx| sequence_probs[idx])
                        .collect();
                    let length_rewards: Vec<Float> = length_indices
                        .iter()
                        .map(|&idx| sequence_rewards[idx])
                        .collect();

                    let length_temp = self.optimize_temperature(&length_probs, &length_rewards)?;
                    self.length_calibrators.insert(seq_len, length_temp);
                }
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Optimize temperature parameter using negative log-likelihood
    fn optimize_temperature(&self, probs: &[Float], rewards: &[Float]) -> Result<Float> {
        let mut best_temp = 1.0;
        let mut best_loss = Float::INFINITY;

        // Grid search for optimal temperature
        for temp_candidate in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let mut loss = 0.0;

            for (&prob, &reward) in probs.iter().zip(rewards.iter()) {
                if prob > 0.0 && prob <= 1.0 {
                    let calibrated_prob = self.apply_temperature(prob, temp_candidate);
                    // Use MSE between calibrated probability and normalized reward
                    let normalized_reward = (reward + 1.0) / 2.0; // Assume rewards in [-1, 1]
                    loss += (calibrated_prob - normalized_reward).powi(2);
                }
            }

            if loss < best_loss {
                best_loss = loss;
                best_temp = temp_candidate;
            }
        }

        Ok(best_temp)
    }

    /// Apply temperature scaling to a probability
    fn apply_temperature(&self, prob: Float, temperature: Float) -> Float {
        if prob <= 0.0 || prob >= 1.0 {
            return prob;
        }

        let logit = prob.ln() - (1.0 - prob).ln();
        let scaled_logit = logit / temperature;
        1.0 / (1.0 + (-scaled_logit).exp())
    }

    /// Calibrate sequence probabilities
    pub fn calibrate_sequence(
        &self,
        sequence_probabilities: &Array3<Float>,
        sequence_lengths: &Array1<usize>,
    ) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "calibrate sequence probabilities".to_string(),
            });
        }

        let batch_size = sequence_probabilities.shape()[0];
        let mut calibrated_probs = Array1::zeros(batch_size);

        for batch_idx in 0..batch_size {
            let seq_len = sequence_lengths[batch_idx];
            let seq_data = sequence_probabilities.slice(s![batch_idx, ..seq_len, ..]);

            // Compute sequence probability using aggregation method
            let seq_prob = match self.aggregation_method {
                SequenceAggregation::Mean => {
                    let log_probs: Float = seq_data
                        .map_axis(Axis(1), |row| {
                            let max_prob =
                                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                            if max_prob > 0.0 {
                                max_prob.ln()
                            } else {
                                Float::NEG_INFINITY
                            }
                        })
                        .mean()
                        .unwrap_or(0.0);
                    log_probs.exp()
                }
                SequenceAggregation::GeometricMean => {
                    let log_sum: Float = seq_data
                        .map_axis(Axis(1), |row| {
                            let max_prob =
                                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                            if max_prob > 0.0 {
                                max_prob.ln()
                            } else {
                                Float::NEG_INFINITY
                            }
                        })
                        .sum();
                    (log_sum / seq_len as Float).exp()
                }
                SequenceAggregation::Min => seq_data
                    .map_axis(Axis(1), |row| {
                        row.iter().cloned().fold(Float::INFINITY, Float::min)
                    })
                    .iter()
                    .cloned()
                    .fold(Float::INFINITY, Float::min),
                SequenceAggregation::Product => seq_data
                    .map_axis(Axis(1), |row| {
                        row.iter().cloned().fold(Float::NEG_INFINITY, Float::max)
                    })
                    .product(),
                SequenceAggregation::LengthNormalized => {
                    let log_sum: Float = seq_data
                        .map_axis(Axis(1), |row| {
                            let max_prob =
                                row.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                            if max_prob > 0.0 {
                                max_prob.ln()
                            } else {
                                Float::NEG_INFINITY
                            }
                        })
                        .sum();
                    (log_sum / seq_len as Float).exp()
                }
            };

            // Apply temperature calibration
            let temperature = self
                .length_calibrators
                .get(&seq_len)
                .copied()
                .unwrap_or(self.sequence_temperature);

            let calibrated_prob = self.apply_temperature(seq_prob, temperature);
            calibrated_probs[batch_idx] = calibrated_prob.clamp(1e-10, 1.0 - 1e-10);
        }

        Ok(calibrated_probs)
    }
}

/// Verbalized confidence extraction for language models
///
/// Extracts explicit confidence statements from model outputs and calibrates them.
#[derive(Debug, Clone)]
pub struct VerbalizedConfidenceCalibrator {
    /// Confidence phrase patterns and their associated confidence levels
    confidence_patterns: HashMap<String, Float>,
    /// Calibration mapping from verbalized to actual confidence
    calibration_mapping: Array1<Float>,
    /// Confidence bins for calibration
    confidence_bins: Array1<Float>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl VerbalizedConfidenceCalibrator {
    /// Create a new verbalized confidence calibrator
    pub fn new() -> Self {
        let mut confidence_patterns = HashMap::new();

        // Initialize with common confidence phrases
        confidence_patterns.insert("I am certain".to_string(), 0.9);
        confidence_patterns.insert("I am confident".to_string(), 0.8);
        confidence_patterns.insert("I believe".to_string(), 0.6);
        confidence_patterns.insert("I think".to_string(), 0.5);
        confidence_patterns.insert("I'm not sure".to_string(), 0.3);
        confidence_patterns.insert("I don't know".to_string(), 0.1);
        confidence_patterns.insert("definitely".to_string(), 0.85);
        confidence_patterns.insert("probably".to_string(), 0.7);
        confidence_patterns.insert("possibly".to_string(), 0.4);
        confidence_patterns.insert("unlikely".to_string(), 0.2);

        Self {
            confidence_patterns,
            calibration_mapping: Array1::zeros(0),
            confidence_bins: Array1::zeros(0),
            is_fitted: false,
        }
    }

    /// Add a custom confidence pattern
    pub fn add_confidence_pattern(&mut self, pattern: String, confidence: Float) {
        self.confidence_patterns
            .insert(pattern, confidence.clamp(0.0, 1.0));
    }

    /// Extract confidence from text using pattern matching
    pub fn extract_verbalized_confidence(&self, text: &str) -> Float {
        let text_lower = text.to_lowercase();

        for (pattern, &confidence) in &self.confidence_patterns {
            if text_lower.contains(&pattern.to_lowercase()) {
                return confidence;
            }
        }

        // Default confidence if no pattern matches
        0.5
    }

    /// Fit calibration mapping from verbalized to actual confidence
    pub fn fit(
        &mut self,
        model_outputs: &[String],
        true_probabilities: &Array1<Float>,
    ) -> Result<()> {
        if model_outputs.len() != true_probabilities.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatched lengths between outputs and probabilities".to_string(),
            ));
        }

        // Extract verbalized confidences
        let verbalized_confidences: Vec<Float> = model_outputs
            .iter()
            .map(|text| self.extract_verbalized_confidence(text))
            .collect();

        // Create calibration bins
        let n_bins = 10;
        self.confidence_bins = Array1::linspace(0.0, 1.0, n_bins + 1);
        self.calibration_mapping = Array1::zeros(n_bins);

        // Compute bin-wise calibration mapping
        for bin_idx in 0..n_bins {
            let bin_start = self.confidence_bins[bin_idx];
            let bin_end = self.confidence_bins[bin_idx + 1];

            let bin_indices: Vec<usize> = verbalized_confidences
                .iter()
                .enumerate()
                .filter_map(|(idx, &conf)| {
                    if conf >= bin_start && conf < bin_end {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            if !bin_indices.is_empty() {
                let bin_true_probs: Float = bin_indices
                    .iter()
                    .map(|&idx| true_probabilities[idx])
                    .sum::<Float>()
                    / bin_indices.len() as Float;

                self.calibration_mapping[bin_idx] = bin_true_probs;
            } else {
                // Use linear interpolation for empty bins
                let bin_mid = (bin_start + bin_end) / 2.0;
                self.calibration_mapping[bin_idx] = bin_mid;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Calibrate verbalized confidence
    pub fn calibrate_verbalized_confidence(&self, text: &str) -> Result<Float> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "calibrate verbalized confidence".to_string(),
            });
        }

        let verbalized_conf = self.extract_verbalized_confidence(text);

        // Find the appropriate calibration bin
        let n_bins = self.calibration_mapping.len();
        for bin_idx in 0..n_bins {
            let bin_start = self.confidence_bins[bin_idx];
            let bin_end = self.confidence_bins[bin_idx + 1];

            if verbalized_conf >= bin_start && verbalized_conf < bin_end {
                return Ok(self.calibration_mapping[bin_idx]);
            }
        }

        // Fallback to linear interpolation
        Ok(verbalized_conf)
    }
}

/// Attention-based calibration using attention weights as confidence indicators
#[derive(Debug, Clone)]
pub struct AttentionBasedCalibrator {
    /// Attention entropy thresholds for confidence estimation
    entropy_thresholds: Array1<Float>,
    /// Calibration parameters for attention-based confidence
    attention_calibration_params: Array2<Float>,
    /// Attention aggregation method
    aggregation_method: AttentionAggregation,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub enum AttentionAggregation {
    /// Maximum attention weight
    Max,
    /// Mean attention weight
    Mean,
    /// Entropy of attention distribution
    Entropy,
    /// Standard deviation of attention weights
    Std,
}

impl AttentionBasedCalibrator {
    /// Create a new attention-based calibrator
    pub fn new(aggregation_method: AttentionAggregation) -> Self {
        Self {
            entropy_thresholds: Array1::zeros(0),
            attention_calibration_params: Array2::zeros((0, 0)),
            aggregation_method,
            is_fitted: false,
        }
    }

    /// Compute attention-based confidence score
    pub fn compute_attention_confidence(
        &self,
        attention_weights: &Array3<Float>, // [batch, sequence, sequence]
    ) -> Result<Array1<Float>> {
        let (batch_size, seq_len, _) = attention_weights.dim();
        let mut confidence_scores = Array1::zeros(batch_size);

        for batch_idx in 0..batch_size {
            let batch_attention = attention_weights.slice(s![batch_idx, .., ..]);

            // Compute confidence based on method
            let score = match self.aggregation_method {
                AttentionAggregation::Max => batch_attention
                    .iter()
                    .cloned()
                    .fold(Float::NEG_INFINITY, Float::max),
                AttentionAggregation::Mean => batch_attention.mean().unwrap_or(0.0),
                AttentionAggregation::Entropy => {
                    // Compute entropy of attention distribution
                    let mut entropy = 0.0;
                    for &weight in batch_attention.iter() {
                        if weight > 0.0 {
                            entropy -= weight * weight.ln();
                        }
                    }
                    entropy
                }
                AttentionAggregation::Std => {
                    let mean = batch_attention.mean().unwrap_or(0.0);
                    let variance = batch_attention
                        .mapv(|x| (x - mean).powi(2))
                        .mean()
                        .unwrap_or(0.0);
                    variance.sqrt()
                }
            };

            confidence_scores[batch_idx] = score;
        }

        Ok(confidence_scores)
    }

    /// Fit attention-based calibration parameters
    pub fn fit(
        &mut self,
        attention_weights: &Array3<Float>,
        true_confidences: &Array1<Float>,
    ) -> Result<()> {
        let attention_scores = self.compute_attention_confidence(attention_weights)?;

        // Create bins for calibration
        let n_bins = 10;
        let min_score = attention_scores
            .iter()
            .cloned()
            .fold(Float::INFINITY, Float::min);
        let max_score = attention_scores
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);

        self.entropy_thresholds = Array1::linspace(min_score, max_score, n_bins + 1);
        self.attention_calibration_params = Array2::zeros((n_bins, 2)); // [slope, intercept]

        // Fit calibration parameters for each bin
        for bin_idx in 0..n_bins {
            let bin_start = self.entropy_thresholds[bin_idx];
            let bin_end = self.entropy_thresholds[bin_idx + 1];

            let bin_indices: Vec<usize> = attention_scores
                .iter()
                .enumerate()
                .filter_map(|(idx, &score)| {
                    if score >= bin_start && score < bin_end {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            if bin_indices.len() >= 2 {
                // Fit linear regression for this bin
                let bin_attention_scores: Vec<Float> = bin_indices
                    .iter()
                    .map(|&idx| attention_scores[idx])
                    .collect();
                let bin_true_confidences: Vec<Float> = bin_indices
                    .iter()
                    .map(|&idx| true_confidences[idx])
                    .collect();

                let (slope, intercept) =
                    self.linear_regression(&bin_attention_scores, &bin_true_confidences);
                self.attention_calibration_params[[bin_idx, 0]] = slope;
                self.attention_calibration_params[[bin_idx, 1]] = intercept;
            } else {
                // Default to identity mapping
                self.attention_calibration_params[[bin_idx, 0]] = 1.0;
                self.attention_calibration_params[[bin_idx, 1]] = 0.0;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Simple linear regression implementation
    fn linear_regression(&self, x: &[Float], y: &[Float]) -> (Float, Float) {
        let n = x.len() as Float;
        let sum_x: Float = x.iter().sum();
        let sum_y: Float = y.iter().sum();
        let sum_xy: Float = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: Float = x.iter().map(|xi| xi * xi).sum();

        let slope = if n * sum_x2 - sum_x * sum_x != 0.0 {
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        } else {
            0.0
        };

        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Calibrate confidence using attention weights
    pub fn calibrate_attention_confidence(
        &self,
        attention_weights: &Array3<Float>,
    ) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "calibrate attention confidence".to_string(),
            });
        }

        let attention_scores = self.compute_attention_confidence(attention_weights)?;
        let mut calibrated_confidences = Array1::zeros(attention_scores.len());

        for (idx, &score) in attention_scores.iter().enumerate() {
            // Find appropriate calibration bin
            let mut bin_idx = 0;
            for i in 0..self.entropy_thresholds.len() - 1 {
                if score >= self.entropy_thresholds[i] && score < self.entropy_thresholds[i + 1] {
                    bin_idx = i;
                    break;
                }
            }

            // Apply calibration
            let slope = self.attention_calibration_params[[bin_idx, 0]];
            let intercept = self.attention_calibration_params[[bin_idx, 1]];

            calibrated_confidences[idx] = (slope * score + intercept).clamp(0.0, 1.0);
        }

        Ok(calibrated_confidences)
    }
}

/// Simple temperature scaling for internal use
#[derive(Debug, Clone)]
struct TemperatureScaling {
    temperature: Float,
    is_fitted: bool,
}

impl TemperatureScaling {
    fn new() -> Self {
        Self {
            temperature: 1.0,
            is_fitted: false,
        }
    }

    fn fit(&mut self, probabilities: &Array1<Float>, labels: &Array1<i32>) -> Result<()> {
        // Simple grid search for optimal temperature
        let mut best_temp = 1.0;
        let mut best_loss = Float::INFINITY;

        for temp_candidate in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let mut loss = 0.0;

            for (&prob, &label) in probabilities.iter().zip(labels.iter()) {
                if prob > 0.0 && prob < 1.0 {
                    let calibrated_prob = self.apply_temperature_scaling(prob, temp_candidate);
                    let target = if label == 1 { 1.0 } else { 0.0 };
                    loss += (calibrated_prob - target).powi(2);
                }
            }

            if loss < best_loss {
                best_loss = loss;
                best_temp = temp_candidate;
            }
        }

        self.temperature = best_temp;
        self.is_fitted = true;
        Ok(())
    }

    fn calibrate_single(&self, probability: Float) -> Result<Float> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "calibrate single probability".to_string(),
            });
        }
        Ok(self.apply_temperature_scaling(probability, self.temperature))
    }

    fn apply_temperature_scaling(&self, prob: Float, temperature: Float) -> Float {
        if prob <= 0.0 || prob >= 1.0 {
            return prob;
        }

        let logit = prob.ln() - (1.0 - prob).ln();
        let scaled_logit = logit / temperature;
        1.0 / (1.0 + (-scaled_logit).exp())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_token_level_calibrator_creation() {
        let calibrator = TokenLevelCalibrator::new(100);
        assert_eq!(calibrator.max_seq_length, 100);
        assert!(!calibrator.is_fitted);
    }

    #[test]
    fn test_sequence_level_calibrator_creation() {
        let calibrator = SequenceLevelCalibrator::new(SequenceAggregation::Mean);
        assert!(!calibrator.is_fitted);
    }

    #[test]
    fn test_verbalized_confidence_extraction() {
        let calibrator = VerbalizedConfidenceCalibrator::new();

        assert_eq!(
            calibrator.extract_verbalized_confidence("I am certain this is correct"),
            0.9
        );
        assert_eq!(
            calibrator.extract_verbalized_confidence("I think this might be right"),
            0.5
        );
        assert_eq!(
            calibrator.extract_verbalized_confidence("No confidence phrases here"),
            0.5
        );
    }

    #[test]
    fn test_attention_based_calibrator_creation() {
        let calibrator = AttentionBasedCalibrator::new(AttentionAggregation::Entropy);
        assert!(!calibrator.is_fitted);
    }

    #[test]
    fn test_token_level_calibrator_fitting() {
        let mut calibrator = TokenLevelCalibrator::new(3);

        // Create dummy data
        let token_probs = Array3::from_shape_vec(
            (2, 3, 4),
            vec![
                0.1, 0.2, 0.3, 0.4, // batch 0, position 0
                0.2, 0.3, 0.4, 0.1, // batch 0, position 1
                0.3, 0.4, 0.1, 0.2, // batch 0, position 2
                0.4, 0.1, 0.2, 0.3, // batch 1, position 0
                0.1, 0.4, 0.3, 0.2, // batch 1, position 1
                0.2, 0.1, 0.4, 0.3, // batch 1, position 2
            ],
        )
        .unwrap();

        let true_tokens = array![[3, 2, 1], [0, 1, 2]];

        let result = calibrator.fit(&token_probs, &true_tokens, None);
        assert!(result.is_ok());
        assert!(calibrator.is_fitted);
    }

    #[test]
    fn test_sequence_level_calibrator_fitting() {
        let mut calibrator = SequenceLevelCalibrator::new(SequenceAggregation::Mean);

        // Create dummy data
        let seq_probs = Array3::from_shape_vec(
            (2, 3, 4),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.1, 0.3, 0.4, 0.1, 0.2, 0.4, 0.1, 0.2, 0.3,
                0.1, 0.4, 0.3, 0.2, 0.2, 0.1, 0.4, 0.3,
            ],
        )
        .unwrap();

        let seq_rewards = array![0.8, 0.6];
        let seq_lengths = array![3, 3];

        let result = calibrator.fit(&seq_probs, &seq_rewards, &seq_lengths);
        assert!(result.is_ok());
        assert!(calibrator.is_fitted);
    }

    #[test]
    fn test_verbalized_confidence_calibrator_fitting() {
        let mut calibrator = VerbalizedConfidenceCalibrator::new();

        let model_outputs = vec![
            "I am certain this is correct".to_string(),
            "I think this might be right".to_string(),
            "I'm not sure about this".to_string(),
        ];

        let true_probs = array![0.95, 0.6, 0.3];

        let result = calibrator.fit(&model_outputs, &true_probs);
        assert!(result.is_ok());
        assert!(calibrator.is_fitted);
    }

    #[test]
    fn test_attention_confidence_computation() {
        let calibrator = AttentionBasedCalibrator::new(AttentionAggregation::Mean);

        // Create dummy attention weights [batch=1, seq=3, seq=3]
        let attention = Array3::from_shape_vec(
            (1, 3, 3),
            vec![
                0.1, 0.2, 0.7, // position 0
                0.3, 0.4, 0.3, // position 1
                0.6, 0.2, 0.2, // position 2
            ],
        )
        .unwrap();

        let result = calibrator.compute_attention_confidence(&attention);
        assert!(result.is_ok());

        let confidence_scores = result.unwrap();
        assert_eq!(confidence_scores.len(), 1);
        assert!(confidence_scores[0] > 0.0);
        assert!(confidence_scores[0] <= 1.0);
    }
}
