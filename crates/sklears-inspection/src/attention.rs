//! Attention and Activation Analysis
//!
//! This module provides comprehensive attention and activation analysis methods for neural networks,
//! including attention visualization, activation maximization, feature visualization,
//! gradient-weighted class activation mapping (Grad-CAM), and neural network dissection.

use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Configuration for attention and activation analysis
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Type of attention mechanism to analyze
    pub attention_type: AttentionType,
    /// Number of optimization steps for activation maximization
    pub max_iterations: usize,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Regularization strength
    pub regularization: Float,
    /// Target layer for analysis
    pub target_layer: Option<usize>,
    /// Target class for class-specific analysis
    pub target_class: Option<usize>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            attention_type: AttentionType::SelfAttention,
            max_iterations: 100,
            learning_rate: 0.01,
            regularization: 0.01,
            target_layer: None,
            target_class: None,
            random_seed: Some(42),
        }
    }
}

/// Types of attention mechanisms
#[derive(Debug, Clone, Copy)]
pub enum AttentionType {
    /// Self-attention (transformer-style)
    SelfAttention,
    /// Cross-attention
    CrossAttention,
    /// Multi-head attention
    MultiHeadAttention,
    /// Spatial attention (for CNNs)
    SpatialAttention,
    /// Channel attention
    ChannelAttention,
}

/// Configuration for Grad-CAM analysis
#[derive(Debug, Clone)]
pub struct GradCAMConfig {
    /// Target layer name or index
    pub target_layer: String,
    /// Target class for analysis
    pub target_class: usize,
    /// Whether to use guided backpropagation
    pub use_guided_backprop: bool,
    /// Upsampling method for visualization
    pub upsampling_method: UpsamplingMethod,
}

/// Upsampling methods for visualization
#[derive(Debug, Clone, Copy)]
pub enum UpsamplingMethod {
    Bilinear,
    Nearest,
    Bicubic,
}

/// Result of attention analysis
#[derive(Debug, Clone)]
pub struct AttentionResult {
    /// Attention weights/scores
    pub attention_weights: Array3<Float>, // (batch, heads, seq_len)
    /// Attention patterns
    pub attention_patterns: Array4<Float>, // (batch, heads, seq_len, seq_len)
    /// Average attention per head
    pub head_attention: Array2<Float>, // (heads, seq_len)
    /// Attention entropy (measure of attention distribution)
    pub attention_entropy: Array1<Float>, // (heads,)
    /// Most attended positions
    pub top_attended_positions: Vec<Vec<usize>>, // per head
}

/// Result of activation maximization
#[derive(Debug, Clone)]
pub struct ActivationMaximizationResult {
    /// Optimized input that maximizes activation
    pub optimal_input: Array2<Float>,
    /// Maximum activation value achieved
    pub max_activation: Float,
    /// Activation trajectory during optimization
    pub activation_trajectory: Array1<Float>,
    /// Input trajectory during optimization
    pub input_trajectory: Array3<Float>, // (iterations, height, width)
    /// Convergence information
    pub converged: bool,
    /// Number of iterations used
    pub iterations_used: usize,
}

/// Result of feature visualization
#[derive(Debug, Clone)]
pub struct FeatureVisualizationResult {
    /// Visualized features for each filter/neuron
    pub feature_maps: Array3<Float>, // (filters, height, width)
    /// Activation statistics
    pub activation_stats: Vec<ActivationStats>,
    /// Feature importance scores
    pub importance_scores: Array1<Float>,
    /// Preferred stimuli for each neuron
    pub preferred_stimuli: Array3<Float>, // (neurons, height, width)
}

/// Statistics for neuron activations
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Mean activation
    pub mean: Float,
    /// Standard deviation
    pub std: Float,
    /// Maximum activation
    pub max: Float,
    /// Minimum activation
    pub min: Float,
    /// Sparsity (fraction of zero activations)
    pub sparsity: Float,
    /// Selectivity (measure of how selective the neuron is)
    pub selectivity: Float,
}

/// Result of Grad-CAM analysis
#[derive(Debug, Clone)]
pub struct GradCAMResult {
    /// Class activation map
    pub activation_map: Array2<Float>,
    /// Guided gradients (if requested)
    pub guided_gradients: Option<Array2<Float>>,
    /// Guided Grad-CAM
    pub guided_gradcam: Option<Array2<Float>>,
    /// Raw gradients
    pub gradients: Array2<Float>,
    /// Feature maps from target layer
    pub feature_maps: Array3<Float>, // (channels, height, width)
    /// Class probability
    pub class_probability: Float,
}

/// Result of neural network dissection
#[derive(Debug, Clone)]
pub struct NetworkDissectionResult {
    /// Concept activations per neuron
    pub concept_activations: HashMap<String, Array1<Float>>,
    /// Neuron-concept alignment scores
    pub alignment_scores: HashMap<String, Array1<Float>>,
    /// Top concepts per neuron
    pub top_concepts_per_neuron: Vec<Vec<String>>,
    /// Top neurons per concept
    pub top_neurons_per_concept: HashMap<String, Vec<usize>>,
    /// Concept coverage statistics
    pub concept_coverage: HashMap<String, Float>,
}

/// Trait for models that support attention and activation analysis
pub trait AttentionAnalyzer {
    /// Extract attention weights from the model
    fn extract_attention_weights(&self, input: &ArrayView2<Float>) -> SklResult<Array3<Float>>;

    /// Get activations from a specific layer
    fn get_layer_activations(
        &self,
        input: &ArrayView2<Float>,
        layer: usize,
    ) -> SklResult<Array2<Float>>;

    /// Compute gradients with respect to input
    fn compute_gradients(
        &self,
        input: &ArrayView2<Float>,
        target_class: usize,
    ) -> SklResult<Array2<Float>>;

    /// Get feature maps from convolutional layers
    fn get_feature_maps(&self, input: &ArrayView2<Float>, layer: &str) -> SklResult<Array3<Float>>;

    /// Forward pass with intermediate activations
    fn forward_with_activations(
        &self,
        input: &ArrayView2<Float>,
    ) -> SklResult<HashMap<String, Array2<Float>>>;
}

/// Analyze attention patterns in neural networks
pub fn analyze_attention<M: AttentionAnalyzer>(
    model: &M,
    input: &ArrayView2<Float>,
    config: &AttentionConfig,
) -> SklResult<AttentionResult> {
    // Extract attention weights
    let attention_weights = model.extract_attention_weights(input)?;

    let (batch_size, n_heads, seq_len) = attention_weights.dim();

    // Compute attention patterns (assuming self-attention)
    let attention_patterns = match config.attention_type {
        AttentionType::SelfAttention => compute_self_attention_patterns(&attention_weights),
        AttentionType::MultiHeadAttention => {
            compute_multihead_attention_patterns(&attention_weights)
        }
        _ => compute_self_attention_patterns(&attention_weights), // Default
    };

    // Compute average attention per head
    let head_attention = attention_weights.mean_axis(Axis(0)).unwrap();

    // Compute attention entropy
    let attention_entropy = compute_attention_entropy(&attention_weights);

    // Find most attended positions
    let top_attended_positions = find_top_attended_positions(&attention_weights, 5);

    Ok(AttentionResult {
        attention_weights,
        attention_patterns,
        head_attention,
        attention_entropy,
        top_attended_positions,
    })
}

/// Compute self-attention patterns
fn compute_self_attention_patterns(attention_weights: &Array3<Float>) -> Array4<Float> {
    let (batch_size, n_heads, seq_len) = attention_weights.dim();
    let mut patterns = Array4::zeros((batch_size, n_heads, seq_len, seq_len));

    // For self-attention, create patterns showing which positions attend to which
    for b in 0..batch_size {
        for h in 0..n_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Simple attention pattern computation
                    patterns[[b, h, i, j]] =
                        attention_weights[[b, h, i]] * attention_weights[[b, h, j]];
                }
            }
        }
    }

    patterns
}

/// Compute multi-head attention patterns
fn compute_multihead_attention_patterns(attention_weights: &Array3<Float>) -> Array4<Float> {
    compute_self_attention_patterns(attention_weights)
}

/// Compute attention entropy for each head
fn compute_attention_entropy(attention_weights: &Array3<Float>) -> Array1<Float> {
    let (_, n_heads, _) = attention_weights.dim();
    let mut entropy = Array1::zeros(n_heads);

    for h in 0..n_heads {
        let head_weights = attention_weights.slice(s![.., h, ..]);
        let mut head_entropy = 0.0;
        let total_weight: Float = head_weights.sum();

        if total_weight > 0.0 {
            for &weight in head_weights.iter() {
                let prob = weight / total_weight;
                if prob > 0.0 {
                    head_entropy -= prob * prob.ln();
                }
            }
        }

        entropy[h] = head_entropy;
    }

    entropy
}

/// Find top attended positions for each head
fn find_top_attended_positions(attention_weights: &Array3<Float>, top_k: usize) -> Vec<Vec<usize>> {
    let (_, n_heads, seq_len) = attention_weights.dim();
    let mut top_positions = Vec::new();

    for h in 0..n_heads {
        let head_attention = attention_weights
            .slice(s![.., h, ..])
            .mean_axis(Axis(0))
            .unwrap();
        let mut indexed_attention: Vec<(Float, usize)> = head_attention
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        indexed_attention.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let top_indices: Vec<usize> = indexed_attention
            .into_iter()
            .take(top_k.min(seq_len))
            .map(|(_, idx)| idx)
            .collect();

        top_positions.push(top_indices);
    }

    top_positions
}

/// Perform activation maximization
pub fn maximize_activation<M: AttentionAnalyzer>(
    model: &M,
    target_layer: usize,
    input_shape: (usize, usize),
    config: &AttentionConfig,
) -> SklResult<ActivationMaximizationResult> {
    let (height, width) = input_shape;
    let mut rng = thread_rng();
    let input_dynamic =
        Array2::<Float>::from_shape_fn((height, width), |_| rng.gen_range(-1.0..=1.0)).into_dyn();
    let mut input = input_dynamic
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .unwrap();

    let mut activation_trajectory = Array1::zeros(config.max_iterations);
    let mut input_trajectory = Array3::zeros((config.max_iterations, height, width));

    let mut converged = false;
    let mut iterations_used = 0;

    for iter in 0..config.max_iterations {
        // Get current activation
        let activations = model.get_layer_activations(&input.view(), target_layer)?;
        let current_activation = activations.sum();

        activation_trajectory[iter] = current_activation;
        input_trajectory.slice_mut(s![iter, .., ..]).assign(&input);

        // Compute gradients
        if let Ok(gradients) = model.compute_gradients(&input.view(), 0) {
            // Update input using gradient ascent
            input = &input + config.learning_rate * &gradients;

            // Apply regularization (L2 norm constraint)
            let norm = input.iter().map(|&x| x * x).sum::<Float>().sqrt();
            if norm > 0.0 {
                input.mapv_inplace(|x| x / norm * (1.0 + config.regularization));
            }
        }

        iterations_used = iter + 1;

        // Check convergence
        if iter > 0 {
            let improvement = activation_trajectory[iter] - activation_trajectory[iter - 1];
            if improvement.abs() < 1e-6 {
                converged = true;
                break;
            }
        }
    }

    let max_activation = activation_trajectory[iterations_used - 1];

    Ok(ActivationMaximizationResult {
        optimal_input: input,
        max_activation,
        activation_trajectory,
        input_trajectory,
        converged,
        iterations_used,
    })
}

/// Visualize learned features
pub fn visualize_features<M: AttentionAnalyzer>(
    model: &M,
    layer: &str,
    input_samples: &ArrayView3<Float>, // (samples, height, width)
    config: &AttentionConfig,
) -> SklResult<FeatureVisualizationResult> {
    let (n_samples, height, width) = input_samples.dim();

    // Get feature maps for all samples
    let mut all_feature_maps = Vec::new();
    for i in 0..n_samples {
        let sample = input_samples.slice(s![i, .., ..]);
        let feature_maps = model.get_feature_maps(&sample, layer)?;
        all_feature_maps.push(feature_maps);
    }

    let n_filters = all_feature_maps[0].dim().0;
    let feature_height = all_feature_maps[0].dim().1;
    let feature_width = all_feature_maps[0].dim().2;

    // Compute statistics for each filter
    let mut activation_stats = Vec::new();
    let mut importance_scores = Array1::zeros(n_filters);

    for filter_idx in 0..n_filters {
        let mut all_activations = Vec::new();

        for feature_maps in &all_feature_maps {
            let filter_activations = feature_maps.slice(s![filter_idx, .., ..]);
            all_activations.extend(filter_activations.iter().cloned());
        }

        let stats = compute_activation_statistics(&all_activations);
        importance_scores[filter_idx] = stats.selectivity;
        activation_stats.push(stats);
    }

    // Create averaged feature maps
    let mut feature_maps = Array3::zeros((n_filters, feature_height, feature_width));
    for filter_idx in 0..n_filters {
        for i in 0..n_samples {
            let mut feature_slice = feature_maps.slice_mut(s![filter_idx, .., ..]);
            feature_slice += &all_feature_maps[i].slice(s![filter_idx, .., ..]);
        }
        feature_maps
            .slice_mut(s![filter_idx, .., ..])
            .mapv_inplace(|x| x / n_samples as Float);
    }

    // Find preferred stimuli (simplified)
    let preferred_stimuli = feature_maps.clone();

    Ok(FeatureVisualizationResult {
        feature_maps,
        activation_stats,
        importance_scores,
        preferred_stimuli,
    })
}

/// Compute activation statistics
fn compute_activation_statistics(activations: &[Float]) -> ActivationStats {
    if activations.is_empty() {
        return ActivationStats {
            mean: 0.0,
            std: 0.0,
            max: 0.0,
            min: 0.0,
            sparsity: 0.0,
            selectivity: 0.0,
        };
    }

    let mean = activations.iter().sum::<Float>() / activations.len() as Float;
    let variance = activations
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<Float>()
        / activations.len() as Float;
    let std = variance.sqrt();

    let max = activations
        .iter()
        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
    let min = activations.iter().fold(Float::INFINITY, |a, &b| a.min(b));

    let zero_count = activations.iter().filter(|&&x| x.abs() < 1e-6).count();
    let sparsity = zero_count as Float / activations.len() as Float;

    // Selectivity: normalized standard deviation
    let selectivity = if mean.abs() > 1e-10 {
        std / mean.abs()
    } else {
        0.0
    };

    ActivationStats {
        mean,
        std,
        max,
        min,
        sparsity,
        selectivity,
    }
}

/// Perform Grad-CAM analysis
pub fn compute_gradcam<M: AttentionAnalyzer>(
    model: &M,
    input: &ArrayView2<Float>,
    config: &GradCAMConfig,
) -> SklResult<GradCAMResult> {
    // Get feature maps from target layer
    let feature_maps = model.get_feature_maps(input, &config.target_layer)?;
    let (n_channels, height, width) = feature_maps.dim();

    // Compute gradients of target class score with respect to feature maps
    let gradients = model.compute_gradients(input, config.target_class)?;

    // Compute importance weights (global average pooling of gradients)
    let mut weights = Array1::zeros(n_channels);
    for c in 0..n_channels {
        let channel_gradients = gradients.slice(s![.., ..]); // Simplified - would need proper indexing
        weights[c] = channel_gradients.mean().unwrap_or(0.0);
    }

    // Compute weighted combination of feature maps
    let mut activation_map = Array2::zeros((height, width));
    for c in 0..n_channels {
        let weighted_feature_map = &feature_maps.slice(s![c, .., ..]) * weights[c];
        activation_map = activation_map + weighted_feature_map;
    }

    // Apply ReLU (keep only positive values)
    activation_map.mapv_inplace(|x: Float| x.max(0.0));

    // Normalize to [0, 1]
    let max_val = activation_map
        .iter()
        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
    if max_val > 0.0 {
        activation_map.mapv_inplace(|x| x / max_val);
    }

    // Placeholder for guided gradients and guided Grad-CAM
    let guided_gradients = if config.use_guided_backprop {
        Some(gradients.clone())
    } else {
        None
    };

    let guided_gradcam = if config.use_guided_backprop {
        Some(activation_map.clone())
    } else {
        None
    };

    Ok(GradCAMResult {
        activation_map,
        guided_gradients,
        guided_gradcam,
        gradients,
        feature_maps,
        class_probability: 0.5, // Placeholder
    })
}

/// Perform network dissection analysis
pub fn dissect_network<M: AttentionAnalyzer>(
    model: &M,
    concept_dataset: &ArrayView3<Float>, // (samples, height, width)
    concept_labels: &[String],
    target_layer: &str,
) -> SklResult<NetworkDissectionResult> {
    let n_samples = concept_dataset.dim().0;
    let n_concepts = concept_labels.len();

    // Get activations for all concept samples
    let mut all_activations = Vec::new();
    for i in 0..n_samples {
        let sample = concept_dataset.slice(s![i, .., ..]);
        let activations = model.get_layer_activations(&sample, 0)?; // Simplified layer access
        all_activations.push(activations);
    }

    let n_neurons = all_activations[0].ncols();

    // Compute concept activations
    let mut concept_activations = HashMap::new();
    let mut alignment_scores = HashMap::new();

    for (concept_idx, concept_name) in concept_labels.iter().enumerate() {
        let mut concept_neuron_activations = Array1::zeros(n_neurons);
        let mut concept_alignment = Array1::zeros(n_neurons);

        // For each neuron, compute how well it responds to this concept
        for neuron_idx in 0..n_neurons {
            let mut activations_for_concept = Vec::new();

            for sample_idx in 0..n_samples {
                // Assuming concept samples are organized somehow - this is simplified
                let activation = all_activations[sample_idx][[0, neuron_idx]];
                activations_for_concept.push(activation);
            }

            let mean_activation =
                activations_for_concept.iter().sum::<Float>() / n_samples as Float;
            concept_neuron_activations[neuron_idx] = mean_activation;

            // Compute alignment score (simplified as mean activation)
            concept_alignment[neuron_idx] = mean_activation;
        }

        concept_activations.insert(concept_name.clone(), concept_neuron_activations);
        alignment_scores.insert(concept_name.clone(), concept_alignment);
    }

    // Find top concepts per neuron
    let mut top_concepts_per_neuron = Vec::new();
    for neuron_idx in 0..n_neurons {
        let mut neuron_concept_scores: Vec<(Float, String)> = concept_labels
            .iter()
            .map(|concept_name| {
                let score = alignment_scores[concept_name][neuron_idx];
                (score, concept_name.clone())
            })
            .collect();

        neuron_concept_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let top_concepts: Vec<String> = neuron_concept_scores
            .into_iter()
            .take(3) // Top 3 concepts
            .map(|(_, concept)| concept)
            .collect();

        top_concepts_per_neuron.push(top_concepts);
    }

    // Find top neurons per concept
    let mut top_neurons_per_concept = HashMap::new();
    for concept_name in concept_labels {
        let concept_scores = &alignment_scores[concept_name];
        let mut indexed_scores: Vec<(Float, usize)> = concept_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (score, i))
            .collect();

        indexed_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let top_neurons: Vec<usize> = indexed_scores
            .into_iter()
            .take(5) // Top 5 neurons
            .map(|(_, idx)| idx)
            .collect();

        top_neurons_per_concept.insert(concept_name.clone(), top_neurons);
    }

    // Compute concept coverage
    let mut concept_coverage = HashMap::new();
    for concept_name in concept_labels {
        let concept_scores = &alignment_scores[concept_name];
        let active_neurons = concept_scores.iter().filter(|&&score| score > 0.1).count();
        let coverage = active_neurons as Float / n_neurons as Float;
        concept_coverage.insert(concept_name.clone(), coverage);
    }

    Ok(NetworkDissectionResult {
        concept_activations,
        alignment_scores,
        top_concepts_per_neuron,
        top_neurons_per_concept,
        concept_coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;
    use scirs2_core::ndarray_ext::Ix3;

    // Mock analyzer for testing
    struct MockAnalyzer;

    impl AttentionAnalyzer for MockAnalyzer {
        fn extract_attention_weights(&self, input: &ArrayView2<Float>) -> SklResult<Array3<Float>> {
            let batch_size = input.nrows();
            let seq_len = input.ncols();
            let n_heads = 4;
            let mut rng = thread_rng();
            let result_dynamic =
                Array3::<Float>::from_shape_fn((batch_size, n_heads, seq_len), |_| {
                    rng.gen_range(0.0..=1.0)
                })
                .into_dyn();
            let result = result_dynamic.into_dimensionality::<Ix3>().unwrap();
            Ok(result)
        }

        fn get_layer_activations(
            &self,
            input: &ArrayView2<Float>,
            _layer: usize,
        ) -> SklResult<Array2<Float>> {
            let mut rng = thread_rng();
            let result_dynamic =
                Array2::<Float>::from_shape_fn((input.nrows(), 64), |_| rng.gen_range(0.0..=1.0))
                    .into_dyn();
            let result = result_dynamic
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .unwrap();
            Ok(result)
        }

        fn compute_gradients(
            &self,
            input: &ArrayView2<Float>,
            _target_class: usize,
        ) -> SklResult<Array2<Float>> {
            let mut rng = thread_rng();
            let (rows, cols) = input.dim();
            let result_dynamic =
                Array2::<Float>::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..=1.0))
                    .into_dyn();
            let result = result_dynamic
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .unwrap();
            Ok(result)
        }

        fn get_feature_maps(
            &self,
            input: &ArrayView2<Float>,
            _layer: &str,
        ) -> SklResult<Array3<Float>> {
            let height = input.nrows();
            let width = input.ncols();
            let mut rng = thread_rng();
            let result_dynamic =
                Array3::<Float>::from_shape_fn((32, height / 2, width / 2), |_| {
                    rng.gen_range(0.0..=1.0)
                })
                .into_dyn();
            let result = result_dynamic.into_dimensionality::<Ix3>().unwrap();
            Ok(result)
        }

        fn forward_with_activations(
            &self,
            input: &ArrayView2<Float>,
        ) -> SklResult<HashMap<String, Array2<Float>>> {
            let mut activations = HashMap::new();
            let mut rng = thread_rng();
            let result_dynamic =
                Array2::<Float>::from_shape_fn((input.nrows(), 32), |_| rng.gen_range(0.0..=1.0))
                    .into_dyn();
            let result = result_dynamic
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .unwrap();
            activations.insert("layer1".to_string(), result);
            Ok(activations)
        }
    }

    #[test]
    fn test_attention_analysis() {
        let model = MockAnalyzer;
        let mut rng = thread_rng();
        let input_dynamic =
            Array2::<Float>::from_shape_fn((2, 10), |_| rng.gen_range(0.0..=1.0)).into_dyn();
        let input = input_dynamic
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        let config = AttentionConfig::default();

        let result = analyze_attention(&model, &input.view(), &config).unwrap();

        assert_eq!(result.attention_weights.dim(), (2, 4, 10));
        assert_eq!(result.attention_patterns.dim(), (2, 4, 10, 10));
        assert_eq!(result.head_attention.dim(), (4, 10));
        assert_eq!(result.attention_entropy.len(), 4);
    }

    #[test]
    fn test_activation_maximization() {
        let model = MockAnalyzer;
        let config = AttentionConfig::default();

        let result = maximize_activation(&model, 0, (10, 10), &config).unwrap();

        assert_eq!(result.optimal_input.dim(), (10, 10));
        assert_eq!(result.activation_trajectory.len(), config.max_iterations);
        assert!(result.max_activation > 0.0);
        assert!(result.iterations_used <= config.max_iterations);
    }

    #[test]
    fn test_feature_visualization() {
        let model = MockAnalyzer;
        let mut rng = thread_rng();
        let input_samples_dynamic =
            Array3::<Float>::from_shape_fn((5, 20, 20), |_| rng.gen_range(0.0..=1.0)).into_dyn();
        let input_samples = input_samples_dynamic.into_dimensionality::<Ix3>().unwrap();
        let config = AttentionConfig::default();

        let result = visualize_features(&model, "conv1", &input_samples.view(), &config).unwrap();

        assert_eq!(result.feature_maps.dim().0, 32); // Number of filters
        assert_eq!(result.activation_stats.len(), 32);
        assert_eq!(result.importance_scores.len(), 32);
    }

    #[test]
    fn test_gradcam() {
        let model = MockAnalyzer;
        let mut rng = thread_rng();
        let input_dynamic =
            Array2::<Float>::from_shape_fn((20, 20), |_| rng.gen_range(0.0..=1.0)).into_dyn();
        let input = input_dynamic
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        let config = GradCAMConfig {
            target_layer: "conv1".to_string(),
            target_class: 0,
            use_guided_backprop: true,
            upsampling_method: UpsamplingMethod::Bilinear,
        };

        let result = compute_gradcam(&model, &input.view(), &config).unwrap();

        assert_eq!(result.activation_map.dim(), (10, 10)); // Reduced size from conv layer
        assert!(result.guided_gradients.is_some());
        assert!(result.guided_gradcam.is_some());
    }

    #[test]
    fn test_attention_entropy() {
        let mut rng = thread_rng();
        let attention_weights_dynamic =
            Array3::<Float>::from_shape_fn((2, 3, 5), |_| rng.gen_range(0.0..=1.0)).into_dyn();
        let attention_weights = attention_weights_dynamic
            .into_dimensionality::<Ix3>()
            .unwrap();
        let entropy = compute_attention_entropy(&attention_weights);

        assert_eq!(entropy.len(), 3);
        assert!(entropy.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_activation_statistics() {
        let activations = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let stats = compute_activation_statistics(&activations);

        assert_eq!(stats.mean, 2.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 4.0);
        assert!(stats.std > 0.0);
        assert!(stats.selectivity > 0.0);
    }

    #[test]
    fn test_network_dissection() {
        let model = MockAnalyzer;
        let mut rng = thread_rng();
        let concept_dataset_dynamic =
            Array3::<Float>::from_shape_fn((10, 20, 20), |_| rng.gen_range(0.0..=1.0)).into_dyn();
        let concept_dataset = concept_dataset_dynamic
            .into_dimensionality::<Ix3>()
            .unwrap();
        let concept_labels = vec!["dog".to_string(), "cat".to_string(), "car".to_string()];

        let result =
            dissect_network(&model, &concept_dataset.view(), &concept_labels, "layer1").unwrap();

        assert_eq!(result.concept_activations.len(), 3);
        assert_eq!(result.alignment_scores.len(), 3);
        assert_eq!(result.concept_coverage.len(), 3);
        assert!(!result.top_concepts_per_neuron.is_empty());
    }
}
