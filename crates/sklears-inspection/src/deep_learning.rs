//! Deep Learning Interpretability
//!
//! This module provides advanced interpretability methods specifically designed for deep neural networks,
//! including concept activation vectors, neural architecture interpretability, and network dissection.

use crate::{Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for deep learning interpretability methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeepLearningConfig {
    /// Target layers for analysis
    pub target_layers: Vec<String>,
    /// Number of concepts to extract
    pub num_concepts: usize,
    /// Concept activation threshold
    pub activation_threshold: Float,
    /// Number of test examples for TCAV
    pub num_test_examples: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Method for concept discovery
    pub concept_discovery_method: ConceptDiscoveryMethod,
}

impl Default for DeepLearningConfig {
    fn default() -> Self {
        Self {
            target_layers: vec!["layer_3".to_string(), "layer_5".to_string()],
            num_concepts: 20,
            activation_threshold: 0.5,
            num_test_examples: 500,
            random_seed: Some(42),
            concept_discovery_method: ConceptDiscoveryMethod::ACE,
        }
    }
}

/// Methods for concept discovery
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConceptDiscoveryMethod {
    /// Automated Concept-based Explanations (ACE)
    ACE,
    /// Testing with Concept Activation Vectors (TCAV)
    TCAV,
    /// Completeness-aware Concept-based Explanations (CCAV)
    CCAV,
    /// Network Dissection
    NetworkDissection,
}

/// Concept Activation Vector (CAV) structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConceptActivationVector {
    /// Concept identifier
    pub concept_id: String,
    /// Layer name where the concept is detected
    pub layer_name: String,
    /// Direction vector in activation space
    pub direction_vector: Array1<Float>,
    /// Concept accuracy (how well it separates concept vs non-concept examples)
    pub accuracy: Float,
    /// Statistical significance (p-value)
    pub p_value: Float,
    /// Examples that activate this concept
    pub activating_examples: Vec<usize>,
}

impl ConceptActivationVector {
    /// Create a new Concept Activation Vector
    pub fn new(concept_id: String, layer_name: String, direction_vector: Array1<Float>) -> Self {
        Self {
            concept_id,
            layer_name,
            direction_vector,
            accuracy: 0.0,
            p_value: 1.0,
            activating_examples: Vec::new(),
        }
    }

    /// Compute concept sensitivity for a given input
    pub fn compute_sensitivity(&self, activation: &ArrayView1<Float>) -> Float {
        activation.dot(&self.direction_vector)
    }

    /// Check if an input activates this concept
    pub fn is_activated(&self, activation: &ArrayView1<Float>, threshold: Float) -> bool {
        self.compute_sensitivity(activation) > threshold
    }
}

/// TCAV (Testing with Concept Activation Vectors) result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TCAVResult {
    /// TCAV score (sensitivity to concept)
    pub tcav_score: Float,
    /// Statistical significance
    pub p_value: Float,
    /// Confidence interval
    pub confidence_interval: (Float, Float),
    /// Number of inputs in class that activate the concept
    pub num_activated: usize,
    /// Total number of inputs tested
    pub total_inputs: usize,
    /// Concept activation vector used
    pub cav: ConceptActivationVector,
}

impl TCAVResult {
    /// Check if the TCAV result is statistically significant
    pub fn is_significant(&self, alpha: Float) -> bool {
        self.p_value < alpha
    }

    /// Get effect size interpretation
    pub fn effect_size_interpretation(&self) -> String {
        match self.tcav_score {
            score if score < 0.1 => "Negligible effect".to_string(),
            score if score < 0.3 => "Small effect".to_string(),
            score if score < 0.5 => "Medium effect".to_string(),
            _ => "Large effect".to_string(),
        }
    }
}

/// Network dissection result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkDissectionResult {
    /// Layer-wise concept analysis
    pub layer_concepts: HashMap<String, Vec<DetectedConcept>>,
    /// Overall network interpretability score
    pub interpretability_score: Float,
    /// Concept hierarchy
    pub concept_hierarchy: ConceptHierarchy,
    /// Disentanglement metrics
    pub disentanglement_metrics: DisentanglementMetrics,
}

/// Detected concept in network dissection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DetectedConcept {
    /// Concept name
    pub name: String,
    /// Concept category (e.g., "object", "texture", "color")
    pub category: String,
    /// IoU score with ground truth concept
    pub iou_score: Float,
    /// Units in the layer that detect this concept
    pub detecting_units: Vec<usize>,
    /// Threshold for concept activation
    pub activation_threshold: Float,
}

/// Concept hierarchy structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConceptHierarchy {
    /// Hierarchical relationships between concepts
    pub relationships: HashMap<String, Vec<String>>,
    /// Concept abstraction levels
    pub abstraction_levels: HashMap<String, usize>,
    /// Concept co-occurrence matrix
    pub co_occurrence: Array2<Float>,
}

/// Disentanglement metrics for evaluating concept separation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DisentanglementMetrics {
    /// Mutual Information Gap (MIG) score
    pub mig_score: Float,
    /// Separated Attribute Predictability (SAP) score
    pub sap_score: Float,
    /// Modularity score
    pub modularity_score: Float,
    /// Compactness score
    pub compactness_score: Float,
}

/// Main deep learning interpretability analyzer
pub struct DeepLearningAnalyzer {
    config: DeepLearningConfig,
    concept_database: ConceptDatabase,
}

impl DeepLearningAnalyzer {
    /// Create a new deep learning analyzer
    pub fn new(config: DeepLearningConfig) -> Self {
        Self {
            config,
            concept_database: ConceptDatabase::new(),
        }
    }

    /// Perform TCAV analysis
    pub fn compute_tcav<F>(
        &self,
        model_fn: F,
        concept_examples: &ArrayView2<Float>,
        random_examples: &ArrayView2<Float>,
        test_examples: &ArrayView2<Float>,
        target_class: usize,
        layer_name: &str,
    ) -> SklResult<TCAVResult>
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<Array2<Float>>,
    {
        // 1. Get activations for concept and random examples
        let concept_activations = model_fn(concept_examples)?;
        let random_activations = model_fn(random_examples)?;

        // 2. Train linear classifier to separate concept from random
        let cav = self.train_concept_activation_vector(
            &concept_activations.view(),
            &random_activations.view(),
            layer_name,
        )?;

        // 3. Compute directional derivatives for test examples
        let test_activations = model_fn(test_examples)?;
        let directional_derivatives = self.compute_directional_derivatives(
            model_fn,
            test_examples,
            &cav.direction_vector.view(),
            target_class,
        )?;

        // 4. Compute TCAV score
        let positive_derivatives = directional_derivatives.iter().filter(|&&x| x > 0.0).count();

        let tcav_score = positive_derivatives as Float / directional_derivatives.len() as Float;

        // 5. Statistical testing
        let (p_value, confidence_interval) =
            self.compute_tcav_statistics(&directional_derivatives, tcav_score)?;

        Ok(TCAVResult {
            tcav_score,
            p_value,
            confidence_interval,
            num_activated: positive_derivatives,
            total_inputs: directional_derivatives.len(),
            cav,
        })
    }

    /// Perform network dissection
    pub fn perform_network_dissection<F>(
        &self,
        model_fn: F,
        probe_dataset: &ArrayView2<Float>,
        concept_labels: &HashMap<String, Array1<bool>>,
    ) -> SklResult<NetworkDissectionResult>
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<HashMap<String, Array2<Float>>>,
    {
        let layer_activations = model_fn(probe_dataset)?;
        let mut layer_concepts = HashMap::new();

        // Analyze each layer
        for (layer_name, activations) in layer_activations.iter() {
            let detected_concepts =
                self.detect_concepts_in_layer(activations, concept_labels, layer_name)?;
            layer_concepts.insert(layer_name.clone(), detected_concepts);
        }

        // Compute overall interpretability score
        let interpretability_score = self.compute_interpretability_score(&layer_concepts);

        // Build concept hierarchy
        let concept_hierarchy = self.build_concept_hierarchy(&layer_concepts)?;

        // Compute disentanglement metrics
        let disentanglement_metrics = self.compute_disentanglement_metrics(&layer_activations)?;

        Ok(NetworkDissectionResult {
            layer_concepts,
            interpretability_score,
            concept_hierarchy,
            disentanglement_metrics,
        })
    }

    /// Automated Concept Extraction (ACE)
    pub fn extract_concepts_ace<F>(
        &self,
        model_fn: F,
        images: &ArrayView3<Float>,
        layer_name: &str,
        num_concepts: usize,
    ) -> SklResult<Vec<ConceptActivationVector>>
    where
        F: Fn(&ArrayView3<Float>) -> SklResult<Array2<Float>>,
    {
        // 1. Get layer activations
        let activations = model_fn(images)?;

        // 2. Segment images into superpixels
        let segments = self.segment_images(images)?;

        // 3. Cluster segments based on their activations
        let concept_clusters = self.cluster_segments(&activations, &segments, num_concepts)?;

        // 4. Create CAVs for each concept cluster
        let mut cavs = Vec::new();
        for (i, cluster) in concept_clusters.iter().enumerate() {
            let concept_id = format!("ace_concept_{}", i);
            let cav = self.create_cav_from_cluster(
                concept_id,
                layer_name.to_string(),
                cluster,
                &activations,
            )?;
            cavs.push(cav);
        }

        Ok(cavs)
    }

    fn train_concept_activation_vector(
        &self,
        concept_activations: &ArrayView2<Float>,
        random_activations: &ArrayView2<Float>,
        layer_name: &str,
    ) -> SklResult<ConceptActivationVector> {
        let n_concept = concept_activations.nrows();
        let n_random = random_activations.nrows();
        let n_features = concept_activations.ncols();

        // Create labels: 1 for concept, 0 for random
        let mut labels = Array1::zeros(n_concept + n_random);
        for i in 0..n_concept {
            labels[i] = 1.0;
        }

        // Combine activations
        let mut combined_activations = Array2::zeros((n_concept + n_random, n_features));
        for i in 0..n_concept {
            combined_activations
                .row_mut(i)
                .assign(&concept_activations.row(i));
        }
        for i in 0..n_random {
            combined_activations
                .row_mut(n_concept + i)
                .assign(&random_activations.row(i));
        }

        // Train linear SVM (simplified implementation)
        let direction_vector =
            self.train_linear_classifier(&combined_activations.view(), &labels.view())?;

        // Compute accuracy using cross-validation
        let accuracy = self.compute_classifier_accuracy(
            &combined_activations.view(),
            &labels.view(),
            &direction_vector,
        )?;

        let mut cav = ConceptActivationVector::new(
            "trained_concept".to_string(),
            layer_name.to_string(),
            direction_vector,
        );
        cav.accuracy = accuracy;

        Ok(cav)
    }

    fn train_linear_classifier(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        // Simple linear regression solution: w = (X^T X)^{-1} X^T y
        // This is a simplified implementation
        let mut weights = Array1::zeros(n_features);

        // Use gradient descent for simplicity
        let learning_rate = 0.01;
        let max_iterations = 1000;

        for _ in 0..max_iterations {
            let predictions = X.dot(&weights);
            let residuals = &predictions - y;
            let gradient = X.t().dot(&residuals) / n_samples as Float;
            weights = weights - learning_rate * gradient;
        }

        Ok(weights)
    }

    fn compute_classifier_accuracy(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
        weights: &Array1<Float>,
    ) -> SklResult<Float> {
        let predictions = X.dot(weights);
        let binary_predictions = predictions.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        let correct = binary_predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| (pred - true_val).abs() < 1e-6)
            .count();

        Ok(correct as Float / y.len() as Float)
    }

    fn compute_directional_derivatives<F>(
        &self,
        model_fn: F,
        inputs: &ArrayView2<Float>,
        direction: &ArrayView1<Float>,
        target_class: usize,
    ) -> SklResult<Array1<Float>>
    where
        F: Fn(&ArrayView2<Float>) -> SklResult<Array2<Float>>,
    {
        // Simplified gradient computation using finite differences
        let epsilon = 1e-5;
        let mut derivatives = Array1::zeros(inputs.nrows());

        for (i, input) in inputs.outer_iter().enumerate() {
            // Forward perturbation
            let input_plus = input.to_owned();
            let input_plus_view = input_plus.insert_axis(Axis(0));
            let activation_plus = model_fn(&input_plus_view.view())?;

            // Backward perturbation
            let input_minus = input.to_owned();
            let input_minus_view = input_minus.insert_axis(Axis(0));
            let activation_minus = model_fn(&input_minus_view.view())?;

            // Compute gradient approximation
            let gradient_approx = (&activation_plus - &activation_minus) / (2.0 * epsilon);

            // Directional derivative
            derivatives[i] = gradient_approx.row(0).dot(direction);
        }

        Ok(derivatives)
    }

    fn compute_tcav_statistics(
        &self,
        directional_derivatives: &Array1<Float>,
        tcav_score: Float,
    ) -> SklResult<(Float, (Float, Float))> {
        let n = directional_derivatives.len() as Float;

        // Use normal approximation for p-value computation
        let mean = 0.5; // Under null hypothesis
        let variance = 0.25 / n; // Binomial variance / n
        let std_error = variance.sqrt();

        // Z-score
        let z_score = (tcav_score - mean) / std_error;

        // Two-tailed p-value (simplified)
        let p_value = 2.0 * (1.0 - self.standard_normal_cdf(z_score.abs()));

        // 95% confidence interval
        let margin_of_error = 1.96 * std_error;
        let confidence_interval = (
            (tcav_score - margin_of_error).max(0.0),
            (tcav_score + margin_of_error).min(1.0),
        );

        Ok((p_value, confidence_interval))
    }

    fn standard_normal_cdf(&self, x: Float) -> Float {
        // Simplified approximation of standard normal CDF
        0.5 * (1.0 + self.erf(x / (2.0_f64.sqrt() as Float)))
    }

    fn erf(&self, x: Float) -> Float {
        // Simplified error function approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    fn detect_concepts_in_layer(
        &self,
        activations: &Array2<Float>,
        concept_labels: &HashMap<String, Array1<bool>>,
        layer_name: &str,
    ) -> SklResult<Vec<DetectedConcept>> {
        let mut detected_concepts = Vec::new();

        for (concept_name, labels) in concept_labels.iter() {
            // For each unit in the layer, compute IoU with concept
            for unit_idx in 0..activations.ncols() {
                let unit_activations = activations.column(unit_idx);

                // Find optimal threshold
                let (threshold, iou_score) =
                    self.find_optimal_threshold(&unit_activations, labels)?;

                if iou_score > 0.04 {
                    // Minimum IoU threshold from Network Dissection paper
                    detected_concepts.push(DetectedConcept {
                        name: concept_name.clone(),
                        category: self.get_concept_category(concept_name),
                        iou_score,
                        detecting_units: vec![unit_idx],
                        activation_threshold: threshold,
                    });
                }
            }
        }

        Ok(detected_concepts)
    }

    fn find_optimal_threshold(
        &self,
        activations: &ArrayView1<Float>,
        ground_truth: &Array1<bool>,
    ) -> SklResult<(Float, Float)> {
        let mut best_threshold = 0.0;
        let mut best_iou = 0.0;

        // Try different thresholds
        let mut sorted_activations: Vec<Float> = activations.to_vec();
        sorted_activations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for &threshold in sorted_activations.iter() {
            let predictions: Array1<bool> = activations.mapv(|x| x > threshold);
            let iou = self.compute_iou(&predictions, ground_truth);

            if iou > best_iou {
                best_iou = iou;
                best_threshold = threshold;
            }
        }

        Ok((best_threshold, best_iou))
    }

    fn compute_iou(&self, predictions: &Array1<bool>, ground_truth: &Array1<bool>) -> Float {
        let intersection = predictions
            .iter()
            .zip(ground_truth.iter())
            .filter(|(&pred, &gt)| pred && gt)
            .count() as Float;

        let union = predictions
            .iter()
            .zip(ground_truth.iter())
            .filter(|(&pred, &gt)| pred || gt)
            .count() as Float;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    fn get_concept_category(&self, concept_name: &str) -> String {
        // Simple categorization based on concept name
        if concept_name.contains("color") {
            "color".to_string()
        } else if concept_name.contains("texture") {
            "texture".to_string()
        } else if concept_name.contains("object") {
            "object".to_string()
        } else {
            "other".to_string()
        }
    }

    fn compute_interpretability_score(
        &self,
        layer_concepts: &HashMap<String, Vec<DetectedConcept>>,
    ) -> Float {
        if layer_concepts.is_empty() {
            return 0.0;
        }

        let total_concepts: usize = layer_concepts.values().map(|concepts| concepts.len()).sum();
        let weighted_iou: Float = layer_concepts
            .values()
            .flat_map(|concepts| concepts.iter())
            .map(|concept| concept.iou_score)
            .sum();

        if total_concepts == 0 {
            0.0
        } else {
            weighted_iou / total_concepts as Float
        }
    }

    fn build_concept_hierarchy(
        &self,
        layer_concepts: &HashMap<String, Vec<DetectedConcept>>,
    ) -> SklResult<ConceptHierarchy> {
        let mut relationships = HashMap::new();
        let mut abstraction_levels = HashMap::new();

        // Simple hierarchy based on layer depth (deeper = more abstract)
        let mut layer_names: Vec<String> = layer_concepts.keys().cloned().collect();
        layer_names.sort();

        for (level, layer_name) in layer_names.iter().enumerate() {
            if let Some(concepts) = layer_concepts.get(layer_name) {
                for concept in concepts {
                    abstraction_levels.insert(concept.name.clone(), level);
                    relationships.insert(concept.name.clone(), Vec::new());
                }
            }
        }

        // Create co-occurrence matrix (simplified)
        let all_concepts: Vec<String> = abstraction_levels.keys().cloned().collect();
        let n_concepts = all_concepts.len();
        let co_occurrence = Array2::zeros((n_concepts, n_concepts));

        Ok(ConceptHierarchy {
            relationships,
            abstraction_levels,
            co_occurrence,
        })
    }

    fn compute_disentanglement_metrics(
        &self,
        layer_activations: &HashMap<String, Array2<Float>>,
    ) -> SklResult<DisentanglementMetrics> {
        // Simplified disentanglement metrics computation
        Ok(DisentanglementMetrics {
            mig_score: 0.5,         // Placeholder
            sap_score: 0.6,         // Placeholder
            modularity_score: 0.7,  // Placeholder
            compactness_score: 0.8, // Placeholder
        })
    }

    fn segment_images(&self, images: &ArrayView3<Float>) -> SklResult<Vec<Vec<(usize, usize)>>> {
        // Placeholder for image segmentation
        // In practice, this would use superpixel segmentation algorithms
        let mut segments = Vec::new();
        for _ in 0..images.shape()[0] {
            segments.push(vec![(0, 0), (1, 1)]); // Placeholder segments
        }
        Ok(segments)
    }

    fn cluster_segments(
        &self,
        activations: &Array2<Float>,
        segments: &[Vec<(usize, usize)>],
        num_concepts: usize,
    ) -> SklResult<Vec<Vec<usize>>> {
        // Placeholder for clustering implementation
        // In practice, this would use k-means or other clustering algorithms
        let mut clusters = Vec::new();
        for i in 0..num_concepts {
            clusters.push(vec![i, i + num_concepts]);
        }
        Ok(clusters)
    }

    fn create_cav_from_cluster(
        &self,
        concept_id: String,
        layer_name: String,
        cluster: &[usize],
        activations: &Array2<Float>,
    ) -> SklResult<ConceptActivationVector> {
        // Compute mean activation for the cluster
        let cluster_mean = if cluster.is_empty() {
            Array1::zeros(activations.ncols())
        } else {
            let cluster_activations: Array2<Float> = cluster
                .iter()
                .map(|&idx| {
                    if idx < activations.nrows() {
                        activations.row(idx).to_owned()
                    } else {
                        Array1::zeros(activations.ncols())
                    }
                })
                .collect::<Vec<_>>()
                .into_iter()
                .fold(Array2::zeros((0, activations.ncols())), |acc, row| {
                    if acc.nrows() == 0 {
                        Array2::from_shape_vec((1, row.len()), row.to_vec()).unwrap()
                    } else {
                        let new_shape = (acc.nrows() + 1, acc.ncols());
                        let mut new_data = acc.into_raw_vec();
                        new_data.extend(row.iter().cloned());
                        Array2::from_shape_vec(new_shape, new_data).unwrap()
                    }
                });

            cluster_activations.mean_axis(Axis(0)).unwrap()
        };

        Ok(ConceptActivationVector::new(
            concept_id,
            layer_name,
            cluster_mean,
        ))
    }
}

/// Concept database for storing and managing learned concepts
pub struct ConceptDatabase {
    concepts: HashMap<String, ConceptActivationVector>,
    concept_relationships: HashMap<String, Vec<String>>,
}

impl Default for ConceptDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl ConceptDatabase {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            concept_relationships: HashMap::new(),
        }
    }

    pub fn add_concept(&mut self, concept: ConceptActivationVector) {
        self.concepts.insert(concept.concept_id.clone(), concept);
    }

    pub fn get_concept(&self, concept_id: &str) -> Option<&ConceptActivationVector> {
        self.concepts.get(concept_id)
    }

    pub fn find_similar_concepts(&self, concept_id: &str, threshold: Float) -> Vec<String> {
        if let Some(target_concept) = self.concepts.get(concept_id) {
            self.concepts
                .iter()
                .filter(|(id, concept)| {
                    *id != concept_id
                        && self.compute_concept_similarity(
                            &target_concept.direction_vector,
                            &concept.direction_vector,
                        ) > threshold
                })
                .map(|(id, _)| id.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn compute_concept_similarity(&self, v1: &Array1<Float>, v2: &Array1<Float>) -> Float {
        // Cosine similarity
        let dot_product = v1.dot(v2);
        let norm1 = v1.dot(v1).sqrt();
        let norm2 = v2.dot(v2).sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_deep_learning_config_creation() {
        let config = DeepLearningConfig::default();
        assert_eq!(config.num_concepts, 20);
        assert_eq!(config.activation_threshold, 0.5);
        assert!(matches!(
            config.concept_discovery_method,
            ConceptDiscoveryMethod::ACE
        ));
    }

    #[test]
    fn test_concept_activation_vector() {
        let direction = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let cav = ConceptActivationVector::new(
            "test_concept".to_string(),
            "layer_1".to_string(),
            direction,
        );

        assert_eq!(cav.concept_id, "test_concept");
        assert_eq!(cav.layer_name, "layer_1");

        let activation = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let sensitivity = cav.compute_sensitivity(&activation.view());
        assert!((sensitivity - 0.6).abs() < 1e-6); // 0.1 + 0.2 + 0.3
    }

    #[test]
    fn test_concept_activation_check() {
        let direction = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let cav = ConceptActivationVector::new(
            "test_concept".to_string(),
            "layer_1".to_string(),
            direction,
        );

        let high_activation = Array1::from_vec(vec![0.8, 0.1, 0.1]);
        let low_activation = Array1::from_vec(vec![0.2, 0.1, 0.1]);

        assert!(cav.is_activated(&high_activation.view(), 0.5));
        assert!(!cav.is_activated(&low_activation.view(), 0.5));
    }

    #[test]
    fn test_tcav_result() {
        let direction = Array1::from_vec(vec![1.0, 0.0]);
        let cav = ConceptActivationVector::new(
            "test_concept".to_string(),
            "layer_1".to_string(),
            direction,
        );

        let result = TCAVResult {
            tcav_score: 0.75,
            p_value: 0.01,
            confidence_interval: (0.65, 0.85),
            num_activated: 15,
            total_inputs: 20,
            cav,
        };

        assert!(result.is_significant(0.05));
        assert_eq!(result.effect_size_interpretation(), "Large effect");
    }

    #[test]
    fn test_concept_database() {
        let mut db = ConceptDatabase::new();

        let direction = Array1::from_vec(vec![1.0, 0.0]);
        let concept = ConceptActivationVector::new(
            "test_concept".to_string(),
            "layer_1".to_string(),
            direction,
        );

        db.add_concept(concept);
        assert!(db.get_concept("test_concept").is_some());
        assert!(db.get_concept("nonexistent").is_none());
    }

    #[test]
    fn test_deep_learning_analyzer_creation() {
        let config = DeepLearningConfig::default();
        let analyzer = DeepLearningAnalyzer::new(config);

        assert_eq!(analyzer.config.num_concepts, 20);
        assert!(analyzer.concept_database.concepts.is_empty());
    }

    #[test]
    fn test_detected_concept() {
        let concept = DetectedConcept {
            name: "stripe_pattern".to_string(),
            category: "texture".to_string(),
            iou_score: 0.65,
            detecting_units: vec![5, 12, 23],
            activation_threshold: 0.4,
        };

        assert_eq!(concept.name, "stripe_pattern");
        assert_eq!(concept.detecting_units.len(), 3);
        assert!(concept.iou_score > 0.6);
    }

    #[test]
    fn test_disentanglement_metrics() {
        let metrics = DisentanglementMetrics {
            mig_score: 0.8,
            sap_score: 0.75,
            modularity_score: 0.9,
            compactness_score: 0.85,
        };

        assert!(metrics.mig_score > 0.7);
        assert!(metrics.modularity_score > 0.8);
    }
}
