//! Interpretability and explainability tools for neighbor-based algorithms
//!
//! This module provides comprehensive tools for understanding and explaining
//! neighbor-based model decisions, including neighbor explanations, feature
//! importance analysis, prototype identification, and influence analysis.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::collections::HashMap;

use crate::distance::Distance;
use crate::knn::KNeighborsClassifier;
use crate::NeighborsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Float, Int};

/// Explanation for a single prediction showing neighbor contributions
#[derive(Debug, Clone)]
pub struct NeighborExplanation {
    /// Index of the sample being explained
    pub sample_index: usize,
    /// The prediction made by the model
    pub prediction: Int,
    /// Confidence/probability of the prediction
    pub confidence: Float,
    /// Indices of the k-nearest neighbors
    pub neighbor_indices: Vec<usize>,
    /// Distances to each neighbor
    pub neighbor_distances: Vec<Float>,
    /// Labels of the neighbors
    pub neighbor_labels: Vec<Int>,
    /// Contribution weights of each neighbor to the final prediction
    pub neighbor_weights: Vec<Float>,
    /// Feature importance scores for this specific prediction
    pub feature_importance: Array1<Float>,
    /// Most influential neighbor (closest or highest weight)
    pub most_influential_neighbor: usize,
}

/// Local feature importance explanation
#[derive(Debug, Clone)]
pub struct LocalImportanceExplanation {
    /// Feature importance scores (higher means more important)
    pub feature_scores: Array1<Float>,
    /// Rank of features (0 = most important)
    pub feature_ranks: Vec<usize>,
    /// Names or descriptions of features (optional)
    pub feature_names: Option<Vec<String>>,
    /// The sample being explained
    pub sample: Array1<Float>,
    /// Reference point used for comparison (e.g., class centroid)
    pub reference_point: Option<Array1<Float>>,
}

/// Prototype (representative example) for a class
#[derive(Debug, Clone)]
pub struct Prototype {
    /// Index in the training set
    pub index: usize,
    /// The prototype sample
    pub sample: Array1<Float>,
    /// Class label
    pub class_label: Int,
    /// Representativeness score (higher = more representative)
    pub score: Float,
    /// Average distance to other samples in the same class
    pub avg_intra_class_distance: Float,
    /// Average distance to samples in other classes
    pub avg_inter_class_distance: Float,
    /// Number of samples for which this is the nearest neighbor
    pub influence_count: usize,
}

/// Neighbor influence analysis result
#[derive(Debug, Clone)]
pub struct InfluenceAnalysis {
    /// Sample index being analyzed
    pub sample_index: usize,
    /// Influence score for each training sample
    pub influence_scores: Vec<Float>,
    /// Indices of most influential samples (sorted by influence)
    pub most_influential: Vec<usize>,
    /// Indices of least influential samples
    pub least_influential: Vec<usize>,
    /// Change in prediction if most influential sample is removed
    pub prediction_stability: Float,
}

/// Neighbor-based model explainer
pub struct NeighborExplainer {
    /// The fitted KNN classifier to explain
    classifier: Option<KNeighborsClassifier<sklears_core::traits::Trained>>,
    /// Training data
    X_train: Array2<Float>,
    /// Training labels
    y_train: Array1<Int>,
    /// Distance metric used
    distance_metric: Distance,
    /// Number of neighbors to consider
    k: usize,
    /// Cached prototypes for each class
    prototypes: Option<HashMap<Int, Vec<Prototype>>>,
}

impl NeighborExplainer {
    /// Create a new explainer for a fitted KNN classifier
    pub fn new(
        classifier: KNeighborsClassifier<sklears_core::traits::Trained>,
        X_train: Array2<Float>,
        y_train: Array1<Int>,
    ) -> Result<Self, NeighborsError> {
        if X_train.nrows() != y_train.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![X_train.nrows()],
                actual: vec![y_train.len()],
            });
        }

        Ok(Self {
            distance_metric: classifier.metric.clone(),
            k: classifier.n_neighbors,
            classifier: Some(classifier),
            X_train,
            y_train,
            prototypes: None,
        })
    }

    /// Create an explainer from raw data (will fit a new classifier)
    pub fn from_data(
        X_train: Array2<Float>,
        y_train: Array1<Int>,
        k: usize,
        distance_metric: Distance,
    ) -> Result<Self, NeighborsError> {
        if X_train.nrows() != y_train.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![X_train.nrows()],
                actual: vec![y_train.len()],
            });
        }

        let classifier = KNeighborsClassifier::new(k)
            .with_metric(distance_metric.clone())
            .fit(&X_train, &y_train)?;

        Ok(Self {
            distance_metric,
            k,
            classifier: Some(classifier),
            X_train,
            y_train,
            prototypes: None,
        })
    }

    /// Explain a single prediction
    pub fn explain_prediction(
        &self,
        sample: ArrayView1<Float>,
    ) -> Result<NeighborExplanation, NeighborsError> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| NeighborsError::InvalidInput("No classifier fitted".to_string()))?;

        // Find k-nearest neighbors
        let (neighbor_indices, neighbor_distances) = self.find_neighbors(sample)?;

        // Get neighbor labels
        let neighbor_labels: Vec<Int> = neighbor_indices
            .iter()
            .map(|&idx| self.y_train[idx])
            .collect();

        // Calculate neighbor weights based on distance
        let neighbor_weights = self.calculate_neighbor_weights(&neighbor_distances)?;

        // Make prediction
        let prediction = classifier.predict(&sample.to_owned().insert_axis(Axis(0)))?[0];

        // Calculate confidence based on neighbor agreement
        let confidence =
            self.calculate_prediction_confidence(&neighbor_labels, prediction, &neighbor_weights);

        // Calculate feature importance for this specific prediction
        let feature_importance =
            self.calculate_local_feature_importance(sample, &neighbor_indices)?;

        // Find most influential neighbor
        let most_influential_neighbor = neighbor_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| neighbor_indices[idx])
            .unwrap_or(0);

        Ok(NeighborExplanation {
            sample_index: 0, // Will be set by caller if needed
            prediction,
            confidence,
            neighbor_indices,
            neighbor_distances,
            neighbor_labels,
            neighbor_weights,
            feature_importance,
            most_influential_neighbor,
        })
    }

    /// Explain multiple predictions at once
    pub fn explain_predictions(
        &self,
        X: ArrayView2<Float>,
    ) -> Result<Vec<NeighborExplanation>, NeighborsError> {
        let mut explanations = Vec::with_capacity(X.nrows());

        for (i, sample) in X.outer_iter().enumerate() {
            let mut explanation = self.explain_prediction(sample)?;
            explanation.sample_index = i;
            explanations.push(explanation);
        }

        Ok(explanations)
    }

    /// Calculate local feature importance using permutation-based approach
    pub fn explain_local_importance(
        &self,
        sample: ArrayView1<Float>,
    ) -> Result<LocalImportanceExplanation, NeighborsError> {
        let n_features = sample.len();
        let mut feature_scores = Array1::zeros(n_features);

        // Get baseline prediction
        let baseline_neighbors = self.find_neighbors(sample)?;
        let baseline_distances = baseline_neighbors.1;
        let baseline_score = baseline_distances.iter().sum::<Float>();

        // Test each feature by permuting it
        let mut sample_copy = sample.to_owned();
        for feature_idx in 0..n_features {
            // Calculate feature importance by measuring how much the neighbor
            // distances change when we set this feature to the mean value
            let original_value = sample_copy[feature_idx];
            let mean_value = self.X_train.column(feature_idx).mean().unwrap_or(0.0);

            sample_copy[feature_idx] = mean_value;

            let perturbed_neighbors = self.find_neighbors(sample_copy.view())?;
            let perturbed_distances = perturbed_neighbors.1;
            let perturbed_score = perturbed_distances.iter().sum::<Float>();

            // Higher change means more important feature
            feature_scores[feature_idx] = (perturbed_score - baseline_score).abs();

            // Restore original value
            sample_copy[feature_idx] = original_value;
        }

        // Normalize scores
        let max_score = feature_scores.iter().cloned().fold(0.0f64, f64::max) as Float;
        if max_score > 0.0 {
            feature_scores /= max_score;
        }

        // Create feature ranks
        let mut feature_ranks: Vec<usize> = (0..n_features).collect();
        feature_ranks.sort_by(|&a, &b| feature_scores[b].partial_cmp(&feature_scores[a]).unwrap());

        Ok(LocalImportanceExplanation {
            feature_scores,
            feature_ranks,
            feature_names: None,
            sample: sample.to_owned(),
            reference_point: None,
        })
    }

    /// Identify prototypes (most representative examples) for each class
    pub fn identify_prototypes(
        &mut self,
        n_prototypes_per_class: usize,
    ) -> Result<HashMap<Int, Vec<Prototype>>, NeighborsError> {
        let mut class_prototypes: HashMap<Int, Vec<Prototype>> = HashMap::new();

        // Get unique classes
        let unique_classes: std::collections::HashSet<Int> = self.y_train.iter().cloned().collect();

        for &class_label in &unique_classes {
            let class_indices: Vec<usize> = self
                .y_train
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            if class_indices.is_empty() {
                continue;
            }

            let mut candidates: Vec<Prototype> = Vec::new();

            for &sample_idx in &class_indices {
                let sample = self.X_train.row(sample_idx);

                // Calculate representativeness scores
                let (avg_intra, avg_inter, influence_count) = self
                    .calculate_representativeness_scores(sample_idx, class_label, &class_indices)?;

                // Higher intra-class distance is worse, higher inter-class distance is better
                // More influence (being nearest neighbor to others) is better
                let score = (avg_inter - avg_intra) + (influence_count as Float * 0.1);

                candidates.push(Prototype {
                    index: sample_idx,
                    sample: sample.to_owned(),
                    class_label,
                    score,
                    avg_intra_class_distance: avg_intra,
                    avg_inter_class_distance: avg_inter,
                    influence_count,
                });
            }

            // Sort by score and take top n
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            candidates.truncate(n_prototypes_per_class);

            class_prototypes.insert(class_label, candidates);
        }

        self.prototypes = Some(class_prototypes.clone());
        Ok(class_prototypes)
    }

    /// Analyze the influence of training samples on a prediction
    pub fn analyze_influence(
        &self,
        sample: ArrayView1<Float>,
    ) -> Result<InfluenceAnalysis, NeighborsError> {
        let n_train = self.X_train.nrows();
        let mut influence_scores = vec![0.0; n_train];

        // Get baseline prediction
        let baseline_explanation = self.explain_prediction(sample)?;
        let baseline_prediction = baseline_explanation.prediction;

        // Calculate influence by leave-one-out approach
        for train_idx in 0..n_train {
            // Create training set without this sample
            let mut X_reduced = Array2::zeros((n_train - 1, self.X_train.ncols()));
            let mut y_reduced = Array1::zeros(n_train - 1);

            let mut reduced_idx = 0;
            for i in 0..n_train {
                if i != train_idx {
                    X_reduced.row_mut(reduced_idx).assign(&self.X_train.row(i));
                    y_reduced[reduced_idx] = self.y_train[i];
                    reduced_idx += 1;
                }
            }

            // Fit new classifier without this sample
            let temp_classifier = KNeighborsClassifier::new(self.k)
                .with_metric(self.distance_metric.clone())
                .fit(&X_reduced, &y_reduced)?;

            // Make prediction
            let new_prediction =
                temp_classifier.predict(&sample.to_owned().insert_axis(Axis(0)))?[0];

            // Calculate influence as change in prediction confidence
            influence_scores[train_idx] = if new_prediction != baseline_prediction {
                1.0
            } else {
                0.0
            };
        }

        // Find most and least influential samples
        let mut indexed_scores: Vec<(usize, Float)> = influence_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let most_influential: Vec<usize> = indexed_scores
            .iter()
            .take(10)
            .map(|&(idx, _)| idx)
            .collect();

        let least_influential: Vec<usize> = indexed_scores
            .iter()
            .skip(indexed_scores.len().saturating_sub(10))
            .map(|&(idx, _)| idx)
            .collect();

        // Calculate prediction stability
        let stability = influence_scores.iter().sum::<Float>() / n_train as Float;

        Ok(InfluenceAnalysis {
            sample_index: 0,
            influence_scores,
            most_influential,
            least_influential,
            prediction_stability: 1.0 - stability,
        })
    }

    /// Get cached prototypes if available
    pub fn get_prototypes(&self) -> Option<&HashMap<Int, Vec<Prototype>>> {
        self.prototypes.as_ref()
    }

    /// Generate a comprehensive interpretability report
    pub fn generate_report(&mut self, sample: ArrayView1<Float>) -> Result<String, NeighborsError> {
        let mut report = String::from("# Neighbor-Based Model Explanation Report\n\n");

        // 1. Basic prediction explanation
        let explanation = self.explain_prediction(sample)?;
        report.push_str("## Prediction Summary\n\n");
        report.push_str(&format!(
            "**Predicted Class:** {}\n",
            explanation.prediction
        ));
        report.push_str(&format!("**Confidence:** {:.3}\n", explanation.confidence));
        report.push_str(&format!(
            "**Most Influential Neighbor:** Sample #{}\n\n",
            explanation.most_influential_neighbor
        ));

        // 2. Neighbor details
        report.push_str("## Nearest Neighbors Analysis\n\n");
        report.push_str("| Rank | Sample # | Distance | Label | Weight | Contribution |\n");
        report.push_str("|------|----------|----------|-------|--------|--------------|\n");

        for (rank, (((idx, dist), label), weight)) in explanation
            .neighbor_indices
            .iter()
            .zip(explanation.neighbor_distances.iter())
            .zip(explanation.neighbor_labels.iter())
            .zip(explanation.neighbor_weights.iter())
            .enumerate()
        {
            let contribution = if *label == explanation.prediction {
                "✓"
            } else {
                "✗"
            };
            report.push_str(&format!(
                "| {} | {} | {:.4} | {} | {:.3} | {} |\n",
                rank + 1,
                idx,
                dist,
                label,
                weight,
                contribution
            ));
        }

        // 3. Feature importance
        report.push_str("\n## Local Feature Importance\n\n");
        let importance = self.explain_local_importance(sample)?;
        report.push_str("| Rank | Feature | Importance Score |\n");
        report.push_str("|------|---------|------------------|\n");

        for (rank, &feature_idx) in importance.feature_ranks.iter().take(10).enumerate() {
            let feature_name = if let Some(names) = &importance.feature_names {
                if let Some(name) = names.get(feature_idx) {
                    name.clone()
                } else {
                    format!("Feature_{}", feature_idx)
                }
            } else {
                format!("Feature_{}", feature_idx)
            };

            report.push_str(&format!(
                "| {} | {} | {:.4} |\n",
                rank + 1,
                feature_name,
                importance.feature_scores[feature_idx]
            ));
        }

        // 4. Prototypes (if available)
        if self.prototypes.is_none() {
            self.identify_prototypes(3)?;
        }

        if let Some(prototypes) = &self.prototypes {
            report.push_str("\n## Class Prototypes\n\n");
            for (&class_label, class_prototypes) in prototypes.iter() {
                report.push_str(&format!("### Class {}\n\n", class_label));
                report.push_str("| Rank | Sample # | Score | Intra Distance | Inter Distance |\n");
                report.push_str("|------|----------|-------|----------------|----------------|\n");

                for (rank, prototype) in class_prototypes.iter().enumerate() {
                    report.push_str(&format!(
                        "| {} | {} | {:.3} | {:.4} | {:.4} |\n",
                        rank + 1,
                        prototype.index,
                        prototype.score,
                        prototype.avg_intra_class_distance,
                        prototype.avg_inter_class_distance
                    ));
                }
                report.push('\n');
            }
        }

        Ok(report)
    }

    // Helper methods

    fn find_neighbors(
        &self,
        sample: ArrayView1<Float>,
    ) -> Result<(Vec<usize>, Vec<Float>), NeighborsError> {
        let mut distances_with_indices: Vec<(usize, Float)> =
            Vec::with_capacity(self.X_train.nrows());

        for (idx, train_sample) in self.X_train.outer_iter().enumerate() {
            let distance = self
                .distance_metric
                .calculate(&sample, &train_sample.view());
            distances_with_indices.push((idx, distance));
        }

        // Sort by distance and take k nearest
        distances_with_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances_with_indices.truncate(self.k);

        let indices: Vec<usize> = distances_with_indices.iter().map(|&(idx, _)| idx).collect();
        let distances: Vec<Float> = distances_with_indices
            .iter()
            .map(|&(_, dist)| dist)
            .collect();

        Ok((indices, distances))
    }

    fn calculate_neighbor_weights(
        &self,
        distances: &[Float],
    ) -> Result<Vec<Float>, NeighborsError> {
        if distances.is_empty() {
            return Ok(vec![]);
        }

        // Use inverse distance weighting (with small epsilon to avoid division by zero)
        let epsilon = 1e-10;
        let weights: Vec<Float> = distances.iter().map(|&d| 1.0 / (d + epsilon)).collect();

        // Normalize weights
        let sum: Float = weights.iter().sum();
        if sum > 0.0 {
            Ok(weights.iter().map(|w| w / sum).collect())
        } else {
            Ok(vec![1.0 / weights.len() as Float; weights.len()])
        }
    }

    fn calculate_prediction_confidence(
        &self,
        neighbor_labels: &[Int],
        prediction: Int,
        weights: &[Float],
    ) -> Float {
        if neighbor_labels.is_empty() || weights.is_empty() {
            return 0.0;
        }

        // Sum weights of neighbors that agree with prediction
        let agreeing_weight: Float = neighbor_labels
            .iter()
            .zip(weights.iter())
            .filter(|(&label, _)| label == prediction)
            .map(|(_, &weight)| weight)
            .sum();

        agreeing_weight
    }

    fn calculate_local_feature_importance(
        &self,
        _sample: ArrayView1<Float>,
        neighbor_indices: &[usize],
    ) -> Result<Array1<Float>, NeighborsError> {
        let n_features = self.X_train.ncols();
        let mut importance = Array1::zeros(n_features);

        if neighbor_indices.is_empty() {
            return Ok(importance);
        }

        // Calculate variance of each feature among neighbors
        // Higher variance suggests more discriminative power
        for feature_idx in 0..n_features {
            let feature_values: Vec<Float> = neighbor_indices
                .iter()
                .map(|&idx| self.X_train[(idx, feature_idx)])
                .collect();

            if feature_values.len() > 1 {
                let mean = feature_values.iter().sum::<Float>() / feature_values.len() as Float;
                let variance = feature_values
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<Float>()
                    / (feature_values.len() - 1) as Float;
                importance[feature_idx] = variance.sqrt(); // Use std dev as importance
            }
        }

        // Normalize
        let max_importance = importance.iter().cloned().fold(0.0f64, f64::max) as Float;
        if max_importance > 0.0 {
            importance /= max_importance;
        }

        Ok(importance)
    }

    fn calculate_representativeness_scores(
        &self,
        sample_idx: usize,
        class_label: Int,
        class_indices: &[usize],
    ) -> Result<(Float, Float, usize), NeighborsError> {
        let sample = self.X_train.row(sample_idx);

        // Calculate average intra-class distance
        let intra_distances: Vec<Float> = class_indices
            .iter()
            .filter(|&&idx| idx != sample_idx)
            .map(|&idx| {
                let other_sample = self.X_train.row(idx);
                self.distance_metric.calculate(&sample, &other_sample)
            })
            .collect();

        let avg_intra = if !intra_distances.is_empty() {
            intra_distances.iter().sum::<Float>() / intra_distances.len() as Float
        } else {
            0.0
        };

        // Calculate average inter-class distance
        let inter_distances: Vec<Float> = self
            .y_train
            .iter()
            .enumerate()
            .filter(|(idx, &label)| *idx != sample_idx && label != class_label)
            .map(|(idx, _)| {
                let other_sample = self.X_train.row(idx);
                self.distance_metric.calculate(&sample, &other_sample)
            })
            .collect();

        let avg_inter = if !inter_distances.is_empty() {
            inter_distances.iter().sum::<Float>() / inter_distances.len() as Float
        } else {
            0.0
        };

        // Calculate influence count (how many samples have this as nearest neighbor)
        let mut influence_count = 0;
        for (other_idx, other_sample) in self.X_train.outer_iter().enumerate() {
            if other_idx == sample_idx {
                continue;
            }

            let (nearest_neighbors, _) = self.find_neighbors_for_sample(other_sample, 1)?;
            if !nearest_neighbors.is_empty() && nearest_neighbors[0] == sample_idx {
                influence_count += 1;
            }
        }

        Ok((avg_intra, avg_inter, influence_count))
    }

    fn find_neighbors_for_sample(
        &self,
        sample: ArrayView1<Float>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<Float>), NeighborsError> {
        let mut distances_with_indices: Vec<(usize, Float)> =
            Vec::with_capacity(self.X_train.nrows());

        for (idx, train_sample) in self.X_train.outer_iter().enumerate() {
            let distance = self
                .distance_metric
                .calculate(&sample, &train_sample.view());
            distances_with_indices.push((idx, distance));
        }

        // Sort by distance and take k nearest
        distances_with_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances_with_indices.truncate(k);

        let indices: Vec<usize> = distances_with_indices.iter().map(|&(idx, _)| idx).collect();
        let distances: Vec<Float> = distances_with_indices
            .iter()
            .map(|&(_, dist)| dist)
            .collect();

        Ok((indices, distances))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[allow(non_snake_case)]
    fn create_test_data() -> (Array2<Float>, Array1<Int>) {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1, 9.0, 9.0, 9.1, 9.1],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        (X, y)
    }

    #[test]
    fn test_explainer_creation() {
        let (X, y) = create_test_data();
        let result = NeighborExplainer::from_data(X, y, 3, Distance::Euclidean);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prediction_explanation() {
        let (X, y) = create_test_data();
        let explainer = NeighborExplainer::from_data(X.clone(), y, 3, Distance::Euclidean).unwrap();

        let test_sample = Array1::from_vec(vec![1.05, 1.05]);
        let explanation = explainer.explain_prediction(test_sample.view());

        assert!(explanation.is_ok());
        let explanation = explanation.unwrap();
        assert_eq!(explanation.prediction, 0); // Should predict class 0
        assert!(explanation.confidence > 0.0);
        assert_eq!(explanation.neighbor_indices.len(), 3);
    }

    #[test]
    fn test_local_importance() {
        let (X, y) = create_test_data();
        let explainer = NeighborExplainer::from_data(X, y, 3, Distance::Euclidean).unwrap();

        let test_sample = Array1::from_vec(vec![1.05, 1.05]);
        let importance = explainer.explain_local_importance(test_sample.view());

        assert!(importance.is_ok());
        let importance = importance.unwrap();
        assert_eq!(importance.feature_scores.len(), 2);
        assert_eq!(importance.feature_ranks.len(), 2);
    }

    #[test]
    fn test_prototype_identification() {
        let (X, y) = create_test_data();
        let mut explainer = NeighborExplainer::from_data(X, y, 3, Distance::Euclidean).unwrap();

        let prototypes = explainer.identify_prototypes(1);
        assert!(prototypes.is_ok());

        let prototypes = prototypes.unwrap();
        assert_eq!(prototypes.len(), 3); // Three classes

        for (&class_label, class_prototypes) in prototypes.iter() {
            assert!((0..=2).contains(&class_label));
            assert_eq!(class_prototypes.len(), 1); // Requested 1 prototype per class
        }
    }

    #[test]
    fn test_influence_analysis() {
        let (X, y) = create_test_data();
        let explainer = NeighborExplainer::from_data(X, y, 3, Distance::Euclidean).unwrap();

        let test_sample = Array1::from_vec(vec![1.05, 1.05]);
        let influence = explainer.analyze_influence(test_sample.view());

        assert!(influence.is_ok());
        let influence = influence.unwrap();
        assert_eq!(influence.influence_scores.len(), 6); // Training set size
        assert!(!influence.most_influential.is_empty());
        assert!(influence.prediction_stability >= 0.0 && influence.prediction_stability <= 1.0);
    }

    #[test]
    fn test_multiple_explanations() {
        let (X, y) = create_test_data();
        let explainer = NeighborExplainer::from_data(X.clone(), y, 3, Distance::Euclidean).unwrap();

        let test_X = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 5.05, 5.05]).unwrap();

        let explanations = explainer.explain_predictions(test_X.view());
        assert!(explanations.is_ok());

        let explanations = explanations.unwrap();
        assert_eq!(explanations.len(), 2);
        assert_eq!(explanations[0].sample_index, 0);
        assert_eq!(explanations[1].sample_index, 1);
    }

    #[test]
    fn test_report_generation() {
        let (X, y) = create_test_data();
        let mut explainer = NeighborExplainer::from_data(X, y, 3, Distance::Euclidean).unwrap();

        let test_sample = Array1::from_vec(vec![1.05, 1.05]);
        let report = explainer.generate_report(test_sample.view());

        assert!(report.is_ok());
        let report = report.unwrap();
        assert!(report.contains("Prediction Summary"));
        assert!(report.contains("Nearest Neighbors Analysis"));
        assert!(report.contains("Local Feature Importance"));
        assert!(report.contains("Class Prototypes"));
    }

    #[test]
    fn test_neighbor_weights_calculation() {
        let (X, y) = create_test_data();
        let explainer = NeighborExplainer::from_data(X, y, 3, Distance::Euclidean).unwrap();

        let distances = vec![0.1, 0.2, 0.3];
        let weights = explainer.calculate_neighbor_weights(&distances);

        assert!(weights.is_ok());
        let weights = weights.unwrap();
        assert_eq!(weights.len(), 3);

        // Check normalization
        let sum: Float = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check inverse relationship (smaller distance = higher weight)
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }
}
