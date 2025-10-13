//! SHAP (SHapley Additive exPlanations) for tree interpretability
//!
//! This module provides SHAP value computation for tree-based models,
//! offering explanations for individual predictions by quantifying
//! the contribution of each feature.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result, SklearsError};

/// SHAP explanation for a single instance
#[derive(Debug, Clone)]
pub struct ShapExplanation {
    /// SHAP values for each feature
    pub values: Array1<f64>,
    /// Base value (expected model output)
    pub base_value: f64,
    /// Input feature values
    pub feature_values: Array1<f64>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Prediction from the model
    pub prediction: f64,
}

/// SHAP explanation for multiple instances
#[derive(Debug, Clone)]
pub struct ShapExplanations {
    /// SHAP values for each instance and feature
    pub values: Array2<f64>,
    /// Base values for each instance
    pub base_values: Array1<f64>,
    /// Input feature values for all instances
    pub feature_values: Array2<f64>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Predictions from the model
    pub predictions: Array1<f64>,
}

/// Configuration for SHAP computation
#[derive(Debug, Clone)]
pub struct ShapConfig {
    /// Check that SHAP values sum to prediction difference
    pub check_additivity: bool,
    /// Maximum tree depth to traverse for SHAP computation
    pub max_depth: Option<usize>,
    /// Approximate SHAP values for faster computation
    pub approximate: bool,
    /// Sample size for approximation (if approximate=true)
    pub sample_size: Option<usize>,
}

impl Default for ShapConfig {
    fn default() -> Self {
        Self {
            check_additivity: true,
            max_depth: None,
            approximate: false,
            sample_size: None,
        }
    }
}

/// Internal representation of a tree node for SHAP computation
#[derive(Debug, Clone)]
struct ShapTreeNode {
    /// Feature index for split (-1 for leaf)
    feature: i32,
    /// Threshold value for split
    threshold: f64,
    /// Left child index
    left_child: Option<usize>,
    /// Right child index  
    right_child: Option<usize>,
    /// Node value (for leaves)
    value: f64,
    /// Number of samples in this node
    node_sample_weight: f64,
}

/// TreeSHAP explainer for single trees
pub struct TreeShapExplainer {
    /// Tree structure converted for SHAP computation
    nodes: Vec<ShapTreeNode>,
    /// Number of features
    n_features: usize,
    /// Expected value from training data
    expected_value: f64,
    /// Configuration
    config: ShapConfig,
}

impl TreeShapExplainer {
    /// Create a new TreeSHAP explainer from a decision tree classifier (placeholder)
    pub fn from_classifier(n_features: usize, config: ShapConfig) -> Result<Self> {
        // Placeholder implementation - in real implementation, would extract tree structure
        let nodes = vec![ShapTreeNode {
            feature: -1,
            threshold: 0.0,
            left_child: None,
            right_child: None,
            value: 0.0,
            node_sample_weight: 1.0,
        }];
        let expected_value = 0.0;

        Ok(Self {
            nodes,
            n_features,
            expected_value,
            config,
        })
    }

    /// Create a new TreeSHAP explainer from a decision tree regressor (placeholder)
    pub fn from_regressor(n_features: usize, config: ShapConfig) -> Result<Self> {
        // Placeholder implementation - in real implementation, would extract tree structure
        let nodes = vec![ShapTreeNode {
            feature: -1,
            threshold: 0.0,
            left_child: None,
            right_child: None,
            value: 0.0,
            node_sample_weight: 1.0,
        }];
        let expected_value = 0.0;

        Ok(Self {
            nodes,
            n_features,
            expected_value,
            config,
        })
    }

    /// Explain a single instance
    pub fn explain_instance(&self, features: &Array1<f64>) -> Result<ShapExplanation> {
        if features.len() != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features,
                features.len()
            )));
        }

        let mut shap_values = Array1::zeros(self.n_features);
        let prediction = self.tree_predict(features);

        // Compute SHAP values using TreeSHAP algorithm
        self.compute_shap_values(features, &mut shap_values, 0, 1.0, 1.0, -1)?;

        // Verify additivity if requested
        if self.config.check_additivity {
            let sum_shap = shap_values.sum() + self.expected_value;
            let diff = (sum_shap - prediction).abs();
            if diff > 1e-6 {
                return Err(SklearsError::InvalidInput(format!(
                    "SHAP values do not sum to prediction difference: {} vs {}",
                    sum_shap, prediction
                )));
            }
        }

        Ok(ShapExplanation {
            values: shap_values,
            base_value: self.expected_value,
            feature_values: features.clone(),
            feature_names: None,
            prediction,
        })
    }

    /// Explain multiple instances
    pub fn explain_instances(&self, features: &Array2<f64>) -> Result<ShapExplanations> {
        let n_instances = features.nrows();
        let mut all_shap_values = Array2::zeros((n_instances, self.n_features));
        let mut predictions = Array1::zeros(n_instances);
        let base_values = Array1::from_elem(n_instances, self.expected_value);

        for i in 0..n_instances {
            let instance = features.row(i);
            let explanation = self.explain_instance(&instance.to_owned())?;
            all_shap_values.row_mut(i).assign(&explanation.values);
            predictions[i] = explanation.prediction;
        }

        Ok(ShapExplanations {
            values: all_shap_values,
            base_values,
            feature_values: features.clone(),
            feature_names: None,
            predictions,
        })
    }

    /// Recursive TreeSHAP computation
    fn compute_shap_values(
        &self,
        features: &Array1<f64>,
        shap_values: &mut Array1<f64>,
        node_idx: usize,
        parent_fraction_zero: f64,
        parent_fraction_one: f64,
        parent_feature_idx: i32,
    ) -> Result<()> {
        if node_idx >= self.nodes.len() {
            return Ok(());
        }

        let node = &self.nodes[node_idx];

        // Leaf node - distribute the value
        if node.feature == -1 {
            if parent_feature_idx >= 0 {
                let idx = parent_feature_idx as usize;
                if idx < self.n_features {
                    shap_values[idx] += (parent_fraction_one - parent_fraction_zero) * node.value;
                }
            }
            return Ok(());
        }

        // Internal node - recurse to children
        if let (Some(left_idx), Some(right_idx)) = (node.left_child, node.right_child) {
            let feature_idx = node.feature as usize;
            let goes_left = if feature_idx < features.len() {
                features[feature_idx] <= node.threshold
            } else {
                true // Default to left if feature is missing
            };

            let left_node = &self.nodes[left_idx];
            let right_node = &self.nodes[right_idx];

            // Compute fractions for left and right children
            let total_weight = left_node.node_sample_weight + right_node.node_sample_weight;
            let left_fraction = if total_weight > 0.0 {
                left_node.node_sample_weight / total_weight
            } else {
                0.5
            };
            let right_fraction = 1.0 - left_fraction;

            if goes_left {
                // Instance goes left
                self.compute_shap_values(
                    features,
                    shap_values,
                    left_idx,
                    parent_fraction_zero * left_fraction,
                    parent_fraction_one,
                    node.feature,
                )?;
                self.compute_shap_values(
                    features,
                    shap_values,
                    right_idx,
                    parent_fraction_zero * right_fraction,
                    parent_fraction_one * 0.0,
                    node.feature,
                )?;
            } else {
                // Instance goes right
                self.compute_shap_values(
                    features,
                    shap_values,
                    left_idx,
                    parent_fraction_zero * left_fraction,
                    parent_fraction_one * 0.0,
                    node.feature,
                )?;
                self.compute_shap_values(
                    features,
                    shap_values,
                    right_idx,
                    parent_fraction_zero * right_fraction,
                    parent_fraction_one,
                    node.feature,
                )?;
            }
        }

        Ok(())
    }

    /// Predict using the tree (simplified implementation)
    fn tree_predict(&self, features: &Array1<f64>) -> f64 {
        let mut node_idx = 0;

        loop {
            if node_idx >= self.nodes.len() {
                break;
            }

            let node = &self.nodes[node_idx];

            // Leaf node
            if node.feature == -1 {
                return node.value;
            }

            // Internal node
            let feature_idx = node.feature as usize;
            let feature_value = if feature_idx < features.len() {
                features[feature_idx]
            } else {
                0.0 // Default value for missing features
            };

            if feature_value <= node.threshold {
                if let Some(left_idx) = node.left_child {
                    node_idx = left_idx;
                } else {
                    break;
                }
            } else {
                if let Some(right_idx) = node.right_child {
                    node_idx = right_idx;
                } else {
                    break;
                }
            }
        }

        0.0 // Default return
    }
}

/// SHAP explainer for ensemble models (Random Forest, Gradient Boosting)
pub struct EnsembleShapExplainer {
    /// Individual tree explainers
    tree_explainers: Vec<TreeShapExplainer>,
    /// Weights for each tree (for gradient boosting)
    tree_weights: Array1<f64>,
    /// Expected value for the ensemble
    expected_value: f64,
    /// Configuration
    config: ShapConfig,
}

impl EnsembleShapExplainer {
    /// Create SHAP explainer for Random Forest classifier (placeholder)
    pub fn from_random_forest_classifier(
        n_features: usize,
        n_trees: usize,
        config: ShapConfig,
    ) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            tree_explainers: vec![],
            tree_weights: Array1::ones(n_trees),
            expected_value: 0.0,
            config,
        })
    }

    /// Create SHAP explainer for Random Forest regressor (placeholder)
    pub fn from_random_forest_regressor(
        n_features: usize,
        n_trees: usize,
        config: ShapConfig,
    ) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            tree_explainers: vec![],
            tree_weights: Array1::ones(n_trees),
            expected_value: 0.0,
            config,
        })
    }

    /// Create SHAP explainer for Gradient Boosting classifier (placeholder)
    pub fn from_gradient_boosting_classifier(
        n_features: usize,
        n_trees: usize,
        config: ShapConfig,
    ) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            tree_explainers: vec![],
            tree_weights: Array1::ones(n_trees),
            expected_value: 0.0,
            config,
        })
    }

    /// Create SHAP explainer for Gradient Boosting regressor (placeholder)
    pub fn from_gradient_boosting_regressor(
        n_features: usize,
        n_trees: usize,
        config: ShapConfig,
    ) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            tree_explainers: vec![],
            tree_weights: Array1::ones(n_trees),
            expected_value: 0.0,
            config,
        })
    }

    /// Explain a single instance by aggregating SHAP values from all trees
    pub fn explain_instance(&self, features: &Array1<f64>) -> Result<ShapExplanation> {
        if self.tree_explainers.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No tree explainers available".to_string(),
            ));
        }

        let n_features = features.len();
        let mut aggregated_shap = Array1::zeros(n_features);
        let mut total_prediction = 0.0;

        // Aggregate SHAP values from all trees
        for (i, explainer) in self.tree_explainers.iter().enumerate() {
            let tree_explanation = explainer.explain_instance(features)?;
            let weight = if i < self.tree_weights.len() {
                self.tree_weights[i]
            } else {
                1.0 / self.tree_explainers.len() as f64
            };

            aggregated_shap = aggregated_shap + &tree_explanation.values * weight;
            total_prediction += tree_explanation.prediction * weight;
        }

        Ok(ShapExplanation {
            values: aggregated_shap,
            base_value: self.expected_value,
            feature_values: features.clone(),
            feature_names: None,
            prediction: total_prediction,
        })
    }

    /// Explain multiple instances
    pub fn explain_instances(&self, features: &Array2<f64>) -> Result<ShapExplanations> {
        let n_instances = features.nrows();
        let n_features = features.ncols();
        let mut all_shap_values = Array2::zeros((n_instances, n_features));
        let mut predictions = Array1::zeros(n_instances);
        let base_values = Array1::from_elem(n_instances, self.expected_value);

        for i in 0..n_instances {
            let instance = features.row(i);
            let explanation = self.explain_instance(&instance.to_owned())?;
            all_shap_values.row_mut(i).assign(&explanation.values);
            predictions[i] = explanation.prediction;
        }

        Ok(ShapExplanations {
            values: all_shap_values,
            base_values,
            feature_values: features.clone(),
            feature_names: None,
            predictions,
        })
    }
}

/// Utility functions for SHAP analysis
pub mod utils {
    use super::*;

    /// Compute global feature importance from SHAP values
    pub fn global_feature_importance(explanations: &ShapExplanations) -> Array1<f64> {
        explanations.values.map_axis(Axis(0), |shap_vals| {
            shap_vals.iter().map(|&x| x.abs()).sum::<f64>() / shap_vals.len() as f64
        })
    }

    /// Find top contributing features for an instance
    pub fn top_features(explanation: &ShapExplanation, n_top: usize) -> Vec<(usize, f64)> {
        let mut feature_impacts: Vec<(usize, f64)> = explanation
            .values
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs()))
            .collect();

        feature_impacts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        feature_impacts.into_iter().take(n_top).collect()
    }

    /// Compute SHAP interaction values (placeholder)
    pub fn interaction_values(explanation: &ShapExplanation) -> Result<Array2<f64>> {
        let n_features = explanation.values.len();
        // Placeholder implementation - return zeros
        Ok(Array2::zeros((n_features, n_features)))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_shap_explanation_creation() {
        let shap_values = array![0.1, -0.2, 0.3];
        let feature_values = array![1.0, 2.0, 3.0];

        let explanation = ShapExplanation {
            values: shap_values.clone(),
            base_value: 0.5,
            feature_values: feature_values.clone(),
            feature_names: None,
            prediction: 0.7,
        };

        assert_eq!(explanation.values.len(), 3);
        assert_eq!(explanation.base_value, 0.5);
        assert_eq!(explanation.prediction, 0.7);
    }

    #[test]
    fn test_shap_config_default() {
        let config = ShapConfig::default();
        assert!(config.check_additivity);
        assert!(!config.approximate);
        assert_eq!(config.max_depth, None);
        assert_eq!(config.sample_size, None);
    }

    #[test]
    fn test_top_features_utility() {
        let shap_values = array![0.1, -0.5, 0.3, -0.2];
        let feature_values = array![1.0, 2.0, 3.0, 4.0];

        let explanation = ShapExplanation {
            values: shap_values,
            base_value: 0.0,
            feature_values,
            feature_names: None,
            prediction: 0.0,
        };

        let top_2 = utils::top_features(&explanation, 2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].0, 1); // Feature 1 has highest absolute SHAP value (0.5)
        assert_eq!(top_2[1].0, 2); // Feature 2 has second highest (0.3)
    }

    #[test]
    fn test_global_feature_importance() {
        let shap_values = array![[0.1, -0.2, 0.3], [0.2, -0.1, 0.1], [-0.1, 0.3, -0.2]];
        let base_values = array![0.0, 0.0, 0.0];
        let feature_values = array![[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [0.9, 1.9, 2.9]];
        let predictions = array![0.2, 0.2, 0.0];

        let explanations = ShapExplanations {
            values: shap_values,
            base_values,
            feature_values,
            feature_names: None,
            predictions,
        };

        let importance = utils::global_feature_importance(&explanations);
        assert_eq!(importance.len(), 3);
        // All importance values should be positive (absolute values)
        assert!(importance.iter().all(|&x| x >= 0.0));
    }
}
