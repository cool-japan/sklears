//! Model interpretation and explainability tools for neural networks.
//!
//! This module provides gradient-based attribution methods to understand
//! how neural networks make their predictions and which input features
//! are most important for the decision.

use crate::NeuralResult;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::SliceRandomExt;
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Configuration for interpretation methods
#[derive(Debug, Clone)]
pub struct InterpretationConfig {
    /// Number of steps for integrated gradients
    pub n_steps: usize,
    /// Baseline for integrated gradients (typically zeros)
    pub baseline: Option<Array2<f64>>,
    /// Whether to use absolute values for attribution
    pub use_abs: bool,
    /// Noise level for SmoothGrad
    pub noise_level: f64,
    /// Number of samples for SmoothGrad
    pub n_samples: usize,
}

impl Default for InterpretationConfig {
    fn default() -> Self {
        Self {
            n_steps: 50,
            baseline: None,
            use_abs: false,
            noise_level: 0.1,
            n_samples: 50,
        }
    }
}

/// Attribution scores for input features
#[derive(Debug, Clone)]
pub struct AttributionScores {
    /// Attribution scores for each input feature
    pub scores: Array2<f64>,
    /// Total attribution (should sum close to prediction difference)
    pub total_attribution: f64,
    /// Method used for attribution
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl AttributionScores {
    /// Get the most important features (highest absolute attribution)
    pub fn top_features(&self, k: usize) -> Vec<(usize, f64)> {
        let mut feature_scores: Vec<(usize, f64)> = self
            .scores
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(idx, col)| {
                let total_score = col.iter().map(|&x| x.abs()).sum::<f64>();
                (idx, total_score)
            })
            .collect();

        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        feature_scores.into_iter().take(k).collect()
    }

    /// Normalize attribution scores to [0, 1] range
    pub fn normalize(&mut self) {
        let max_val = self.scores.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        if max_val > 0.0 {
            self.scores.mapv_inplace(|x| x / max_val);
        }
    }
}

/// Trait for models that support gradient-based interpretation
pub trait InterpretableModel {
    /// Compute gradients of the output with respect to input
    fn compute_gradients(
        &self,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<Array2<f64>>;

    /// Forward pass returning both prediction and intermediate activations
    fn forward_with_activations(
        &self,
        input: &Array2<f64>,
    ) -> NeuralResult<(Array2<f64>, Vec<Array2<f64>>)>;

    /// Get the number of classes for classification models
    fn num_classes(&self) -> usize;
}

/// Model interpretation methods
pub struct ModelInterpreter {
    config: InterpretationConfig,
}

impl ModelInterpreter {
    /// Create a new model interpreter with default configuration
    pub fn new() -> Self {
        Self {
            config: InterpretationConfig::default(),
        }
    }

    /// Create a new model interpreter with custom configuration
    pub fn with_config(config: InterpretationConfig) -> Self {
        Self { config }
    }

    /// Compute vanilla gradients (saliency maps)
    pub fn vanilla_gradients<M: InterpretableModel>(
        &self,
        model: &M,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<AttributionScores> {
        let gradients = model.compute_gradients(input, target_class)?;

        let scores = if self.config.use_abs {
            gradients.mapv(|x: f64| x.abs())
        } else {
            gradients
        };

        let total_attribution = scores.sum();
        let mut metadata = HashMap::new();
        metadata.insert(
            "max_gradient".to_string(),
            scores.iter().fold(0.0f64, |a, &b| a.max(b.abs())),
        );
        metadata.insert("mean_gradient".to_string(), scores.mean().unwrap_or(0.0));

        Ok(AttributionScores {
            scores,
            total_attribution,
            method: "VanillaGradients".to_string(),
            metadata,
        })
    }

    /// Compute integrated gradients
    pub fn integrated_gradients<M: InterpretableModel>(
        &self,
        model: &M,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<AttributionScores> {
        let baseline = self
            .config
            .baseline
            .as_ref()
            .map(|b| b.clone())
            .unwrap_or_else(|| Array2::zeros(input.raw_dim()));

        if baseline.raw_dim() != input.raw_dim() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", input.raw_dim()),
                actual: format!("{:?}", baseline.raw_dim()),
            });
        }

        let mut integrated_gradients: Array2<f64> = Array2::zeros(input.raw_dim());

        // Compute path integral
        for i in 0..self.config.n_steps {
            let alpha = (i as f64 + 1.0) / self.config.n_steps as f64;
            let interpolated = &baseline + &((&*input - &baseline) * alpha);

            let gradients = model.compute_gradients(&interpolated, target_class)?;
            integrated_gradients = integrated_gradients + gradients;
        }

        // Scale by difference and average over steps
        let difference = input - &baseline;
        let scores = difference * integrated_gradients / self.config.n_steps as f64;

        let scores = if self.config.use_abs {
            scores.mapv(|x: f64| x.abs())
        } else {
            scores
        };

        let total_attribution = scores.sum();
        let mut metadata = HashMap::new();
        metadata.insert("n_steps".to_string(), self.config.n_steps as f64);
        metadata.insert(
            "max_attribution".to_string(),
            scores.iter().fold(0.0, |a, &b| a.max(b.abs())),
        );

        Ok(AttributionScores {
            scores,
            total_attribution,
            method: "IntegratedGradients".to_string(),
            metadata,
        })
    }

    /// Compute SmoothGrad (gradients with noise averaging)
    pub fn smooth_grad<M: InterpretableModel>(
        &self,
        model: &M,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<AttributionScores> {
        use scirs2_core::random::essentials::Normal;
        use scirs2_core::random::prelude::*;

        let mut rng = thread_rng();
        let noise_dist = Normal::new(0.0, self.config.noise_level).map_err(|_| {
            SklearsError::InvalidParameter {
                name: "noise_level".to_string(),
                reason: "Invalid noise level for normal distribution".to_string(),
            }
        })?;

        let mut total_gradients = Array2::zeros(input.raw_dim());

        for _ in 0..self.config.n_samples {
            // Add noise to input
            let noise = Array2::from_shape_fn(input.raw_dim(), |_| rng.sample(noise_dist));
            let noisy_input = input + &noise;

            // Compute gradients for noisy input
            let gradients = model.compute_gradients(&noisy_input, target_class)?;
            total_gradients = total_gradients + gradients;
        }

        // Average gradients
        let scores = total_gradients / self.config.n_samples as f64;

        let scores = if self.config.use_abs {
            scores.mapv(|x: f64| x.abs())
        } else {
            scores
        };

        let total_attribution = scores.sum();
        let mut metadata = HashMap::new();
        metadata.insert("n_samples".to_string(), self.config.n_samples as f64);
        metadata.insert("noise_level".to_string(), self.config.noise_level);
        metadata.insert("variance".to_string(), scores.var(1.0));

        Ok(AttributionScores {
            scores,
            total_attribution,
            method: "SmoothGrad".to_string(),
            metadata,
        })
    }

    /// Compute Guided Backpropagation (modified ReLU gradients)
    pub fn guided_backprop<M: InterpretableModel>(
        &self,
        model: &M,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<AttributionScores> {
        // Note: This is a simplified version. Full implementation would require
        // modifying the ReLU backward pass to only pass positive gradients
        // where both input and gradient are positive.

        let mut gradients = model.compute_gradients(input, target_class)?;

        // For now, use a heuristic: zero out gradients where input is negative
        gradients.zip_mut_with(input, |grad, &inp| {
            if inp <= 0.0 {
                *grad = 0.0;
            }
        });

        let scores = if self.config.use_abs {
            gradients.mapv(|x: f64| x.abs())
        } else {
            gradients
        };

        let total_attribution = scores.sum();
        let mut metadata = HashMap::new();
        metadata.insert(
            "positive_gradients".to_string(),
            scores.iter().filter(|&&x| x > 0.0).count() as f64,
        );

        Ok(AttributionScores {
            scores,
            total_attribution,
            method: "GuidedBackprop".to_string(),
            metadata,
        })
    }

    /// Layer-wise Relevance Propagation (LRP) - simplified implementation
    pub fn layer_relevance_propagation<M: InterpretableModel>(
        &self,
        model: &M,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<AttributionScores> {
        // Get forward pass with intermediate activations
        let (output, activations) = model.forward_with_activations(input)?;

        // Start with output relevance
        let mut relevance = if let Some(class) = target_class {
            let mut r = Array2::zeros(output.raw_dim());
            for (i, mut row) in r.axis_iter_mut(Axis(0)).enumerate() {
                if let Some(class_val) = output.get((i, class)) {
                    row[class] = *class_val;
                }
            }
            r
        } else {
            output.clone()
        };

        // This is a simplified LRP implementation
        // Full LRP would require layer-by-layer relevance propagation
        // using specific rules (e.g., LRP-0, LRP-γ, LRP-ε)

        let scores = if self.config.use_abs {
            relevance.mapv(|x: f64| x.abs())
        } else {
            relevance
        };

        let total_attribution = scores.sum();
        let mut metadata = HashMap::new();
        metadata.insert("n_layers".to_string(), activations.len() as f64);

        Ok(AttributionScores {
            scores,
            total_attribution,
            method: "LayerRelevancePropagation".to_string(),
            metadata,
        })
    }

    /// Generate comprehensive attribution report
    pub fn comprehensive_analysis<M: InterpretableModel>(
        &self,
        model: &M,
        input: &Array2<f64>,
        target_class: Option<usize>,
    ) -> NeuralResult<Vec<AttributionScores>> {
        let methods = vec![
            (
                "vanilla",
                self.vanilla_gradients(model, input, target_class)?,
            ),
            (
                "integrated",
                self.integrated_gradients(model, input, target_class)?,
            ),
            ("smooth", self.smooth_grad(model, input, target_class)?),
            ("guided", self.guided_backprop(model, input, target_class)?),
            (
                "lrp",
                self.layer_relevance_propagation(model, input, target_class)?,
            ),
        ];

        Ok(methods.into_iter().map(|(_, scores)| scores).collect())
    }
}

impl Default for ModelInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature importance analyzer for understanding model behavior
pub struct FeatureImportanceAnalyzer;

impl FeatureImportanceAnalyzer {
    /// Compute permutation-based feature importance
    pub fn permutation_importance<M, F>(
        model: &M,
        input: &Array2<f64>,
        targets: &Array1<f64>,
        metric_fn: F,
        n_repeats: usize,
    ) -> NeuralResult<Array1<f64>>
    where
        M: InterpretableModel,
        F: Fn(&Array2<f64>, &Array1<f64>) -> f64,
    {
        use scirs2_core::random::prelude::*;

        let n_features = input.ncols();
        let mut importances = Array1::zeros(n_features);
        let mut rng = thread_rng();

        // Get baseline score
        let (baseline_pred, _) = model.forward_with_activations(input)?;
        let baseline_score = metric_fn(&baseline_pred, targets);

        for feature_idx in 0..n_features {
            let mut feature_scores = Vec::new();

            for _ in 0..n_repeats {
                // Create permuted data
                let mut permuted_input = input.clone();
                let mut feature_values: Vec<f64> = permuted_input.column(feature_idx).to_vec();
                feature_values.shuffle(&mut rng);

                for (i, &val) in feature_values.iter().enumerate() {
                    permuted_input[[i, feature_idx]] = val;
                }

                // Compute score with permuted feature
                let (permuted_pred, _) = model.forward_with_activations(&permuted_input)?;
                let permuted_score = metric_fn(&permuted_pred, targets);

                // Importance is the decrease in score
                feature_scores.push(baseline_score - permuted_score);
            }

            // Average importance across repeats
            importances[feature_idx] = feature_scores.iter().sum::<f64>() / n_repeats as f64;
        }

        Ok(importances)
    }

    /// Compute SHAP-like values using sampling
    pub fn shap_values<M: InterpretableModel>(
        model: &M,
        input: &Array2<f64>,
        background: &Array2<f64>,
        n_samples: usize,
    ) -> NeuralResult<Array2<f64>> {
        use scirs2_core::random::prelude::*;

        let mut rng = thread_rng();
        let n_features = input.ncols();
        let n_instances = input.nrows();
        let mut shap_values = Array2::zeros((n_instances, n_features));

        for instance_idx in 0..n_instances {
            let instance = input.row(instance_idx);

            for _ in 0..n_samples {
                // Random coalition (subset of features)
                let coalition: Vec<bool> = (0..n_features).map(|_| rng.gen_bool(0.5)).collect();

                // Create two instances: with and without current feature
                for feature_idx in 0..n_features {
                    let mut with_feature = background
                        .row(rng.gen_range(0..background.nrows()))
                        .to_owned();
                    let mut without_feature = with_feature.clone();

                    // Apply coalition
                    for (i, &include) in coalition.iter().enumerate() {
                        if include {
                            with_feature[i] = instance[i];
                            without_feature[i] = instance[i];
                        }
                    }

                    // Add current feature to "with_feature" instance
                    with_feature[feature_idx] = instance[feature_idx];

                    // Compute marginal contribution
                    let with_input = with_feature.insert_axis(Axis(0));
                    let without_input = without_feature.insert_axis(Axis(0));

                    let (with_pred, _) = model.forward_with_activations(&with_input)?;
                    let (without_pred, _) = model.forward_with_activations(&without_input)?;

                    let contribution =
                        with_pred.mean().unwrap_or(0.0) - without_pred.mean().unwrap_or(0.0);
                    shap_values[[instance_idx, feature_idx]] += contribution;
                }
            }

            // Average over samples
            for feature_idx in 0..n_features {
                shap_values[[instance_idx, feature_idx]] /= n_samples as f64;
            }
        }

        Ok(shap_values)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    struct MockModel {
        weights: Array2<f64>,
    }

    impl InterpretableModel for MockModel {
        fn compute_gradients(
            &self,
            input: &Array2<f64>,
            _target_class: Option<usize>,
        ) -> NeuralResult<Array2<f64>> {
            // Simple mock: gradients are just the weights repeated for each sample
            let n_samples = input.nrows();
            let mut gradients = Array2::zeros(input.raw_dim());

            for i in 0..n_samples {
                for j in 0..input.ncols() {
                    gradients[[i, j]] = self.weights[[0, j % self.weights.ncols()]];
                }
            }

            Ok(gradients)
        }

        fn forward_with_activations(
            &self,
            input: &Array2<f64>,
        ) -> NeuralResult<(Array2<f64>, Vec<Array2<f64>>)> {
            let output = input.dot(&self.weights.t());
            Ok((output, vec![input.clone()]))
        }

        fn num_classes(&self) -> usize {
            self.weights.nrows()
        }
    }

    #[test]
    fn test_vanilla_gradients() {
        let model = MockModel {
            weights: Array2::from_shape_vec((1, 3), vec![0.5, -0.3, 0.8]).unwrap(),
        };

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, 0.5, 1.5]).unwrap();

        let interpreter = ModelInterpreter::new();
        let result = interpreter.vanilla_gradients(&model, &input, None).unwrap();

        assert_eq!(result.method, "VanillaGradients");
        assert_eq!(result.scores.raw_dim(), input.raw_dim());
    }

    #[test]
    fn test_integrated_gradients() {
        let model = MockModel {
            weights: Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap(),
        };

        let input = Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).unwrap();

        let config = InterpretationConfig {
            n_steps: 10,
            baseline: Some(Array2::zeros((1, 2))),
            ..Default::default()
        };

        let interpreter = ModelInterpreter::with_config(config);
        let result = interpreter
            .integrated_gradients(&model, &input, None)
            .unwrap();

        assert_eq!(result.method, "IntegratedGradients");
        assert_eq!(result.scores.raw_dim(), input.raw_dim());
    }

    #[test]
    fn test_attribution_scores() {
        let scores = Array2::from_shape_vec((2, 3), vec![0.5, -0.3, 0.8, -0.2, 0.9, -0.1]).unwrap();

        let mut attribution = AttributionScores {
            scores,
            total_attribution: 1.6,
            method: "Test".to_string(),
            metadata: HashMap::new(),
        };

        let top_features = attribution.top_features(2);
        assert_eq!(top_features.len(), 2);

        attribution.normalize();
        let max_val = attribution
            .scores
            .iter()
            .map(|&x| x.abs())
            .fold(0.0, f64::max);
        assert!(max_val <= 1.0 + 1e-10);
    }
}
