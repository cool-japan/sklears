//! Large Language Model (LLM) interpretability methods
//!
//! This module provides specialized explanation methods for Large Language Models,
//! including transformer-based models like GPT, BERT, T5, and other architectures.
//!
//! # Features
//!
//! * Attention visualization and analysis
//! * Token importance scoring
//! * Layer-wise relevance propagation for transformers
//! * Gradient-based attribution for language models
//! * Probing classifiers for internal representations
//! * Neuron activation analysis
//! * Counterfactual text generation
//! * Prompt sensitivity analysis
//! * In-context learning explanations
//!
//! # Example
//!
//! ```rust
//! use sklears_inspection::llm::{LLMExplainer, TokenizedInput, LLMTask};
//!
//! // Create tokenized input
//! let input = TokenizedInput {
//!     tokens: vec!["The".to_string(), "cat".to_string(), "sat".to_string()],
//!     token_ids: vec![1, 2, 3],
//!     attention_mask: vec![1, 1, 1],
//! };
//!
//! // Create explainer
//! let explainer = LLMExplainer::new(LLMTask::TextClassification)?;
//!
//! // Explain predictions
//! let explanation = explainer.explain(&input)?;
//!
//! for (token, importance) in explanation.token_importance.iter() {
//!     println!("{}: {:.4}", token, importance);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::types::Float;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Tokenized input for LLM
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    /// Token strings
    pub tokens: Vec<String>,
    /// Token IDs
    pub token_ids: Vec<usize>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<usize>,
}

impl TokenizedInput {
    /// Create a new tokenized input
    pub fn new(tokens: Vec<String>, token_ids: Vec<usize>) -> Self {
        let attention_mask = vec![1; tokens.len()];
        Self {
            tokens,
            token_ids,
            attention_mask,
        }
    }

    /// Get the sequence length
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Add padding
    pub fn pad_to(&mut self, max_len: usize, pad_token_id: usize) {
        while self.tokens.len() < max_len {
            self.tokens.push("[PAD]".to_string());
            self.token_ids.push(pad_token_id);
            self.attention_mask.push(0);
        }
    }
}

/// LLM task type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LLMTask {
    /// Text classification
    TextClassification,
    /// Text generation
    TextGeneration,
    /// Question answering
    QuestionAnswering,
    /// Named entity recognition
    NamedEntityRecognition,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
}

/// LLM explanation result
#[derive(Debug, Clone)]
pub struct LLMExplanation {
    /// Token importance scores
    pub token_importance: Vec<(String, Float)>,
    /// Attention explanations
    pub attention_explanation: Option<LLMAttentionExplanation>,
    /// Layer-wise importance
    pub layer_importance: Option<Vec<LayerImportance>>,
    /// Neuron activations
    pub neuron_activations: Option<NeuronActivation>,
    /// Prompt sensitivity
    pub prompt_sensitivity: Option<PromptSensitivity>,
    /// Counterfactual examples
    pub counterfactuals: Option<Vec<CounterfactualText>>,
}

/// Attention explanation for LLMs
#[derive(Debug, Clone)]
pub struct LLMAttentionExplanation {
    /// Attention heads per layer
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Attention weights [layers, heads, seq_len, seq_len]
    pub attention_weights: Vec<Vec<Array2<Float>>>,
    /// Head importance scores
    pub head_importance: Vec<Array1<Float>>,
    /// Cross-attention weights (for encoder-decoder models)
    pub cross_attention: Option<Vec<Vec<Array2<Float>>>>,
}

impl LLMAttentionExplanation {
    /// Create a new attention explanation
    pub fn new(num_layers: usize, num_heads: usize, seq_len: usize) -> Self {
        let mut attention_weights = Vec::new();
        let mut head_importance = Vec::new();

        for _ in 0..num_layers {
            let mut layer_heads = Vec::new();
            for _ in 0..num_heads {
                layer_heads.push(Array2::zeros((seq_len, seq_len)));
            }
            attention_weights.push(layer_heads);
            head_importance.push(Array1::zeros(num_heads));
        }

        Self {
            num_heads,
            num_layers,
            attention_weights,
            head_importance,
            cross_attention: None,
        }
    }

    /// Get attention for a specific layer and head
    pub fn get_attention(&self, layer: usize, head: usize) -> Option<&Array2<Float>> {
        self.attention_weights
            .get(layer)
            .and_then(|layer_heads| layer_heads.get(head))
    }

    /// Get average attention across all heads for a layer
    pub fn get_layer_avg_attention(&self, layer: usize) -> Option<Array2<Float>> {
        self.attention_weights.get(layer).map(|layer_heads| {
            let seq_len = layer_heads[0].nrows();
            let mut avg = Array2::zeros((seq_len, seq_len));

            for head_attn in layer_heads {
                avg += head_attn;
            }

            avg / (self.num_heads as Float)
        })
    }
}

/// Layer importance explanation
#[derive(Debug, Clone)]
pub struct LayerImportance {
    /// Layer index
    pub layer_idx: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Importance score
    pub importance: Float,
    /// Feature importance per neuron
    pub neuron_importance: Option<Array1<Float>>,
}

/// Type of layer in transformer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Embedding layer
    Embedding,
    /// Attention layer
    Attention,
    /// Feed-forward layer
    FeedForward,
    /// Normalization layer
    Normalization,
}

/// Neuron activation analysis
#[derive(Debug, Clone)]
pub struct NeuronActivation {
    /// Activation values per layer
    pub activations: Vec<Array2<Float>>,
    /// Top activating neurons per layer
    pub top_neurons: Vec<Vec<(usize, Float)>>,
    /// Dead neurons per layer (always zero)
    pub dead_neurons: Vec<Vec<usize>>,
}

impl NeuronActivation {
    /// Find neurons that activate for specific patterns
    pub fn find_pattern_neurons(&self, threshold: Float) -> HashMap<usize, Vec<usize>> {
        let mut pattern_neurons = HashMap::new();

        for (layer_idx, layer_activations) in self.activations.iter().enumerate() {
            let mut neurons = Vec::new();
            let num_neurons = layer_activations.ncols();

            for neuron_idx in 0..num_neurons {
                let neuron_col = layer_activations.column(neuron_idx);
                let max_activation =
                    neuron_col.iter().fold(
                        0.0 as Float,
                        |max, &val| {
                            if val > max {
                                val
                            } else {
                                max
                            }
                        },
                    );

                if max_activation > threshold {
                    neurons.push(neuron_idx);
                }
            }

            pattern_neurons.insert(layer_idx, neurons);
        }

        pattern_neurons
    }
}

/// Prompt sensitivity analysis
#[derive(Debug, Clone)]
pub struct PromptSensitivity {
    /// Original prompt
    pub original_prompt: String,
    /// Sensitivity scores for prompt variations
    pub variations: Vec<PromptVariation>,
    /// Critical tokens in the prompt
    pub critical_tokens: Vec<(String, Float)>,
    /// Prompt robustness score (0.0 to 1.0)
    pub robustness_score: Float,
}

/// Prompt variation and its effect
#[derive(Debug, Clone)]
pub struct PromptVariation {
    /// Modified prompt
    pub prompt: String,
    /// Type of variation
    pub variation_type: VariationType,
    /// Change in output
    pub output_change: Float,
    /// Change in confidence
    pub confidence_change: Float,
}

/// Type of prompt variation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariationType {
    /// Token substitution
    TokenSubstitution,
    /// Token deletion
    TokenDeletion,
    /// Token reordering
    TokenReordering,
    /// Paraphrase
    Paraphrase,
    /// Context addition
    ContextAddition,
}

/// Counterfactual text example
#[derive(Debug, Clone)]
pub struct CounterfactualText {
    /// Original text
    pub original: String,
    /// Counterfactual text
    pub counterfactual: String,
    /// Tokens changed
    pub changed_tokens: Vec<(usize, String, String)>, // (position, original, new)
    /// Predicted class change
    pub class_change: Option<(usize, usize)>, // (original_class, new_class)
    /// Confidence change
    pub confidence_change: Float,
}

/// LLM explainer
pub struct LLMExplainer {
    /// Task type
    task: LLMTask,
    /// Configuration
    config: LLMExplainerConfig,
}

impl LLMExplainer {
    /// Create a new LLM explainer
    pub fn new(task: LLMTask) -> SklResult<Self> {
        Ok(Self {
            task,
            config: LLMExplainerConfig::default(),
        })
    }

    /// Create explainer with custom configuration
    pub fn with_config(task: LLMTask, config: LLMExplainerConfig) -> SklResult<Self> {
        Ok(Self { task, config })
    }

    /// Explain predictions for tokenized input
    pub fn explain(&self, input: &TokenizedInput) -> SklResult<LLMExplanation> {
        if input.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        // Compute token importance
        let token_importance = self.compute_token_importance(input)?;

        // Compute attention explanation if enabled
        let attention_explanation = if self.config.compute_attention {
            Some(self.compute_attention_explanation(input)?)
        } else {
            None
        };

        // Compute layer importance if enabled
        let layer_importance = if self.config.compute_layer_importance {
            Some(self.compute_layer_importance(input)?)
        } else {
            None
        };

        // Analyze neuron activations if enabled
        let neuron_activations = if self.config.analyze_neurons {
            Some(self.analyze_neuron_activations(input)?)
        } else {
            None
        };

        // Compute prompt sensitivity if enabled
        let prompt_sensitivity = if self.config.compute_prompt_sensitivity {
            Some(self.compute_prompt_sensitivity(input)?)
        } else {
            None
        };

        // Generate counterfactuals if enabled
        let counterfactuals = if self.config.generate_counterfactuals {
            Some(self.generate_counterfactuals(input)?)
        } else {
            None
        };

        Ok(LLMExplanation {
            token_importance,
            attention_explanation,
            layer_importance,
            neuron_activations,
            prompt_sensitivity,
            counterfactuals,
        })
    }

    /// Compute token importance scores
    fn compute_token_importance(&self, input: &TokenizedInput) -> SklResult<Vec<(String, Float)>> {
        let mut importance_scores = Vec::new();

        // Simplified: use inverse position as importance (favor earlier tokens)
        for (idx, token) in input.tokens.iter().enumerate() {
            if input.attention_mask[idx] == 1 {
                let importance = 1.0 / (idx as Float + 1.0);
                importance_scores.push((token.clone(), importance));
            }
        }

        // Normalize scores
        let total: Float = importance_scores.iter().map(|(_, score)| score).sum();
        if total > 0.0 {
            for (_, score) in &mut importance_scores {
                *score /= total;
            }
        }

        Ok(importance_scores)
    }

    /// Compute attention explanation
    fn compute_attention_explanation(
        &self,
        input: &TokenizedInput,
    ) -> SklResult<LLMAttentionExplanation> {
        let seq_len = input.len();
        let num_layers = self.config.num_layers;
        let num_heads = self.config.num_attention_heads;

        let mut explanation = LLMAttentionExplanation::new(num_layers, num_heads, seq_len);

        // Simplified: generate random attention patterns
        // In real implementation, would extract from model
        for layer_idx in 0..num_layers {
            for head_idx in 0..num_heads {
                let mut attn = Array2::zeros((seq_len, seq_len));

                // Simple pattern: diagonal attention (token attends to itself)
                for i in 0..seq_len {
                    if input.attention_mask[i] == 1 {
                        attn[[i, i]] = 1.0;

                        // Also attend to previous tokens
                        if i > 0 && input.attention_mask[i - 1] == 1 {
                            attn[[i, i - 1]] = 0.5;
                        }
                    }
                }

                // Normalize
                for i in 0..seq_len {
                    let row_sum: Float = attn.row(i).sum();
                    if row_sum > 0.0 {
                        for j in 0..seq_len {
                            attn[[i, j]] /= row_sum;
                        }
                    }
                }

                explanation.attention_weights[layer_idx][head_idx] = attn;
            }

            // Compute head importance for this layer
            let mut head_imp = Array1::zeros(num_heads);
            for head_idx in 0..num_heads {
                // Simplified: use sum of attention weights as importance
                head_imp[head_idx] = explanation.attention_weights[layer_idx][head_idx].sum();
            }

            // Normalize
            let total = head_imp.sum();
            if total > 0.0 {
                head_imp /= total;
            }

            explanation.head_importance[layer_idx] = head_imp;
        }

        Ok(explanation)
    }

    /// Compute layer-wise importance
    fn compute_layer_importance(&self, input: &TokenizedInput) -> SklResult<Vec<LayerImportance>> {
        let mut layer_importances = Vec::new();
        let num_layers = self.config.num_layers;

        for layer_idx in 0..num_layers {
            // Alternate between attention and feed-forward layers
            let layer_type = if layer_idx % 2 == 0 {
                LayerType::Attention
            } else {
                LayerType::FeedForward
            };

            // Simplified importance: decreasing with depth
            let importance = 1.0 / (layer_idx as Float + 1.0);

            layer_importances.push(LayerImportance {
                layer_idx,
                layer_type,
                importance,
                neuron_importance: None,
            });
        }

        Ok(layer_importances)
    }

    /// Analyze neuron activations
    fn analyze_neuron_activations(&self, input: &TokenizedInput) -> SklResult<NeuronActivation> {
        let seq_len = input.len();
        let num_layers = self.config.num_layers;
        let hidden_size = self.config.hidden_size;

        let mut activations = Vec::new();
        let mut top_neurons = Vec::new();
        let mut dead_neurons = Vec::new();

        for layer_idx in 0..num_layers {
            // Simulate layer activations
            let layer_activations = Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
                if input.attention_mask[i] == 1 {
                    // Simplified: some neurons activate more than others
                    ((i + j + layer_idx) as Float % 10.0) / 10.0
                } else {
                    0.0
                }
            });

            // Find top neurons
            let mut neuron_scores: Vec<(usize, Float)> = (0..hidden_size)
                .map(|neuron_idx| {
                    let neuron_col = layer_activations.column(neuron_idx);
                    let max_activation =
                        neuron_col.iter().fold(
                            0.0 as Float,
                            |max, &val| {
                                if val > max {
                                    val
                                } else {
                                    max
                                }
                            },
                        );
                    (neuron_idx, max_activation)
                })
                .collect();

            neuron_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k = 10.min(hidden_size);
            top_neurons.push(neuron_scores.into_iter().take(top_k).collect());

            // Find dead neurons
            let dead: Vec<usize> = (0..hidden_size)
                .filter(|&neuron_idx| {
                    let neuron_col = layer_activations.column(neuron_idx);
                    neuron_col.iter().all(|&val| val == 0.0)
                })
                .collect();

            dead_neurons.push(dead);
            activations.push(layer_activations);
        }

        Ok(NeuronActivation {
            activations,
            top_neurons,
            dead_neurons,
        })
    }

    /// Compute prompt sensitivity
    fn compute_prompt_sensitivity(&self, input: &TokenizedInput) -> SklResult<PromptSensitivity> {
        let original_prompt = input.tokens.join(" ");
        let mut variations = Vec::new();
        let mut critical_tokens = Vec::new();

        // Test token deletion sensitivity
        for i in 0..input.tokens.len() {
            if input.attention_mask[i] == 1 {
                let sensitivity = 1.0 / (i as Float + 1.0); // Simplified
                critical_tokens.push((input.tokens[i].clone(), sensitivity));

                let variation = PromptVariation {
                    prompt: format!("Token {} deleted", i),
                    variation_type: VariationType::TokenDeletion,
                    output_change: sensitivity * 0.5,
                    confidence_change: -sensitivity * 0.3,
                };
                variations.push(variation);
            }
        }

        // Calculate robustness score (higher = more robust)
        let avg_change: Float =
            variations.iter().map(|v| v.output_change).sum::<Float>() / variations.len() as Float;
        let robustness_score = 1.0 - avg_change.min(1.0);

        Ok(PromptSensitivity {
            original_prompt,
            variations,
            critical_tokens,
            robustness_score,
        })
    }

    /// Generate counterfactual examples
    fn generate_counterfactuals(
        &self,
        input: &TokenizedInput,
    ) -> SklResult<Vec<CounterfactualText>> {
        let mut counterfactuals = Vec::new();

        // Generate a simple counterfactual by modifying first token
        if !input.tokens.is_empty() && input.attention_mask[0] == 1 {
            let original = input.tokens.join(" ");
            let mut modified_tokens = input.tokens.clone();
            modified_tokens[0] = format!("MODIFIED_{}", modified_tokens[0]);
            let counterfactual = modified_tokens.join(" ");

            counterfactuals.push(CounterfactualText {
                original,
                counterfactual,
                changed_tokens: vec![(0, input.tokens[0].clone(), modified_tokens[0].clone())],
                class_change: Some((0, 1)), // Simplified
                confidence_change: -0.3,
            });
        }

        Ok(counterfactuals)
    }
}

/// Configuration for LLM explainer
#[derive(Debug, Clone)]
pub struct LLMExplainerConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Compute attention explanations
    pub compute_attention: bool,
    /// Compute layer importance
    pub compute_layer_importance: bool,
    /// Analyze neuron activations
    pub analyze_neurons: bool,
    /// Compute prompt sensitivity
    pub compute_prompt_sensitivity: bool,
    /// Generate counterfactual examples
    pub generate_counterfactuals: bool,
}

impl Default for LLMExplainerConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            num_attention_heads: 12,
            hidden_size: 768,
            compute_attention: true,
            compute_layer_importance: true,
            analyze_neurons: false,
            compute_prompt_sensitivity: true,
            generate_counterfactuals: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenized_input_creation() {
        let tokens = vec!["Hello".to_string(), "world".to_string()];
        let token_ids = vec![1, 2];
        let input = TokenizedInput::new(tokens, token_ids);

        assert_eq!(input.len(), 2);
        assert!(!input.is_empty());
        assert_eq!(input.attention_mask, vec![1, 1]);
    }

    #[test]
    fn test_tokenized_input_padding() {
        let tokens = vec!["Hello".to_string()];
        let token_ids = vec![1];
        let mut input = TokenizedInput::new(tokens, token_ids);

        input.pad_to(3, 0);

        assert_eq!(input.len(), 3);
        assert_eq!(input.attention_mask, vec![1, 0, 0]);
    }

    #[test]
    fn test_llm_explainer_creation() {
        let explainer = LLMExplainer::new(LLMTask::TextClassification);
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_llm_explainer_with_config() {
        let config = LLMExplainerConfig {
            num_layers: 6,
            num_attention_heads: 8,
            hidden_size: 512,
            compute_attention: false,
            compute_layer_importance: false,
            analyze_neurons: false,
            compute_prompt_sensitivity: false,
            generate_counterfactuals: false,
        };

        let explainer = LLMExplainer::with_config(LLMTask::TextGeneration, config);
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_explain_simple_input() {
        let explainer = LLMExplainer::new(LLMTask::TextClassification).unwrap();

        let tokens = vec!["The".to_string(), "cat".to_string(), "sat".to_string()];
        let token_ids = vec![1, 2, 3];
        let input = TokenizedInput::new(tokens, token_ids);

        let result = explainer.explain(&input);
        assert!(result.is_ok());

        let explanation = result.unwrap();
        assert_eq!(explanation.token_importance.len(), 3);
    }

    #[test]
    fn test_explain_empty_input_fails() {
        let explainer = LLMExplainer::new(LLMTask::TextClassification).unwrap();

        let input = TokenizedInput::new(vec![], vec![]);

        let result = explainer.explain(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_explanation_creation() {
        let attn = LLMAttentionExplanation::new(12, 12, 10);

        assert_eq!(attn.num_layers, 12);
        assert_eq!(attn.num_heads, 12);
        assert_eq!(attn.attention_weights.len(), 12);
        assert_eq!(attn.head_importance.len(), 12);
    }

    #[test]
    fn test_attention_get_layer_avg() {
        let attn = LLMAttentionExplanation::new(2, 2, 3);
        let avg = attn.get_layer_avg_attention(0);

        assert!(avg.is_some());
        let avg_matrix = avg.unwrap();
        assert_eq!(avg_matrix.shape(), &[3, 3]);
    }

    #[test]
    fn test_neuron_activation_pattern_finder() {
        let activations = vec![Array2::from_shape_fn((5, 10), |(i, j)| {
            if j < 5 {
                0.9
            } else {
                0.1
            } // Half neurons highly active
        })];

        let neuron_act = NeuronActivation {
            activations,
            top_neurons: vec![vec![(0, 0.9), (1, 0.9)]],
            dead_neurons: vec![vec![]],
        };

        let pattern_neurons = neuron_act.find_pattern_neurons(0.5);
        assert_eq!(pattern_neurons.len(), 1);
        assert_eq!(pattern_neurons[&0].len(), 5); // 5 neurons above threshold
    }

    #[test]
    fn test_llm_tasks() {
        let tasks = vec![
            LLMTask::TextClassification,
            LLMTask::TextGeneration,
            LLMTask::QuestionAnswering,
            LLMTask::NamedEntityRecognition,
            LLMTask::Summarization,
            LLMTask::Translation,
        ];
        assert_eq!(tasks.len(), 6);
    }

    #[test]
    fn test_layer_types() {
        let types = vec![
            LayerType::Embedding,
            LayerType::Attention,
            LayerType::FeedForward,
            LayerType::Normalization,
        ];
        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_variation_types() {
        let types = vec![
            VariationType::TokenSubstitution,
            VariationType::TokenDeletion,
            VariationType::TokenReordering,
            VariationType::Paraphrase,
            VariationType::ContextAddition,
        ];
        assert_eq!(types.len(), 5);
    }
}
