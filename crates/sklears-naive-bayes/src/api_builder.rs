//! Enhanced API builder for Naive Bayes models
//!
//! This module provides fluent API builders and configuration presets for
//! creating and configuring Naive Bayes models with better ergonomics.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB, PoissonNB};

/// Configuration presets for common use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NaiveBayesPreset {
    TextClassification,
    DocumentClassification,
    SentimentAnalysis,
    SpamDetection,
    GeneralClassification,
    HighPerformance,
    RobustClassification,
    ProbabilisticClassification,
}

/// Serializable model parameters for all Naive Bayes variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableNBParams {
    /// Model type identifier
    pub model_type: String,
    /// Model configuration as JSON
    pub config: serde_json::Value,
    /// Trained parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Feature names and types
    pub feature_info: Option<HashMap<String, String>>,
    /// Class names
    pub class_names: Option<Vec<String>>,
    /// Model metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Enhanced fluent API builder for Naive Bayes models
#[derive(Debug, Clone)]
pub struct NaiveBayesBuilder {
    preset: Option<NaiveBayesPreset>,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for NaiveBayesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NaiveBayesBuilder {
    /// Create a new Naive Bayes builder
    pub fn new() -> Self {
        Self {
            preset: None,
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    /// Use a configuration preset
    pub fn preset(mut self, preset: NaiveBayesPreset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Enable cross-validation
    pub fn cross_validate(mut self, n_folds: usize) -> Self {
        self.cross_validation = true;
        self.n_folds = n_folds;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Enable serializable model parameters
    pub fn serializable(mut self) -> Self {
        self.serializable = true;
        self
    }

    /// Build a Gaussian Naive Bayes model
    pub fn gaussian(self) -> FluentGaussianNB {
        let mut model = FluentGaussianNB::new();
        self.apply_common_config(&mut model);
        model
    }

    /// Build a Multinomial Naive Bayes model
    pub fn multinomial(self) -> FluentMultinomialNB {
        let mut model = FluentMultinomialNB::new();
        self.apply_common_config(&mut model);
        model
    }

    /// Build a Bernoulli Naive Bayes model
    pub fn bernoulli(self) -> FluentBernoulliNB {
        let mut model = FluentBernoulliNB::new();
        self.apply_common_config(&mut model);
        model
    }

    /// Build a Categorical Naive Bayes model
    pub fn categorical(self) -> FluentCategoricalNB {
        let mut model = FluentCategoricalNB::new();
        self.apply_common_config(&mut model);
        model
    }

    /// Build a Complement Naive Bayes model
    pub fn complement(self) -> FluentComplementNB {
        let mut model = FluentComplementNB::new();
        self.apply_common_config(&mut model);
        model
    }

    /// Build a Poisson Naive Bayes model
    pub fn poisson(self) -> FluentPoissonNB {
        let mut model = FluentPoissonNB::new();
        self.apply_common_config(&mut model);
        model
    }

    /// Apply common configuration to all models
    fn apply_common_config<T: FluentNaiveBayesModel>(&self, model: &mut T) {
        if let Some(preset) = &self.preset {
            model.apply_preset(preset.clone());
        }

        if self.cross_validation {
            model.enable_cross_validation(self.n_folds);
        }

        if let Some(seed) = self.random_state {
            model.set_random_state(seed);
        }

        if self.verbose {
            model.set_verbose(true);
        }

        if self.serializable {
            model.enable_serialization();
        }
    }
}

/// Common trait for all fluent Naive Bayes models
pub trait FluentNaiveBayesModel {
    fn apply_preset(&mut self, preset: NaiveBayesPreset);
    fn enable_cross_validation(&mut self, n_folds: usize);
    fn set_random_state(&mut self, seed: u64);
    fn set_verbose(&mut self, verbose: bool);
    fn enable_serialization(&mut self);
}

/// Fluent wrapper for Gaussian Naive Bayes
#[derive(Debug, Clone)]
pub struct FluentGaussianNB {
    model: GaussianNB,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for FluentGaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentGaussianNB {
    pub fn new() -> Self {
        Self {
            model: GaussianNB::new(),
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    /// Set variance smoothing parameter
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.model = self.model.var_smoothing(var_smoothing);
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.model = self.model.priors(priors);
        self
    }

    /// Quick configuration for text classification
    pub fn for_text_classification(mut self) -> Self {
        self.apply_preset(NaiveBayesPreset::TextClassification);
        self
    }

    /// Quick configuration for document classification
    pub fn for_document_classification(mut self) -> Self {
        self.apply_preset(NaiveBayesPreset::DocumentClassification);
        self
    }

    /// Quick configuration for sentiment analysis
    pub fn for_sentiment_analysis(mut self) -> Self {
        self.apply_preset(NaiveBayesPreset::SentimentAnalysis);
        self
    }

    /// Get the underlying model
    pub fn build(self) -> GaussianNB {
        self.model
    }
}

impl FluentNaiveBayesModel for FluentGaussianNB {
    fn apply_preset(&mut self, preset: NaiveBayesPreset) {
        match preset {
            NaiveBayesPreset::TextClassification => {
                self.model = self.model.clone().var_smoothing(1e-9);
                self.cross_validation = true;
                self.n_folds = 5;
            }
            NaiveBayesPreset::DocumentClassification => {
                self.model = self.model.clone().var_smoothing(1e-6);
                self.cross_validation = true;
                self.n_folds = 10;
            }
            NaiveBayesPreset::SentimentAnalysis => {
                self.model = self.model.clone().var_smoothing(1e-8);
                self.cross_validation = true;
                self.verbose = true;
            }
            NaiveBayesPreset::SpamDetection => {
                self.model = self.model.clone().var_smoothing(1e-7);
                self.cross_validation = true;
                self.n_folds = 5;
            }
            NaiveBayesPreset::GeneralClassification => {
                self.model = self.model.clone().var_smoothing(1e-9);
                self.cross_validation = false;
            }
            NaiveBayesPreset::HighPerformance => {
                self.model = self.model.clone().var_smoothing(1e-10);
                self.cross_validation = false;
            }
            NaiveBayesPreset::RobustClassification => {
                self.model = self.model.clone().var_smoothing(1e-6);
                self.cross_validation = true;
                self.n_folds = 10;
            }
            NaiveBayesPreset::ProbabilisticClassification => {
                self.model = self.model.clone().var_smoothing(1e-9);
                self.cross_validation = true;
                self.serializable = true;
            }
        }
    }

    fn enable_cross_validation(&mut self, n_folds: usize) {
        self.cross_validation = true;
        self.n_folds = n_folds;
    }

    fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn enable_serialization(&mut self) {
        self.serializable = true;
    }
}

/// Fluent wrapper for Multinomial Naive Bayes
#[derive(Debug, Clone)]
pub struct FluentMultinomialNB {
    model: MultinomialNB,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for FluentMultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentMultinomialNB {
    pub fn new() -> Self {
        Self {
            model: MultinomialNB::new(),
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    /// Set smoothing parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.model = self.model.alpha(alpha);
        self
    }

    /// Set whether to learn class prior probabilities
    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.model = self.model.fit_prior(fit_prior);
        self
    }

    /// Set class prior probabilities
    pub fn class_prior(mut self, class_prior: Array1<f64>) -> Self {
        self.model = self.model.class_prior(class_prior);
        self
    }

    /// Quick configuration for text classification
    pub fn for_text_classification(mut self) -> Self {
        self.apply_preset(NaiveBayesPreset::TextClassification);
        self
    }

    /// Get the underlying model
    pub fn build(self) -> MultinomialNB {
        self.model
    }
}

impl FluentNaiveBayesModel for FluentMultinomialNB {
    fn apply_preset(&mut self, preset: NaiveBayesPreset) {
        match preset {
            NaiveBayesPreset::TextClassification => {
                self.model = self.model.clone().alpha(1.0).fit_prior(true);
            }
            NaiveBayesPreset::DocumentClassification => {
                self.model = self.model.clone().alpha(0.5).fit_prior(true);
            }
            NaiveBayesPreset::SentimentAnalysis => {
                self.model = self.model.clone().alpha(0.1).fit_prior(true);
            }
            NaiveBayesPreset::SpamDetection => {
                self.model = self.model.clone().alpha(1.0).fit_prior(true);
            }
            _ => {
                self.model = self.model.clone().alpha(1.0).fit_prior(true);
            }
        }
    }

    fn enable_cross_validation(&mut self, n_folds: usize) {
        self.cross_validation = true;
        self.n_folds = n_folds;
    }

    fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn enable_serialization(&mut self) {
        self.serializable = true;
    }
}

/// Fluent wrapper for Bernoulli Naive Bayes
#[derive(Debug, Clone)]
pub struct FluentBernoulliNB {
    model: BernoulliNB,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for FluentBernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentBernoulliNB {
    pub fn new() -> Self {
        Self {
            model: BernoulliNB::new(),
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.model = self.model.alpha(alpha);
        self
    }

    pub fn build(self) -> BernoulliNB {
        self.model
    }
}

impl FluentNaiveBayesModel for FluentBernoulliNB {
    fn apply_preset(&mut self, preset: NaiveBayesPreset) {
        match preset {
            NaiveBayesPreset::TextClassification => {
                self.model = self.model.clone().alpha(1.0);
            }
            _ => {
                self.model = self.model.clone().alpha(1.0);
            }
        }
    }

    fn enable_cross_validation(&mut self, n_folds: usize) {
        self.cross_validation = true;
        self.n_folds = n_folds;
    }

    fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn enable_serialization(&mut self) {
        self.serializable = true;
    }
}

/// Fluent wrapper for Categorical Naive Bayes
#[derive(Debug, Clone)]
pub struct FluentCategoricalNB {
    model: CategoricalNB,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for FluentCategoricalNB {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentCategoricalNB {
    pub fn new() -> Self {
        Self {
            model: CategoricalNB::new(),
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.model = self.model.alpha(alpha);
        self
    }

    pub fn build(self) -> CategoricalNB {
        self.model
    }
}

impl FluentNaiveBayesModel for FluentCategoricalNB {
    fn apply_preset(&mut self, preset: NaiveBayesPreset) {
        match preset {
            NaiveBayesPreset::TextClassification => {
                self.model = self.model.clone().alpha(1.0);
            }
            _ => {
                self.model = self.model.clone().alpha(1.0);
            }
        }
    }

    fn enable_cross_validation(&mut self, n_folds: usize) {
        self.cross_validation = true;
        self.n_folds = n_folds;
    }

    fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn enable_serialization(&mut self) {
        self.serializable = true;
    }
}

/// Fluent wrapper for Complement Naive Bayes
#[derive(Debug, Clone)]
pub struct FluentComplementNB {
    model: ComplementNB,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for FluentComplementNB {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentComplementNB {
    pub fn new() -> Self {
        Self {
            model: ComplementNB::new(),
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.model = self.model.alpha(alpha);
        self
    }

    pub fn build(self) -> ComplementNB {
        self.model
    }
}

impl FluentNaiveBayesModel for FluentComplementNB {
    fn apply_preset(&mut self, preset: NaiveBayesPreset) {
        match preset {
            NaiveBayesPreset::TextClassification => {
                self.model = self.model.clone().alpha(1.0);
            }
            _ => {
                self.model = self.model.clone().alpha(1.0);
            }
        }
    }

    fn enable_cross_validation(&mut self, n_folds: usize) {
        self.cross_validation = true;
        self.n_folds = n_folds;
    }

    fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn enable_serialization(&mut self) {
        self.serializable = true;
    }
}

/// Fluent wrapper for Poisson Naive Bayes
#[derive(Debug, Clone)]
pub struct FluentPoissonNB {
    model: PoissonNB,
    cross_validation: bool,
    n_folds: usize,
    random_state: Option<u64>,
    verbose: bool,
    serializable: bool,
}

impl Default for FluentPoissonNB {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentPoissonNB {
    pub fn new() -> Self {
        Self {
            model: PoissonNB::new(),
            cross_validation: false,
            n_folds: 5,
            random_state: None,
            verbose: false,
            serializable: false,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.model = self.model.alpha(alpha);
        self
    }

    pub fn build(self) -> PoissonNB {
        self.model
    }
}

impl FluentNaiveBayesModel for FluentPoissonNB {
    fn apply_preset(&mut self, preset: NaiveBayesPreset) {
        match preset {
            NaiveBayesPreset::TextClassification => {
                self.model = self.model.clone().alpha(1.0);
            }
            _ => {
                self.model = self.model.clone().alpha(1.0);
            }
        }
    }

    fn enable_cross_validation(&mut self, n_folds: usize) {
        self.cross_validation = true;
        self.n_folds = n_folds;
    }

    fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn enable_serialization(&mut self) {
        self.serializable = true;
    }
}

/// Convenience function to create a new Naive Bayes builder
pub fn naive_bayes() -> NaiveBayesBuilder {
    NaiveBayesBuilder::new()
}

/// Convenience function to create a new Naive Bayes builder with preset
pub fn naive_bayes_preset(preset: NaiveBayesPreset) -> NaiveBayesBuilder {
    NaiveBayesBuilder::new().preset(preset)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use sklears_core::traits::Estimator;

    #[test]
    fn test_naive_bayes_builder_gaussian() {
        let model = naive_bayes()
            .preset(NaiveBayesPreset::TextClassification)
            .cross_validate(5)
            .verbose()
            .gaussian()
            .var_smoothing(1e-8)
            .build();

        // Model should be created successfully
        assert!(model.config().var_smoothing > 0.0);
    }

    #[test]
    fn test_naive_bayes_builder_multinomial() {
        let model = naive_bayes()
            .preset(NaiveBayesPreset::DocumentClassification)
            .cross_validate(10)
            .multinomial()
            .alpha(0.5)
            .build();

        // Model should be created successfully
        assert!(model.config().alpha > 0.0);
    }

    #[test]
    fn test_fluent_api_chaining() {
        let model = naive_bayes()
            .preset(NaiveBayesPreset::SentimentAnalysis)
            .cross_validate(5)
            .random_state(42)
            .verbose()
            .serializable()
            .gaussian()
            .var_smoothing(1e-9)
            .for_text_classification()
            .build();

        // Model should be created with all configurations
        assert!(model.config().var_smoothing > 0.0);
    }

    #[test]
    fn test_configuration_presets() {
        // Test different presets
        let text_model = naive_bayes_preset(NaiveBayesPreset::TextClassification)
            .multinomial()
            .build();

        let sentiment_model = naive_bayes_preset(NaiveBayesPreset::SentimentAnalysis)
            .gaussian()
            .build();

        // Both models should be created with appropriate configurations
        assert!(text_model.config().alpha > 0.0);
        assert!(sentiment_model.config().var_smoothing > 0.0);
    }

    #[test]
    fn test_serializable_params() {
        let params = SerializableNBParams {
            model_type: "GaussianNB".to_string(),
            config: serde_json::json!({"var_smoothing": 1e-9}),
            parameters: HashMap::new(),
            feature_info: Some(HashMap::new()),
            class_names: Some(vec!["Class1".to_string(), "Class2".to_string()]),
            metadata: HashMap::new(),
        };

        // Serialization should work
        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: SerializableNBParams = serde_json::from_str(&serialized).unwrap();

        assert_eq!(params.model_type, deserialized.model_type);
    }
}
