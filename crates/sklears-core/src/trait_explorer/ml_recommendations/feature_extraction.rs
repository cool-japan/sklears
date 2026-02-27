//! Feature extraction system for converting trait information to numerical features
//!
//! This module provides comprehensive feature extraction capabilities including
//! text processing, TF-IDF vectorization, n-gram analysis, semantic features,
//! and statistical feature computation.

use crate::api_reference_generator::{MethodInfo, TraitInfo};
use crate::error::{Result, SklearsError};
use crate::trait_explorer::ml_recommendations::data_types::*;
use crate::trait_explorer::ml_recommendations::neural_networks::NeuralEmbeddingModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// SciRS2 Compliance: Use scirs2_autograd for ndarray functionality
use scirs2_core::ndarray::{s, Array1, ArrayView1};

/// Feature extraction system for converting trait information to numerical features
pub struct TraitFeatureExtractor {
    /// Cache for extracted features
    feature_cache: HashMap<String, Array1<f64>>,
    /// Configuration
    config: FeatureExtractionConfig,
    /// Neural embedding model for semantic features
    embedding_model: Option<NeuralEmbeddingModel>,
    /// Text processing vocabulary
    vocabulary: HashMap<String, usize>,
    /// TF-IDF weights for text features
    tfidf_weights: HashMap<String, f64>,
    /// N-gram cache for efficiency
    ngram_cache: HashMap<String, Vec<f64>>,
    /// Statistical feature cache
    stats_cache: HashMap<String, Vec<f64>>,
}

impl TraitFeatureExtractor {
    /// Create a new feature extractor with default configuration
    pub fn new() -> Self {
        Self {
            feature_cache: HashMap::new(),
            config: FeatureExtractionConfig::default(),
            embedding_model: None,
            vocabulary: HashMap::new(),
            tfidf_weights: HashMap::new(),
            ngram_cache: HashMap::new(),
            stats_cache: HashMap::new(),
        }
    }

    /// Create a feature extractor with custom configuration
    pub fn with_config(config: FeatureExtractionConfig) -> Self {
        let embedding_model = if config.embedding_config.pretrained_model.is_some() {
            // In a real implementation, this would load a pre-trained model
            Some(NeuralEmbeddingModel::new(
                config.embedding_config.dimensions,
                config.embedding_config.dimensions / 2,
                &[128, 64],
            ))
        } else {
            None
        };

        Self {
            feature_cache: HashMap::new(),
            config,
            embedding_model,
            vocabulary: HashMap::new(),
            tfidf_weights: HashMap::new(),
            ngram_cache: HashMap::new(),
            stats_cache: HashMap::new(),
        }
    }

    /// Extract numerical features from trait context
    pub fn extract_context_features(&mut self, context: &TraitContext) -> Result<Array1<f64>> {
        // Check cache first
        if let Some(cached) = self.feature_cache.get(&context.trait_name) {
            return Ok(cached.clone());
        }

        let mut features = Vec::new();

        // Basic numerical features
        features.extend(self.extract_basic_features(context));

        // Text-based features from description
        let text_features = self.extract_text_features(&context.description)?;
        features.extend(text_features.iter());

        // Keyword features
        let keyword_features = self.extract_keyword_features(&context.keywords)?;
        features.extend(keyword_features.iter());

        // Semantic features (if neural embeddings enabled)
        if let Some(ref model) = self.embedding_model {
            let semantic_features = self.extract_semantic_features(context, model)?;
            features.extend(semantic_features.iter());
        }

        // Statistical features
        let statistical_features = self.extract_statistical_features(context)?;
        features.extend(statistical_features.iter());

        // API complexity features
        let api_features = self.extract_api_complexity_features(context);
        features.extend(api_features);

        // Memory and performance features
        let memory_features = self.extract_memory_features(context);
        features.extend(memory_features);

        // Concurrency features
        let concurrency_features = self.extract_concurrency_features(context);
        features.extend(concurrency_features);

        // Stability and maturity features
        let stability_features = self.extract_stability_features(context);
        features.extend(stability_features);

        // Normalize and pad features to desired dimension
        features = self.normalize_features(features)?;
        features.resize(self.config.embedding_config.dimensions, 0.0);

        let feature_array = Array1::from_vec(features);

        // Cache the result
        self.feature_cache
            .insert(context.trait_name.clone(), feature_array.clone());

        Ok(feature_array)
    }

    /// Extract features from trait information
    pub fn extract_trait_features(&mut self, trait_info: &TraitInfo) -> Result<Array1<f64>> {
        let mut features = Vec::new();

        // Method count and complexity
        features.push(trait_info.methods.len() as f64);

        // Average method complexity
        let avg_complexity = if trait_info.methods.is_empty() {
            0.0
        } else {
            trait_info
                .methods
                .iter()
                .map(|m| self.calculate_method_complexity(m))
                .sum::<f64>()
                / trait_info.methods.len() as f64
        };
        features.push(avg_complexity);

        // Generic parameter count
        features.push(trait_info.generics.len() as f64);

        // Documentation quality score
        let doc_score = self.calculate_documentation_quality(&trait_info.description);
        features.push(doc_score);

        // Text features from trait name and description
        let combined_text = format!("{} {}", trait_info.name, trait_info.description);
        let text_features = self.extract_text_features(&combined_text)?;
        features.extend(text_features.iter());

        // Method signature complexity
        let signature_complexity = self.calculate_signature_complexity(&trait_info.methods);
        features.push(signature_complexity);

        // Associated types and constants
        features.push(trait_info.associated_types.len() as f64);
        features.push(trait_info.constants.len() as f64);

        // Normalize and pad features
        features = self.normalize_features(features)?;
        features.resize(self.config.embedding_config.dimensions, 0.0);

        Ok(Array1::from_vec(features))
    }

    /// Extract basic numerical features from trait context
    fn extract_basic_features(&self, context: &TraitContext) -> Vec<f64> {
        vec![
            context.complexity_score,
            context.usage_frequency as f64 / 10000.0, // Normalize
            context.performance_impact,
            context.learning_curve_difficulty,
            if context.is_experimental { 1.0 } else { 0.0 },
            context.community_adoption_rate,
            context.documentation_quality,
            context.keywords.len() as f64 / 10.0, // Normalize
            context.related_crates.len() as f64 / 5.0, // Normalize
            context.feature_flags.len() as f64 / 3.0, // Normalize
        ]
    }

    /// Extract text-based features using TF-IDF and n-grams
    fn extract_text_features(&mut self, text: &str) -> Result<Vec<f64>> {
        // Check cache first
        if let Some(cached) = self.ngram_cache.get(text) {
            return Ok(cached.clone());
        }

        let processed_text = self.preprocess_text(text);
        let words: Vec<String> = processed_text.split_whitespace().map(String::from).collect();

        let mut features = vec![0.0; 64]; // Fixed size for text features

        // Build vocabulary if empty
        if self.vocabulary.is_empty() {
            self.build_vocabulary(&[&processed_text]);
        }

        // TF-IDF features
        let tfidf_features = self.calculate_tfidf_features(&words)?;
        features[..tfidf_features.len().min(32)].copy_from_slice(&tfidf_features[..tfidf_features.len().min(32)]);

        // Character-level n-gram features
        let char_features = self.extract_character_ngrams(&processed_text)?;
        features[32..].copy_from_slice(&char_features[..32.min(char_features.len())]);

        // Word-level n-gram features
        let word_ngram_features = self.extract_word_ngrams(&words)?;
        for (i, feature) in word_ngram_features.iter().take(features.len()).enumerate() {
            features[i] = (features[i] + feature) / 2.0; // Combine with existing features
        }

        // Cache the result
        self.ngram_cache.insert(text.to_string(), features.clone());

        Ok(features)
    }

    /// Extract keyword-based features
    fn extract_keyword_features(&self, keywords: &[String]) -> Result<Vec<f64>> {
        let mut features = vec![0.0; 16]; // Fixed size for keyword features

        // Predefined important keywords and their weights
        let important_keywords = vec![
            ("async", 1.0),
            ("sync", 0.8),
            ("iterator", 0.9),
            ("error", 0.7),
            ("serialize", 0.8),
            ("clone", 0.6),
            ("debug", 0.5),
            ("hash", 0.7),
            ("ord", 0.6),
            ("eq", 0.6),
            ("send", 0.9),
            ("copy", 0.5),
            ("display", 0.4),
            ("from", 0.7),
            ("into", 0.7),
            ("default", 0.5),
        ];

        for (i, (keyword, weight)) in important_keywords.iter().take(features.len()).enumerate() {
            if keywords.iter().any(|k| k.to_lowercase().contains(&keyword.to_lowercase())) {
                features[i] = *weight;
            }
        }

        Ok(features)
    }

    /// Extract semantic features using neural embeddings
    fn extract_semantic_features(
        &self,
        context: &TraitContext,
        model: &NeuralEmbeddingModel,
    ) -> Result<Array1<f64>> {
        // Create input vector from basic features
        let basic_features = self.extract_basic_features(context);
        let input = Array1::from_vec(basic_features);

        // Pad input to required dimension
        let input_dim = model.get_input_dim();
        let mut padded_input = Array1::zeros(input_dim);
        let copy_len = input.len().min(input_dim);
        padded_input
            .slice_mut(s![..copy_len])
            .assign(&input.slice(s![..copy_len]));

        model.forward(&padded_input)
    }

    /// Extract statistical features from context
    fn extract_statistical_features(&mut self, context: &TraitContext) -> Result<Vec<f64>> {
        let cache_key = format!("stats_{}", context.trait_name);

        // Check cache first
        if let Some(cached) = self.stats_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut features = Vec::new();

        // Variance and standard deviation of numerical features
        let numerical_values = vec![
            context.complexity_score,
            context.performance_impact,
            context.learning_curve_difficulty,
            context.community_adoption_rate,
            context.documentation_quality,
        ];

        let stats = self.calculate_descriptive_statistics(&numerical_values);
        features.extend(stats);

        // Distribution features
        let distribution_features = self.calculate_distribution_features(&numerical_values);
        features.extend(distribution_features);

        // Time-based features (if applicable)
        let time_features = self.extract_time_based_features(context);
        features.extend(time_features);

        // Cache the result
        self.stats_cache.insert(cache_key, features.clone());

        Ok(features)
    }

    /// Extract API complexity features
    fn extract_api_complexity_features(&self, context: &TraitContext) -> Vec<f64> {
        let api = &context.api_complexity;

        vec![
            api.method_count as f64 / 20.0, // Normalize by typical max
            api.associated_type_count as f64 / 5.0,
            api.generic_parameter_count as f64 / 3.0,
            if api.has_default_implementations { 1.0 } else { 0.0 },
            if api.has_blanket_implementations { 1.0 } else { 0.0 },
            if api.requires_unsafe { 1.0 } else { 0.0 },
        ]
    }

    /// Extract memory characteristics features
    fn extract_memory_features(&self, context: &TraitContext) -> Vec<f64> {
        let memory = &context.memory_characteristics;

        let overhead_score = match memory.memory_overhead {
            MemoryOverhead::None => 0.0,
            MemoryOverhead::Low => 0.25,
            MemoryOverhead::Medium => 0.5,
            MemoryOverhead::High => 1.0,
        };

        let allocation_score = match memory.allocation_pattern {
            AllocationPattern::StackOnly => 0.0,
            AllocationPattern::HeapOptional => 0.33,
            AllocationPattern::HeapRequired => 0.66,
            AllocationPattern::ZeroCopy => 1.0,
        };

        let cache_score = match memory.cache_efficiency {
            CacheEfficiency::Excellent => 1.0,
            CacheEfficiency::Good => 0.75,
            CacheEfficiency::Average => 0.5,
            CacheEfficiency::Poor => 0.0,
        };

        vec![overhead_score, allocation_score, cache_score]
    }

    /// Extract concurrency safety features
    fn extract_concurrency_features(&self, context: &TraitContext) -> Vec<f64> {
        let concurrency = &context.concurrency_safety;

        let thread_safety_score = match concurrency.thread_safety {
            ThreadSafety::NotThreadSafe => 0.0,
            ThreadSafety::ThreadLocal => 0.25,
            ThreadSafety::ThreadSafe => 0.75,
            ThreadSafety::LockFree => 1.0,
        };

        vec![
            thread_safety_score,
            if concurrency.requires_synchronization { 1.0 } else { 0.0 },
            if concurrency.lock_free_available { 1.0 } else { 0.0 },
            if concurrency.send_sync_bounds.implements_send { 1.0 } else { 0.0 },
            if concurrency.send_sync_bounds.implements_sync { 1.0 } else { 0.0 },
            concurrency.send_sync_bounds.conditional_bounds.len() as f64 / 3.0,
        ]
    }

    /// Extract stability and maturity features
    fn extract_stability_features(&self, context: &TraitContext) -> Vec<f64> {
        let stability_score = match context.stability_level {
            StabilityLevel::Experimental => 0.0,
            StabilityLevel::Unstable => 0.33,
            StabilityLevel::Stable => 1.0,
            StabilityLevel::Deprecated => -0.5,
        };

        vec![
            stability_score,
            context.community_adoption_rate,
            context.usage_frequency as f64 / 100000.0, // Large scale normalization
        ]
    }

    /// Preprocess text for feature extraction
    fn preprocess_text(&self, text: &str) -> String {
        let mut processed = text.to_lowercase();

        // Remove punctuation if configured
        if self.config.text_processing.remove_stop_words {
            processed = self.remove_stop_words(&processed);
        }

        // Apply stemming if configured
        if self.config.text_processing.enable_stemming {
            processed = self.apply_stemming(&processed);
        }

        // Filter by word length
        let words: Vec<&str> = processed
            .split_whitespace()
            .filter(|word| {
                word.len() >= self.config.text_processing.min_word_length
                    && word.len() <= self.config.text_processing.max_word_length
            })
            .collect();

        words.join(" ")
    }

    /// Remove stop words from text
    fn remove_stop_words(&self, text: &str) -> String {
        let stop_words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "among", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those",
        ];

        text.split_whitespace()
            .filter(|word| !stop_words.contains(word))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Apply basic stemming
    fn apply_stemming(&self, text: &str) -> String {
        // Simple suffix removal stemming
        text.split_whitespace()
            .map(|word| {
                if word.ends_with("ing") && word.len() > 4 {
                    &word[..word.len() - 3]
                } else if word.ends_with("ed") && word.len() > 3 {
                    &word[..word.len() - 2]
                } else if word.ends_with("s") && word.len() > 2 {
                    &word[..word.len() - 1]
                } else {
                    word
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Calculate TF-IDF features for words
    fn calculate_tfidf_features(&self, words: &[String]) -> Result<Vec<f64>> {
        let mut features = vec![0.0; 32];

        for (i, word) in words.iter().take(32).enumerate() {
            if let Some(&_vocab_idx) = self.vocabulary.get(word) {
                let tf = words.iter().filter(|w| *w == word).count() as f64 / words.len() as f64;
                let idf = self.tfidf_weights.get(word).unwrap_or(&1.0);

                // Apply TF weighting scheme
                let weighted_tf = match self.config.text_processing.tf_weighting {
                    TfWeighting::Raw => tf,
                    TfWeighting::Log => (1.0 + tf).ln(),
                    TfWeighting::DoubleNormalization => 0.5 + 0.5 * tf,
                    TfWeighting::Binary => if tf > 0.0 { 1.0 } else { 0.0 },
                };

                features[i] = if self.config.text_processing.idf_weighting {
                    weighted_tf * idf
                } else {
                    weighted_tf
                };
            }
        }

        Ok(features)
    }

    /// Extract character-level n-gram features
    fn extract_character_ngrams(&self, text: &str) -> Result<Vec<f64>> {
        let mut features = vec![0.0; 32];
        let text_bytes = text.as_bytes();

        // Character frequency features
        for (i, &byte) in text_bytes.iter().take(32).enumerate() {
            features[i] = byte as f64 / 255.0; // Normalize to [0,1]
        }

        // N-gram features for configured ranges
        for &(n_min, n_max) in &self.config.text_processing.ngram_ranges {
            for n in n_min..=n_max {
                if n <= text.len() {
                    for i in 0..=(text.len() - n) {
                        let ngram = &text[i..i + n];
                        let hash = self.simple_hash(ngram) % features.len();
                        features[hash] += 1.0 / text.len() as f64;
                    }
                }
            }
        }

        Ok(features)
    }

    /// Extract word-level n-gram features
    fn extract_word_ngrams(&self, words: &[String]) -> Result<Vec<f64>> {
        let mut features = vec![0.0; 32];

        for &(n_min, n_max) in &self.config.text_processing.ngram_ranges {
            for n in n_min..=n_max {
                if n <= words.len() {
                    for i in 0..=(words.len() - n) {
                        let ngram = words[i..i + n].join(" ");
                        let hash = self.simple_hash(&ngram) % features.len();
                        features[hash] += 1.0 / words.len() as f64;
                    }
                }
            }
        }

        Ok(features)
    }

    /// Simple hash function for n-grams
    fn simple_hash(&self, text: &str) -> usize {
        text.bytes().map(|b| b as usize).sum::<usize>()
    }

    /// Calculate descriptive statistics
    fn calculate_descriptive_statistics(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![0.0; 5];
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let std_dev = variance.sqrt();

        // Skewness approximation
        let skewness = if std_dev > 0.0 {
            values
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / values.len() as f64
        } else {
            0.0
        };

        // Kurtosis approximation
        let kurtosis = if std_dev > 0.0 {
            values
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / values.len() as f64
                - 3.0
        } else {
            0.0
        };

        vec![mean, variance, std_dev, skewness, kurtosis]
    }

    /// Calculate distribution features
    fn calculate_distribution_features(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![0.0; 3];
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted_values[0];
        let max_val = sorted_values[sorted_values.len() - 1];
        let range = max_val - min_val;

        vec![min_val, max_val, range]
    }

    /// Extract time-based features (placeholder for future enhancement)
    fn extract_time_based_features(&self, _context: &TraitContext) -> Vec<f64> {
        // Placeholder for time-based features like:
        // - Age of the trait
        // - Adoption velocity
        // - Version stability
        vec![0.0; 3]
    }

    /// Calculate method complexity score
    fn calculate_method_complexity(&self, method: &MethodInfo) -> f64 {
        // Parameter complexity
        let param_complexity = method.parameters.len() as f64 * 0.1;

        // Return type complexity
        let return_complexity = if method.return_type.contains("Result") {
            0.2
        } else if method.return_type.contains("Option") {
            0.1
        } else {
            0.0
        };

        // Generic complexity
        let generic_complexity = method.return_type.matches('<').count() as f64 * 0.1;

        // Lifetime complexity
        let lifetime_complexity = method.return_type.matches('\'').count() as f64 * 0.05;

        (param_complexity + return_complexity + generic_complexity + lifetime_complexity).min(1.0)
    }

    /// Calculate documentation quality score
    fn calculate_documentation_quality(&self, doc: &str) -> f64 {
        if doc.is_empty() {
            return 0.0;
        }

        let word_count = doc.split_whitespace().count();
        let has_examples = doc.contains("example") || doc.contains("Example");
        let has_errors = doc.contains("Error") || doc.contains("error");
        let has_panics = doc.contains("Panic") || doc.contains("panic");
        let has_safety = doc.contains("Safety") || doc.contains("safety");
        let has_usage = doc.contains("Usage") || doc.contains("usage");

        let mut score = (word_count as f64 / 100.0).min(1.0); // Base score from length

        if has_examples { score += 0.2; }
        if has_errors { score += 0.1; }
        if has_panics { score += 0.1; }
        if has_safety { score += 0.1; }
        if has_usage { score += 0.1; }

        score.min(1.0)
    }

    /// Calculate signature complexity for multiple methods
    fn calculate_signature_complexity(&self, methods: &[MethodInfo]) -> f64 {
        if methods.is_empty() {
            return 0.0;
        }

        let total_complexity: f64 = methods
            .iter()
            .map(|method| self.calculate_method_complexity(method))
            .sum();

        total_complexity / methods.len() as f64
    }

    /// Build vocabulary from text corpus
    fn build_vocabulary(&mut self, texts: &[&str]) {
        let mut word_counts = HashMap::new();
        let mut doc_counts = HashMap::new();

        for text in texts {
            let words: Vec<&str> = text.split_whitespace().collect();
            let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();

            for word in &words {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }

            for word in unique_words {
                *doc_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Build vocabulary with most frequent words
        let mut word_freq: Vec<(String, usize)> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        for (i, (word, _)) in word_freq.iter().take(1000).enumerate() {
            self.vocabulary.insert(word.clone(), i);
        }

        // Calculate IDF weights
        let total_docs = texts.len() as f64;
        for (word, doc_count) in doc_counts {
            let idf = (total_docs / doc_count as f64).ln();
            self.tfidf_weights.insert(word, idf);
        }
    }

    /// Normalize features using the configured normalization method
    fn normalize_features(&self, mut features: Vec<f64>) -> Result<Vec<f64>> {
        match self.config.normalization.method {
            NormalizationMethod::StandardScaling => {
                let mean = features.iter().sum::<f64>() / features.len() as f64;
                let variance = features
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / features.len() as f64;
                let std_dev = variance.sqrt().max(1e-8);

                for feature in &mut features {
                    *feature = (*feature - mean) / std_dev;
                }
            }
            NormalizationMethod::MinMaxScaling => {
                let min_val = features.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = features.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let range = (max_val - min_val).max(1e-8);

                for feature in &mut features {
                    *feature = (*feature - min_val) / range;
                }
            }
            NormalizationMethod::RobustScaling => {
                let mut sorted_features = features.clone();
                sorted_features.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = sorted_features[sorted_features.len() / 2];
                let q25 = sorted_features[sorted_features.len() / 4];
                let q75 = sorted_features[3 * sorted_features.len() / 4];
                let iqr = (q75 - q25).max(1e-8);

                for feature in &mut features {
                    *feature = (*feature - median) / iqr;
                }
            }
            NormalizationMethod::UnitVector => {
                let norm = features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt().max(1e-8);
                for feature in &mut features {
                    *feature = *feature / norm;
                }
            }
            _ => {
                // Other normalization methods would be implemented here
            }
        }

        // Apply clipping if configured
        if let Some((min_bound, max_bound)) = self.config.normalization.clipping_bounds {
            for feature in &mut features {
                *feature = feature.max(min_bound).min(max_bound);
            }
        }

        Ok(features)
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.feature_cache.clear();
        self.ngram_cache.clear();
        self.stats_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        (
            self.feature_cache.len(),
            self.ngram_cache.len(),
            self.stats_cache.len(),
        )
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get configuration
    pub fn get_config(&self) -> &FeatureExtractionConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: FeatureExtractionConfig) {
        self.config = config;
        // Clear caches as configuration change may affect feature extraction
        self.clear_cache();
    }
}

impl Default for TraitFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            text_processing: TextProcessingConfig::default(),
            embedding_config: EmbeddingConfig::default(),
            feature_selection: FeatureSelectionConfig::default(),
            normalization: NormalizationConfig::default(),
        }
    }
}

impl Default for TextProcessingConfig {
    fn default() -> Self {
        Self {
            enable_stemming: true,
            enable_lemmatization: false,
            remove_stop_words: true,
            min_word_length: 2,
            max_word_length: 20,
            ngram_ranges: vec![(1, 1), (2, 2)],
            tf_weighting: TfWeighting::Log,
            idf_weighting: true,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimensions: 256,
            pretrained_model: None,
            fine_tuning: None,
            context_window: 512,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            selection_method: FeatureSelectionMethod::VarianceThreshold,
            num_features: Some(100),
            selection_threshold: Some(0.01),
            cv_folds: 5,
        }
    }
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            method: NormalizationMethod::StandardScaling,
            per_feature: true,
            clipping_bounds: Some((-3.0, 3.0)),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = TraitFeatureExtractor::new();
        assert_eq!(extractor.vocabulary_size(), 0);
        assert_eq!(extractor.cache_stats(), (0, 0, 0));
    }

    #[test]
    fn test_text_preprocessing() {
        let extractor = TraitFeatureExtractor::new();
        let processed = extractor.preprocess_text("This is a TEST with UPPERCASE");
        assert!(processed.contains("test"));
        assert!(processed.contains("uppercase"));
    }

    #[test]
    fn test_documentation_quality_scoring() {
        let extractor = TraitFeatureExtractor::new();

        let empty_doc = "";
        assert_eq!(extractor.calculate_documentation_quality(empty_doc), 0.0);

        let good_doc = "This trait provides example usage with error handling and panic safety";
        let score = extractor.calculate_documentation_quality(good_doc);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_method_complexity_calculation() {
        let extractor = TraitFeatureExtractor::new();

        let simple_method = MethodInfo {
            name: "simple".to_string(),
            parameters: vec![],
            return_type: "()".to_string(),
            description: "Simple method".to_string(),
            examples: vec![],
        };

        let complex_method = MethodInfo {
            name: "complex".to_string(),
            parameters: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            return_type: "Result<Option<Vec<T>>, Error>".to_string(),
            description: "Complex method".to_string(),
            examples: vec![],
        };

        let simple_complexity = extractor.calculate_method_complexity(&simple_method);
        let complex_complexity = extractor.calculate_method_complexity(&complex_method);

        assert!(complex_complexity > simple_complexity);
        assert!(complex_complexity <= 1.0);
    }

    #[test]
    fn test_feature_normalization() {
        let extractor = TraitFeatureExtractor::new();
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let normalized = extractor.normalize_features(features).expect("normalize_features should succeed");

        // For standard scaling, mean should be approximately 0
        let mean = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!((mean).abs() < 1e-10);
    }

    #[test]
    fn test_basic_features_extraction() {
        let extractor = TraitFeatureExtractor::new();
        let context = TraitContext {
            trait_name: "TestTrait".to_string(),
            description: "A test trait".to_string(),
            complexity_score: 0.5,
            usage_frequency: 1000,
            performance_impact: 0.2,
            learning_curve_difficulty: 0.3,
            is_experimental: false,
            community_adoption_rate: 0.8,
            keywords: vec!["test".to_string()],
            related_crates: vec!["crate1".to_string()],
            rust_version_requirement: None,
            feature_flags: vec![],
            documentation_quality: 0.7,
            stability_level: StabilityLevel::Stable,
            api_complexity: ApiComplexity {
                method_count: 3,
                associated_type_count: 1,
                generic_parameter_count: 0,
                has_default_implementations: true,
                has_blanket_implementations: false,
                requires_unsafe: false,
            },
            memory_characteristics: MemoryCharacteristics {
                memory_overhead: MemoryOverhead::Low,
                allocation_pattern: AllocationPattern::StackOnly,
                cache_efficiency: CacheEfficiency::Good,
            },
            concurrency_safety: ConcurrencySafety {
                thread_safety: ThreadSafety::ThreadSafe,
                requires_synchronization: false,
                lock_free_available: true,
                send_sync_bounds: SendSyncBounds {
                    implements_send: true,
                    implements_sync: true,
                    conditional_bounds: vec![],
                },
            },
        };

        let basic_features = extractor.extract_basic_features(&context);
        assert_eq!(basic_features.len(), 10);
        assert!(basic_features.iter().all(|&f| f >= 0.0));
    }
}