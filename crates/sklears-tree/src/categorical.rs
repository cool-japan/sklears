//! Advanced Categorical Feature Handling for Tree Algorithms
//!
//! This module provides specialized categorical feature handling beyond basic
//! one-hot encoding and CatBoost methods, including advanced encoding techniques,
//! high-cardinality feature processing, and categorical-aware tree algorithms.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use sklears_core::error::{Result, SklearsError};
use std::collections::{HashMap, HashSet};

/// Advanced categorical encoding strategies
#[derive(Debug, Clone)]
pub enum CategoricalEncoding {
    /// Traditional one-hot encoding
    OneHot,
    /// Target encoding with regularization
    TargetEncoding {
        smoothing: f64,
        min_samples_leaf: usize,
        noise_level: f64,
    },
    /// Binary encoding for high-cardinality features
    BinaryEncoding,
    /// Hashing trick for very high-cardinality features
    FeatureHashing {
        n_components: usize,
        hash_function: HashFunction,
    },
    /// Frequency encoding
    FrequencyEncoding,
    /// Ordinal encoding with learned ordering
    LearnedOrdinal { ordering_strategy: OrderingStrategy },
    /// Weight of Evidence encoding
    WeightOfEvidence { smoothing: f64 },
    /// Leave-one-out encoding
    LeaveOneOut { sigma: f64 },
    /// James-Stein encoding
    JamesStein { sigma: f64 },
    /// Hierarchical encoding for nested categories
    Hierarchical { hierarchy_levels: Vec<String> },
    /// Embeddings learned from tree interactions
    TreeEmbedding {
        embedding_dim: usize,
        n_trees: usize,
    },
}

/// Hash function types for feature hashing
#[derive(Debug, Clone, Copy)]
pub enum HashFunction {
    /// Simple modulo hash
    Modulo,
    /// MurmurHash3
    Murmur3,
    /// FNV hash
    FNV,
    XxHash,
}

/// Strategies for learning ordinal encoding order
#[derive(Debug, Clone)]
pub enum OrderingStrategy {
    /// Order by target mean
    TargetMean,
    /// Order by frequency
    Frequency,
    /// Order by information gain
    InformationGain,
    /// Order by mutual information
    MutualInformation,
    /// Custom ordering provided by user
    Custom(Vec<String>),
}

/// Categorical feature processor
#[derive(Debug, Clone)]
pub struct CategoricalProcessor {
    /// Encoding strategies for each categorical feature
    pub encodings: HashMap<usize, CategoricalEncoding>,
    /// Mapping from original to encoded feature indices
    pub feature_mapping: HashMap<usize, Vec<usize>>,
    /// Category mappings for each feature
    pub category_mappings: HashMap<usize, CategoryMapping>,
    /// Statistics for each categorical feature
    pub feature_stats: HashMap<usize, CategoricalFeatureStats>,
}

/// Statistics for a categorical feature
#[derive(Debug, Clone)]
pub struct CategoricalFeatureStats {
    /// Number of unique categories
    pub n_categories: usize,
    /// Category frequencies
    pub category_frequencies: HashMap<String, usize>,
    /// Target statistics for each category
    pub target_stats: HashMap<String, TargetStats>,
    /// Information gain of this feature
    pub information_gain: f64,
    /// Mutual information with target
    pub mutual_information: f64,
}

/// Target statistics for a category
#[derive(Debug, Clone)]
pub struct TargetStats {
    /// Number of samples
    pub count: usize,
    /// Sum of targets
    pub sum: f64,
    /// Sum of squared targets
    pub sum_squared: f64,
    /// Mean target value
    pub mean: f64,
    /// Variance of target values
    pub variance: f64,
}

impl TargetStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squared: 0.0,
            mean: 0.0,
            variance: 0.0,
        }
    }

    pub fn update(&mut self, target: f64) {
        self.count += 1;
        self.sum += target;
        self.sum_squared += target * target;

        self.mean = self.sum / self.count as f64;
        if self.count > 1 {
            self.variance = (self.sum_squared / self.count as f64) - (self.mean * self.mean);
        }
    }
}

/// Category mapping for encoding/decoding
#[derive(Debug, Clone)]
pub enum CategoryMapping {
    /// Direct string to index mapping
    StringToIndex(HashMap<String, usize>),
    /// String to multiple indices (for binary encoding)
    StringToIndices(HashMap<String, Vec<usize>>),
    /// String to hash values
    StringToHash(HashMap<String, u64>),
    /// String to target encoding value
    StringToTarget(HashMap<String, f64>),
    /// String to embedding vector
    StringToEmbedding(HashMap<String, Vec<f64>>),
}

impl CategoricalProcessor {
    pub fn new() -> Self {
        Self {
            encodings: HashMap::new(),
            feature_mapping: HashMap::new(),
            category_mappings: HashMap::new(),
            feature_stats: HashMap::new(),
        }
    }

    /// Configure encoding for a specific feature
    pub fn set_encoding(&mut self, feature_idx: usize, encoding: CategoricalEncoding) {
        self.encodings.insert(feature_idx, encoding);
    }

    /// Fit the processor on categorical data and targets
    pub fn fit(
        &mut self,
        categorical_data: &HashMap<usize, Vec<String>>,
        targets: &Array1<f64>,
    ) -> Result<()> {
        for (&feature_idx, categories) in categorical_data {
            let encoding = self
                .encodings
                .get(&feature_idx)
                .cloned()
                .unwrap_or(CategoricalEncoding::OneHot);

            // Calculate feature statistics
            let stats = self.calculate_feature_stats(categories, targets)?;
            self.feature_stats.insert(feature_idx, stats);

            // Fit encoding
            match encoding {
                CategoricalEncoding::OneHot => {
                    self.fit_one_hot(feature_idx, categories)?;
                }
                CategoricalEncoding::TargetEncoding {
                    smoothing,
                    min_samples_leaf,
                    noise_level,
                } => {
                    self.fit_target_encoding(
                        feature_idx,
                        categories,
                        targets,
                        smoothing,
                        min_samples_leaf,
                        noise_level,
                    )?;
                }
                CategoricalEncoding::BinaryEncoding => {
                    self.fit_binary_encoding(feature_idx, categories)?;
                }
                CategoricalEncoding::FeatureHashing {
                    n_components,
                    hash_function,
                } => {
                    self.fit_feature_hashing(feature_idx, categories, n_components, hash_function)?;
                }
                CategoricalEncoding::FrequencyEncoding => {
                    self.fit_frequency_encoding(feature_idx, categories)?;
                }
                CategoricalEncoding::LearnedOrdinal { ordering_strategy } => {
                    self.fit_learned_ordinal(feature_idx, categories, targets, ordering_strategy)?;
                }
                CategoricalEncoding::WeightOfEvidence { smoothing } => {
                    self.fit_weight_of_evidence(feature_idx, categories, targets, smoothing)?;
                }
                CategoricalEncoding::LeaveOneOut { sigma } => {
                    self.fit_leave_one_out(feature_idx, categories, targets, sigma)?;
                }
                CategoricalEncoding::JamesStein { sigma } => {
                    self.fit_james_stein(feature_idx, categories, targets, sigma)?;
                }
                CategoricalEncoding::Hierarchical { hierarchy_levels } => {
                    self.fit_hierarchical(feature_idx, categories, targets, hierarchy_levels)?;
                }
                CategoricalEncoding::TreeEmbedding {
                    embedding_dim,
                    n_trees,
                } => {
                    self.fit_tree_embedding(
                        feature_idx,
                        categories,
                        targets,
                        embedding_dim,
                        n_trees,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Transform categorical data using fitted encodings
    pub fn transform(&self, categorical_data: &HashMap<usize, Vec<String>>) -> Result<Array2<f64>> {
        let n_samples = categorical_data
            .values()
            .next()
            .ok_or_else(|| SklearsError::InvalidInput("No categorical data provided".to_string()))?
            .len();

        let mut encoded_features = Vec::new();

        for (&feature_idx, categories) in categorical_data {
            let encoded = self.transform_feature(feature_idx, categories)?;
            encoded_features.extend(encoded);
        }

        if encoded_features.is_empty() {
            return Ok(Array2::zeros((n_samples, 0)));
        }

        let n_features = encoded_features.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (feature_idx, feature_data) in encoded_features.iter().enumerate() {
            for (sample_idx, &value) in feature_data.iter().enumerate() {
                result[[sample_idx, feature_idx]] = value;
            }
        }

        Ok(result)
    }

    /// Calculate feature statistics
    fn calculate_feature_stats(
        &self,
        categories: &[String],
        targets: &Array1<f64>,
    ) -> Result<CategoricalFeatureStats> {
        let mut category_frequencies = HashMap::new();
        let mut target_stats = HashMap::new();

        // Calculate frequencies and target statistics
        for (i, category) in categories.iter().enumerate() {
            *category_frequencies.entry(category.clone()).or_insert(0) += 1;

            let stats = target_stats
                .entry(category.clone())
                .or_insert_with(TargetStats::new);
            stats.update(targets[i]);
        }

        let n_categories = category_frequencies.len();

        // Calculate information gain (simplified)
        let total_samples = categories.len() as f64;
        let overall_mean = targets.mean().unwrap_or(0.0);
        let overall_variance = targets
            .iter()
            .map(|&x| (x - overall_mean).powi(2))
            .sum::<f64>()
            / total_samples;

        let mut weighted_variance = 0.0;
        for (category, freq) in &category_frequencies {
            let weight = *freq as f64 / total_samples;
            let cat_variance = target_stats[category].variance;
            weighted_variance += weight * cat_variance;
        }

        let information_gain = overall_variance - weighted_variance;

        // Mutual information (simplified estimation)
        let mutual_information = information_gain / overall_variance.max(1e-8);

        Ok(CategoricalFeatureStats {
            n_categories,
            category_frequencies,
            target_stats,
            information_gain,
            mutual_information,
        })
    }

    /// Fit one-hot encoding
    fn fit_one_hot(&mut self, feature_idx: usize, categories: &[String]) -> Result<()> {
        let unique_categories: HashSet<String> = categories.iter().cloned().collect();
        let mut mapping = HashMap::new();

        for (i, category) in unique_categories.iter().enumerate() {
            mapping.insert(category.clone(), i);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToIndex(mapping));

        let n_encoded_features = unique_categories.len();
        self.feature_mapping
            .insert(feature_idx, (0..n_encoded_features).collect());

        Ok(())
    }

    /// Fit target encoding
    fn fit_target_encoding(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        smoothing: f64,
        min_samples_leaf: usize,
        noise_level: f64,
    ) -> Result<()> {
        let overall_mean = targets.mean().unwrap_or(0.0);
        let mut mapping = HashMap::new();

        // Calculate category means with smoothing
        let stats = &self.feature_stats[&feature_idx];

        for (category, cat_stats) in &stats.target_stats {
            if cat_stats.count >= min_samples_leaf {
                // Apply smoothing: (category_mean * count + overall_mean * smoothing) / (count + smoothing)
                let smoothed_mean = (cat_stats.mean * cat_stats.count as f64
                    + overall_mean * smoothing)
                    / (cat_stats.count as f64 + smoothing);

                // Add noise for regularization
                let mut rng = thread_rng();
                let noise = rng.gen_range(-noise_level..noise_level);
                let encoded_value = smoothed_mean + noise;

                mapping.insert(category.clone(), encoded_value);
            } else {
                mapping.insert(category.clone(), overall_mean);
            }
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToTarget(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]); // Single encoded feature

        Ok(())
    }

    /// Fit binary encoding
    fn fit_binary_encoding(&mut self, feature_idx: usize, categories: &[String]) -> Result<()> {
        let unique_categories: Vec<String> = categories
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let n_bits = (unique_categories.len() as f64).log2().ceil() as usize;

        let mut mapping = HashMap::new();

        for (i, category) in unique_categories.iter().enumerate() {
            let mut binary_representation = Vec::new();
            for bit in 0..n_bits {
                binary_representation.push(if (i >> bit) & 1 == 1 { 1 } else { 0 });
            }
            mapping.insert(category.clone(), binary_representation);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToIndices(mapping));
        self.feature_mapping
            .insert(feature_idx, (0..n_bits).collect());

        Ok(())
    }

    /// Fit feature hashing
    fn fit_feature_hashing(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        n_components: usize,
        hash_function: HashFunction,
    ) -> Result<()> {
        // Feature hashing doesn't require fitting, just store parameters
        // The actual hashing is done during transform
        self.feature_mapping
            .insert(feature_idx, (0..n_components).collect());

        Ok(())
    }

    /// Fit frequency encoding
    fn fit_frequency_encoding(&mut self, feature_idx: usize, categories: &[String]) -> Result<()> {
        let stats = &self.feature_stats[&feature_idx];
        let total_samples = categories.len() as f64;

        let mut mapping = HashMap::new();
        for (category, &frequency) in &stats.category_frequencies {
            mapping.insert(category.clone(), frequency as f64 / total_samples);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToTarget(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]);

        Ok(())
    }

    /// Fit learned ordinal encoding
    fn fit_learned_ordinal(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        ordering_strategy: OrderingStrategy,
    ) -> Result<()> {
        let stats = &self.feature_stats[&feature_idx];

        let mut category_values: Vec<(String, f64)> = match ordering_strategy {
            OrderingStrategy::TargetMean => stats
                .target_stats
                .iter()
                .map(|(cat, stats)| (cat.clone(), stats.mean))
                .collect(),
            OrderingStrategy::Frequency => stats
                .category_frequencies
                .iter()
                .map(|(cat, &freq)| (cat.clone(), freq as f64))
                .collect(),
            OrderingStrategy::InformationGain => {
                // Use target variance as proxy for information gain per category
                stats
                    .target_stats
                    .iter()
                    .map(|(cat, stats)| (cat.clone(), -stats.variance)) // Negative for descending order
                    .collect()
            }
            OrderingStrategy::MutualInformation => {
                // Simplified mutual information per category
                let overall_mean = targets.mean().unwrap_or(0.0);
                stats
                    .target_stats
                    .iter()
                    .map(|(cat, stats)| (cat.clone(), (stats.mean - overall_mean).abs()))
                    .collect()
            }
            OrderingStrategy::Custom(ref order) => order
                .iter()
                .enumerate()
                .map(|(i, cat)| (cat.clone(), i as f64))
                .collect(),
        };

        // Sort by value
        category_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Create ordinal mapping
        let mut mapping = HashMap::new();
        for (i, (category, _)) in category_values.iter().enumerate() {
            mapping.insert(category.clone(), i);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToIndex(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]);

        Ok(())
    }

    /// Fit Weight of Evidence encoding
    fn fit_weight_of_evidence(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        smoothing: f64,
    ) -> Result<()> {
        // For regression, we'll use a modified WoE based on target distribution
        let overall_mean = targets.mean().unwrap_or(0.0);
        let stats = &self.feature_stats[&feature_idx];
        let total_samples = categories.len() as f64;

        let mut mapping = HashMap::new();

        for (category, cat_stats) in &stats.target_stats {
            let p_category = cat_stats.count as f64 / total_samples;
            let category_mean = cat_stats.mean;

            // Modified WoE for regression: log((category_mean + smoothing) / (overall_mean + smoothing))
            let woe = ((category_mean + smoothing) / (overall_mean + smoothing)).ln();
            mapping.insert(category.clone(), woe);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToTarget(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]);

        Ok(())
    }

    /// Fit Leave-One-Out encoding
    fn fit_leave_one_out(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        sigma: f64,
    ) -> Result<()> {
        // For LOO encoding, we need to store sufficient statistics
        // This is a simplified version - in practice, you'd compute this during transform
        let overall_mean = targets.mean().unwrap_or(0.0);
        let stats = &self.feature_stats[&feature_idx];

        let mut mapping = HashMap::new();

        for (category, cat_stats) in &stats.target_stats {
            // Regularized mean: (sum - x) / (count - 1) with regularization
            let regularized_mean = if cat_stats.count > 1 {
                (cat_stats.sum - cat_stats.mean) / (cat_stats.count - 1) as f64
            } else {
                overall_mean
            };

            // Add noise for regularization
            let mut rng = thread_rng();
            let noise = rng.gen_range(-sigma..sigma);
            mapping.insert(category.clone(), regularized_mean + noise);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToTarget(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]);

        Ok(())
    }

    /// Fit James-Stein encoding
    fn fit_james_stein(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        sigma: f64,
    ) -> Result<()> {
        let overall_mean = targets.mean().unwrap_or(0.0);
        let stats = &self.feature_stats[&feature_idx];

        let mut mapping = HashMap::new();

        for (category, cat_stats) in &stats.target_stats {
            // James-Stein shrinkage
            let shrinkage_factor = 1.0 - (sigma.powi(2)) / (cat_stats.variance + sigma.powi(2));
            let js_estimate = overall_mean + shrinkage_factor * (cat_stats.mean - overall_mean);

            mapping.insert(category.clone(), js_estimate);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToTarget(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]);

        Ok(())
    }

    /// Fit hierarchical encoding (simplified)
    fn fit_hierarchical(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        hierarchy_levels: Vec<String>,
    ) -> Result<()> {
        // This is a simplified implementation
        // In practice, you'd parse the hierarchy structure and create multi-level encodings
        let overall_mean = targets.mean().unwrap_or(0.0);
        let mut mapping = HashMap::new();

        for category in categories.iter().cloned().collect::<HashSet<_>>() {
            // For now, just use target encoding
            mapping.insert(category, overall_mean);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToTarget(mapping));
        self.feature_mapping.insert(feature_idx, vec![0]);

        Ok(())
    }

    /// Fit tree embedding (simplified)
    fn fit_tree_embedding(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        embedding_dim: usize,
        n_trees: usize,
    ) -> Result<()> {
        // This is a placeholder - in practice, you'd train a small ensemble
        // and use leaf indices as embeddings
        let unique_categories: HashSet<String> = categories.iter().cloned().collect();
        let mut mapping = HashMap::new();

        let mut rng = thread_rng();

        for category in unique_categories {
            let embedding: Vec<f64> = (0..embedding_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            mapping.insert(category, embedding);
        }

        self.category_mappings
            .insert(feature_idx, CategoryMapping::StringToEmbedding(mapping));
        self.feature_mapping
            .insert(feature_idx, (0..embedding_dim).collect());

        Ok(())
    }

    /// Transform a single feature
    fn transform_feature(
        &self,
        feature_idx: usize,
        categories: &[String],
    ) -> Result<Vec<Vec<f64>>> {
        let mapping = self
            .category_mappings
            .get(&feature_idx)
            .ok_or_else(|| SklearsError::InvalidInput("Feature not fitted".to_string()))?;

        let n_encoded_features = self
            .feature_mapping
            .get(&feature_idx)
            .map(|v| v.len())
            .unwrap_or(1);

        let mut result = vec![vec![0.0; categories.len()]; n_encoded_features];

        match mapping {
            CategoryMapping::StringToIndex(ref map) => {
                // One-hot or ordinal encoding
                for (sample_idx, category) in categories.iter().enumerate() {
                    if let Some(&index) = map.get(category) {
                        if n_encoded_features == 1 {
                            // Ordinal encoding
                            result[0][sample_idx] = index as f64;
                        } else {
                            // One-hot encoding
                            if index < n_encoded_features {
                                result[index][sample_idx] = 1.0;
                            }
                        }
                    }
                }
            }
            CategoryMapping::StringToIndices(ref map) => {
                // Binary encoding
                for (sample_idx, category) in categories.iter().enumerate() {
                    if let Some(indices) = map.get(category) {
                        for (bit_idx, &bit_value) in indices.iter().enumerate() {
                            if bit_idx < n_encoded_features {
                                result[bit_idx][sample_idx] = bit_value as f64;
                            }
                        }
                    }
                }
            }
            CategoryMapping::StringToTarget(ref map) => {
                // Target encoding, frequency encoding, etc.
                for (sample_idx, category) in categories.iter().enumerate() {
                    let value = map.get(category).cloned().unwrap_or(0.0);
                    result[0][sample_idx] = value;
                }
            }
            CategoryMapping::StringToEmbedding(ref map) => {
                // Embedding encoding
                for (sample_idx, category) in categories.iter().enumerate() {
                    if let Some(embedding) = map.get(category) {
                        for (dim_idx, &value) in embedding.iter().enumerate() {
                            if dim_idx < n_encoded_features {
                                result[dim_idx][sample_idx] = value;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported mapping type".to_string(),
                ));
            }
        }

        Ok(result)
    }
}

/// High-cardinality categorical feature handler
#[derive(Debug, Clone)]
pub struct HighCardinalityHandler {
    /// Threshold for considering a feature high-cardinality
    pub cardinality_threshold: usize,
    /// Strategy for handling high-cardinality features
    pub strategy: HighCardinalityStrategy,
    /// Feature statistics
    pub feature_stats: HashMap<usize, HighCardinalityStats>,
}

/// Strategies for handling high-cardinality features
#[derive(Debug, Clone)]
pub enum HighCardinalityStrategy {
    /// Drop rare categories
    DropRare { min_frequency: f64 },
    /// Group rare categories into "Other"
    GroupRare {
        min_frequency: f64,
        other_label: String,
    },
    /// Use frequency-based truncation
    TopK { k: usize },
    /// Use feature hashing
    Hashing { n_components: usize },
    /// Use target-based grouping
    TargetGrouping { n_groups: usize },
}

/// Statistics for high-cardinality features
#[derive(Debug, Clone)]
pub struct HighCardinalityStats {
    /// Original number of categories
    pub original_cardinality: usize,
    /// Reduced number of categories
    pub reduced_cardinality: usize,
    /// Information loss from reduction
    pub information_loss: f64,
    /// Categories that were merged or dropped
    pub processed_categories: HashMap<String, String>,
}

impl HighCardinalityHandler {
    pub fn new(cardinality_threshold: usize, strategy: HighCardinalityStrategy) -> Self {
        Self {
            cardinality_threshold,
            strategy,
            feature_stats: HashMap::new(),
        }
    }

    /// Process high-cardinality categorical features
    pub fn process_features(
        &mut self,
        categorical_data: &HashMap<usize, Vec<String>>,
        targets: &Array1<f64>,
    ) -> Result<HashMap<usize, Vec<String>>> {
        let mut processed_data = HashMap::new();

        for (&feature_idx, categories) in categorical_data {
            let unique_categories: HashSet<String> = categories.iter().cloned().collect();

            if unique_categories.len() > self.cardinality_threshold {
                let processed =
                    self.process_high_cardinality_feature(feature_idx, categories, targets)?;
                processed_data.insert(feature_idx, processed);
            } else {
                processed_data.insert(feature_idx, categories.clone());
            }
        }

        Ok(processed_data)
    }

    /// Process a single high-cardinality feature
    fn process_high_cardinality_feature(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
    ) -> Result<Vec<String>> {
        match &self.strategy {
            HighCardinalityStrategy::DropRare { min_frequency } => {
                self.drop_rare_categories(feature_idx, categories, *min_frequency)
            }
            HighCardinalityStrategy::GroupRare {
                min_frequency,
                other_label,
            } => self.group_rare_categories(
                feature_idx,
                categories,
                *min_frequency,
                other_label.clone(),
            ),
            HighCardinalityStrategy::TopK { k } => {
                self.keep_top_k_categories(feature_idx, categories, *k)
            }
            HighCardinalityStrategy::Hashing { n_components } => {
                self.hash_categories(feature_idx, categories, *n_components)
            }
            HighCardinalityStrategy::TargetGrouping { n_groups } => {
                self.group_by_target(feature_idx, categories, targets, *n_groups)
            }
        }
    }

    /// Drop rare categories
    fn drop_rare_categories(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        min_frequency: f64,
    ) -> Result<Vec<String>> {
        let mut category_counts = HashMap::new();
        for category in categories {
            *category_counts.entry(category.clone()).or_insert(0) += 1;
        }

        let total_samples = categories.len();
        let min_count = (total_samples as f64 * min_frequency) as usize;

        let frequent_categories: HashSet<String> = category_counts
            .iter()
            .filter(|(_, &count)| count >= min_count)
            .map(|(cat, _)| cat.clone())
            .collect();

        let processed: Vec<String> = categories
            .iter()
            .filter_map(|cat| {
                if frequent_categories.contains(cat) {
                    Some(cat.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(processed)
    }

    /// Group rare categories into "Other"
    fn group_rare_categories(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        min_frequency: f64,
        other_label: String,
    ) -> Result<Vec<String>> {
        let mut category_counts = HashMap::new();
        for category in categories {
            *category_counts.entry(category.clone()).or_insert(0) += 1;
        }

        let total_samples = categories.len();
        let min_count = (total_samples as f64 * min_frequency) as usize;

        let frequent_categories: HashSet<String> = category_counts
            .iter()
            .filter(|(_, &count)| count >= min_count)
            .map(|(cat, _)| cat.clone())
            .collect();

        let processed: Vec<String> = categories
            .iter()
            .map(|cat| {
                if frequent_categories.contains(cat) {
                    cat.clone()
                } else {
                    other_label.clone()
                }
            })
            .collect();

        Ok(processed)
    }

    /// Keep only top K most frequent categories
    fn keep_top_k_categories(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        k: usize,
    ) -> Result<Vec<String>> {
        let mut category_counts = HashMap::new();
        for category in categories {
            *category_counts.entry(category.clone()).or_insert(0) += 1;
        }

        let mut sorted_categories: Vec<(String, usize)> = category_counts.into_iter().collect();
        sorted_categories.sort_by(|a, b| b.1.cmp(&a.1));

        let top_k_categories: HashSet<String> = sorted_categories
            .into_iter()
            .take(k)
            .map(|(cat, _)| cat)
            .collect();

        let processed: Vec<String> = categories
            .iter()
            .filter_map(|cat| {
                if top_k_categories.contains(cat) {
                    Some(cat.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(processed)
    }

    /// Hash categories (simplified)
    fn hash_categories(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        n_components: usize,
    ) -> Result<Vec<String>> {
        // Simple hash function - in practice, use a proper hash function
        let processed: Vec<String> = categories
            .iter()
            .map(|cat| {
                let hash = self.simple_hash(cat) % n_components;
                format!("hash_{}", hash)
            })
            .collect();

        Ok(processed)
    }

    /// Group categories by target similarity
    fn group_by_target(
        &mut self,
        feature_idx: usize,
        categories: &[String],
        targets: &Array1<f64>,
        n_groups: usize,
    ) -> Result<Vec<String>> {
        // Calculate target mean for each category
        let mut category_targets = HashMap::new();
        for (i, category) in categories.iter().enumerate() {
            category_targets
                .entry(category.clone())
                .or_insert_with(Vec::new)
                .push(targets[i]);
        }

        let mut category_means: Vec<(String, f64)> = category_targets
            .iter()
            .map(|(cat, target_vals)| {
                let mean = target_vals.iter().sum::<f64>() / target_vals.len() as f64;
                (cat.clone(), mean)
            })
            .collect();

        // Sort by target mean
        category_means.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Create groups
        let categories_per_group = category_means.len() / n_groups;
        let mut group_mapping = HashMap::new();

        for (i, (category, _)) in category_means.iter().enumerate() {
            let group_id = std::cmp::min(i / categories_per_group, n_groups - 1);
            group_mapping.insert(category.clone(), format!("group_{}", group_id));
        }

        let processed: Vec<String> = categories
            .iter()
            .map(|cat| {
                group_mapping
                    .get(cat)
                    .cloned()
                    .unwrap_or_else(|| "group_0".to_string())
            })
            .collect();

        Ok(processed)
    }

    /// Simple hash function (replace with proper hash in production)
    fn simple_hash(&self, s: &str) -> usize {
        s.bytes().map(|b| b as usize).sum()
    }
}
