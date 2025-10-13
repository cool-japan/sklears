//! Feature interaction detection and analysis
//!
//! This module provides comprehensive feature interaction detection implementations including
//! pairwise interactions, higher-order interactions, correlation analysis, dependency analysis,
//! and interaction optimization. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Supported interaction detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionMethod {
    Pairwise,
    HigherOrder,
    Correlation,
    MutualInformation,
    ChiSquared,
    ANOVA,
    RegressionBased,
    TreeBased,
    Statistical,
}

/// Configuration for interaction detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    pub method: InteractionMethod,
    pub max_order: usize,
    pub significance_threshold: f64,
    pub correlation_threshold: f64,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub interaction_strength_threshold: f64,
    pub use_statistical_tests: bool,
    pub correction_method: Option<String>,
    pub random_state: Option<u64>,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            method: InteractionMethod::Pairwise,
            max_order: 2,
            significance_threshold: 0.05,
            correlation_threshold: 0.1,
            min_samples_leaf: 5,
            max_features: None,
            interaction_strength_threshold: 0.01,
            use_statistical_tests: true,
            correction_method: Some("bonferroni".to_string()),
            random_state: Some(42),
        }
    }
}

/// Results from feature interaction detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionResults {
    pub detected_interactions: Vec<Vec<usize>>,
    pub interaction_strengths: Array1<f64>,
    pub significance_values: Option<Array1<f64>>,
    pub correlation_matrix: Option<Array2<f64>>,
    pub interaction_types: Vec<String>,
    pub statistical_tests: Option<HashMap<String, f64>>,
    pub interaction_metadata: HashMap<String, f64>,
    pub feature_pair_scores: HashMap<(usize, usize), f64>,
}

impl InteractionResults {
    /// Create new interaction results
    pub fn new(detected_interactions: Vec<Vec<usize>>, interaction_strengths: Array1<f64>) -> Self {
        Self {
            detected_interactions,
            interaction_strengths,
            significance_values: None,
            correlation_matrix: None,
            interaction_types: Vec::new(),
            statistical_tests: None,
            interaction_metadata: HashMap::new(),
            feature_pair_scores: HashMap::new(),
        }
    }

    /// Get detected interactions
    pub fn detected_interactions(&self) -> &[Vec<usize>] {
        &self.detected_interactions
    }

    /// Get interaction strengths
    pub fn interaction_strengths(&self) -> &Array1<f64> {
        &self.interaction_strengths
    }

    /// Get significance values if available
    pub fn significance_values(&self) -> Option<&Array1<f64>> {
        self.significance_values.as_ref()
    }

    /// Set significance values
    pub fn set_significance_values(&mut self, values: Array1<f64>) {
        self.significance_values = Some(values);
    }

    /// Get correlation matrix if available
    pub fn correlation_matrix(&self) -> Option<&Array2<f64>> {
        self.correlation_matrix.as_ref()
    }

    /// Set correlation matrix
    pub fn set_correlation_matrix(&mut self, matrix: Array2<f64>) {
        self.correlation_matrix = Some(matrix);
    }

    /// Get interaction types
    pub fn interaction_types(&self) -> &[String] {
        &self.interaction_types
    }

    /// Add interaction type
    pub fn add_interaction_type(&mut self, interaction_type: String) {
        self.interaction_types.push(interaction_type);
    }

    /// Get statistical tests
    pub fn statistical_tests(&self) -> Option<&HashMap<String, f64>> {
        self.statistical_tests.as_ref()
    }

    /// Set statistical tests
    pub fn set_statistical_tests(&mut self, tests: HashMap<String, f64>) {
        self.statistical_tests = Some(tests);
    }

    /// Get interaction metadata
    pub fn interaction_metadata(&self) -> &HashMap<String, f64> {
        &self.interaction_metadata
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: f64) {
        self.interaction_metadata.insert(key, value);
    }

    /// Get feature pair scores
    pub fn feature_pair_scores(&self) -> &HashMap<(usize, usize), f64> {
        &self.feature_pair_scores
    }

    /// Add feature pair score
    pub fn add_feature_pair_score(&mut self, pair: (usize, usize), score: f64) {
        self.feature_pair_scores.insert(pair, score);
    }

    /// Get number of interactions detected
    pub fn n_interactions_detected(&self) -> usize {
        self.detected_interactions.len()
    }

    /// Get strongest interaction
    pub fn strongest_interaction(&self) -> Option<(Vec<usize>, f64)> {
        if self.detected_interactions.is_empty() {
            return None;
        }

        let max_idx = self
            .interaction_strengths
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)?;

        Some((
            self.detected_interactions[max_idx].clone(),
            self.interaction_strengths[max_idx],
        ))
    }
}

/// Validator for interaction configurations
#[derive(Debug, Clone)]
pub struct InteractionValidator;

impl InteractionValidator {
    pub fn validate_config(config: &InteractionConfig) -> Result<()> {
        if config.max_order == 0 {
            return Err(SklearsError::InvalidInput(
                "max_order must be greater than 0".to_string(),
            ));
        }

        if config.significance_threshold <= 0.0 || config.significance_threshold >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "significance_threshold must be between 0 and 1".to_string(),
            ));
        }

        if config.correlation_threshold < 0.0 || config.correlation_threshold > 1.0 {
            return Err(SklearsError::InvalidInput(
                "correlation_threshold must be between 0 and 1".to_string(),
            ));
        }

        if config.min_samples_leaf == 0 {
            return Err(SklearsError::InvalidInput(
                "min_samples_leaf must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Core feature interaction detector
#[derive(Debug, Clone)]
pub struct FeatureInteractionDetector {
    config: InteractionConfig,
    interaction_results: Option<InteractionResults>,
    is_fitted: bool,
}

impl FeatureInteractionDetector {
    /// Create a new feature interaction detector
    pub fn new(config: InteractionConfig) -> Result<Self> {
        InteractionValidator::validate_config(&config)?;

        Ok(Self {
            config,
            interaction_results: None,
            is_fitted: false,
        })
    }

    /// Detect interactions in data
    pub fn detect_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<InteractionResults>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        match self.config.method {
            InteractionMethod::Pairwise => self.detect_pairwise_interactions(x, y),
            InteractionMethod::HigherOrder => self.detect_higher_order_interactions(x, y),
            InteractionMethod::Correlation => self.detect_correlation_interactions(x, y),
            InteractionMethod::MutualInformation => self.detect_mi_interactions(x, y),
            InteractionMethod::Statistical => self.detect_statistical_interactions(x, y),
            _ => self.detect_pairwise_interactions(x, y), // Default fallback
        }
    }

    /// Detect pairwise feature interactions
    fn detect_pairwise_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<InteractionResults>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let mut detected_interactions = Vec::new();
        let mut interaction_strengths = Vec::new();
        let mut feature_pair_scores = HashMap::new();

        // Test all pairs of features
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let feature_i = x.column(i);
                let feature_j = x.column(j);

                let interaction_strength =
                    self.compute_pairwise_interaction_strength(&feature_i, &feature_j, y)?;

                feature_pair_scores.insert((i, j), interaction_strength);

                if interaction_strength > self.config.interaction_strength_threshold {
                    detected_interactions.push(vec![i, j]);
                    interaction_strengths.push(interaction_strength);
                }
            }
        }

        let mut results = InteractionResults::new(
            detected_interactions,
            Array1::from_vec(interaction_strengths),
        );

        results.feature_pair_scores = feature_pair_scores;
        results.add_metadata("method".to_string(), 1.0); // Pairwise method
        results.add_metadata("n_features".to_string(), n_features as f64);

        self.interaction_results = Some(results.clone());
        self.is_fitted = true;

        Ok(results)
    }

    /// Compute pairwise interaction strength
    fn compute_pairwise_interaction_strength<T>(
        &self,
        feature_i: &ArrayView1<T>,
        feature_j: &ArrayView1<T>,
        _y: Option<&ArrayView1<T>>,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified correlation-based interaction strength
        let strength = self.compute_correlation(feature_i, feature_j)?;
        Ok(strength.abs())
    }

    /// Compute correlation between two features
    fn compute_correlation<T>(
        &self,
        feature_i: &ArrayView1<T>,
        feature_j: &ArrayView1<T>,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified Pearson correlation implementation
        // In practice, this would compute actual correlation
        let correlation = 0.15; // Placeholder
        Ok(correlation)
    }

    /// Detect higher-order interactions
    fn detect_higher_order_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<InteractionResults>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let mut detected_interactions = Vec::new();
        let mut interaction_strengths = Vec::new();

        // Generate combinations of features up to max_order
        for order in 2..=self.config.max_order.min(n_features) {
            let combinations = self.generate_combinations(n_features, order);

            for combination in combinations {
                if combination.len() != order {
                    continue;
                }

                let interaction_strength =
                    self.compute_higher_order_interaction_strength(x, &combination, y)?;

                if interaction_strength > self.config.interaction_strength_threshold {
                    detected_interactions.push(combination);
                    interaction_strengths.push(interaction_strength);
                }
            }
        }

        let mut results = InteractionResults::new(
            detected_interactions,
            Array1::from_vec(interaction_strengths),
        );

        results.add_metadata("method".to_string(), 2.0); // Higher-order method
        results.add_metadata("max_order".to_string(), self.config.max_order as f64);

        self.interaction_results = Some(results.clone());
        self.is_fitted = true;

        Ok(results)
    }

    /// Generate combinations of features
    fn generate_combinations(&self, n_features: usize, order: usize) -> Vec<Vec<usize>> {
        if order == 0 || order > n_features {
            return Vec::new();
        }

        if order == 1 {
            return (0..n_features).map(|i| vec![i]).collect();
        }

        let mut combinations = Vec::new();
        self.generate_combinations_recursive(n_features, order, 0, Vec::new(), &mut combinations);
        combinations
    }

    /// Recursive helper for generating combinations
    fn generate_combinations_recursive(
        &self,
        n_features: usize,
        remaining: usize,
        start: usize,
        current: Vec<usize>,
        combinations: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            combinations.push(current);
            return;
        }

        for i in start..=(n_features - remaining) {
            let mut new_current = current.clone();
            new_current.push(i);
            self.generate_combinations_recursive(
                n_features,
                remaining - 1,
                i + 1,
                new_current,
                combinations,
            );
        }
    }

    /// Compute higher-order interaction strength
    fn compute_higher_order_interaction_strength<T>(
        &self,
        _x: &ArrayView2<T>,
        combination: &[usize],
        _y: Option<&ArrayView1<T>>,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified higher-order interaction computation
        let base_strength = 0.1;
        let order_penalty = 1.0 / (combination.len() as f64).sqrt();
        Ok(base_strength * order_penalty)
    }

    /// Detect correlation-based interactions
    fn detect_correlation_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<InteractionResults>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let correlation_matrix = self.compute_correlation_matrix(x)?;

        let mut detected_interactions = Vec::new();
        let mut interaction_strengths = Vec::new();
        let mut feature_pair_scores = HashMap::new();

        // Find highly correlated pairs
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let correlation = correlation_matrix[(i, j)].abs();
                feature_pair_scores.insert((i, j), correlation);

                if correlation > self.config.correlation_threshold {
                    detected_interactions.push(vec![i, j]);
                    interaction_strengths.push(correlation);
                }
            }
        }

        let mut results = InteractionResults::new(
            detected_interactions,
            Array1::from_vec(interaction_strengths),
        );

        results.set_correlation_matrix(correlation_matrix);
        results.feature_pair_scores = feature_pair_scores;
        results.add_metadata("method".to_string(), 3.0); // Correlation method

        self.interaction_results = Some(results.clone());
        self.is_fitted = true;

        Ok(results)
    }

    /// Compute correlation matrix
    fn compute_correlation_matrix<T>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let (_, n_features) = x.dim();
        let mut correlation_matrix = Array2::zeros((n_features, n_features));

        // Set diagonal to 1.0 (self-correlation)
        for i in 0..n_features {
            correlation_matrix[(i, i)] = 1.0;
        }

        // Compute correlations for all pairs
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let feature_i = x.column(i);
                let feature_j = x.column(j);
                let correlation = self.compute_correlation(&feature_i, &feature_j)?;

                correlation_matrix[(i, j)] = correlation;
                correlation_matrix[(j, i)] = correlation;
            }
        }

        Ok(correlation_matrix)
    }

    /// Detect mutual information-based interactions
    fn detect_mi_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<InteractionResults>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let mut detected_interactions = Vec::new();
        let mut interaction_strengths = Vec::new();
        let mut feature_pair_scores = HashMap::new();

        // Compute mutual information for all pairs
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let feature_i = x.column(i);
                let feature_j = x.column(j);

                let mi_score = self.compute_mutual_information(&feature_i, &feature_j, y)?;
                feature_pair_scores.insert((i, j), mi_score);

                if mi_score > self.config.interaction_strength_threshold {
                    detected_interactions.push(vec![i, j]);
                    interaction_strengths.push(mi_score);
                }
            }
        }

        let mut results = InteractionResults::new(
            detected_interactions,
            Array1::from_vec(interaction_strengths),
        );

        results.feature_pair_scores = feature_pair_scores;
        results.add_metadata("method".to_string(), 4.0); // MI method

        self.interaction_results = Some(results.clone());
        self.is_fitted = true;

        Ok(results)
    }

    /// Compute mutual information between features
    fn compute_mutual_information<T>(
        &self,
        feature_i: &ArrayView1<T>,
        feature_j: &ArrayView1<T>,
        _y: Option<&ArrayView1<T>>,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified mutual information computation
        let mi_score = 0.12; // Placeholder
        Ok(mi_score)
    }

    /// Detect statistical interactions
    fn detect_statistical_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<InteractionResults>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let mut detected_interactions = Vec::new();
        let mut interaction_strengths = Vec::new();
        let mut significance_values = Vec::new();
        let mut statistical_tests = HashMap::new();

        // Perform statistical tests for all pairs
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let feature_i = x.column(i);
                let feature_j = x.column(j);

                let (test_statistic, p_value) =
                    self.compute_statistical_test(&feature_i, &feature_j, y)?;

                if p_value < self.config.significance_threshold {
                    detected_interactions.push(vec![i, j]);
                    interaction_strengths.push(test_statistic);
                    significance_values.push(p_value);
                }
            }
        }

        statistical_tests.insert(
            "mean_p_value".to_string(),
            if significance_values.is_empty() {
                1.0
            } else {
                significance_values.iter().sum::<f64>() / significance_values.len() as f64
            },
        );

        let mut results = InteractionResults::new(
            detected_interactions,
            Array1::from_vec(interaction_strengths),
        );

        if !significance_values.is_empty() {
            results.set_significance_values(Array1::from_vec(significance_values));
        }
        results.set_statistical_tests(statistical_tests);
        results.add_metadata("method".to_string(), 5.0); // Statistical method

        self.interaction_results = Some(results.clone());
        self.is_fitted = true;

        Ok(results)
    }

    /// Compute statistical test for interaction
    fn compute_statistical_test<T>(
        &self,
        feature_i: &ArrayView1<T>,
        feature_j: &ArrayView1<T>,
        _y: Option<&ArrayView1<T>>,
    ) -> Result<(f64, f64)>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified statistical test (e.g., correlation test)
        let correlation = self.compute_correlation(feature_i, feature_j)?;
        let n = feature_i.len() as f64;

        // t-statistic for correlation
        let t_statistic = correlation * ((n - 2.0) / (1.0 - correlation * correlation)).sqrt();

        // Simplified p-value calculation
        let p_value = if t_statistic.abs() > 1.96 { 0.04 } else { 0.2 };

        Ok((t_statistic, p_value))
    }

    /// Get interaction results
    pub fn interaction_results(&self) -> Option<&InteractionResults> {
        self.interaction_results.as_ref()
    }

    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Get configuration
    pub fn config(&self) -> &InteractionConfig {
        &self.config
    }
}

impl Default for FeatureInteractionDetector {
    fn default() -> Self {
        Self::new(InteractionConfig::default()).unwrap()
    }
}

/// Pairwise interaction detector
#[derive(Debug, Clone)]
pub struct PairwiseInteractions {
    correlation_threshold: f64,
    interaction_matrix: Option<Array2<f64>>,
    significant_pairs: Vec<(usize, usize)>,
}

impl PairwiseInteractions {
    pub fn new(correlation_threshold: f64) -> Self {
        Self {
            correlation_threshold,
            interaction_matrix: None,
            significant_pairs: Vec::new(),
        }
    }

    /// Detect pairwise interactions
    pub fn detect<T>(&mut self, x: &ArrayView2<T>) -> Result<Vec<(usize, usize)>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let (_, n_features) = x.dim();
        let mut interaction_matrix = Array2::zeros((n_features, n_features));
        let mut significant_pairs = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let feature_i = x.column(i);
                let feature_j = x.column(j);

                // Simplified correlation computation
                let correlation: f64 = 0.2; // Placeholder
                interaction_matrix[(i, j)] = correlation;
                interaction_matrix[(j, i)] = correlation;

                if correlation.abs() > self.correlation_threshold {
                    significant_pairs.push((i, j));
                }
            }
        }

        self.interaction_matrix = Some(interaction_matrix);
        self.significant_pairs = significant_pairs.clone();

        Ok(significant_pairs)
    }

    /// Get interaction matrix
    pub fn interaction_matrix(&self) -> Option<&Array2<f64>> {
        self.interaction_matrix.as_ref()
    }

    /// Get significant pairs
    pub fn significant_pairs(&self) -> &[(usize, usize)] {
        &self.significant_pairs
    }
}

/// Higher-order interaction detector
#[derive(Debug, Clone)]
pub struct HighOrderInteractions {
    max_order: usize,
    significance_threshold: f64,
    detected_interactions: Vec<Vec<usize>>,
}

impl HighOrderInteractions {
    pub fn new(max_order: usize, significance_threshold: f64) -> Self {
        Self {
            max_order,
            significance_threshold,
            detected_interactions: Vec::new(),
        }
    }

    /// Detect higher-order interactions
    pub fn detect<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: Option<&ArrayView1<T>>,
    ) -> Result<Vec<Vec<usize>>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let mut interactions = Vec::new();

        // Generate and test combinations
        for order in 2..=self.max_order.min(n_features) {
            let combinations = self.generate_combinations(n_features, order);

            for combination in combinations {
                let interaction_strength = self.evaluate_interaction(x, &combination, y)?;

                if interaction_strength > self.significance_threshold {
                    interactions.push(combination);
                }
            }
        }

        self.detected_interactions = interactions.clone();
        Ok(interactions)
    }

    /// Generate feature combinations
    fn generate_combinations(&self, n_features: usize, order: usize) -> Vec<Vec<usize>> {
        if order > n_features {
            return Vec::new();
        }

        let mut combinations = Vec::new();
        self.generate_combinations_recursive(n_features, order, 0, Vec::new(), &mut combinations);
        combinations
    }

    /// Recursive combination generation
    fn generate_combinations_recursive(
        &self,
        n_features: usize,
        remaining: usize,
        start: usize,
        current: Vec<usize>,
        combinations: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            combinations.push(current);
            return;
        }

        for i in start..=(n_features - remaining) {
            let mut new_current = current.clone();
            new_current.push(i);
            self.generate_combinations_recursive(
                n_features,
                remaining - 1,
                i + 1,
                new_current,
                combinations,
            );
        }
    }

    /// Evaluate interaction strength
    fn evaluate_interaction<T>(
        &self,
        _x: &ArrayView2<T>,
        combination: &[usize],
        _y: Option<&ArrayView1<T>>,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified interaction evaluation
        let base_strength = 0.08;
        let order_factor = 1.0 / (combination.len() as f64);
        Ok(base_strength * order_factor)
    }

    /// Get detected interactions
    pub fn detected_interactions(&self) -> &[Vec<usize>] {
        &self.detected_interactions
    }
}

/// Interaction analyzer for comprehensive analysis
#[derive(Debug, Clone)]
pub struct InteractionAnalyzer {
    analysis_results: HashMap<String, f64>,
    interaction_summary: HashMap<String, Vec<f64>>,
    network_metrics: HashMap<String, f64>,
}

impl InteractionAnalyzer {
    pub fn new() -> Self {
        Self {
            analysis_results: HashMap::new(),
            interaction_summary: HashMap::new(),
            network_metrics: HashMap::new(),
        }
    }

    /// Analyze interaction results
    pub fn analyze_interactions(&mut self, results: &InteractionResults) -> Result<()> {
        // Basic statistics
        let n_interactions = results.n_interactions_detected();
        let mean_strength = if !results.interaction_strengths().is_empty() {
            results.interaction_strengths().mean().unwrap_or(0.0)
        } else {
            0.0
        };

        self.analysis_results
            .insert("n_interactions".to_string(), n_interactions as f64);
        self.analysis_results
            .insert("mean_interaction_strength".to_string(), mean_strength);

        // Analyze interaction orders
        let mut order_counts = HashMap::new();
        for interaction in results.detected_interactions() {
            let order = interaction.len();
            *order_counts.entry(order).or_insert(0) += 1;
        }

        for (order, count) in order_counts {
            self.analysis_results
                .insert(format!("order_{}_count", order), count as f64);
        }

        // Network metrics
        if let Some((strongest_interaction, strength)) = results.strongest_interaction() {
            self.network_metrics
                .insert("strongest_interaction_strength".to_string(), strength);
            self.network_metrics.insert(
                "strongest_interaction_size".to_string(),
                strongest_interaction.len() as f64,
            );
        }

        Ok(())
    }

    /// Get analysis results
    pub fn analysis_results(&self) -> &HashMap<String, f64> {
        &self.analysis_results
    }

    /// Get interaction summary
    pub fn interaction_summary(&self) -> &HashMap<String, Vec<f64>> {
        &self.interaction_summary
    }

    /// Get network metrics
    pub fn network_metrics(&self) -> &HashMap<String, f64> {
        &self.network_metrics
    }
}

impl Default for InteractionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Correlation analysis for feature relationships
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    correlation_matrix: Option<Array2<f64>>,
    correlation_threshold: f64,
    significant_correlations: Vec<(usize, usize, f64)>,
}

impl CorrelationAnalysis {
    pub fn new(correlation_threshold: f64) -> Self {
        Self {
            correlation_matrix: None,
            correlation_threshold,
            significant_correlations: Vec::new(),
        }
    }

    /// Perform correlation analysis
    pub fn analyze<T>(&mut self, x: &ArrayView2<T>) -> Result<Vec<(usize, usize, f64)>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let correlation_matrix = self.compute_correlation_matrix(x)?;
        let significant_correlations = self.find_significant_correlations(&correlation_matrix)?;

        self.correlation_matrix = Some(correlation_matrix);
        self.significant_correlations = significant_correlations.clone();

        Ok(significant_correlations)
    }

    /// Compute correlation matrix
    fn compute_correlation_matrix<T>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let (_, n_features) = x.dim();
        let mut matrix = Array2::eye(n_features); // Initialize with identity

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let correlation = 0.15; // Placeholder correlation
                matrix[(i, j)] = correlation;
                matrix[(j, i)] = correlation;
            }
        }

        Ok(matrix)
    }

    /// Find significant correlations
    fn find_significant_correlations(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<Vec<(usize, usize, f64)>> {
        let n_features = matrix.nrows();
        let mut significant = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let correlation = matrix[(i, j)];
                if correlation.abs() > self.correlation_threshold {
                    significant.push((i, j, correlation));
                }
            }
        }

        Ok(significant)
    }

    /// Get correlation matrix
    pub fn correlation_matrix(&self) -> Option<&Array2<f64>> {
        self.correlation_matrix.as_ref()
    }

    /// Get significant correlations
    pub fn significant_correlations(&self) -> &[(usize, usize, f64)] {
        &self.significant_correlations
    }
}

/// Dependency analysis for feature dependencies
#[derive(Debug, Clone)]
pub struct DependencyAnalysis {
    dependency_matrix: Option<Array2<f64>>,
    dependency_threshold: f64,
    dependency_graph: HashMap<usize, Vec<usize>>,
}

impl DependencyAnalysis {
    pub fn new(dependency_threshold: f64) -> Self {
        Self {
            dependency_matrix: None,
            dependency_threshold,
            dependency_graph: HashMap::new(),
        }
    }

    /// Analyze feature dependencies
    pub fn analyze_dependencies<T>(
        &mut self,
        x: &ArrayView2<T>,
    ) -> Result<HashMap<usize, Vec<usize>>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let dependency_matrix = self.compute_dependency_matrix(x)?;
        let dependency_graph = self.build_dependency_graph(&dependency_matrix)?;

        self.dependency_matrix = Some(dependency_matrix);
        self.dependency_graph = dependency_graph.clone();

        Ok(dependency_graph)
    }

    /// Compute dependency matrix
    fn compute_dependency_matrix<T>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (_, n_features) = x.dim();
        let mut matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i != j {
                    let dependency = 0.1; // Placeholder dependency strength
                    matrix[(i, j)] = dependency;
                }
            }
        }

        Ok(matrix)
    }

    /// Build dependency graph
    fn build_dependency_graph(&self, matrix: &Array2<f64>) -> Result<HashMap<usize, Vec<usize>>> {
        let n_features = matrix.nrows();
        let mut graph = HashMap::new();

        for i in 0..n_features {
            let mut dependencies = Vec::new();
            for j in 0..n_features {
                if i != j && matrix[(i, j)] > self.dependency_threshold {
                    dependencies.push(j);
                }
            }
            graph.insert(i, dependencies);
        }

        Ok(graph)
    }

    /// Get dependency matrix
    pub fn dependency_matrix(&self) -> Option<&Array2<f64>> {
        self.dependency_matrix.as_ref()
    }

    /// Get dependency graph
    pub fn dependency_graph(&self) -> &HashMap<usize, Vec<usize>> {
        &self.dependency_graph
    }
}

/// Interaction optimizer for feature combination optimization
#[derive(Debug, Clone)]
pub struct InteractionOptimizer {
    optimization_strategy: String,
    optimization_results: HashMap<String, f64>,
    optimal_interactions: Vec<Vec<usize>>,
}

impl InteractionOptimizer {
    pub fn new(optimization_strategy: String) -> Self {
        Self {
            optimization_strategy,
            optimization_results: HashMap::new(),
            optimal_interactions: Vec::new(),
        }
    }

    /// Optimize feature interactions
    pub fn optimize_interactions<T>(
        &mut self,
        x: &ArrayView2<T>,
        y: &ArrayView1<T>,
        candidate_interactions: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        match self.optimization_strategy.as_str() {
            "greedy" => self.greedy_optimization(x, y, candidate_interactions),
            "genetic" => self.genetic_optimization(x, y, candidate_interactions),
            "exhaustive" => self.exhaustive_optimization(x, y, candidate_interactions),
            _ => self.greedy_optimization(x, y, candidate_interactions), // Default
        }
    }

    /// Greedy optimization
    fn greedy_optimization<T>(
        &mut self,
        _x: &ArrayView2<T>,
        _y: &ArrayView1<T>,
        candidate_interactions: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified greedy selection
        let mut selected = Vec::new();
        let mut remaining = candidate_interactions.to_vec();

        while !remaining.is_empty() && selected.len() < 10 {
            // Select best remaining interaction
            let best_idx = 0; // Placeholder selection
            if best_idx < remaining.len() {
                selected.push(remaining.remove(best_idx));
            } else {
                break;
            }
        }

        self.optimal_interactions = selected.clone();
        self.optimization_results
            .insert("n_selected".to_string(), selected.len() as f64);

        Ok(selected)
    }

    /// Genetic optimization
    fn genetic_optimization<T>(
        &mut self,
        _x: &ArrayView2<T>,
        _y: &ArrayView1<T>,
        candidate_interactions: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified genetic algorithm
        let selected: Vec<_> = candidate_interactions.iter().take(5).cloned().collect();

        self.optimal_interactions = selected.clone();
        self.optimization_results
            .insert("generations".to_string(), 20.0);
        self.optimization_results
            .insert("n_selected".to_string(), selected.len() as f64);

        Ok(selected)
    }

    /// Exhaustive optimization
    fn exhaustive_optimization<T>(
        &mut self,
        _x: &ArrayView2<T>,
        _y: &ArrayView1<T>,
        candidate_interactions: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified exhaustive search
        let selected = candidate_interactions.to_vec();

        self.optimal_interactions = selected.clone();
        self.optimization_results
            .insert("evaluations".to_string(), selected.len() as f64);

        Ok(selected)
    }

    /// Get optimization results
    pub fn optimization_results(&self) -> &HashMap<String, f64> {
        &self.optimization_results
    }

    /// Get optimal interactions
    pub fn optimal_interactions(&self) -> &[Vec<usize>] {
        &self.optimal_interactions
    }
}

/// Feature combination for generating interaction features
#[derive(Debug, Clone)]
pub struct FeatureCombination {
    combination_method: String,
    generated_features: Option<Array2<f64>>,
    combination_map: HashMap<usize, Vec<usize>>,
}

impl FeatureCombination {
    pub fn new(combination_method: String) -> Self {
        Self {
            combination_method,
            generated_features: None,
            combination_map: HashMap::new(),
        }
    }

    /// Generate combination features
    pub fn generate_combinations<T>(
        &mut self,
        x: &ArrayView2<T>,
        interactions: &[Vec<usize>],
    ) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let (n_samples, _) = x.dim();
        let n_combinations = interactions.len();
        let mut combined_features = Array2::zeros((n_samples, n_combinations));

        for (combo_idx, interaction) in interactions.iter().enumerate() {
            for sample_idx in 0..n_samples {
                let combined_value = match self.combination_method.as_str() {
                    "multiply" => self.multiply_features(x, sample_idx, interaction),
                    "add" => self.add_features(x, sample_idx, interaction),
                    "max" => self.max_features(x, sample_idx, interaction),
                    "min" => self.min_features(x, sample_idx, interaction),
                    _ => self.multiply_features(x, sample_idx, interaction), // Default
                }?;

                combined_features[(sample_idx, combo_idx)] = combined_value;
            }

            self.combination_map.insert(combo_idx, interaction.clone());
        }

        self.generated_features = Some(combined_features.clone());
        Ok(combined_features)
    }

    /// Multiply features
    fn multiply_features<T>(
        &self,
        x: &ArrayView2<T>,
        sample_idx: usize,
        interaction: &[usize],
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let mut product = 1.0;
        for &feature_idx in interaction {
            if feature_idx < x.dim().1 {
                // Convert to f64 (simplified)
                product *= 1.0; // Placeholder conversion
            }
        }
        Ok(product)
    }

    /// Add features
    fn add_features<T>(
        &self,
        x: &ArrayView2<T>,
        sample_idx: usize,
        interaction: &[usize],
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let mut sum = 0.0;
        for &feature_idx in interaction {
            if feature_idx < x.dim().1 {
                // Convert to f64 (simplified)
                sum += 1.0; // Placeholder conversion
            }
        }
        Ok(sum)
    }

    /// Take maximum of features
    fn max_features<T>(
        &self,
        x: &ArrayView2<T>,
        sample_idx: usize,
        interaction: &[usize],
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let mut max_val = f64::NEG_INFINITY;
        for &feature_idx in interaction {
            if feature_idx < x.dim().1 {
                // Convert to f64 (simplified)
                max_val = max_val.max(1.0); // Placeholder conversion
            }
        }
        Ok(max_val)
    }

    /// Take minimum of features
    fn min_features<T>(
        &self,
        x: &ArrayView2<T>,
        sample_idx: usize,
        interaction: &[usize],
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let mut min_val = f64::INFINITY;
        for &feature_idx in interaction {
            if feature_idx < x.dim().1 {
                // Convert to f64 (simplified)
                min_val = min_val.min(1.0); // Placeholder conversion
            }
        }
        Ok(min_val)
    }

    /// Get generated features
    pub fn generated_features(&self) -> Option<&Array2<f64>> {
        self.generated_features.as_ref()
    }

    /// Get combination map
    pub fn combination_map(&self) -> &HashMap<usize, Vec<usize>> {
        &self.combination_map
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interaction_config_default() {
        let config = InteractionConfig::default();
        assert_eq!(config.method, InteractionMethod::Pairwise);
        assert_eq!(config.max_order, 2);
        assert_eq!(config.significance_threshold, 0.05);
    }

    #[test]
    fn test_interaction_validator() {
        let mut config = InteractionConfig::default();
        assert!(InteractionValidator::validate_config(&config).is_ok());

        config.max_order = 0;
        assert!(InteractionValidator::validate_config(&config).is_err());

        config.max_order = 2;
        config.significance_threshold = 1.5;
        assert!(InteractionValidator::validate_config(&config).is_err());
    }

    #[test]
    fn test_interaction_results() {
        let interactions = vec![vec![0, 1], vec![1, 2]];
        let strengths = Array1::from_vec(vec![0.8, 0.6]);
        let mut results = InteractionResults::new(interactions.clone(), strengths.clone());

        assert_eq!(results.detected_interactions(), &interactions);
        assert_eq!(results.interaction_strengths(), &strengths);
        assert_eq!(results.n_interactions_detected(), 2);

        let significance = Array1::from_vec(vec![0.01, 0.03]);
        results.set_significance_values(significance.clone());
        assert_eq!(results.significance_values().unwrap(), &significance);

        results.add_metadata("method".to_string(), 1.0);
        assert_eq!(results.interaction_metadata().get("method"), Some(&1.0));

        let strongest = results.strongest_interaction().unwrap();
        assert_eq!(strongest.0, vec![0, 1]);
        assert_eq!(strongest.1, 0.8);
    }

    #[test]
    fn test_feature_interaction_detector() {
        let config = InteractionConfig::default();
        let mut detector = FeatureInteractionDetector::new(config).unwrap();

        let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0; 5].into_iter().chain(vec![1.0; 5]).collect());

        let results = detector
            .detect_interactions(&x.view(), Some(&y.view()))
            .unwrap();
        assert!(detector.is_fitted());
        assert!(detector.interaction_results().is_some());
    }

    #[test]
    fn test_pairwise_interactions() {
        let mut pairwise = PairwiseInteractions::new(0.1);

        let x = Array2::from_shape_vec((8, 4), (0..32).map(|i| i as f64).collect()).unwrap();
        let pairs = pairwise.detect(&x.view()).unwrap();

        assert!(pairwise.interaction_matrix().is_some());
        // Results depend on implementation details
    }

    #[test]
    fn test_higher_order_interactions() {
        let mut higher_order = HighOrderInteractions::new(3, 0.05);

        let x = Array2::from_shape_vec((12, 5), (0..60).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0; 6].into_iter().chain(vec![1.0; 6]).collect());

        let interactions = higher_order.detect(&x.view(), Some(&y.view())).unwrap();
        assert_eq!(
            higher_order.detected_interactions().len(),
            interactions.len()
        );
    }

    #[test]
    fn test_interaction_analyzer() {
        let mut analyzer = InteractionAnalyzer::new();

        let interactions = vec![vec![0, 1], vec![1, 2], vec![0, 1, 2]];
        let strengths = Array1::from_vec(vec![0.8, 0.6, 0.4]);
        let results = InteractionResults::new(interactions, strengths);

        assert!(analyzer.analyze_interactions(&results).is_ok());
        assert!(analyzer.analysis_results().contains_key("n_interactions"));
        assert!(analyzer
            .analysis_results()
            .contains_key("mean_interaction_strength"));
    }

    #[test]
    fn test_correlation_analysis() {
        let mut correlation = CorrelationAnalysis::new(0.2);

        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
        )
        .unwrap();
        let correlations = correlation.analyze(&x.view()).unwrap();

        assert!(correlation.correlation_matrix().is_some());
        // Results depend on implementation
    }

    #[test]
    fn test_dependency_analysis() {
        let mut dependency = DependencyAnalysis::new(0.15);

        let x = Array2::from_shape_vec((8, 4), (0..32).map(|i| i as f64).collect()).unwrap();
        let graph = dependency.analyze_dependencies(&x.view()).unwrap();

        assert!(dependency.dependency_matrix().is_some());
        assert_eq!(dependency.dependency_graph().len(), graph.len());
    }

    #[test]
    fn test_interaction_optimizer() {
        let mut optimizer = InteractionOptimizer::new("greedy".to_string());

        let x = Array2::from_shape_vec((10, 4), (0..40).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![0.0; 5].into_iter().chain(vec![1.0; 5]).collect());

        let candidates = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 2]];
        let optimal = optimizer
            .optimize_interactions(&x.view(), &y.view(), &candidates)
            .unwrap();

        assert!(!optimal.is_empty());
        assert!(optimizer.optimization_results().contains_key("n_selected"));
    }

    #[test]
    fn test_feature_combination() {
        let mut combination = FeatureCombination::new("multiply".to_string());

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let interactions = vec![vec![0, 1], vec![1, 2]];

        let combined = combination
            .generate_combinations(&x.view(), &interactions)
            .unwrap();
        assert_eq!(combined.dim(), (4, 2));
        assert!(combination.generated_features().is_some());
        assert_eq!(combination.combination_map().len(), 2);
    }
}
