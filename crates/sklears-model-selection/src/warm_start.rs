//! Warm-start mechanisms for optimization algorithms
//!
//! This module provides functionality to reuse previous optimization results
//! to speed up new optimization runs by initializing with good starting points
//! and leveraging historical performance data.

use crate::ParameterValue;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Configuration for warm-start mechanisms
#[derive(Debug, Clone)]
pub struct WarmStartConfig {
    /// Maximum number of previous evaluations to store
    pub max_history_size: usize,
    /// Whether to use top-k initialization
    pub use_top_k_init: bool,
    /// Number of top configurations to use for initialization
    pub top_k: usize,
    /// Whether to use surrogate model warm-start
    pub use_surrogate_warmstart: bool,
    /// Weight decay for older evaluations
    pub weight_decay: f64,
    /// Minimum weight for historical evaluations
    pub min_weight: f64,
    /// Whether to adapt parameter ranges based on history
    pub adapt_parameter_ranges: bool,
    /// Whether to use transfer learning from similar problems
    pub use_transfer_learning: bool,
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            use_top_k_init: true,
            top_k: 10,
            use_surrogate_warmstart: true,
            weight_decay: 0.95,
            min_weight: 0.1,
            adapt_parameter_ranges: true,
            use_transfer_learning: false,
        }
    }
}

/// Historical evaluation record
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvaluationRecord {
    /// Parameter configuration
    pub parameters: ParameterValue,
    /// Score achieved
    pub score: f64,
    /// Timestamp of evaluation
    pub timestamp: u64,
    /// Cross-validation fold scores (if available)
    pub cv_scores: Option<Vec<f64>>,
    /// Standard deviation of score
    pub score_std: Option<f64>,
    /// Duration of evaluation in milliseconds
    pub duration_ms: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl EvaluationRecord {
    pub fn new(parameters: ParameterValue, score: f64) -> Self {
        Self {
            parameters,
            score,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cv_scores: None,
            score_std: None,
            duration_ms: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_cv_scores(mut self, cv_scores: Vec<f64>) -> Self {
        self.score_std = Some(statistical_std(&cv_scores));
        self.cv_scores = Some(cv_scores);
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Calculate age-based weight
    pub fn age_weight(&self, decay_factor: f64, min_weight: f64) -> f64 {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let age_hours = (current_time - self.timestamp) / 3600;
        let weight = decay_factor.powf(age_hours as f64);
        weight.max(min_weight)
    }
}

/// Optimization history store
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// All evaluation records
    records: Vec<EvaluationRecord>,
    /// Configuration
    config: WarmStartConfig,
    /// Problem signature for transfer learning
    problem_signature: Option<String>,
}

impl OptimizationHistory {
    pub fn new(config: WarmStartConfig) -> Self {
        Self {
            records: Vec::new(),
            config,
            problem_signature: None,
        }
    }

    /// Add a new evaluation record
    pub fn add_record(&mut self, record: EvaluationRecord) {
        self.records.push(record);

        // Maintain size limit
        if self.records.len() > self.config.max_history_size {
            // Remove oldest records, but prefer to keep high-scoring ones
            self.records.sort_by(|a, b| {
                let score_diff = b
                    .score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if score_diff == std::cmp::Ordering::Equal {
                    b.timestamp.cmp(&a.timestamp) // Newer first if same score
                } else {
                    score_diff
                }
            });

            self.records.truncate(self.config.max_history_size);
        }
    }

    /// Get top-k best configurations
    pub fn get_top_k(&self, k: usize) -> Vec<&EvaluationRecord> {
        let mut sorted_records = self.records.iter().collect::<Vec<_>>();
        sorted_records.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_records.into_iter().take(k).collect()
    }

    /// Get weighted historical data
    pub fn get_weighted_data(&self) -> Vec<(ParameterValue, f64, f64)> {
        self.records
            .iter()
            .map(|record| {
                let weight = record.age_weight(self.config.weight_decay, self.config.min_weight);
                (record.parameters.clone(), record.score, weight)
            })
            .collect()
    }

    /// Get similar configurations based on parameter distance
    pub fn get_similar_configs(
        &self,
        target_params: &ParameterValue,
        similarity_threshold: f64,
        max_count: usize,
    ) -> Vec<&EvaluationRecord> {
        let mut similar = self
            .records
            .iter()
            .filter_map(|record| {
                let distance = parameter_distance(target_params, &record.parameters);
                if distance <= similarity_threshold {
                    Some((record, distance))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Sort by distance (closer first) then by score (better first)
        similar.sort_by(|a, b| {
            let dist_cmp = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
            if dist_cmp == std::cmp::Ordering::Equal {
                b.0.score
                    .partial_cmp(&a.0.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                dist_cmp
            }
        });

        similar
            .into_iter()
            .take(max_count)
            .map(|(record, _)| record)
            .collect()
    }

    /// Analyze parameter ranges from history
    pub fn analyze_parameter_ranges(&self) -> HashMap<String, (f64, f64, f64)> {
        let mut param_analysis = HashMap::new();

        if self.records.is_empty() {
            return param_analysis;
        }

        // Group records by parameter type and analyze ranges
        let mut float_params: HashMap<String, Vec<f64>> = HashMap::new();

        for record in &self.records {
            if let ParameterValue::Float(val) = &record.parameters {
                float_params
                    .entry("default".to_string())
                    .or_default()
                    .push(*val);
            }
            // Add more parameter type handling as needed
        }

        for (param_name, values) in float_params {
            if !values.is_empty() {
                let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mean_val = values.iter().sum::<f64>() / values.len() as f64;

                param_analysis.insert(param_name, (min_val, max_val, mean_val));
            }
        }

        param_analysis
    }

    /// Get evaluation statistics
    pub fn get_statistics(&self) -> OptimizationStatistics {
        if self.records.is_empty() {
            return OptimizationStatistics::default();
        }

        let scores: Vec<f64> = self.records.iter().map(|r| r.score).collect();
        let durations: Vec<u64> = self.records.iter().filter_map(|r| r.duration_ms).collect();

        let best_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let score_std = statistical_std(&scores);

        let mean_duration = if !durations.is_empty() {
            durations.iter().sum::<u64>() as f64 / durations.len() as f64
        } else {
            0.0
        };

        OptimizationStatistics {
            total_evaluations: self.records.len(),
            best_score,
            worst_score,
            mean_score,
            score_std,
            mean_duration_ms: mean_duration,
            unique_configurations: self.count_unique_configurations(),
        }
    }

    fn count_unique_configurations(&self) -> usize {
        let mut unique_params = std::collections::HashSet::new();
        for record in &self.records {
            unique_params.insert(format!("{:?}", record.parameters));
        }
        unique_params.len()
    }

    /// Set problem signature for transfer learning
    pub fn set_problem_signature(&mut self, signature: String) {
        self.problem_signature = Some(signature);
    }

    /// Get problem signature
    pub fn problem_signature(&self) -> Option<&str> {
        self.problem_signature.as_deref()
    }

    /// Export history to JSON
    #[cfg(feature = "serde")]
    pub fn export_to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.records)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to export history: {}", e)))
    }

    /// Import history from JSON
    #[cfg(feature = "serde")]
    pub fn import_from_json(&mut self, json_data: &str) -> Result<()> {
        let imported_records: Vec<EvaluationRecord> = serde_json::from_str(json_data)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to import history: {}", e)))?;

        for record in imported_records {
            self.add_record(record);
        }

        Ok(())
    }
}

/// Statistics about optimization history
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub total_evaluations: usize,
    pub best_score: f64,
    pub worst_score: f64,
    pub mean_score: f64,
    pub score_std: f64,
    pub mean_duration_ms: f64,
    pub unique_configurations: usize,
}

impl Default for OptimizationStatistics {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            best_score: f64::NEG_INFINITY,
            worst_score: f64::INFINITY,
            mean_score: 0.0,
            score_std: 0.0,
            mean_duration_ms: 0.0,
            unique_configurations: 0,
        }
    }
}

/// Warm-start strategy for optimization initialization
#[derive(Debug, Clone)]
pub enum WarmStartStrategy {
    /// Use top-k best configurations from history
    TopK(usize),
    /// Use weighted sampling based on historical performance
    WeightedSampling(usize),
    /// Use surrogate model predictions to generate initial points
    SurrogateModel(usize),
    /// Use clustering to find diverse good regions
    ClusterBased(usize),
    /// Combine multiple strategies
    Combined(Vec<WarmStartStrategy>),
}

/// Warm-start initializer
pub struct WarmStartInitializer {
    history: OptimizationHistory,
    strategy: WarmStartStrategy,
    config: WarmStartConfig,
}

impl WarmStartInitializer {
    pub fn new(
        history: OptimizationHistory,
        strategy: WarmStartStrategy,
        config: WarmStartConfig,
    ) -> Self {
        Self {
            history,
            strategy,
            config,
        }
    }

    /// Generate initial points for optimization
    pub fn generate_initial_points(&self, n_points: usize) -> Vec<ParameterValue> {
        match &self.strategy {
            WarmStartStrategy::TopK(k) => self.generate_top_k_points(*k.min(&n_points)),
            WarmStartStrategy::WeightedSampling(n) => {
                self.generate_weighted_sample_points(*n.min(&n_points))
            }
            WarmStartStrategy::SurrogateModel(n) => {
                self.generate_surrogate_points(*n.min(&n_points))
            }
            WarmStartStrategy::ClusterBased(n) => {
                self.generate_cluster_based_points(*n.min(&n_points))
            }
            WarmStartStrategy::Combined(strategies) => {
                self.generate_combined_points(strategies, n_points)
            }
        }
    }

    fn generate_top_k_points(&self, k: usize) -> Vec<ParameterValue> {
        self.history
            .get_top_k(k)
            .into_iter()
            .map(|record| record.parameters.clone())
            .collect()
    }

    fn generate_weighted_sample_points(&self, n: usize) -> Vec<ParameterValue> {
        let weighted_data = self.history.get_weighted_data();
        if weighted_data.is_empty() {
            return Vec::new();
        }

        // Create cumulative weight distribution
        let total_weight: f64 = weighted_data.iter().map(|(_, _, w)| w).sum();
        let mut cumulative_weights = Vec::new();
        let mut running_sum = 0.0;

        for (_, _, weight) in &weighted_data {
            running_sum += weight / total_weight;
            cumulative_weights.push(running_sum);
        }

        // Sample based on weights
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();
        let mut selected = Vec::new();

        for _ in 0..n {
            let random_val: f64 = rng.gen();
            for (i, &cum_weight) in cumulative_weights.iter().enumerate() {
                if random_val <= cum_weight {
                    selected.push(weighted_data[i].0.clone());
                    break;
                }
            }
        }

        selected
    }

    fn generate_surrogate_points(&self, n: usize) -> Vec<ParameterValue> {
        // Simplified surrogate model approach
        // In practice, this would use Gaussian Process or similar
        let top_configs = self.history.get_top_k(n.min(10));

        if top_configs.is_empty() {
            return Vec::new();
        }

        // For now, return top configurations with some variations
        // A real implementation would use a surrogate model to predict good areas
        top_configs
            .into_iter()
            .take(n)
            .map(|record| record.parameters.clone())
            .collect()
    }

    fn generate_cluster_based_points(&self, n: usize) -> Vec<ParameterValue> {
        // Simplified clustering approach
        // Group similar high-performing configurations and sample from each cluster
        let good_configs = self.history.get_top_k(n * 2); // Get more than needed

        if good_configs.is_empty() {
            return Vec::new();
        }

        // Simple distance-based clustering
        let mut clusters: Vec<Vec<EvaluationRecord>> = Vec::new();
        let similarity_threshold = 0.5;

        for config in good_configs {
            let mut assigned = false;
            for cluster in &mut clusters {
                let cluster_center: &EvaluationRecord = &cluster[0];
                let distance = parameter_distance(&config.parameters, &cluster_center.parameters);

                if distance <= similarity_threshold {
                    cluster.push(config.clone());
                    assigned = true;
                    break;
                }
            }

            if !assigned {
                clusters.push(vec![config.clone()]);
            }
        }

        // Sample one from each cluster, up to n points
        clusters
            .into_iter()
            .take(n)
            .map(|cluster| cluster[0].parameters.clone())
            .collect()
    }

    fn generate_combined_points(
        &self,
        strategies: &[WarmStartStrategy],
        n_points: usize,
    ) -> Vec<ParameterValue> {
        let points_per_strategy = n_points / strategies.len().max(1);
        let mut all_points = Vec::new();

        for strategy in strategies {
            let strategy_initializer = WarmStartInitializer::new(
                self.history.clone(),
                strategy.clone(),
                self.config.clone(),
            );
            let points = strategy_initializer.generate_initial_points(points_per_strategy);
            all_points.extend(points);
        }

        // Fill remaining points with top-k if needed
        if all_points.len() < n_points {
            let remaining = n_points - all_points.len();
            let top_points = self.generate_top_k_points(remaining);
            all_points.extend(top_points);
        }

        all_points.into_iter().take(n_points).collect()
    }

    /// Update history with new evaluation
    pub fn update_history(&mut self, record: EvaluationRecord) {
        self.history.add_record(record);
    }

    /// Get optimization statistics
    pub fn statistics(&self) -> OptimizationStatistics {
        self.history.get_statistics()
    }

    /// Get history reference
    pub fn history(&self) -> &OptimizationHistory {
        &self.history
    }

    /// Get mutable history reference
    pub fn history_mut(&mut self) -> &mut OptimizationHistory {
        &mut self.history
    }
}

/// Transfer learning for warm-start across similar problems
pub struct TransferLearning {
    /// Historical data from multiple problems
    problem_histories: HashMap<String, OptimizationHistory>,
    /// Similarity threshold for transfer
    similarity_threshold: f64,
}

impl TransferLearning {
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            problem_histories: HashMap::new(),
            similarity_threshold,
        }
    }

    /// Add history from a solved problem
    pub fn add_problem_history(&mut self, problem_id: String, history: OptimizationHistory) {
        self.problem_histories.insert(problem_id, history);
    }

    /// Get transfer learning recommendations for a new problem
    pub fn get_transfer_recommendations(
        &self,
        problem_signature: &str,
        n_recommendations: usize,
    ) -> Vec<ParameterValue> {
        let mut all_recommendations = Vec::new();

        for history in self.problem_histories.values() {
            if let Some(hist_signature) = history.problem_signature() {
                let similarity =
                    self.calculate_problem_similarity(problem_signature, hist_signature);

                if similarity >= self.similarity_threshold {
                    let top_configs = history.get_top_k(n_recommendations);
                    for config in top_configs {
                        all_recommendations
                            .push((config.parameters.clone(), config.score * similarity));
                    }
                }
            }
        }

        // Sort by weighted score and return top recommendations
        all_recommendations
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        all_recommendations
            .into_iter()
            .take(n_recommendations)
            .map(|(params, _)| params)
            .collect()
    }

    fn calculate_problem_similarity(&self, sig1: &str, sig2: &str) -> f64 {
        // Simplified similarity calculation
        // In practice, this could use more sophisticated methods
        let words1: std::collections::HashSet<&str> = sig1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = sig2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

// Helper functions

fn parameter_distance(p1: &ParameterValue, p2: &ParameterValue) -> f64 {
    match (p1, p2) {
        (ParameterValue::Float(v1), ParameterValue::Float(v2)) => (v1 - v2).abs(),
        (ParameterValue::Integer(v1), ParameterValue::Integer(v2)) => (*v1 - *v2).abs() as f64,
        (ParameterValue::String(s1), ParameterValue::String(s2)) => {
            if s1 == s2 {
                0.0
            } else {
                1.0
            }
        }
        _ => 1.0, // Different types are maximally distant
    }
}

fn statistical_std(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_record() {
        let params = ParameterValue::Float(1.5);
        let record = EvaluationRecord::new(params.clone(), 0.85)
            .with_cv_scores(vec![0.8, 0.85, 0.9])
            .with_duration(1500)
            .with_metadata("algorithm".to_string(), "random_forest".to_string());

        assert_eq!(record.score, 0.85);
        assert_eq!(record.parameters, params);
        assert!(record.cv_scores.is_some());
        assert!(record.score_std.is_some());
        assert_eq!(record.duration_ms, Some(1500));
        assert_eq!(
            record.metadata.get("algorithm"),
            Some(&"random_forest".to_string())
        );
    }

    #[test]
    fn test_optimization_history() {
        let config = WarmStartConfig::default();
        let mut history = OptimizationHistory::new(config);

        // Add some records
        let record1 = EvaluationRecord::new(ParameterValue::Float(1.0), 0.8);
        let record2 = EvaluationRecord::new(ParameterValue::Float(2.0), 0.9);
        let record3 = EvaluationRecord::new(ParameterValue::Float(3.0), 0.7);

        history.add_record(record1);
        history.add_record(record2);
        history.add_record(record3);

        // Test top-k
        let top_2 = history.get_top_k(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].score, 0.9); // Best score first
        assert_eq!(top_2[1].score, 0.8); // Second best

        // Test statistics
        let stats = history.get_statistics();
        assert_eq!(stats.total_evaluations, 3);
        assert_eq!(stats.best_score, 0.9);
        assert_eq!(stats.worst_score, 0.7);
        assert!((stats.mean_score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_warm_start_initializer() {
        let config = WarmStartConfig::default();
        let mut history = OptimizationHistory::new(config.clone());

        // Add some historical data
        for i in 0..10 {
            let score = 0.5 + (i as f64) * 0.05; // Increasing scores
            let record = EvaluationRecord::new(ParameterValue::Float(i as f64), score);
            history.add_record(record);
        }

        let initializer = WarmStartInitializer::new(history, WarmStartStrategy::TopK(5), config);

        let initial_points = initializer.generate_initial_points(3);
        assert_eq!(initial_points.len(), 3);

        // Should get the top-scoring configurations
        if let ParameterValue::Float(val) = &initial_points[0] {
            assert!(*val >= 7.0); // Should be from high-scoring configurations
        }
    }

    #[test]
    fn test_parameter_distance() {
        let p1 = ParameterValue::Float(1.0);
        let p2 = ParameterValue::Float(2.0);
        let p3 = ParameterValue::String("test".to_string());
        let p4 = ParameterValue::String("test".to_string());

        assert_eq!(parameter_distance(&p1, &p2), 1.0);
        assert_eq!(parameter_distance(&p3, &p4), 0.0);
        assert_eq!(parameter_distance(&p1, &p3), 1.0); // Different types
    }

    #[test]
    fn test_transfer_learning() {
        let mut transfer = TransferLearning::new(0.5);

        let config = WarmStartConfig::default();
        let mut history1 = OptimizationHistory::new(config.clone());
        history1.set_problem_signature("classification tree depth".to_string());

        let record = EvaluationRecord::new(ParameterValue::Float(10.0), 0.95);
        history1.add_record(record);

        transfer.add_problem_history("problem1".to_string(), history1);

        let recommendations =
            transfer.get_transfer_recommendations("classification tree depth optimization", 2);

        assert!(!recommendations.is_empty());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_json_serialization() {
        let config = WarmStartConfig::default();
        let mut history = OptimizationHistory::new(config.clone());

        let record = EvaluationRecord::new(ParameterValue::Float(1.5), 0.85)
            .with_cv_scores(vec![0.8, 0.85, 0.9]);

        history.add_record(record);

        let json = history.export_to_json().unwrap();
        assert!(json.contains("1.5"));
        assert!(json.contains("0.85"));

        let mut new_history = OptimizationHistory::new(config);
        new_history.import_from_json(&json).unwrap();

        assert_eq!(new_history.records.len(), 1);
        assert_eq!(new_history.records[0].score, 0.85);
    }
}
