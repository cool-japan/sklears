//! Adaptive Resource Allocation for hyperparameter optimization
//!
//! This module implements adaptive resource allocation strategies that dynamically
//! adjust the computational budget for different hyperparameter configurations
//! based on their performance, uncertainty, and potential for improvement.
//!
//! Key strategies include:
//! - Multi-Armed Bandit allocation
//! - Successive Halving with adaptive thresholds
//! - Resource allocation based on upper confidence bounds
//! - Dynamic budget reallocation based on learning curves

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use std::fmt::Debug;

/// Resource allocation strategy
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// Equal allocation to all configurations
    Uniform,
    /// Multi-armed bandit allocation based on Upper Confidence Bound
    UCB { exploration_factor: f64 },
    /// Thompson sampling for resource allocation
    ThompsonSampling,
    /// Successive halving with adaptive promotion
    SuccessiveHalving { reduction_factor: f64 },
    /// Custom allocation function
    Custom(fn(&[ResourceConfiguration]) -> Vec<f64>),
}

/// Configuration being evaluated with allocated resources
#[derive(Debug, Clone)]
pub struct ResourceConfiguration {
    /// Unique identifier for this configuration
    pub id: String,
    /// Parameter values for this configuration
    pub parameters: HashMap<String, f64>,
    /// Number of evaluations performed
    pub evaluations: usize,
    /// Performance scores from evaluations
    pub scores: Vec<f64>,
    /// Current mean performance
    pub mean_score: f64,
    /// Performance variance
    pub variance: f64,
    /// Confidence interval width
    pub confidence_width: f64,
    /// Total resources allocated (e.g., training epochs, CV folds)
    pub resources_allocated: f64,
    /// Resource efficiency (score per unit resource)
    pub efficiency: f64,
    /// Whether this configuration is still active
    pub is_active: bool,
}

impl ResourceConfiguration {
    /// Create a new resource configuration
    pub fn new(id: String, parameters: HashMap<String, f64>) -> Self {
        Self {
            id,
            parameters,
            evaluations: 0,
            scores: Vec::new(),
            mean_score: 0.0,
            variance: 0.0,
            confidence_width: f64::INFINITY,
            resources_allocated: 0.0,
            efficiency: 0.0,
            is_active: true,
        }
    }

    /// Add a new evaluation result
    pub fn add_evaluation(&mut self, score: f64, resources_used: f64) {
        self.scores.push(score);
        self.evaluations += 1;
        self.resources_allocated += resources_used;

        // Update statistics
        self.update_statistics();
    }

    /// Update internal statistics
    pub fn update_statistics(&mut self) {
        if self.scores.is_empty() {
            return;
        }

        // Calculate mean
        self.mean_score = self.scores.iter().sum::<f64>() / self.scores.len() as f64;

        // Calculate variance
        if self.scores.len() > 1 {
            let variance = self
                .scores
                .iter()
                .map(|&x| (x - self.mean_score).powi(2))
                .sum::<f64>()
                / (self.scores.len() - 1) as f64;
            self.variance = variance;

            // Calculate 95% confidence interval width
            let std_error = (variance / self.scores.len() as f64).sqrt();
            self.confidence_width = 1.96 * std_error;
        }

        // Calculate efficiency
        if self.resources_allocated > 0.0 {
            self.efficiency = self.mean_score / self.resources_allocated;
        }
    }

    /// Calculate Upper Confidence Bound
    pub fn upper_confidence_bound(&self, exploration_factor: f64, total_evaluations: usize) -> f64 {
        if self.evaluations == 0 {
            return f64::INFINITY;
        }

        let confidence_bonus = exploration_factor
            * (2.0 * (total_evaluations as f64).ln() / self.evaluations as f64).sqrt();

        self.mean_score + confidence_bonus
    }

    /// Calculate Lower Confidence Bound
    pub fn lower_confidence_bound(&self, exploration_factor: f64, total_evaluations: usize) -> f64 {
        if self.evaluations == 0 {
            return f64::NEG_INFINITY;
        }

        let confidence_bonus = exploration_factor
            * (2.0 * (total_evaluations as f64).ln() / self.evaluations as f64).sqrt();

        self.mean_score - confidence_bonus
    }

    /// Check if this configuration should be promoted (continue receiving resources)
    pub fn should_promote(&self, threshold_percentile: f64, all_scores: &[f64]) -> bool {
        if all_scores.is_empty() || self.evaluations == 0 {
            return true;
        }

        let mut sorted_scores = all_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (threshold_percentile * sorted_scores.len() as f64) as usize;
        let threshold = sorted_scores
            .get(threshold_idx)
            .unwrap_or(&f64::NEG_INFINITY);

        self.mean_score >= *threshold
    }
}

/// Adaptive resource allocator
pub struct AdaptiveResourceAllocator {
    /// Allocation strategy to use
    strategy: AllocationStrategy,
    /// All configurations being evaluated
    configurations: Vec<ResourceConfiguration>,
    /// Total resource budget
    total_budget: f64,
    /// Resources used so far
    resources_used: f64,
    /// Minimum resource allocation per configuration
    min_allocation: f64,
    /// Maximum resource allocation per configuration
    max_allocation: f64,
    /// Random number generator
    rng: StdRng,
}

impl AdaptiveResourceAllocator {
    pub fn new(
        strategy: AllocationStrategy,
        total_budget: f64,
        min_allocation: f64,
        max_allocation: f64,
        random_state: Option<u64>,
    ) -> Self {
        let rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        Self {
            strategy,
            configurations: Vec::new(),
            total_budget,
            resources_used: 0.0,
            min_allocation,
            max_allocation,
            rng,
        }
    }

    /// Add a new configuration to evaluate
    pub fn add_configuration(&mut self, config: ResourceConfiguration) {
        self.configurations.push(config);
    }

    /// Get next resource allocations for all active configurations
    pub fn allocate_resources(&mut self) -> Vec<(String, f64)> {
        let active_configs: Vec<ResourceConfiguration> = self
            .configurations
            .iter()
            .filter(|c| c.is_active)
            .cloned()
            .collect();

        if active_configs.is_empty() {
            return Vec::new();
        }

        let remaining_budget = self.total_budget - self.resources_used;
        if remaining_budget <= 0.0 {
            return Vec::new();
        }

        let active_refs: Vec<&ResourceConfiguration> = active_configs.iter().collect();
        let allocations = match &self.strategy {
            AllocationStrategy::Uniform => self.uniform_allocation(&active_refs, remaining_budget),
            AllocationStrategy::UCB { exploration_factor } => {
                self.ucb_allocation(&active_refs, remaining_budget, *exploration_factor)
            }
            AllocationStrategy::ThompsonSampling => {
                self.thompson_sampling_allocation(&active_configs, remaining_budget)
            }
            AllocationStrategy::SuccessiveHalving { reduction_factor } => self
                .successive_halving_allocation(&active_refs, remaining_budget, *reduction_factor),
            AllocationStrategy::Custom(allocation_fn) => {
                self.custom_allocation(&active_refs, remaining_budget, allocation_fn)
            }
        };

        // Apply min/max constraints
        let constrained_allocations: Vec<(String, f64)> = allocations
            .into_iter()
            .map(|(id, alloc)| (id, alloc.clamp(self.min_allocation, self.max_allocation)))
            .collect();

        // Update resources used
        let total_allocated: f64 = constrained_allocations.iter().map(|(_, alloc)| alloc).sum();
        self.resources_used += total_allocated;

        constrained_allocations
    }

    /// Uniform resource allocation
    fn uniform_allocation(
        &self,
        configs: &[&ResourceConfiguration],
        budget: f64,
    ) -> Vec<(String, f64)> {
        let allocation_per_config = budget / configs.len() as f64;
        configs
            .iter()
            .map(|config| (config.id.clone(), allocation_per_config))
            .collect()
    }

    /// Upper Confidence Bound based allocation
    fn ucb_allocation(
        &self,
        configs: &[&ResourceConfiguration],
        budget: f64,
        exploration_factor: f64,
    ) -> Vec<(String, f64)> {
        let total_evaluations: usize = configs.iter().map(|c| c.evaluations).sum();

        // Calculate UCB scores
        let ucb_scores: Vec<f64> = configs
            .iter()
            .map(|config| config.upper_confidence_bound(exploration_factor, total_evaluations))
            .collect();

        // Allocate proportional to UCB scores
        let total_ucb: f64 = ucb_scores.iter().sum();
        if total_ucb <= 0.0 {
            return self.uniform_allocation(configs, budget);
        }

        configs
            .iter()
            .zip(ucb_scores.iter())
            .map(|(config, &ucb)| {
                let allocation = budget * (ucb / total_ucb);
                (config.id.clone(), allocation)
            })
            .collect()
    }

    /// Thompson sampling based allocation
    fn thompson_sampling_allocation(
        &mut self,
        configs: &[ResourceConfiguration],
        budget: f64,
    ) -> Vec<(String, f64)> {
        let mut samples = Vec::new();

        for config in configs {
            let sample = if config.evaluations == 0 {
                // For untested configurations, sample from prior
                self.rng.gen::<f64>()
            } else {
                // Sample from posterior (using normal approximation)
                let std_dev = if config.variance > 0.0 {
                    (config.variance / config.evaluations as f64).sqrt()
                } else {
                    0.1 // Small default standard deviation
                };

                use scirs2_core::random::essentials::Normal;
                let normal = Normal::new(config.mean_score, std_dev)
                    .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                self.rng.sample(normal)
            };
            samples.push(sample);
        }

        // Allocate proportional to samples
        let total_samples: f64 = samples.iter().sum();
        if total_samples <= 0.0 {
            let config_refs: Vec<&ResourceConfiguration> = configs.iter().collect();
            return self.uniform_allocation(&config_refs, budget);
        }

        configs
            .iter()
            .zip(samples.iter())
            .map(|(config, &sample)| {
                let allocation = budget * (sample / total_samples);
                (config.id.clone(), allocation)
            })
            .collect()
    }

    /// Successive halving allocation
    fn successive_halving_allocation(
        &self,
        configs: &[&ResourceConfiguration],
        budget: f64,
        reduction_factor: f64,
    ) -> Vec<(String, f64)> {
        // Sort configurations by performance
        let mut config_scores: Vec<(usize, f64)> = configs
            .iter()
            .enumerate()
            .map(|(i, config)| (i, config.mean_score))
            .collect();

        config_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Determine how many configurations to keep in next round
        let num_to_keep = ((configs.len() as f64) * reduction_factor).ceil() as usize;
        let num_to_keep = num_to_keep.max(1);

        let mut allocations = vec![(String::new(), 0.0); configs.len()];
        let allocation_per_kept = budget / num_to_keep as f64;

        for (config_idx, _score) in config_scores
            .iter()
            .take(num_to_keep.min(config_scores.len()))
        {
            allocations[*config_idx] = (configs[*config_idx].id.clone(), allocation_per_kept);
        }

        allocations
            .into_iter()
            .filter(|(id, _)| !id.is_empty())
            .collect()
    }

    /// Custom allocation function
    fn custom_allocation(
        &self,
        configs: &[&ResourceConfiguration],
        budget: f64,
        allocation_fn: &fn(&[ResourceConfiguration]) -> Vec<f64>,
    ) -> Vec<(String, f64)> {
        let config_refs: Vec<ResourceConfiguration> = configs.iter().map(|&c| c.clone()).collect();
        let allocations = allocation_fn(&config_refs);

        if allocations.len() != configs.len() {
            return self.uniform_allocation(configs, budget);
        }

        let total_weight: f64 = allocations.iter().sum();
        if total_weight <= 0.0 {
            return self.uniform_allocation(configs, budget);
        }

        configs
            .iter()
            .zip(allocations.iter())
            .map(|(config, &weight)| {
                let allocation = budget * (weight / total_weight);
                (config.id.clone(), allocation)
            })
            .collect()
    }

    /// Update configuration with evaluation result
    pub fn update_configuration(
        &mut self,
        config_id: &str,
        score: f64,
        resources_used: f64,
    ) -> Result<()> {
        let config = self
            .configurations
            .iter_mut()
            .find(|c| c.id == config_id)
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!("Configuration {} not found", config_id))
            })?;

        config.add_evaluation(score, resources_used);
        Ok(())
    }

    /// Deactivate underperforming configurations
    pub fn prune_configurations(&mut self, min_evaluations: usize, percentile_threshold: f64) {
        if self.configurations.len() <= 1 {
            return;
        }

        // Only consider configurations with minimum evaluations
        let eligible_scores: Vec<f64> = self
            .configurations
            .iter()
            .filter(|c| c.evaluations >= min_evaluations)
            .map(|c| c.mean_score)
            .collect();

        if eligible_scores.is_empty() {
            return;
        }

        for config in &mut self.configurations {
            if config.evaluations >= min_evaluations
                && !config.should_promote(percentile_threshold, &eligible_scores)
            {
                config.is_active = false;
            }
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> AllocationStatistics {
        let active_count = self.configurations.iter().filter(|c| c.is_active).count();
        let total_count = self.configurations.len();

        let best_score = self
            .configurations
            .iter()
            .filter(|c| c.evaluations > 0)
            .map(|c| c.mean_score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let total_evaluations: usize = self.configurations.iter().map(|c| c.evaluations).sum();

        AllocationStatistics {
            active_configurations: active_count,
            total_configurations: total_count,
            resources_used: self.resources_used,
            resources_remaining: self.total_budget - self.resources_used,
            best_score,
            total_evaluations,
        }
    }

    /// Get active configurations
    pub fn active_configurations(&self) -> Vec<&ResourceConfiguration> {
        self.configurations.iter().filter(|c| c.is_active).collect()
    }

    /// Get best configuration so far
    pub fn best_configuration(&self) -> Option<&ResourceConfiguration> {
        self.configurations
            .iter()
            .filter(|c| c.evaluations > 0)
            .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
    }
}

/// Statistics about resource allocation
#[derive(Debug, Clone)]
pub struct AllocationStatistics {
    pub active_configurations: usize,
    pub total_configurations: usize,
    pub resources_used: f64,
    pub resources_remaining: f64,
    pub best_score: f64,
    pub total_evaluations: usize,
}

/// Configuration for adaptive resource allocation
#[derive(Debug, Clone)]
pub struct AdaptiveAllocationConfig {
    /// Total resource budget
    pub total_budget: f64,
    /// Minimum resource allocation per configuration
    pub min_allocation: f64,
    /// Maximum resource allocation per configuration
    pub max_allocation: f64,
    /// Minimum evaluations before pruning
    pub min_evaluations_for_pruning: usize,
    /// Percentile threshold for pruning (bottom X% get pruned)
    pub pruning_percentile: f64,
    /// How often to perform pruning (in allocation rounds)
    pub pruning_frequency: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for AdaptiveAllocationConfig {
    fn default() -> Self {
        Self {
            total_budget: 1000.0,
            min_allocation: 1.0,
            max_allocation: 100.0,
            min_evaluations_for_pruning: 3,
            pruning_percentile: 0.25,
            pruning_frequency: 5,
            random_state: None,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_configuration() {
        let mut config = ResourceConfiguration::new("test".to_string(), HashMap::new());
        assert_eq!(config.evaluations, 0);
        assert!(config.is_active);

        config.add_evaluation(0.8, 10.0);
        assert_eq!(config.evaluations, 1);
        assert_eq!(config.mean_score, 0.8);
        assert_eq!(config.resources_allocated, 10.0);
    }

    #[test]
    fn test_ucb_calculation() {
        let mut config = ResourceConfiguration::new("test".to_string(), HashMap::new());
        config.add_evaluation(0.8, 10.0);

        let ucb = config.upper_confidence_bound(1.0, 10);
        assert!(ucb > config.mean_score);
    }

    #[test]
    fn test_uniform_allocation() {
        let allocator =
            AdaptiveResourceAllocator::new(AllocationStrategy::Uniform, 100.0, 1.0, 50.0, Some(42));

        let configs = vec![
            ResourceConfiguration::new("1".to_string(), HashMap::new()),
            ResourceConfiguration::new("2".to_string(), HashMap::new()),
        ];
        let config_refs: Vec<&ResourceConfiguration> = configs.iter().collect();

        let allocations = allocator.uniform_allocation(&config_refs, 100.0);
        assert_eq!(allocations.len(), 2);
        assert_eq!(allocations[0].1, 50.0);
        assert_eq!(allocations[1].1, 50.0);
    }

    #[test]
    fn test_configuration_pruning() {
        let mut config = ResourceConfiguration::new("test".to_string(), HashMap::new());
        config.add_evaluation(0.1, 10.0); // Low score

        let all_scores = vec![0.8, 0.9, 0.7, 0.1]; // This config is worst
        assert!(!config.should_promote(0.5, &all_scores)); // Should be pruned
    }

    #[test]
    fn test_allocation_statistics() {
        let mut allocator =
            AdaptiveResourceAllocator::new(AllocationStrategy::Uniform, 100.0, 1.0, 50.0, Some(42));

        allocator.add_configuration(ResourceConfiguration::new("1".to_string(), HashMap::new()));
        allocator.add_configuration(ResourceConfiguration::new("2".to_string(), HashMap::new()));

        let stats = allocator.get_statistics();
        assert_eq!(stats.active_configurations, 2);
        assert_eq!(stats.total_configurations, 2);
        assert_eq!(stats.resources_used, 0.0);
    }
}
