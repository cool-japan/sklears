//! Budget-constrained kernel approximation methods
//!
//! This module provides kernel approximation methods that operate within
//! specified computational budgets for time, memory, or operations.

use crate::{Nystroem, RBFSampler};
use scirs2_core::ndarray::ndarray_linalg::{Norm, SVD};
use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform},
};
use std::collections::HashMap;
use std::time::Instant;

/// Budget constraint types
#[derive(Debug, Clone)]
/// BudgetConstraint
pub enum BudgetConstraint {
    /// Time budget in seconds
    Time { max_seconds: f64 },
    /// Memory budget in bytes
    Memory { max_bytes: usize },
    /// Operations budget (approximate number of operations)
    Operations { max_operations: usize },
    /// Combined budget with multiple constraints
    Combined {
        max_seconds: Option<f64>,

        max_bytes: Option<usize>,
        max_operations: Option<usize>,
    },
}

/// Budget usage tracking
#[derive(Debug, Clone)]
/// BudgetUsage
pub struct BudgetUsage {
    /// Time used in seconds
    pub time_used: f64,
    /// Memory used in bytes
    pub memory_used: usize,
    /// Operations performed
    pub operations_used: usize,
}

impl Default for BudgetUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl BudgetUsage {
    /// Create a new budget usage tracker
    pub fn new() -> Self {
        Self {
            time_used: 0.0,
            memory_used: 0,
            operations_used: 0,
        }
    }

    /// Check if budget is within constraints
    pub fn is_within_budget(&self, constraint: &BudgetConstraint) -> bool {
        match constraint {
            BudgetConstraint::Time { max_seconds } => self.time_used <= *max_seconds,
            BudgetConstraint::Memory { max_bytes } => self.memory_used <= *max_bytes,
            BudgetConstraint::Operations { max_operations } => {
                self.operations_used <= *max_operations
            }
            BudgetConstraint::Combined {
                max_seconds,
                max_bytes,
                max_operations,
            } => {
                let time_ok = max_seconds.map_or(true, |max| self.time_used <= max);
                let memory_ok = max_bytes.map_or(true, |max| self.memory_used <= max);
                let ops_ok = max_operations.map_or(true, |max| self.operations_used <= max);
                time_ok && memory_ok && ops_ok
            }
        }
    }
}

/// Optimization strategy for budget-constrained approximation
#[derive(Debug, Clone)]
/// OptimizationStrategy
pub enum OptimizationStrategy {
    /// Maximize approximation quality within budget
    MaxQuality,
    /// Minimize computational cost for target quality
    MinCost { target_quality: f64 },
    /// Balance quality and cost
    Balanced {
        quality_weight: f64,

        cost_weight: f64,
    },
    /// Greedy approach - stop when budget is exhausted
    Greedy,
}

/// Configuration for budget-constrained approximation
#[derive(Debug, Clone)]
/// BudgetConstrainedConfig
pub struct BudgetConstrainedConfig {
    /// Budget constraint
    pub budget: BudgetConstraint,
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Minimum number of components to test
    pub min_components: usize,
    /// Maximum number of components to test
    pub max_components: usize,
    /// Step size for component search
    pub step_size: usize,
    /// Number of trials for each configuration
    pub n_trials: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Early stopping tolerance
    pub early_stopping_tolerance: f64,
}

impl Default for BudgetConstrainedConfig {
    fn default() -> Self {
        Self {
            budget: BudgetConstraint::Time { max_seconds: 10.0 },
            strategy: OptimizationStrategy::MaxQuality,
            min_components: 10,
            max_components: 1000,
            step_size: 10,
            n_trials: 3,
            random_seed: None,
            early_stopping_tolerance: 0.01,
        }
    }
}

/// Result from budget-constrained optimization
#[derive(Debug, Clone)]
/// BudgetOptimizationResult
pub struct BudgetOptimizationResult {
    /// Optimal number of components
    pub optimal_components: usize,
    /// Quality score achieved
    pub quality_score: f64,
    /// Budget usage
    pub budget_usage: BudgetUsage,
    /// All tested configurations
    pub tested_configs: HashMap<usize, (f64, BudgetUsage)>,
    /// Whether optimization completed or was stopped due to budget
    pub completed: bool,
}

/// Budget-constrained RBF sampler
#[derive(Debug, Clone)]
/// BudgetConstrainedRBFSampler
pub struct BudgetConstrainedRBFSampler {
    gamma: f64,
    config: BudgetConstrainedConfig,
}

impl Default for BudgetConstrainedRBFSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl BudgetConstrainedRBFSampler {
    /// Create a new budget-constrained RBF sampler
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            config: BudgetConstrainedConfig::default(),
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: BudgetConstrainedConfig) -> Self {
        self.config = config;
        self
    }

    /// Set budget constraint
    pub fn budget(mut self, budget: BudgetConstraint) -> Self {
        self.config.budget = budget;
        self
    }

    /// Set optimization strategy
    pub fn strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Find optimal configuration within budget
    pub fn find_optimal_config(&self, x: &Array2<f64>) -> Result<BudgetOptimizationResult> {
        let _start_time = Instant::now();
        let mut total_usage = BudgetUsage::new();
        let mut tested_configs = HashMap::new();
        let mut best_quality = f64::NEG_INFINITY;
        let mut best_components = self.config.min_components;
        let mut completed = true;

        // Split data for validation
        let n_samples = x.nrows();
        let split_idx = (n_samples as f64 * 0.8) as usize;
        let x_train = x
            .slice(scirs2_core::ndarray::s![..split_idx, ..])
            .to_owned();
        let x_val = x
            .slice(scirs2_core::ndarray::s![split_idx.., ..])
            .to_owned();

        // Test different numbers of components
        for n_components in
            (self.config.min_components..=self.config.max_components).step_by(self.config.step_size)
        {
            let _config_start = Instant::now();
            let mut config_usage = BudgetUsage::new();
            let mut trial_qualities = Vec::new();

            // Check if we have budget for this configuration
            if !total_usage.is_within_budget(&self.config.budget) {
                completed = false;
                break;
            }

            // Run multiple trials
            for trial in 0..self.config.n_trials {
                let trial_start = Instant::now();

                // Create sampler
                let seed = self.config.random_seed.map(|s| s + trial as u64);
                let sampler = if let Some(s) = seed {
                    RBFSampler::new(n_components)
                        .gamma(self.gamma)
                        .random_state(s)
                } else {
                    RBFSampler::new(n_components).gamma(self.gamma)
                };

                // Fit and transform
                let fitted = sampler.fit(&x_train, &())?;
                let _x_train_transformed = fitted.transform(&x_train)?;
                let x_val_transformed = fitted.transform(&x_val)?;

                // Compute quality
                let quality = self.compute_quality(&x_val, &x_val_transformed)?;
                trial_qualities.push(quality);

                // Update trial usage
                let trial_time = trial_start.elapsed().as_secs_f64();
                config_usage.time_used += trial_time;
                config_usage.memory_used += self.estimate_memory_usage(n_components, x.ncols());
                config_usage.operations_used += self.estimate_operations(n_components, x.nrows());

                // Check budget constraints
                if !config_usage.is_within_budget(&self.config.budget) {
                    completed = false;
                    break;
                }
            }

            // Average quality across trials
            let avg_quality = trial_qualities.iter().sum::<f64>() / trial_qualities.len() as f64;

            // Update total usage
            total_usage.time_used += config_usage.time_used;
            total_usage.memory_used = total_usage.memory_used.max(config_usage.memory_used);
            total_usage.operations_used += config_usage.operations_used;

            // Store configuration result
            tested_configs.insert(n_components, (avg_quality, config_usage));

            // Update best configuration based on strategy
            if self.is_better_config(avg_quality, n_components, best_quality, best_components) {
                best_quality = avg_quality;
                best_components = n_components;
            }

            // Early stopping check
            if self.should_stop_early(avg_quality, &tested_configs) {
                break;
            }

            // Check if we're out of budget
            if !total_usage.is_within_budget(&self.config.budget) {
                completed = false;
                break;
            }
        }

        Ok(BudgetOptimizationResult {
            optimal_components: best_components,
            quality_score: best_quality,
            budget_usage: total_usage,
            tested_configs,
            completed,
        })
    }

    /// Compute quality score for transformed data
    fn compute_quality(&self, x: &Array2<f64>, x_transformed: &Array2<f64>) -> Result<f64> {
        // Compute reconstruction quality (simplified)
        let n_samples = x.nrows().min(100); // Limit for efficiency
        let x_subset = x.slice(scirs2_core::ndarray::s![..n_samples, ..]);
        let x_transformed_subset = x_transformed.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        // Compute approximate kernel matrix
        let k_approx = x_transformed_subset.dot(&x_transformed_subset.t());

        // Compute exact kernel matrix (small subset)
        let mut k_exact = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_subset.row(i) - &x_subset.row(j);
                let squared_norm = diff.dot(&diff);
                k_exact[[i, j]] = (-self.gamma * squared_norm).exp();
            }
        }

        // Compute alignment score
        let k_exact_norm = k_exact.norm_l2();
        let k_approx_norm = k_approx.norm_l2();
        let alignment = if k_exact_norm > 1e-12 && k_approx_norm > 1e-12 {
            (&k_exact * &k_approx).sum() / (k_exact_norm * k_approx_norm)
        } else {
            0.0
        };

        Ok(alignment)
    }

    /// Estimate memory usage for a given configuration
    fn estimate_memory_usage(&self, n_components: usize, n_features: usize) -> usize {
        // Rough estimate: components * features * sizeof(f64) + overhead
        n_components * n_features * 8 + 1024
    }

    /// Estimate number of operations for a given configuration
    fn estimate_operations(&self, n_components: usize, n_samples: usize) -> usize {
        // Rough estimate: fitting + transformation operations
        n_components * n_samples * 10
    }

    /// Check if a configuration is better based on optimization strategy
    fn is_better_config(
        &self,
        quality: f64,
        components: usize,
        best_quality: f64,
        best_components: usize,
    ) -> bool {
        match &self.config.strategy {
            OptimizationStrategy::MaxQuality => quality > best_quality,
            OptimizationStrategy::MinCost { target_quality } => {
                if quality >= *target_quality {
                    components < best_components || best_quality < *target_quality
                } else {
                    quality > best_quality
                }
            }
            OptimizationStrategy::Balanced {
                quality_weight,
                cost_weight,
            } => {
                let score = quality * quality_weight - (components as f64) * cost_weight;
                let best_score =
                    best_quality * quality_weight - (best_components as f64) * cost_weight;
                score > best_score
            }
            OptimizationStrategy::Greedy => quality > best_quality,
        }
    }

    /// Check if early stopping should be applied
    fn should_stop_early(
        &self,
        _current_quality: f64,
        tested_configs: &HashMap<usize, (f64, BudgetUsage)>,
    ) -> bool {
        if tested_configs.len() < 3 {
            return false;
        }

        // Check if improvement is minimal
        let mut recent_qualities: Vec<f64> = tested_configs
            .values()
            .map(|(quality, _)| *quality)
            .collect();
        recent_qualities.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = recent_qualities.len();
        if n >= 3 {
            let improvement = recent_qualities[n - 1] - recent_qualities[n - 3];
            improvement < self.config.early_stopping_tolerance
        } else {
            false
        }
    }
}

/// Fitted budget-constrained RBF sampler
pub struct FittedBudgetConstrainedRBFSampler {
    fitted_rbf: crate::rbf_sampler::RBFSampler<sklears_core::traits::Trained>,
    optimization_result: BudgetOptimizationResult,
}

impl Fit<Array2<f64>, ()> for BudgetConstrainedRBFSampler {
    type Fitted = FittedBudgetConstrainedRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Find optimal configuration within budget
        let optimization_result = self.find_optimal_config(x)?;

        // Fit RBF sampler with optimal configuration
        let rbf_sampler = RBFSampler::new(optimization_result.optimal_components).gamma(self.gamma);
        let fitted_rbf = rbf_sampler.fit(x, &())?;

        Ok(FittedBudgetConstrainedRBFSampler {
            fitted_rbf,
            optimization_result,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedBudgetConstrainedRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_rbf.transform(x)
    }
}

impl FittedBudgetConstrainedRBFSampler {
    /// Get the optimization result
    pub fn optimization_result(&self) -> &BudgetOptimizationResult {
        &self.optimization_result
    }

    /// Get the optimal number of components
    pub fn optimal_components(&self) -> usize {
        self.optimization_result.optimal_components
    }

    /// Get the quality score
    pub fn quality_score(&self) -> f64 {
        self.optimization_result.quality_score
    }

    /// Get the budget usage
    pub fn budget_usage(&self) -> &BudgetUsage {
        &self.optimization_result.budget_usage
    }

    /// Check if optimization completed within budget
    pub fn completed(&self) -> bool {
        self.optimization_result.completed
    }
}

/// Budget-constrained Nyström method
#[derive(Debug, Clone)]
/// BudgetConstrainedNystroem
pub struct BudgetConstrainedNystroem {
    kernel: crate::nystroem::Kernel,
    config: BudgetConstrainedConfig,
}

impl Default for BudgetConstrainedNystroem {
    fn default() -> Self {
        Self::new()
    }
}

impl BudgetConstrainedNystroem {
    /// Create a new budget-constrained Nyström method
    pub fn new() -> Self {
        Self {
            kernel: crate::nystroem::Kernel::Rbf { gamma: 1.0 },
            config: BudgetConstrainedConfig::default(),
        }
    }

    /// Set kernel type
    pub fn kernel(mut self, kernel: crate::nystroem::Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set gamma parameter (for RBF kernel)
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.kernel = crate::nystroem::Kernel::Rbf { gamma };
        self
    }

    /// Set configuration
    pub fn config(mut self, config: BudgetConstrainedConfig) -> Self {
        self.config = config;
        self
    }

    /// Find optimal configuration within budget
    pub fn find_optimal_config(&self, x: &Array2<f64>) -> Result<BudgetOptimizationResult> {
        let _start_time = Instant::now();
        let mut total_usage = BudgetUsage::new();
        let mut tested_configs = HashMap::new();
        let mut best_quality = f64::NEG_INFINITY;
        let mut best_components = self.config.min_components;
        let mut completed = true;

        // Test different numbers of components
        for n_components in
            (self.config.min_components..=self.config.max_components).step_by(self.config.step_size)
        {
            let _config_start = Instant::now();
            let mut config_usage = BudgetUsage::new();
            let mut trial_qualities = Vec::new();

            // Check if we have budget for this configuration
            if !total_usage.is_within_budget(&self.config.budget) {
                completed = false;
                break;
            }

            // Run multiple trials
            for trial in 0..self.config.n_trials {
                let trial_start = Instant::now();

                // Create Nyström method
                let seed = self.config.random_seed.map(|s| s + trial as u64);
                let nystroem = if let Some(s) = seed {
                    Nystroem::new(self.kernel.clone(), n_components).random_state(s)
                } else {
                    Nystroem::new(self.kernel.clone(), n_components)
                };

                // Fit and transform
                let fitted = nystroem.fit(x, &())?;
                let x_transformed = fitted.transform(x)?;

                // Compute quality (simplified)
                let quality = self.compute_nystroem_quality(&x_transformed)?;
                trial_qualities.push(quality);

                // Update trial usage
                let trial_time = trial_start.elapsed().as_secs_f64();
                config_usage.time_used += trial_time;
                config_usage.memory_used += n_components * x.ncols() * 8;
                config_usage.operations_used += n_components * x.nrows() * 5;

                // Check budget constraints
                if !config_usage.is_within_budget(&self.config.budget) {
                    completed = false;
                    break;
                }
            }

            // Average quality across trials
            let avg_quality = trial_qualities.iter().sum::<f64>() / trial_qualities.len() as f64;

            // Update total usage
            total_usage.time_used += config_usage.time_used;
            total_usage.memory_used = total_usage.memory_used.max(config_usage.memory_used);
            total_usage.operations_used += config_usage.operations_used;

            // Store configuration result
            tested_configs.insert(n_components, (avg_quality, config_usage));

            // Update best configuration
            if avg_quality > best_quality {
                best_quality = avg_quality;
                best_components = n_components;
            }

            // Check if we're out of budget
            if !total_usage.is_within_budget(&self.config.budget) {
                completed = false;
                break;
            }
        }

        Ok(BudgetOptimizationResult {
            optimal_components: best_components,
            quality_score: best_quality,
            budget_usage: total_usage,
            tested_configs,
            completed,
        })
    }

    /// Compute quality score for Nyström approximation
    fn compute_nystroem_quality(&self, x_transformed: &Array2<f64>) -> Result<f64> {
        // Compute effective rank as quality measure
        let (_, s, _) = x_transformed
            .svd(true, true)
            .map_err(|_| SklearsError::InvalidInput("SVD computation failed".to_string()))?;

        let s_sum = s.sum();
        if s_sum == 0.0 {
            return Ok(0.0);
        }

        let s_normalized = &s / s_sum;
        let entropy = -s_normalized
            .iter()
            .filter(|&&x| x > 1e-12)
            .map(|&x| x * x.ln())
            .sum::<f64>();

        Ok(entropy.exp())
    }
}

/// Fitted budget-constrained Nyström method
pub struct FittedBudgetConstrainedNystroem {
    fitted_nystroem: crate::nystroem::Nystroem<sklears_core::traits::Trained>,
    optimization_result: BudgetOptimizationResult,
}

impl Fit<Array2<f64>, ()> for BudgetConstrainedNystroem {
    type Fitted = FittedBudgetConstrainedNystroem;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Find optimal configuration within budget
        let optimization_result = self.find_optimal_config(x)?;

        // Fit Nyström method with optimal configuration
        let nystroem = Nystroem::new(self.kernel, optimization_result.optimal_components);
        let fitted_nystroem = nystroem.fit(x, &())?;

        Ok(FittedBudgetConstrainedNystroem {
            fitted_nystroem,
            optimization_result,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedBudgetConstrainedNystroem {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_nystroem.transform(x)
    }
}

impl FittedBudgetConstrainedNystroem {
    /// Get the optimization result
    pub fn optimization_result(&self) -> &BudgetOptimizationResult {
        &self.optimization_result
    }

    /// Get the optimal number of components
    pub fn optimal_components(&self) -> usize {
        self.optimization_result.optimal_components
    }

    /// Get the quality score
    pub fn quality_score(&self) -> f64 {
        self.optimization_result.quality_score
    }

    /// Get the budget usage
    pub fn budget_usage(&self) -> &BudgetUsage {
        &self.optimization_result.budget_usage
    }

    /// Check if optimization completed within budget
    pub fn completed(&self) -> bool {
        self.optimization_result.completed
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_budget_constrained_rbf_sampler() {
        let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| (i as f64) * 0.01).collect())
            .unwrap();

        let config = BudgetConstrainedConfig {
            budget: BudgetConstraint::Time { max_seconds: 5.0 },
            min_components: 10,
            max_components: 50,
            step_size: 10,
            n_trials: 2,
            ..Default::default()
        };

        let sampler = BudgetConstrainedRBFSampler::new().gamma(0.5).config(config);

        let fitted = sampler.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 100);
        assert!(fitted.optimal_components() >= 10);
        assert!(fitted.optimal_components() <= 50);
        assert!(fitted.quality_score() >= 0.0);
        assert!(fitted.budget_usage().time_used <= 5.0);
    }

    #[test]
    fn test_budget_constrained_nystroem() {
        let x =
            Array2::from_shape_vec((80, 3), (0..240).map(|i| (i as f64) * 0.02).collect()).unwrap();

        let config = BudgetConstrainedConfig {
            budget: BudgetConstraint::Memory { max_bytes: 100000 },
            min_components: 5,
            max_components: 30,
            step_size: 5,
            n_trials: 2,
            ..Default::default()
        };

        let nystroem = BudgetConstrainedNystroem::new().gamma(1.0).config(config);

        let fitted = nystroem.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 80);
        assert!(fitted.optimal_components() >= 5);
        assert!(fitted.optimal_components() <= 30);
        assert!(fitted.budget_usage().memory_used <= 100000);
    }

    #[test]
    fn test_budget_constraint_types() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| (i as f64) * 0.05).collect()).unwrap();

        let constraints = vec![
            BudgetConstraint::Time { max_seconds: 10.0 },
            BudgetConstraint::Memory { max_bytes: 100000 },
            BudgetConstraint::Operations {
                max_operations: 50000,
            },
            BudgetConstraint::Combined {
                max_seconds: Some(10.0),
                max_bytes: Some(100000),
                max_operations: Some(50000),
            },
        ];

        for constraint in constraints {
            let config = BudgetConstrainedConfig {
                budget: constraint,
                min_components: 5,
                max_components: 20,
                step_size: 5,
                n_trials: 1,
                ..Default::default()
            };

            let sampler = BudgetConstrainedRBFSampler::new()
                .gamma(0.8)
                .config(config.clone());

            let fitted = sampler.fit(&x, &()).unwrap();
            let usage = fitted.budget_usage();

            // Verify budget constraints are respected
            assert!(
                usage.is_within_budget(&config.budget),
                "Budget constraint violated"
            );
        }
    }

    #[test]
    fn test_optimization_strategies() {
        let x =
            Array2::from_shape_vec((60, 3), (0..180).map(|i| (i as f64) * 0.03).collect()).unwrap();

        let strategies = vec![
            OptimizationStrategy::MaxQuality,
            OptimizationStrategy::MinCost {
                target_quality: 0.7,
            },
            OptimizationStrategy::Balanced {
                quality_weight: 1.0,
                cost_weight: 0.01,
            },
            OptimizationStrategy::Greedy,
        ];

        for strategy in strategies {
            let config = BudgetConstrainedConfig {
                budget: BudgetConstraint::Time { max_seconds: 4.0 },
                strategy,
                min_components: 10,
                max_components: 30,
                step_size: 10,
                n_trials: 1,
                ..Default::default()
            };

            let sampler = BudgetConstrainedRBFSampler::new().gamma(0.5).config(config);

            let fitted = sampler.fit(&x, &()).unwrap();

            assert!(fitted.optimal_components() >= 10);
            assert!(fitted.optimal_components() <= 30);
            assert!(fitted.quality_score() >= 0.0);
        }
    }

    #[test]
    fn test_budget_usage_tracking() {
        let mut usage = BudgetUsage::new();
        usage.time_used = 2.5;
        usage.memory_used = 1000;
        usage.operations_used = 500;

        let constraint1 = BudgetConstraint::Time { max_seconds: 3.0 };
        let constraint2 = BudgetConstraint::Memory { max_bytes: 800 };
        let constraint3 = BudgetConstraint::Operations {
            max_operations: 600,
        };
        let constraint4 = BudgetConstraint::Combined {
            max_seconds: Some(3.0),
            max_bytes: Some(1200),
            max_operations: Some(600),
        };

        assert!(usage.is_within_budget(&constraint1));
        assert!(!usage.is_within_budget(&constraint2));
        assert!(usage.is_within_budget(&constraint3));
        assert!(usage.is_within_budget(&constraint4));
    }

    #[test]
    fn test_early_stopping() {
        let x =
            Array2::from_shape_vec((40, 2), (0..80).map(|i| (i as f64) * 0.05).collect()).unwrap();

        let config = BudgetConstrainedConfig {
            budget: BudgetConstraint::Time { max_seconds: 10.0 },
            min_components: 5,
            max_components: 50,
            step_size: 5,
            n_trials: 1,
            early_stopping_tolerance: 0.001,
            ..Default::default()
        };

        let sampler = BudgetConstrainedRBFSampler::new().gamma(0.3).config(config);

        let result = sampler.find_optimal_config(&x).unwrap();

        // Should find a reasonable solution
        assert!(result.optimal_components >= 5);
        assert!(result.optimal_components <= 50);
        assert!(result.quality_score >= 0.0);
    }

    #[test]
    fn test_reproducibility() {
        let x =
            Array2::from_shape_vec((50, 3), (0..150).map(|i| (i as f64) * 0.02).collect()).unwrap();

        let config = BudgetConstrainedConfig {
            budget: BudgetConstraint::Time { max_seconds: 3.0 },
            min_components: 10,
            max_components: 30,
            step_size: 10,
            n_trials: 2,
            random_seed: Some(42),
            ..Default::default()
        };

        let sampler1 = BudgetConstrainedRBFSampler::new()
            .gamma(0.7)
            .config(config.clone());

        let sampler2 = BudgetConstrainedRBFSampler::new().gamma(0.7).config(config);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        assert_eq!(fitted1.optimal_components(), fitted2.optimal_components());
        assert_abs_diff_eq!(
            fitted1.quality_score(),
            fitted2.quality_score(),
            epsilon = 1e-10
        );
    }
}
