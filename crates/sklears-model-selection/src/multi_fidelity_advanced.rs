//! Advanced Multi-Fidelity Optimization Techniques
//!
//! This module extends multi-fidelity optimization with:
//! - Progressive Resource Allocation
//! - Coarse-to-Fine Optimization Strategies
//! - Adaptive Fidelity Selection
//! - Budget Allocation Algorithms
//!
//! These techniques enable more efficient hyperparameter optimization by intelligently
//! allocating computational resources across different fidelity levels.

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::types::Float;
use std::collections::HashMap;

// ============================================================================
// Progressive Resource Allocation
// ============================================================================

/// Progressive resource allocation strategy
#[derive(Debug, Clone)]
pub enum ProgressiveAllocationStrategy {
    /// Geometric progression: each level gets r^i times more resources
    Geometric {
        base_resources: Float,
        ratio: Float,
        n_levels: usize,
    },
    /// Exponential progression: exponentially increasing resources
    Exponential {
        initial_resources: Float,
        growth_rate: Float,
        n_levels: usize,
    },
    /// Fibonacci progression: resources follow Fibonacci sequence
    Fibonacci {
        base_resources: Float,
        n_levels: usize,
    },
    /// Adaptive progression based on performance feedback
    Adaptive {
        initial_resources: Float,
        adaptation_rate: Float,
        performance_threshold: Float,
    },
    /// Custom progression with user-defined schedule
    Custom { resource_schedule: Vec<Float> },
}

/// Configuration for progressive resource allocation
#[derive(Debug, Clone)]
pub struct ProgressiveAllocationConfig {
    pub strategy: ProgressiveAllocationStrategy,
    pub total_budget: Float,
    pub min_resource_per_config: Float,
    pub max_resource_per_config: Float,
    pub early_stopping_threshold: Option<Float>,
    pub random_state: Option<u64>,
}

impl Default for ProgressiveAllocationConfig {
    fn default() -> Self {
        Self {
            strategy: ProgressiveAllocationStrategy::Geometric {
                base_resources: 1.0,
                ratio: 2.0,
                n_levels: 5,
            },
            total_budget: 100.0,
            min_resource_per_config: 0.1,
            max_resource_per_config: 50.0,
            early_stopping_threshold: Some(0.01),
            random_state: None,
        }
    }
}

/// Progressive resource allocator
pub struct ProgressiveAllocator {
    config: ProgressiveAllocationConfig,
    allocation_history: Vec<AllocationRecord>,
    current_level: usize,
    remaining_budget: Float,
}

impl ProgressiveAllocator {
    pub fn new(config: ProgressiveAllocationConfig) -> Self {
        let remaining_budget = config.total_budget;
        Self {
            config,
            allocation_history: Vec::new(),
            current_level: 0,
            remaining_budget,
        }
    }

    /// Allocate resources progressively to configurations
    pub fn allocate(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
    ) -> Result<AllocationPlan, Box<dyn std::error::Error>> {
        // Clone strategy to avoid borrowing issues
        let strategy = self.config.strategy.clone();
        match strategy {
            ProgressiveAllocationStrategy::Geometric {
                base_resources,
                ratio,
                n_levels,
            } => self.allocate_geometric(configurations, base_resources, ratio, n_levels),
            ProgressiveAllocationStrategy::Exponential {
                initial_resources,
                growth_rate,
                n_levels,
            } => {
                self.allocate_exponential(configurations, initial_resources, growth_rate, n_levels)
            }
            ProgressiveAllocationStrategy::Fibonacci {
                base_resources,
                n_levels,
            } => self.allocate_fibonacci(configurations, base_resources, n_levels),
            ProgressiveAllocationStrategy::Adaptive {
                initial_resources,
                adaptation_rate,
                performance_threshold,
            } => self.allocate_adaptive(
                configurations,
                initial_resources,
                adaptation_rate,
                performance_threshold,
            ),
            ProgressiveAllocationStrategy::Custom { resource_schedule } => {
                self.allocate_custom(configurations, &resource_schedule)
            }
        }
    }

    /// Geometric progression allocation
    fn allocate_geometric(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
        base_resources: Float,
        ratio: Float,
        n_levels: usize,
    ) -> Result<AllocationPlan, Box<dyn std::error::Error>> {
        let mut allocations = Vec::new();
        let mut promoted_configs = configurations.to_vec();

        for level in 0..n_levels {
            let resources_at_level = base_resources * ratio.powi(level as i32);
            let resources_clamped = resources_at_level
                .max(self.config.min_resource_per_config)
                .min(self.config.max_resource_per_config);

            // Allocate to each configuration
            for (idx, config) in promoted_configs.iter().enumerate() {
                if self.remaining_budget >= resources_clamped {
                    allocations.push(ConfigAllocation {
                        config_id: idx,
                        level,
                        resources: resources_clamped,
                        estimated_performance: config.performance,
                    });
                    self.remaining_budget -= resources_clamped;
                }
            }

            // Promote top performing configurations to next level
            if level < n_levels - 1 {
                promoted_configs = self.select_top_configs(&promoted_configs, ratio);
            }
        }

        Ok(AllocationPlan {
            allocations,
            total_resources_used: self.config.total_budget - self.remaining_budget,
            n_levels,
            final_configs: promoted_configs.len(),
        })
    }

    /// Exponential progression allocation
    fn allocate_exponential(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
        initial_resources: Float,
        growth_rate: Float,
        n_levels: usize,
    ) -> Result<AllocationPlan, Box<dyn std::error::Error>> {
        let mut allocations = Vec::new();
        let mut promoted_configs = configurations.to_vec();

        for level in 0..n_levels {
            let resources_at_level = initial_resources * (growth_rate * level as Float).exp();
            let resources_clamped = resources_at_level
                .max(self.config.min_resource_per_config)
                .min(self.config.max_resource_per_config);

            for (idx, config) in promoted_configs.iter().enumerate() {
                if self.remaining_budget >= resources_clamped {
                    allocations.push(ConfigAllocation {
                        config_id: idx,
                        level,
                        resources: resources_clamped,
                        estimated_performance: config.performance,
                    });
                    self.remaining_budget -= resources_clamped;
                }
            }

            if level < n_levels - 1 {
                let survival_rate = (-growth_rate).exp();
                promoted_configs = self.select_top_configs(&promoted_configs, 1.0 / survival_rate);
            }
        }

        Ok(AllocationPlan {
            allocations,
            total_resources_used: self.config.total_budget - self.remaining_budget,
            n_levels,
            final_configs: promoted_configs.len(),
        })
    }

    /// Fibonacci progression allocation
    fn allocate_fibonacci(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
        base_resources: Float,
        n_levels: usize,
    ) -> Result<AllocationPlan, Box<dyn std::error::Error>> {
        let mut allocations = Vec::new();
        let mut promoted_configs = configurations.to_vec();

        // Generate Fibonacci sequence
        let fib_sequence = self.generate_fibonacci(n_levels);

        for (level, &fib) in fib_sequence.iter().enumerate() {
            let resources_at_level = base_resources * fib as Float;
            let resources_clamped = resources_at_level
                .max(self.config.min_resource_per_config)
                .min(self.config.max_resource_per_config);

            for (idx, config) in promoted_configs.iter().enumerate() {
                if self.remaining_budget >= resources_clamped {
                    allocations.push(ConfigAllocation {
                        config_id: idx,
                        level,
                        resources: resources_clamped,
                        estimated_performance: config.performance,
                    });
                    self.remaining_budget -= resources_clamped;
                }
            }

            if level < n_levels - 1 {
                let ratio = if fib > 0 {
                    fib_sequence[level + 1] as Float / fib as Float
                } else {
                    2.0
                };
                promoted_configs = self.select_top_configs(&promoted_configs, ratio);
            }
        }

        Ok(AllocationPlan {
            allocations,
            total_resources_used: self.config.total_budget - self.remaining_budget,
            n_levels,
            final_configs: promoted_configs.len(),
        })
    }

    /// Adaptive progression allocation based on performance feedback
    fn allocate_adaptive(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
        initial_resources: Float,
        adaptation_rate: Float,
        performance_threshold: Float,
    ) -> Result<AllocationPlan, Box<dyn std::error::Error>> {
        let mut allocations = Vec::new();
        let mut active_configs = configurations.to_vec();
        let mut level = 0;
        let mut current_resources = initial_resources;

        while !active_configs.is_empty() && self.remaining_budget > 0.0 {
            // Allocate current resources to active configurations
            for (idx, config) in active_configs.iter().enumerate() {
                let allocated = current_resources
                    .max(self.config.min_resource_per_config)
                    .min(self.config.max_resource_per_config)
                    .min(self.remaining_budget);

                if allocated > 0.0 {
                    allocations.push(ConfigAllocation {
                        config_id: idx,
                        level,
                        resources: allocated,
                        estimated_performance: config.performance,
                    });
                    self.remaining_budget -= allocated;
                }
            }

            // Compute performance statistics
            let performances: Vec<Float> = active_configs.iter().map(|c| c.performance).collect();
            let mean_perf = performances.iter().sum::<Float>() / performances.len() as Float;
            let max_perf = performances
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            // Adapt resource allocation based on performance spread
            let perf_spread = max_perf - mean_perf;
            if perf_spread > performance_threshold {
                // Good separation: increase resources for next round
                current_resources *= 1.0 + adaptation_rate;
            } else {
                // Poor separation: keep or slightly decrease resources
                current_resources *= 1.0 - adaptation_rate * 0.5;
            }

            // Filter configurations above threshold
            active_configs.retain(|c| c.performance >= mean_perf);

            level += 1;

            // Safety check to prevent infinite loops
            if level > 100 {
                break;
            }
        }

        Ok(AllocationPlan {
            allocations,
            total_resources_used: self.config.total_budget - self.remaining_budget,
            n_levels: level,
            final_configs: active_configs.len(),
        })
    }

    /// Custom progression allocation
    fn allocate_custom(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
        resource_schedule: &[Float],
    ) -> Result<AllocationPlan, Box<dyn std::error::Error>> {
        let mut allocations = Vec::new();
        let mut promoted_configs = configurations.to_vec();

        for (level, &resources_at_level) in resource_schedule.iter().enumerate() {
            let resources_clamped = resources_at_level
                .max(self.config.min_resource_per_config)
                .min(self.config.max_resource_per_config);

            for (idx, config) in promoted_configs.iter().enumerate() {
                if self.remaining_budget >= resources_clamped {
                    allocations.push(ConfigAllocation {
                        config_id: idx,
                        level,
                        resources: resources_clamped,
                        estimated_performance: config.performance,
                    });
                    self.remaining_budget -= resources_clamped;
                }
            }

            // Halve configurations each level
            if level < resource_schedule.len() - 1 {
                promoted_configs = self.select_top_configs(&promoted_configs, 2.0);
            }
        }

        Ok(AllocationPlan {
            allocations,
            total_resources_used: self.config.total_budget - self.remaining_budget,
            n_levels: resource_schedule.len(),
            final_configs: promoted_configs.len(),
        })
    }

    // Helper methods

    fn select_top_configs(
        &self,
        configs: &[ConfigurationWithPerformance],
        ratio: Float,
    ) -> Vec<ConfigurationWithPerformance> {
        let n_keep = (configs.len() as Float / ratio).ceil() as usize;
        let n_keep = n_keep.max(1).min(configs.len());

        let mut sorted_configs = configs.to_vec();
        sorted_configs.sort_by(|a, b| b.performance.partial_cmp(&a.performance).unwrap());
        sorted_configs.truncate(n_keep);
        sorted_configs
    }

    fn generate_fibonacci(&self, n: usize) -> Vec<usize> {
        let mut fib = vec![1, 1];
        for i in 2..n {
            fib.push(fib[i - 1] + fib[i - 2]);
        }
        fib.truncate(n);
        fib
    }
}

/// Configuration with performance information
#[derive(Debug, Clone)]
pub struct ConfigurationWithPerformance {
    pub config_id: usize,
    pub parameters: HashMap<String, Float>,
    pub performance: Float,
    pub resources_used: Float,
}

/// Allocation for a specific configuration
#[derive(Debug, Clone)]
pub struct ConfigAllocation {
    pub config_id: usize,
    pub level: usize,
    pub resources: Float,
    pub estimated_performance: Float,
}

/// Plan for resource allocation
#[derive(Debug, Clone)]
pub struct AllocationPlan {
    pub allocations: Vec<ConfigAllocation>,
    pub total_resources_used: Float,
    pub n_levels: usize,
    pub final_configs: usize,
}

/// Allocation record for history tracking
#[derive(Debug, Clone)]
struct AllocationRecord {
    level: usize,
    resources: Float,
    n_configs: usize,
}

// ============================================================================
// Coarse-to-Fine Optimization Strategies
// ============================================================================

/// Coarse-to-fine optimization strategy
#[derive(Debug, Clone)]
pub enum CoarseToFineStrategy {
    /// Grid refinement: start with coarse grid, refine promising regions
    GridRefinement {
        initial_grid_size: usize,
        refinement_factor: Float,
        n_refinement_levels: usize,
    },
    /// Hierarchical sampling: progressively finer sampling
    HierarchicalSampling {
        initial_samples: usize,
        samples_per_level: usize,
        focus_radius: Float,
    },
    /// Zoom-in strategy: iteratively zoom into best regions
    ZoomIn {
        zoom_factor: Float,
        n_zoom_levels: usize,
        top_k_regions: usize,
    },
    /// Multi-scale optimization
    MultiScale {
        scales: Vec<Float>,
        samples_per_scale: usize,
    },
}

/// Configuration for coarse-to-fine optimization
#[derive(Debug, Clone)]
pub struct CoarseToFineConfig {
    pub strategy: CoarseToFineStrategy,
    pub parameter_bounds: HashMap<String, (Float, Float)>,
    pub convergence_tolerance: Float,
    pub max_evaluations: usize,
    pub random_state: Option<u64>,
}

impl Default for CoarseToFineConfig {
    fn default() -> Self {
        Self {
            strategy: CoarseToFineStrategy::GridRefinement {
                initial_grid_size: 10,
                refinement_factor: 0.5,
                n_refinement_levels: 3,
            },
            parameter_bounds: HashMap::new(),
            convergence_tolerance: 1e-4,
            max_evaluations: 1000,
            random_state: None,
        }
    }
}

/// Coarse-to-fine optimizer
pub struct CoarseToFineOptimizer {
    config: CoarseToFineConfig,
    evaluation_history: Vec<EvaluationPoint>,
    current_bounds: HashMap<String, (Float, Float)>,
    rng: StdRng,
}

impl CoarseToFineOptimizer {
    pub fn new(config: CoarseToFineConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.random_state.unwrap_or(42));
        let current_bounds = config.parameter_bounds.clone();

        Self {
            config,
            evaluation_history: Vec::new(),
            current_bounds,
            rng,
        }
    }

    /// Optimize using coarse-to-fine strategy
    pub fn optimize<F>(
        &mut self,
        objective_fn: F,
    ) -> Result<CoarseToFineResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Float,
    {
        // Clone strategy to avoid borrowing issues
        let strategy = self.config.strategy.clone();
        match strategy {
            CoarseToFineStrategy::GridRefinement {
                initial_grid_size,
                refinement_factor,
                n_refinement_levels,
            } => self.optimize_grid_refinement(
                &objective_fn,
                initial_grid_size,
                refinement_factor,
                n_refinement_levels,
            ),
            CoarseToFineStrategy::HierarchicalSampling {
                initial_samples,
                samples_per_level,
                focus_radius,
            } => self.optimize_hierarchical(
                &objective_fn,
                initial_samples,
                samples_per_level,
                focus_radius,
            ),
            CoarseToFineStrategy::ZoomIn {
                zoom_factor,
                n_zoom_levels,
                top_k_regions,
            } => self.optimize_zoomin(&objective_fn, zoom_factor, n_zoom_levels, top_k_regions),
            CoarseToFineStrategy::MultiScale {
                scales,
                samples_per_scale,
            } => self.optimize_multiscale(&objective_fn, &scales, samples_per_scale),
        }
    }

    fn optimize_grid_refinement<F>(
        &mut self,
        objective_fn: &F,
        initial_grid_size: usize,
        refinement_factor: Float,
        n_refinement_levels: usize,
    ) -> Result<CoarseToFineResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Float,
    {
        let mut best_point = None;
        let mut best_value = f64::NEG_INFINITY;

        for level in 0..n_refinement_levels {
            // Sample grid at current resolution
            let grid_size = (initial_grid_size as Float * (1.0 + level as Float)).ceil() as usize;
            let points = self.generate_grid_points(grid_size)?;

            // Evaluate all points
            for params in points {
                let value = objective_fn(&params);
                self.evaluation_history.push(EvaluationPoint {
                    parameters: params.clone(),
                    value,
                    level,
                });

                if value > best_value {
                    best_value = value;
                    best_point = Some(params);
                }
            }

            // Refine bounds around best point
            if let Some(ref best) = best_point {
                self.refine_bounds(best, refinement_factor);
            }
        }

        Ok(CoarseToFineResult {
            best_parameters: best_point.unwrap_or_default(),
            best_value,
            n_evaluations: self.evaluation_history.len(),
            convergence_history: self.get_convergence_history(),
        })
    }

    fn optimize_hierarchical<F>(
        &mut self,
        objective_fn: &F,
        initial_samples: usize,
        samples_per_level: usize,
        focus_radius: Float,
    ) -> Result<CoarseToFineResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Float,
    {
        let mut best_point = None;
        let mut best_value = f64::NEG_INFINITY;
        let mut level = 0;

        // Initial random sampling
        for _ in 0..initial_samples {
            let params = self.sample_from_bounds(&self.current_bounds.clone())?;
            let value = objective_fn(&params);
            self.evaluation_history.push(EvaluationPoint {
                parameters: params.clone(),
                value,
                level,
            });

            if value > best_value {
                best_value = value;
                best_point = Some(params);
            }
        }

        // Hierarchical refinement
        while self.evaluation_history.len() < self.config.max_evaluations {
            level += 1;

            let best_params = best_point.clone();
            if let Some(best) = best_params {
                // Focus sampling around best point
                for _ in 0..samples_per_level {
                    let params = self.sample_around_point(&best, focus_radius)?;
                    let value = objective_fn(&params);
                    self.evaluation_history.push(EvaluationPoint {
                        parameters: params.clone(),
                        value,
                        level,
                    });

                    if value > best_value {
                        best_value = value;
                        best_point = Some(params);
                    }
                }
            }

            // Reduce focus radius for next level
            let new_radius = focus_radius * 0.5;
            if new_radius < self.config.convergence_tolerance {
                break;
            }
        }

        Ok(CoarseToFineResult {
            best_parameters: best_point.unwrap_or_default(),
            best_value,
            n_evaluations: self.evaluation_history.len(),
            convergence_history: self.get_convergence_history(),
        })
    }

    fn optimize_zoomin<F>(
        &mut self,
        objective_fn: &F,
        zoom_factor: Float,
        n_zoom_levels: usize,
        top_k_regions: usize,
    ) -> Result<CoarseToFineResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Float,
    {
        let mut regions = vec![self.current_bounds.clone()];
        let mut best_point = None;
        let mut best_value = f64::NEG_INFINITY;

        for level in 0..n_zoom_levels {
            let mut region_performances = Vec::new();

            // Evaluate each region
            for region in &regions {
                let samples = self.sample_from_bounds(region)?;
                let value = objective_fn(&samples);

                self.evaluation_history.push(EvaluationPoint {
                    parameters: samples.clone(),
                    value,
                    level,
                });

                if value > best_value {
                    best_value = value;
                    best_point = Some(samples.clone());
                }

                region_performances.push((region.clone(), samples, value));
            }

            // Select top-k regions
            region_performances.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            region_performances.truncate(top_k_regions);

            // Zoom into top regions
            regions = region_performances
                .iter()
                .map(|(region, center, _)| self.zoom_region(region, center, zoom_factor))
                .collect();
        }

        Ok(CoarseToFineResult {
            best_parameters: best_point.unwrap_or_default(),
            best_value,
            n_evaluations: self.evaluation_history.len(),
            convergence_history: self.get_convergence_history(),
        })
    }

    fn optimize_multiscale<F>(
        &mut self,
        objective_fn: &F,
        scales: &[Float],
        samples_per_scale: usize,
    ) -> Result<CoarseToFineResult, Box<dyn std::error::Error>>
    where
        F: Fn(&HashMap<String, Float>) -> Float,
    {
        let mut best_point = None;
        let mut best_value = f64::NEG_INFINITY;

        for (level, &scale) in scales.iter().enumerate() {
            // Generate samples at current scale
            for _ in 0..samples_per_scale {
                let current_bounds = self.current_bounds.clone();
                let mut params = self.sample_from_bounds(&current_bounds)?;

                // Apply scale perturbation
                for (_, value) in params.iter_mut() {
                    let perturbation = self.rng.gen_range(-scale..scale);
                    *value += perturbation;
                }

                let value = objective_fn(&params);
                self.evaluation_history.push(EvaluationPoint {
                    parameters: params.clone(),
                    value,
                    level,
                });

                if value > best_value {
                    best_value = value;
                    best_point = Some(params);
                }
            }

            // Refine bounds for next scale
            if let Some(ref best) = best_point {
                self.refine_bounds(best, 0.5);
            }
        }

        Ok(CoarseToFineResult {
            best_parameters: best_point.unwrap_or_default(),
            best_value,
            n_evaluations: self.evaluation_history.len(),
            convergence_history: self.get_convergence_history(),
        })
    }

    // Helper methods

    fn generate_grid_points(
        &self,
        grid_size: usize,
    ) -> Result<Vec<HashMap<String, Float>>, Box<dyn std::error::Error>> {
        let mut points = Vec::new();

        // Generate grid for each parameter
        let param_names: Vec<_> = self.current_bounds.keys().cloned().collect();

        if param_names.is_empty() {
            return Ok(points);
        }

        // Simple 1D grid for each parameter (can be extended to multi-D)
        for param_name in &param_names {
            if let Some(&(min, max)) = self.current_bounds.get(param_name) {
                for i in 0..grid_size {
                    let value = min + (max - min) * (i as Float) / ((grid_size - 1) as Float);
                    let mut params = HashMap::new();
                    params.insert(param_name.clone(), value);
                    points.push(params);
                }
            }
        }

        Ok(points)
    }

    fn sample_from_bounds(
        &mut self,
        bounds: &HashMap<String, (Float, Float)>,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        let mut params = HashMap::new();

        for (param_name, &(min, max)) in bounds {
            let value = self.rng.gen_range(min..max);
            params.insert(param_name.clone(), value);
        }

        Ok(params)
    }

    fn sample_around_point(
        &mut self,
        center: &HashMap<String, Float>,
        radius: Float,
    ) -> Result<HashMap<String, Float>, Box<dyn std::error::Error>> {
        let mut params = HashMap::new();

        for (param_name, &center_value) in center {
            if let Some(&(min, max)) = self.current_bounds.get(param_name) {
                let perturbation = self.rng.gen_range(-radius..radius);
                let value = (center_value + perturbation).max(min).min(max);
                params.insert(param_name.clone(), value);
            }
        }

        Ok(params)
    }

    fn refine_bounds(&mut self, best_point: &HashMap<String, Float>, refinement_factor: Float) {
        for (param_name, &value) in best_point {
            if let Some(&(min, max)) = self.current_bounds.get(param_name) {
                let range = max - min;
                let new_range = range * refinement_factor;
                let new_min = (value - new_range / 2.0).max(min);
                let new_max = (value + new_range / 2.0).min(max);
                self.current_bounds
                    .insert(param_name.clone(), (new_min, new_max));
            }
        }
    }

    fn zoom_region(
        &self,
        region: &HashMap<String, (Float, Float)>,
        center: &HashMap<String, Float>,
        zoom_factor: Float,
    ) -> HashMap<String, (Float, Float)> {
        let mut zoomed = HashMap::new();

        for (param_name, &(min, max)) in region {
            if let Some(&center_value) = center.get(param_name) {
                let range = max - min;
                let new_range = range * zoom_factor;
                let new_min = (center_value - new_range / 2.0).max(min);
                let new_max = (center_value + new_range / 2.0).min(max);
                zoomed.insert(param_name.clone(), (new_min, new_max));
            }
        }

        zoomed
    }

    fn get_convergence_history(&self) -> Vec<Float> {
        let mut best_so_far = f64::NEG_INFINITY;
        self.evaluation_history
            .iter()
            .map(|point| {
                if point.value > best_so_far {
                    best_so_far = point.value;
                }
                best_so_far
            })
            .collect()
    }
}

/// Evaluation point in optimization
#[derive(Debug, Clone)]
struct EvaluationPoint {
    parameters: HashMap<String, Float>,
    value: Float,
    level: usize,
}

/// Result of coarse-to-fine optimization
#[derive(Debug, Clone)]
pub struct CoarseToFineResult {
    pub best_parameters: HashMap<String, Float>,
    pub best_value: Float,
    pub n_evaluations: usize,
    pub convergence_history: Vec<Float>,
}

// ============================================================================
// Adaptive Fidelity Selection
// ============================================================================

/// Adaptive fidelity selection strategy
#[derive(Debug, Clone)]
pub enum AdaptiveFidelityStrategy {
    /// UCB-based fidelity selection
    UpperConfidenceBound { exploration_weight: Float },
    /// Expected Improvement with fidelity cost
    ExpectedImprovement { cost_weight: Float },
    /// Information gain maximization
    InformationGain { uncertainty_threshold: Float },
    /// Cost-aware Thompson sampling
    ThompsonSampling {
        prior_alpha: Float,
        prior_beta: Float,
    },
}

/// Adaptive fidelity selector
pub struct AdaptiveFidelitySelector {
    strategy: AdaptiveFidelityStrategy,
    fidelity_stats: HashMap<usize, FidelityStatistics>,
}

impl AdaptiveFidelitySelector {
    pub fn new(strategy: AdaptiveFidelityStrategy) -> Self {
        Self {
            strategy,
            fidelity_stats: HashMap::new(),
        }
    }

    /// Select fidelity level for next evaluation
    pub fn select_fidelity(
        &mut self,
        available_fidelities: &[usize],
        budget_remaining: Float,
    ) -> usize {
        match &self.strategy {
            AdaptiveFidelityStrategy::UpperConfidenceBound { exploration_weight } => {
                self.select_ucb(available_fidelities, *exploration_weight)
            }
            AdaptiveFidelityStrategy::ExpectedImprovement { cost_weight } => {
                self.select_ei(available_fidelities, *cost_weight, budget_remaining)
            }
            AdaptiveFidelityStrategy::InformationGain {
                uncertainty_threshold,
            } => self.select_ig(available_fidelities, *uncertainty_threshold),
            AdaptiveFidelityStrategy::ThompsonSampling {
                prior_alpha,
                prior_beta,
            } => self.select_thompson(available_fidelities, *prior_alpha, *prior_beta),
        }
    }

    /// Update fidelity statistics after evaluation
    pub fn update(&mut self, fidelity: usize, performance: Float, cost: Float) {
        let stats = self
            .fidelity_stats
            .entry(fidelity)
            .or_insert_with(|| FidelityStatistics {
                n_evaluations: 0,
                total_performance: 0.0,
                total_cost: 0.0,
                mean_performance: 0.0,
                mean_cost: 0.0,
            });

        stats.n_evaluations += 1;
        stats.total_performance += performance;
        stats.total_cost += cost;
        stats.mean_performance = stats.total_performance / stats.n_evaluations as Float;
        stats.mean_cost = stats.total_cost / stats.n_evaluations as Float;
    }

    fn select_ucb(&self, fidelities: &[usize], exploration_weight: Float) -> usize {
        let total_evals: usize = self.fidelity_stats.values().map(|s| s.n_evaluations).sum();

        let mut best_fidelity = fidelities[0];
        let mut best_score = f64::NEG_INFINITY;

        for &fidelity in fidelities {
            let score = if let Some(stats) = self.fidelity_stats.get(&fidelity) {
                let exploitation = stats.mean_performance / stats.mean_cost.max(1e-6);
                let exploration = exploration_weight
                    * ((total_evals as Float).ln() / stats.n_evaluations as Float).sqrt();
                exploitation + exploration
            } else {
                f64::INFINITY // Unvisited fidelities get highest priority
            };

            if score > best_score {
                best_score = score;
                best_fidelity = fidelity;
            }
        }

        best_fidelity
    }

    fn select_ei(&self, fidelities: &[usize], cost_weight: Float, _budget: Float) -> usize {
        let mut best_fidelity = fidelities[0];
        let mut best_score = f64::NEG_INFINITY;

        for &fidelity in fidelities {
            let score = if let Some(stats) = self.fidelity_stats.get(&fidelity) {
                stats.mean_performance - cost_weight * stats.mean_cost
            } else {
                0.0 // Default for unvisited
            };

            if score > best_score {
                best_score = score;
                best_fidelity = fidelity;
            }
        }

        best_fidelity
    }

    fn select_ig(&self, fidelities: &[usize], _threshold: Float) -> usize {
        // Prefer fidelities with less data (higher uncertainty)
        let mut best_fidelity = fidelities[0];
        let mut min_evals = usize::MAX;

        for &fidelity in fidelities {
            let n_evals = self
                .fidelity_stats
                .get(&fidelity)
                .map(|s| s.n_evaluations)
                .unwrap_or(0);
            if n_evals < min_evals {
                min_evals = n_evals;
                best_fidelity = fidelity;
            }
        }

        best_fidelity
    }

    fn select_thompson(
        &self,
        fidelities: &[usize],
        prior_alpha: Float,
        prior_beta: Float,
    ) -> usize {
        let mut rng = StdRng::seed_from_u64(42);
        let mut best_fidelity = fidelities[0];
        let mut best_sample = f64::NEG_INFINITY;

        for &fidelity in fidelities {
            // Beta distribution sampling (simplified)
            let (alpha, beta) = if let Some(stats) = self.fidelity_stats.get(&fidelity) {
                (
                    prior_alpha + stats.total_performance,
                    prior_beta + stats.n_evaluations as Float - stats.total_performance,
                )
            } else {
                (prior_alpha, prior_beta)
            };

            // Simple sampling (in practice would use proper Beta distribution)
            let sample = rng.gen_range(0.0..(alpha / (alpha + beta)));

            if sample > best_sample {
                best_sample = sample;
                best_fidelity = fidelity;
            }
        }

        best_fidelity
    }
}

#[derive(Debug, Clone)]
struct FidelityStatistics {
    n_evaluations: usize,
    total_performance: Float,
    total_cost: Float,
    mean_performance: Float,
    mean_cost: Float,
}

// ============================================================================
// Budget Allocation Algorithms
// ============================================================================

/// Budget allocation strategy
#[derive(Debug, Clone)]
pub enum BudgetAllocationStrategy {
    /// Equal allocation to all configurations
    Equal,
    /// Proportional to estimated performance
    Proportional { min_allocation: Float },
    /// Rank-based allocation
    RankBased { allocation_ratios: Vec<Float> },
    /// Adaptive allocation based on uncertainty
    UncertaintyBased { exploration_factor: Float },
    /// Racing algorithm
    Racing { confidence_level: Float },
}

/// Budget allocator
pub struct BudgetAllocator {
    strategy: BudgetAllocationStrategy,
    total_budget: Float,
    allocations: HashMap<usize, Float>,
}

impl BudgetAllocator {
    pub fn new(strategy: BudgetAllocationStrategy, total_budget: Float) -> Self {
        Self {
            strategy,
            total_budget,
            allocations: HashMap::new(),
        }
    }

    /// Allocate budget among configurations
    pub fn allocate(
        &mut self,
        configurations: &[ConfigurationWithPerformance],
    ) -> HashMap<usize, Float> {
        // Clone strategy to avoid borrowing issues
        let strategy = self.strategy.clone();
        match strategy {
            BudgetAllocationStrategy::Equal => self.allocate_equal(configurations),
            BudgetAllocationStrategy::Proportional { min_allocation } => {
                self.allocate_proportional(configurations, min_allocation)
            }
            BudgetAllocationStrategy::RankBased { allocation_ratios } => {
                self.allocate_rank_based(configurations, &allocation_ratios)
            }
            BudgetAllocationStrategy::UncertaintyBased { exploration_factor } => {
                self.allocate_uncertainty(configurations, exploration_factor)
            }
            BudgetAllocationStrategy::Racing { confidence_level } => {
                self.allocate_racing(configurations, confidence_level)
            }
        }
    }

    fn allocate_equal(
        &mut self,
        configs: &[ConfigurationWithPerformance],
    ) -> HashMap<usize, Float> {
        let per_config = self.total_budget / configs.len() as Float;
        configs
            .iter()
            .enumerate()
            .map(|(i, _)| (i, per_config))
            .collect()
    }

    fn allocate_proportional(
        &mut self,
        configs: &[ConfigurationWithPerformance],
        min_allocation: Float,
    ) -> HashMap<usize, Float> {
        let total_perf: Float = configs.iter().map(|c| c.performance).sum();
        let available = self.total_budget - min_allocation * configs.len() as Float;

        configs
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let proportional = if total_perf > 0.0 {
                    (c.performance / total_perf) * available
                } else {
                    0.0
                };
                (i, min_allocation + proportional)
            })
            .collect()
    }

    fn allocate_rank_based(
        &mut self,
        configs: &[ConfigurationWithPerformance],
        ratios: &[Float],
    ) -> HashMap<usize, Float> {
        let mut sorted: Vec<_> = configs.iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.performance.partial_cmp(&a.1.performance).unwrap());

        let total_ratio: Float = ratios.iter().sum();
        sorted
            .iter()
            .enumerate()
            .map(|(rank, (orig_idx, _))| {
                let ratio = ratios.get(rank).cloned().unwrap_or(0.0);
                let allocation = (ratio / total_ratio) * self.total_budget;
                (*orig_idx, allocation)
            })
            .collect()
    }

    fn allocate_uncertainty(
        &mut self,
        configs: &[ConfigurationWithPerformance],
        exploration: Float,
    ) -> HashMap<usize, Float> {
        // Simplified: allocate based on inverse of resources already used
        let total_inverse: Float = configs.iter().map(|c| 1.0 / (c.resources_used + 1.0)).sum();

        configs
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let uncertainty = 1.0 / (c.resources_used + 1.0);
                let exploitation = c.performance;
                let score = exploitation + exploration * uncertainty;
                let allocation = (score / total_inverse) * self.total_budget;
                (i, allocation)
            })
            .collect()
    }

    fn allocate_racing(
        &mut self,
        configs: &[ConfigurationWithPerformance],
        _confidence: Float,
    ) -> HashMap<usize, Float> {
        // Racing: allocate more to statistically indistinguishable configurations
        let initial = self.total_budget / configs.len() as Float;

        configs
            .iter()
            .enumerate()
            .map(|(i, _)| (i, initial))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_allocation_config() {
        let config = ProgressiveAllocationConfig::default();
        assert_eq!(config.total_budget, 100.0);
        assert_eq!(config.min_resource_per_config, 0.1);
    }

    #[test]
    fn test_coarse_to_fine_config() {
        let config = CoarseToFineConfig::default();
        assert_eq!(config.max_evaluations, 1000);
    }

    #[test]
    fn test_adaptive_fidelity_selector() {
        let strategy = AdaptiveFidelityStrategy::UpperConfidenceBound {
            exploration_weight: 1.0,
        };
        let selector = AdaptiveFidelitySelector::new(strategy);
        assert!(selector.fidelity_stats.is_empty());
    }

    #[test]
    fn test_budget_allocator() {
        let strategy = BudgetAllocationStrategy::Equal;
        let allocator = BudgetAllocator::new(strategy, 100.0);
        assert_eq!(allocator.total_budget, 100.0);
    }

    #[test]
    fn test_progressive_allocator() {
        let config = ProgressiveAllocationConfig::default();
        let allocator = ProgressiveAllocator::new(config);
        assert_eq!(allocator.remaining_budget, 100.0);
    }

    #[test]
    fn test_coarse_to_fine_optimizer() {
        let config = CoarseToFineConfig::default();
        let optimizer = CoarseToFineOptimizer::new(config);
        assert!(optimizer.evaluation_history.is_empty());
    }
}
