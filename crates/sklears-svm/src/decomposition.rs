//! Decomposition methods for large-scale SVM optimization
//!
//! This module provides decomposition algorithms for solving large SVM problems
//! by breaking them down into smaller sub-problems that can be solved efficiently.
//! The main approaches include:
//! - Chunked Sequential Minimal Optimization (SMO)
//! - Matrix decomposition strategies
//! - Hierarchical decomposition
//! - Working set selection algorithms

use crate::kernels::Kernel;
use crate::smo::{SmoConfig, SmoSolver};
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, types::Float};
use std::collections::HashMap;

/// Configuration for decomposition algorithms
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Maximum working set size
    pub max_working_set_size: usize,
    /// Minimum working set size
    pub min_working_set_size: usize,
    /// Number of decomposition levels
    pub decomposition_levels: usize,
    /// Overlap between working sets
    pub working_set_overlap: usize,
    /// Strategy for working set selection
    pub selection_strategy: WorkingSetSelectionStrategy,
    /// Maximum iterations per decomposition step
    pub max_iterations_per_step: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to use hierarchical decomposition
    pub use_hierarchical: bool,
    /// Cache size for kernel evaluations
    pub kernel_cache_size: usize,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_working_set_size: 2000,
            min_working_set_size: 100,
            decomposition_levels: 3,
            working_set_overlap: 50,
            selection_strategy: WorkingSetSelectionStrategy::MaximalViolating,
            max_iterations_per_step: 1000,
            tolerance: 1e-6,
            use_hierarchical: true,
            kernel_cache_size: 256,
        }
    }
}

/// Strategies for selecting working sets in decomposition
#[derive(Debug, Clone, Copy)]
pub enum WorkingSetSelectionStrategy {
    /// Select samples with maximum KKT condition violations
    MaximalViolating,
    /// Random selection with gradient-based weighting
    WeightedRandom,
    /// Steepest feasible direction
    SteepestFeasible,
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Block-wise selection for structured problems
    BlockWise,
}

/// Decomposition-based SVM solver
pub struct DecompositionSolver {
    config: DecompositionConfig,
    kernel: Box<dyn Kernel>,
    working_sets: Vec<WorkingSet>,
    kernel_cache: HashMap<(usize, usize), Float>,
    convergence_history: Vec<Float>,
}

impl DecompositionSolver {
    /// Create a new decomposition solver
    pub fn new(kernel: Box<dyn Kernel>, config: DecompositionConfig) -> Self {
        let cache_capacity = config.kernel_cache_size * config.kernel_cache_size;
        Self {
            config,
            kernel,
            working_sets: Vec::new(),
            kernel_cache: HashMap::with_capacity(cache_capacity),
            convergence_history: Vec::new(),
        }
    }

    /// Solve the SVM problem using decomposition
    pub fn solve(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        c: Float,
        initial_alpha: Option<&Array1<Float>>,
    ) -> Result<(Array1<Float>, Float)> {
        let n_samples = x.nrows();

        // Initialize alpha coefficients
        let mut alpha = initial_alpha
            .cloned()
            .unwrap_or_else(|| Array1::zeros(n_samples));

        // Create initial decomposition
        self.create_initial_decomposition(n_samples)?;

        // Main decomposition loop
        let mut iteration = 0;
        let mut converged = false;

        while !converged
            && iteration < self.config.max_iterations_per_step * self.config.decomposition_levels
        {
            // Solve each working set
            let mut global_change = 0.0;

            // Process working sets one by one to avoid borrowing conflicts
            for i in 0..self.working_sets.len() {
                if !self.working_sets[i].active || self.working_sets[i].indices.len() < 2 {
                    continue;
                }

                // Extract working set data
                let ws_size = self.working_sets[i].indices.len();
                let mut ws_x = Array2::zeros((ws_size, x.ncols()));
                let mut ws_y = Array1::zeros(ws_size);
                let mut ws_alpha = Array1::zeros(ws_size);

                for (j, &idx) in self.working_sets[i].indices.iter().enumerate() {
                    ws_x.row_mut(j).assign(&x.row(idx));
                    ws_y[j] = y[idx];
                    ws_alpha[j] = alpha[idx];
                }

                // For now, use a simplified approach - this would need integration with actual SMO solver
                let new_alpha = ws_alpha.clone(); // Placeholder

                // Compute change in alpha
                let change = (&new_alpha - &ws_alpha).mapv(|x| x.abs()).sum();

                // Update global alpha
                for (j, &idx) in self.working_sets[i].indices.iter().enumerate() {
                    alpha[idx] = new_alpha[j];
                }

                self.working_sets[i].last_change = change;
                global_change += change;
            }

            // Update working sets based on convergence
            self.update_working_sets(&alpha, x, y, c)?;

            // Check convergence
            self.convergence_history.push(global_change);
            converged = self.check_convergence(global_change);

            iteration += 1;
        }

        // Compute bias term
        let bias = self.compute_bias(&alpha, x, y, c)?;

        Ok((alpha, bias))
    }

    /// Create initial decomposition into working sets
    fn create_initial_decomposition(&mut self, n_samples: usize) -> Result<()> {
        self.working_sets.clear();

        match self.config.selection_strategy {
            WorkingSetSelectionStrategy::BlockWise => {
                self.create_block_decomposition(n_samples)?;
            }
            _ => {
                self.create_overlapping_decomposition(n_samples)?;
            }
        }

        Ok(())
    }

    /// Create block-wise decomposition
    fn create_block_decomposition(&mut self, n_samples: usize) -> Result<()> {
        let block_size = self.config.max_working_set_size;
        let mut start = 0;

        while start < n_samples {
            let end = (start + block_size).min(n_samples);
            let indices: Vec<usize> = (start..end).collect();

            self.working_sets.push(WorkingSet {
                indices,
                active: true,
                last_change: Float::INFINITY,
                priority: 1.0,
            });

            start = end;
        }

        Ok(())
    }

    /// Create overlapping decomposition
    fn create_overlapping_decomposition(&mut self, n_samples: usize) -> Result<()> {
        let step_size = self.config.max_working_set_size - self.config.working_set_overlap;
        let mut start = 0;

        while start < n_samples {
            let end = (start + self.config.max_working_set_size).min(n_samples);
            let indices: Vec<usize> = (start..end).collect();

            if indices.len() >= self.config.min_working_set_size {
                self.working_sets.push(WorkingSet {
                    indices,
                    active: true,
                    last_change: Float::INFINITY,
                    priority: 1.0,
                });
            }

            start += step_size;
        }

        Ok(())
    }

    /// Solve a single working set using SMO
    fn solve_working_set(
        &mut self,
        working_set: &mut WorkingSet,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: &mut Array1<Float>,
        c: Float,
    ) -> Result<Float> {
        if !working_set.active || working_set.indices.len() < 2 {
            return Ok(0.0);
        }

        // Extract working set data
        let ws_size = working_set.indices.len();
        let mut ws_x = Array2::zeros((ws_size, x.ncols()));
        let mut ws_y = Array1::zeros(ws_size);
        let mut ws_alpha = Array1::zeros(ws_size);

        for (i, &idx) in working_set.indices.iter().enumerate() {
            ws_x.row_mut(i).assign(&x.row(idx));
            ws_y[i] = y[idx];
            ws_alpha[i] = alpha[idx];
        }

        // Create kernel matrix for working set
        let kernel_matrix = self.compute_cached_kernel_matrix(&ws_x, &working_set.indices);

        // Solve sub-problem using SMO
        let mut smo_config = SmoConfig::default();
        smo_config.max_iter = self.config.max_iterations_per_step;
        smo_config.tol = self.config.tolerance;
        smo_config.c = c;

        // Create a dummy linear kernel for SMO (we'll use the pre-computed matrix)
        use crate::kernels::LinearKernel;
        let dummy_kernel = LinearKernel;
        let smo_solver = SmoSolver::new(smo_config, dummy_kernel);

        // For now, use a simplified approach - this would need integration with the actual SMO solver
        let new_alpha = ws_alpha.clone(); // Placeholder

        // Compute change in alpha
        let change = (&new_alpha - &ws_alpha).mapv(|x| x.abs()).sum();

        // Update global alpha
        for (i, &idx) in working_set.indices.iter().enumerate() {
            alpha[idx] = new_alpha[i];
        }

        working_set.last_change = change;
        Ok(change)
    }

    /// Compute cached kernel matrix for working set
    fn compute_cached_kernel_matrix(
        &mut self,
        x: &Array2<Float>,
        indices: &[usize],
    ) -> Array2<Float> {
        let n = x.nrows();
        let mut kernel_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let key = (indices[i].min(indices[j]), indices[i].max(indices[j]));

                let k_val = if let Some(&cached_val) = self.kernel_cache.get(&key) {
                    cached_val
                } else {
                    let val = self
                        .kernel
                        .compute(x.row(i).to_owned().view(), x.row(j).to_owned().view());

                    // Cache management - remove oldest entries if cache is full
                    if self.kernel_cache.len()
                        >= self.config.kernel_cache_size * self.config.kernel_cache_size
                    {
                        // Simple LRU-like eviction (remove random entry)
                        if let Some(key_to_remove) = self.kernel_cache.keys().next().copied() {
                            self.kernel_cache.remove(&key_to_remove);
                        }
                    }

                    self.kernel_cache.insert(key, val);
                    val
                };

                kernel_matrix[[i, j]] = k_val;
                kernel_matrix[[j, i]] = k_val;
            }
        }

        kernel_matrix
    }

    /// Update working sets based on convergence and violation patterns
    fn update_working_sets(
        &mut self,
        alpha: &Array1<Float>,
        x: &Array2<Float>,
        y: &Array1<Float>,
        c: Float,
    ) -> Result<()> {
        // Compute KKT violations for each sample
        let violations = self.compute_kkt_violations(alpha, x, y, c)?;

        // Update working set priorities and activity
        for working_set in &mut self.working_sets {
            let avg_violation: Float = working_set
                .indices
                .iter()
                .map(|&i| violations[i])
                .sum::<Float>()
                / working_set.indices.len() as Float;

            working_set.priority = avg_violation;
            working_set.active = working_set.last_change > self.config.tolerance * 0.1
                || avg_violation > self.config.tolerance;
        }

        // Optionally create new working sets for high-violation regions
        if self.config.use_hierarchical {
            self.create_adaptive_working_sets(&violations)?;
        }

        Ok(())
    }

    /// Compute KKT condition violations for all samples
    fn compute_kkt_violations(
        &self,
        alpha: &Array1<Float>,
        x: &Array2<Float>,
        y: &Array1<Float>,
        c: Float,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut violations = Array1::zeros(n_samples);

        // Compute decision function values
        let mut decision_values: Array1<Float> = Array1::zeros(n_samples);
        for i in 0..n_samples {
            for j in 0..n_samples {
                if alpha[j] > 0.0 {
                    let k_val = self
                        .kernel
                        .compute(x.row(i).to_owned().view(), x.row(j).to_owned().view());
                    decision_values[i] += alpha[j] * y[j] * k_val;
                }
            }
        }

        // Compute KKT violations
        for i in 0..n_samples {
            let yi_f: Float = y[i] * decision_values[i];

            let one = 1.0 as Float;
            let zero = 0.0 as Float;

            violations[i] = if alpha[i] < 1e-8 {
                // At lower bound
                (one - yi_f).max(zero)
            } else if alpha[i] > c - 1e-8 {
                // At upper bound
                (yi_f - one).max(zero)
            } else {
                // Free variable
                (one - yi_f).abs()
            };
        }

        Ok(violations)
    }

    /// Create adaptive working sets for high-violation regions
    fn create_adaptive_working_sets(&mut self, violations: &Array1<Float>) -> Result<()> {
        let n_samples = violations.len();
        let threshold = violations.iter().fold(0.0, |acc, &x| acc + x) / n_samples as Float;

        // Find high-violation samples
        let high_violation_indices: Vec<usize> = (0..n_samples)
            .filter(|&i| violations[i] > threshold * 2.0)
            .collect();

        if high_violation_indices.len() >= self.config.min_working_set_size {
            // Create new working set for high-violation samples
            let mut new_working_set = WorkingSet {
                indices: high_violation_indices,
                active: true,
                last_change: Float::INFINITY,
                priority: threshold * 2.0,
            };

            // Limit size if too large
            if new_working_set.indices.len() > self.config.max_working_set_size {
                new_working_set
                    .indices
                    .truncate(self.config.max_working_set_size);
            }

            self.working_sets.push(new_working_set);
        }

        Ok(())
    }

    /// Check global convergence
    fn check_convergence(&self, change: Float) -> bool {
        if self.convergence_history.len() < 3 {
            return false;
        }

        // Check if change is below tolerance
        if change < self.config.tolerance {
            return true;
        }

        // Check if convergence has stagnated
        let recent_changes: Float =
            self.convergence_history.iter().rev().take(3).sum::<Float>() / 3.0;

        recent_changes < self.config.tolerance * 10.0
    }

    /// Compute bias term from support vectors
    fn compute_bias(
        &self,
        alpha: &Array1<Float>,
        x: &Array2<Float>,
        y: &Array1<Float>,
        c: Float,
    ) -> Result<Float> {
        let n_samples = x.nrows();
        let mut bias_sum = 0.0;
        let mut n_free_sv = 0;

        for i in 0..n_samples {
            if alpha[i] > 1e-8 && alpha[i] < c - 1e-8 {
                // Free support vector
                let mut decision_value = 0.0;
                for j in 0..n_samples {
                    if alpha[j] > 1e-8 {
                        let k_val = self
                            .kernel
                            .compute(x.row(i).to_owned().view(), x.row(j).to_owned().view());
                        decision_value += alpha[j] * y[j] * k_val;
                    }
                }
                bias_sum += y[i] - decision_value;
                n_free_sv += 1;
            }
        }

        if n_free_sv > 0 {
            Ok(bias_sum / n_free_sv as Float)
        } else {
            Ok(0.0)
        }
    }

    /// Get convergence statistics
    pub fn get_convergence_history(&self) -> &[Float] {
        &self.convergence_history
    }

    /// Get number of active working sets
    pub fn get_active_working_sets(&self) -> usize {
        self.working_sets.iter().filter(|ws| ws.active).count()
    }
}

/// Working set for decomposition algorithm
#[derive(Debug, Clone)]
struct WorkingSet {
    /// Indices of samples in this working set
    indices: Vec<usize>,
    /// Whether this working set is active
    active: bool,
    /// Last change in objective function
    last_change: Float,
    /// Priority for selection
    priority: Float,
}

/// Hierarchical decomposition for very large problems
pub struct HierarchicalDecomposer {
    levels: Vec<DecompositionLevel>,
    config: DecompositionConfig,
}

impl HierarchicalDecomposer {
    /// Create new hierarchical decomposer
    pub fn new(config: DecompositionConfig) -> Self {
        Self {
            levels: Vec::new(),
            config,
        }
    }

    /// Decompose problem hierarchically
    pub fn decompose(&mut self, n_samples: usize) -> Result<()> {
        self.levels.clear();

        let mut current_size = n_samples;
        let mut level = 0;

        while current_size > self.config.max_working_set_size
            && level < self.config.decomposition_levels
        {
            let reduction_factor =
                (self.config.max_working_set_size as Float / current_size as Float).sqrt();
            let new_size = (current_size as Float * reduction_factor).ceil() as usize;

            self.levels.push(DecompositionLevel {
                level,
                original_size: current_size,
                reduced_size: new_size,
                reduction_factor,
                mapping: self.create_level_mapping(current_size, new_size)?,
            });

            current_size = new_size;
            level += 1;
        }

        Ok(())
    }

    /// Create mapping between levels
    fn create_level_mapping(&self, from_size: usize, to_size: usize) -> Result<Vec<Vec<usize>>> {
        let cluster_size = (from_size as Float / to_size as Float).ceil() as usize;
        let mut mapping = Vec::with_capacity(to_size);

        for i in 0..to_size {
            let start = i * cluster_size;
            let end = ((i + 1) * cluster_size).min(from_size);
            mapping.push((start..end).collect());
        }

        Ok(mapping)
    }
}

/// Single level in hierarchical decomposition
#[derive(Debug, Clone)]
struct DecompositionLevel {
    level: usize,
    original_size: usize,
    reduced_size: usize,
    reduction_factor: Float,
    mapping: Vec<Vec<usize>>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{LinearKernel, RbfKernel};

    #[test]
    fn test_decomposition_solver_creation() {
        let kernel = Box::new(LinearKernel);
        let config = DecompositionConfig::default();
        let solver = DecompositionSolver::new(kernel, config);
        assert_eq!(solver.working_sets.len(), 0);
    }

    #[test]
    fn test_block_decomposition() {
        let kernel = Box::new(RbfKernel::new(1.0));
        let mut config = DecompositionConfig::default();
        config.max_working_set_size = 100;
        config.selection_strategy = WorkingSetSelectionStrategy::BlockWise;

        let mut solver = DecompositionSolver::new(kernel, config);
        solver.create_initial_decomposition(250).unwrap();

        assert_eq!(solver.working_sets.len(), 3); // 250 / 100 = 2.5 -> 3 blocks
    }

    #[test]
    fn test_hierarchical_decomposer() {
        let config = DecompositionConfig::default();
        let mut decomposer = HierarchicalDecomposer::new(config);

        decomposer.decompose(10000).unwrap();
        assert!(!decomposer.levels.is_empty());
    }
}
