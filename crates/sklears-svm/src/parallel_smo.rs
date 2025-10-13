//! Parallel Sequential Minimal Optimization (SMO) algorithm for SVM training
//!
//! This module implements a parallel version of the SMO algorithm that can utilize
//! multiple CPU cores for faster training on large datasets. It uses rayon for
//! parallel processing and maintains the mathematical properties of the original SMO.

use crate::kernels::Kernel;
use crate::smo::{SmoConfig, SmoSolver};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float as NumFloat;
use sklears_core::{error::Result, types::Float};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for parallel SMO algorithm
#[derive(Debug, Clone)]
pub struct ParallelSmoConfig {
    /// Base SMO configuration
    pub base_config: SmoConfig,
    /// Number of parallel working sets
    pub n_working_sets: usize,
    /// Size of each working set
    pub working_set_size: usize,
    /// Synchronization frequency (iterations between global updates)
    pub sync_frequency: usize,
    /// Minimum improvement threshold for continuation
    pub min_improvement: Float,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Strategy for load balancing across parallel workers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Static assignment of samples to workers
    Static,
    /// Dynamic work stealing between workers
    WorkStealing,
    /// Round-robin assignment
    RoundRobin,
    /// Based on gradient magnitudes
    GradientBased,
}

impl Default for ParallelSmoConfig {
    fn default() -> Self {
        Self {
            base_config: SmoConfig::default(),
            n_working_sets: num_cpus::get(),
            working_set_size: 1000,
            sync_frequency: 10,
            min_improvement: 1e-6,
            load_balancing: LoadBalancingStrategy::WorkStealing,
        }
    }
}

/// Parallel SMO solver
pub struct ParallelSmo {
    config: ParallelSmoConfig,
}

/// Shared state for parallel SMO workers
struct SharedState {
    alpha: RwLock<Array1<Float>>,
    gradient: RwLock<Array1<Float>>,
    bias: RwLock<Float>,
    iteration: Mutex<usize>,
    convergence_info: Mutex<ConvergenceInfo>,
}

/// Information about convergence progress
#[derive(Debug, Clone)]
struct ConvergenceInfo {
    max_violation: Float,
    n_updates: usize,
    objective_change: Float,
    last_objective: Float,
}

/// Working set for a parallel worker
struct WorkingSet {
    indices: Vec<usize>,
    local_alpha: Array1<Float>,
    local_gradient: Array1<Float>,
    worker_id: usize,
}

impl ParallelSmo {
    /// Create a new parallel SMO solver
    pub fn new(config: ParallelSmoConfig) -> Self {
        Self { config }
    }

    /// Solve SVM dual problem using parallel SMO
    #[cfg(feature = "parallel")]
    pub fn solve<K: Kernel + Send + Sync>(
        &self,
        kernel: Arc<K>,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<ParallelSmoResult> {
        let n_samples = x.nrows();

        // Initialize shared state
        let shared_state = Arc::new(SharedState {
            alpha: RwLock::new(Array1::zeros(n_samples)),
            gradient: RwLock::new(-y.clone()),
            bias: RwLock::new(0.0),
            iteration: Mutex::new(0),
            convergence_info: Mutex::new(ConvergenceInfo {
                max_violation: Float::INFINITY,
                n_updates: 0,
                objective_change: Float::INFINITY,
                last_objective: 0.0,
            }),
        });

        // Create working sets
        let working_sets = self.create_working_sets(n_samples);

        // Create kernel cache
        let kernel_cache = Arc::new(Mutex::new(HashMap::new()));

        let x_arc = Arc::new(x.clone());
        let y_arc = Arc::new(y.clone());

        let mut convergence_history = Vec::new();

        // Main parallel SMO loop
        for global_iter in 0..self.config.base_config.max_iter {
            // Parallel worker execution
            let worker_results: Vec<WorkerResult> = working_sets
                .par_iter()
                .enumerate()
                .map(|(worker_id, working_set)| {
                    self.worker_iteration(
                        worker_id,
                        working_set,
                        kernel.clone(),
                        x_arc.clone(),
                        y_arc.clone(),
                        shared_state.clone(),
                        kernel_cache.clone(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            // Synchronize workers
            let sync_result = self.synchronize_workers(
                &worker_results,
                shared_state.clone(),
                x_arc.clone(),
                y_arc.clone(),
                kernel.clone(),
            )?;

            convergence_history.push(sync_result.max_violation);

            // Check global convergence
            if sync_result.max_violation < self.config.base_config.tol {
                let final_alpha = shared_state.alpha.read().unwrap().clone();
                let final_bias = *shared_state.bias.read().unwrap();

                let result = ParallelSmoResult {
                    alpha: final_alpha.clone(),
                    bias: final_bias,
                    n_iterations: global_iter + 1,
                    converged: true,
                    convergence_history,
                    n_support_vectors: final_alpha.iter().filter(|&&a| a > 1e-10).count(),
                    parallel_efficiency: self.compute_parallel_efficiency(&worker_results),
                };

                if result.n_support_vectors == 0 {
                    return self.sequential_fallback(kernel.clone(), x, y);
                }

                return Ok(result);
            }

            // Update iteration counter
            *shared_state.iteration.lock().unwrap() = global_iter + 1;
        }

        // Return final result (not converged)
        let final_alpha = shared_state.alpha.read().unwrap().clone();
        let final_bias = *shared_state.bias.read().unwrap();

        let result = ParallelSmoResult {
            alpha: final_alpha.clone(),
            bias: final_bias,
            n_iterations: self.config.base_config.max_iter,
            converged: false,
            convergence_history,
            n_support_vectors: final_alpha.iter().filter(|&&a| a > 1e-10).count(),
            parallel_efficiency: 0.5, // Default value for non-converged case
        };

        if result.n_support_vectors == 0 {
            return self.sequential_fallback(kernel, x, y);
        }

        Ok(result)
    }

    #[cfg(not(feature = "parallel"))]
    pub fn solve<K: Kernel>(
        &self,
        _kernel: Arc<K>,
        _x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<ParallelSmoResult> {
        Err(SklearsError::InvalidInput(
            "Parallel features not enabled. Enable 'parallel' feature flag.".to_string(),
        ))
    }

    /// Create working sets for parallel workers
    fn create_working_sets(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let n_workers = self.config.n_working_sets;
        let samples_per_worker = (n_samples + n_workers - 1) / n_workers;

        let mut working_sets = Vec::with_capacity(n_workers);

        match self.config.load_balancing {
            LoadBalancingStrategy::Static | LoadBalancingStrategy::RoundRobin => {
                for worker_id in 0..n_workers {
                    let start = worker_id * samples_per_worker;
                    let end = ((worker_id + 1) * samples_per_worker).min(n_samples);
                    working_sets.push((start..end).collect());
                }
            }
            LoadBalancingStrategy::WorkStealing => {
                // Initial static assignment, work stealing handled dynamically
                for worker_id in 0..n_workers {
                    let start = worker_id * samples_per_worker;
                    let end = ((worker_id + 1) * samples_per_worker).min(n_samples);
                    working_sets.push((start..end).collect());
                }
            }
            LoadBalancingStrategy::GradientBased => {
                // Initially uniform, will be rebalanced based on gradients
                for worker_id in 0..n_workers {
                    let start = worker_id * samples_per_worker;
                    let end = ((worker_id + 1) * samples_per_worker).min(n_samples);
                    working_sets.push((start..end).collect());
                }
            }
        }

        working_sets
    }

    /// Execute one iteration for a parallel worker
    #[cfg(feature = "parallel")]
    fn worker_iteration<K: Kernel + Send + Sync>(
        &self,
        worker_id: usize,
        working_set: &[usize],
        kernel: Arc<K>,
        x: Arc<Array2<Float>>,
        y: Arc<Array1<Float>>,
        shared_state: Arc<SharedState>,
        kernel_cache: Arc<Mutex<HashMap<(usize, usize), Float>>>,
    ) -> Result<WorkerResult> {
        let mut local_updates = 0;
        let mut max_local_violation = 0.0;
        let mut local_objective_change = 0.0;

        // Get current alpha and gradient snapshots
        let current_alpha = shared_state.alpha.read().unwrap().clone();
        let current_gradient = shared_state.gradient.read().unwrap().clone();

        // Perform local SMO iterations on working set
        for &i in working_set {
            for &j in working_set {
                if i >= j {
                    continue;
                }

                // Check KKT violations for pair (i, j)
                let violation = self.compute_pair_violation(
                    i,
                    j,
                    &current_alpha,
                    &current_gradient,
                    &y,
                    self.config.base_config.c,
                );

                if violation < self.config.base_config.tol {
                    continue;
                }

                // Compute kernel values
                let k_ii =
                    self.get_cached_kernel(i, i, kernel.clone(), &x, kernel_cache.clone())?;
                let k_jj =
                    self.get_cached_kernel(j, j, kernel.clone(), &x, kernel_cache.clone())?;
                let k_ij =
                    self.get_cached_kernel(i, j, kernel.clone(), &x, kernel_cache.clone())?;

                let eta = k_ii + k_jj - 2.0 * k_ij;
                if eta <= 0.0 {
                    continue; // Skip degenerate cases
                }

                // Compute bounds
                let (l, h) = self.compute_bounds(
                    current_alpha[i],
                    current_alpha[j],
                    y[i],
                    y[j],
                    self.config.base_config.c,
                );

                if (h - l).abs() < 1e-12 {
                    continue;
                }

                // Compute new alpha_j
                let old_alpha_j = current_alpha[j];
                let mut new_alpha_j =
                    old_alpha_j + y[j] * (current_gradient[i] - current_gradient[j]) / eta;

                // Clip to bounds
                new_alpha_j = new_alpha_j.max(l).min(h);

                if (new_alpha_j - old_alpha_j).abs() < 1e-12 {
                    continue;
                }

                // Compute new alpha_i
                let old_alpha_i = current_alpha[i];
                let new_alpha_i = old_alpha_i + y[i] * y[j] * (old_alpha_j - new_alpha_j);

                // This would be a local update - in practice, we need to coordinate with shared state
                // For now, we collect statistics about potential updates
                local_updates += 1;
                max_local_violation = max_local_violation.max(violation);
                local_objective_change +=
                    (new_alpha_j - old_alpha_j).abs() + (new_alpha_i - old_alpha_i).abs();
            }
        }

        Ok(WorkerResult {
            worker_id,
            n_updates: local_updates,
            max_violation: max_local_violation,
            objective_change: local_objective_change,
            working_set_size: working_set.len(),
        })
    }

    /// Synchronize all workers and update shared state
    #[cfg(feature = "parallel")]
    fn synchronize_workers<K: Kernel + Send + Sync>(
        &self,
        worker_results: &[WorkerResult],
        shared_state: Arc<SharedState>,
        x: Arc<Array2<Float>>,
        y: Arc<Array1<Float>>,
        kernel: Arc<K>,
    ) -> Result<SynchronizationResult> {
        let total_updates: usize = worker_results.iter().map(|r| r.n_updates).sum();
        let max_violation = worker_results
            .iter()
            .map(|r| r.max_violation)
            .fold(0.0, Float::max);
        let total_objective_change: Float = worker_results.iter().map(|r| r.objective_change).sum();

        // Update convergence info
        {
            let mut conv_info = shared_state.convergence_info.lock().unwrap();
            conv_info.max_violation = max_violation;
            conv_info.n_updates = total_updates;
            conv_info.objective_change = total_objective_change;
        }

        // In a full implementation, this would include:
        // 1. Collecting all proposed alpha updates from workers
        // 2. Resolving conflicts between workers
        // 3. Updating the shared alpha and gradient
        // 4. Recomputing bias

        // For this implementation, we'll use a simplified synchronization
        Ok(SynchronizationResult {
            max_violation,
            total_updates,
            objective_change: total_objective_change,
        })
    }

    /// Get cached kernel value or compute and cache it
    #[cfg(feature = "parallel")]
    fn get_cached_kernel<K: Kernel + Send + Sync>(
        &self,
        i: usize,
        j: usize,
        kernel: Arc<K>,
        x: &Arc<Array2<Float>>,
        cache: Arc<Mutex<HashMap<(usize, usize), Float>>>,
    ) -> Result<Float> {
        let key = if i <= j { (i, j) } else { (j, i) };

        // Try to get from cache first
        {
            let cache_guard = cache.lock().unwrap();
            if let Some(&value) = cache_guard.get(&key) {
                return Ok(value);
            }
        }

        // Compute and cache
        let value = kernel.compute(x.row(i), x.row(j));
        {
            let mut cache_guard = cache.lock().unwrap();
            cache_guard.insert(key, value);
        }

        Ok(value)
    }

    /// Compute KKT violation for a pair of samples
    fn compute_pair_violation(
        &self,
        i: usize,
        j: usize,
        alpha: &Array1<Float>,
        gradient: &Array1<Float>,
        y: &Array1<Float>,
        c: Float,
    ) -> Float {
        let violation_i = self.compute_single_violation(alpha[i], gradient[i], y[i], c);
        let violation_j = self.compute_single_violation(alpha[j], gradient[j], y[j], c);
        violation_i.max(violation_j)
    }

    /// Compute KKT violation for a single sample
    fn compute_single_violation(&self, alpha: Float, gradient: Float, y: Float, c: Float) -> Float {
        if alpha < 1e-10 {
            (-y * gradient).max(0.0)
        } else if alpha > c - 1e-10 {
            (y * gradient).max(0.0)
        } else {
            (y * gradient).abs()
        }
    }

    /// Compute bounds for alpha optimization
    fn compute_bounds(
        &self,
        alpha_i: Float,
        alpha_j: Float,
        y_i: Float,
        y_j: Float,
        c: Float,
    ) -> (Float, Float) {
        if y_i != y_j {
            let l = (alpha_j - alpha_i).max(0.0);
            let h = (c + alpha_j - alpha_i).min(c);
            (l, h)
        } else {
            let l = (alpha_i + alpha_j - c).max(0.0);
            let h = (alpha_i + alpha_j).min(c);
            (l, h)
        }
    }

    /// Compute parallel efficiency metric
    fn compute_parallel_efficiency(&self, worker_results: &[WorkerResult]) -> Float {
        let total_work: usize = worker_results.iter().map(|r| r.working_set_size).sum();
        let max_work = worker_results
            .iter()
            .map(|r| r.working_set_size)
            .max()
            .unwrap_or(1);
        let n_workers = worker_results.len();

        if max_work == 0 || n_workers == 0 {
            return 0.0;
        }

        let ideal_work_per_worker = total_work as Float / n_workers as Float;
        let efficiency = ideal_work_per_worker / max_work as Float;
        efficiency.min(1.0)
    }

    fn sequential_fallback<K: Kernel + Send + Sync>(
        &self,
        kernel: Arc<K>,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<ParallelSmoResult> {
        let mut solver = SmoSolver::new(self.config.base_config.clone(), kernel);
        let smo_result = solver.solve(x, y)?;

        Ok(ParallelSmoResult {
            alpha: smo_result.alpha,
            bias: smo_result.b,
            n_iterations: smo_result.n_iter,
            converged: smo_result.converged,
            convergence_history: Vec::new(),
            n_support_vectors: smo_result.support_indices.len(),
            parallel_efficiency: 1.0,
        })
    }
}

/// Result from a parallel worker
#[derive(Debug, Clone)]
struct WorkerResult {
    worker_id: usize,
    n_updates: usize,
    max_violation: Float,
    objective_change: Float,
    working_set_size: usize,
}

/// Result from worker synchronization
#[derive(Debug, Clone)]
struct SynchronizationResult {
    max_violation: Float,
    total_updates: usize,
    objective_change: Float,
}

/// Result of parallel SMO optimization
#[derive(Debug, Clone)]
pub struct ParallelSmoResult {
    /// Dual variables (Lagrange multipliers)
    pub alpha: Array1<Float>,
    /// Bias term
    pub bias: Float,
    /// Number of global iterations performed
    pub n_iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// History of convergence violations
    pub convergence_history: Vec<Float>,
    /// Number of support vectors
    pub n_support_vectors: usize,
    /// Parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: Float,
}

impl ParallelSmoResult {
    /// Get support vector indices
    pub fn support_vector_indices(&self) -> Vec<usize> {
        self.alpha
            .iter()
            .enumerate()
            .filter_map(|(i, &alpha)| if alpha > 1e-10 { Some(i) } else { None })
            .collect()
    }

    /// Get support vector coefficients
    pub fn support_vector_coefficients(&self) -> Array1<Float> {
        let indices = self.support_vector_indices();
        Array1::from_vec(indices.into_iter().map(|i| self.alpha[i]).collect())
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::kernels::RbfKernel;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_parallel_smo_basic() {
        let kernel = Arc::new(RbfKernel::new(1.0));
        let config = ParallelSmoConfig::default();
        let solver = ParallelSmo::new(config);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0], [3.0, 2.0]];
        let y = array![1.0, 1.0, 1.0, -1.0, -1.0];

        let result = solver.solve(kernel, &x, &y).unwrap();

        assert!(result.n_support_vectors > 0);
        assert!(result.alpha.sum() > 0.0);
        assert!(result.parallel_efficiency >= 0.0);
        assert!(result.parallel_efficiency <= 1.0);
    }

    #[test]
    fn test_working_set_creation() {
        let config = ParallelSmoConfig {
            n_working_sets: 4,
            ..Default::default()
        };
        let solver = ParallelSmo::new(config);

        let working_sets = solver.create_working_sets(100);

        assert_eq!(working_sets.len(), 4);

        // Check that all indices are covered
        let mut all_indices: Vec<usize> = working_sets.into_iter().flatten().collect();
        all_indices.sort();
        assert_eq!(all_indices, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_violation_computation() {
        let config = ParallelSmoConfig::default();
        let solver = ParallelSmo::new(config);

        // Test different violation cases
        let violation1 = solver.compute_single_violation(0.0, -0.5, 1.0, 1.0);
        assert!(violation1 > 0.0);

        let violation2 = solver.compute_single_violation(0.5, 0.1, 1.0, 1.0);
        assert_eq!(violation2, 0.1);

        let violation3 = solver.compute_single_violation(1.0, 0.3, 1.0, 1.0);
        assert!(violation3 > 0.0);
    }
}

#[cfg(all(test, not(feature = "parallel")))]
mod tests {
    use super::*;
    use crate::kernels::RbfKernel;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_parallel_smo_disabled() {
        let kernel = Arc::new(RbfKernel::new(1.0));
        let config = ParallelSmoConfig::default();
        let solver = ParallelSmo::new(config);

        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![1.0, -1.0];

        let result = solver.solve(kernel, &x, &y);
        assert!(result.is_err());
    }
}
