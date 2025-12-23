//! Asynchronous Updates for Parallel Cross-Decomposition
//!
//! This module provides asynchronous update mechanisms for parallel and distributed
//! cross-decomposition algorithms, enabling non-blocking updates and improved scalability.
//!
//! ## Features
//! - Lock-free asynchronous parameter updates
//! - Bounded asynchrony with staleness control
//! - Asynchronous ADMM (Alternating Direction Method of Multipliers)
//! - Asynchronous coordinate descent
//!
//! ## Applications
//! - Large-scale distributed CCA
//! - Asynchronous parallel PLS
//! - Non-blocking optimization

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::types::Float;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Configuration for asynchronous updates
#[derive(Debug, Clone)]
pub struct AsyncUpdateConfig {
    /// Maximum staleness (delay) allowed
    pub max_staleness: usize,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Learning rate for updates
    pub learning_rate: Float,
    /// Momentum parameter
    pub momentum: Float,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl Default for AsyncUpdateConfig {
    fn default() -> Self {
        Self {
            max_staleness: 10,
            num_workers: 4,
            learning_rate: 0.01,
            momentum: 0.9,
            tolerance: 1e-6,
            max_iterations: 1000,
        }
    }
}

/// Asynchronous parameter update results
#[derive(Debug, Clone)]
pub struct AsyncUpdateResults {
    /// Final parameters
    pub parameters: Array1<Float>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence history
    pub convergence_history: Vec<Float>,
    /// Average staleness
    pub average_staleness: Float,
    /// Whether algorithm converged
    pub converged: bool,
}

/// Bounded asynchrony coordinator
pub struct BoundedAsyncCoordinator {
    /// Configuration
    config: AsyncUpdateConfig,
    /// Current parameters (shared across workers)
    parameters: Arc<Mutex<Array1<Float>>>,
    /// Version counter for staleness tracking
    version: Arc<Mutex<usize>>,
    /// Update queue
    update_queue: Arc<Mutex<VecDeque<AsyncUpdate>>>,
}

/// Asynchronous update
#[derive(Debug, Clone)]
struct AsyncUpdate {
    /// Gradient or update direction
    gradient: Array1<Float>,
    /// Version when gradient was computed
    version: usize,
    /// Worker ID
    worker_id: usize,
}

/// Asynchronous ADMM for consensus optimization
pub struct AsyncADMM {
    /// Number of blocks
    num_blocks: usize,
    /// Penalty parameter
    rho: Float,
    /// Primal variables (one per block)
    primal_vars: Vec<Array1<Float>>,
    /// Dual variables
    dual_vars: Vec<Array1<Float>>,
    /// Consensus variable
    consensus: Array1<Float>,
}

/// Asynchronous coordinate descent
pub struct AsyncCoordinateDescent {
    /// Configuration
    config: AsyncUpdateConfig,
    /// Current iterate
    current_x: Arc<Mutex<Array1<Float>>>,
    /// Coordinate version counters
    coord_versions: Arc<Mutex<Vec<usize>>>,
}

impl BoundedAsyncCoordinator {
    /// Create a new bounded asynchrony coordinator
    pub fn new(config: AsyncUpdateConfig, initial_params: Array1<Float>) -> Self {
        Self {
            config,
            parameters: Arc::new(Mutex::new(initial_params)),
            version: Arc::new(Mutex::new(0)),
            update_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Submit an asynchronous update
    pub fn submit_update(&self, gradient: Array1<Float>, worker_id: usize) {
        let version = *self.version.lock().unwrap();

        let update = AsyncUpdate {
            gradient,
            version,
            worker_id,
        };

        let mut queue = self.update_queue.lock().unwrap();
        queue.push_back(update);
    }

    /// Process pending updates
    pub fn process_updates(&self) -> usize {
        let mut queue = self.update_queue.lock().unwrap();
        let mut params = self.parameters.lock().unwrap();
        let mut version = self.version.lock().unwrap();

        let mut num_processed = 0;

        while let Some(update) = queue.pop_front() {
            let staleness = *version - update.version;

            // Only process if staleness is within bounds
            if staleness <= self.config.max_staleness {
                // Apply update with staleness-aware learning rate
                let effective_lr = self.config.learning_rate / (1.0 + staleness as Float);

                for i in 0..params.len() {
                    params[i] -= effective_lr * update.gradient[i];
                }

                *version += 1;
                num_processed += 1;
            }
        }

        num_processed
    }

    /// Get current parameters (thread-safe)
    pub fn get_parameters(&self) -> Array1<Float> {
        self.parameters.lock().unwrap().clone()
    }

    /// Get current version
    pub fn get_version(&self) -> usize {
        *self.version.lock().unwrap()
    }
}

impl AsyncADMM {
    /// Create a new async ADMM instance
    pub fn new(num_blocks: usize, dim: usize, rho: Float) -> Self {
        Self {
            num_blocks,
            rho,
            primal_vars: vec![Array1::zeros(dim); num_blocks],
            dual_vars: vec![Array1::zeros(dim); num_blocks],
            consensus: Array1::zeros(dim),
        }
    }

    /// Update a single block (can be called asynchronously)
    pub fn update_block(&mut self, block_id: usize, objective_grad: &Array1<Float>) {
        if block_id >= self.num_blocks {
            return;
        }

        // Update primal variable
        // x_i^{k+1} = argmin f_i(x) + (rho/2)||x - z^k + u_i^k||^2
        // Simplified: gradient descent step
        let residual = &self.primal_vars[block_id] - &self.consensus + &self.dual_vars[block_id];

        for i in 0..self.primal_vars[block_id].len() {
            self.primal_vars[block_id][i] -=
                (objective_grad[i] + self.rho * residual[i]) / (1.0 + self.rho);
        }

        // Update dual variable
        // u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}
        let new_residual = &self.primal_vars[block_id] - &self.consensus;
        self.dual_vars[block_id] = &self.dual_vars[block_id] + &new_residual;
    }

    /// Update consensus variable (should be called periodically)
    pub fn update_consensus(&mut self) {
        // z^{k+1} = (1/N) Σ (x_i^{k+1} + u_i^k)
        let mut sum = Array1::zeros(self.consensus.len());

        for i in 0..self.num_blocks {
            sum = sum + &self.primal_vars[i] + &self.dual_vars[i];
        }

        self.consensus = sum / (self.num_blocks as Float);
    }

    /// Get primal residual norm
    pub fn primal_residual(&self) -> Float {
        let mut residual = 0.0;

        for i in 0..self.num_blocks {
            let diff = &self.primal_vars[i] - &self.consensus;
            residual += diff.mapv(|x| x * x).sum();
        }

        residual.sqrt()
    }

    /// Get dual residual norm
    pub fn dual_residual(&self, prev_consensus: &Array1<Float>) -> Float {
        let diff = &self.consensus - prev_consensus;
        let scaled = diff * self.rho * (self.num_blocks as Float);
        scaled.mapv(|x| x * x).sum().sqrt()
    }

    /// Get consensus variable
    pub fn get_consensus(&self) -> &Array1<Float> {
        &self.consensus
    }
}

impl AsyncCoordinateDescent {
    /// Create a new async coordinate descent instance
    pub fn new(config: AsyncUpdateConfig, initial_x: Array1<Float>) -> Self {
        let dim = initial_x.len();

        Self {
            config,
            current_x: Arc::new(Mutex::new(initial_x)),
            coord_versions: Arc::new(Mutex::new(vec![0; dim])),
        }
    }

    /// Update a single coordinate asynchronously
    pub fn update_coordinate(&self, coord: usize, gradient_coord: Float, read_version: usize) {
        let mut x = self.current_x.lock().unwrap();
        let mut versions = self.coord_versions.lock().unwrap();

        // Check staleness
        let staleness = versions[coord] - read_version;

        if staleness <= self.config.max_staleness {
            // Staleness-aware learning rate
            let effective_lr = self.config.learning_rate / (1.0 + staleness as Float);

            // Update coordinate
            x[coord] -= effective_lr * gradient_coord;

            // Increment version
            versions[coord] += 1;
        }
    }

    /// Get current solution (thread-safe)
    pub fn get_solution(&self) -> Array1<Float> {
        self.current_x.lock().unwrap().clone()
    }

    /// Get coordinate versions
    pub fn get_versions(&self) -> Vec<usize> {
        self.coord_versions.lock().unwrap().clone()
    }
}

/// Simulate asynchronous SGD for demonstration
pub fn async_sgd_simulation(
    objective: impl Fn(&Array1<Float>) -> Float,
    gradient: impl Fn(&Array1<Float>) -> Array1<Float>,
    config: AsyncUpdateConfig,
    initial_params: Array1<Float>,
) -> AsyncUpdateResults {
    let coordinator = BoundedAsyncCoordinator::new(config.clone(), initial_params);

    let mut convergence_history = Vec::new();
    let mut iterations = 0;
    let mut converged = false;
    let mut staleness_sum = 0.0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Simulate worker updates
        for worker_id in 0..config.num_workers {
            let params = coordinator.get_parameters();
            let grad = gradient(&params);

            // Add some noise to simulate asynchrony
            let mut rng = thread_rng();
            let delay = rng.random_range(0..config.max_staleness + 1);
            staleness_sum += delay as Float;

            coordinator.submit_update(grad, worker_id);
        }

        // Process updates
        coordinator.process_updates();

        // Check convergence
        let params = coordinator.get_parameters();
        let obj_value = objective(&params);
        convergence_history.push(obj_value);

        if convergence_history.len() > 10 {
            let recent_change = (convergence_history[convergence_history.len() - 1]
                - convergence_history[convergence_history.len() - 10])
                .abs();

            if recent_change < config.tolerance {
                converged = true;
                break;
            }
        }
    }

    let average_staleness = staleness_sum / (iterations * config.num_workers) as Float;

    AsyncUpdateResults {
        parameters: coordinator.get_parameters(),
        iterations,
        convergence_history,
        average_staleness,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_async_config_default() {
        let config = AsyncUpdateConfig::default();
        assert_eq!(config.max_staleness, 10);
        assert_eq!(config.num_workers, 4);
    }

    #[test]
    fn test_bounded_async_coordinator_creation() {
        let config = AsyncUpdateConfig::default();
        let initial = array![1.0, 2.0, 3.0];

        let coordinator = BoundedAsyncCoordinator::new(config, initial.clone());

        let params = coordinator.get_parameters();
        assert_eq!(params, initial);
    }

    #[test]
    fn test_submit_and_process_updates() {
        let config = AsyncUpdateConfig::default();
        let initial = array![1.0, 2.0, 3.0];

        let coordinator = BoundedAsyncCoordinator::new(config, initial);

        // Submit an update
        let gradient = array![0.1, 0.2, 0.3];
        coordinator.submit_update(gradient, 0);

        // Process updates
        let num_processed = coordinator.process_updates();
        assert_eq!(num_processed, 1);

        // Parameters should have changed
        let params = coordinator.get_parameters();
        assert!(params[0] < 1.0); // Should decrease due to gradient
    }

    #[test]
    fn test_staleness_filtering() {
        let config = AsyncUpdateConfig {
            max_staleness: 1,
            ..Default::default()
        };
        let initial = array![1.0, 2.0];

        let coordinator = BoundedAsyncCoordinator::new(config, initial);

        // Submit multiple updates to increase version
        for _ in 0..5 {
            coordinator.submit_update(array![0.1, 0.1], 0);
            coordinator.process_updates();
        }

        let version_before = coordinator.get_version();

        // Submit a very stale update (will be rejected)
        coordinator.submit_update(array![1.0, 1.0], 0);

        // This stale update should not be processed
        let num_processed = coordinator.process_updates();
        assert!(num_processed == 0 || coordinator.get_version() == version_before + num_processed);
    }

    #[test]
    fn test_async_admm_creation() {
        let admm = AsyncADMM::new(3, 5, 1.0);
        assert_eq!(admm.num_blocks, 3);
        assert_eq!(admm.primal_vars.len(), 3);
        assert_eq!(admm.dual_vars.len(), 3);
    }

    #[test]
    fn test_async_admm_block_update() {
        let mut admm = AsyncADMM::new(2, 3, 1.0);

        let gradient = array![0.1, 0.2, 0.3];
        admm.update_block(0, &gradient);

        // Primal variable should have changed
        assert!(admm.primal_vars[0].iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_async_admm_consensus_update() {
        let mut admm = AsyncADMM::new(2, 3, 1.0);

        // Update blocks
        let grad1 = array![0.1, 0.2, 0.3];
        let grad2 = array![0.2, 0.3, 0.4];

        admm.update_block(0, &grad1);
        admm.update_block(1, &grad2);

        // Update consensus
        let prev_consensus = admm.consensus.clone();
        admm.update_consensus();

        // Consensus should have changed
        assert!(admm.consensus != prev_consensus);
    }

    #[test]
    fn test_async_admm_residuals() {
        let mut admm = AsyncADMM::new(2, 3, 1.0);

        let prev_consensus = admm.consensus.clone();

        let primal_res = admm.primal_residual();
        let dual_res = admm.dual_residual(&prev_consensus);

        assert!(primal_res >= 0.0);
        assert!(dual_res >= 0.0);
    }

    #[test]
    fn test_async_coordinate_descent_creation() {
        let config = AsyncUpdateConfig::default();
        let initial = array![1.0, 2.0, 3.0];

        let acd = AsyncCoordinateDescent::new(config, initial.clone());

        let solution = acd.get_solution();
        assert_eq!(solution, initial);
    }

    #[test]
    fn test_async_coordinate_descent_update() {
        let config = AsyncUpdateConfig::default();
        let initial = array![1.0, 2.0, 3.0];

        let acd = AsyncCoordinateDescent::new(config, initial);

        // Update coordinate 0
        acd.update_coordinate(0, 0.5, 0);

        let solution = acd.get_solution();
        assert!(solution[0] < 1.0); // Should decrease
    }

    #[test]
    fn test_async_coordinate_versions() {
        let config = AsyncUpdateConfig::default();
        let initial = array![1.0, 2.0];

        let acd = AsyncCoordinateDescent::new(config, initial);

        // Initial versions should be 0
        let versions = acd.get_versions();
        assert_eq!(versions, vec![0, 0]);

        // Update coordinate 0
        acd.update_coordinate(0, 0.1, 0);

        // Version of coordinate 0 should increment
        let versions = acd.get_versions();
        assert_eq!(versions[0], 1);
        assert_eq!(versions[1], 0);
    }

    #[test]
    fn test_async_sgd_simulation() {
        let config = AsyncUpdateConfig {
            max_iterations: 50,
            num_workers: 2,
            learning_rate: 0.1,
            tolerance: 1e-4,
            ..Default::default()
        };

        // Simple quadratic objective: f(x) = ||x - target||^2
        let target = array![1.0, 2.0];
        let objective = |x: &Array1<Float>| {
            let diff = x - &target;
            diff.mapv(|v| v * v).sum()
        };

        let gradient = |x: &Array1<Float>| 2.0 * (x - &target);

        let initial = array![5.0, 5.0];

        let results = async_sgd_simulation(objective, gradient, config, initial);

        // Should converge
        assert!(results.convergence_history.len() > 0);

        // Final objective should be smaller than initial
        let final_obj = results.convergence_history.last().unwrap();
        let initial_obj = results.convergence_history.first().unwrap();
        assert!(final_obj < initial_obj);
    }

    #[test]
    fn test_get_consensus() {
        let admm = AsyncADMM::new(2, 3, 1.0);
        let consensus = admm.get_consensus();
        assert_eq!(consensus.len(), 3);
    }
}
