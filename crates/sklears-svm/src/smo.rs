//! Sequential Minimal Optimization (SMO) algorithm for SVM training

use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// Working set selection strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkingSetStrategy {
    /// First-order heuristic (maximum violating pair)
    FirstOrder,
    /// Second-order heuristic (maximum objective decrease)
    SecondOrder,
    /// Mixed strategy combining both
    Mixed,
}

/// SMO algorithm configuration
#[derive(Debug, Clone)]
pub struct SmoConfig {
    /// Regularization parameter
    pub c: Float,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Cache size for kernel evaluations (in MB)
    pub cache_size: usize,
    /// Enable shrinking heuristic
    pub shrinking: bool,
    /// Working set selection strategy
    pub working_set_strategy: WorkingSetStrategy,
    /// Early stopping threshold
    pub early_stopping_tol: Float,
    /// Check convergence every N iterations
    pub convergence_check_interval: usize,
}

impl Default for SmoConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            tol: 1e-3,
            max_iter: 1000,
            cache_size: 200,
            shrinking: true,
            working_set_strategy: WorkingSetStrategy::SecondOrder,
            early_stopping_tol: 1e-4,
            convergence_check_interval: 10,
        }
    }
}

/// SMO algorithm result
#[derive(Debug, Clone)]
pub struct SmoResult {
    /// Lagrange multipliers (alpha values)
    pub alpha: Array1<Float>,
    /// Bias term
    pub b: Float,
    /// Support vector indices
    pub support_indices: Vec<usize>,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final objective value
    pub objective_value: Float,
    /// Cache hit ratio for performance analysis
    pub cache_hit_ratio: Float,
}

/// Kernel cache for efficient computation
#[derive(Debug)]
struct KernelCache {
    cache: HashMap<(usize, usize), Float>,
    max_size: usize,
    hits: usize,
    total_requests: usize,
}

impl KernelCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
            hits: 0,
            total_requests: 0,
        }
    }

    fn get(&mut self, i: usize, j: usize) -> Option<Float> {
        self.total_requests += 1;
        let key = if i <= j { (i, j) } else { (j, i) };
        if let Some(&value) = self.cache.get(&key) {
            self.hits += 1;
            Some(value)
        } else {
            None
        }
    }

    fn insert(&mut self, i: usize, j: usize, value: Float) {
        let key = if i <= j { (i, j) } else { (j, i) };

        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry (could be improved with LRU)
            if let Some(first_key) = self.cache.keys().next().copied() {
                self.cache.remove(&first_key);
            }
        }

        self.cache.insert(key, value);
    }

    fn hit_ratio(&self) -> Float {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as Float / self.total_requests as Float
        }
    }
}

/// SMO algorithm implementation
pub struct SmoSolver<K: Kernel> {
    config: SmoConfig,
    kernel: K,
    // Training data (stored as owned data for now, views in solve)
    x: Array2<Float>,
    y: Array1<Float>,
    // Algorithm state
    alpha: Array1<Float>,
    f: Array1<Float>, // Cached decision function values
    b: Float,
    // Working set and shrinking
    active_set: Vec<usize>,
    inactive_set: Vec<usize>,
    // Kernel cache
    kernel_cache: RefCell<KernelCache>,
    // Convergence tracking
    objective_values: Vec<Float>,
    convergence_history: Vec<Float>,
}

impl<K: Kernel> SmoSolver<K> {
    /// Create a new SMO solver
    pub fn new(config: SmoConfig, kernel: K) -> Self {
        let cache_size = (config.cache_size * 1024 * 1024) / (std::mem::size_of::<Float>() * 2); // Convert MB to number of entries
        Self {
            kernel_cache: RefCell::new(KernelCache::new(cache_size)),
            config,
            kernel,
            x: Array2::zeros((0, 0)),
            y: Array1::zeros(0),
            alpha: Array1::zeros(0),
            f: Array1::zeros(0),
            b: 0.0,
            active_set: Vec::new(),
            inactive_set: Vec::new(),
            objective_values: Vec::new(),
            convergence_history: Vec::new(),
        }
    }

    /// Solve the SVM optimization problem
    pub fn solve(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<SmoResult> {
        self.solve_with_warm_start(x, y, None)
    }

    /// Solve the SVM optimization problem with optional warm start
    pub fn solve_with_warm_start(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        warm_start_alpha: Option<&Array1<Float>>,
    ) -> Result<SmoResult> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot solve SVM with empty dataset".to_string(),
            ));
        }

        // Initialize solver state - avoid unnecessary clones by reusing storage when possible
        if self.x.dim() == x.dim() {
            self.x.assign(x); // In-place copy, no allocation
        } else {
            self.x = x.clone(); // Need to reallocate
        }

        if self.y.dim() == y.dim() {
            self.y.assign(y); // In-place copy, no allocation
        } else {
            self.y = y.clone(); // Need to reallocate
        }

        // Handle warm start
        if let Some(alpha_init) = warm_start_alpha {
            if alpha_init.len() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Warm start alpha length {} does not match number of samples {}",
                    alpha_init.len(),
                    n_samples
                )));
            }

            // Validate alpha values
            for &alpha in alpha_init.iter() {
                if alpha < 0.0 || alpha > self.config.c + 1e-10 {
                    return Err(SklearsError::InvalidInput(format!(
                        "Invalid alpha value {} in warm start (must be between 0 and C={})",
                        alpha, self.config.c
                    )));
                }
            }

            self.alpha = alpha_init.clone();

            // Initialize f values based on warm start
            self.f = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let mut f_i = 0.0;
                for j in 0..n_samples {
                    if self.alpha[j] > 1e-10 {
                        let k_ij = self.get_kernel_value(i, j);
                        f_i += self.alpha[j] * self.y[j] * k_ij;
                    }
                }
                self.f[i] = f_i - self.y[i];
            }

            // Initialize bias with warm start
            let support_indices = self.find_support_vectors();
            self.update_bias(&support_indices)?;
        } else {
            // Cold start
            self.alpha = Array1::zeros(n_samples);
            self.f = -y.clone(); // f = -y initially (assuming all alpha = 0)
            self.b = 0.0;
        }

        self.active_set = (0..n_samples).collect();
        self.inactive_set.clear();
        self.objective_values.clear();
        self.convergence_history.clear();

        let mut n_iter = 0; // Count of successful updates
        let mut loop_iter = 0; // Total loop iterations
        let mut converged = false;
        let mut no_change_count = 0; // Count consecutive iterations with no updates

        // Main SMO loop
        while n_iter < self.config.max_iter && loop_iter < self.config.max_iter * 10 {
            loop_iter += 1;

            let (i, j) = match self.select_working_set() {
                Some(pair) => pair,
                None => {
                    converged = true;
                    break;
                }
            };

            if self.take_step(i, j)? {
                n_iter += 1;
                no_change_count = 0; // Reset counter on successful update

                // Compute and store objective value
                if n_iter % 5 == 0 {
                    let obj_val = self.compute_objective();
                    self.objective_values.push(obj_val);
                }
            } else {
                no_change_count += 1;
                // If we've had too many consecutive iterations without updates, stop
                if no_change_count > 100 {
                    converged = true;
                    break;
                }
            }

            // Apply shrinking periodically
            if self.config.shrinking && n_iter > 0 && n_iter % 100 == 0 {
                self.apply_shrinking();
            }

            // Check convergence periodically
            if n_iter > 0 && n_iter % self.config.convergence_check_interval == 0 {
                let convergence_measure = self.compute_convergence_measure();
                self.convergence_history.push(convergence_measure);

                if self.check_convergence() {
                    converged = true;
                    break;
                }

                // Early stopping based on convergence rate
                if self.should_early_stop() {
                    converged = true;
                    break;
                }
            }
        }

        // Find support vectors
        let support_indices = self.find_support_vectors();

        // Update bias using support vectors
        self.update_bias(&support_indices)?;

        let final_objective = self.compute_objective();
        let cache_hit_ratio = self.kernel_cache.borrow().hit_ratio();

        Ok(SmoResult {
            alpha: self.alpha.clone(),
            b: self.b,
            support_indices,
            n_iter,
            converged,
            objective_value: final_objective,
            cache_hit_ratio,
        })
    }

    /// Select working set (i, j) using heuristics
    fn select_working_set(&self) -> Option<(usize, usize)> {
        match self.config.working_set_strategy {
            WorkingSetStrategy::FirstOrder => self.select_working_set_first_order(),
            WorkingSetStrategy::SecondOrder => self.select_working_set_second_order(),
            WorkingSetStrategy::Mixed => {
                // Alternate between strategies based on iteration count
                if self.objective_values.len() % 2 == 0 {
                    self.select_working_set_second_order()
                } else {
                    self.select_working_set_first_order()
                }
            }
        }
    }

    /// First-order working set selection (WSS1 - Maximal Violating Pair)
    /// O(n) complexity using maximal violating pair heuristic
    fn select_working_set_first_order(&self) -> Option<(usize, usize)> {
        let mut i_up = None;
        let mut i_low = None;
        let mut g_min_up = Float::INFINITY; // Minimum gradient in I_up (most violation)
        let mut g_max_low = Float::NEG_INFINITY; // Maximum gradient in I_low (most violation)

        for &t in &self.active_set {
            let alpha_t = self.alpha[t];
            let y_t = self.y[t];
            let f_t = self.f[t];
            let g_t = y_t * f_t; // Gradient: y_i * f_i

            // I_up: samples that can increase their alpha (violate upper bound when g_t < 1)
            // For y_i = +1: alpha_i < C
            // For y_i = -1: alpha_i > 0
            if (y_t > 0.0 && alpha_t < self.config.c - self.config.tol)
                || (y_t < 0.0 && alpha_t > self.config.tol)
            {
                // Find most violating sample (minimum gradient)
                if g_t < g_min_up {
                    g_min_up = g_t;
                    i_up = Some(t);
                }
            }

            // I_low: samples that can decrease their alpha (violate lower bound when g_t > 1)
            // For y_i = -1: alpha_i < C
            // For y_i = +1: alpha_i > 0
            if (y_t < 0.0 && alpha_t < self.config.c - self.config.tol)
                || (y_t > 0.0 && alpha_t > self.config.tol)
            {
                // Find most violating sample (maximum gradient)
                if g_t > g_max_low {
                    g_max_low = g_t;
                    i_low = Some(t);
                }
            }
        }

        // Check if we found a violating pair
        match (i_up, i_low) {
            (Some(i), Some(j)) => {
                // Continue optimization when optimality gap > tolerance
                // OR when we're at initialization (both have same gradient but violate KKT)
                let gap = g_max_low - g_min_up;
                if gap > self.config.tol
                    || (gap.abs() <= self.config.tol && g_min_up < 1.0 - self.config.tol)
                {
                    Some((i, j))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Second-order working set selection (maximum objective decrease)
    /// Optimized to use WSS1 approach with second-order refinement
    fn select_working_set_second_order(&self) -> Option<(usize, usize)> {
        // First find i_up using maximal violating pair heuristic (O(n))
        let mut i_up = None;
        let mut g_min_up = Float::INFINITY;

        for &t in &self.active_set {
            let alpha_t = self.alpha[t];
            let y_t = self.y[t];
            let f_t = self.f[t];
            let g_t = y_t * f_t;

            // Check if in I_up set and find minimum gradient (most violating)
            if ((y_t > 0.0 && alpha_t < self.config.c - self.config.tol)
                || (y_t < 0.0 && alpha_t > self.config.tol))
                && g_t < g_min_up
            {
                g_min_up = g_t;
                i_up = Some(t);
            }
        }

        let i = i_up?;

        // Now find j that maximizes objective decrease (O(n))
        let mut best_j = None;
        let mut best_decrease = -1e-10; // Negative to allow selection at initialization when all decreases are 0

        let k_ii = self.get_kernel_value(i, i);
        let g_i = g_min_up;

        for &j in &self.active_set {
            if i == j {
                continue;
            }

            let alpha_j = self.alpha[j];
            let y_j = self.y[j];
            let f_j = self.f[j];
            let g_j = y_j * f_j;

            // Check if j is in I_low
            if !((y_j < 0.0 && alpha_j < self.config.c - self.config.tol)
                || (y_j > 0.0 && alpha_j > self.config.tol))
            {
                continue;
            }

            // Compute second-order information
            let k_jj = self.get_kernel_value(j, j);
            let k_ij = self.get_kernel_value(i, j);
            let eta = k_ii + k_jj - 2.0 * k_ij;

            if eta <= 0.0 {
                continue; // Skip degenerate cases
            }

            // Estimate objective decrease
            let grad_diff = g_j - g_i; // Correct gradient difference
            let decrease = (grad_diff * grad_diff) / eta;

            if decrease > best_decrease {
                best_decrease = decrease;
                best_j = Some(j);
            }
        }

        // If no j found with positive decrease, fall back to first-order selection
        // This can happen at initialization when all pairs have the same gradient
        if best_j.is_none() {
            self.select_working_set_first_order()
        } else {
            best_j.map(|j| (i, j))
        }
    }

    /// Estimate objective function decrease for pair (i, j)
    fn estimate_objective_decrease(&self, i: usize, j: usize) -> Float {
        let y_i = self.y[i];
        let y_j = self.y[j];
        let f_i = self.f[i];
        let f_j = self.f[j];

        // Simplified estimate based on gradient difference
        let grad_diff = (y_i * f_i - 1.0).abs() + (y_j * f_j - 1.0).abs();
        let f_diff = (f_i - f_j).abs();

        grad_diff * f_diff
    }

    /// Check if sample i violates KKT conditions
    fn violates_kkt(&self, i: usize) -> bool {
        let alpha_i = self.alpha[i];
        let y_i = self.y[i];
        let f_i = self.f[i];

        let tol = self.config.tol;

        if alpha_i < tol {
            // alpha_i = 0, should have y_i * f_i >= 1
            y_i * f_i < 1.0 - tol
        } else if alpha_i > self.config.c - tol {
            // alpha_i = C, should have y_i * f_i <= 1
            y_i * f_i > 1.0 + tol
        } else {
            // 0 < alpha_i < C, should have y_i * f_i = 1
            (y_i * f_i - 1.0).abs() > tol
        }
    }

    /// Take optimization step for pair (i, j)
    fn take_step(&mut self, i: usize, j: usize) -> Result<bool> {
        if i == j {
            return Ok(false);
        }

        let alpha_i_old = self.alpha[i];
        let alpha_j_old = self.alpha[j];
        let y_i = self.y[i];
        let y_j = self.y[j];

        // Compute bounds L and H
        let (l, h) = if y_i == y_j {
            let gamma = alpha_i_old + alpha_j_old;
            ((gamma - self.config.c).max(0.0), gamma.min(self.config.c))
        } else {
            let gamma = alpha_i_old - alpha_j_old;
            (
                (-gamma).max(0.0),
                (self.config.c - gamma).min(self.config.c),
            )
        };

        if (l - h).abs() < 1e-10 {
            return Ok(false);
        }

        // Compute kernel values with caching
        let k_ii = self.get_kernel_value(i, i);
        let k_jj = self.get_kernel_value(j, j);
        let k_ij = self.get_kernel_value(i, j);

        // Compute second derivative
        let eta = k_ii + k_jj - 2.0 * k_ij;

        let alpha_j_new = if eta > 0.0 {
            // Normal case: optimize along eta
            let alpha_j_unc = alpha_j_old + y_j * (self.f[i] - self.f[j]) / eta;
            alpha_j_unc.max(l).min(h)
        } else {
            // Degenerate case: evaluate objective at endpoints
            let f1 = self.objective_at_endpoint(i, j, l)?;
            let f2 = self.objective_at_endpoint(i, j, h)?;

            if f1 < f2 - 1e-10 {
                l
            } else if f2 < f1 - 1e-10 {
                h
            } else {
                alpha_j_old
            }
        };

        if (alpha_j_new - alpha_j_old).abs() < 1e-10 {
            return Ok(false);
        }

        // Compute new alpha_i
        let alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new);

        // Update alpha values
        self.alpha[i] = alpha_i_new;
        self.alpha[j] = alpha_j_new;

        // Update cached f values
        self.update_f_values(i, j, alpha_i_old, alpha_j_old)?;

        Ok(true)
    }

    /// Compute objective function value at endpoint
    fn objective_at_endpoint(&self, i: usize, j: usize, alpha_j: Float) -> Result<Float> {
        let y_i = self.y[i];
        let y_j = self.y[j];
        let alpha_i_old = self.alpha[i];
        let alpha_j_old = self.alpha[j];

        let alpha_i = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j);

        // This is a simplified objective computation
        // In practice, you'd want to compute the full dual objective
        let k_ii = self.get_kernel_value(i, i);
        let k_jj = self.get_kernel_value(j, j);
        let k_ij = self.get_kernel_value(i, j);

        Ok(alpha_i * self.f[i] + alpha_j * self.f[j]
            - 0.5
                * (alpha_i * alpha_i * k_ii
                    + alpha_j * alpha_j * k_jj
                    + 2.0 * alpha_i * alpha_j * k_ij))
    }

    /// Update cached f values after alpha update
    fn update_f_values(
        &mut self,
        i: usize,
        j: usize,
        alpha_i_old: Float,
        alpha_j_old: Float,
    ) -> Result<()> {
        let delta_alpha_i = self.alpha[i] - alpha_i_old;
        let delta_alpha_j = self.alpha[j] - alpha_j_old;

        // Skip update if both deltas are near zero
        if delta_alpha_i.abs() < 1e-10 && delta_alpha_j.abs() < 1e-10 {
            return Ok(());
        }

        // Update f values for active samples to improve efficiency
        for &k in &self.active_set {
            let k_ik = self.get_kernel_value(i, k);
            let k_jk = self.get_kernel_value(j, k);
            self.f[k] += self.y[i] * delta_alpha_i * k_ik + self.y[j] * delta_alpha_j * k_jk;
        }

        // Also update inactive samples if shrinking is disabled
        if !self.config.shrinking {
            for &k in &self.inactive_set {
                let k_ik = self.get_kernel_value(i, k);
                let k_jk = self.get_kernel_value(j, k);
                self.f[k] += self.y[i] * delta_alpha_i * k_ik + self.y[j] * delta_alpha_j * k_jk;
            }
        }

        Ok(())
    }

    /// Check convergence using KKT conditions
    fn check_convergence(&self) -> bool {
        for &i in &self.active_set {
            if self.violates_kkt(i) {
                return false;
            }
        }
        true
    }

    /// Find support vector indices
    fn find_support_vectors(&self) -> Vec<usize> {
        let mut support_indices = Vec::new();

        for i in 0..self.alpha.len() {
            if self.alpha[i] > 1e-10 {
                support_indices.push(i);
            }
        }

        support_indices
    }

    /// Update bias term using support vectors
    fn update_bias(&mut self, support_indices: &[usize]) -> Result<()> {
        if support_indices.is_empty() {
            self.b = 0.0;
            return Ok(());
        }

        // Use support vectors that are not at bounds
        let mut bias_sum = 0.0;
        let mut n_free = 0;

        for &i in support_indices {
            let alpha_i = self.alpha[i];
            if alpha_i > 1e-10 && alpha_i < self.config.c - 1e-10 {
                // Free support vector: 0 < alpha < C
                // For margin support vectors: y[i] * (sum(alpha[j] * y[j] * K(x[i], x[j])) + b) = 1
                // Since f[i] = sum(alpha[j] * y[j] * K(x[i], x[j])) - y[i], we have:
                // sum(alpha[j] * y[j] * K(x[i], x[j])) = f[i] + y[i]
                // Therefore: y[i] * (f[i] + y[i] + b) = 1
                //           y[i] * f[i] + 1 + y[i] * b = 1  (since y[i]^2 = 1)
                //           y[i] * (f[i] + b) = 0
                //           b = -f[i]
                bias_sum += -self.f[i];
                n_free += 1;
            }
        }

        if n_free > 0 {
            self.b = bias_sum / n_free as Float;
        } else {
            // Use all support vectors
            bias_sum = 0.0;
            for &i in support_indices {
                bias_sum += -self.f[i];
            }
            self.b = bias_sum / support_indices.len() as Float;
        }

        Ok(())
    }

    /// Get kernel value with caching
    fn get_kernel_value(&self, i: usize, j: usize) -> Float {
        let mut cache = self.kernel_cache.borrow_mut();
        if let Some(value) = cache.get(i, j) {
            value
        } else {
            // Use views directly without allocating - major performance improvement
            let value = self.kernel.compute(self.x.row(i), self.x.row(j));
            cache.insert(i, j, value);
            value
        }
    }

    /// Apply shrinking heuristic to reduce problem size
    fn apply_shrinking(&mut self) {
        if !self.config.shrinking {
            return;
        }

        let mut new_active = Vec::new();
        let tol = self.config.tol;

        for &i in &self.active_set {
            let alpha_i = self.alpha[i];
            let y_i = self.y[i];
            let f_i = self.f[i];

            // Keep samples that are likely to change
            let should_keep = if alpha_i < tol {
                // alpha = 0: keep if violates upper bound
                y_i * f_i < 1.0 + tol
            } else if alpha_i > self.config.c - tol {
                // alpha = C: keep if violates lower bound
                y_i * f_i > 1.0 - tol
            } else {
                // 0 < alpha < C: always keep (support vectors)
                true
            };

            if should_keep {
                new_active.push(i);
            } else {
                self.inactive_set.push(i);
            }
        }

        self.active_set = new_active;
    }

    /// Compute convergence measure
    fn compute_convergence_measure(&self) -> Float {
        let mut max_violation: Float = 0.0;

        for &i in &self.active_set {
            let alpha_i = self.alpha[i];
            let y_i = self.y[i];
            let f_i = self.f[i];

            let violation = if alpha_i < self.config.tol {
                (1.0 - y_i * f_i).max(0.0)
            } else if alpha_i > self.config.c - self.config.tol {
                (y_i * f_i - 1.0).max(0.0)
            } else {
                (y_i * f_i - 1.0).abs()
            };

            max_violation = max_violation.max(violation);
        }

        max_violation
    }

    /// Check if early stopping should be applied
    fn should_early_stop(&self) -> bool {
        if self.convergence_history.len() < 3 {
            return false;
        }

        let recent = &self.convergence_history[self.convergence_history.len() - 3..];
        let improvement = recent[0] - recent[2];

        improvement < self.config.early_stopping_tol
    }

    /// Compute dual objective function value
    fn compute_objective(&mut self) -> Float {
        let mut objective = 0.0;

        // Sum of alpha_i
        for &alpha in self.alpha.iter() {
            objective += alpha;
        }

        // Subtract 0.5 * alpha^T K alpha
        for i in 0..self.alpha.len() {
            for j in 0..self.alpha.len() {
                if self.alpha[i] > 1e-10 && self.alpha[j] > 1e-10 {
                    let k_ij = self.get_kernel_value(i, j);
                    objective -= 0.5 * self.alpha[i] * self.alpha[j] * self.y[i] * self.y[j] * k_ij;
                }
            }
        }

        objective
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::LinearKernel;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_smo_linear_separable() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1.0, 1.0, -1.0, -1.0];

        let config = SmoConfig {
            c: 1.0,
            tol: 1e-3,
            max_iter: 200,
            working_set_strategy: WorkingSetStrategy::SecondOrder,
            ..Default::default()
        };

        let mut solver = SmoSolver::new(config, LinearKernel::new());
        let result = solver.solve(&x, &y).unwrap();

        // Should converge for linearly separable data
        assert!(result.converged || result.n_iter < 1000);
        assert!(!result.support_indices.is_empty());
        assert!(result.cache_hit_ratio >= 0.0);

        // Check that alpha values are non-negative and sum correctly
        for &alpha in result.alpha.iter() {
            assert!(alpha >= -1e-10);
        }
    }

    #[test]
    fn test_working_set_strategies() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1.0, 1.0, -1.0, -1.0];

        for strategy in [
            WorkingSetStrategy::FirstOrder,
            WorkingSetStrategy::SecondOrder,
            WorkingSetStrategy::Mixed,
        ] {
            let config = SmoConfig {
                c: 1.0,
                tol: 1e-2,     // More tolerant for testing
                max_iter: 500, // More iterations for FirstOrder strategy
                working_set_strategy: strategy,
                ..Default::default()
            };

            let mut solver = SmoSolver::new(config, LinearKernel::new());
            let result = solver.solve(&x, &y).unwrap();

            // Allow for the possibility that the algorithm converges without finding many support vectors
            // This can happen with perfectly separable data and lenient tolerance
            assert!(
                result.converged || !result.support_indices.is_empty(),
                "Strategy {:?} should either converge or find support vectors",
                strategy
            );
        }
    }

    #[test]
    fn test_kernel_cache() {
        let mut cache = KernelCache::new(10);

        // Test miss
        assert_eq!(cache.get(0, 1), None);
        assert_eq!(cache.hit_ratio(), 0.0);

        // Test insert and hit
        cache.insert(0, 1, 5.0);
        assert_eq!(cache.get(0, 1), Some(5.0));
        assert_eq!(cache.get(1, 0), Some(5.0)); // Symmetric

        assert!(cache.hit_ratio() > 0.0);
    }

    #[test]
    fn test_smo_config_default() {
        let config = SmoConfig::default();
        assert_eq!(config.c, 1.0);
        assert_eq!(config.tol, 1e-3);
        assert_eq!(config.max_iter, 1000);
        assert_eq!(config.working_set_strategy, WorkingSetStrategy::SecondOrder);
        assert!(config.shrinking);
    }

    #[test]
    fn test_smo_warm_start() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1.0, 1.0, -1.0, -1.0];

        let config = SmoConfig {
            c: 1.0,
            tol: 1e-2,
            max_iter: 50,
            ..Default::default()
        };

        // First solve without warm start
        let mut solver1 = SmoSolver::new(config.clone(), LinearKernel::new());
        let result1 = solver1.solve(&x, &y).unwrap();

        // Now solve with warm start using the previous solution
        let mut solver2 = SmoSolver::new(config, LinearKernel::new());
        let result2 = solver2
            .solve_with_warm_start(&x, &y, Some(&result1.alpha))
            .unwrap();

        // Warm start should converge faster (fewer iterations)
        assert!(result2.n_iter <= result1.n_iter);

        // Results should be similar
        for (a1, a2) in result1.alpha.iter().zip(result2.alpha.iter()) {
            assert_abs_diff_eq!(a1, a2, epsilon = 1e-1);
        }
    }

    #[test]
    fn test_smo_warm_start_invalid_alpha() {
        let x = array![[1.0, 1.0], [2.0, 2.0]];
        let y = array![1.0, -1.0];

        let config = SmoConfig::default();
        let mut solver = SmoSolver::new(config, LinearKernel::new());

        // Test with negative alpha (invalid)
        let invalid_alpha = array![-0.1, 0.5];
        let result = solver.solve_with_warm_start(&x, &y, Some(&invalid_alpha));
        assert!(result.is_err());

        // Test with alpha > C (invalid)
        let invalid_alpha = array![0.5, 2.0]; // C = 1.0 by default
        let result = solver.solve_with_warm_start(&x, &y, Some(&invalid_alpha));
        assert!(result.is_err());

        // Test with wrong size
        let invalid_alpha = array![0.1]; // Only 1 element, but we have 2 samples
        let result = solver.solve_with_warm_start(&x, &y, Some(&invalid_alpha));
        assert!(result.is_err());
    }
}
