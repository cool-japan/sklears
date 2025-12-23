//! Performance Optimization for Multi-Output Learning
//!
//! This module provides optimized algorithms and utilities for improving computational
//! efficiency in multi-output learning scenarios.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

// ============================================================================
// Early Stopping Criteria
// ============================================================================

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Minimum improvement required to continue training
    pub min_delta: Float,
    /// Number of iterations with no improvement before stopping
    pub patience: usize,
    /// Metric to monitor ("loss" or "validation_score")
    pub monitor: String,
    /// Whether higher metric values are better
    pub mode_max: bool,
    /// Restore best weights when stopping
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            min_delta: 1e-4,
            patience: 10,
            monitor: "loss".to_string(),
            mode_max: false,
            restore_best_weights: true,
        }
    }
}

/// Early stopping tracker
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    config: EarlyStoppingConfig,
    best_value: Option<Float>,
    best_iteration: usize,
    wait_count: usize,
    should_stop: bool,
}

impl EarlyStopping {
    /// Create a new early stopping tracker
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            config,
            best_value: None,
            best_iteration: 0,
            wait_count: 0,
            should_stop: false,
        }
    }

    /// Update with new metric value
    pub fn update(&mut self, value: Float, iteration: usize) -> bool {
        match self.best_value {
            None => {
                self.best_value = Some(value);
                self.best_iteration = iteration;
                false
            }
            Some(best) => {
                let is_improvement = if self.config.mode_max {
                    value > best + self.config.min_delta
                } else {
                    value < best - self.config.min_delta
                };

                if is_improvement {
                    self.best_value = Some(value);
                    self.best_iteration = iteration;
                    self.wait_count = 0;
                    false
                } else {
                    self.wait_count += 1;
                    if self.wait_count >= self.config.patience {
                        self.should_stop = true;
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }

    /// Check if should stop
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Get best value
    pub fn best_value(&self) -> Option<Float> {
        self.best_value
    }

    /// Get best iteration
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }
}

// ============================================================================
// Warm Start Multi-Output Regressor
// ============================================================================

/// Configuration for warm start regressor
#[derive(Debug, Clone)]
pub struct WarmStartRegressorConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: Float,
    /// L2 regularization
    pub alpha: Float,
    /// Tolerance for convergence
    pub tol: Float,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Verbosity level
    pub verbose: bool,
}

impl Default for WarmStartRegressorConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            learning_rate: 0.01,
            alpha: 0.0001,
            tol: 1e-4,
            early_stopping: Some(EarlyStoppingConfig::default()),
            verbose: false,
        }
    }
}

/// Warm Start Multi-Output Regressor
///
/// Multi-output regressor with warm start capabilities for iterative optimization.
/// Supports resuming training from previous state and early stopping.
///
/// # Examples
///
/// ```rust
/// use sklears_multioutput::performance::{WarmStartRegressor, WarmStartRegressorConfig};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Predict};
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
///
/// let mut config = WarmStartRegressorConfig::default();
/// config.max_iter = 100;
///
/// let model = WarmStartRegressor::new().config(config);
/// let trained = model.fit(&X.view(), &y.view()).unwrap();
///
/// // Continue training with warm start
/// let continued = trained.continue_training(&X.view(), &y.view(), 50).unwrap();
///
/// let predictions = continued.predict(&X.view()).unwrap();
/// assert_eq!(predictions.dim(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct WarmStartRegressor<S = Untrained> {
    state: S,
    config: WarmStartRegressorConfig,
}

/// Trained state for Warm Start Regressor
#[derive(Debug, Clone)]
pub struct WarmStartRegressorTrained {
    /// Coefficient matrix
    pub coef: Array2<Float>,
    /// Intercept vector
    pub intercept: Array1<Float>,
    /// Number of features
    pub n_features: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Loss history
    pub loss_history: Vec<Float>,
    /// Best loss achieved
    pub best_loss: Float,
    /// Best iteration
    pub best_iter: usize,
    /// Best coefficients (if early stopping enabled)
    pub best_coef: Option<Array2<Float>>,
    /// Best intercept (if early stopping enabled)
    pub best_intercept: Option<Array1<Float>>,
    /// Whether converged
    pub converged: bool,
    /// Configuration
    pub config: WarmStartRegressorConfig,
}

impl WarmStartRegressor<Untrained> {
    /// Create a new warm start regressor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: WarmStartRegressorConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: WarmStartRegressorConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Enable early stopping
    pub fn early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.config.early_stopping = Some(config);
        self
    }
}

impl Default for WarmStartRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for WarmStartRegressor<Untrained> {
    type Fitted = WarmStartRegressor<WarmStartRegressorTrained>;

    fn fit(self, X: &ArrayView2<Float>, y: &ArrayView2<Float>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outputs = y.ncols();

        // Initialize coefficients
        let mut coef = Array2::zeros((n_features, n_outputs));
        let mut intercept = Array1::zeros(n_outputs);

        let mut loss_history = Vec::new();
        let mut best_loss = Float::INFINITY;
        let mut best_iter = 0;
        let mut best_coef = None;
        let mut best_intercept = None;

        let mut early_stopping = self
            .config
            .early_stopping
            .as_ref()
            .map(|cfg| EarlyStopping::new(cfg.clone()));

        let mut converged = false;

        // Gradient descent with early stopping
        for iter in 0..self.config.max_iter {
            let mut total_loss = 0.0;

            // Compute predictions and gradients
            for i in 0..n_samples {
                let x_i = X.row(i);
                let y_i = y.row(i);

                // Prediction
                let pred = coef.t().dot(&x_i) + &intercept;

                // Error
                let error = &y_i - &pred;
                total_loss += error.mapv(|x| x.powi(2)).sum();

                // Update coefficients
                for j in 0..n_features {
                    for k in 0..n_outputs {
                        let gradient = -error[k] * x_i[j] + self.config.alpha * coef[[j, k]];
                        coef[[j, k]] -= self.config.learning_rate * gradient;
                    }
                }

                // Update intercept
                for k in 0..n_outputs {
                    intercept[k] += self.config.learning_rate * error[k];
                }
            }

            // Average loss
            let avg_loss = total_loss / (n_samples as Float * n_outputs as Float);
            loss_history.push(avg_loss);

            // Track best model
            if avg_loss < best_loss {
                best_loss = avg_loss;
                best_iter = iter;
                if self.config.early_stopping.is_some() {
                    best_coef = Some(coef.clone());
                    best_intercept = Some(intercept.clone());
                }
            }

            // Check convergence
            if iter > 0 && (loss_history[iter - 1] - avg_loss).abs() < self.config.tol {
                converged = true;
                if self.config.verbose {
                    println!("Converged at iteration {}", iter);
                }
                break;
            }

            // Early stopping
            if let Some(ref mut es) = early_stopping {
                if es.update(avg_loss, iter) {
                    if self.config.verbose {
                        println!("Early stopping at iteration {}", iter);
                    }
                    break;
                }
            }

            if self.config.verbose && iter % 100 == 0 {
                println!("Iteration {}: loss = {:.6}", iter, avg_loss);
            }
        }

        // Restore best weights if early stopping is enabled
        if let Some(cfg) = &self.config.early_stopping {
            if cfg.restore_best_weights {
                if let Some(ref best_c) = best_coef {
                    coef = best_c.clone();
                }
                if let Some(ref best_i) = best_intercept {
                    intercept = best_i.clone();
                }
            }
        }

        Ok(WarmStartRegressor {
            state: WarmStartRegressorTrained {
                coef,
                intercept,
                n_features,
                n_outputs,
                n_iter: loss_history.len(),
                loss_history,
                best_loss,
                best_iter,
                best_coef,
                best_intercept,
                converged,
                config: self.config,
            },
            config: WarmStartRegressorConfig::default(),
        })
    }
}

impl WarmStartRegressor<WarmStartRegressorTrained> {
    /// Continue training from current state
    pub fn continue_training(
        mut self,
        X: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
        additional_iterations: usize,
    ) -> SklResult<Self> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if X.ncols() != self.state.n_features || y.ncols() != self.state.n_outputs {
            return Err(SklearsError::InvalidInput(
                "Feature or output dimensions do not match".to_string(),
            ));
        }

        let n_samples = X.nrows();

        let mut early_stopping = self
            .state
            .config
            .early_stopping
            .as_ref()
            .map(|cfg| EarlyStopping::new(cfg.clone()));

        // Continue from where we left off
        for iter in 0..additional_iterations {
            let mut total_loss = 0.0;

            // Gradient descent step
            for i in 0..n_samples {
                let x_i = X.row(i);
                let y_i = y.row(i);

                let pred = self.state.coef.t().dot(&x_i) + &self.state.intercept;
                let error = &y_i - &pred;
                total_loss += error.mapv(|x| x.powi(2)).sum();

                // Update coefficients
                for j in 0..self.state.n_features {
                    for k in 0..self.state.n_outputs {
                        let gradient =
                            -error[k] * x_i[j] + self.state.config.alpha * self.state.coef[[j, k]];
                        self.state.coef[[j, k]] -= self.state.config.learning_rate * gradient;
                    }
                }

                // Update intercept
                for k in 0..self.state.n_outputs {
                    self.state.intercept[k] += self.state.config.learning_rate * error[k];
                }
            }

            let avg_loss = total_loss / (n_samples as Float * self.state.n_outputs as Float);
            self.state.loss_history.push(avg_loss);

            // Update best
            if avg_loss < self.state.best_loss {
                self.state.best_loss = avg_loss;
                self.state.best_iter = self.state.n_iter + iter;
                if self.state.config.early_stopping.is_some() {
                    self.state.best_coef = Some(self.state.coef.clone());
                    self.state.best_intercept = Some(self.state.intercept.clone());
                }
            }

            // Check convergence
            let loss_len = self.state.loss_history.len();
            if loss_len > 1 {
                let prev_loss = self.state.loss_history[loss_len - 2];
                if (prev_loss - avg_loss).abs() < self.state.config.tol {
                    self.state.converged = true;
                    break;
                }
            }

            // Early stopping
            if let Some(ref mut es) = early_stopping {
                if es.update(avg_loss, self.state.n_iter + iter) {
                    break;
                }
            }
        }

        self.state.n_iter += additional_iterations;
        Ok(self)
    }

    /// Get training history
    pub fn loss_history(&self) -> &[Float] {
        &self.state.loss_history
    }

    /// Get best loss
    pub fn best_loss(&self) -> Float {
        self.state.best_loss
    }

    /// Check if converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get coefficients
    pub fn coef(&self) -> &Array2<Float> {
        &self.state.coef
    }

    /// Get number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>>
    for WarmStartRegressor<WarmStartRegressorTrained>
{
    fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                X.ncols()
            )));
        }

        let n_samples = X.nrows();
        let mut predictions = Array2::zeros((n_samples, self.state.n_outputs));

        for i in 0..n_samples {
            let x_i = X.row(i);
            let pred = self.state.coef.t().dot(&x_i) + &self.state.intercept;
            predictions.row_mut(i).assign(&pred);
        }

        Ok(predictions)
    }
}

impl Estimator for WarmStartRegressor<Untrained> {
    type Config = WarmStartRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for WarmStartRegressor<WarmStartRegressorTrained> {
    type Config = WarmStartRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}

// ============================================================================
// Fast Prediction Cache
// ============================================================================

/// Prediction cache for fast repeated predictions
#[derive(Debug, Clone)]
pub struct PredictionCache {
    /// Cached predictions keyed by input hash
    cache: HashMap<u64, Array2<Float>>,
    /// Maximum cache size
    max_size: usize,
    /// Number of cache hits
    hits: usize,
    /// Number of cache misses
    misses: usize,
}

impl PredictionCache {
    /// Create a new prediction cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Get cached prediction
    pub fn get(&mut self, X: &ArrayView2<Float>) -> Option<Array2<Float>> {
        let hash = self.hash_input(X);
        if let Some(pred) = self.cache.get(&hash) {
            self.hits += 1;
            Some(pred.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store prediction in cache
    pub fn put(&mut self, X: &ArrayView2<Float>, prediction: Array2<Float>) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(first_key) = self.cache.keys().next().copied() {
                self.cache.remove(&first_key);
            }
        }
        let hash = self.hash_input(X);
        self.cache.insert(hash, prediction);
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize, Float) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as Float / total as Float
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Simple hash function for input
    fn hash_input(&self, X: &ArrayView2<Float>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &val in X.iter() {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_early_stopping_basic() {
        let config = EarlyStoppingConfig {
            min_delta: 0.1,
            patience: 3,
            mode_max: false,
            ..Default::default()
        };

        let mut es = EarlyStopping::new(config);

        assert!(!es.update(1.0, 0));
        assert!(!es.update(0.8, 1)); // Improvement (1.0 - 0.8 = 0.2 > min_delta)
        assert!(!es.update(0.79, 2)); // No improvement #1 (0.8 - 0.79 = 0.01 < min_delta)
        assert!(!es.update(0.78, 3)); // No improvement #2
        assert!(es.update(0.77, 4)); // No improvement #3, should stop after patience (3)
    }

    #[test]
    fn test_early_stopping_mode_max() {
        let config = EarlyStoppingConfig {
            min_delta: 0.01,
            patience: 2,
            mode_max: true,
            ..Default::default()
        };

        let mut es = EarlyStopping::new(config);

        assert!(!es.update(0.5, 0));
        assert!(!es.update(0.6, 1)); // Improvement
        assert!(!es.update(0.59, 2)); // No improvement
        assert!(es.update(0.58, 3)); // Should stop
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_warm_start_regressor_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let model = WarmStartRegressor::new().max_iter(100).learning_rate(0.1);

        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (3, 2));
        assert!(trained.n_iter() > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_warm_start_continue_training() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0]];

        let model = WarmStartRegressor::new().max_iter(10).learning_rate(0.1);

        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let initial_iter = trained.n_iter();
        let initial_loss = trained.loss_history().last().copied().unwrap();

        // Continue training
        let continued = trained.continue_training(&X.view(), &y.view(), 20).unwrap();
        let final_loss = continued.loss_history().last().copied().unwrap();

        assert!(continued.n_iter() > initial_iter);
        // Loss should generally decrease (or stay similar)
        assert!(final_loss <= initial_loss + 1.0); // Allow some tolerance
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_warm_start_with_early_stopping() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let es_config = EarlyStoppingConfig {
            patience: 5,
            min_delta: 1e-6,
            ..Default::default()
        };

        let model = WarmStartRegressor::new()
            .max_iter(1000)
            .early_stopping(es_config)
            .learning_rate(0.1);

        let trained = model.fit(&X.view(), &y.view()).unwrap();

        // Should stop early due to convergence
        assert!(trained.n_iter() < 1000);
        assert!(trained.best_loss() < Float::INFINITY);
    }

    #[test]
    fn test_prediction_cache_basic() {
        let mut cache = PredictionCache::new(10);

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let pred = array![[1.0, 2.0], [2.0, 3.0]];

        // Cache miss
        assert!(cache.get(&X.view()).is_none());

        // Store and retrieve
        cache.put(&X.view(), pred.clone());
        let cached = cache.get(&X.view()).unwrap();

        assert_eq!(cached.dim(), pred.dim());
        assert_eq!(cache.stats().0, 1); // 1 hit
        assert_eq!(cache.stats().1, 1); // 1 miss
    }

    #[test]
    fn test_prediction_cache_eviction() {
        let mut cache = PredictionCache::new(2);

        let X1 = array![[1.0, 2.0]];
        let X2 = array![[2.0, 3.0]];
        let X3 = array![[3.0, 4.0]];
        let pred = array![[1.0, 2.0]];

        cache.put(&X1.view(), pred.clone());
        cache.put(&X2.view(), pred.clone());
        cache.put(&X3.view(), pred.clone()); // Should evict oldest

        assert_eq!(cache.cache.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = PredictionCache::new(10);

        let X = array![[1.0, 2.0]];
        let pred = array![[1.0, 2.0]];

        cache.get(&X.view()); // miss
        cache.put(&X.view(), pred);
        cache.get(&X.view()); // hit
        cache.get(&X.view()); // hit

        let (hits, misses, hit_rate) = cache.stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
        assert_abs_diff_eq!(hit_rate, 2.0 / 3.0, epsilon = 1e-6);
    }
}
