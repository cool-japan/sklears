//! Bayesian Optimization framework for hyperparameter tuning and global optimization

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Predict,
};

use crate::gpr::{GaussianProcessRegressor, GprTrained};

///
/// let kernel = RBF::new(1.0);
/// let gpr = GaussianProcessRegressor::new().kernel(Box::new(kernel));
/// let optimizer = BayesianOptimizer::new(gpr)
///     .acquisition(AcquisitionFunction::ExpectedImprovement)
///     .xi(0.01);
///
/// // Define bounds and initial points
/// let bounds = array![[0.0, 10.0]]; // 1D optimization between 0 and 10
/// let X_init = array![[1.0], [5.0], [9.0]];
/// let y_init = array![1.0, 25.0, 81.0]; // f(x) = x^2
///
/// let mut bo = optimizer.fit_initial(&X_init.view(), &y_init.view()).unwrap();
/// let next_point = bo.suggest_next_point(&bounds.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BayesianOptimizer {
    gp: Option<GaussianProcessRegressor<GprTrained>>,
    acquisition: AcquisitionFunction,
    xi: f64,
    n_restarts: usize,
    random_state: Option<u64>,
}

/// Available acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected Improvement - balances exploration and exploitation
    ExpectedImprovement,
    /// Probability of Improvement - conservative acquisition function
    ProbabilityOfImprovement,
    /// Upper Confidence Bound - optimistic acquisition function
    UpperConfidenceBound { beta: f64 },
    /// Entropy Search - information-theoretic acquisition
    EntropySearch,
}

/// Result of Bayesian optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best point found during optimization
    pub best_point: Array1<f64>,
    /// Best value found during optimization
    pub best_value: f64,
    /// All evaluated points
    pub all_points: Array2<f64>,
    /// All evaluated values
    pub all_values: Array1<f64>,
    /// Number of iterations completed
    pub n_iterations: usize,
}

impl BayesianOptimizer {
    /// Create a new BayesianOptimizer instance
    pub fn new(_gp: GaussianProcessRegressor) -> Self {
        Self {
            gp: None,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            xi: 0.01,
            n_restarts: 10,
            random_state: None,
        }
    }

    /// Set the acquisition function
    pub fn acquisition(mut self, acquisition: AcquisitionFunction) -> Self {
        self.acquisition = acquisition;
        self
    }

    /// Set the exploration parameter
    pub fn xi(mut self, xi: f64) -> Self {
        self.xi = xi;
        self
    }

    /// Set the number of random restarts for acquisition optimization
    pub fn n_restarts(mut self, n_restarts: usize) -> Self {
        self.n_restarts = n_restarts;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Create a builder for the Bayesian optimizer
    pub fn builder() -> BayesianOptimizerBuilder {
        BayesianOptimizerBuilder::new()
    }

    /// Fit the initial Gaussian Process model
    pub fn fit_initial(
        self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<BayesianOptimizerFitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Create a default GP if none provided
        let gp = match self.gp {
            Some(gp) => gp,
            None => {
                use crate::kernels::RBF;
                use sklears_core::traits::Fit;

                let kernel = RBF::new(1.0);
                let gpr = GaussianProcessRegressor::new().kernel(Box::new(kernel));
                gpr.fit(&X.to_owned(), &y.to_owned())?
            }
        };

        let current_best = y.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Ok(BayesianOptimizerFitted {
            gp,
            X_obs: X.to_owned(),
            y_obs: y.to_owned(),
            current_best,
            acquisition: self.acquisition,
            xi: self.xi,
            n_restarts: self.n_restarts,
            random_state: self.random_state,
        })
    }
}

/// Fitted Bayesian Optimizer that can suggest new points
#[derive(Debug, Clone)]
pub struct BayesianOptimizerFitted {
    gp: GaussianProcessRegressor<GprTrained>,
    X_obs: Array2<f64>,
    y_obs: Array1<f64>,
    current_best: f64,
    acquisition: AcquisitionFunction,
    xi: f64,
    n_restarts: usize,
    random_state: Option<u64>,
}

impl BayesianOptimizerFitted {
    /// Suggest the next point to evaluate
    pub fn suggest_next_point(&self, bounds: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        if bounds.nrows() != self.X_obs.ncols() {
            return Err(SklearsError::InvalidInput(
                "Bounds dimension must match feature dimension".to_string(),
            ));
        }

        let mut best_x = Array1::<f64>::zeros(bounds.nrows());
        let mut best_acq = f64::NEG_INFINITY;

        // Multi-restart optimization
        for restart in 0..self.n_restarts {
            // Random starting point within bounds
            let mut x_start = Array1::<f64>::zeros(bounds.nrows());
            let mut rng = self.random_state.unwrap_or(42) + restart as u64 * 1337;

            for i in 0..bounds.nrows() {
                let min_bound = bounds[[i, 0]];
                let max_bound = bounds[[i, 1]];
                // Simple pseudo-random number generation
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let random_val = (rng % 1000000) as f64 / 1000000.0;
                x_start[i] = min_bound + random_val * (max_bound - min_bound);
            }

            // Simple gradient-free optimization (grid search for simplicity)
            let optimized_x = self.optimize_acquisition_internal(&x_start, bounds)?;
            let acq_val = self.evaluate_acquisition(&optimized_x)?;

            if acq_val > best_acq {
                best_acq = acq_val;
                best_x = optimized_x;
            }
        }

        Ok(best_x)
    }

    /// Update the optimizer with a new observation
    pub fn update(
        mut self,
        x_new: &ArrayView1<f64>,
        y_new: f64,
    ) -> SklResult<BayesianOptimizerFitted> {
        // Add new observation
        let mut X_new = Array2::<f64>::zeros((self.X_obs.nrows() + 1, self.X_obs.ncols()));
        X_new
            .slice_mut(s![..self.X_obs.nrows(), ..])
            .assign(&self.X_obs);
        X_new.row_mut(self.X_obs.nrows()).assign(x_new);

        let mut y_new_vec = Array1::<f64>::zeros(self.y_obs.len() + 1);
        y_new_vec
            .slice_mut(s![..self.y_obs.len()])
            .assign(&self.y_obs);
        y_new_vec[self.y_obs.len()] = y_new;

        // Update current best
        self.current_best = self.current_best.max(y_new);

        // Refit GP
        use crate::kernels::RBF;
        use sklears_core::traits::Fit;

        let kernel = RBF::new(1.0);
        let gpr = GaussianProcessRegressor::new().kernel(Box::new(kernel));
        let new_gp = gpr.fit(&X_new, &y_new_vec)?;

        Ok(BayesianOptimizerFitted {
            gp: new_gp,
            X_obs: X_new,
            y_obs: y_new_vec,
            current_best: self.current_best,
            acquisition: self.acquisition,
            xi: self.xi,
            n_restarts: self.n_restarts,
            random_state: self.random_state,
        })
    }

    /// Get the current best observed value
    pub fn get_best_value(&self) -> f64 {
        self.current_best
    }

    /// Get the point that achieved the best value
    pub fn get_best_point(&self) -> SklResult<Array1<f64>> {
        let best_idx = self
            .y_obs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| SklearsError::InvalidInput("No observations available".to_string()))?;

        Ok(self.X_obs.row(best_idx).to_owned())
    }

    /// Optimize the acquisition function (simplified implementation)
    fn optimize_acquisition_internal(
        &self,
        x_start: &Array1<f64>,
        bounds: &ArrayView2<f64>,
    ) -> SklResult<Array1<f64>> {
        // Simple grid search for acquisition function optimization
        let n_grid = 20;
        let mut best_x = x_start.clone();
        let mut best_acq = self.evaluate_acquisition(&best_x)?;

        // Grid search over each dimension
        for dim in 0..bounds.nrows() {
            let min_bound = bounds[[dim, 0]];
            let max_bound = bounds[[dim, 1]];

            for i in 0..n_grid {
                let mut x_test = best_x.clone();
                x_test[dim] =
                    min_bound + (i as f64 / (n_grid - 1) as f64) * (max_bound - min_bound);

                let acq_val = self.evaluate_acquisition(&x_test)?;
                if acq_val > best_acq {
                    best_acq = acq_val;
                    best_x = x_test;
                }
            }
        }

        Ok(best_x)
    }

    /// Evaluate the acquisition function at a given point
    fn evaluate_acquisition(&self, x: &Array1<f64>) -> SklResult<f64> {
        let X_test = x.view().insert_axis(Axis(0));
        let (mean, std) = self.gp.predict_with_std(&X_test.to_owned())?;
        let mu = mean[0];
        let sigma = std[0];

        let acq_val = match self.acquisition {
            AcquisitionFunction::ExpectedImprovement => {
                if sigma == 0.0 {
                    0.0
                } else {
                    let z = (mu - self.current_best - self.xi) / sigma;
                    let phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
                    let pdf = (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
                    (mu - self.current_best - self.xi) * phi + sigma * pdf
                }
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                if sigma == 0.0 {
                    if mu > self.current_best + self.xi {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let z = (mu - self.current_best - self.xi) / sigma;
                    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
                }
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mu + beta * sigma,
            AcquisitionFunction::EntropySearch => {
                // Simplified entropy search (just use uncertainty)
                sigma
            }
        };

        Ok(acq_val)
    }
}

impl BayesianOptimizerFitted {
    /// Predict at given points
    pub fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        self.gp.predict(X)
    }

    /// Predict variance at given points
    pub fn predict_variance(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (_mean, std) = self.gp.predict_with_std(X)?;
        // Variance is the square of standard deviation
        Ok(std.mapv(|s| s * s))
    }

    /// Compute acquisition value at a given point
    pub fn acquisition_value(&self, x: &Array1<f64>, current_best: f64) -> SklResult<f64> {
        let X_test = x.view().insert_axis(Axis(0));
        let (mean, std) = self.gp.predict_with_std(&X_test.to_owned())?;
        let mu = mean[0];
        let sigma = std[0];

        let acq_val = match self.acquisition {
            AcquisitionFunction::ExpectedImprovement => {
                if sigma == 0.0 {
                    0.0
                } else {
                    let z = (mu - current_best - self.xi) / sigma;
                    let phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
                    let pdf = (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
                    (mu - current_best - self.xi) * phi + sigma * pdf
                }
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                if sigma == 0.0 {
                    if mu > current_best + self.xi {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let z = (mu - current_best - self.xi) / sigma;
                    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
                }
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mu + beta * sigma,
            AcquisitionFunction::EntropySearch => {
                // Simplified entropy search (just use uncertainty)
                sigma
            }
        };

        Ok(acq_val)
    }

    /// Optimize acquisition function to find next point
    pub fn optimize_acquisition(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        n_restarts: usize,
    ) -> SklResult<Array1<f64>> {
        let mut best_x = Array1::<f64>::zeros(bounds.nrows());
        let mut best_acq = f64::NEG_INFINITY;

        // Multi-restart optimization
        for restart in 0..n_restarts {
            // Random starting point within bounds
            let mut x_start = Array1::<f64>::zeros(bounds.nrows());
            let mut rng = self.random_state.unwrap_or(42) + restart as u64 * 1337;

            for i in 0..bounds.nrows() {
                let min_bound = bounds[[i, 0]];
                let max_bound = bounds[[i, 1]];
                // Simple pseudo-random number generation
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let random_val = (rng % 1000000) as f64 / 1000000.0;
                x_start[i] = min_bound + random_val * (max_bound - min_bound);
            }

            // Simple gradient-free optimization (grid search for simplicity)
            let optimized_x =
                self.optimize_acquisition_from_start(&x_start, bounds, current_best)?;
            let acq_val = self.acquisition_value(&optimized_x, current_best)?;

            if acq_val > best_acq {
                best_acq = acq_val;
                best_x = optimized_x;
            }
        }

        Ok(best_x)
    }

    /// Optimize from a starting point
    fn optimize_acquisition_from_start(
        &self,
        x_start: &Array1<f64>,
        bounds: &Array2<f64>,
        current_best: f64,
    ) -> SklResult<Array1<f64>> {
        // Simple grid search for acquisition function optimization
        let n_grid = 20;
        let mut best_x = x_start.clone();
        let mut best_acq = self.acquisition_value(&best_x, current_best)?;

        // Grid search over each dimension
        for dim in 0..bounds.nrows() {
            let min_bound = bounds[[dim, 0]];
            let max_bound = bounds[[dim, 1]];

            for i in 0..n_grid {
                let mut x_test = best_x.clone();
                x_test[dim] =
                    min_bound + (i as f64 / (n_grid - 1) as f64) * (max_bound - min_bound);

                let acq_val = self.acquisition_value(&x_test, current_best)?;
                if acq_val > best_acq {
                    best_acq = acq_val;
                    best_x = x_test;
                }
            }
        }

        Ok(best_x)
    }

    /// Add observation to the model
    pub fn add_observation(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> SklResult<()> {
        // Add new observations
        let mut X_new = Array2::<f64>::zeros((self.X_obs.nrows() + X.nrows(), self.X_obs.ncols()));
        X_new
            .slice_mut(s![..self.X_obs.nrows(), ..])
            .assign(&self.X_obs);
        X_new.slice_mut(s![self.X_obs.nrows().., ..]).assign(X);

        let mut y_new = Array1::<f64>::zeros(self.y_obs.len() + y.len());
        y_new.slice_mut(s![..self.y_obs.len()]).assign(&self.y_obs);
        y_new.slice_mut(s![self.y_obs.len()..]).assign(y);

        // Update current best
        for &val in y.iter() {
            self.current_best = self.current_best.max(val);
        }

        // Refit GP
        use crate::kernels::RBF;
        use sklears_core::traits::Fit;

        let kernel = RBF::new(1.0);
        let gpr = GaussianProcessRegressor::new().kernel(Box::new(kernel));
        self.gp = gpr.fit(&X_new, &y_new)?;
        self.X_obs = X_new;
        self.y_obs = y_new;

        Ok(())
    }
}

/// Builder for Bayesian optimizer
#[derive(Debug)]
pub struct BayesianOptimizerBuilder {
    acquisition: AcquisitionFunction,
    xi: f64,
    n_restarts: usize,
    random_state: Option<u64>,
}

impl BayesianOptimizerBuilder {
    pub fn new() -> Self {
        Self {
            acquisition: AcquisitionFunction::ExpectedImprovement,
            xi: 0.01,
            n_restarts: 10,
            random_state: None,
        }
    }

    pub fn acquisition_function(mut self, acquisition: AcquisitionFunction) -> Self {
        self.acquisition = acquisition;
        self
    }

    pub fn xi(mut self, xi: f64) -> Self {
        self.xi = xi;
        self
    }

    pub fn n_restarts(mut self, n_restarts: usize) -> Self {
        self.n_restarts = n_restarts;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn build(self) -> BayesianOptimizer {
        BayesianOptimizer {
            gp: None,
            acquisition: self.acquisition,
            xi: self.xi,
            n_restarts: self.n_restarts,
            random_state: self.random_state,
        }
    }
}

impl Default for BayesianOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
