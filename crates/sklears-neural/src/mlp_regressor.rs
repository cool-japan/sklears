//! Multi-layer Perceptron (MLP) regressor implementation.

use crate::{
    activation::Activation,
    solvers::{AdamSolver, LambSolver, LarsSolver, LearningRateSchedule, SgdSolver, Solver},
    utils::{
        create_batches_regression, initialize_biases, initialize_weights, EarlyStopping, WeightInit,
    },
    NeuralResult,
};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{error::SklearsError, traits::Fit, traits::Predict};
use std::collections::HashMap;

/// Type alias for neural network weights and biases
type WeightsAndBiases = (Vec<Array2<f64>>, Vec<Array1<f64>>);

/// Multi-layer Perceptron regressor
#[derive(Debug, Clone)]
pub struct MLPRegressor<State = sklears_core::traits::Untrained> {
    // Hyperparameters
    pub hidden_layer_sizes: Vec<usize>,
    pub activation: Activation,
    pub solver: Solver,
    pub alpha: f64, // L2 regularization term
    pub batch_size: Option<usize>,
    pub learning_rate_init: f64,
    pub max_iter: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
    pub tol: f64,
    pub verbose: bool,
    pub warm_start: bool,
    pub momentum: f64,
    pub nesterovs_momentum: bool,
    pub early_stopping: bool,
    pub validation_fraction: f64,
    pub beta_1: f64,  // Adam parameter
    pub beta_2: f64,  // Adam parameter
    pub epsilon: f64, // Adam parameter
    pub n_iter_no_change: usize,
    pub max_fun: usize,
    pub learning_rate: LearningRateSchedule,
    pub power_t: f64,
    pub weight_init: WeightInit,

    // Trained state
    state: State,
}

/// Trained MLP regressor
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainedMLPRegressor {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub n_features_in: usize,
    pub n_outputs: usize,
    pub n_layers: usize,
    pub activation: Activation,
    pub loss: f64,
    pub n_iter: usize,
}

impl MLPRegressor<sklears_core::traits::Untrained> {
    /// Create a new MLP regressor with default parameters
    pub fn new() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            activation: Activation::Relu,
            solver: Solver::Adam,
            alpha: 0.0001,
            batch_size: None,
            learning_rate_init: 0.001,
            max_iter: 200,
            shuffle: true,
            random_state: None,
            tol: 1e-4,
            verbose: false,
            warm_start: false,
            momentum: 0.9,
            nesterovs_momentum: true,
            early_stopping: false,
            validation_fraction: 0.1,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            n_iter_no_change: 10,
            max_fun: 15000,
            learning_rate: LearningRateSchedule::Constant,
            power_t: 0.5,
            weight_init: WeightInit::Xavier,
            state: sklears_core::traits::Untrained,
        }
    }

    /// Set the hidden layer sizes
    pub fn hidden_layer_sizes(mut self, sizes: &[usize]) -> Self {
        self.hidden_layer_sizes = sizes.to_vec();
        self
    }

    /// Set the activation function
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set the solver
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set the L2 regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the initial learning rate
    pub fn learning_rate_init(mut self, learning_rate: f64) -> Self {
        self.learning_rate_init = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to shuffle the training data
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the tolerance for optimization
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set weight initialization strategy
    pub fn weight_init(mut self, weight_init: WeightInit) -> Self {
        self.weight_init = weight_init;
        self
    }
}

impl Fit<Array2<f64>, Array2<f64>> for MLPRegressor<sklears_core::traits::Untrained> {
    type Fitted = MLPRegressor<TrainedMLPRegressor>;

    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> NeuralResult<MLPRegressor<TrainedMLPRegressor>> {
        let (n_samples, n_features) = x.dim();
        let (n_samples_y, n_outputs) = y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("x.nrows()={n_samples}"),
                actual: format!("y.nrows()={n_samples_y}"),
            });
        }

        // Initialize random number generator
        let mut rng = match self.random_state {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(42),
        };

        // Build layer sizes
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend_from_slice(&self.hidden_layer_sizes);
        layer_sizes.push(n_outputs);

        // Initialize weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let w = initialize_weights(layer_sizes[i], layer_sizes[i + 1], &self.weight_init);
            let b = initialize_biases(layer_sizes[i + 1], &self.weight_init);
            weights.push(w);
            biases.push(b);
        }

        // Determine batch size
        let batch_size = self.batch_size.unwrap_or(n_samples.min(200));

        // Initialize solver
        let mut solver = match self.solver {
            Solver::Sgd => SolverType::Sgd(SgdSolver::new(
                self.learning_rate_init,
                self.momentum,
                self.nesterovs_momentum,
                self.learning_rate,
                self.power_t,
                self.learning_rate_init,
            )),
            Solver::Adam => SolverType::Adam(AdamSolver::new(
                self.learning_rate_init,
                self.beta_1,
                self.beta_2,
                self.epsilon,
            )),
            Solver::AdamW => {
                return Err(SklearsError::NotImplemented(
                    "AdamW solver not yet integrated with MLP".to_string(),
                ));
            }
            Solver::RMSprop => {
                return Err(SklearsError::NotImplemented(
                    "RMSprop solver not yet integrated with MLP".to_string(),
                ));
            }
            Solver::Nadam => {
                return Err(SklearsError::NotImplemented(
                    "Nadam solver not yet integrated with MLP".to_string(),
                ));
            }
            Solver::Lbfgs => {
                return Err(SklearsError::NotImplemented(
                    "L-BFGS solver not yet implemented".to_string(),
                ));
            }
            Solver::Lars => SolverType::Lars(LarsSolver::new(
                self.learning_rate_init,
                self.momentum,
                self.alpha, // Use alpha as weight decay
                0.001,      // lars_coefficient
                1e-8,       // epsilon
                1.0,        // trust_coefficient
            )),
            Solver::Lamb => SolverType::Lamb(LambSolver::new(
                self.learning_rate_init,
                self.beta_1,
                self.beta_2,
                self.epsilon,
                self.alpha, // Use alpha as weight decay
                1.0,        // trust_coefficient
            )),
        };

        // Initialize solver state
        match &mut solver {
            SolverType::Sgd(sgd) => sgd.initialize(&weights, &biases),
            SolverType::Adam(adam) => adam.initialize(&weights, &biases),
            SolverType::Lars(lars) => lars.initialize(&weights, &biases),
            SolverType::Lamb(lamb) => lamb.initialize(&weights, &biases),
        }

        // Training loop
        let mut best_loss = f64::INFINITY;
        let mut n_iter = 0;
        let mut early_stopping = if self.early_stopping {
            Some(EarlyStopping::new(self.n_iter_no_change, self.tol, true))
        } else {
            None
        };

        for epoch in 0..self.max_iter {
            let batches = create_batches_regression(x, y, batch_size, self.shuffle);

            let mut total_loss = 0.0;
            let mut total_samples = 0;

            for (batch_x, batch_y) in batches {
                // Forward pass
                let (activations, loss) =
                    self.forward_pass(&batch_x, &batch_y, &weights, &biases)?;

                // Backward pass
                let (weight_grads, bias_grads) =
                    self.backward_pass(&batch_x, &batch_y, &activations, &weights, &biases)?;

                // Update parameters
                match &mut solver {
                    SolverType::Sgd(sgd) => {
                        sgd.update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)?;
                    }
                    SolverType::Adam(adam) => {
                        adam.update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)?;
                    }
                    SolverType::Lars(lars) => {
                        lars.update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)?;
                    }
                    SolverType::Lamb(lamb) => {
                        lamb.update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)?;
                    }
                }

                total_loss += loss * batch_x.nrows() as f64;
                total_samples += batch_x.nrows();
            }

            let avg_loss = total_loss / total_samples as f64;
            n_iter = epoch + 1;

            if self.verbose && epoch % 10 == 0 {
                println!("Epoch {epoch}: Loss = {avg_loss:.6}");
            }

            // Check for early stopping
            if let Some(ref mut early_stop) = early_stopping {
                if early_stop.should_stop(avg_loss, &weights, &biases) {
                    if self.verbose {
                        println!("Early stopping at epoch {epoch}");
                    }
                    if let Some((best_weights, best_biases)) = early_stop.get_best_weights() {
                        weights = best_weights.to_vec();
                        biases = best_biases.to_vec();
                    }
                    break;
                }
            }

            // Check for convergence
            if (best_loss - avg_loss).abs() < self.tol {
                if self.verbose {
                    println!("Converged at epoch {epoch}");
                }
                break;
            }

            best_loss = avg_loss;
        }

        Ok(MLPRegressor {
            hidden_layer_sizes: self.hidden_layer_sizes,
            activation: self.activation,
            solver: self.solver,
            alpha: self.alpha,
            batch_size: self.batch_size,
            learning_rate_init: self.learning_rate_init,
            max_iter: self.max_iter,
            shuffle: self.shuffle,
            random_state: self.random_state,
            tol: self.tol,
            verbose: self.verbose,
            warm_start: self.warm_start,
            momentum: self.momentum,
            nesterovs_momentum: self.nesterovs_momentum,
            early_stopping: self.early_stopping,
            validation_fraction: self.validation_fraction,
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            n_iter_no_change: self.n_iter_no_change,
            max_fun: self.max_fun,
            learning_rate: self.learning_rate,
            power_t: self.power_t,
            weight_init: self.weight_init,
            state: TrainedMLPRegressor {
                weights,
                biases,
                n_features_in: n_features,
                n_outputs,
                n_layers: layer_sizes.len(),
                activation: self.activation,
                loss: best_loss,
                n_iter,
            },
        })
    }
}

impl MLPRegressor<sklears_core::traits::Untrained> {
    fn forward_pass(
        &self,
        x: &Array2<f64>,
        y_true: &Array2<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> NeuralResult<(Vec<Array2<f64>>, f64)> {
        let mut activations = vec![x.clone()];

        // Forward through hidden layers
        for i in 0..weights.len() - 1 {
            let z = activations[i].dot(&weights[i]) + &biases[i];
            let a = self.activation.apply(&z);
            activations.push(a);
        }

        // Output layer (linear activation for regression)
        let z_out =
            activations.last().unwrap().dot(weights.last().unwrap()) + biases.last().unwrap();
        let y_pred = z_out; // Linear activation (identity)
        activations.push(y_pred.clone());

        // Compute mean squared error loss
        let loss = self.compute_loss(&y_pred, y_true, weights)?;

        Ok((activations, loss))
    }

    fn backward_pass(
        &self,
        _x: &Array2<f64>,
        y_true: &Array2<f64>,
        activations: &[Array2<f64>],
        weights: &[Array2<f64>],
        _biases: &[Array1<f64>],
    ) -> NeuralResult<WeightsAndBiases> {
        let n_layers = weights.len();
        let mut weight_grads = vec![Array2::zeros((0, 0)); n_layers];
        let mut bias_grads = vec![Array1::zeros(0); n_layers];
        let n_samples = y_true.nrows() as f64;

        // Output layer gradient (linear activation + MSE loss)
        let mut delta = (&activations[n_layers] - y_true) * (2.0 / n_samples);

        // Gradients for output layer
        weight_grads[n_layers - 1] = activations[n_layers - 1].t().dot(&delta);
        bias_grads[n_layers - 1] = delta.sum_axis(Axis(0));

        // Add L2 regularization to weight gradients
        if self.alpha > 0.0 {
            weight_grads[n_layers - 1] =
                &weight_grads[n_layers - 1] + &weights[n_layers - 1] * (2.0 * self.alpha);
        }

        // Backpropagate through hidden layers
        for i in (0..n_layers - 1).rev() {
            // Compute delta for current layer
            delta =
                delta.dot(&weights[i + 1].t()) * &self.activation.derivative(&activations[i + 1]);

            // Compute gradients
            weight_grads[i] = activations[i].t().dot(&delta);
            bias_grads[i] = delta.sum_axis(Axis(0));

            // Add L2 regularization
            if self.alpha > 0.0 {
                weight_grads[i] = &weight_grads[i] + &weights[i] * (2.0 * self.alpha);
            }
        }

        Ok((weight_grads, bias_grads))
    }

    fn compute_loss(
        &self,
        y_pred: &Array2<f64>,
        y_true: &Array2<f64>,
        weights: &[Array2<f64>],
    ) -> NeuralResult<f64> {
        let n_samples = y_pred.nrows() as f64;

        // Mean squared error loss
        let diff = y_pred - y_true;
        let mut loss = (&diff * &diff).sum() / (2.0 * n_samples);

        // Add L2 regularization
        if self.alpha > 0.0 {
            let mut reg_loss = 0.0;
            for weight in weights {
                reg_loss += weight.mapv(|x| x * x).sum();
            }
            loss += self.alpha * reg_loss;
        }

        Ok(loss)
    }
}

impl Predict<Array2<f64>, Array2<f64>> for MLPRegressor<TrainedMLPRegressor> {
    fn predict(&self, x: &Array2<f64>) -> NeuralResult<Array2<f64>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.state.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: self.state.n_features_in,
                actual: n_features,
            });
        }

        let mut activations = x.clone();

        // Forward through hidden layers
        for i in 0..self.state.weights.len() - 1 {
            let z = activations.dot(&self.state.weights[i]) + &self.state.biases[i];
            activations = self.state.activation.apply(&z);
        }

        // Output layer (linear)
        let z_out =
            activations.dot(self.state.weights.last().unwrap()) + self.state.biases.last().unwrap();

        Ok(z_out)
    }
}

impl MLPRegressor<TrainedMLPRegressor> {
    /// Get the loss from training
    pub fn loss(&self) -> f64 {
        self.state.loss
    }

    /// Get the number of iterations performed during training
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }
}

impl Default for MLPRegressor<sklears_core::traits::Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum SolverType {
    Sgd(SgdSolver),
    Adam(AdamSolver),
    Lars(LarsSolver),
    Lamb(LambSolver),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mlp_regressor_creation() {
        let mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10, 5])
            .activation(Activation::Relu)
            .max_iter(100)
            .random_state(42);

        assert_eq!(mlp.hidden_layer_sizes, vec![10, 5]);
        assert_eq!(mlp.activation, Activation::Relu);
        assert_eq!(mlp.max_iter, 100);
        assert_eq!(mlp.random_state, Some(42));
    }

    #[test]
    fn test_mlp_regressor_fit_predict_single_output() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![
            [3.0], // 1*1 + 2*1 = 3
            [5.0], // 2*1 + 3*1 = 5
            [7.0], // 3*1 + 4*1 = 7
            [9.0], // 4*1 + 5*1 = 9
        ];

        let mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[5])
            .max_iter(100)
            .random_state(42);

        let trained_mlp = mlp.fit(&x, &y).unwrap();
        let predictions = trained_mlp.predict(&x).unwrap();

        assert_eq!(predictions.dim(), (4, 1));

        // Check that predictions are reasonable (finite values)
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_mlp_regressor_fit_predict_multi_output() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![[3.0, 6.0], [5.0, 10.0], [7.0, 14.0], [9.0, 18.0],];

        let mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[5])
            .max_iter(50)
            .random_state(42);

        let trained_mlp = mlp.fit(&x, &y).unwrap();
        let predictions = trained_mlp.predict(&x).unwrap();

        assert_eq!(predictions.dim(), (4, 2));

        // Check that predictions are reasonable (finite values)
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_mlp_regressor_dimension_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0]]; // Wrong number of samples

        let mlp = MLPRegressor::new().random_state(42);
        let result = mlp.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_mlp_regressor_simple_linear_function() {
        // Test with a simple linear function: y = 2*x1 + 3*x2
        let x = array![[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0],];
        let y = array![
            [5.0],  // 2*1 + 3*1 = 5
            [7.0],  // 2*2 + 3*1 = 7
            [8.0],  // 2*1 + 3*2 = 8
            [10.0], // 2*2 + 3*2 = 10
        ];

        let mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(200)
            .learning_rate_init(0.01)
            .random_state(42);

        let trained_mlp = mlp.fit(&x, &y).unwrap();
        let predictions = trained_mlp.predict(&x).unwrap();

        // The network should be able to learn this simple linear function reasonably well
        // We don't expect perfect fit, but should be better than random
        let mut total_error = 0.0;
        for i in 0..y.nrows() {
            total_error += (predictions[[i, 0]] - y[[i, 0]]).abs();
        }
        let mean_absolute_error = total_error / y.nrows() as f64;

        // Should have some learning (error should be less than the range of y values)
        assert!(mean_absolute_error < 5.0);
    }
}
