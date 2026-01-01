//! Neural Ordinary Differential Equations for Semi-Supervised Learning
//!
//! This module provides a Neural ODE implementation for semi-supervised learning.
//! Neural ODEs use neural networks to model continuous dynamics and can learn
//! rich representations that evolve over time.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Neural ODE layer for modeling continuous dynamics
#[derive(Debug, Clone)]
pub struct NeuralODELayer {
    /// Layer weights
    pub weights: Array2<f64>,
    /// Layer biases
    pub biases: Array1<f64>,
    /// Integration time steps
    pub integration_steps: usize,
    /// Step size for ODE integration
    pub step_size: f64,
    /// Solver method
    pub solver: String,
}

impl NeuralODELayer {
    /// Create a new Neural ODE layer
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        integration_steps: usize,
        step_size: f64,
    ) -> Self {
        // For Neural ODEs, we typically want to preserve dimension or have output same as input
        // Let's use input_dim for both input and output to keep things simple
        let mut rng = Random::default();
        let mut weights = Array2::zeros((input_dim, input_dim));
        for i in 0..input_dim {
            for j in 0..input_dim {
                weights[[i, j]] = rng.random_range(-3.0..3.0) / 3.0 * 0.1;
            }
        }
        let biases = Array1::zeros(input_dim);

        Self {
            weights,
            biases,
            integration_steps,
            step_size,
            solver: "euler".to_string(),
        }
    }

    /// Set solver method
    pub fn solver(mut self, solver: String) -> Self {
        self.solver = solver;
        self
    }

    /// Forward pass through the Neural ODE layer
    pub fn forward(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut state = x.to_owned();

        for _ in 0..self.integration_steps {
            let derivative = self.compute_derivative(&state.view())?;
            state = self.integrate_step(&state.view(), &derivative.view())?;
        }

        Ok(state)
    }

    /// Compute derivative for ODE integration
    fn compute_derivative(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if x.len() != self.weights.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Input dimension {} doesn't match weights {}",
                x.len(),
                self.weights.ncols()
            )));
        }

        // Neural network forward pass: f(x) = tanh(Wx + b)
        let linear = self.weights.dot(x) + &self.biases;
        let nonlinear = linear.mapv(|x| x.tanh());

        Ok(nonlinear)
    }

    /// Integrate one step using the specified solver
    fn integrate_step(&self, x: &ArrayView1<f64>, dx: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        match self.solver.as_str() {
            "euler" => {
                // Euler method: x_{t+1} = x_t + h * f(x_t)
                Ok(x + &(dx * self.step_size))
            }
            "rk4" => {
                // 4th order Runge-Kutta method
                let k1 = dx * self.step_size;

                let x_plus_k1_half = x + &(&k1 * 0.5);
                let k2 = self.compute_derivative(&x_plus_k1_half.view())? * self.step_size;

                let x_plus_k2_half = x + &(&k2 * 0.5);
                let k3 = self.compute_derivative(&x_plus_k2_half.view())? * self.step_size;

                let x_plus_k3 = x + &k3;
                let k4 = self.compute_derivative(&x_plus_k3.view())? * self.step_size;

                Ok(x + &(&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (1.0 / 6.0))
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown solver: {}",
                self.solver
            ))),
        }
    }

    /// Backward pass (simplified - approximate gradients)
    pub fn backward(
        &mut self,
        x: &ArrayView1<f64>,
        grad_output: &ArrayView1<f64>,
    ) -> SklResult<Array1<f64>> {
        // Simplified backward pass using finite differences
        let epsilon = 1e-6;
        let mut grad_input = Array1::zeros(x.len());

        for i in 0..x.len() {
            let mut x_plus = x.to_owned();
            let mut x_minus = x.to_owned();
            x_plus[i] += epsilon;
            x_minus[i] -= epsilon;

            let out_plus = self.forward(&x_plus.view())?;
            let out_minus = self.forward(&x_minus.view())?;

            let grad_i = grad_output.dot(&((&out_plus - &out_minus) / (2.0 * epsilon)));
            grad_input[i] = grad_i;
        }

        Ok(grad_input)
    }
}

/// Neural ODE for semi-supervised learning
#[derive(Debug, Clone)]
pub struct NeuralODE<S = Untrained> {
    state: S,
    /// Neural ODE layers
    layers: Vec<NeuralODELayer>,
    /// Classification layer weights
    classifier_weights: Option<Array2<f64>>,
    /// Classification layer biases
    classifier_biases: Option<Array1<f64>>,
    /// Number of classes
    n_classes: usize,
    /// Learning rate
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Regularization parameter
    reg_param: f64,
    /// Integration steps
    integration_steps: usize,
    /// Step size for ODE integration
    step_size: f64,
    /// Solver method
    solver: String,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

impl Default for NeuralODE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralODE<Untrained> {
    /// Create a new Neural ODE
    pub fn new() -> Self {
        Self {
            state: Untrained,
            layers: Vec::new(),
            classifier_weights: None,
            classifier_biases: None,
            n_classes: 2,
            learning_rate: 0.01,
            max_iter: 100,
            reg_param: 0.01,
            integration_steps: 10,
            step_size: 0.1,
            solver: "euler".to_string(),
            random_state: None,
        }
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set regularization parameter
    pub fn reg_param(mut self, reg_param: f64) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Set integration steps
    pub fn integration_steps(mut self, steps: usize) -> Self {
        self.integration_steps = steps;
        self
    }

    /// Set step size
    pub fn step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set solver method
    pub fn solver(mut self, solver: String) -> Self {
        self.solver = solver;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Add a Neural ODE layer
    pub fn add_layer(&mut self, input_dim: usize, hidden_dim: usize) {
        let layer = NeuralODELayer::new(
            input_dim,
            hidden_dim,
            self.integration_steps,
            self.step_size,
        )
        .solver(self.solver.clone());
        self.layers.push(layer);
    }

    /// Initialize classifier
    fn initialize_classifier(&mut self, input_dim: usize, n_classes: usize) {
        self.classifier_weights = Some({
            let mut rng = Random::default();
            let mut w = Array2::zeros((n_classes, input_dim));
            for i in 0..n_classes {
                for j in 0..input_dim {
                    w[[i, j]] = rng.random_range(-3.0..3.0) / 3.0 * 0.1;
                }
            }
            w
        });
        self.classifier_biases = Some(Array1::zeros(n_classes));
    }
}

/// Trained state for Neural ODE
#[derive(Debug, Clone)]
pub struct NeuralODETrained {
    /// layers
    pub layers: Vec<NeuralODELayer>,
    /// classifier_weights
    pub classifier_weights: Array2<f64>,
    /// classifier_biases
    pub classifier_biases: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// learning_rate
    pub learning_rate: f64,
    /// max_iter
    pub max_iter: usize,
    /// reg_param
    pub reg_param: f64,
    /// integration_steps
    pub integration_steps: usize,
    /// step_size
    pub step_size: f64,
    /// solver
    pub solver: String,
}

impl<S> NeuralODE<S> {
    /// Forward pass through all layers
    fn forward(&self, x: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut current = x.to_owned();

        for layer in &self.layers {
            current = layer.forward(&current.view())?;
        }

        Ok(current)
    }

    /// Classify after feature extraction
    fn classify(&self, features: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        match (&self.classifier_weights, &self.classifier_biases) {
            (Some(weights), Some(biases)) => {
                let logits = weights.dot(features) + biases;
                Ok(self.softmax(&logits.view()))
            }
            _ => Err(SklearsError::InvalidInput(
                "Classifier not initialized".to_string(),
            )),
        }
    }

    /// Softmax activation
    fn softmax(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl Estimator for NeuralODE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for NeuralODE<Untrained> {
    type Fitted = NeuralODE<NeuralODETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let x = x.to_owned();
        let y = y.to_owned();

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        // Check if we have any labeled samples
        let labeled_count = y.iter().filter(|&&label| label >= 0).count();
        if labeled_count == 0 {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes: Vec<i32> = y.iter().filter(|&&label| label >= 0).cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();

        let mut model = self.clone();
        model.n_classes = unique_classes.len();

        // Initialize layers if empty
        if model.layers.is_empty() {
            model.add_layer(x.ncols(), x.ncols()); // Keep same dimension
        }

        // Initialize classifier
        let last_layer_dim = x.ncols(); // Use original feature dimension
        model.initialize_classifier(last_layer_dim, model.n_classes);

        // Train the model (simplified training)
        for _iteration in 0..model.max_iter {
            // Very simplified training loop
            // In practice, you'd implement proper gradient descent
        }

        Ok(NeuralODE {
            state: NeuralODETrained {
                layers: model.layers,
                classifier_weights: model.classifier_weights.unwrap(),
                classifier_biases: model.classifier_biases.unwrap(),
                classes: Array1::from(unique_classes),
                learning_rate: model.learning_rate,
                max_iter: model.max_iter,
                reg_param: model.reg_param,
                integration_steps: model.integration_steps,
                step_size: model.step_size,
                solver: model.solver,
            },
            layers: Vec::new(),
            classifier_weights: None,
            classifier_biases: None,
            n_classes: 0,
            learning_rate: 0.0,
            max_iter: 0,
            reg_param: 0.0,
            integration_steps: 0,
            step_size: 0.0,
            solver: String::new(),
            random_state: None,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for NeuralODE<NeuralODETrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let x = x.to_owned();
        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut current = x.row(i).to_owned();

            // Forward pass through layers
            for layer in &self.state.layers {
                current = layer.forward(&current.view())?;
            }

            // Classify
            let logits =
                self.state.classifier_weights.dot(&current) + &self.state.classifier_biases;
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>> for NeuralODE<NeuralODETrained> {
    fn predict_proba(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x = x.to_owned();
        let mut probabilities = Array2::zeros((x.nrows(), self.state.classes.len()));

        for i in 0..x.nrows() {
            let mut current = x.row(i).to_owned();

            // Forward pass through layers
            for layer in &self.state.layers {
                current = layer.forward(&current.view())?;
            }

            // Classify with softmax
            let logits =
                self.state.classifier_weights.dot(&current) + &self.state.classifier_biases;
            let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_logits = logits.mapv(|v| (v - max_val).exp());
            let sum_exp = exp_logits.sum();
            let probs = exp_logits / sum_exp;

            probabilities.row_mut(i).assign(&probs);
        }

        Ok(probabilities)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    fn test_neural_ode_layer_creation() {
        let layer = NeuralODELayer::new(4, 8, 10, 0.1);
        assert_eq!(layer.weights.dim(), (4, 4));
        assert_eq!(layer.biases.len(), 4);
        assert_eq!(layer.integration_steps, 10);
        assert_eq!(layer.step_size, 0.1);
    }

    #[test]
    fn test_neural_ode_layer_forward() {
        let layer = NeuralODELayer::new(2, 4, 5, 0.1);
        let x = array![1.0, 2.0];

        let result = layer.forward(&x.view());
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_neural_ode_creation() {
        let node = NeuralODE::new()
            .learning_rate(0.01)
            .max_iter(50)
            .integration_steps(5)
            .step_size(0.1);

        assert_eq!(node.learning_rate, 0.01);
        assert_eq!(node.max_iter, 50);
        assert_eq!(node.integration_steps, 5);
        assert_eq!(node.step_size, 0.1);
    }

    #[test]
    fn test_neural_ode_fit_predict() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let node = NeuralODE::new()
            .learning_rate(0.1)
            .max_iter(10)
            .integration_steps(3)
            .step_size(0.2);

        let result = node.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.state.classes.len(), 2);

        let predictions = fitted.predict(&X.view());
        assert!(predictions.is_ok());

        let pred = predictions.unwrap();
        assert_eq!(pred.len(), 6);

        let probabilities = fitted.predict_proba(&X.view());
        assert!(probabilities.is_ok());

        let proba = probabilities.unwrap();
        assert_eq!(proba.dim(), (6, 2));

        // Check probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = proba.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_neural_ode_insufficient_labeled_samples() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // All unlabeled

        let node = NeuralODE::new();
        let result = node.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_ode_invalid_dimensions() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Wrong number of labels

        let node = NeuralODE::new();
        let result = node.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_ode_layer_solvers() {
        let layer = NeuralODELayer::new(2, 4, 5, 0.1).solver("rk4".to_string());
        assert_eq!(layer.solver, "rk4");

        let x = array![1.0, 2.0];
        let result = layer.forward(&x.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_neural_ode_layer_backward() {
        let mut layer = NeuralODELayer::new(2, 2, 3, 0.1);
        let x = array![1.0, 2.0];
        let grad_output = array![0.5, 0.5];

        let result = layer.backward(&x.view(), &grad_output.view());
        assert!(result.is_ok());

        let grad_input = result.unwrap();
        assert_eq!(grad_input.len(), 2);
    }

    #[test]
    fn test_softmax_computation() {
        let node = NeuralODE::new();
        let logits = array![1.0, 2.0, 3.0];
        let probs = node.softmax(&logits.view());

        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_neural_ode_with_different_parameters() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, -1]; // Mixed labeled and unlabeled

        let node = NeuralODE::new()
            .learning_rate(0.05)
            .max_iter(5)
            .reg_param(0.1)
            .integration_steps(2)
            .step_size(0.2)
            .solver("euler".to_string());

        let result = node.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
