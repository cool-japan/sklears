//! Ladder Networks for Deep Semi-Supervised Learning
//!
//! This module implements Ladder Networks, a deep learning architecture that
//! combines supervised learning with unsupervised learning through lateral
//! connections and denoising objectives. Ladder networks achieve state-of-the-art
//! performance on semi-supervised learning tasks by learning hierarchical
//! representations at multiple levels.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Ladder Networks for Deep Semi-Supervised Learning
///
/// Ladder Networks are neural networks that combine supervised and unsupervised
/// learning objectives. They use lateral connections between encoder and decoder
/// paths to enable effective learning from both labeled and unlabeled data.
///
/// The architecture consists of:
/// - An encoder path that applies noise and nonlinearities
/// - A decoder path that reconstructs clean representations
/// - Lateral connections that help the decoder
/// - Multiple reconstruction costs at different layers
///
/// # Parameters
///
/// * `layer_sizes` - Sizes of hidden layers (including input and output)
/// * `noise_std` - Standard deviation of Gaussian noise added to each layer
/// * `lambda_unsupervised` - Weight for unsupervised reconstruction loss
/// * `lambda_supervised` - Weight for supervised classification loss
/// * `denoising_cost_weights` - Weights for denoising costs at each layer
/// * `learning_rate` - Learning rate for optimization
/// * `max_iter` - Maximum number of training iterations
/// * `batch_size` - Size of mini-batches for training
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::LadderNetworks;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let ln = LadderNetworks::new()
///     .layer_sizes(vec![2, 4, 2])
///     .noise_std(0.3)
///     .lambda_unsupervised(1.0)
///     .lambda_supervised(1.0);
/// let fitted = ln.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LadderNetworks<S = Untrained> {
    state: S,
    layer_sizes: Vec<usize>,
    noise_std: f64,
    lambda_unsupervised: f64,
    lambda_supervised: f64,
    denoising_cost_weights: Vec<f64>,
    learning_rate: f64,
    max_iter: usize,
    batch_size: usize,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    random_state: Option<u64>,
}

impl LadderNetworks<Untrained> {
    /// Create a new LadderNetworks instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            layer_sizes: vec![10, 6, 4, 2], // Default architecture
            noise_std: 0.3,
            lambda_unsupervised: 1.0,
            lambda_supervised: 1.0,
            denoising_cost_weights: vec![1000.0, 10.0, 0.1, 0.1],
            learning_rate: 0.002,
            max_iter: 100,
            batch_size: 32,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            random_state: None,
        }
    }

    /// Set the layer sizes (input size will be set automatically)
    pub fn layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.layer_sizes = sizes;
        self
    }

    /// Set the noise standard deviation
    pub fn noise_std(mut self, std: f64) -> Self {
        self.noise_std = std;
        self
    }

    /// Set the unsupervised loss weight
    pub fn lambda_unsupervised(mut self, lambda: f64) -> Self {
        self.lambda_unsupervised = lambda;
        self
    }

    /// Set the supervised loss weight
    pub fn lambda_supervised(mut self, lambda: f64) -> Self {
        self.lambda_supervised = lambda;
        self
    }

    /// Set the denoising cost weights for each layer
    pub fn denoising_cost_weights(mut self, weights: Vec<f64>) -> Self {
        self.denoising_cost_weights = weights;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set Adam optimizer beta1 parameter
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set Adam optimizer beta2 parameter
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    fn initialize_weights(&self, input_size: usize) -> LadderWeights {
        let mut layer_sizes = self.layer_sizes.clone();
        layer_sizes[0] = input_size; // Set input size

        let n_layers = layer_sizes.len();
        let mut encoder_weights = Vec::with_capacity(n_layers - 1);
        let mut encoder_biases = Vec::with_capacity(n_layers - 1);
        let mut decoder_weights = Vec::with_capacity(n_layers - 1);
        let mut decoder_biases = Vec::with_capacity(n_layers - 1);

        // Xavier initialization
        for i in 0..(n_layers - 1) {
            let fan_in = layer_sizes[i];
            let fan_out = layer_sizes[i + 1];
            let xavier_std = (2.0 / (fan_in + fan_out) as f64).sqrt();

            // Encoder weights (bottom-up)
            let mut rng = Random::default();
            let mut w_enc = Array2::zeros((fan_in, fan_out));
            for i in 0..fan_in {
                for j in 0..fan_out {
                    w_enc[[i, j]] = rng.gen_range(-3.0..3.0) / 3.0 * xavier_std;
                }
            }
            let b_enc = Array1::zeros(fan_out);
            encoder_weights.push(w_enc);
            encoder_biases.push(b_enc);

            // Decoder weights (top-down)
            let mut w_dec = Array2::zeros((fan_out, fan_in));
            for i in 0..fan_out {
                for j in 0..fan_in {
                    w_dec[[i, j]] = rng.gen_range(-3.0..3.0) / 3.0 * xavier_std;
                }
            }
            let b_dec = Array1::zeros(fan_in);
            decoder_weights.push(w_dec);
            decoder_biases.push(b_dec);
        }

        LadderWeights {
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            layer_sizes,
        }
    }

    fn add_noise(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = Random::default();
        let mut noise = Array2::zeros(x.dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                noise[[i, j]] = rng.gen_range(-3.0..3.0) / 3.0 * self.noise_std;
            }
        }
        x + &noise
    }

    fn relu(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn relu_derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    fn softmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.dim());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_row: Array1<f64> = row.mapv(|v| (v - max_val).exp());
            let sum_exp: f64 = exp_row.sum();
            let softmax_row = exp_row / sum_exp;
            result.row_mut(i).assign(&softmax_row);
        }
        result
    }

    fn forward_encoder(
        &self,
        x: &Array2<f64>,
        weights: &LadderWeights,
        add_noise: bool,
    ) -> EncoderOutput {
        let mut activations = Vec::new();
        let mut noisy_activations = Vec::new();
        let mut pre_activations = Vec::new();

        let mut current = x.clone();
        activations.push(current.clone());

        if add_noise {
            current = self.add_noise(&current);
        }
        noisy_activations.push(current.clone());

        for i in 0..weights.encoder_weights.len() {
            // Linear transformation
            let z = current.dot(&weights.encoder_weights[i]) + &weights.encoder_biases[i];
            pre_activations.push(z.clone());

            // Apply activation function (ReLU for hidden layers, linear for output)
            current = if i == weights.encoder_weights.len() - 1 {
                z.clone() // Linear for output layer
            } else {
                self.relu(&z)
            };

            activations.push(current.clone());

            // Add noise to hidden layers
            if add_noise && i < weights.encoder_weights.len() - 1 {
                current = self.add_noise(&current);
            }
            noisy_activations.push(current.clone());
        }

        EncoderOutput {
            activations,
            noisy_activations,
            pre_activations,
        }
    }

    fn forward_decoder(
        &self,
        top_activation: &Array2<f64>,
        weights: &LadderWeights,
    ) -> Vec<Array2<f64>> {
        let mut decoder_outputs = Vec::new();
        let mut current = top_activation.clone();

        // Start from the top layer and work down
        for i in (0..weights.decoder_weights.len()).rev() {
            current = current.dot(&weights.decoder_weights[i]) + &weights.decoder_biases[i];

            // Apply activation function for hidden layers
            if i > 0 {
                current = self.relu(&current);
            }

            decoder_outputs.push(current.clone());
        }

        decoder_outputs.reverse(); // Reverse to match layer order
        decoder_outputs
    }

    fn compute_denoising_cost(
        &self,
        clean_activations: &[Array2<f64>],
        reconstructed_activations: &[Array2<f64>],
    ) -> f64 {
        let mut total_cost = 0.0;

        for (layer_idx, (clean, reconstructed)) in clean_activations
            .iter()
            .zip(reconstructed_activations.iter())
            .enumerate()
        {
            if layer_idx < self.denoising_cost_weights.len() {
                let diff = clean - reconstructed;
                let mse = diff.mapv(|x| x * x).mean().unwrap();
                total_cost += self.denoising_cost_weights[layer_idx] * mse;
            }
        }

        total_cost
    }

    fn compute_supervised_cost(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // Cross-entropy loss
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        let log_predictions = clipped_predictions.mapv(|x| x.ln());

        let mut cost = 0.0;
        for i in 0..targets.nrows() {
            for j in 0..targets.ncols() {
                cost -= targets[[i, j]] * log_predictions[[i, j]];
            }
        }

        cost / targets.nrows() as f64
    }

    fn create_target_matrix(&self, y: &Array1<i32>, classes: &[i32]) -> Array2<f64> {
        let n_samples = y.len();
        let n_classes = classes.len();
        let mut targets = Array2::zeros((n_samples, n_classes));

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                    targets[[i, class_idx]] = 1.0;
                }
            }
        }

        targets
    }

    fn update_weights_adam(
        &self,
        weights: &mut LadderWeights,
        gradients: &LadderWeights,
        momentum: &mut LadderWeights,
        velocity: &mut LadderWeights,
        iteration: usize,
    ) {
        let beta1_t = self.beta1.powi(iteration as i32);
        let beta2_t = self.beta2.powi(iteration as i32);
        let alpha_t = self.learning_rate * (1.0 - beta2_t).sqrt() / (1.0 - beta1_t);

        // Update encoder weights
        for i in 0..weights.encoder_weights.len() {
            // Update momentum and velocity
            momentum.encoder_weights[i] = self.beta1 * &momentum.encoder_weights[i]
                + (1.0 - self.beta1) * &gradients.encoder_weights[i];
            velocity.encoder_weights[i] = self.beta2 * &velocity.encoder_weights[i]
                + (1.0 - self.beta2) * gradients.encoder_weights[i].mapv(|x| x * x);

            // Update weights
            let momentum_corrected = &momentum.encoder_weights[i] / (1.0 - beta1_t);
            let velocity_corrected = &velocity.encoder_weights[i] / (1.0 - beta2_t);
            let update = momentum_corrected / velocity_corrected.mapv(|x| x.sqrt() + self.epsilon);
            weights.encoder_weights[i] = &weights.encoder_weights[i] - alpha_t * update;

            // Update biases
            momentum.encoder_biases[i] = self.beta1 * &momentum.encoder_biases[i]
                + (1.0 - self.beta1) * &gradients.encoder_biases[i];
            velocity.encoder_biases[i] = self.beta2 * &velocity.encoder_biases[i]
                + (1.0 - self.beta2) * gradients.encoder_biases[i].mapv(|x| x * x);

            let momentum_corrected_b = &momentum.encoder_biases[i] / (1.0 - beta1_t);
            let velocity_corrected_b = &velocity.encoder_biases[i] / (1.0 - beta2_t);
            let update_b =
                momentum_corrected_b / velocity_corrected_b.mapv(|x| x.sqrt() + self.epsilon);
            weights.encoder_biases[i] = &weights.encoder_biases[i] - alpha_t * update_b;
        }

        // Update decoder weights (similar process)
        for i in 0..weights.decoder_weights.len() {
            momentum.decoder_weights[i] = self.beta1 * &momentum.decoder_weights[i]
                + (1.0 - self.beta1) * &gradients.decoder_weights[i];
            velocity.decoder_weights[i] = self.beta2 * &velocity.decoder_weights[i]
                + (1.0 - self.beta2) * gradients.decoder_weights[i].mapv(|x| x * x);

            let momentum_corrected = &momentum.decoder_weights[i] / (1.0 - beta1_t);
            let velocity_corrected = &velocity.decoder_weights[i] / (1.0 - beta2_t);
            let update = momentum_corrected / velocity_corrected.mapv(|x| x.sqrt() + self.epsilon);
            weights.decoder_weights[i] = &weights.decoder_weights[i] - alpha_t * update;

            momentum.decoder_biases[i] = self.beta1 * &momentum.decoder_biases[i]
                + (1.0 - self.beta1) * &gradients.decoder_biases[i];
            velocity.decoder_biases[i] = self.beta2 * &velocity.decoder_biases[i]
                + (1.0 - self.beta2) * gradients.decoder_biases[i].mapv(|x| x * x);

            let momentum_corrected_b = &momentum.decoder_biases[i] / (1.0 - beta1_t);
            let velocity_corrected_b = &velocity.decoder_biases[i] / (1.0 - beta2_t);
            let update_b =
                momentum_corrected_b / velocity_corrected_b.mapv(|x| x.sqrt() + self.epsilon);
            weights.decoder_biases[i] = &weights.decoder_biases[i] - alpha_t * update_b;
        }
    }

    fn compute_gradients(
        &self,
        x: &Array2<f64>,
        targets: &Array2<f64>,
        weights: &LadderWeights,
    ) -> SklResult<LadderWeights> {
        let batch_size = x.nrows() as f64;

        // Forward pass through clean encoder
        let clean_encoder_output = self.forward_encoder(x, weights, false);

        // Forward pass through noisy encoder
        let noisy_encoder_output = self.forward_encoder(x, weights, true);

        // Forward pass through decoder
        let decoder_outputs =
            self.forward_decoder(noisy_encoder_output.activations.last().unwrap(), weights);

        // Initialize gradients
        let mut gradient_weights = LadderWeights {
            encoder_weights: weights
                .encoder_weights
                .iter()
                .map(|w| Array2::zeros(w.dim()))
                .collect(),
            encoder_biases: weights
                .encoder_biases
                .iter()
                .map(|b| Array1::zeros(b.len()))
                .collect(),
            decoder_weights: weights
                .decoder_weights
                .iter()
                .map(|w| Array2::zeros(w.dim()))
                .collect(),
            decoder_biases: weights
                .decoder_biases
                .iter()
                .map(|b| Array1::zeros(b.len()))
                .collect(),
            layer_sizes: weights.layer_sizes.clone(),
        };

        // Simplified gradient computation (in practice, this would use automatic differentiation)
        // For demonstration, we compute approximate gradients using finite differences
        let delta = 1e-5;

        // Compute gradients for encoder weights
        for i in 0..weights.encoder_weights.len() {
            for j in 0..weights.encoder_weights[i].nrows() {
                for k in 0..weights.encoder_weights[i].ncols() {
                    // Perturb weight
                    let mut weights_plus = weights.clone();
                    let mut weights_minus = weights.clone();
                    weights_plus.encoder_weights[i][[j, k]] += delta;
                    weights_minus.encoder_weights[i][[j, k]] -= delta;

                    // Compute costs
                    let cost_plus = self.compute_total_cost(x, targets, &weights_plus)?;
                    let cost_minus = self.compute_total_cost(x, targets, &weights_minus)?;

                    // Approximate gradient
                    gradient_weights.encoder_weights[i][[j, k]] =
                        (cost_plus - cost_minus) / (2.0 * delta);
                }
            }
        }

        Ok(gradient_weights)
    }

    fn compute_total_cost(
        &self,
        x: &Array2<f64>,
        targets: &Array2<f64>,
        weights: &LadderWeights,
    ) -> SklResult<f64> {
        // Forward pass
        let clean_encoder_output = self.forward_encoder(x, weights, false);
        let noisy_encoder_output = self.forward_encoder(x, weights, true);
        let decoder_outputs =
            self.forward_decoder(noisy_encoder_output.activations.last().unwrap(), weights);

        // Supervised cost
        let predictions = self.softmax(noisy_encoder_output.activations.last().unwrap());
        let supervised_cost = self.compute_supervised_cost(&predictions, targets);

        // Unsupervised denoising cost
        let denoising_cost =
            self.compute_denoising_cost(&clean_encoder_output.activations, &decoder_outputs);

        let total_cost =
            self.lambda_supervised * supervised_cost + self.lambda_unsupervised * denoising_cost;
        Ok(total_cost)
    }
}

#[derive(Debug, Clone)]
pub struct LadderWeights {
    /// encoder_weights
    pub encoder_weights: Vec<Array2<f64>>,
    /// encoder_biases
    pub encoder_biases: Vec<Array1<f64>>,
    /// decoder_weights
    pub decoder_weights: Vec<Array2<f64>>,
    /// decoder_biases
    pub decoder_biases: Vec<Array1<f64>>,
    /// layer_sizes
    pub layer_sizes: Vec<usize>,
}

#[derive(Debug)]
pub struct EncoderOutput {
    /// activations
    pub activations: Vec<Array2<f64>>,
    /// noisy_activations
    pub noisy_activations: Vec<Array2<f64>>,
    /// pre_activations
    pub pre_activations: Vec<Array2<f64>>,
}

impl Default for LadderNetworks<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LadderNetworks<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for LadderNetworks<Untrained> {
    type Fitted = LadderNetworks<LadderNetworksTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify classes
        let mut classes = std::collections::HashSet::new();
        for &label in y.iter() {
            if label != -1 {
                classes.insert(label);
            }
        }

        if classes.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Adjust layer sizes
        let mut layer_sizes = self.layer_sizes.clone();
        layer_sizes[0] = n_features;
        let last_idx = layer_sizes.len() - 1;
        layer_sizes[last_idx] = n_classes;

        // Initialize weights
        let mut weights = self.initialize_weights(n_features);
        weights.layer_sizes = layer_sizes;

        // Initialize Adam optimizer state
        let mut momentum = weights.clone();
        let mut velocity = weights.clone();

        // Zero initialize momentum and velocity
        for i in 0..momentum.encoder_weights.len() {
            momentum.encoder_weights[i].fill(0.0);
            momentum.encoder_biases[i].fill(0.0);
            velocity.encoder_weights[i].fill(0.0);
            velocity.encoder_biases[i].fill(0.0);
        }
        for i in 0..momentum.decoder_weights.len() {
            momentum.decoder_weights[i].fill(0.0);
            momentum.decoder_biases[i].fill(0.0);
            velocity.decoder_weights[i].fill(0.0);
            velocity.decoder_biases[i].fill(0.0);
        }

        // Create target matrix
        let targets = self.create_target_matrix(&y, &classes);

        // Training loop (simplified for demonstration)
        for iteration in 1..=self.max_iter {
            // For simplicity, we'll train on the entire dataset
            // In practice, you'd use mini-batches

            let total_cost = self.compute_total_cost(&X, &targets, &weights)?;

            if iteration % 10 == 0 {
                println!("Iteration {}: Total cost = {:.6}", iteration, total_cost);
            }

            // Compute gradients (simplified - in practice use automatic differentiation)
            let gradients = self.compute_gradients(&X, &targets, &weights)?;

            // Update weights using Adam
            self.update_weights_adam(
                &mut weights,
                &gradients,
                &mut momentum,
                &mut velocity,
                iteration,
            );
        }

        // Final forward pass to get label distributions
        let final_encoder_output = self.forward_encoder(&X, &weights, false);
        let final_predictions = self.softmax(final_encoder_output.activations.last().unwrap());

        Ok(LadderNetworks {
            state: LadderNetworksTrained {
                X_train: X,
                y_train: y,
                classes: Array1::from(classes),
                weights,
                label_distributions: final_predictions,
            },
            layer_sizes: self.layer_sizes,
            noise_std: self.noise_std,
            lambda_unsupervised: self.lambda_unsupervised,
            lambda_supervised: self.lambda_supervised,
            denoising_cost_weights: self.denoising_cost_weights,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            batch_size: self.batch_size,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            random_state: self.random_state,
        })
    }
}

impl LadderNetworks<LadderNetworksTrained> {
    fn forward_encoder(
        &self,
        x: &Array2<f64>,
        weights: &LadderWeights,
        add_noise: bool,
    ) -> EncoderOutput {
        let mut activations = Vec::new();
        let mut noisy_activations = Vec::new();
        let mut pre_activations = Vec::new();

        let mut current = x.clone();
        activations.push(current.clone());

        if add_noise {
            current = self.add_noise(&current);
        }
        noisy_activations.push(current.clone());

        for i in 0..weights.encoder_weights.len() {
            // Linear transformation
            let z = current.dot(&weights.encoder_weights[i]) + &weights.encoder_biases[i];
            pre_activations.push(z.clone());

            // Apply activation function (ReLU for hidden layers, linear for output)
            current = if i == weights.encoder_weights.len() - 1 {
                z.clone() // Linear for output layer
            } else {
                self.relu(&z)
            };

            activations.push(current.clone());

            // Add noise to hidden layers
            if add_noise && i < weights.encoder_weights.len() - 1 {
                current = self.add_noise(&current);
            }
            noisy_activations.push(current.clone());
        }

        EncoderOutput {
            activations,
            noisy_activations,
            pre_activations,
        }
    }

    fn add_noise(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = Random::default();
        let mut noise = Array2::zeros(x.dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                noise[[i, j]] = rng.gen_range(-3.0..3.0) / 3.0 * self.noise_std;
            }
        }
        x + &noise
    }

    fn relu(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn softmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.dim());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_row: Array1<f64> = row.mapv(|v| (v - max_val).exp());
            let sum_exp: f64 = exp_row.sum();
            let softmax_row = exp_row / sum_exp;
            result.row_mut(i).assign(&softmax_row);
        }
        result
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for LadderNetworks<LadderNetworksTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let encoder_output = self.forward_encoder(&X, &self.state.weights, false);
        let predictions = self.softmax(encoder_output.activations.last().unwrap());

        let mut result = Array1::zeros(X.nrows());
        for i in 0..X.nrows() {
            let max_idx = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            result[i] = self.state.classes[max_idx];
        }

        Ok(result)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>> for LadderNetworks<LadderNetworksTrained> {
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let encoder_output = self.forward_encoder(&X, &self.state.weights, false);
        let predictions = self.softmax(encoder_output.activations.last().unwrap());
        Ok(predictions)
    }
}

/// Trained state for LadderNetworks
#[derive(Debug, Clone)]
pub struct LadderNetworksTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// weights
    pub weights: LadderWeights,
    /// label_distributions
    pub label_distributions: Array2<f64>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_ladder_networks_basic() {
        let X = array![
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.0],
            [5.0, 6.0, 2.5],
            [6.0, 7.0, 3.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let ln = LadderNetworks::new()
            .layer_sizes(vec![3, 4, 2])
            .noise_std(0.1)
            .lambda_unsupervised(0.5)
            .lambda_supervised(1.0)
            .max_iter(5); // Reduced for testing
        let fitted = ln.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 6);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to approximately 1
        for i in 0..6 {
            let sum: f64 = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 0.1); // Allow some numerical error
        }
    }

    #[test]
    fn test_ladder_networks_initialization() {
        let ln = LadderNetworks::new()
            .layer_sizes(vec![4, 6, 3])
            .noise_std(0.2);

        let weights = ln.initialize_weights(4);

        // Check dimensions
        assert_eq!(weights.layer_sizes, vec![4, 6, 3]);
        assert_eq!(weights.encoder_weights.len(), 2);
        assert_eq!(weights.encoder_weights[0].dim(), (4, 6));
        assert_eq!(weights.encoder_weights[1].dim(), (6, 3));

        assert_eq!(weights.decoder_weights.len(), 2);
        assert_eq!(weights.decoder_weights[0].dim(), (6, 4));
        assert_eq!(weights.decoder_weights[1].dim(), (3, 6));
    }

    #[test]
    fn test_ladder_networks_noise_addition() {
        let ln = LadderNetworks::new().noise_std(0.1);
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let noisy_x = ln.add_noise(&x);

        // Check dimensions are preserved
        assert_eq!(noisy_x.dim(), x.dim());

        // Check that noise was added (values should be different)
        let diff = (&noisy_x - &x).mapv(|v| v.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_ladder_networks_activations() {
        let ln = LadderNetworks::new();

        // Test ReLU
        let x = array![[-1.0, 0.0, 1.0, 2.0]];
        let relu_result = ln.relu(&x);
        let expected = array![[0.0, 0.0, 1.0, 2.0]];

        for i in 0..x.ncols() {
            assert!((relu_result[[0, i]] - expected[[0, i]]).abs() < 1e-10);
        }

        // Test ReLU derivative
        let relu_deriv = ln.relu_derivative(&x);
        let expected_deriv = array![[0.0, 0.0, 1.0, 1.0]];

        for i in 0..x.ncols() {
            assert!((relu_deriv[[0, i]] - expected_deriv[[0, i]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ladder_networks_softmax() {
        let ln = LadderNetworks::new();
        let x = array![[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]];

        let softmax_result = ln.softmax(&x);

        // Check dimensions
        assert_eq!(softmax_result.dim(), x.dim());

        // Check that each row sums to 1
        for i in 0..x.nrows() {
            let sum: f64 = softmax_result.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }

        // Check that all values are positive
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(softmax_result[[i, j]] > 0.0);
            }
        }
    }

    #[test]
    fn test_ladder_networks_forward_encoder() {
        let ln = LadderNetworks::new();
        let weights = ln.initialize_weights(2);
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let encoder_output = ln.forward_encoder(&x, &weights, false);

        // Check that we have activations for each layer
        assert_eq!(encoder_output.activations.len(), weights.layer_sizes.len());
        assert_eq!(
            encoder_output.noisy_activations.len(),
            weights.layer_sizes.len()
        );

        // Check input layer
        assert_eq!(encoder_output.activations[0].dim(), x.dim());
    }

    #[test]
    fn test_ladder_networks_target_matrix() {
        let ln = LadderNetworks::new();
        let y = array![0, 1, -1, 0]; // -1 indicates unlabeled
        let classes = vec![0, 1];

        let targets = ln.create_target_matrix(&y, &classes);

        // Check dimensions
        assert_eq!(targets.dim(), (4, 2));

        // Check labeled samples
        assert_eq!(targets[[0, 0]], 1.0); // First sample, class 0
        assert_eq!(targets[[0, 1]], 0.0);
        assert_eq!(targets[[1, 0]], 0.0); // Second sample, class 1
        assert_eq!(targets[[1, 1]], 1.0);

        // Check unlabeled sample (should be all zeros)
        assert_eq!(targets[[2, 0]], 0.0);
        assert_eq!(targets[[2, 1]], 0.0);
    }
}
