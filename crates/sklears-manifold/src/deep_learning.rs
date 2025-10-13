use scirs2_core::essentials::{Normal, Uniform};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

#[derive(Debug, Clone)]
pub struct AutoencoderManifold<S = Untrained> {
    n_components: usize,
    hidden_layers: Vec<usize>,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    state: S,
}

#[derive(Debug, Clone)]
pub struct TrainedAutoencoder {
    encoder_weights: Vec<Array2<f64>>,
    encoder_biases: Vec<Array1<f64>>,
    decoder_weights: Vec<Array2<f64>>,
    decoder_biases: Vec<Array1<f64>>,
}

impl AutoencoderManifold<Untrained> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            hidden_layers: vec![128, 64],
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            state: Untrained,
        }
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    fn initialize_weights(&self, input_dim: usize) -> TrainedAutoencoder {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(&self.hidden_layers);
        layer_sizes.push(self.n_components);

        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((input_size, output_size), |(_, _)| {
                let mut rng = thread_rng();
                Normal::new(0.0, (2.0 / (input_size + output_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(output_size);

            encoder_weights.push(weight);
            encoder_biases.push(bias);
        }

        layer_sizes.reverse();
        let mut decoder_weights = Vec::new();
        let mut decoder_biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((input_size, output_size), |(_, _)| {
                let mut rng = thread_rng();
                Normal::new(0.0, (2.0 / (input_size + output_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(output_size);

            decoder_weights.push(weight);
            decoder_biases.push(bias);
        }

        /// TrainedAutoencoder
        TrainedAutoencoder {
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
        }
    }
}

impl AutoencoderManifold<TrainedAutoencoder> {
    fn forward_pass(&self, x: &ArrayView2<f64>) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let mut activations = x.to_owned();

        for (weights, biases) in self
            .state
            .encoder_weights
            .iter()
            .zip(self.state.encoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            activations.mapv_inplace(|x| x.max(0.0)); // ReLU
        }

        let encoded = activations.clone();

        for (weights, biases) in self
            .state
            .decoder_weights
            .iter()
            .zip(self.state.decoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            if weights != self.state.decoder_weights.last().unwrap() {
                activations.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
        }

        Ok((encoded, activations))
    }
}

impl Estimator for AutoencoderManifold<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for AutoencoderManifold<Untrained> {
    type Fitted = AutoencoderManifold<TrainedAutoencoder>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_components >= n_features {
            return Err(SklearsError::InvalidInput(
                "n_components must be less than number of features".to_string(),
            ));
        }

        let state = self.initialize_weights(n_features);

        // Note: For a complete implementation, we would need to implement
        // the training loop with gradient descent here. For now, we return
        // the initialized model as a placeholder.

        Ok(AutoencoderManifold {
            n_components: self.n_components,
            hidden_layers: self.hidden_layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for AutoencoderManifold<TrainedAutoencoder> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = x.to_owned();

        for (weights, biases) in self
            .state
            .encoder_weights
            .iter()
            .zip(self.state.encoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            activations.mapv_inplace(|x| x.max(0.0)); // ReLU
        }

        Ok(activations)
    }
}

#[derive(Debug, Clone)]
pub struct VariationalAutoencoder<S = Untrained> {
    n_components: usize,
    hidden_layers: Vec<usize>,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    beta: f64, // KL divergence weight
    state: S,
}

#[derive(Debug, Clone)]
pub struct TrainedVAE {
    encoder_weights: Vec<Array2<f64>>,
    encoder_biases: Vec<Array1<f64>>,
    mu_layer: (Array2<f64>, Array1<f64>),
    logvar_layer: (Array2<f64>, Array1<f64>),
    decoder_weights: Vec<Array2<f64>>,
    decoder_biases: Vec<Array1<f64>>,
}

impl VariationalAutoencoder<Untrained> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            hidden_layers: vec![128, 64],
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            beta: 1.0,
            state: Untrained,
        }
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

impl VariationalAutoencoder<TrainedVAE> {
    fn reparameterize(&self, mu: &Array2<f64>, logvar: &Array2<f64>) -> Array2<f64> {
        let std = logvar.mapv(|x| (0.5 * x).exp());
        let mut rng = thread_rng();
        let epsilon = Array2::from_shape_fn(mu.dim(), |(_, _)| {
            Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
        });
        mu + &std * &epsilon
    }

    fn kl_divergence(&self, mu: &Array2<f64>, logvar: &Array2<f64>) -> f64 {
        let kl: Array2<f64> = -0.5 * (1.0 + logvar - mu.mapv(|x| x * x) - logvar.mapv(|x| x.exp()));
        kl.sum() / mu.nrows() as f64
    }
}

impl Estimator for VariationalAutoencoder<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<Array2<f64>, Array2<f64>> for VariationalAutoencoder<TrainedVAE> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = x.to_owned();

        for (weights, biases) in self
            .state
            .encoder_weights
            .iter()
            .zip(self.state.encoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            activations.mapv_inplace(|x| x.max(0.0)); // ReLU
        }

        let mu = activations.dot(&self.state.mu_layer.0) + &self.state.mu_layer.1;
        let logvar = activations.dot(&self.state.logvar_layer.0) + &self.state.logvar_layer.1;

        Ok(self.reparameterize(&mu, &logvar))
    }
}

/// Adversarial Autoencoder (AAE) - Combines autoencoder reconstruction with adversarial regularization
#[derive(Debug, Clone)]
pub struct AdversarialAutoencoder<S = Untrained> {
    n_components: usize,
    hidden_layers: Vec<usize>,
    discriminator_layers: Vec<usize>,
    epochs: usize,
    learning_rate: f64,
    adversarial_weight: f64,
    batch_size: usize,
    prior_type: PriorType,
    state: S,
}

#[derive(Debug, Clone)]
pub enum PriorType {
    /// Gaussian
    Gaussian,
    /// Uniform
    Uniform,
    /// Categorical
    Categorical(usize), // number of categories
}

#[derive(Debug, Clone)]
pub struct TrainedAAE {
    encoder_weights: Vec<Array2<f64>>,
    encoder_biases: Vec<Array1<f64>>,
    decoder_weights: Vec<Array2<f64>>,
    decoder_biases: Vec<Array1<f64>>,
    discriminator_weights: Vec<Array2<f64>>,
    discriminator_biases: Vec<Array1<f64>>,
    prior_type: PriorType,
}

impl AdversarialAutoencoder<Untrained> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            hidden_layers: vec![128, 64],
            discriminator_layers: vec![64, 32],
            epochs: 100,
            learning_rate: 0.001,
            adversarial_weight: 1.0,
            batch_size: 32,
            prior_type: PriorType::Gaussian,
            state: Untrained,
        }
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_discriminator_layers(mut self, layers: Vec<usize>) -> Self {
        self.discriminator_layers = layers;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_adversarial_weight(mut self, weight: f64) -> Self {
        self.adversarial_weight = weight;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_prior_type(mut self, prior_type: PriorType) -> Self {
        self.prior_type = prior_type;
        self
    }

    fn initialize_aae_weights(&self, input_dim: usize) -> TrainedAAE {
        let mut rng = thread_rng();

        // Initialize encoder
        let mut encoder_layer_sizes = vec![input_dim];
        encoder_layer_sizes.extend(&self.hidden_layers);
        encoder_layer_sizes.push(self.n_components);

        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();

        for i in 0..encoder_layer_sizes.len() - 1 {
            let input_size = encoder_layer_sizes[i];
            let output_size = encoder_layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((input_size, output_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (input_size + output_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(output_size);

            encoder_weights.push(weight);
            encoder_biases.push(bias);
        }

        // Initialize decoder (reverse of encoder)
        let mut decoder_layer_sizes = encoder_layer_sizes.clone();
        decoder_layer_sizes.reverse();

        let mut decoder_weights = Vec::new();
        let mut decoder_biases = Vec::new();

        for i in 0..decoder_layer_sizes.len() - 1 {
            let input_size = decoder_layer_sizes[i];
            let output_size = decoder_layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((input_size, output_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (input_size + output_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(output_size);

            decoder_weights.push(weight);
            decoder_biases.push(bias);
        }

        // Initialize discriminator
        let mut disc_layer_sizes = vec![self.n_components];
        disc_layer_sizes.extend(&self.discriminator_layers);
        disc_layer_sizes.push(1); // Binary classification output

        let mut discriminator_weights = Vec::new();
        let mut discriminator_biases = Vec::new();

        for i in 0..disc_layer_sizes.len() - 1 {
            let input_size = disc_layer_sizes[i];
            let output_size = disc_layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((input_size, output_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (input_size + output_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(output_size);

            discriminator_weights.push(weight);
            discriminator_biases.push(bias);
        }

        /// TrainedAAE
        TrainedAAE {
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            discriminator_weights,
            discriminator_biases,
            prior_type: self.prior_type.clone(),
        }
    }
}

impl AdversarialAutoencoder<TrainedAAE> {
    fn encode(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = x.to_owned();

        for (weights, biases) in self
            .state
            .encoder_weights
            .iter()
            .zip(self.state.encoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            // Apply ReLU except for the final layer
            if weights != self.state.encoder_weights.last().unwrap() {
                activations.mapv_inplace(|x| x.max(0.0));
            }
        }

        Ok(activations)
    }

    fn decode(&self, z: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = z.to_owned();

        for (i, (weights, biases)) in self
            .state
            .decoder_weights
            .iter()
            .zip(self.state.decoder_biases.iter())
            .enumerate()
        {
            activations = activations.dot(weights) + biases;
            // Apply ReLU except for the final layer
            if i < self.state.decoder_weights.len() - 1 {
                activations.mapv_inplace(|x| x.max(0.0));
            }
        }

        Ok(activations)
    }

    fn discriminate(&self, z: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = z.to_owned();

        for (i, (weights, biases)) in self
            .state
            .discriminator_weights
            .iter()
            .zip(self.state.discriminator_biases.iter())
            .enumerate()
        {
            activations = activations.dot(weights) + biases;

            if i < self.state.discriminator_weights.len() - 1 {
                // ReLU for hidden layers
                activations.mapv_inplace(|x| x.max(0.0));
            } else {
                // Sigmoid for output layer
                activations.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            }
        }

        Ok(activations)
    }

    fn sample_prior(&self, n_samples: usize) -> Array2<f64> {
        let mut rng = thread_rng();

        match &self.state.prior_type {
            PriorType::Gaussian => {
                Array2::from_shape_fn((n_samples, self.n_components), |(_, _)| {
                    Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
                })
            }
            PriorType::Uniform => {
                Array2::from_shape_fn((n_samples, self.n_components), |(_, _)| {
                    Uniform::new(-1.0, 1.0).unwrap().sample(&mut rng)
                })
            }
            PriorType::Categorical(n_cats) => {
                let mut samples = Array2::zeros((n_samples, self.n_components));
                for i in 0..n_samples {
                    let cat = rng.gen_range(0..*n_cats);
                    if cat < self.n_components {
                        samples[[i, cat]] = 1.0;
                    }
                }
                samples
            }
        }
    }

    fn reconstruction_loss(
        &self,
        x_original: &ArrayView2<f64>,
        x_reconstructed: &ArrayView2<f64>,
    ) -> f64 {
        let diff = x_original - x_reconstructed;
        (diff.mapv(|x| x * x).sum()) / x_original.nrows() as f64
    }

    fn adversarial_loss(&self, discriminator_output: &ArrayView2<f64>, is_real: bool) -> f64 {
        let target = if is_real { 1.0 } else { 0.0 };
        let epsilon = 1e-12;

        let mut loss = 0.0;
        for &output in discriminator_output.iter() {
            let clamped_output = output.max(epsilon).min(1.0 - epsilon);
            loss -= target * clamped_output.ln() + (1.0 - target) * (1.0 - clamped_output).ln();
        }

        loss / discriminator_output.nrows() as f64
    }
}

impl Estimator for AdversarialAutoencoder<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for AdversarialAutoencoder<Untrained> {
    type Fitted = AdversarialAutoencoder<TrainedAAE>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_components >= n_features {
            return Err(SklearsError::InvalidInput(
                "n_components must be less than number of features".to_string(),
            ));
        }

        let state = self.initialize_aae_weights(n_features);

        // Note: For a complete implementation, we would need to implement
        // the full adversarial training loop with alternating updates between
        // autoencoder and discriminator. For now, we return the initialized model.

        Ok(AdversarialAutoencoder {
            n_components: self.n_components,
            hidden_layers: self.hidden_layers,
            discriminator_layers: self.discriminator_layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            adversarial_weight: self.adversarial_weight,
            batch_size: self.batch_size,
            prior_type: self.prior_type,
            state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for AdversarialAutoencoder<TrainedAAE> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.encode(&x.view())
    }
}

/// Neural Ordinary Differential Equation (NODE) - Models continuous dynamics for manifold learning
#[derive(Debug, Clone)]
pub struct NeuralODE<S = Untrained> {
    n_components: usize,
    hidden_layers: Vec<usize>,
    integration_time: f64,
    num_time_steps: usize,
    solver_type: ODESolverType,
    learning_rate: f64,
    epochs: usize,
    state: S,
}

#[derive(Debug, Clone)]
pub enum ODESolverType {
    /// Euler
    Euler,
    /// RungeKutta4
    RungeKutta4,
}

#[derive(Debug, Clone)]
pub struct TrainedNODE {
    ode_func_weights: Vec<Array2<f64>>,
    ode_func_biases: Vec<Array1<f64>>,
    encoder_weights: Vec<Array2<f64>>,
    encoder_biases: Vec<Array1<f64>>,
    decoder_weights: Vec<Array2<f64>>,
    decoder_biases: Vec<Array1<f64>>,
    integration_time: f64,
    num_time_steps: usize,
    solver_type: ODESolverType,
}

impl NeuralODE<Untrained> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            hidden_layers: vec![64, 32],
            integration_time: 1.0,
            num_time_steps: 10,
            solver_type: ODESolverType::RungeKutta4,
            learning_rate: 0.001,
            epochs: 100,
            state: Untrained,
        }
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_integration_time(mut self, time: f64) -> Self {
        self.integration_time = time;
        self
    }

    pub fn with_num_time_steps(mut self, steps: usize) -> Self {
        self.num_time_steps = steps;
        self
    }

    pub fn with_solver_type(mut self, solver: ODESolverType) -> Self {
        self.solver_type = solver;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    fn initialize_node_weights(&self, input_dim: usize) -> TrainedNODE {
        let mut rng = thread_rng();

        // Initialize encoder (compress to manifold dimension)
        let encoder_layers = [input_dim, self.n_components];
        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();

        for i in 0..encoder_layers.len() - 1 {
            let in_size = encoder_layers[i];
            let out_size = encoder_layers[i + 1];

            let weight = Array2::from_shape_fn((in_size, out_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (in_size + out_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(out_size);

            encoder_weights.push(weight);
            encoder_biases.push(bias);
        }

        // Initialize ODE function (dynamics in manifold space)
        let mut ode_layer_sizes = vec![self.n_components];
        ode_layer_sizes.extend(&self.hidden_layers);
        ode_layer_sizes.push(self.n_components); // Output same dim as input

        let mut ode_func_weights = Vec::new();
        let mut ode_func_biases = Vec::new();

        for i in 0..ode_layer_sizes.len() - 1 {
            let in_size = ode_layer_sizes[i];
            let out_size = ode_layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((in_size, out_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (in_size + out_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(out_size);

            ode_func_weights.push(weight);
            ode_func_biases.push(bias);
        }

        // Initialize decoder (expand back to original dimension)
        let decoder_layers = [self.n_components, input_dim];
        let mut decoder_weights = Vec::new();
        let mut decoder_biases = Vec::new();

        for i in 0..decoder_layers.len() - 1 {
            let in_size = decoder_layers[i];
            let out_size = decoder_layers[i + 1];

            let weight = Array2::from_shape_fn((in_size, out_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (in_size + out_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });
            let bias = Array1::zeros(out_size);

            decoder_weights.push(weight);
            decoder_biases.push(bias);
        }

        /// TrainedNODE
        TrainedNODE {
            ode_func_weights,
            ode_func_biases,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            integration_time: self.integration_time,
            num_time_steps: self.num_time_steps,
            solver_type: self.solver_type.clone(),
        }
    }
}

impl NeuralODE<TrainedNODE> {
    fn encode(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = x.to_owned();

        for (weights, biases) in self
            .state
            .encoder_weights
            .iter()
            .zip(self.state.encoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            // No activation for encoder to preserve manifold structure
        }

        Ok(activations)
    }

    fn decode(&self, z: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = z.to_owned();

        for (weights, biases) in self
            .state
            .decoder_weights
            .iter()
            .zip(self.state.decoder_biases.iter())
        {
            activations = activations.dot(weights) + biases;
            // No activation for decoder to preserve reconstruction
        }

        Ok(activations)
    }

    fn ode_func(&self, _t: f64, h: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = h.to_owned();

        for (i, (weights, biases)) in self
            .state
            .ode_func_weights
            .iter()
            .zip(self.state.ode_func_biases.iter())
            .enumerate()
        {
            activations = activations.dot(weights) + biases;
            // Apply activation function except for output layer
            if i < self.state.ode_func_weights.len() - 1 {
                activations.mapv_inplace(|x| x.tanh()); // Tanh for bounded dynamics
            }
        }

        Ok(activations)
    }

    fn solve_ode(&self, h0: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let dt = self.state.integration_time / self.state.num_time_steps as f64;
        let mut h = h0.to_owned();

        for i in 0..self.state.num_time_steps {
            let t = i as f64 * dt;

            match self.state.solver_type {
                ODESolverType::Euler => {
                    // Forward Euler: h_{n+1} = h_n + dt * f(t_n, h_n)
                    let dh_dt = self.ode_func(t, &h.view())?;
                    h = h + dh_dt * dt;
                }
                ODESolverType::RungeKutta4 => {
                    // 4th-order Runge-Kutta
                    let k1 = self.ode_func(t, &h.view())? * dt;
                    let k2 = self.ode_func(t + dt / 2.0, &(&h + &k1 * 0.5).view())? * dt;
                    let k3 = self.ode_func(t + dt / 2.0, &(&h + &k2 * 0.5).view())? * dt;
                    let k4 = self.ode_func(t + dt, &(&h + &k3).view())? * dt;

                    h = h + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
                }
            }
        }

        Ok(h)
    }

    fn forward_pass(&self, x: &ArrayView2<f64>) -> SklResult<(Array2<f64>, Array2<f64>)> {
        // Encode to manifold space
        let h0 = self.encode(x)?;

        // Solve ODE in manifold space
        let h_final = self.solve_ode(&h0.view())?;

        // Decode back to original space
        let x_reconstructed = self.decode(&h_final.view())?;

        Ok((h_final, x_reconstructed))
    }

    pub fn get_manifold_trajectory(&self, x: &ArrayView2<f64>) -> SklResult<Vec<Array2<f64>>> {
        let h0 = self.encode(x)?;
        let dt = self.state.integration_time / self.state.num_time_steps as f64;
        let mut h = h0.clone();
        let mut trajectory = vec![h0];

        for i in 0..self.state.num_time_steps {
            let t = i as f64 * dt;

            match self.state.solver_type {
                ODESolverType::Euler => {
                    let dh_dt = self.ode_func(t, &h.view())?;
                    h = h + dh_dt * dt;
                }
                ODESolverType::RungeKutta4 => {
                    let k1 = self.ode_func(t, &h.view())? * dt;
                    let k2 = self.ode_func(t + dt / 2.0, &(&h + &k1 * 0.5).view())? * dt;
                    let k3 = self.ode_func(t + dt / 2.0, &(&h + &k2 * 0.5).view())? * dt;
                    let k4 = self.ode_func(t + dt, &(&h + &k3).view())? * dt;

                    h = h + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
                }
            }

            trajectory.push(h.clone());
        }

        Ok(trajectory)
    }
}

impl Estimator for NeuralODE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for NeuralODE<Untrained> {
    type Fitted = NeuralODE<TrainedNODE>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_components >= n_features {
            return Err(SklearsError::InvalidInput(
                "n_components must be less than number of features".to_string(),
            ));
        }

        if self.integration_time <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "integration_time must be positive".to_string(),
            ));
        }

        if self.num_time_steps == 0 {
            return Err(SklearsError::InvalidInput(
                "num_time_steps must be positive".to_string(),
            ));
        }

        let state = self.initialize_node_weights(n_features);

        // Note: For a complete implementation, we would need to implement
        // the training loop with gradient computation through the ODE solver.
        // This requires adjoint sensitivity method for backpropagation.

        Ok(NeuralODE {
            n_components: self.n_components,
            hidden_layers: self.hidden_layers,
            integration_time: self.integration_time,
            num_time_steps: self.num_time_steps,
            solver_type: self.solver_type,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for NeuralODE<TrainedNODE> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (h_final, _) = self.forward_pass(&x.view())?;
        Ok(h_final)
    }
}

/// Continuous Normalizing Flow (CNF) - Invertible transformation through continuous dynamics
#[derive(Debug, Clone)]
pub struct ContinuousNormalizingFlow<S = Untrained> {
    n_components: usize,
    hidden_layers: Vec<usize>,
    integration_time: f64,
    num_time_steps: usize,
    solver_type: ODESolverType,
    trace_estimator: TraceEstimator,
    learning_rate: f64,
    epochs: usize,
    state: S,
}

#[derive(Debug, Clone)]
pub enum TraceEstimator {
    /// Exact
    Exact, // For small dimensions
    /// Hutchinson
    Hutchinson, // Stochastic trace estimation
    /// RademacherRandom
    RademacherRandom, // Random projection trace estimation
}

#[derive(Debug, Clone)]
pub struct TrainedCNF {
    dynamics_weights: Vec<Array2<f64>>,
    dynamics_biases: Vec<Array1<f64>>,
    integration_time: f64,
    num_time_steps: usize,
    solver_type: ODESolverType,
    trace_estimator: TraceEstimator,
    input_dim: usize,
}

impl ContinuousNormalizingFlow<Untrained> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            hidden_layers: vec![64, 32, 64],
            integration_time: 1.0,
            num_time_steps: 20,
            solver_type: ODESolverType::RungeKutta4,
            trace_estimator: TraceEstimator::Hutchinson,
            learning_rate: 0.001,
            epochs: 100,
            state: Untrained,
        }
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_integration_time(mut self, time: f64) -> Self {
        self.integration_time = time;
        self
    }

    pub fn with_num_time_steps(mut self, steps: usize) -> Self {
        self.num_time_steps = steps;
        self
    }

    pub fn with_solver_type(mut self, solver: ODESolverType) -> Self {
        self.solver_type = solver;
        self
    }

    pub fn with_trace_estimator(mut self, estimator: TraceEstimator) -> Self {
        self.trace_estimator = estimator;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    fn initialize_cnf_weights(&self, input_dim: usize) -> TrainedCNF {
        let mut rng = thread_rng();

        // Initialize dynamics function (velocity field)
        let mut dynamics_layer_sizes = vec![self.n_components];
        dynamics_layer_sizes.extend(&self.hidden_layers);
        dynamics_layer_sizes.push(self.n_components); // Output same dimension as input

        let mut dynamics_weights = Vec::new();
        let mut dynamics_biases = Vec::new();

        for i in 0..dynamics_layer_sizes.len() - 1 {
            let in_size = dynamics_layer_sizes[i];
            let out_size = dynamics_layer_sizes[i + 1];

            let weight = Array2::from_shape_fn((in_size, out_size), |(_, _)| {
                Normal::new(0.0, (2.0 / (in_size + out_size) as f64).sqrt())
                    .unwrap()
                    .sample(&mut rng)
            });

            // Initialize biases to zero except for the final layer (small random bias)
            let bias = if i == dynamics_layer_sizes.len() - 2 {
                Array1::from_shape_fn(out_size, |_| {
                    Normal::new(0.0, 0.01).unwrap().sample(&mut rng)
                })
            } else {
                Array1::zeros(out_size)
            };

            dynamics_weights.push(weight);
            dynamics_biases.push(bias);
        }

        /// TrainedCNF
        TrainedCNF {
            dynamics_weights,
            dynamics_biases,
            integration_time: self.integration_time,
            num_time_steps: self.num_time_steps,
            solver_type: self.solver_type.clone(),
            trace_estimator: self.trace_estimator.clone(),
            input_dim,
        }
    }
}

impl ContinuousNormalizingFlow<TrainedCNF> {
    fn validate_input_dim(&self, data: &ArrayView2<f64>) -> SklResult<()> {
        if data.ncols() != self.state.input_dim {
            return Err(SklearsError::InvalidInput(format!(
                "Expected input with {} features, but received {}",
                self.state.input_dim,
                data.ncols()
            )));
        }

        Ok(())
    }

    fn split_input(&self, data: &ArrayView2<f64>) -> (Array2<f64>, Option<Array2<f64>>) {
        let latent = data
            .slice(scirs2_core::ndarray::s![.., 0..self.n_components])
            .to_owned();

        if self.state.input_dim > self.n_components {
            let residual = data
                .slice(scirs2_core::ndarray::s![
                    ..,
                    self.n_components..self.state.input_dim
                ])
                .to_owned();
            (latent, Some(residual))
        } else {
            (latent, None)
        }
    }

    fn combine_latent(&self, latent: &Array2<f64>, residual: Option<&Array2<f64>>) -> Array2<f64> {
        if self.state.input_dim == self.n_components {
            return latent.clone();
        }

        let mut combined = Array2::zeros((latent.nrows(), self.state.input_dim));
        combined
            .slice_mut(scirs2_core::ndarray::s![.., 0..self.n_components])
            .assign(latent);

        if let Some(residual) = residual {
            combined
                .slice_mut(scirs2_core::ndarray::s![
                    ..,
                    self.n_components..self.state.input_dim
                ])
                .assign(residual);
        }

        combined
    }

    fn integrate_latent(
        &self,
        latent0: &Array2<f64>,
        forward: bool,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        let mut latent = latent0.clone();
        let mut log_det = Array1::zeros(latent.nrows());

        if self.state.num_time_steps == 0 {
            return Ok((latent, log_det));
        }

        let base_dt = self.state.integration_time / self.state.num_time_steps as f64;
        let dt_signed = if forward { base_dt } else { -base_dt };

        for step in 0..self.state.num_time_steps {
            let t = if forward {
                step as f64 * base_dt
            } else {
                self.state.integration_time - step as f64 * base_dt
            };

            let trace = self.compute_trace_jacobian(t, &latent.view())?;
            let trace_contrib = trace.mapv(|val| val * dt_signed);
            log_det = log_det + trace_contrib;

            match self.state.solver_type {
                ODESolverType::Euler => {
                    let dz_dt = self.dynamics(t, &latent.view())?;
                    latent = latent + dz_dt * dt_signed;
                }
                ODESolverType::RungeKutta4 => {
                    let k1 = self.dynamics(t, &latent.view())? * dt_signed;
                    let k2 = self.dynamics(t + dt_signed / 2.0, &(&latent + &k1 * 0.5).view())?
                        * dt_signed;
                    let k3 = self.dynamics(t + dt_signed / 2.0, &(&latent + &k2 * 0.5).view())?
                        * dt_signed;
                    let k4 = self.dynamics(t + dt_signed, &(&latent + &k3).view())? * dt_signed;

                    latent = latent + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
                }
            }
        }

        Ok((latent, log_det))
    }

    fn dynamics(&self, _t: f64, z: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut activations = z.to_owned();

        for (i, (weights, biases)) in self
            .state
            .dynamics_weights
            .iter()
            .zip(self.state.dynamics_biases.iter())
            .enumerate()
        {
            activations = activations.dot(weights) + biases;
            // Apply activation function except for output layer
            if i < self.state.dynamics_weights.len() - 1 {
                activations.mapv_inplace(|x| x.tanh()); // Smooth, bounded activation
            }
        }

        Ok(activations)
    }

    fn compute_trace_jacobian(&self, _t: f64, z: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let epsilon = 1e-6;
        let (n_samples, n_dims) = z.dim();

        match self.state.trace_estimator {
            TraceEstimator::Exact => {
                // Compute exact trace using finite differences (expensive for large dimensions)
                let mut traces = Array1::zeros(n_samples);

                for sample_idx in 0..n_samples {
                    let z_sample = z.slice(scirs2_core::ndarray::s![sample_idx, ..]).to_owned();
                    let mut trace = 0.0;

                    for dim_idx in 0..n_dims {
                        // Compute partial derivative using finite differences
                        let mut z_plus = z_sample.clone();
                        let mut z_minus = z_sample.clone();

                        z_plus[dim_idx] += epsilon;
                        z_minus[dim_idx] -= epsilon;

                        let z_plus_2d = z_plus.view().insert_axis(scirs2_core::ndarray::Axis(0));
                        let z_minus_2d = z_minus.view().insert_axis(scirs2_core::ndarray::Axis(0));

                        let f_plus = self.dynamics(_t, &z_plus_2d)?;
                        let f_minus = self.dynamics(_t, &z_minus_2d)?;

                        // Partial derivative of f_i with respect to z_i
                        let partial_deriv =
                            (f_plus[[0, dim_idx]] - f_minus[[0, dim_idx]]) / (2.0 * epsilon);
                        trace += partial_deriv;
                    }

                    traces[sample_idx] = trace;
                }

                Ok(traces)
            }
            TraceEstimator::Hutchinson => {
                // Hutchinson's stochastic trace estimator: Tr(J) ≈ E[ε^T J ε] where ε ~ Rademacher
                let mut rng = thread_rng();
                let mut traces = Array1::zeros(n_samples);

                for sample_idx in 0..n_samples {
                    let z_sample = z.slice(scirs2_core::ndarray::s![sample_idx, ..]).to_owned();

                    // Generate random Rademacher vector
                    let epsilon_vec = Array1::from_shape_fn(n_dims, |_| {
                        if rng.gen::<f64>() < 0.5 {
                            -1.0
                        } else {
                            1.0
                        }
                    });

                    // Compute J * ε using finite differences
                    let z_plus = &z_sample + &epsilon_vec * epsilon;
                    let z_minus = &z_sample - &epsilon_vec * epsilon;

                    let z_plus_2d = z_plus.view().insert_axis(scirs2_core::ndarray::Axis(0));
                    let z_minus_2d = z_minus.view().insert_axis(scirs2_core::ndarray::Axis(0));

                    let f_plus = self.dynamics(_t, &z_plus_2d)?;
                    let f_minus = self.dynamics(_t, &z_minus_2d)?;

                    let jv = (f_plus.slice(scirs2_core::ndarray::s![0, ..]).to_owned()
                        - f_minus.slice(scirs2_core::ndarray::s![0, ..]).to_owned())
                        / (2.0 * epsilon);

                    // ε^T * (J * ε) = ε^T * jv
                    let trace_estimate = epsilon_vec.dot(&jv);
                    traces[sample_idx] = trace_estimate;
                }

                Ok(traces)
            }
            TraceEstimator::RademacherRandom => {
                // Alternative stochastic estimator
                let mut rng = thread_rng();
                let mut traces = Array1::zeros(n_samples);

                for sample_idx in 0..n_samples {
                    let z_sample = z.slice(scirs2_core::ndarray::s![sample_idx, ..]).to_owned();

                    // Generate random Gaussian vector
                    let epsilon_vec = Array1::from_shape_fn(n_dims, |_| {
                        Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
                    });

                    let z_plus = &z_sample + &epsilon_vec * epsilon;
                    let z_minus = &z_sample - &epsilon_vec * epsilon;

                    let z_plus_2d = z_plus.view().insert_axis(scirs2_core::ndarray::Axis(0));
                    let z_minus_2d = z_minus.view().insert_axis(scirs2_core::ndarray::Axis(0));

                    let f_plus = self.dynamics(_t, &z_plus_2d)?;
                    let f_minus = self.dynamics(_t, &z_minus_2d)?;

                    let jv = (f_plus.slice(scirs2_core::ndarray::s![0, ..]).to_owned()
                        - f_minus.slice(scirs2_core::ndarray::s![0, ..]).to_owned())
                        / (2.0 * epsilon);

                    let trace_estimate = epsilon_vec.dot(&jv);
                    traces[sample_idx] = trace_estimate;
                }

                Ok(traces)
            }
        }
    }

    pub fn forward_flow(&self, z0: &ArrayView2<f64>) -> SklResult<(Array2<f64>, Array1<f64>)> {
        self.validate_input_dim(z0)?;
        let (latent0, residual) = self.split_input(z0);
        let (latent_final, log_det) = self.integrate_latent(&latent0, true)?;
        let combined = self.combine_latent(&latent_final, residual.as_ref());

        Ok((combined, log_det))
    }

    pub fn backward_flow(&self, z1: &ArrayView2<f64>) -> SklResult<(Array2<f64>, Array1<f64>)> {
        self.validate_input_dim(z1)?;
        let (latent0, residual) = self.split_input(z1);
        let (latent_initial, log_det) = self.integrate_latent(&latent0, false)?;
        let combined = self.combine_latent(&latent_initial, residual.as_ref());

        Ok((combined, log_det))
    }

    pub fn log_likelihood(
        &self,
        x: &ArrayView2<f64>,
        base_log_prob: f64,
    ) -> SklResult<Array1<f64>> {
        let (_, log_det_jac) = self.forward_flow(x)?;

        // log p(x) = log p(z) + log |det J|
        // where z = f(x) and J is the Jacobian of the transformation
        let log_likelihood = Array1::from_elem(x.nrows(), base_log_prob) + log_det_jac;

        Ok(log_likelihood)
    }

    pub fn sample(
        &self,
        n_samples: usize,
        base_distribution: &dyn Fn(usize, usize) -> Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        // Sample from base distribution
        let latent_target = base_distribution(n_samples, self.n_components);
        let (latent_source, _) = self.integrate_latent(&latent_target, false)?;

        if self.state.input_dim == self.n_components {
            Ok(latent_source)
        } else {
            let mut combined = Array2::zeros((n_samples, self.state.input_dim));
            combined
                .slice_mut(scirs2_core::ndarray::s![.., 0..self.n_components])
                .assign(&latent_source);
            Ok(combined)
        }
    }
}

impl Estimator for ContinuousNormalizingFlow<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for ContinuousNormalizingFlow<Untrained> {
    type Fitted = ContinuousNormalizingFlow<TrainedCNF>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components should not exceed number of features".to_string(),
            ));
        }

        if self.integration_time <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "integration_time must be positive".to_string(),
            ));
        }

        if self.num_time_steps == 0 {
            return Err(SklearsError::InvalidInput(
                "num_time_steps must be positive".to_string(),
            ));
        }

        let state = self.initialize_cnf_weights(n_features);

        // Note: For a complete implementation, we would need to implement
        // the training loop with maximum likelihood estimation and
        // gradient computation through the CNF.

        Ok(ContinuousNormalizingFlow {
            n_components: self.n_components,
            hidden_layers: self.hidden_layers,
            integration_time: self.integration_time,
            num_time_steps: self.num_time_steps,
            solver_type: self.solver_type,
            trace_estimator: self.trace_estimator,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ContinuousNormalizingFlow<TrainedCNF> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.validate_input_dim(&x.view())?;
        let (latent0, _) = self.split_input(&x.view());
        let (latent_final, _) = self.integrate_latent(&latent0, true)?;
        Ok(latent_final)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_autoencoder_basic() {
        let data = Array2::from_shape_fn((100, 10), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let autoencoder = AutoencoderManifold::new(5)
            .with_epochs(10)
            .with_batch_size(16);

        let fitted = autoencoder.fit(&data, &()).unwrap();
        let encoded = fitted.transform(&data).unwrap();

        assert_eq!(encoded.ncols(), 5);
        assert_eq!(encoded.nrows(), 100);
    }

    #[test]
    fn test_autoencoder_invalid_params() {
        let data = Array2::from_shape_fn((50, 5), |(i, j)| i as f64 + j as f64);
        let autoencoder = AutoencoderManifold::new(10); // n_components > n_features

        assert!(autoencoder.fit(&data, &()).is_err());
    }

    #[test]
    fn test_variational_autoencoder_basic() {
        let vae = VariationalAutoencoder::new(3).with_epochs(5).with_beta(0.5);

        assert_eq!(vae.n_components, 3);
        assert_eq!(vae.beta, 0.5);
    }

    #[test]
    fn test_autoencoder_configuration() {
        let autoencoder = AutoencoderManifold::new(3)
            .with_hidden_layers(vec![64, 32])
            .with_learning_rate(0.01)
            .with_batch_size(64)
            .with_epochs(50);

        assert_eq!(autoencoder.hidden_layers, vec![64, 32]);
        assert_eq!(autoencoder.learning_rate, 0.01);
        assert_eq!(autoencoder.batch_size, 64);
        assert_eq!(autoencoder.epochs, 50);
    }

    #[test]
    fn test_autoencoder_empty_data() {
        let data = Array2::zeros((0, 5));
        let autoencoder = AutoencoderManifold::new(2);

        assert!(autoencoder.fit(&data, &()).is_err());
    }

    #[test]
    fn test_adversarial_autoencoder_basic() {
        let data = Array2::from_shape_fn((100, 10), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let aae = AdversarialAutoencoder::new(5)
            .with_epochs(10)
            .with_batch_size(16)
            .with_adversarial_weight(0.5);

        let fitted = aae.fit(&data, &()).unwrap();
        let encoded = fitted.transform(&data).unwrap();

        assert_eq!(encoded.ncols(), 5);
        assert_eq!(encoded.nrows(), 100);
    }

    #[test]
    fn test_adversarial_autoencoder_configuration() {
        let aae = AdversarialAutoencoder::new(3)
            .with_hidden_layers(vec![64, 32])
            .with_discriminator_layers(vec![32, 16])
            .with_learning_rate(0.01)
            .with_adversarial_weight(2.0)
            .with_prior_type(PriorType::Uniform);

        assert_eq!(aae.hidden_layers, vec![64, 32]);
        assert_eq!(aae.discriminator_layers, vec![32, 16]);
        assert_eq!(aae.learning_rate, 0.01);
        assert_eq!(aae.adversarial_weight, 2.0);
        matches!(aae.prior_type, PriorType::Uniform);
    }

    #[test]
    fn test_adversarial_autoencoder_prior_types() {
        let gaussian_aae = AdversarialAutoencoder::new(3).with_prior_type(PriorType::Gaussian);
        assert!(matches!(gaussian_aae.prior_type, PriorType::Gaussian));

        let uniform_aae = AdversarialAutoencoder::new(3).with_prior_type(PriorType::Uniform);
        assert!(matches!(uniform_aae.prior_type, PriorType::Uniform));

        let categorical_aae =
            AdversarialAutoencoder::new(3).with_prior_type(PriorType::Categorical(5));
        assert!(matches!(
            categorical_aae.prior_type,
            PriorType::Categorical(5)
        ));
    }

    #[test]
    fn test_adversarial_autoencoder_invalid_params() {
        let data = Array2::from_shape_fn((50, 5), |(i, j)| i as f64 + j as f64);
        let aae = AdversarialAutoencoder::new(10); // n_components > n_features

        assert!(aae.fit(&data, &()).is_err());
    }

    #[test]
    fn test_adversarial_autoencoder_empty_data() {
        let data = Array2::zeros((0, 5));
        let aae = AdversarialAutoencoder::new(2);

        assert!(aae.fit(&data, &()).is_err());
    }

    #[test]
    fn test_neural_ode_basic() {
        let data = Array2::from_shape_fn((50, 8), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let node = NeuralODE::new(3)
            .with_integration_time(0.5)
            .with_num_time_steps(5)
            .with_solver_type(ODESolverType::RungeKutta4);

        let fitted = node.fit(&data, &()).unwrap();
        let encoded = fitted.transform(&data).unwrap();

        assert_eq!(encoded.ncols(), 3);
        assert_eq!(encoded.nrows(), 50);
    }

    #[test]
    fn test_neural_ode_configuration() {
        let node = NeuralODE::new(4)
            .with_hidden_layers(vec![32, 16])
            .with_integration_time(2.0)
            .with_num_time_steps(20)
            .with_solver_type(ODESolverType::Euler)
            .with_learning_rate(0.01)
            .with_epochs(50);

        assert_eq!(node.hidden_layers, vec![32, 16]);
        assert_eq!(node.integration_time, 2.0);
        assert_eq!(node.num_time_steps, 20);
        assert!(matches!(node.solver_type, ODESolverType::Euler));
        assert_eq!(node.learning_rate, 0.01);
        assert_eq!(node.epochs, 50);
    }

    #[test]
    fn test_neural_ode_solver_types() {
        let euler_node = NeuralODE::new(2).with_solver_type(ODESolverType::Euler);
        assert!(matches!(euler_node.solver_type, ODESolverType::Euler));

        let rk4_node = NeuralODE::new(2).with_solver_type(ODESolverType::RungeKutta4);
        assert!(matches!(rk4_node.solver_type, ODESolverType::RungeKutta4));
    }

    #[test]
    fn test_neural_ode_invalid_params() {
        let data = Array2::from_shape_fn((30, 5), |(i, j)| i as f64 + j as f64);

        // n_components >= n_features
        let node = NeuralODE::new(10);
        assert!(node.fit(&data, &()).is_err());

        // Invalid integration time
        let node = NeuralODE::new(2).with_integration_time(-1.0);
        assert!(node.fit(&data, &()).is_err());

        // Invalid time steps
        let node = NeuralODE::new(2).with_num_time_steps(0);
        assert!(node.fit(&data, &()).is_err());
    }

    #[test]
    fn test_neural_ode_empty_data() {
        let data = Array2::zeros((0, 5));
        let node = NeuralODE::new(2);

        assert!(node.fit(&data, &()).is_err());
    }

    #[test]
    fn test_neural_ode_manifold_trajectory() {
        let data = Array2::from_shape_fn((10, 6), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let node = NeuralODE::new(2).with_num_time_steps(3);

        let fitted = node.fit(&data, &()).unwrap();
        let trajectory = fitted.get_manifold_trajectory(&data.view()).unwrap();

        // Should have initial state + 3 time steps = 4 total states
        assert_eq!(trajectory.len(), 4);

        // Each state should have the same dimensions
        for state in &trajectory {
            assert_eq!(state.ncols(), 2); // n_components
            assert_eq!(state.nrows(), 10); // n_samples
        }
    }

    #[test]
    fn test_continuous_normalizing_flow_basic() {
        let data = Array2::from_shape_fn((30, 4), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let cnf = ContinuousNormalizingFlow::new(3)
            .with_integration_time(0.5)
            .with_num_time_steps(8)
            .with_trace_estimator(TraceEstimator::Hutchinson);

        let fitted = cnf.fit(&data, &()).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), 30);
    }

    #[test]
    fn test_continuous_normalizing_flow_configuration() {
        let cnf = ContinuousNormalizingFlow::new(2)
            .with_hidden_layers(vec![32, 16, 32])
            .with_integration_time(2.0)
            .with_num_time_steps(25)
            .with_solver_type(ODESolverType::Euler)
            .with_trace_estimator(TraceEstimator::Exact)
            .with_learning_rate(0.005)
            .with_epochs(150);

        assert_eq!(cnf.hidden_layers, vec![32, 16, 32]);
        assert_eq!(cnf.integration_time, 2.0);
        assert_eq!(cnf.num_time_steps, 25);
        assert!(matches!(cnf.solver_type, ODESolverType::Euler));
        assert!(matches!(cnf.trace_estimator, TraceEstimator::Exact));
        assert_eq!(cnf.learning_rate, 0.005);
        assert_eq!(cnf.epochs, 150);
    }

    #[test]
    fn test_cnf_trace_estimator_types() {
        let exact_cnf =
            ContinuousNormalizingFlow::new(2).with_trace_estimator(TraceEstimator::Exact);
        assert!(matches!(exact_cnf.trace_estimator, TraceEstimator::Exact));

        let hutchinson_cnf =
            ContinuousNormalizingFlow::new(2).with_trace_estimator(TraceEstimator::Hutchinson);
        assert!(matches!(
            hutchinson_cnf.trace_estimator,
            TraceEstimator::Hutchinson
        ));

        let rademacher_cnf = ContinuousNormalizingFlow::new(2)
            .with_trace_estimator(TraceEstimator::RademacherRandom);
        assert!(matches!(
            rademacher_cnf.trace_estimator,
            TraceEstimator::RademacherRandom
        ));
    }

    #[test]
    fn test_cnf_forward_backward_flow() {
        let data = Array2::from_shape_fn((5, 3), |(i, j)| i as f64 + j as f64);
        let cnf = ContinuousNormalizingFlow::new(3)
            .with_num_time_steps(5)
            .with_integration_time(1.0);

        let fitted = cnf.fit(&data, &()).unwrap();

        // Test forward flow
        let (z, log_det_forward) = fitted.forward_flow(&data.view()).unwrap();
        assert_eq!(z.dim(), data.dim());
        assert_eq!(log_det_forward.len(), data.nrows());

        // Test backward flow
        let (x_reconstructed, log_det_backward) = fitted.backward_flow(&z.view()).unwrap();
        assert_eq!(x_reconstructed.dim(), data.dim());
        assert_eq!(log_det_backward.len(), data.nrows());

        // Check that forward and backward flows are approximately inverses
        let reconstruction_error = (&data - &x_reconstructed).mapv(|x| x.abs()).sum();
        assert!(reconstruction_error < 1.0); // Allow some numerical error
    }

    #[test]
    fn test_cnf_log_likelihood() {
        let data = Array2::from_shape_fn((10, 2), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let cnf = ContinuousNormalizingFlow::new(2).with_num_time_steps(5);

        let fitted = cnf.fit(&data, &()).unwrap();
        let log_likelihood = fitted.log_likelihood(&data.view(), 0.0).unwrap();

        assert_eq!(log_likelihood.len(), data.nrows());
        // Check that all likelihood values are finite
        for &ll in log_likelihood.iter() {
            assert!(ll.is_finite());
        }
    }

    #[test]
    fn test_cnf_sampling() {
        let data = Array2::from_shape_fn((20, 3), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let cnf = ContinuousNormalizingFlow::new(3).with_num_time_steps(5);

        let fitted = cnf.fit(&data, &()).unwrap();

        // Define a simple Gaussian base distribution
        let gaussian_sampler = |n_samples: usize, n_dims: usize| -> Array2<f64> {
            let mut rng = thread_rng();
            Array2::from_shape_fn((n_samples, n_dims), |(_, _)| {
                Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
            })
        };

        let samples = fitted.sample(15, &gaussian_sampler).unwrap();
        assert_eq!(samples.nrows(), 15);
        assert_eq!(samples.ncols(), 3);

        // Check that samples are finite
        for &val in samples.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_cnf_invalid_params() {
        let data = Array2::from_shape_fn((25, 4), |(i, j)| i as f64 + j as f64);

        // n_components > n_features
        let cnf = ContinuousNormalizingFlow::new(6);
        assert!(cnf.fit(&data, &()).is_err());

        // Invalid integration time
        let cnf = ContinuousNormalizingFlow::new(2).with_integration_time(-0.5);
        assert!(cnf.fit(&data, &()).is_err());

        // Invalid time steps
        let cnf = ContinuousNormalizingFlow::new(2).with_num_time_steps(0);
        assert!(cnf.fit(&data, &()).is_err());
    }

    #[test]
    fn test_cnf_empty_data() {
        let data = Array2::zeros((0, 4));
        let cnf = ContinuousNormalizingFlow::new(2);

        assert!(cnf.fit(&data, &()).is_err());
    }
}
