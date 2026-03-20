use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;

pub struct AutoencoderFeatureExtractor {
    pub encoding_dim: usize,
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub regularization: f64,
    pub random_state: Option<u64>,
    pub use_sigmoid: bool,
    pub tolerance: f64,
}

impl AutoencoderFeatureExtractor {
    pub fn new() -> Self {
        Self {
            encoding_dim: 32,
            learning_rate: 0.01,
            n_epochs: 100,
            batch_size: 32,
            regularization: 0.01,
            random_state: None,
            use_sigmoid: true,
            tolerance: 1e-6,
        }
    }

    pub fn encoding_dim(mut self, dim: usize) -> Self {
        self.encoding_dim = dim;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn use_sigmoid(mut self, use_sig: bool) -> Self {
        self.use_sigmoid = use_sig;
        self
    }

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    fn sigmoid(&self, x: f64) -> f64 {
        if self.use_sigmoid {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.max(0.0)
        }
    }

    fn activation_derivative(&self, x: f64) -> f64 {
        if self.use_sigmoid {
            let sig = self.sigmoid(x);
            sig * (1.0 - sig)
        } else {
            if x > 0.0 { 1.0 } else { 0.0 }
        }
    }

    fn initialize_weights(&self, input_dim: usize, output_dim: usize) -> Array2<f64> {
        let mut rng = match self.random_state {
            Some(seed) => Random::seed_from_u64(seed),
            None => Random::seed_from_u64(42),
        };

        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let mut weights = Array2::zeros((input_dim, output_dim));

        for i in 0..input_dim {
            for j in 0..output_dim {
                weights[(i, j)] = rng.random_range(0.0..1.0) * scale - scale / 2.0;
            }
        }

        weights
    }

    fn encode(
        &self,
        x: &Array2<f64>,
        encoder_weights: &Array2<f64>,
        encoder_bias: &Array1<f64>,
    ) -> Array2<f64> {
        let linear = x.dot(encoder_weights) + encoder_bias;
        let mut encoded = Array2::zeros(linear.raw_dim());

        for i in 0..linear.nrows() {
            for j in 0..linear.ncols() {
                encoded[(i, j)] = self.sigmoid(linear[(i, j)]);
            }
        }

        encoded
    }

    fn decode(
        &self,
        encoded: &Array2<f64>,
        decoder_weights: &Array2<f64>,
        decoder_bias: &Array1<f64>,
    ) -> Array2<f64> {
        let linear = encoded.dot(decoder_weights) + decoder_bias;
        let mut decoded = Array2::zeros(linear.raw_dim());

        for i in 0..linear.nrows() {
            for j in 0..linear.ncols() {
                decoded[(i, j)] = self.sigmoid(linear[(i, j)]);
            }
        }

        decoded
    }

    fn compute_loss(
        &self,
        original: &Array2<f64>,
        reconstructed: &Array2<f64>,
        encoder_weights: &Array2<f64>,
        decoder_weights: &Array2<f64>,
    ) -> f64 {
        let diff = original - reconstructed;
        let mse = diff.iter().map(|&x| x * x).sum::<f64>() / (original.len() as f64);

        let l2_reg = self.regularization * (
            encoder_weights.iter().map(|&x| x * x).sum::<f64>() +
            decoder_weights.iter().map(|&x| x * x).sum::<f64>()
        );

        mse + l2_reg
    }
}

pub struct FittedAutoencoderFeatureExtractor {
    extractor: AutoencoderFeatureExtractor,
    encoder_weights: Array2<f64>,
    encoder_bias: Array1<f64>,
    decoder_weights: Array2<f64>,
    decoder_bias: Array1<f64>,
    input_dim: usize,
}

impl Estimator<Untrained> for AutoencoderFeatureExtractor {
    fn estimator_type(&self) -> &'static str {
        "AutoencoderFeatureExtractor"
    }

    fn complexity(&self) -> f64 {
        self.encoding_dim as f64
    }
}

impl Fit<Array2<f64>, ()> for AutoencoderFeatureExtractor {
    type Output = FittedAutoencoderFeatureExtractor;

    fn fit(&self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Output> {
        let (n_samples, input_dim) = x.dim();

        if n_samples == 0 || input_dim == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let mut encoder_weights = self.initialize_weights(input_dim, self.encoding_dim);
        let mut encoder_bias = Array1::zeros(self.encoding_dim);
        let mut decoder_weights = self.initialize_weights(self.encoding_dim, input_dim);
        let mut decoder_bias = Array1::zeros(input_dim);

        // Simplified training loop
        for epoch in 0..self.n_epochs {
            let encoded = self.encode(x, &encoder_weights, &encoder_bias);
            let decoded = self.decode(&encoded, &decoder_weights, &decoder_bias);

            let loss = self.compute_loss(x, &decoded, &encoder_weights, &decoder_weights);

            if loss < self.tolerance {
                break;
            }

            // Simplified gradient updates (placeholder)
            let lr = self.learning_rate;
            let decay = 0.999;

            encoder_weights *= decay;
            decoder_weights *= decay;
        }

        Ok(FittedAutoencoderFeatureExtractor {
            extractor: self.clone(),
            encoder_weights,
            encoder_bias,
            decoder_weights,
            decoder_bias,
            input_dim,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedAutoencoderFeatureExtractor {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, input_dim) = x.dim();

        if input_dim != self.input_dim {
            return Err(SklearsError::InvalidInput(format!(
                "Input dimension {} does not match fitted dimension {}",
                input_dim, self.input_dim
            )));
        }

        if n_samples == 0 {
            return Ok(Array2::zeros((0, self.extractor.encoding_dim)));
        }

        Ok(self.extractor.encode(x, &self.encoder_weights, &self.encoder_bias))
    }
}

impl FittedAutoencoderFeatureExtractor {
    pub fn reconstruct(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let encoded = self.transform(x)?;
        Ok(self.extractor.decode(&encoded, &self.decoder_weights, &self.decoder_bias))
    }

    pub fn reconstruction_error(&self, x: &Array2<f64>) -> SklResult<f64> {
        let reconstructed = self.reconstruct(x)?;
        let diff = x - &reconstructed;
        let mse = diff.iter().map(|&x| x * x).sum::<f64>() / (x.len() as f64);
        Ok(mse)
    }

    pub fn get_decoder_weights(&self) -> &Array2<f64> {
        &self.decoder_weights
    }

    pub fn get_encoder_weights(&self) -> &Array2<f64> {
        &self.encoder_weights
    }
}