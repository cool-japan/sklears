use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;

pub struct CNNFeatureExtractor {
    pub filters: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub activation: CNNActivation,
    pub pooling_size: usize,
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub regularization: f64,
    pub random_state: Option<u64>,
    pub dropout_rate: f64,
    pub use_batch_norm: bool,
}

impl CNNFeatureExtractor {
    pub fn new() -> Self {
        Self {
            filters: vec![32, 64, 128],
            kernel_sizes: vec![3, 3, 3],
            strides: vec![1, 1, 1],
            activation: CNNActivation::ReLU,
            pooling_size: 2,
            learning_rate: 0.001,
            n_epochs: 50,
            batch_size: 32,
            regularization: 0.01,
            random_state: None,
            dropout_rate: 0.2,
            use_batch_norm: true,
        }
    }

    pub fn filters(mut self, filters: Vec<usize>) -> Self {
        self.filters = filters;
        self
    }

    pub fn kernel_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.kernel_sizes = sizes;
        self
    }

    pub fn strides(mut self, strides: Vec<usize>) -> Self {
        self.strides = strides;
        self
    }

    pub fn activation(mut self, activation: CNNActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn pooling_size(mut self, size: usize) -> Self {
        self.pooling_size = size;
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

    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn use_batch_norm(mut self, use_norm: bool) -> Self {
        self.use_batch_norm = use_norm;
        self
    }

    fn initialize_filters(&self, input_channels: usize, output_channels: usize, kernel_size: usize) -> Array4<f64> {
        let mut rng = match self.random_state {
            Some(seed) => Random::seed_from_u64(seed),
            None => Random::seed_from_u64(42),
        };

        let fan_in = input_channels * kernel_size * kernel_size;
        let fan_out = output_channels * kernel_size * kernel_size;
        let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();

        let mut filters = Array4::zeros((output_channels, input_channels, kernel_size, kernel_size));

        for i in 0..output_channels {
            for j in 0..input_channels {
                for k in 0..kernel_size {
                    for l in 0..kernel_size {
                        filters[(i, j, k, l)] = rng.gen_range(-scale..scale);
                    }
                }
            }
        }

        filters
    }

    fn convolve2d(&self, input: &Array3<f64>, filters: &Array4<f64>, stride: usize) -> Array3<f64> {
        let (input_height, input_width, input_channels) = input.dim();
        let (num_filters, _, kernel_height, kernel_width) = filters.dim();

        let output_height = (input_height - kernel_height) / stride + 1;
        let output_width = (input_width - kernel_width) / stride + 1;

        let mut output = Array3::zeros((output_height, output_width, num_filters));

        for f in 0..num_filters {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = 0.0;

                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            for c in 0..input_channels {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                if ih < input_height && iw < input_width {
                                    sum += input[(ih, iw, c)] * filters[(f, c, kh, kw)];
                                }
                            }
                        }
                    }

                    output[(oh, ow, f)] = self.activation.apply(sum);
                }
            }
        }

        output
    }

    fn max_pool2d(&self, input: &Array3<f64>, pool_size: usize) -> Array3<f64> {
        let (input_height, input_width, channels) = input.dim();
        let output_height = input_height / pool_size;
        let output_width = input_width / pool_size;

        let mut output = Array3::zeros((output_height, output_width, channels));

        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut max_val = f64::NEG_INFINITY;

                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * pool_size + ph;
                            let iw = ow * pool_size + pw;

                            if ih < input_height && iw < input_width {
                                max_val = max_val.max(input[(ih, iw, c)]);
                            }
                        }
                    }

                    output[(oh, ow, c)] = max_val;
                }
            }
        }

        output
    }

    fn batch_normalize(&self, input: &Array3<f64>) -> Array3<f64> {
        if !self.use_batch_norm {
            return input.clone();
        }

        let (height, width, channels) = input.dim();
        let mut normalized = Array3::zeros((height, width, channels));

        for c in 0..channels {
            let mut channel_values = Vec::new();

            for h in 0..height {
                for w in 0..width {
                    channel_values.push(input[(h, w, c)]);
                }
            }

            let mean = channel_values.iter().sum::<f64>() / channel_values.len() as f64;
            let variance = channel_values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / channel_values.len() as f64;
            let std_dev = (variance + 1e-8).sqrt();

            for h in 0..height {
                for w in 0..width {
                    normalized[(h, w, c)] = (input[(h, w, c)] - mean) / std_dev;
                }
            }
        }

        normalized
    }

    fn apply_dropout(&self, input: &Array3<f64>) -> Array3<f64> {
        let mut rng = Random::seed_from_u64(42);
        let mut result = input.clone();

        for elem in result.iter_mut() {
            if rng.gen_range(0.0..1.0) < self.dropout_rate {
                *elem = 0.0;
            } else {
                *elem /= 1.0 - self.dropout_rate;
            }
        }

        result
    }

    fn global_average_pool(&self, input: &Array3<f64>) -> Array1<f64> {
        let (height, width, channels) = input.dim();
        let mut output = Array1::zeros(channels);

        for c in 0..channels {
            let mut sum = 0.0;
            for h in 0..height {
                for w in 0..width {
                    sum += input[(h, w, c)];
                }
            }
            output[c] = sum / (height * width) as f64;
        }

        output
    }
}

pub struct FittedCNNFeatureExtractor {
    extractor: CNNFeatureExtractor,
    conv_filters: Vec<Array4<f64>>,
    input_shape: (usize, usize, usize),
    output_dim: usize,
}

impl Estimator<Untrained> for CNNFeatureExtractor {
    fn estimator_type(&self) -> &'static str {
        "CNNFeatureExtractor"
    }

    fn complexity(&self) -> f64 {
        self.filters.iter().sum::<usize>() as f64
    }
}

impl Fit<Array3<f64>, ()> for CNNFeatureExtractor {
    type Output = FittedCNNFeatureExtractor;

    fn fit(&self, x: &Array3<f64>, _y: &()) -> SklResult<Self::Output> {
        let (height, width, channels) = x.dim();

        if height == 0 || width == 0 || channels == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.filters.len() != self.kernel_sizes.len() || self.filters.len() != self.strides.len() {
            return Err(SklearsError::InvalidInput("Mismatched filter configuration".to_string()));
        }

        let mut conv_filters = Vec::new();
        let mut current_channels = channels;

        for (i, &num_filters) in self.filters.iter().enumerate() {
            let kernel_size = self.kernel_sizes[i];
            let filters = self.initialize_filters(current_channels, num_filters, kernel_size);
            conv_filters.push(filters);
            current_channels = num_filters;
        }

        let mut current_shape = (height, width, channels);
        for (i, _) in self.filters.iter().enumerate() {
            let stride = self.strides[i];
            let kernel_size = self.kernel_sizes[i];

            current_shape.0 = (current_shape.0 - kernel_size) / stride + 1;
            current_shape.1 = (current_shape.1 - kernel_size) / stride + 1;
            current_shape.2 = self.filters[i];

            current_shape.0 /= self.pooling_size;
            current_shape.1 /= self.pooling_size;
        }

        let output_dim = current_shape.2;

        Ok(FittedCNNFeatureExtractor {
            extractor: self.clone(),
            conv_filters,
            input_shape: (height, width, channels),
            output_dim,
        })
    }
}

impl Transform<Array3<f64>, Array1<f64>> for FittedCNNFeatureExtractor {
    fn transform(&self, x: &Array3<f64>) -> SklResult<Array1<f64>> {
        let (height, width, channels) = x.dim();

        if (height, width, channels) != self.input_shape {
            return Err(SklearsError::InvalidInput(format!(
                "Input shape {:?} does not match fitted shape {:?}",
                (height, width, channels), self.input_shape
            )));
        }

        let mut current = x.clone();

        for (i, filters) in self.conv_filters.iter().enumerate() {
            let stride = self.extractor.strides[i];

            current = self.extractor.convolve2d(&current, filters, stride);

            if self.extractor.use_batch_norm {
                current = self.extractor.batch_normalize(&current);
            }

            current = self.extractor.max_pool2d(&current, self.extractor.pooling_size);

            if i == self.conv_filters.len() - 1 {
                current = self.extractor.apply_dropout(&current);
            }
        }

        Ok(self.extractor.global_average_pool(&current))
    }
}

impl FittedCNNFeatureExtractor {
    pub fn get_feature_maps(&self, x: &Array3<f64>, layer: usize) -> SklResult<Array3<f64>> {
        if layer >= self.conv_filters.len() {
            return Err(SklearsError::InvalidInput("Layer index out of bounds".to_string()));
        }

        let mut current = x.clone();

        for i in 0..=layer {
            let filters = &self.conv_filters[i];
            let stride = self.extractor.strides[i];

            current = self.extractor.convolve2d(&current, filters, stride);

            if self.extractor.use_batch_norm {
                current = self.extractor.batch_normalize(&current);
            }

            current = self.extractor.max_pool2d(&current, self.extractor.pooling_size);
        }

        Ok(current)
    }

    pub fn get_output_shape(&self) -> usize {
        self.output_dim
    }

    pub fn get_num_layers(&self) -> usize {
        self.conv_filters.len()
    }
}