use super::neural_types::*;
use super::neural_activations::*;

pub struct DenseLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub activation: Option<CNNActivation>,
    pub use_bias: bool,
}

impl DenseLayer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Option<CNNActivation>) -> Self {
        let mut rng = Random::seed_from_u64(42);
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let mut weights = Array2::zeros((input_dim, output_dim));
        for i in 0..input_dim {
            for j in 0..output_dim {
                weights[(i, j)] = rng.gen_range(-scale..scale);
            }
        }

        Self {
            input_dim,
            output_dim,
            weights,
            bias: Array1::zeros(output_dim),
            activation,
            use_bias: true,
        }
    }

    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        if !use_bias {
            self.bias = Array1::zeros(self.output_dim);
        }
        self
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let linear_output = input.dot(&self.weights);

        let output_with_bias = if self.use_bias {
            &linear_output + &self.bias
        } else {
            linear_output
        };

        if let Some(activation) = &self.activation {
            self.apply_activation(&output_with_bias, activation)
        } else {
            output_with_bias
        }
    }

    fn apply_activation(&self, input: &Array2<f64>, activation: &CNNActivation) -> Array2<f64> {
        let mut result = Array2::zeros(input.raw_dim());

        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                result[(i, j)] = activation.apply(input[(i, j)]);
            }
        }

        result
    }

    pub fn update_weights(&mut self, weight_updates: &Array2<f64>, bias_updates: &Array1<f64>, learning_rate: f64) {
        self.weights = &self.weights - learning_rate * weight_updates;
        if self.use_bias {
            self.bias = &self.bias - learning_rate * bias_updates;
        }
    }

    pub fn get_parameters(&self) -> (Array2<f64>, Array1<f64>) {
        (self.weights.clone(), self.bias.clone())
    }

    pub fn set_parameters(&mut self, weights: Array2<f64>, bias: Array1<f64>) {
        self.weights = weights;
        self.bias = bias;
    }
}

pub struct Conv2DLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Array4<f64>,
    pub bias: Array1<f64>,
    pub activation: Option<CNNActivation>,
    pub use_bias: bool,
}

impl Conv2DLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Option<CNNActivation>,
    ) -> Self {
        let mut rng = Random::seed_from_u64(42);
        let fan_in = in_channels * kernel_size * kernel_size;
        let fan_out = out_channels * kernel_size * kernel_size;
        let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();

        let mut weights = Array4::zeros((out_channels, in_channels, kernel_size, kernel_size));
        for i in 0..out_channels {
            for j in 0..in_channels {
                for k in 0..kernel_size {
                    for l in 0..kernel_size {
                        weights[(i, j, k, l)] = rng.gen_range(-scale..scale);
                    }
                }
            }
        }

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias: Array1::zeros(out_channels),
            activation,
            use_bias: true,
        }
    }

    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let padded_input = self.apply_padding(input);
        let convolved = self.convolve(&padded_input);

        if let Some(activation) = &self.activation {
            self.apply_activation(&convolved, activation)
        } else {
            convolved
        }
    }

    fn apply_padding(&self, input: &Array3<f64>) -> Array3<f64> {
        if self.padding == 0 {
            return input.clone();
        }

        let (height, width, channels) = input.dim();
        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        let mut padded = Array3::zeros((padded_height, padded_width, channels));

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    padded[(h + self.padding, w + self.padding, c)] = input[(h, w, c)];
                }
            }
        }

        padded
    }

    fn convolve(&self, input: &Array3<f64>) -> Array3<f64> {
        let (input_height, input_width, _) = input.dim();
        let output_height = (input_height - self.kernel_size) / self.stride + 1;
        let output_width = (input_width - self.kernel_size) / self.stride + 1;

        let mut output = Array3::zeros((output_height, output_width, self.out_channels));

        for f in 0..self.out_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = if self.use_bias { self.bias[f] } else { 0.0 };

                    for kh in 0..self.kernel_size {
                        for kw in 0..self.kernel_size {
                            for c in 0..self.in_channels {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;
                                sum += input[(ih, iw, c)] * self.weights[(f, c, kh, kw)];
                            }
                        }
                    }

                    output[(oh, ow, f)] = sum;
                }
            }
        }

        output
    }

    fn apply_activation(&self, input: &Array3<f64>, activation: &CNNActivation) -> Array3<f64> {
        let mut result = Array3::zeros(input.raw_dim());

        for i in 0..input.dim().0 {
            for j in 0..input.dim().1 {
                for k in 0..input.dim().2 {
                    result[(i, j, k)] = activation.apply(input[(i, j, k)]);
                }
            }
        }

        result
    }
}

pub struct MaxPool2DLayer {
    pub pool_size: usize,
    pub stride: usize,
}

impl MaxPool2DLayer {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size, stride }
    }

    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (input_height, input_width, channels) = input.dim();
        let output_height = (input_height - self.pool_size) / self.stride + 1;
        let output_width = (input_width - self.pool_size) / self.stride + 1;

        let mut output = Array3::zeros((output_height, output_width, channels));

        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut max_val = f64::NEG_INFINITY;

                    for ph in 0..self.pool_size {
                        for pw in 0..self.pool_size {
                            let ih = oh * self.stride + ph;
                            let iw = ow * self.stride + pw;

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
}

pub struct AvgPool2DLayer {
    pub pool_size: usize,
    pub stride: usize,
}

impl AvgPool2DLayer {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size, stride }
    }

    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (input_height, input_width, channels) = input.dim();
        let output_height = (input_height - self.pool_size) / self.stride + 1;
        let output_width = (input_width - self.pool_size) / self.stride + 1;

        let mut output = Array3::zeros((output_height, output_width, channels));

        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = 0.0;
                    let mut count = 0;

                    for ph in 0..self.pool_size {
                        for pw in 0..self.pool_size {
                            let ih = oh * self.stride + ph;
                            let iw = ow * self.stride + pw;

                            if ih < input_height && iw < input_width {
                                sum += input[(ih, iw, c)];
                                count += 1;
                            }
                        }
                    }

                    output[(oh, ow, c)] = if count > 0 { sum / count as f64 } else { 0.0 };
                }
            }
        }

        output
    }
}

pub struct DropoutLayer {
    pub dropout_rate: f64,
    pub training: bool,
}

impl DropoutLayer {
    pub fn new(dropout_rate: f64) -> Self {
        Self {
            dropout_rate,
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward_2d(&self, input: &Array2<f64>) -> Array2<f64> {
        if !self.training || self.dropout_rate == 0.0 {
            return input.clone();
        }

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

    pub fn forward_3d(&self, input: &Array3<f64>) -> Array3<f64> {
        if !self.training || self.dropout_rate == 0.0 {
            return input.clone();
        }

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
}

pub struct BatchNormLayer {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub affine: bool,
    pub running_mean: Array1<f64>,
    pub running_var: Array1<f64>,
    pub weight: Array1<f64>,
    pub bias: Array1<f64>,
    pub training: bool,
}

impl BatchNormLayer {
    pub fn new(num_features: usize, eps: f64, momentum: f64, affine: bool) -> Self {
        Self {
            num_features,
            eps,
            momentum,
            affine,
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            weight: Array1::ones(num_features),
            bias: Array1::zeros(num_features),
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward_2d(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let (batch_size, features) = input.dim();

        if features != self.num_features {
            panic!("Input features {} do not match layer features {}", features, self.num_features);
        }

        let (mean, var) = if self.training {
            let mean = self.compute_mean_2d(input);
            let var = self.compute_variance_2d(input, &mean);

            self.update_running_stats(&mean, &var);
            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        let mut normalized = Array2::zeros((batch_size, features));

        for i in 0..batch_size {
            for j in 0..features {
                let normalized_val = (input[(i, j)] - mean[j]) / (var[j] + self.eps).sqrt();
                normalized[(i, j)] = if self.affine {
                    self.weight[j] * normalized_val + self.bias[j]
                } else {
                    normalized_val
                };
            }
        }

        normalized
    }

    pub fn forward_3d(&mut self, input: &Array3<f64>) -> Array3<f64> {
        let (height, width, channels) = input.dim();

        if channels != self.num_features {
            panic!("Input channels {} do not match layer features {}", channels, self.num_features);
        }

        let (mean, var) = if self.training {
            let mean = self.compute_mean_3d(input);
            let var = self.compute_variance_3d(input, &mean);

            self.update_running_stats(&mean, &var);
            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        let mut normalized = Array3::zeros((height, width, channels));

        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let normalized_val = (input[(h, w, c)] - mean[c]) / (var[c] + self.eps).sqrt();
                    normalized[(h, w, c)] = if self.affine {
                        self.weight[c] * normalized_val + self.bias[c]
                    } else {
                        normalized_val
                    };
                }
            }
        }

        normalized
    }

    fn compute_mean_2d(&self, input: &Array2<f64>) -> Array1<f64> {
        let (batch_size, features) = input.dim();
        let mut mean = Array1::zeros(features);

        for j in 0..features {
            let mut sum = 0.0;
            for i in 0..batch_size {
                sum += input[(i, j)];
            }
            mean[j] = sum / batch_size as f64;
        }

        mean
    }

    fn compute_variance_2d(&self, input: &Array2<f64>, mean: &Array1<f64>) -> Array1<f64> {
        let (batch_size, features) = input.dim();
        let mut var = Array1::zeros(features);

        for j in 0..features {
            let mut sum_sq_diff = 0.0;
            for i in 0..batch_size {
                let diff = input[(i, j)] - mean[j];
                sum_sq_diff += diff * diff;
            }
            var[j] = sum_sq_diff / batch_size as f64;
        }

        var
    }

    fn compute_mean_3d(&self, input: &Array3<f64>) -> Array1<f64> {
        let (height, width, channels) = input.dim();
        let mut mean = Array1::zeros(channels);

        for c in 0..channels {
            let mut sum = 0.0;
            for h in 0..height {
                for w in 0..width {
                    sum += input[(h, w, c)];
                }
            }
            mean[c] = sum / (height * width) as f64;
        }

        mean
    }

    fn compute_variance_3d(&self, input: &Array3<f64>, mean: &Array1<f64>) -> Array1<f64> {
        let (height, width, channels) = input.dim();
        let mut var = Array1::zeros(channels);

        for c in 0..channels {
            let mut sum_sq_diff = 0.0;
            for h in 0..height {
                for w in 0..width {
                    let diff = input[(h, w, c)] - mean[c];
                    sum_sq_diff += diff * diff;
                }
            }
            var[c] = sum_sq_diff / (height * width) as f64;
        }

        var
    }

    fn update_running_stats(&mut self, batch_mean: &Array1<f64>, batch_var: &Array1<f64>) {
        for i in 0..self.num_features {
            self.running_mean[i] = (1.0 - self.momentum) * self.running_mean[i] + self.momentum * batch_mean[i];
            self.running_var[i] = (1.0 - self.momentum) * self.running_var[i] + self.momentum * batch_var[i];
        }
    }
}