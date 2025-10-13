//! Convolutional layers for neural networks.
//!
//! This module provides 1D and 2D convolution layers for building
//! convolutional neural networks (CNNs).

use crate::{activation::Activation, NeuralResult};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Array5, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, Rng};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;

/// Helper function to apply activation to a scalar value
fn apply_activation_scalar<T: FloatBounds>(activation: &Activation, x: T) -> T {
    match activation {
        Activation::Identity => x,
        Activation::Logistic => {
            let exp_neg = (-x).exp();
            T::one() / (T::one() + exp_neg)
        }
        Activation::Tanh => x.tanh(),
        Activation::Relu => x.max(T::zero()),
        _ => x, // For other activations, just return input for now
    }
}

/// Padding strategy for convolution operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// No padding (output size decreases)
    Valid,
    /// Padding to maintain input size
    Same,
    /// Custom padding amount
    Custom(usize),
}

/// 1D Convolution Layer
#[derive(Debug, Clone)]
pub struct Conv1D {
    /// Number of output filters/channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride for convolution
    pub stride: usize,
    /// Padding strategy
    pub padding: Padding,
    /// Dilation rate
    pub dilation: usize,
    /// Activation function
    pub activation: Option<Activation>,
    /// Whether to use bias
    pub use_bias: bool,

    // Learnable parameters
    /// Weights: (out_channels, in_channels, kernel_size)
    pub weights: Option<Array3<f64>>,
    /// Biases: (out_channels,)
    pub biases: Option<Array1<f64>>,

    // Cached values for backpropagation
    last_input: Option<Array2<f64>>,
    last_output: Option<Array2<f64>>,
}

impl Conv1D {
    pub fn new(
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: Padding,
        dilation: usize,
        activation: Option<Activation>,
        use_bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
            weights: None,
            biases: None,
            last_input: None,
            last_output: None,
        }
    }

    /// Initialize weights and biases
    pub fn initialize(&mut self, in_channels: usize, rng: &mut impl Rng) -> NeuralResult<()> {
        // He initialization for ReLU-like activations
        let fan_in = in_channels * self.kernel_size;
        let std = (2.0 / fan_in as f64).sqrt();

        // Initialize weights
        let mut weights = Array3::zeros((self.out_channels, in_channels, self.kernel_size));
        for weight in weights.iter_mut() {
            let normal_dist = Normal::new(0.0, std).unwrap();
            *weight = rng.sample(normal_dist);
        }
        self.weights = Some(weights);

        // Initialize biases
        if self.use_bias {
            self.biases = Some(Array1::zeros(self.out_channels));
        }

        Ok(())
    }

    /// Calculate output size given input size
    pub fn output_size(&self, input_size: usize) -> usize {
        let effective_kernel_size = self.dilation * (self.kernel_size - 1) + 1;
        let padding_amount = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (effective_kernel_size - 1) / 2,
            Padding::Custom(p) => p,
        };

        (input_size + 2 * padding_amount - effective_kernel_size) / self.stride + 1
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array2<f64>) -> NeuralResult<Array2<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "forward".to_string(),
            })?;

        let (batch_size, in_channels, input_length) = if input.ndim() == 2 {
            (1, input.nrows(), input.ncols())
        } else {
            return Err(SklearsError::ShapeMismatch {
                expected: "2D array (channels, length) or 3D array (batch, channels, length)"
                    .to_string(),
                actual: format!("{}D array", input.ndim()),
            });
        };

        let output_length = self.output_size(input_length);
        let mut output = Array2::zeros((self.out_channels, output_length));

        // Apply padding
        let padded_input = self.apply_padding(input)?;

        // Convolution operation
        for out_ch in 0..self.out_channels {
            for i in 0..output_length {
                let mut sum = 0.0;

                for in_ch in 0..in_channels {
                    for k in 0..self.kernel_size {
                        let input_idx = i * self.stride + k * self.dilation;
                        if input_idx < padded_input.ncols() {
                            sum += weights[[out_ch, in_ch, k]] * padded_input[[in_ch, input_idx]];
                        }
                    }
                }

                // Add bias if enabled
                if let Some(ref biases) = self.biases {
                    sum += biases[out_ch];
                }

                output[[out_ch, i]] = sum;
            }
        }

        // Apply activation function
        if let Some(ref activation) = self.activation {
            output.mapv_inplace(|x| apply_activation_scalar(activation, x));
        }

        // Cache for backpropagation
        self.last_input = Some(input.clone());
        self.last_output = Some(output.clone());

        Ok(output)
    }

    /// Apply padding to input
    fn apply_padding(&self, input: &Array2<f64>) -> NeuralResult<Array2<f64>> {
        let padding_amount = match self.padding {
            Padding::Valid => return Ok(input.clone()),
            Padding::Same => {
                let effective_kernel_size = self.dilation * (self.kernel_size - 1) + 1;
                (effective_kernel_size - 1) / 2
            }
            Padding::Custom(p) => p,
        };

        if padding_amount == 0 {
            return Ok(input.clone());
        }

        let (in_channels, input_length) = input.dim();
        let padded_length = input_length + 2 * padding_amount;
        let mut padded = Array2::zeros((in_channels, padded_length));

        // Copy original data to center
        for ch in 0..in_channels {
            for i in 0..input_length {
                padded[[ch, i + padding_amount]] = input[[ch, i]];
            }
        }

        Ok(padded)
    }
}

/// 2D Convolution Layer
#[derive(Debug, Clone)]
pub struct Conv2D {
    /// Number of output filters/channels
    pub out_channels: usize,
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding strategy
    pub padding: Padding,
    /// Dilation rate (height, width)
    pub dilation: (usize, usize),
    /// Activation function
    pub activation: Option<Activation>,
    /// Whether to use bias
    pub use_bias: bool,

    // Learnable parameters
    /// Weights: (out_channels, in_channels, kernel_height, kernel_width)
    pub weights: Option<Array4<f64>>,
    /// Biases: (out_channels,)
    pub biases: Option<Array1<f64>>,

    // Cached values for backpropagation
    last_input: Option<Array3<f64>>,
    last_output: Option<Array3<f64>>,
}

impl Conv2D {
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: Padding,
        dilation: (usize, usize),
        activation: Option<Activation>,
        use_bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
            weights: None,
            biases: None,
            last_input: None,
            last_output: None,
        }
    }

    /// Initialize weights and biases
    pub fn initialize(&mut self, in_channels: usize, rng: &mut impl Rng) -> NeuralResult<()> {
        // He initialization
        let fan_in = in_channels * self.kernel_size.0 * self.kernel_size.1;
        let std = (2.0 / fan_in as f64).sqrt();

        // Initialize weights
        let mut weights = Array4::zeros((
            self.out_channels,
            in_channels,
            self.kernel_size.0,
            self.kernel_size.1,
        ));

        for weight in weights.iter_mut() {
            let normal_dist = Normal::new(0.0, std).unwrap();
            *weight = rng.sample(normal_dist);
        }
        self.weights = Some(weights);

        // Initialize biases
        if self.use_bias {
            self.biases = Some(Array1::zeros(self.out_channels));
        }

        Ok(())
    }

    /// Calculate output size given input size
    pub fn output_size(&self, input_size: (usize, usize)) -> (usize, usize) {
        let effective_kernel_h = self.dilation.0 * (self.kernel_size.0 - 1) + 1;
        let effective_kernel_w = self.dilation.1 * (self.kernel_size.1 - 1) + 1;

        let padding_h = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (effective_kernel_h - 1) / 2,
            Padding::Custom(p) => p,
        };

        let padding_w = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (effective_kernel_w - 1) / 2,
            Padding::Custom(p) => p,
        };

        let output_h = (input_size.0 + 2 * padding_h - effective_kernel_h) / self.stride.0 + 1;
        let output_w = (input_size.1 + 2 * padding_w - effective_kernel_w) / self.stride.1 + 1;

        (output_h, output_w)
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array3<f64>) -> NeuralResult<Array3<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "forward".to_string(),
            })?;

        let (in_channels, input_h, input_w) = input.dim();
        let (output_h, output_w) = self.output_size((input_h, input_w));
        let mut output = Array3::zeros((self.out_channels, output_h, output_w));

        // Apply padding
        let padded_input = self.apply_padding_2d(input)?;

        // Convolution operation
        for out_ch in 0..self.out_channels {
            for out_y in 0..output_h {
                for out_x in 0..output_w {
                    let mut sum = 0.0;

                    for in_ch in 0..in_channels {
                        for ky in 0..self.kernel_size.0 {
                            for kx in 0..self.kernel_size.1 {
                                let input_y = out_y * self.stride.0 + ky * self.dilation.0;
                                let input_x = out_x * self.stride.1 + kx * self.dilation.1;

                                if input_y < padded_input.len_of(Axis(1))
                                    && input_x < padded_input.len_of(Axis(2))
                                {
                                    sum += weights[[out_ch, in_ch, ky, kx]]
                                        * padded_input[[in_ch, input_y, input_x]];
                                }
                            }
                        }
                    }

                    // Add bias if enabled
                    if let Some(ref biases) = self.biases {
                        sum += biases[out_ch];
                    }

                    output[[out_ch, out_y, out_x]] = sum;
                }
            }
        }

        // Apply activation function
        if let Some(ref activation) = self.activation {
            output.mapv_inplace(|x| apply_activation_scalar(activation, x));
        }

        // Cache for backpropagation
        self.last_input = Some(input.clone());
        self.last_output = Some(output.clone());

        Ok(output)
    }

    /// Apply 2D padding to input
    fn apply_padding_2d(&self, input: &Array3<f64>) -> NeuralResult<Array3<f64>> {
        let padding_amount = match self.padding {
            Padding::Valid => return Ok(input.clone()),
            Padding::Same => {
                let effective_kernel_h = self.dilation.0 * (self.kernel_size.0 - 1) + 1;
                let effective_kernel_w = self.dilation.1 * (self.kernel_size.1 - 1) + 1;
                ((effective_kernel_h - 1) / 2, (effective_kernel_w - 1) / 2)
            }
            Padding::Custom(p) => (p, p),
        };

        if padding_amount.0 == 0 && padding_amount.1 == 0 {
            return Ok(input.clone());
        }

        let (in_channels, input_h, input_w) = input.dim();
        let padded_h = input_h + 2 * padding_amount.0;
        let padded_w = input_w + 2 * padding_amount.1;
        let mut padded = Array3::zeros((in_channels, padded_h, padded_w));

        // Copy original data to center
        for ch in 0..in_channels {
            for y in 0..input_h {
                for x in 0..input_w {
                    padded[[ch, y + padding_amount.0, x + padding_amount.1]] = input[[ch, y, x]];
                }
            }
        }

        Ok(padded)
    }
}

/// Pooling operations for dimensionality reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    Max,
    Average,
    AdaptiveMax,
    AdaptiveAverage,
}

/// 2D Pooling Layer
#[derive(Debug, Clone)]
pub struct Pool2D {
    /// Type of pooling operation
    pub pool_type: PoolingType,
    /// Pool size (height, width)
    pub pool_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding strategy
    pub padding: Padding,

    // Cached values for backpropagation (max pooling)
    last_input: Option<Array3<f64>>,
    max_indices: Option<Array3<(usize, usize)>>,
}

impl Pool2D {
    pub fn new(
        pool_type: PoolingType,
        pool_size: (usize, usize),
        stride: (usize, usize),
        padding: Padding,
    ) -> Self {
        Self {
            pool_type,
            pool_size,
            stride,
            padding,
            last_input: None,
            max_indices: None,
        }
    }

    /// Calculate output size
    pub fn output_size(&self, input_size: (usize, usize)) -> (usize, usize) {
        let padding_amount = match self.padding {
            Padding::Valid => (0, 0),
            Padding::Same => ((self.pool_size.0 - 1) / 2, (self.pool_size.1 - 1) / 2),
            Padding::Custom(p) => (p, p),
        };

        let output_h = (input_size.0 + 2 * padding_amount.0 - self.pool_size.0) / self.stride.0 + 1;
        let output_w = (input_size.1 + 2 * padding_amount.1 - self.pool_size.1) / self.stride.1 + 1;

        (output_h, output_w)
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array3<f64>) -> NeuralResult<Array3<f64>> {
        let (in_channels, input_h, input_w) = input.dim();
        let (output_h, output_w) = self.output_size((input_h, input_w));
        let mut output = Array3::zeros((in_channels, output_h, output_w));

        match self.pool_type {
            PoolingType::Max => {
                let mut max_indices = Array3::from_elem((in_channels, output_h, output_w), (0, 0));

                for ch in 0..in_channels {
                    for out_y in 0..output_h {
                        for out_x in 0..output_w {
                            let mut max_val = f64::NEG_INFINITY;
                            let mut max_idx = (0, 0);

                            for py in 0..self.pool_size.0 {
                                for px in 0..self.pool_size.1 {
                                    let input_y = out_y * self.stride.0 + py;
                                    let input_x = out_x * self.stride.1 + px;

                                    if input_y < input_h && input_x < input_w {
                                        let val = input[[ch, input_y, input_x]];
                                        if val > max_val {
                                            max_val = val;
                                            max_idx = (input_y, input_x);
                                        }
                                    }
                                }
                            }

                            output[[ch, out_y, out_x]] = max_val;
                            max_indices[[ch, out_y, out_x]] = max_idx;
                        }
                    }
                }

                self.max_indices = Some(max_indices);
            }
            PoolingType::Average => {
                for ch in 0..in_channels {
                    for out_y in 0..output_h {
                        for out_x in 0..output_w {
                            let mut sum = 0.0;
                            let mut count = 0;

                            for py in 0..self.pool_size.0 {
                                for px in 0..self.pool_size.1 {
                                    let input_y = out_y * self.stride.0 + py;
                                    let input_x = out_x * self.stride.1 + px;

                                    if input_y < input_h && input_x < input_w {
                                        sum += input[[ch, input_y, input_x]];
                                        count += 1;
                                    }
                                }
                            }

                            output[[ch, out_y, out_x]] =
                                if count > 0 { sum / count as f64 } else { 0.0 };
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Adaptive pooling not yet implemented".to_string(),
                ));
            }
        }

        // Cache for backpropagation
        self.last_input = Some(input.clone());

        Ok(output)
    }
}

/// 3D Convolution Layer for video/volumetric data
#[derive(Debug, Clone)]
pub struct Conv3D {
    /// Number of output filters/channels
    pub out_channels: usize,
    /// Kernel size (depth, height, width)
    pub kernel_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding strategy
    pub padding: Padding,
    /// Dilation rate (depth, height, width)
    pub dilation: (usize, usize, usize),
    /// Activation function
    pub activation: Option<Activation>,
    /// Whether to use bias
    pub use_bias: bool,

    // Learnable parameters
    /// Weights: (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
    pub weights: Option<Array5<f64>>,
    /// Biases: (out_channels,)
    pub biases: Option<Array1<f64>>,

    // Cached values for backpropagation
    last_input: Option<Array4<f64>>,
    last_output: Option<Array4<f64>>,
}

impl Conv3D {
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: Padding,
        dilation: (usize, usize, usize),
        activation: Option<Activation>,
        use_bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
            weights: None,
            biases: None,
            last_input: None,
            last_output: None,
        }
    }

    /// Initialize weights and biases
    pub fn initialize(&mut self, in_channels: usize, rng: &mut impl Rng) -> NeuralResult<()> {
        // He initialization
        let fan_in = in_channels * self.kernel_size.0 * self.kernel_size.1 * self.kernel_size.2;
        let std = (2.0 / fan_in as f64).sqrt();

        // Initialize weights
        let mut weights = Array5::zeros((
            self.out_channels,
            in_channels,
            self.kernel_size.0,
            self.kernel_size.1,
            self.kernel_size.2,
        ));

        for weight in weights.iter_mut() {
            let normal_dist = Normal::new(0.0, std).unwrap();
            *weight = rng.sample(normal_dist);
        }
        self.weights = Some(weights);

        // Initialize biases
        if self.use_bias {
            self.biases = Some(Array1::zeros(self.out_channels));
        }

        Ok(())
    }

    /// Calculate output size given input size
    pub fn output_size(&self, input_size: (usize, usize, usize)) -> (usize, usize, usize) {
        let effective_kernel_d = self.dilation.0 * (self.kernel_size.0 - 1) + 1;
        let effective_kernel_h = self.dilation.1 * (self.kernel_size.1 - 1) + 1;
        let effective_kernel_w = self.dilation.2 * (self.kernel_size.2 - 1) + 1;

        let padding_d = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (effective_kernel_d - 1) / 2,
            Padding::Custom(p) => p,
        };

        let padding_h = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (effective_kernel_h - 1) / 2,
            Padding::Custom(p) => p,
        };

        let padding_w = match self.padding {
            Padding::Valid => 0,
            Padding::Same => (effective_kernel_w - 1) / 2,
            Padding::Custom(p) => p,
        };

        let output_d = (input_size.0 + 2 * padding_d - effective_kernel_d) / self.stride.0 + 1;
        let output_h = (input_size.1 + 2 * padding_h - effective_kernel_h) / self.stride.1 + 1;
        let output_w = (input_size.2 + 2 * padding_w - effective_kernel_w) / self.stride.2 + 1;

        (output_d, output_h, output_w)
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array4<f64>) -> NeuralResult<Array4<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "forward".to_string(),
            })?;

        let (in_channels, input_d, input_h, input_w) = input.dim();
        let (output_d, output_h, output_w) = self.output_size((input_d, input_h, input_w));
        let mut output = Array4::zeros((self.out_channels, output_d, output_h, output_w));

        // Apply padding
        let padded_input = self.apply_padding_3d(input)?;

        // Convolution operation
        for out_ch in 0..self.out_channels {
            for out_z in 0..output_d {
                for out_y in 0..output_h {
                    for out_x in 0..output_w {
                        let mut sum = 0.0;

                        for in_ch in 0..in_channels {
                            for kz in 0..self.kernel_size.0 {
                                for ky in 0..self.kernel_size.1 {
                                    for kx in 0..self.kernel_size.2 {
                                        let input_z = out_z * self.stride.0 + kz * self.dilation.0;
                                        let input_y = out_y * self.stride.1 + ky * self.dilation.1;
                                        let input_x = out_x * self.stride.2 + kx * self.dilation.2;

                                        if input_z < padded_input.len_of(Axis(1))
                                            && input_y < padded_input.len_of(Axis(2))
                                            && input_x < padded_input.len_of(Axis(3))
                                        {
                                            sum += weights[[out_ch, in_ch, kz, ky, kx]]
                                                * padded_input[[in_ch, input_z, input_y, input_x]];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if enabled
                        if let Some(ref biases) = self.biases {
                            sum += biases[out_ch];
                        }

                        output[[out_ch, out_z, out_y, out_x]] = sum;
                    }
                }
            }
        }

        // Apply activation function
        if let Some(ref activation) = self.activation {
            output.mapv_inplace(|x| apply_activation_scalar(activation, x));
        }

        // Cache for backpropagation
        self.last_input = Some(input.clone());
        self.last_output = Some(output.clone());

        Ok(output)
    }

    /// Apply 3D padding to input
    fn apply_padding_3d(&self, input: &Array4<f64>) -> NeuralResult<Array4<f64>> {
        let padding_amount = match self.padding {
            Padding::Valid => return Ok(input.clone()),
            Padding::Same => {
                let effective_kernel_d = self.dilation.0 * (self.kernel_size.0 - 1) + 1;
                let effective_kernel_h = self.dilation.1 * (self.kernel_size.1 - 1) + 1;
                let effective_kernel_w = self.dilation.2 * (self.kernel_size.2 - 1) + 1;
                (
                    (effective_kernel_d - 1) / 2,
                    (effective_kernel_h - 1) / 2,
                    (effective_kernel_w - 1) / 2,
                )
            }
            Padding::Custom(p) => (p, p, p),
        };

        if padding_amount.0 == 0 && padding_amount.1 == 0 && padding_amount.2 == 0 {
            return Ok(input.clone());
        }

        let (in_channels, input_d, input_h, input_w) = input.dim();
        let padded_d = input_d + 2 * padding_amount.0;
        let padded_h = input_h + 2 * padding_amount.1;
        let padded_w = input_w + 2 * padding_amount.2;
        let mut padded = Array4::zeros((in_channels, padded_d, padded_h, padded_w));

        // Copy original data to center
        for ch in 0..in_channels {
            for z in 0..input_d {
                for y in 0..input_h {
                    for x in 0..input_w {
                        padded[[
                            ch,
                            z + padding_amount.0,
                            y + padding_amount.1,
                            x + padding_amount.2,
                        ]] = input[[ch, z, y, x]];
                    }
                }
            }
        }

        Ok(padded)
    }
}

/// 3D Pooling Layer
#[derive(Debug, Clone)]
pub struct Pool3D {
    /// Type of pooling operation
    pub pool_type: PoolingType,
    /// Pool size (depth, height, width)
    pub pool_size: (usize, usize, usize),
    /// Stride (depth, height, width)
    pub stride: (usize, usize, usize),
    /// Padding strategy
    pub padding: Padding,

    // Cached values for backpropagation (max pooling)
    last_input: Option<Array4<f64>>,
    max_indices: Option<Array4<(usize, usize, usize)>>,
}

impl Pool3D {
    pub fn new(
        pool_type: PoolingType,
        pool_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: Padding,
    ) -> Self {
        Self {
            pool_type,
            pool_size,
            stride,
            padding,
            last_input: None,
            max_indices: None,
        }
    }

    /// Calculate output size
    pub fn output_size(&self, input_size: (usize, usize, usize)) -> (usize, usize, usize) {
        let padding_amount = match self.padding {
            Padding::Valid => (0, 0, 0),
            Padding::Same => (
                (self.pool_size.0 - 1) / 2,
                (self.pool_size.1 - 1) / 2,
                (self.pool_size.2 - 1) / 2,
            ),
            Padding::Custom(p) => (p, p, p),
        };

        let output_d = (input_size.0 + 2 * padding_amount.0 - self.pool_size.0) / self.stride.0 + 1;
        let output_h = (input_size.1 + 2 * padding_amount.1 - self.pool_size.1) / self.stride.1 + 1;
        let output_w = (input_size.2 + 2 * padding_amount.2 - self.pool_size.2) / self.stride.2 + 1;

        (output_d, output_h, output_w)
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array4<f64>) -> NeuralResult<Array4<f64>> {
        let (in_channels, input_d, input_h, input_w) = input.dim();
        let (output_d, output_h, output_w) = self.output_size((input_d, input_h, input_w));
        let mut output = Array4::zeros((in_channels, output_d, output_h, output_w));

        match self.pool_type {
            PoolingType::Max => {
                let mut max_indices =
                    Array4::from_elem((in_channels, output_d, output_h, output_w), (0, 0, 0));

                for ch in 0..in_channels {
                    for out_z in 0..output_d {
                        for out_y in 0..output_h {
                            for out_x in 0..output_w {
                                let mut max_val = f64::NEG_INFINITY;
                                let mut max_idx = (0, 0, 0);

                                for pz in 0..self.pool_size.0 {
                                    for py in 0..self.pool_size.1 {
                                        for px in 0..self.pool_size.2 {
                                            let input_z = out_z * self.stride.0 + pz;
                                            let input_y = out_y * self.stride.1 + py;
                                            let input_x = out_x * self.stride.2 + px;

                                            if input_z < input_d
                                                && input_y < input_h
                                                && input_x < input_w
                                            {
                                                let val = input[[ch, input_z, input_y, input_x]];
                                                if val > max_val {
                                                    max_val = val;
                                                    max_idx = (input_z, input_y, input_x);
                                                }
                                            }
                                        }
                                    }
                                }

                                output[[ch, out_z, out_y, out_x]] = max_val;
                                max_indices[[ch, out_z, out_y, out_x]] = max_idx;
                            }
                        }
                    }
                }

                self.max_indices = Some(max_indices);
            }
            PoolingType::Average => {
                for ch in 0..in_channels {
                    for out_z in 0..output_d {
                        for out_y in 0..output_h {
                            for out_x in 0..output_w {
                                let mut sum = 0.0;
                                let mut count = 0;

                                for pz in 0..self.pool_size.0 {
                                    for py in 0..self.pool_size.1 {
                                        for px in 0..self.pool_size.2 {
                                            let input_z = out_z * self.stride.0 + pz;
                                            let input_y = out_y * self.stride.1 + py;
                                            let input_x = out_x * self.stride.2 + px;

                                            if input_z < input_d
                                                && input_y < input_h
                                                && input_x < input_w
                                            {
                                                sum += input[[ch, input_z, input_y, input_x]];
                                                count += 1;
                                            }
                                        }
                                    }
                                }

                                output[[ch, out_z, out_y, out_x]] =
                                    if count > 0 { sum / count as f64 } else { 0.0 };
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Adaptive 3D pooling not yet implemented".to_string(),
                ));
            }
        }

        // Cache for backpropagation
        self.last_input = Some(input.clone());

        Ok(output)
    }
}

/// CNN building blocks for easy network construction
pub mod cnn_blocks {
    use super::*;

    /// A complete convolutional block with conv, activation, and optional pooling
    #[derive(Debug, Clone)]
    pub struct ConvBlock2D {
        pub conv: Conv2D,
        pub pool: Option<Pool2D>,
    }

    impl ConvBlock2D {
        pub fn new(
            out_channels: usize,
            kernel_size: (usize, usize),
            stride: (usize, usize),
            activation: Activation,
            pool_size: Option<(usize, usize)>,
        ) -> Self {
            let conv = Conv2D::new(
                out_channels,
                kernel_size,
                stride,
                Padding::Same,
                (1, 1),
                Some(activation),
                true,
            );

            let pool = pool_size.map(|size| {
                Pool2D::new(
                    PoolingType::Max,
                    size,
                    size, // stride same as pool size
                    Padding::Valid,
                )
            });

            Self { conv, pool }
        }

        pub fn initialize(&mut self, in_channels: usize, rng: &mut impl Rng) -> NeuralResult<()> {
            self.conv.initialize(in_channels, rng)
        }

        pub fn forward(&mut self, input: &Array3<f64>) -> NeuralResult<Array3<f64>> {
            let conv_output = self.conv.forward(input)?;

            if let Some(ref mut pool) = self.pool {
                pool.forward(&conv_output)
            } else {
                Ok(conv_output)
            }
        }

        pub fn output_size(&self, input_size: (usize, usize)) -> (usize, usize) {
            let conv_size = self.conv.output_size(input_size);
            if let Some(ref pool) = self.pool {
                pool.output_size(conv_size)
            } else {
                conv_size
            }
        }
    }

    /// ResNet-style residual block
    #[derive(Debug, Clone)]
    pub struct ResidualBlock2D {
        pub conv1: Conv2D,
        pub conv2: Conv2D,
        pub shortcut: Option<Conv2D>, // For dimension matching
    }

    impl ResidualBlock2D {
        pub fn new(channels: usize, stride: (usize, usize), activation: Activation) -> Self {
            let conv1 = Conv2D::new(
                channels,
                (3, 3),
                (1, 1),
                Padding::Same,
                (1, 1),
                Some(activation),
                true,
            );

            let conv2 = Conv2D::new(
                channels,
                (3, 3),
                stride,
                Padding::Same,
                (1, 1),
                None, // No activation on second conv
                true,
            );

            // Shortcut connection for dimension matching if needed
            let shortcut = if stride != (1, 1) {
                Some(Conv2D::new(
                    channels,
                    (1, 1),
                    stride,
                    Padding::Valid,
                    (1, 1),
                    None,
                    false,
                ))
            } else {
                None
            };

            Self {
                conv1,
                conv2,
                shortcut,
            }
        }

        pub fn initialize(&mut self, in_channels: usize, rng: &mut impl Rng) -> NeuralResult<()> {
            self.conv1.initialize(in_channels, rng)?;
            self.conv2.initialize(in_channels, rng)?;

            if let Some(ref mut shortcut) = self.shortcut {
                shortcut.initialize(in_channels, rng)?;
            }

            Ok(())
        }

        pub fn forward(&mut self, input: &Array3<f64>) -> NeuralResult<Array3<f64>> {
            let x = self.conv1.forward(input)?;
            let x = self.conv2.forward(&x)?;

            // Add residual connection
            let residual = if let Some(ref mut shortcut) = self.shortcut {
                shortcut.forward(input)?
            } else {
                input.clone()
            };

            // Element-wise addition
            let mut output = x;
            output += &residual;

            // Apply activation after addition (ReLU typically)
            if let Some(activation) = &self.conv1.activation {
                output.mapv_inplace(|x| apply_activation_scalar(activation, x));
            }

            Ok(output)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use scirs2_core::ndarray::{Array3, Array4};
    use scirs2_core::random::ChaCha8Rng;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_conv1d_forward() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut conv = Conv1D::new(
            2, // out_channels
            3, // kernel_size
            1, // stride
            Padding::Valid,
            1, // dilation
            Some(Activation::Relu),
            true,
        );

        conv.initialize(1, &mut rng).unwrap();

        let input = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = conv.forward(&input).unwrap();

        assert_eq!(output.dim(), (2, 3)); // 2 channels, length 3 (5-3+1)
    }

    #[test]
    fn test_conv2d_forward() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut conv = Conv2D::new(
            2,      // out_channels
            (3, 3), // kernel_size
            (1, 1), // stride
            Padding::Same,
            (1, 1), // dilation
            Some(Activation::Relu),
            true,
        );

        conv.initialize(1, &mut rng).unwrap();

        let input = Array3::zeros((1, 5, 5));
        let output = conv.forward(&input).unwrap();

        assert_eq!(output.dim(), (2, 5, 5)); // Same size due to padding
    }

    #[test]
    fn test_pool2d_max() {
        let mut pool = Pool2D::new(PoolingType::Max, (2, 2), (2, 2), Padding::Valid);

        let mut input = Array3::zeros((1, 4, 4));
        input[[0, 0, 0]] = 1.0;
        input[[0, 0, 1]] = 2.0;
        input[[0, 1, 0]] = 3.0;
        input[[0, 1, 1]] = 4.0;

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.dim(), (1, 2, 2));
        assert_eq!(output[[0, 0, 0]], 4.0); // Max of [1,2,3,4]
    }

    #[test]
    fn test_conv_block() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut block = cnn_blocks::ConvBlock2D::new(
            4,      // out_channels
            (3, 3), // kernel_size
            (1, 1), // stride
            Activation::Relu,
            Some((2, 2)), // pool_size
        );

        block.initialize(3, &mut rng).unwrap();

        let input = Array3::zeros((3, 8, 8));
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dim(), (4, 4, 4)); // 8x8 -> conv -> 8x8 -> pool -> 4x4
    }

    #[test]
    fn test_conv3d_forward() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut conv = Conv3D::new(
            2,         // out_channels
            (3, 3, 3), // kernel_size
            (1, 1, 1), // stride
            Padding::Same,
            (1, 1, 1), // dilation
            Some(Activation::Relu),
            true,
        );

        conv.initialize(1, &mut rng).unwrap();

        let input = Array4::zeros((1, 4, 4, 4));
        let output = conv.forward(&input).unwrap();

        assert_eq!(output.dim(), (2, 4, 4, 4)); // Same size due to padding
    }

    #[test]
    fn test_pool3d_max() {
        let mut pool = Pool3D::new(PoolingType::Max, (2, 2, 2), (2, 2, 2), Padding::Valid);

        let mut input = Array4::zeros((1, 4, 4, 4));
        input[[0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;
        input[[0, 1, 0, 0]] = 5.0;
        input[[0, 1, 0, 1]] = 6.0;
        input[[0, 1, 1, 0]] = 7.0;
        input[[0, 1, 1, 1]] = 8.0;

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.dim(), (1, 2, 2, 2));
        assert_eq!(output[[0, 0, 0, 0]], 8.0); // Max of [1,2,3,4,5,6,7,8]
    }

    // TODO: Implement DepthwiseSeparableConv2D and uncomment this test
    // #[test]
    // fn test_depthwise_separable_conv() {
    //     let mut rng = ChaCha8Rng::seed_from_u64(42);
    //     let mut conv = DepthwiseSeparableConv2D::new(
    //         8, // out_channels
    //         (3, 3), // kernel_size
    //         (1, 1), // stride
    //         Padding::Same,
    //         (1, 1), // dilation
    //         Some(Activation::Relu),
    //         true,
    //         2, // depth_multiplier
    //     );
    //
    //     conv.initialize(4, &mut rng).unwrap();
    //
    //     let input = Array3::zeros((4, 8, 8));
    //     let output = conv.forward(&input).unwrap();
    //
    //     assert_eq!(output.dim(), (8, 8, 8)); // Same spatial size due to padding
    // }

    // TODO: Implement GroupConv2D and uncomment this test
    // #[test]
    // fn test_group_conv() {
    //     let mut rng = ChaCha8Rng::seed_from_u64(42);
    //     let mut conv = GroupConv2D::new(
    //         8, // out_channels
    //         (3, 3), // kernel_size
    //         (1, 1), // stride
    //         Padding::Same,
    //         (1, 1), // dilation
    //         2, // groups
    //         Some(Activation::Relu),
    //         true,
    //     );
    //
    //     conv.initialize(4, &mut rng).unwrap();
    //
    //     let input = Array3::zeros((4, 8, 8));
    //     let output = conv.forward(&input).unwrap();
    //
    //     assert_eq!(output.dim(), (8, 8, 8)); // Same spatial size due to padding
    // }
}
