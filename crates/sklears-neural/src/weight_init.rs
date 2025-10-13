//! Weight initialization strategies for neural networks.
//!
//! This module provides various weight initialization methods that are crucial
//! for proper neural network training. Different initialization strategies are
//! optimal for different activation functions and architectures.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::FromPrimitive;
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::prelude::StdRng;
use scirs2_core::random::{Distribution, Rng, SeedableRng};
use sklears_core::{error::SklearsError, types::FloatBounds};

/// Weight initialization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitStrategy {
    /// Zero initialization (not recommended for hidden layers)
    Zeros,
    /// Uniform initialization with given range
    Uniform { low: f64, high: f64 },
    /// Normal/Gaussian initialization
    Normal { mean: f64, std: f64 },
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization  
    XavierNormal,
    /// He uniform initialization (good for ReLU)
    HeUniform,
    /// He normal initialization (good for ReLU)
    HeNormal,
    /// LeCun uniform initialization
    LeCunUniform,
    /// LeCun normal initialization
    LeCunNormal,
    /// Orthogonal initialization
    Orthogonal { gain: f64 },
    /// Truncated normal initialization
    TruncatedNormal {
        mean: f64,
        std: f64,
        low: f64,
        high: f64,
    },
    /// Variance scaling initialization (general form)
    VarianceScaling {
        scale: f64,
        mode: ScalingMode,
        distribution: ScalingDistribution,
    },
}

/// Mode for variance scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    /// Scale by fan-in (number of input units)
    FanIn,
    /// Scale by fan-out (number of output units)
    FanOut,
    /// Scale by average of fan-in and fan-out
    FanAvg,
}

/// Distribution for variance scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingDistribution {
    /// Uniform distribution
    Uniform,
    /// Normal distribution
    Normal,
    /// Truncated normal distribution
    TruncatedNormal,
}

impl Default for InitStrategy {
    fn default() -> Self {
        InitStrategy::XavierUniform
    }
}

/// Weight initializer
pub struct WeightInitializer<T: FloatBounds> {
    strategy: InitStrategy,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatBounds> WeightInitializer<T> {
    /// Create a new weight initializer with the given strategy
    pub fn new(strategy: InitStrategy) -> Self {
        Self {
            strategy,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Initialize a 2D weight matrix
    pub fn initialize_2d<R: Rng>(
        &self,
        rng: &mut R,
        shape: (usize, usize),
    ) -> NeuralResult<Array2<T>> {
        let (rows, cols) = shape;
        let fan_in = rows;
        let fan_out = cols;

        match self.strategy {
            InitStrategy::Zeros => Ok(Array2::zeros(shape)),

            InitStrategy::Uniform { low, high } => {
                let uniform = Uniform::new(low, high).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid uniform distribution: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(uniform.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::Normal { mean, std } => {
                let normal = Normal::new(mean, std).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid normal distribution: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(normal.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::XavierUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let uniform = Uniform::new(-limit, limit).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid Xavier uniform: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(uniform.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let normal = Normal::new(0.0, std).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid Xavier normal: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(normal.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::HeUniform => {
                let limit = (6.0 / fan_in as f64).sqrt();
                let uniform = Uniform::new(-limit, limit).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid He uniform: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(uniform.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::HeNormal => {
                let std = (2.0 / fan_in as f64).sqrt();
                let normal = Normal::new(0.0, std)
                    .map_err(|e| SklearsError::InvalidInput(format!("Invalid He normal: {}", e)))?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(normal.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::LeCunUniform => {
                let limit = (3.0 / fan_in as f64).sqrt();
                let uniform = Uniform::new(-limit, limit).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid LeCun uniform: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(uniform.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::LeCunNormal => {
                let std = (1.0 / fan_in as f64).sqrt();
                let normal = Normal::new(0.0, std).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid LeCun normal: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(normal.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            InitStrategy::Orthogonal { gain } => self.orthogonal_init(rng, shape, gain),

            InitStrategy::TruncatedNormal {
                mean,
                std,
                low,
                high,
            } => self.truncated_normal_init(rng, shape, mean, std, low, high),

            InitStrategy::VarianceScaling {
                scale,
                mode,
                distribution,
            } => self.variance_scaling_init(rng, shape, scale, mode, distribution, fan_in, fan_out),
        }
    }

    /// Initialize a 1D bias vector
    pub fn initialize_1d<R: Rng>(&self, rng: &mut R, size: usize) -> NeuralResult<Array1<T>> {
        // For biases, typically use zeros or small values
        match self.strategy {
            InitStrategy::Zeros => Ok(Array1::zeros(size)),
            InitStrategy::Uniform { low, high } => {
                let uniform = Uniform::new(low, high).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid uniform distribution: {}", e))
                })?;
                let values: Vec<T> = (0..size)
                    .map(|_| FromPrimitive::from_f64(uniform.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Ok(Array1::from_vec(values))
            }
            InitStrategy::Normal { mean, std } => {
                let normal = Normal::new(mean, std).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid normal distribution: {}", e))
                })?;
                let values: Vec<T> = (0..size)
                    .map(|_| FromPrimitive::from_f64(normal.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Ok(Array1::from_vec(values))
            }
            // For most other strategies, use zeros for biases
            _ => Ok(Array1::zeros(size)),
        }
    }

    /// Orthogonal initialization using QR decomposition
    fn orthogonal_init<R: Rng>(
        &self,
        rng: &mut R,
        shape: (usize, usize),
        gain: f64,
    ) -> NeuralResult<Array2<T>> {
        let (rows, cols) = shape;

        // Generate random matrix
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            SklearsError::InvalidInput(format!("Invalid normal for orthogonal: {}", e))
        })?;
        let values: Vec<f64> = (0..rows * cols).map(|_| normal.sample(rng)).collect();
        let mut matrix = Array2::from_shape_vec(shape, values)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        // Perform QR decomposition (simplified version)
        // This is a basic implementation - in practice, you'd use a more robust QR decomposition
        self.gram_schmidt_qr(&mut matrix)?;

        // Apply gain
        matrix.mapv_inplace(|x| x * gain);

        // Convert to target type
        Ok(matrix.mapv(|x| FromPrimitive::from_f64(x).unwrap_or(T::zero())))
    }

    /// Simplified Gram-Schmidt QR decomposition
    fn gram_schmidt_qr(&self, matrix: &mut Array2<f64>) -> NeuralResult<()> {
        let (rows, cols) = matrix.dim();
        let min_dim = rows.min(cols);

        for i in 0..min_dim {
            // Normalize column i
            let mut col_i = matrix.column_mut(i);
            let norm = col_i.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                col_i.mapv_inplace(|x| x / norm);
            }

            // Orthogonalize subsequent columns
            for j in (i + 1)..cols {
                let dot_product = matrix.column(i).dot(&matrix.column(j));

                // Get column i values before taking mutable reference to column j
                let col_i_values: Vec<f64> = matrix.column(i).to_vec();
                let mut col_j = matrix.column_mut(j);

                for (c_j, &c_i) in col_j.iter_mut().zip(col_i_values.iter()) {
                    *c_j -= dot_product * c_i;
                }
            }
        }

        Ok(())
    }

    /// Truncated normal initialization
    fn truncated_normal_init<R: Rng>(
        &self,
        rng: &mut R,
        shape: (usize, usize),
        mean: f64,
        std: f64,
        low: f64,
        high: f64,
    ) -> NeuralResult<Array2<T>> {
        let (rows, cols) = shape;
        let normal = Normal::new(mean, std)
            .map_err(|e| SklearsError::InvalidInput(format!("Invalid truncated normal: {}", e)))?;

        let mut values = Vec::with_capacity(rows * cols);

        // Generate values within bounds
        for _ in 0..rows * cols {
            let mut val = normal.sample(rng);
            let mut attempts = 0;

            // Rejection sampling to stay within bounds
            while (val < low || val > high) && attempts < 100 {
                val = normal.sample(rng);
                attempts += 1;
            }

            // If we can't generate a valid value, clamp it
            if val < low {
                val = low;
            } else if val > high {
                val = high;
            }

            values.push(FromPrimitive::from_f64(val).unwrap_or(T::zero()));
        }

        Array2::from_shape_vec(shape, values)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
    }

    /// Variance scaling initialization
    fn variance_scaling_init<R: Rng>(
        &self,
        rng: &mut R,
        shape: (usize, usize),
        scale: f64,
        mode: ScalingMode,
        distribution: ScalingDistribution,
        fan_in: usize,
        fan_out: usize,
    ) -> NeuralResult<Array2<T>> {
        let (rows, cols) = shape;

        let fan = match mode {
            ScalingMode::FanIn => fan_in as f64,
            ScalingMode::FanOut => fan_out as f64,
            ScalingMode::FanAvg => (fan_in + fan_out) as f64 / 2.0,
        };

        let variance = scale / fan;

        match distribution {
            ScalingDistribution::Uniform => {
                let limit = (3.0 * variance).sqrt();
                let uniform = Uniform::new(-limit, limit).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid variance scaling uniform: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(uniform.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            ScalingDistribution::Normal => {
                let std = variance.sqrt();
                let normal = Normal::new(0.0, std).map_err(|e| {
                    SklearsError::InvalidInput(format!("Invalid variance scaling normal: {}", e))
                })?;
                let values: Vec<T> = (0..rows * cols)
                    .map(|_| FromPrimitive::from_f64(normal.sample(rng)).unwrap_or(T::zero()))
                    .collect();
                Array2::from_shape_vec(shape, values)
                    .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
            }

            ScalingDistribution::TruncatedNormal => {
                let std = variance.sqrt();
                self.truncated_normal_init(rng, shape, 0.0, std, -2.0 * std, 2.0 * std)
            }
        }
    }
}

/// Convenience functions for common initialization strategies
impl<T: FloatBounds> WeightInitializer<T> {
    /// Xavier/Glorot uniform initialization
    pub fn xavier_uniform() -> Self {
        Self::new(InitStrategy::XavierUniform)
    }

    /// Xavier/Glorot normal initialization
    pub fn xavier_normal() -> Self {
        Self::new(InitStrategy::XavierNormal)
    }

    /// He uniform initialization (good for ReLU)
    pub fn he_uniform() -> Self {
        Self::new(InitStrategy::HeUniform)
    }

    /// He normal initialization (good for ReLU)
    pub fn he_normal() -> Self {
        Self::new(InitStrategy::HeNormal)
    }

    /// LeCun uniform initialization
    pub fn lecun_uniform() -> Self {
        Self::new(InitStrategy::LeCunUniform)
    }

    /// LeCun normal initialization
    pub fn lecun_normal() -> Self {
        Self::new(InitStrategy::LeCunNormal)
    }

    /// Orthogonal initialization with default gain of 1.0
    pub fn orthogonal() -> Self {
        Self::new(InitStrategy::Orthogonal { gain: 1.0 })
    }

    /// Orthogonal initialization with custom gain
    pub fn orthogonal_with_gain(gain: f64) -> Self {
        Self::new(InitStrategy::Orthogonal { gain })
    }

    /// Zero initialization
    pub fn zeros() -> Self {
        Self::new(InitStrategy::Zeros)
    }

    /// Uniform initialization
    pub fn uniform(low: f64, high: f64) -> Self {
        Self::new(InitStrategy::Uniform { low, high })
    }

    /// Normal initialization
    pub fn normal(mean: f64, std: f64) -> Self {
        Self::new(InitStrategy::Normal { mean, std })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::random::rngs::StdRng;

    #[test]
    fn test_xavier_uniform_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::xavier_uniform();

        let weights = initializer.initialize_2d(&mut rng, (100, 50)).unwrap();

        // Check shape
        assert_eq!(weights.dim(), (100, 50));

        // Check that values are in expected range
        let fan_in = 100;
        let fan_out = 50;
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();

        for &val in weights.iter() {
            assert!(val >= -limit && val <= limit);
        }

        // Check approximate mean and variance
        let mean = weights.mean().unwrap();
        let variance = weights.mapv(|x| (x - mean) * (x - mean)).mean().unwrap();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
        assert!(variance > 0.001); // Should have some variance
    }

    #[test]
    fn test_he_normal_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::he_normal();

        let weights = initializer.initialize_2d(&mut rng, (128, 64)).unwrap();

        // Check shape
        assert_eq!(weights.dim(), (128, 64));

        // For He initialization, variance should be approximately 2/fan_in
        let fan_in = 128;
        let expected_variance = 2.0 / fan_in as f64;

        let mean = weights.mean().unwrap();
        let actual_variance = weights.mapv(|x| (x - mean) * (x - mean)).mean().unwrap();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(actual_variance, expected_variance, epsilon = 0.01);
    }

    #[test]
    fn test_orthogonal_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::orthogonal();

        let weights = initializer.initialize_2d(&mut rng, (10, 10)).unwrap();

        // Check shape
        assert_eq!(weights.dim(), (10, 10));

        // For square orthogonal matrix, columns should be approximately orthonormal
        // W^T * W should be approximately identity
        let weights_t = weights.t();
        let product = weights_t.dot(&weights);

        // Check diagonal elements are close to 1
        for i in 0..10 {
            assert_abs_diff_eq!(product[[i, i]], 1.0, epsilon = 0.1);
        }

        // Check off-diagonal elements are close to 0
        for i in 0..10 {
            for j in 0..10 {
                if i != j {
                    assert_abs_diff_eq!(product[[i, j]], 0.0, epsilon = 0.1);
                }
            }
        }
    }

    #[test]
    fn test_truncated_normal_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::new(InitStrategy::TruncatedNormal {
            mean: 0.0,
            std: 1.0,
            low: -2.0,
            high: 2.0,
        });

        let weights = initializer.initialize_2d(&mut rng, (100, 100)).unwrap();

        // Check that all values are within bounds
        for &val in weights.iter() {
            assert!(val >= -2.0 && val <= 2.0);
        }

        // Check approximate mean
        let mean = weights.mean().unwrap();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.2);
    }

    #[test]
    fn test_zeros_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::zeros();

        let weights = initializer.initialize_2d(&mut rng, (50, 30)).unwrap();
        let biases = initializer.initialize_1d(&mut rng, 30).unwrap();

        // Check that all values are zero
        for &val in weights.iter() {
            assert_eq!(val, 0.0);
        }

        for &val in biases.iter() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_uniform_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::uniform(-0.5, 0.5);

        let weights = initializer.initialize_2d(&mut rng, (100, 50)).unwrap();

        // Check that all values are within bounds
        for &val in weights.iter() {
            assert!(val >= -0.5 && val <= 0.5);
        }

        // Check approximate mean (should be close to 0 for symmetric range)
        let mean = weights.mean().unwrap();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_variance_scaling_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let initializer = WeightInitializer::<f64>::new(InitStrategy::VarianceScaling {
            scale: 2.0,
            mode: ScalingMode::FanIn,
            distribution: ScalingDistribution::Normal,
        });

        let weights = initializer.initialize_2d(&mut rng, (100, 50)).unwrap();

        // Check variance
        let fan_in = 100;
        let expected_variance = 2.0 / fan_in as f64;

        let mean = weights.mean().unwrap();
        let actual_variance = weights.mapv(|x| (x - mean) * (x - mean)).mean().unwrap();

        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(actual_variance, expected_variance, epsilon = 0.005);
    }

    #[test]
    fn test_bias_initialization() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test zeros bias initialization (default for most strategies)
        let initializer = WeightInitializer::<f64>::xavier_uniform();
        let biases = initializer.initialize_1d(&mut rng, 10).unwrap();

        for &val in biases.iter() {
            assert_eq!(val, 0.0);
        }

        // Test custom bias initialization
        let initializer = WeightInitializer::<f64>::uniform(-0.1, 0.1);
        let biases = initializer.initialize_1d(&mut rng, 10).unwrap();

        for &val in biases.iter() {
            assert!(val >= -0.1 && val <= 0.1);
        }
    }
}
