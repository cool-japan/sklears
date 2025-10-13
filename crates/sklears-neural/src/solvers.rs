//! Optimization solvers for neural networks.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Gradient clipping methods
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GradientClipping {
    /// No gradient clipping
    None,
    /// Clip gradients by norm (global norm clipping)
    ByNorm(f64),
    /// Clip gradients by value (element-wise clipping)
    ByValue(f64),
}

/// Gradient clipping utilities
pub struct GradientClipper;

impl GradientClipper {
    /// Apply gradient clipping to weights and biases gradients
    pub fn clip_gradients(
        weight_grads: &mut [Array2<f64>],
        bias_grads: &mut [Array1<f64>],
        clipping: GradientClipping,
    ) {
        match clipping {
            GradientClipping::None => {}
            GradientClipping::ByNorm(max_norm) => {
                Self::clip_by_global_norm(weight_grads, bias_grads, max_norm);
            }
            GradientClipping::ByValue(max_value) => {
                Self::clip_by_value(weight_grads, bias_grads, max_value);
            }
        }
    }

    /// Clip gradients by global norm
    fn clip_by_global_norm(
        weight_grads: &mut [Array2<f64>],
        bias_grads: &mut [Array1<f64>],
        max_norm: f64,
    ) {
        // Compute global norm
        let mut global_norm_sq = 0.0;

        for grad in weight_grads.iter() {
            global_norm_sq += grad.iter().map(|x| x * x).sum::<f64>();
        }

        for grad in bias_grads.iter() {
            global_norm_sq += grad.iter().map(|x| x * x).sum::<f64>();
        }

        let global_norm = global_norm_sq.sqrt();

        // Apply clipping if necessary
        if global_norm > max_norm {
            let clip_coeff = max_norm / global_norm;

            for grad in weight_grads.iter_mut() {
                grad.mapv_inplace(|x| x * clip_coeff);
            }

            for grad in bias_grads.iter_mut() {
                grad.mapv_inplace(|x| x * clip_coeff);
            }
        }
    }

    /// Clip gradients by value (element-wise)
    fn clip_by_value(
        weight_grads: &mut [Array2<f64>],
        bias_grads: &mut [Array1<f64>],
        max_value: f64,
    ) {
        for grad in weight_grads.iter_mut() {
            grad.mapv_inplace(|x| x.max(-max_value).min(max_value));
        }

        for grad in bias_grads.iter_mut() {
            grad.mapv_inplace(|x| x.max(-max_value).min(max_value));
        }
    }
}
/// Solver algorithms for weight optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Solver {
    /// Limited-memory BFGS algorithm
    Lbfgs,
    /// Stochastic Gradient Descent
    Sgd,
    /// Adam optimizer
    #[default]
    Adam,
    /// AdamW optimizer (Adam with decoupled weight decay)
    AdamW,
    /// RMSprop optimizer
    RMSprop,
    /// Nadam optimizer (Nesterov-accelerated Adam)
    Nadam,
    /// LARS optimizer (Layer-wise Adaptive Rate Scaling)
    Lars,
    /// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training)
    Lamb,
}

/// Learning rate schedule for SGD
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LearningRateSchedule {
    /// Constant learning rate
    #[default]
    Constant,
    /// Inverted scaling: lr = lr_init / (t^power)
    InvScaling,
    /// Adaptive learning rate based on loss improvement
    Adaptive,
    /// Exponential decay: lr = lr_init * decay_rate^(step / decay_steps)
    ExponentialDecay,
    /// Cosine annealing: lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * step / max_steps)) / 2
    CosineAnnealing,
    /// Step decay: lr = lr_init * decay_factor^floor(step / step_size)
    StepDecay,
    /// Warmup: lr increases linearly from 0 to lr_init over warmup_steps
    Warmup,
    /// Cyclical learning rate: lr cycles between lr_min and lr_max
    CyclicalLR,
}

/// SGD solver for neural network optimization
#[derive(Debug, Clone)]
pub struct SgdSolver {
    pub learning_rate: f64,
    pub momentum: f64,
    pub nesterovs_momentum: bool,
    pub schedule: LearningRateSchedule,
    pub power_t: f64,
    pub eta0: f64,
    // Additional parameters for learning rate schedules
    pub decay_rate: f64,     // for exponential decay
    pub decay_steps: usize,  // for exponential decay
    pub max_steps: usize,    // for cosine annealing and warmup
    pub lr_min: f64,         // for cosine annealing and cyclical LR
    pub lr_max: f64,         // for cyclical LR
    pub step_size: usize,    // for step decay and cyclical LR
    pub decay_factor: f64,   // for step decay
    pub warmup_steps: usize, // for warmup
    pub cycle_length: usize, // for cyclical LR
    velocity_weights: Vec<Array2<f64>>,
    velocity_biases: Vec<Array1<f64>>,
    t: usize,
}

impl SgdSolver {
    pub fn new(
        learning_rate: f64,
        momentum: f64,
        nesterovs_momentum: bool,
        schedule: LearningRateSchedule,
        power_t: f64,
        eta0: f64,
    ) -> Self {
        Self {
            learning_rate,
            momentum,
            nesterovs_momentum,
            schedule,
            power_t,
            eta0,
            // Default values for additional parameters
            decay_rate: 0.9,
            decay_steps: 1000,
            max_steps: 10000,
            lr_min: 0.0,
            lr_max: learning_rate,
            step_size: 1000,
            decay_factor: 0.1,
            warmup_steps: 1000,
            cycle_length: 2000,
            velocity_weights: Vec::new(),
            velocity_biases: Vec::new(),
            t: 0,
        }
    }

    /// Create a new SGD solver with full configuration
    pub fn new_with_schedule_params(
        learning_rate: f64,
        momentum: f64,
        nesterovs_momentum: bool,
        schedule: LearningRateSchedule,
        power_t: f64,
        eta0: f64,
        decay_rate: f64,
        decay_steps: usize,
        max_steps: usize,
        lr_min: f64,
        lr_max: f64,
        step_size: usize,
        decay_factor: f64,
        warmup_steps: usize,
        cycle_length: usize,
    ) -> Self {
        Self {
            learning_rate,
            momentum,
            nesterovs_momentum,
            schedule,
            power_t,
            eta0,
            decay_rate,
            decay_steps,
            max_steps,
            lr_min,
            lr_max,
            step_size,
            decay_factor,
            warmup_steps,
            cycle_length,
            velocity_weights: Vec::new(),
            velocity_biases: Vec::new(),
            t: 0,
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.velocity_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.velocity_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.t = 0;
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        self.t += 1;
        let lr = self.get_learning_rate();

        for i in 0..weights.len() {
            // Update velocity for weights
            self.velocity_weights[i] =
                &self.velocity_weights[i] * self.momentum - &weight_grads[i] * lr;

            if self.nesterovs_momentum {
                // Nesterov momentum
                weights[i] =
                    &weights[i] + &self.velocity_weights[i] * self.momentum - &weight_grads[i] * lr;
            } else {
                // Standard momentum
                weights[i] = &weights[i] + &self.velocity_weights[i];
            }

            // Update velocity for biases
            self.velocity_biases[i] =
                &self.velocity_biases[i] * self.momentum - &bias_grads[i] * lr;

            if self.nesterovs_momentum {
                biases[i] =
                    &biases[i] + &self.velocity_biases[i] * self.momentum - &bias_grads[i] * lr;
            } else {
                biases[i] = &biases[i] + &self.velocity_biases[i];
            }
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> f64 {
        match self.schedule {
            LearningRateSchedule::Constant => self.learning_rate,
            LearningRateSchedule::InvScaling => self.eta0 / (self.t as f64).powf(self.power_t),
            LearningRateSchedule::Adaptive => {
                // This would need loss history to implement properly
                // For now, return constant rate
                self.learning_rate
            }
            LearningRateSchedule::ExponentialDecay => {
                // lr = lr_init * decay_rate^(step / decay_steps)
                self.learning_rate
                    * self
                        .decay_rate
                        .powf(self.t as f64 / self.decay_steps as f64)
            }
            LearningRateSchedule::CosineAnnealing => {
                // lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * step / max_steps)) / 2
                let progress = (self.t as f64 / self.max_steps as f64).min(1.0);
                self.lr_min
                    + (self.learning_rate - self.lr_min)
                        * (1.0 + (std::f64::consts::PI * progress).cos())
                        / 2.0
            }
            LearningRateSchedule::StepDecay => {
                // lr = lr_init * decay_factor^floor(step / step_size)
                let step_count = self.t / self.step_size;
                self.learning_rate * self.decay_factor.powi(step_count as i32)
            }
            LearningRateSchedule::Warmup => {
                // Linear warmup: lr increases linearly from 0 to lr_init over warmup_steps
                if self.t <= self.warmup_steps {
                    self.learning_rate * (self.t as f64 / self.warmup_steps as f64)
                } else {
                    self.learning_rate
                }
            }
            LearningRateSchedule::CyclicalLR => {
                // Triangular cyclical learning rate
                let cycle = 1.0 + (self.t as f64 / (2.0 * self.cycle_length as f64));
                let x = (2.0 * cycle.fract() - 1.0).abs();
                self.lr_min + (self.lr_max - self.lr_min) * (1.0 - x).max(0.0)
            }
        }
    }
}

/// Adam optimizer for neural network optimization
#[derive(Debug, Clone)]
pub struct AdamSolver {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m_weights: Vec<Array2<f64>>,
    v_weights: Vec<Array2<f64>>,
    m_biases: Vec<Array1<f64>>,
    v_biases: Vec<Array1<f64>>,
    t: usize,
}

impl AdamSolver {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
            t: 0,
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.m_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.v_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.m_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.v_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.t = 0;
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        self.t += 1;
        let lr_t = self.learning_rate * ((1.0 - self.beta2.powi(self.t as i32)).sqrt())
            / (1.0 - self.beta1.powi(self.t as i32));

        for i in 0..weights.len() {
            // Update biased first moment estimate for weights
            self.m_weights[i] =
                &self.m_weights[i] * self.beta1 + &weight_grads[i] * (1.0 - self.beta1);

            // Update biased second raw moment estimate for weights
            self.v_weights[i] = &self.v_weights[i] * self.beta2
                + &weight_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            // Update weights
            let update =
                &self.m_weights[i] / (&self.v_weights[i].mapv(|x| x.sqrt()) + self.epsilon);
            weights[i] = &weights[i] - &update * lr_t;

            // Update biased first moment estimate for biases
            self.m_biases[i] = &self.m_biases[i] * self.beta1 + &bias_grads[i] * (1.0 - self.beta1);

            // Update biased second raw moment estimate for biases
            self.v_biases[i] = &self.v_biases[i] * self.beta2
                + &bias_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            // Update biases
            let bias_update =
                &self.m_biases[i] / (&self.v_biases[i].mapv(|x| x.sqrt()) + self.epsilon);
            biases[i] = &biases[i] - &bias_update * lr_t;
        }

        Ok(())
    }
}

/// AdamW optimizer with decoupled weight decay
#[derive(Debug, Clone)]
pub struct AdamWSolver {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    m_weights: Vec<Array2<f64>>,
    v_weights: Vec<Array2<f64>>,
    m_biases: Vec<Array1<f64>>,
    v_biases: Vec<Array1<f64>>,
    t: usize,
}

impl AdamWSolver {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
            t: 0,
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.m_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.v_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.m_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.v_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.t = 0;
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr_t = self.learning_rate * (bias_correction2.sqrt()) / bias_correction1;

        for i in 0..weights.len() {
            // Update biased first moment estimate for weights
            self.m_weights[i] =
                &self.m_weights[i] * self.beta1 + &weight_grads[i] * (1.0 - self.beta1);

            // Update biased second raw moment estimate for weights
            self.v_weights[i] = &self.v_weights[i] * self.beta2
                + &weight_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            // AdamW update with decoupled weight decay
            let update =
                &self.m_weights[i] / (&self.v_weights[i].mapv(|x| x.sqrt()) + self.epsilon);
            weights[i] =
                &weights[i] * (1.0 - self.learning_rate * self.weight_decay) - &update * lr_t;

            // Update biased first moment estimate for biases (no weight decay)
            self.m_biases[i] = &self.m_biases[i] * self.beta1 + &bias_grads[i] * (1.0 - self.beta1);

            // Update biased second raw moment estimate for biases
            self.v_biases[i] = &self.v_biases[i] * self.beta2
                + &bias_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            // Update biases (no weight decay)
            let bias_update =
                &self.m_biases[i] / (&self.v_biases[i].mapv(|x| x.sqrt()) + self.epsilon);
            biases[i] = &biases[i] - &bias_update * lr_t;
        }

        Ok(())
    }
}

/// RMSprop optimizer
#[derive(Debug, Clone)]
pub struct RMSpropSolver {
    pub learning_rate: f64,
    pub alpha: f64, // decay rate
    pub epsilon: f64,
    pub momentum: f64,
    pub centered: bool, // whether to use centered RMSprop
    v_weights: Vec<Array2<f64>>,
    v_biases: Vec<Array1<f64>>,
    momentum_weights: Vec<Array2<f64>>,
    momentum_biases: Vec<Array1<f64>>,
    // For centered RMSprop
    g_weights: Vec<Array2<f64>>,
    g_biases: Vec<Array1<f64>>,
}

impl RMSpropSolver {
    pub fn new(
        learning_rate: f64,
        alpha: f64,
        epsilon: f64,
        momentum: f64,
        centered: bool,
    ) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            momentum,
            centered,
            v_weights: Vec::new(),
            v_biases: Vec::new(),
            momentum_weights: Vec::new(),
            momentum_biases: Vec::new(),
            g_weights: Vec::new(),
            g_biases: Vec::new(),
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.v_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.v_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.momentum_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.momentum_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();

        if self.centered {
            self.g_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
            self.g_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        }
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        for i in 0..weights.len() {
            // Update accumulated squared gradients for weights
            self.v_weights[i] = &self.v_weights[i] * self.alpha
                + &weight_grads[i].mapv(|x| x * x) * (1.0 - self.alpha);

            let denominator = if self.centered {
                // Update accumulated gradients for centered RMSprop
                self.g_weights[i] =
                    &self.g_weights[i] * self.alpha + &weight_grads[i] * (1.0 - self.alpha);
                (&self.v_weights[i] - &self.g_weights[i].mapv(|x| x * x))
                    .mapv(|x| (x + self.epsilon).sqrt())
            } else {
                self.v_weights[i].mapv(|x| (x + self.epsilon).sqrt())
            };

            // Compute update
            let update = &weight_grads[i] / &denominator;

            if self.momentum > 0.0 {
                // Apply momentum
                self.momentum_weights[i] =
                    &self.momentum_weights[i] * self.momentum + &update * self.learning_rate;
                weights[i] = &weights[i] - &self.momentum_weights[i];
            } else {
                // Direct update
                weights[i] = &weights[i] - &update * self.learning_rate;
            }

            // Same process for biases
            self.v_biases[i] = &self.v_biases[i] * self.alpha
                + &bias_grads[i].mapv(|x| x * x) * (1.0 - self.alpha);

            let bias_denominator = if self.centered {
                self.g_biases[i] =
                    &self.g_biases[i] * self.alpha + &bias_grads[i] * (1.0 - self.alpha);
                (&self.v_biases[i] - &self.g_biases[i].mapv(|x| x * x))
                    .mapv(|x| (x + self.epsilon).sqrt())
            } else {
                self.v_biases[i].mapv(|x| (x + self.epsilon).sqrt())
            };

            let bias_update = &bias_grads[i] / &bias_denominator;

            if self.momentum > 0.0 {
                self.momentum_biases[i] =
                    &self.momentum_biases[i] * self.momentum + &bias_update * self.learning_rate;
                biases[i] = &biases[i] - &self.momentum_biases[i];
            } else {
                biases[i] = &biases[i] - &bias_update * self.learning_rate;
            }
        }

        Ok(())
    }
}

/// Nadam optimizer (Nesterov-accelerated Adam)
#[derive(Debug, Clone)]
pub struct NadamSolver {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m_weights: Vec<Array2<f64>>,
    v_weights: Vec<Array2<f64>>,
    m_biases: Vec<Array1<f64>>,
    v_biases: Vec<Array1<f64>>,
    t: usize,
}

impl NadamSolver {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
            t: 0,
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.m_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.v_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.m_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.v_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.t = 0;
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        self.t += 1;
        let beta1_t = self.beta1.powi(self.t as i32);
        let beta2_t = self.beta2.powi(self.t as i32);
        let lr_t = self.learning_rate / (1.0 - beta1_t);

        for i in 0..weights.len() {
            // Update biased first moment estimate
            self.m_weights[i] =
                &self.m_weights[i] * self.beta1 + &weight_grads[i] * (1.0 - self.beta1);

            // Update biased second raw moment estimate
            self.v_weights[i] = &self.v_weights[i] * self.beta2
                + &weight_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            // Bias-corrected second moment estimate
            let v_corrected = &self.v_weights[i] / (1.0 - beta2_t);

            // Nadam update: combines current gradient with momentum term
            let m_hat = (&self.m_weights[i] * self.beta1 + &weight_grads[i] * (1.0 - self.beta1))
                / (1.0 - beta1_t);
            let update = &m_hat / (&v_corrected.mapv(|x| x.sqrt()) + self.epsilon);
            weights[i] = &weights[i] - &update * self.learning_rate;

            // Same for biases
            self.m_biases[i] = &self.m_biases[i] * self.beta1 + &bias_grads[i] * (1.0 - self.beta1);
            self.v_biases[i] = &self.v_biases[i] * self.beta2
                + &bias_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            let v_bias_corrected = &self.v_biases[i] / (1.0 - beta2_t);
            let m_bias_hat = (&self.m_biases[i] * self.beta1 + &bias_grads[i] * (1.0 - self.beta1))
                / (1.0 - beta1_t);
            let bias_update = &m_bias_hat / (&v_bias_corrected.mapv(|x| x.sqrt()) + self.epsilon);
            biases[i] = &biases[i] - &bias_update * self.learning_rate;
        }

        Ok(())
    }
}

/// LARS optimizer (Layer-wise Adaptive Rate Scaling)
/// Designed for large batch training with layer-wise adaptive learning rates
#[derive(Debug, Clone)]
pub struct LarsSolver {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub lars_coefficient: f64,  // typically 0.001
    pub epsilon: f64,           // small constant to avoid division by zero
    pub trust_coefficient: f64, // trust ratio coefficient, typically 1.0
    momentum_weights: Vec<Array2<f64>>,
    momentum_biases: Vec<Array1<f64>>,
}

impl LarsSolver {
    pub fn new(
        learning_rate: f64,
        momentum: f64,
        weight_decay: f64,
        lars_coefficient: f64,
        epsilon: f64,
        trust_coefficient: f64,
    ) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            lars_coefficient,
            epsilon,
            trust_coefficient,
            momentum_weights: Vec::new(),
            momentum_biases: Vec::new(),
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.momentum_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.momentum_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        for i in 0..weights.len() {
            // Compute norms for layer-wise adaptation
            let weight_norm = weights[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            let grad_norm = weight_grads[i].iter().map(|x| x * x).sum::<f64>().sqrt();

            // Compute layer-wise learning rate using LARS formula
            let lars_lr = if grad_norm > self.epsilon && weight_norm > self.epsilon {
                self.learning_rate * self.lars_coefficient * weight_norm
                    / (grad_norm + self.weight_decay * weight_norm)
            } else {
                self.learning_rate
            };

            // Apply trust coefficient (layer-wise trust ratio)
            let effective_lr = lars_lr * self.trust_coefficient;

            // Add weight decay to gradients
            let weight_grad_with_decay = &weight_grads[i] + &weights[i] * self.weight_decay;

            // Update momentum for weights
            self.momentum_weights[i] =
                &self.momentum_weights[i] * self.momentum + &weight_grad_with_decay * effective_lr;

            // Update weights
            weights[i] = &weights[i] - &self.momentum_weights[i];

            // For biases, use standard SGD with momentum (no LARS scaling)
            self.momentum_biases[i] =
                &self.momentum_biases[i] * self.momentum + &bias_grads[i] * self.learning_rate;
            biases[i] = &biases[i] - &self.momentum_biases[i];
        }

        Ok(())
    }
}

/// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training)
/// Combines layer-wise adaptation from LARS with Adam's moments
#[derive(Debug, Clone)]
pub struct LambSolver {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub trust_coefficient: f64, // trust ratio coefficient, typically 1.0
    m_weights: Vec<Array2<f64>>,
    v_weights: Vec<Array2<f64>>,
    m_biases: Vec<Array1<f64>>,
    v_biases: Vec<Array1<f64>>,
    t: usize,
}

impl LambSolver {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
        trust_coefficient: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            trust_coefficient,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
            t: 0,
        }
    }

    pub fn initialize(&mut self, weights: &[Array2<f64>], biases: &[Array1<f64>]) {
        self.m_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.v_weights = weights.iter().map(|w| Array2::zeros(w.dim())).collect();
        self.m_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.v_biases = biases.iter().map(|b| Array1::zeros(b.len())).collect();
        self.t = 0;
    }

    pub fn update_params(
        &mut self,
        weights: &mut [Array2<f64>],
        biases: &mut [Array1<f64>],
        weight_grads: &[Array2<f64>],
        bias_grads: &[Array1<f64>],
    ) -> NeuralResult<()> {
        self.t += 1;

        for i in 0..weights.len() {
            // Add weight decay to gradients
            let weight_grad_with_decay = &weight_grads[i] + &weights[i] * self.weight_decay;

            // Update biased first moment estimate
            self.m_weights[i] =
                &self.m_weights[i] * self.beta1 + &weight_grad_with_decay * (1.0 - self.beta1);

            // Update biased second raw moment estimate
            self.v_weights[i] = &self.v_weights[i] * self.beta2
                + &weight_grad_with_decay.mapv(|x| x * x) * (1.0 - self.beta2);

            // Bias correction
            let m_hat = &self.m_weights[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &self.v_weights[i] / (1.0 - self.beta2.powi(self.t as i32));

            // Compute Adam update
            let adam_update = &m_hat / (&v_hat.mapv(|x| x.sqrt()) + self.epsilon);

            // Compute norms for layer-wise adaptation
            let weight_norm = weights[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            let adam_update_norm = adam_update.iter().map(|x| x * x).sum::<f64>().sqrt();

            // Compute layer-wise learning rate using LAMB formula
            let trust_ratio = if weight_norm > 0.0 && adam_update_norm > 0.0 {
                (weight_norm / adam_update_norm).min(self.trust_coefficient)
            } else {
                1.0
            };

            // Apply layer-wise adaptive learning rate
            let effective_lr = self.learning_rate * trust_ratio;

            // Update weights
            weights[i] = &weights[i] - &adam_update * effective_lr;

            // For biases, use standard Adam (no layer-wise adaptation)
            self.m_biases[i] = &self.m_biases[i] * self.beta1 + &bias_grads[i] * (1.0 - self.beta1);
            self.v_biases[i] = &self.v_biases[i] * self.beta2
                + &bias_grads[i].mapv(|x| x * x) * (1.0 - self.beta2);

            let m_bias_hat = &self.m_biases[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_bias_hat = &self.v_biases[i] / (1.0 - self.beta2.powi(self.t as i32));

            let bias_update = &m_bias_hat / (&v_bias_hat.mapv(|x| x.sqrt()) + self.epsilon);
            biases[i] = &biases[i] - &bias_update * self.learning_rate;
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};
    use scirs2_core::numeric::Signed;

    #[test]
    fn test_sgd_solver_initialization() {
        let mut solver =
            SgdSolver::new(0.01, 0.9, false, LearningRateSchedule::Constant, 0.5, 0.01);

        let weights = vec![Array2::zeros((3, 2)), Array2::zeros((2, 1))];
        let biases = vec![Array1::zeros(2), Array1::zeros(1)];

        solver.initialize(&weights, &biases);

        assert_eq!(solver.velocity_weights.len(), 2);
        assert_eq!(solver.velocity_biases.len(), 2);
        assert_eq!(solver.t, 0);
    }

    #[test]
    fn test_adam_solver_initialization() {
        let mut solver = AdamSolver::new(0.001, 0.9, 0.999, 1e-8);

        let weights = vec![Array2::zeros((3, 2)), Array2::zeros((2, 1))];
        let biases = vec![Array1::zeros(2), Array1::zeros(1)];

        solver.initialize(&weights, &biases);

        assert_eq!(solver.m_weights.len(), 2);
        assert_eq!(solver.v_weights.len(), 2);
        assert_eq!(solver.m_biases.len(), 2);
        assert_eq!(solver.v_biases.len(), 2);
        assert_eq!(solver.t, 0);
    }

    #[test]
    fn test_sgd_learning_rate_schedules() {
        let solver = SgdSolver::new(0.01, 0.0, false, LearningRateSchedule::Constant, 0.5, 0.1);
        assert_abs_diff_eq!(solver.get_learning_rate(), 0.01, epsilon = 1e-10);

        let mut solver =
            SgdSolver::new(0.01, 0.0, false, LearningRateSchedule::InvScaling, 0.5, 0.1);
        solver.t = 4;
        let expected_lr = 0.1 / (4.0_f64).powf(0.5);
        assert_abs_diff_eq!(solver.get_learning_rate(), expected_lr, epsilon = 1e-10);
    }

    #[test]
    fn test_parameter_updates_change_weights() {
        let mut solver = SgdSolver::new(0.1, 0.0, false, LearningRateSchedule::Constant, 0.5, 0.01);

        let mut weights = vec![array![[1.0, 2.0], [3.0, 4.0]]];
        let mut biases = vec![array![0.5, -0.5]];
        let weight_grads = vec![array![[0.1, 0.2], [0.3, 0.4]]];
        let bias_grads = vec![array![0.1, -0.1]];

        let original_weights = weights[0].clone();
        let original_biases = biases[0].clone();

        solver.initialize(&weights, &biases);
        solver
            .update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)
            .unwrap();

        // Check that weights and biases have changed
        assert!(!weights[0].abs_diff_eq(&original_weights, 1e-10));
        assert!(!biases[0].abs_diff_eq(&original_biases, 1e-10));
    }

    #[test]
    fn test_adamw_solver_initialization() {
        let mut solver = AdamWSolver::new(0.001, 0.9, 0.999, 1e-8, 0.01);

        let weights = vec![Array2::zeros((3, 2)), Array2::zeros((2, 1))];
        let biases = vec![Array1::zeros(2), Array1::zeros(1)];

        solver.initialize(&weights, &biases);

        assert_eq!(solver.m_weights.len(), 2);
        assert_eq!(solver.v_weights.len(), 2);
        assert_eq!(solver.weight_decay, 0.01);
        assert_eq!(solver.t, 0);
    }

    #[test]
    fn test_rmsprop_solver_initialization() {
        let mut solver = RMSpropSolver::new(0.001, 0.9, 1e-8, 0.0, false);

        let weights = vec![Array2::zeros((3, 2)), Array2::zeros((2, 1))];
        let biases = vec![Array1::zeros(2), Array1::zeros(1)];

        solver.initialize(&weights, &biases);

        assert_eq!(solver.v_weights.len(), 2);
        assert_eq!(solver.v_biases.len(), 2);
        assert!(!solver.centered);
    }

    #[test]
    fn test_rmsprop_centered_initialization() {
        let mut solver = RMSpropSolver::new(0.001, 0.9, 1e-8, 0.0, true);

        let weights = vec![Array2::zeros((3, 2))];
        let biases = vec![Array1::zeros(2)];

        solver.initialize(&weights, &biases);

        assert!(solver.centered);
        assert_eq!(solver.g_weights.len(), 1);
        assert_eq!(solver.g_biases.len(), 1);
    }

    #[test]
    fn test_nadam_solver_initialization() {
        let mut solver = NadamSolver::new(0.001, 0.9, 0.999, 1e-8);

        let weights = vec![Array2::zeros((3, 2)), Array2::zeros((2, 1))];
        let biases = vec![Array1::zeros(2), Array1::zeros(1)];

        solver.initialize(&weights, &biases);

        assert_eq!(solver.m_weights.len(), 2);
        assert_eq!(solver.v_weights.len(), 2);
        assert_eq!(solver.t, 0);
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut solver = AdamWSolver::new(0.1, 0.9, 0.999, 1e-8, 0.1);

        let mut weights = vec![array![[1.0, 2.0]]];
        let mut biases = vec![array![0.5]];
        let weight_grads = vec![array![[0.0, 0.0]]]; // Zero gradients to isolate weight decay
        let bias_grads = vec![array![0.0]];

        let original_weight_magnitude = weights[0].iter().map(|x| x.abs()).sum::<f64>();

        solver.initialize(&weights, &biases);
        solver
            .update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)
            .unwrap();

        let new_weight_magnitude = weights[0].iter().map(|x| x.abs()).sum::<f64>();

        // Weight magnitude should decrease due to weight decay
        assert!(new_weight_magnitude < original_weight_magnitude);
    }

    #[test]
    fn test_lars_solver() {
        let mut solver = LarsSolver::new(0.1, 0.9, 0.01, 0.001, 1e-8, 1.0);

        let mut weights = vec![array![[1.0, 2.0], [3.0, 4.0]]];
        let mut biases = vec![array![0.5, -0.5]];
        let weight_grads = vec![array![[0.1, 0.2], [0.3, 0.4]]];
        let bias_grads = vec![array![0.1, -0.1]];

        let original_weights = weights[0].clone();
        let original_biases = biases[0].clone();

        solver.initialize(&weights, &biases);
        solver
            .update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)
            .unwrap();

        // Check that parameters have changed
        assert!(!weights[0].abs_diff_eq(&original_weights, 1e-10));
        assert!(!biases[0].abs_diff_eq(&original_biases, 1e-10));
    }

    #[test]
    fn test_lamb_solver() {
        let mut solver = LambSolver::new(0.001, 0.9, 0.999, 1e-8, 0.01, 1.0);

        let mut weights = vec![array![[1.0, 2.0], [3.0, 4.0]]];
        let mut biases = vec![array![0.5, -0.5]];
        let weight_grads = vec![array![[0.1, 0.2], [0.3, 0.4]]];
        let bias_grads = vec![array![0.1, -0.1]];

        let original_weights = weights[0].clone();
        let original_biases = biases[0].clone();

        solver.initialize(&weights, &biases);
        solver
            .update_params(&mut weights, &mut biases, &weight_grads, &bias_grads)
            .unwrap();

        // Check that parameters have changed
        assert!(!weights[0].abs_diff_eq(&original_weights, 1e-10));
        assert!(!biases[0].abs_diff_eq(&original_biases, 1e-10));
    }

    #[test]
    fn test_gradient_clipping_by_norm() {
        let mut weight_grads = vec![array![[10.0, 20.0], [30.0, 40.0]]];
        let mut bias_grads = vec![array![5.0, -5.0]];

        // Compute original norm
        let original_norm = (weight_grads[0].iter().map(|x| x * x).sum::<f64>()
            + bias_grads[0].iter().map(|x| x * x).sum::<f64>())
        .sqrt();

        let max_norm = 10.0;
        GradientClipper::clip_gradients(
            &mut weight_grads,
            &mut bias_grads,
            GradientClipping::ByNorm(max_norm),
        );

        // Compute new norm
        let new_norm = (weight_grads[0].iter().map(|x| x * x).sum::<f64>()
            + bias_grads[0].iter().map(|x| x * x).sum::<f64>())
        .sqrt();

        assert!(original_norm > max_norm);
        assert_abs_diff_eq!(new_norm, max_norm, epsilon = 1e-6);
    }

    #[test]
    fn test_gradient_clipping_by_value() {
        let mut weight_grads = vec![array![[10.0, -15.0], [20.0, -25.0]]];
        let mut bias_grads = vec![array![12.0, -8.0]];

        let max_value = 5.0;
        GradientClipper::clip_gradients(
            &mut weight_grads,
            &mut bias_grads,
            GradientClipping::ByValue(max_value),
        );

        // Check that all values are within [-max_value, max_value]
        for grad in &weight_grads {
            for &value in grad.iter() {
                assert!(value >= -max_value && value <= max_value);
            }
        }

        for grad in &bias_grads {
            for &value in grad.iter() {
                assert!(value >= -max_value && value <= max_value);
            }
        }
    }

    #[test]
    fn test_learning_rate_schedules() {
        // Test exponential decay
        let mut solver = SgdSolver::new_with_schedule_params(
            1.0,
            0.0,
            false,
            LearningRateSchedule::ExponentialDecay,
            0.5,
            0.1,
            0.9,
            100,
            1000,
            0.0,
            1.0,
            500,
            0.1,
            100,
            1000,
        );
        solver.t = 100;
        let lr = solver.get_learning_rate();
        let expected = 1.0 * 0.9_f64.powf(100.0 / 100.0);
        assert_abs_diff_eq!(lr, expected, epsilon = 1e-6);

        // Test cosine annealing
        let mut solver = SgdSolver::new_with_schedule_params(
            1.0,
            0.0,
            false,
            LearningRateSchedule::CosineAnnealing,
            0.5,
            0.1,
            0.9,
            100,
            1000,
            0.0,
            1.0,
            500,
            0.1,
            100,
            1000,
        );
        solver.t = 500;
        let lr = solver.get_learning_rate();
        let progress = 500.0 / 1000.0;
        let expected = 0.0 + (1.0 - 0.0) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        assert_abs_diff_eq!(lr, expected, epsilon = 1e-6);

        // Test warmup
        let mut solver = SgdSolver::new_with_schedule_params(
            1.0,
            0.0,
            false,
            LearningRateSchedule::Warmup,
            0.5,
            0.1,
            0.9,
            100,
            1000,
            0.0,
            1.0,
            500,
            0.1,
            100,
            1000,
        );
        solver.t = 50;
        let lr = solver.get_learning_rate();
        let expected = 1.0 * (50.0 / 100.0);
        assert_abs_diff_eq!(lr, expected, epsilon = 1e-6);
    }
}
