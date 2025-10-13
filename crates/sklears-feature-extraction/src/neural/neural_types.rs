pub use scirs2_core::ndarray::{s, Array1, Array2, Axis};
pub use scirs2_core::random::Random;
pub use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError, Transform},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy)]
pub enum CNNActivation {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum AttentionType {
    /// Standard scaled dot-product attention
    Scaled,
    /// Additive attention (Bahdanau-style)
    Additive,
    /// Multiplicative attention
    Multiplicative,
    /// Multi-head attention
    MultiHead,
}

#[derive(Clone)]
pub struct TransformerLayerWeights {
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
    pub w_ff1: Array2<f64>,
    pub w_ff2: Array2<f64>,
    pub layer_norm_scale: Array1<f64>,
    pub layer_norm_bias: Array1<f64>,
}