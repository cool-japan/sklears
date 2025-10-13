//! Dictionary initialization strategies

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{thread_rng, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result, types::Float};

/// Initialization strategies
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InitializationStrategy {
    Random,
    DataDriven,
    Orthogonal,
    SVD,
}

/// Initialization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InitializationConfig {
    pub strategy: InitializationStrategy,
    pub scale: Float,
    pub normalize: bool,
}

impl Default for InitializationConfig {
    fn default() -> Self {
        Self {
            strategy: InitializationStrategy::Random,
            scale: 1.0,
            normalize: true,
        }
    }
}

/// Dictionary initializer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictionaryInitializer {
    config: InitializationConfig,
}

/// Random initializer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RandomInitializer {
    scale: Float,
}

/// Data-driven initializer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataDrivenInitializer {
    normalize: bool,
}

/// Orthogonal initializer
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrthogonalInitializer {
    scale: Float,
}

impl DictionaryInitializer {
    pub fn new(config: InitializationConfig) -> Self {
        Self { config }
    }

    pub fn initialize(
        &self,
        n_atoms: usize,
        n_features: usize,
        data: Option<&Array2<Float>>,
    ) -> Result<Array2<Float>> {
        let mut rng = thread_rng();
        let mut dictionary = Array2::zeros((n_atoms, n_features));

        // Simple random initialization
        for i in 0..n_atoms {
            for j in 0..n_features {
                dictionary[[i, j]] = (rng.gen::<Float>() - 0.5) * 2.0 * self.config.scale;
            }
        }

        if self.config.normalize {
            // Normalize atoms
            for mut row in dictionary.rows_mut() {
                let norm = row.mapv(|x| x * x).sum().sqrt();
                if norm > 0.0 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
        }

        Ok(dictionary)
    }
}
