//! Evaluation and metrics for dictionary learning

use scirs2_core::ndarray::Array2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result, types::Float};

/// Reconstruction error metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReconstructionError {
    pub mse: Float,
    pub rmse: Float,
    pub mae: Float,
    pub relative_error: Float,
}

/// Sparsity metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparsityMetrics {
    pub l0_norm: Float,
    pub l1_norm: Float,
    pub sparsity_ratio: Float,
    pub gini_coefficient: Float,
}

/// Coherence metrics for dictionary quality
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CoherenceMetrics {
    pub mutual_coherence: Float,
    pub spark: usize,
    pub condition_number: Float,
}

/// Dictionary quality assessment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictionaryQuality {
    pub reconstruction_error: ReconstructionError,
    pub sparsity_metrics: SparsityMetrics,
    pub coherence_metrics: CoherenceMetrics,
}

/// Dictionary metrics calculator
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictionaryMetrics;

impl DictionaryMetrics {
    pub fn new() -> Self {
        Self
    }

    pub fn compute_reconstruction_error(
        &self,
        original: &Array2<Float>,
        reconstructed: &Array2<Float>,
    ) -> Result<ReconstructionError> {
        let diff = original - reconstructed;
        let squared_diff = diff.mapv(|x| x * x);
        let mse = squared_diff.mean().unwrap_or(0.0);
        let rmse = mse.sqrt();

        let abs_diff = diff.mapv(|x| x.abs());
        let mae = abs_diff.mean().unwrap_or(0.0);

        let original_norm = original.mapv(|x| x * x).sum().sqrt();
        let relative_error = if original_norm > 0.0 {
            diff.mapv(|x| x * x).sum().sqrt() / original_norm
        } else {
            0.0
        };

        Ok(ReconstructionError {
            mse,
            rmse,
            mae,
            relative_error,
        })
    }

    pub fn compute_sparsity_metrics(&self, codes: &Array2<Float>) -> Result<SparsityMetrics> {
        let total_elements = codes.len() as Float;
        let non_zero_count = codes.iter().filter(|&&x| x.abs() > 1e-10).count() as Float;

        let l0_norm = non_zero_count;
        let l1_norm = codes.mapv(|x| x.abs()).sum();
        let sparsity_ratio = 1.0 - (non_zero_count / total_elements);
        let gini_coefficient = 0.0; // Simplified placeholder

        Ok(SparsityMetrics {
            l0_norm,
            l1_norm,
            sparsity_ratio,
            gini_coefficient,
        })
    }

    pub fn compute_coherence_metrics(
        &self,
        dictionary: &Array2<Float>,
    ) -> Result<CoherenceMetrics> {
        let n_atoms = dictionary.nrows();

        // Compute mutual coherence (simplified)
        let mut max_coherence: Float = 0.0;
        for i in 0..n_atoms {
            for j in (i + 1)..n_atoms {
                let atom_i = dictionary.row(i);
                let atom_j = dictionary.row(j);
                let coherence = atom_i.dot(&atom_j).abs();
                max_coherence = max_coherence.max(coherence);
            }
        }

        Ok(CoherenceMetrics {
            mutual_coherence: max_coherence,
            spark: n_atoms,        // Simplified
            condition_number: 1.0, // Placeholder
        })
    }

    pub fn compute_quality(
        &self,
        original: &Array2<Float>,
        dictionary: &Array2<Float>,
        codes: &Array2<Float>,
    ) -> Result<DictionaryQuality> {
        // Reconstruct data
        let reconstructed = codes.dot(dictionary);

        let reconstruction_error = self.compute_reconstruction_error(original, &reconstructed)?;
        let sparsity_metrics = self.compute_sparsity_metrics(codes)?;
        let coherence_metrics = self.compute_coherence_metrics(dictionary)?;

        Ok(DictionaryQuality {
            reconstruction_error,
            sparsity_metrics,
            coherence_metrics,
        })
    }
}

impl Default for DictionaryMetrics {
    fn default() -> Self {
        Self::new()
    }
}
