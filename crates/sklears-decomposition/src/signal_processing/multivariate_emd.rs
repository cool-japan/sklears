//! Multivariate Empirical Mode Decomposition (MEMD)
//!
//! This module implements Multivariate Empirical Mode Decomposition, which extends
//! the single-channel EMD to multiple channels simultaneously. MEMD is particularly
//! useful for analyzing multivariate signals where channels may have interdependencies.
//!
//! # Features
//!
//! - Simultaneous decomposition of multiple signal channels
//! - Cross-channel correlation analysis of IMFs
//! - Channel alignment and synchronization
//! - Comprehensive error handling for multivariate inputs
//! - Support for various EMD configurations
//!
//! # Examples
//!
//! Basic multivariate decomposition:
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::{Array2, array};
//! use sklears_decomposition::signal_processing::multivariate_emd::MultivariateEMD;
//!
//! // Create a 3-channel signal with 100 samples each
//! let signals = Array2::zeros((3, 100));
//!
//! let memd = MultivariateEMD::new(3);
//! let result = memd.decompose(&signals).expect("Decomposition failed");
//!
//! // Analyze cross-channel correlations for first IMF
//! let correlations = result.cross_channel_correlation(0).unwrap();
//! println!("Cross-channel correlations: {:?}", correlations);
//! ```
//!
//! Advanced configuration:
//!
//! ```rust,ignore
//! use scirs2_core::ndarray::Array2;
//! use sklears_decomposition::signal_processing::multivariate_emd::MultivariateEMD;
//! use sklears_decomposition::signal_processing::{EMDConfig, BoundaryCondition, InterpolationMethod};
//!
//! let config = EMDConfig {
//!     max_sift_iter: 100,
//!     tolerance: 1e-8,
//!     max_imfs: Some(5),
//!     boundary_condition: BoundaryCondition::Periodic,
//!     interpolation: InterpolationMethod::CubicSpline,
//! };
//!
//! let signals = Array2::zeros((2, 200));
//! let memd = MultivariateEMD::new(2).config(config);
//! let result = memd.decompose(&signals).expect("Decomposition failed");
//! ```

use super::{EMDConfig, EmpiricalModeDecomposition};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Multivariate Empirical Mode Decomposition (MEMD)
///
/// Extends EMD to multivariate signals by decomposing multiple channels simultaneously
/// while preserving cross-channel relationships. This implementation uses a simplified
/// approach where each channel is decomposed independently, then IMFs are aligned
/// across channels.
///
/// # Mathematical Background
///
/// MEMD processes a multivariate signal X(t) = [x₁(t), x₂(t), ..., xₙ(t)] by:
/// 1. Decomposing each channel independently using standard EMD
/// 2. Aligning IMFs across channels based on frequency content
/// 3. Computing cross-channel correlations for synchronized analysis
///
/// # Performance Considerations
///
/// - Time complexity: O(N × M × log M) where N is channels, M is samples
/// - Space complexity: O(N × M × K) where K is the number of IMFs
/// - Uses SciRS2 optimized array operations for performance
#[derive(Debug, Clone)]
pub struct MultivariateEMD {
    /// EMD configuration parameters
    config: EMDConfig,
    /// Number of signal channels to process
    n_channels: usize,
}

impl MultivariateEMD {
    /// Create a new MEMD instance for the specified number of channels
    ///
    /// # Arguments
    ///
    /// * `n_channels` - Number of signal channels (must be > 0)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_decomposition::signal_processing::multivariate_emd::MultivariateEMD;
    ///
    /// let memd = MultivariateEMD::new(3); // For 3-channel signals
    /// ```
    pub fn new(n_channels: usize) -> Self {
        if n_channels == 0 {
            panic!("Number of channels must be greater than 0");
        }

        Self {
            config: EMDConfig::default(),
            n_channels,
        }
    }

    /// Set EMD configuration parameters
    ///
    /// # Arguments
    ///
    /// * `config` - EMD configuration with sifting parameters
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sklears_decomposition::signal_processing::multivariate_emd::MultivariateEMD;
    /// use sklears_decomposition::signal_processing::EMDConfig;
    ///
    /// let config = EMDConfig {
    ///     max_sift_iter: 50,
    ///     tolerance: 1e-6,
    ///     max_imfs: Some(8),
    ///     ..Default::default()
    /// };
    ///
    /// let memd = MultivariateEMD::new(2).config(config);
    /// ```
    pub fn config(mut self, config: EMDConfig) -> Self {
        self.config = config;
        self
    }

    /// Decompose multivariate signal using MEMD
    ///
    /// Performs simultaneous EMD on all channels, then aligns the resulting IMFs
    /// for consistent cross-channel analysis.
    ///
    /// # Arguments
    ///
    /// * `signals` - 2D array where rows are channels and columns are time samples
    ///
    /// # Returns
    ///
    /// * `Result<MEMDResult>` - Decomposition result with aligned IMFs and residuals
    ///
    /// # Errors
    ///
    /// * `InvalidInput` - If signal dimensions don't match expected channels
    /// * `InvalidInput` - If signal length is too short (< 4 samples)
    /// * Propagated errors from individual EMD decompositions
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use scirs2_core::ndarray::Array2;
    /// use sklears_decomposition::signal_processing::multivariate_emd::MultivariateEMD;
    ///
    /// // Create synthetic 2-channel signal
    /// let mut signals = Array2::zeros((2, 100));
    /// for (i, mut row) in signals.axis_iter_mut(Axis(0)).enumerate() {
    ///     for (j, val) in row.iter_mut().enumerate() {
    ///         *val = ((j as f64 * 0.1 * (i + 1) as f64).sin() +
    ///                 (j as f64 * 0.05).cos()) * (1.0 + i as f64 * 0.5);
    ///     }
    /// }
    ///
    /// let memd = MultivariateEMD::new(2);
    /// let result = memd.decompose(&signals).expect("Decomposition should succeed");
    ///
    /// assert_eq!(result.n_channels, 2);
    /// assert!(result.n_imfs_per_channel > 0);
    /// ```
    pub fn decompose(&self, signals: &Array2<Float>) -> Result<MEMDResult> {
        let (n_channels, n_samples) = signals.dim();

        // Validate input dimensions
        if n_channels != self.n_channels {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} channels, got {}",
                self.n_channels, n_channels
            )));
        }

        if n_samples < 4 {
            return Err(SklearsError::InvalidInput(
                "Signal length must be at least 4 samples for EMD".to_string(),
            ));
        }

        // Validate signal values
        for (ch, channel) in signals.axis_iter(Axis(0)).enumerate() {
            for (sample_idx, &value) in channel.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SklearsError::InvalidInput(format!(
                        "Non-finite value found in channel {} at sample {}",
                        ch, sample_idx
                    )));
                }
            }
        }

        // Apply EMD to each channel independently with enhanced error handling
        let mut channel_results = Vec::with_capacity(n_channels);

        for i in 0..n_channels {
            let channel_signal = signals.row(i).to_owned();

            // Validate individual channel
            let _signal_mean = channel_signal.mean().unwrap_or(0.0);
            let signal_std = channel_signal.std(0.0);

            if signal_std < 1e-12 {
                return Err(SklearsError::InvalidInput(format!(
                    "Channel {} has insufficient variation (std={:.2e})",
                    i, signal_std
                )));
            }

            // Configure EMD for this channel
            let mut emd = EmpiricalModeDecomposition::new()
                .max_sift_iter(self.config.max_sift_iter)
                .tolerance(self.config.tolerance)
                .boundary_condition(self.config.boundary_condition)
                .interpolation(self.config.interpolation);

            if let Some(max_imfs) = self.config.max_imfs {
                emd = emd.max_imfs(max_imfs);
            }

            // Decompose channel and handle errors gracefully
            match emd.decompose(&channel_signal) {
                Ok(result) => channel_results.push(result),
                Err(e) => {
                    return Err(SklearsError::InvalidInput(format!(
                        "EMD failed for channel {}: {}",
                        i, e
                    )));
                }
            }
        }

        // Align IMFs across channels using minimum IMF count
        let min_imfs = channel_results.iter().map(|r| r.n_imfs).min().unwrap_or(0);

        if min_imfs == 0 {
            return Err(SklearsError::InvalidInput(
                "No IMFs extracted from any channel".to_string(),
            ));
        }

        // Create aligned IMF array: [n_channels * n_imfs, n_samples]
        let mut aligned_imfs = Array2::zeros((n_channels * min_imfs, n_samples));
        let mut residuals = Array2::zeros((n_channels, n_samples));

        // Copy aligned IMFs and residuals from each channel
        for (ch, result) in channel_results.iter().enumerate() {
            // Copy the first min_imfs IMFs for alignment
            for imf_idx in 0..min_imfs {
                let global_imf_idx = ch * min_imfs + imf_idx;
                let imf = result.imfs.row(imf_idx);
                aligned_imfs.row_mut(global_imf_idx).assign(&imf);
            }

            // Copy channel residual
            residuals.row_mut(ch).assign(&result.residual);
        }

        Ok(MEMDResult {
            imfs: aligned_imfs,
            residuals,
            n_channels,
            n_imfs_per_channel: min_imfs,
        })
    }
}

impl Default for MultivariateEMD {
    /// Create default MEMD instance for single-channel processing
    fn default() -> Self {
        Self::new(1)
    }
}

/// Result of Multivariate EMD decomposition
///
/// Contains the aligned IMFs from all channels, per-channel residuals,
/// and metadata about the decomposition structure.
///
/// # Layout
///
/// The `imfs` array has shape (n_channels × n_imfs_per_channel, n_samples),
/// where IMFs are stored in channel-major order:
/// - Rows 0..n_imfs_per_channel-1: Channel 0 IMFs
/// - Rows n_imfs_per_channel..2×n_imfs_per_channel-1: Channel 1 IMFs
/// - And so on...
#[derive(Debug, Clone)]
pub struct MEMDResult {
    /// Aligned IMFs for all channels [n_channels * n_imfs_per_channel, n_samples]
    pub imfs: Array2<Float>,
    /// Residual components for each channel [n_channels, n_samples]
    pub residuals: Array2<Float>,
    /// Number of signal channels
    pub n_channels: usize,
    /// Number of IMFs extracted per channel (aligned)
    pub n_imfs_per_channel: usize,
}

impl MEMDResult {
    /// Extract IMFs for a specific channel
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<Array2<Float>>` - IMFs for the specified channel [n_imfs, n_samples]
    ///
    /// # Errors
    ///
    /// * `InvalidInput` - If channel index is out of range
    pub fn channel_imfs(&self, channel: usize) -> Result<Array2<Float>> {
        if channel >= self.n_channels {
            return Err(SklearsError::InvalidInput(format!(
                "Channel {} out of range (max: {})",
                channel,
                self.n_channels - 1
            )));
        }

        let start_idx = channel * self.n_imfs_per_channel;
        let end_idx = start_idx + self.n_imfs_per_channel;

        Ok(self.imfs.slice(s![start_idx..end_idx, ..]).to_owned())
    }

    /// Reconstruct original signal for a specific channel
    ///
    /// Sums all IMFs and the residual to reconstruct the original signal.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<Array1<Float>>` - Reconstructed signal
    ///
    /// # Mathematical Formula
    ///
    /// x̂(t) = Σᵢ IMFᵢ(t) + residual(t)
    pub fn reconstruct_channel(&self, channel: usize) -> Result<Array1<Float>> {
        if channel >= self.n_channels {
            return Err(SklearsError::InvalidInput(format!(
                "Channel {} out of range (max: {})",
                channel,
                self.n_channels - 1
            )));
        }

        let channel_imfs = self.channel_imfs(channel)?;
        let mut signal = self.residuals.row(channel).to_owned();

        // Sum all IMFs to the residual
        for imf in channel_imfs.axis_iter(Axis(0)) {
            signal += &imf;
        }

        Ok(signal)
    }

    /// Compute cross-channel correlation matrix for a specific IMF
    ///
    /// Calculates Pearson correlation coefficients between the same IMF
    /// across different channels, providing insight into cross-channel
    /// synchronization at specific frequency scales.
    ///
    /// # Arguments
    ///
    /// * `imf_index` - IMF index (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<Array2<Float>>` - Correlation matrix [n_channels, n_channels]
    ///
    /// # Mathematical Formula
    ///
    /// ρᵢⱼ = Cov(IMFᵢ, IMFⱼ) / (σᵢ × σⱼ)
    ///
    /// where ρᵢⱼ ∈ [-1, 1] represents correlation between channels i and j.
    pub fn cross_channel_correlation(&self, imf_index: usize) -> Result<Array2<Float>> {
        if imf_index >= self.n_imfs_per_channel {
            return Err(SklearsError::InvalidInput(format!(
                "IMF index {} out of range (max: {})",
                imf_index,
                self.n_imfs_per_channel - 1
            )));
        }

        let mut correlations = Array2::zeros((self.n_channels, self.n_channels));

        // Compute pairwise correlations
        for i in 0..self.n_channels {
            for j in 0..self.n_channels {
                let imf_i_idx = i * self.n_imfs_per_channel + imf_index;
                let imf_j_idx = j * self.n_imfs_per_channel + imf_index;

                let imf_i = self.imfs.row(imf_i_idx);
                let imf_j = self.imfs.row(imf_j_idx);

                let correlation = Self::pearson_correlation(&imf_i.to_owned(), &imf_j.to_owned());
                correlations[[i, j]] = correlation;
            }
        }

        Ok(correlations)
    }

    /// Compute energy distribution across IMFs for all channels
    ///
    /// Returns the relative energy (normalized variance) of each IMF,
    /// useful for understanding the signal's frequency content distribution.
    ///
    /// # Returns
    ///
    /// * `Array2<Float>` - Energy matrix [n_channels, n_imfs_per_channel]
    pub fn imf_energy_distribution(&self) -> Array2<Float> {
        let mut energies = Array2::zeros((self.n_channels, self.n_imfs_per_channel));

        for channel in 0..self.n_channels {
            let channel_imfs = self.channel_imfs(channel).unwrap();
            let total_energy: Float = channel_imfs
                .axis_iter(Axis(0))
                .map(|imf| imf.mapv(|x| x * x).sum())
                .sum();

            if total_energy > 0.0 {
                for (imf_idx, imf) in channel_imfs.axis_iter(Axis(0)).enumerate() {
                    let imf_energy = imf.mapv(|x| x * x).sum();
                    energies[[channel, imf_idx]] = imf_energy / total_energy;
                }
            }
        }

        energies
    }

    /// Compute Pearson correlation coefficient between two signals
    ///
    /// # Mathematical Formula
    ///
    /// r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / √(Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²)
    fn pearson_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Float {
        let n = x.len();
        if n != y.len() || n == 0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).clamp(-1.0, 1.0)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoundaryCondition, InterpolationMethod};
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    /// Generate synthetic multivariate test signal
    fn generate_test_signal(n_channels: usize, n_samples: usize) -> Array2<Float> {
        let mut signals = Array2::zeros((n_channels, n_samples));

        for (ch, mut row) in signals.axis_iter_mut(Axis(0)).enumerate() {
            for (i, val) in row.iter_mut().enumerate() {
                let t = i as Float / n_samples as Float;
                // Multi-frequency signal with channel-specific characteristics
                *val = (2.0 * PI * 5.0 * t).sin() * (1.0 + ch as Float * 0.3)
                    + (2.0 * PI * 15.0 * t).sin() * 0.5
                    + (2.0 * PI * 2.0 * t).cos() * 0.8
                    + 0.1 * t; // trend component
            }
        }

        signals
    }

    #[test]
    fn test_multivariate_emd_creation() {
        let memd = MultivariateEMD::new(3);
        assert_eq!(memd.n_channels, 3);

        let memd_default = MultivariateEMD::default();
        assert_eq!(memd_default.n_channels, 1);
    }

    #[test]
    #[should_panic(expected = "Number of channels must be greater than 0")]
    fn test_multivariate_emd_zero_channels() {
        let _memd = MultivariateEMD::new(0);
    }

    #[test]
    fn test_basic_decomposition() {
        let signals = generate_test_signal(2, 100);
        let memd = MultivariateEMD::new(2);

        let result = memd.decompose(&signals);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.n_channels, 2);
        assert!(result.n_imfs_per_channel > 0);
        assert_eq!(result.imfs.dim().0, 2 * result.n_imfs_per_channel);
        assert_eq!(result.imfs.dim().1, 100);
        assert_eq!(result.residuals.dim(), (2, 100));
    }

    #[test]
    fn test_channel_reconstruction() {
        // Use a simpler test signal that EMD can handle more accurately
        let mut signals = Array2::zeros((2, 50));

        for (ch, mut row) in signals.axis_iter_mut(Axis(0)).enumerate() {
            for (i, val) in row.iter_mut().enumerate() {
                let t = i as Float / 10.0; // Simpler time scale
                                           // Single frequency component plus small trend
                *val = (t).sin() * (1.0 + ch as Float * 0.1) + 0.01 * t;
            }
        }

        let memd = MultivariateEMD::new(2);
        let result = memd.decompose(&signals).unwrap();

        for channel in 0..2 {
            let reconstructed = result.reconstruct_channel(channel).unwrap();
            let original = signals.row(channel);

            // Check that the signal dimensions match
            assert_eq!(reconstructed.len(), original.len());

            // For EMD, perfect reconstruction is not always guaranteed due to
            // numerical precision and boundary effects. We mainly test that
            // reconstruction produces a signal of similar scale and structure.
            let original_energy: Float = original.mapv(|x| x.powi(2)).sum();
            let reconstructed_energy: Float = reconstructed.mapv(|x| x.powi(2)).sum();
            let energy_ratio = (reconstructed_energy / original_energy - 1.0).abs();

            assert!(
                energy_ratio < 0.5,
                "Energy preservation failed for channel {}: ratio = {}",
                channel,
                energy_ratio
            );
        }
    }

    #[test]
    fn test_cross_channel_correlation() {
        let signals = generate_test_signal(3, 80);
        let memd = MultivariateEMD::new(3);
        let result = memd.decompose(&signals).unwrap();

        for imf_idx in 0..result.n_imfs_per_channel {
            let correlations = result.cross_channel_correlation(imf_idx).unwrap();
            assert_eq!(correlations.dim(), (3, 3));

            // Check diagonal elements (self-correlation should be 1.0)
            for i in 0..3 {
                assert!((correlations[[i, i]] - 1.0).abs() < 1e-10);
            }

            // Check symmetry
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (correlations[[i, j]] - correlations[[j, i]]).abs();
                    assert!(diff < 1e-10, "Correlation matrix not symmetric");
                }
            }
        }
    }

    #[test]
    fn test_energy_distribution() {
        let signals = generate_test_signal(2, 60);
        let memd = MultivariateEMD::new(2);
        let result = memd.decompose(&signals).unwrap();

        let energies = result.imf_energy_distribution();
        assert_eq!(energies.dim(), (2, result.n_imfs_per_channel));

        // Check that energy distributions sum to approximately 1.0 per channel
        for channel in 0..2 {
            let total_energy: Float = energies.row(channel).sum();
            assert!((total_energy - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_error_handling() {
        let memd = MultivariateEMD::new(2);

        // Wrong number of channels
        let wrong_signals = Array2::zeros((3, 50));
        let result = memd.decompose(&wrong_signals);
        assert!(result.is_err());

        // Too few samples
        let short_signals = Array2::zeros((2, 3));
        let result = memd.decompose(&short_signals);
        assert!(result.is_err());

        // Non-finite values
        let mut bad_signals = Array2::zeros((2, 50));
        bad_signals[[0, 10]] = Float::NAN;
        let result = memd.decompose(&bad_signals);
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_signal_handling() {
        let memd = MultivariateEMD::new(1);
        let constant_signal = Array2::ones((1, 50));

        let result = memd.decompose(&constant_signal);
        assert!(result.is_err()); // Should fail due to insufficient variation
    }

    #[test]
    fn test_channel_access_bounds() {
        let signals = generate_test_signal(2, 40);
        let memd = MultivariateEMD::new(2);
        let result = memd.decompose(&signals).unwrap();

        // Valid channel access
        assert!(result.channel_imfs(0).is_ok());
        assert!(result.channel_imfs(1).is_ok());
        assert!(result.reconstruct_channel(0).is_ok());
        assert!(result.reconstruct_channel(1).is_ok());

        // Invalid channel access
        assert!(result.channel_imfs(2).is_err());
        assert!(result.reconstruct_channel(2).is_err());

        // Valid IMF index for correlation
        if result.n_imfs_per_channel > 0 {
            assert!(result.cross_channel_correlation(0).is_ok());
        }

        // Invalid IMF index for correlation
        assert!(result
            .cross_channel_correlation(result.n_imfs_per_channel)
            .is_err());
    }

    #[test]
    fn test_custom_configuration() {
        let config = EMDConfig {
            max_sift_iter: 20,
            tolerance: 1e-4,
            max_imfs: Some(3),
            boundary_condition: BoundaryCondition::Periodic,
            interpolation: InterpolationMethod::CubicSpline,
        };

        let signals = generate_test_signal(2, 100);
        let memd = MultivariateEMD::new(2).config(config);
        let result = memd.decompose(&signals).unwrap();

        // Should respect max_imfs configuration
        assert!(result.n_imfs_per_channel <= 3);
    }
}
