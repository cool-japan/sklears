//! Real-time and streaming decomposition algorithms
//!
//! This module provides streaming implementations of decomposition algorithms
//! for real-time processing of data streams.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, SVD};
use scirs2_core::ndarray::{Array1, Array2, Axis, Zip};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::VecDeque;

/// Configuration for streaming decomposition algorithms
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Window size for streaming algorithms
    pub window_size: usize,
    /// Forgetting factor for exponential weighting
    pub forgetting_factor: Float,
    /// Update frequency (number of samples before update)
    pub update_frequency: usize,
    /// Minimum number of samples before first update
    pub min_samples: usize,
    /// Enable concept drift detection
    pub detect_drift: bool,
    /// Drift detection threshold
    pub drift_threshold: Float,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            forgetting_factor: 0.95,
            update_frequency: 10,
            min_samples: 50,
            detect_drift: true,
            drift_threshold: 0.1,
        }
    }
}

/// Streaming PCA for real-time dimensionality reduction
#[derive(Debug, Clone)]
pub struct StreamingPCA {
    config: StreamingConfig,
    n_components: usize,
    // Current state
    components: Option<Array2<Float>>,
    explained_variance: Option<Array1<Float>>,
    mean: Option<Array1<Float>>,
    n_samples_seen: usize,
    // Streaming state
    data_buffer: VecDeque<Array1<Float>>,
    covariance_matrix: Option<Array2<Float>>,
    sum_weights: Float,
    // Drift detection
    previous_components: Option<Array2<Float>>,
    drift_detected: bool,
}

impl StreamingPCA {
    /// Create a new streaming PCA
    pub fn new(n_components: usize) -> Self {
        Self {
            config: StreamingConfig::default(),
            n_components,
            components: None,
            explained_variance: None,
            mean: None,
            n_samples_seen: 0,
            data_buffer: VecDeque::new(),
            covariance_matrix: None,
            sum_weights: 0.0,
            previous_components: None,
            drift_detected: false,
        }
    }

    /// Configure the streaming PCA
    pub fn with_config(mut self, config: StreamingConfig) -> Self {
        self.config = config;
        self
    }

    /// Update the PCA with a new data sample
    pub fn partial_fit(&mut self, sample: &Array1<Float>) -> Result<()> {
        let n_features = sample.len();

        // Initialize if first sample
        if self.mean.is_none() {
            self.mean = Some(Array1::zeros(n_features));
            self.covariance_matrix = Some(Array2::zeros((n_features, n_features)));
        }

        // Add to buffer
        self.data_buffer.push_back(sample.clone());
        if self.data_buffer.len() > self.config.window_size {
            self.data_buffer.pop_front();
        }

        self.n_samples_seen += 1;

        // Update running statistics
        self.update_statistics(sample)?;

        // Update decomposition if enough samples
        if self.n_samples_seen >= self.config.min_samples
            && self.n_samples_seen % self.config.update_frequency == 0
        {
            self.update_decomposition()?;
        }

        Ok(())
    }

    /// Update running mean and covariance
    fn update_statistics(&mut self, sample: &Array1<Float>) -> Result<()> {
        let weight = 1.0 / (self.n_samples_seen as Float);
        let alpha = self.config.forgetting_factor;

        if let Some(ref mut mean) = self.mean {
            let delta = sample - &*mean;
            mean.zip_mut_with(&delta, |m, d| *m += d * weight);

            if let Some(ref mut cov) = self.covariance_matrix {
                let delta_mean = sample - &*mean;
                for i in 0..sample.len() {
                    for j in 0..sample.len() {
                        cov[[i, j]] =
                            alpha * cov[[i, j]] + (1.0 - alpha) * delta_mean[i] * delta_mean[j];
                    }
                }
            }
        }

        Ok(())
    }

    /// Update the PCA decomposition
    fn update_decomposition(&mut self) -> Result<()> {
        if let Some(ref cov) = self.covariance_matrix {
            // Eigendecomposition of covariance matrix
            let (eigenvalues, eigenvectors) = cov
                .eigh(scirs2_core::ndarray::ndarray_linalg::UPLO::Upper)
                .map_err(|e| {
                    SklearsError::InvalidOperation(format!("Eigendecomposition failed: {:?}", e))
                })?;

            // Sort by eigenvalue (descending)
            let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
            indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

            // Select top components
            let n_components = self.n_components.min(eigenvalues.len());
            let mut components = Array2::<Float>::zeros((n_components, cov.nrows()));
            let mut explained_variance = Array1::<Float>::zeros(n_components);

            for (i, &idx) in indices.iter().take(n_components).enumerate() {
                components.row_mut(i).assign(&eigenvectors.column(idx));
                explained_variance[i] = eigenvalues[idx].max(0.0);
            }

            // Detect concept drift
            if self.config.detect_drift {
                self.detect_concept_drift(&components)?;
            }

            self.components = Some(components);
            self.explained_variance = Some(explained_variance);
        }

        Ok(())
    }

    /// Detect concept drift in the data
    fn detect_concept_drift(&mut self, new_components: &Array2<Float>) -> Result<()> {
        if let Some(ref prev_components) = self.previous_components {
            let similarity = self.compute_subspace_similarity(prev_components, new_components)?;

            if similarity < (1.0 - self.config.drift_threshold) {
                self.drift_detected = true;
                println!("Concept drift detected! Similarity: {similarity:.4}");
            } else {
                self.drift_detected = false;
            }
        }

        self.previous_components = Some(new_components.clone());
        Ok(())
    }

    /// Compute similarity between two subspaces
    fn compute_subspace_similarity(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Float> {
        // Compute the principal angles between subspaces
        let ab = a.dot(&b.t());
        let (_u, s, _vt) = ab
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidOperation(format!("SVD failed: {:?}", e)))?;

        // Average cosine of principal angles
        let similarity = s.iter().map(|&x| x.powi(2)).sum::<Float>() / s.len() as Float;
        Ok(similarity)
    }

    /// Transform new data using current components
    pub fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if let (Some(ref components), Some(ref mean)) = (&self.components, &self.mean) {
            let centered = x - mean;
            Ok(centered.dot(&components.t()))
        } else {
            Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            })
        }
    }

    /// Get the current components
    pub fn components(&self) -> Option<&Array2<Float>> {
        self.components.as_ref()
    }

    /// Get the explained variance
    pub fn explained_variance(&self) -> Option<&Array1<Float>> {
        self.explained_variance.as_ref()
    }

    /// Check if concept drift was detected
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Get number of samples seen
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Force an immediate update of the decomposition (useful for real-time scenarios)
    pub fn force_update(&mut self) -> Result<()> {
        if self.n_samples_seen >= self.config.min_samples {
            self.update_decomposition()
        } else {
            Err(SklearsError::InvalidParameter {
                name: "min_samples".to_string(),
                reason: "Not enough samples for decomposition update".to_string(),
            })
        }
    }

    /// Reset the streaming PCA (useful when drift is detected)
    pub fn reset(&mut self) {
        self.components = None;
        self.explained_variance = None;
        self.mean = None;
        self.n_samples_seen = 0;
        self.data_buffer.clear();
        self.covariance_matrix = None;
        self.sum_weights = 0.0;
        self.previous_components = None;
        self.drift_detected = false;
    }

    /// Transform a single sample for real-time processing
    pub fn transform_sample(&self, sample: &Array1<Float>) -> Result<Array1<Float>> {
        if let (Some(ref components), Some(ref mean)) = (&self.components, &self.mean) {
            let centered = sample - mean;
            Ok(components.dot(&centered))
        } else {
            Err(SklearsError::NotFitted {
                operation: "transform_sample".to_string(),
            })
        }
    }

    /// Get reconstruction error for monitoring model quality in real-time
    pub fn reconstruction_error(&self, sample: &Array1<Float>) -> Result<Float> {
        if let (Some(ref components), Some(ref mean)) = (&self.components, &self.mean) {
            let centered = sample - mean;
            let transformed = components.dot(&centered);
            let reconstructed = components.t().dot(&transformed);
            let error = (&centered - &reconstructed)
                .mapv(|x| x.powi(2))
                .sum()
                .sqrt();
            Ok(error)
        } else {
            Err(SklearsError::NotFitted {
                operation: "reconstruction_error".to_string(),
            })
        }
    }
}

/// Streaming ICA for real-time blind source separation
#[derive(Debug, Clone)]
pub struct StreamingICA {
    config: StreamingConfig,
    n_components: usize,
    // Current state
    components: Option<Array2<Float>>,
    mixing_matrix: Option<Array2<Float>>,
    mean: Option<Array1<Float>>,
    n_samples_seen: usize,
    // Streaming state
    data_buffer: VecDeque<Array1<Float>>,
    learning_rate: Float,
    // Drift detection
    previous_components: Option<Array2<Float>>,
    drift_detected: bool,
}

impl StreamingICA {
    /// Create a new streaming ICA
    pub fn new(n_components: usize) -> Self {
        Self {
            config: StreamingConfig::default(),
            n_components,
            components: None,
            mixing_matrix: None,
            mean: None,
            n_samples_seen: 0,
            data_buffer: VecDeque::new(),
            learning_rate: 0.01,
            previous_components: None,
            drift_detected: false,
        }
    }

    /// Configure the streaming ICA
    pub fn with_config(mut self, config: StreamingConfig) -> Self {
        self.config = config;
        self
    }

    /// Update the ICA with a new data sample
    pub fn partial_fit(&mut self, sample: &Array1<Float>) -> Result<()> {
        let n_features = sample.len();

        // Initialize if first sample
        if self.mean.is_none() {
            self.mean = Some(Array1::zeros(n_features));
            self.components = Some(Array2::eye(self.n_components));
        }

        // Add to buffer
        self.data_buffer.push_back(sample.clone());
        if self.data_buffer.len() > self.config.window_size {
            self.data_buffer.pop_front();
        }

        self.n_samples_seen += 1;

        // Update running mean
        self.update_mean(sample)?;

        // Update ICA decomposition
        if self.n_samples_seen >= self.config.min_samples
            && self.n_samples_seen % self.config.update_frequency == 0
        {
            self.update_decomposition()?;
        }

        Ok(())
    }

    /// Update running mean
    fn update_mean(&mut self, sample: &Array1<Float>) -> Result<()> {
        let weight = 1.0 / (self.n_samples_seen as Float);

        if let Some(ref mut mean) = self.mean {
            let delta = sample - &*mean;
            mean.zip_mut_with(&delta, |m, d| *m += d * weight);
        }

        Ok(())
    }

    /// Update the ICA decomposition using online learning
    fn update_decomposition(&mut self) -> Result<()> {
        if let (Some(ref mean), Some(_)) = (&self.mean, &self.components) {
            // Process recent samples
            let recent_samples: Vec<_> = self
                .data_buffer
                .iter()
                .rev()
                .take(self.config.update_frequency)
                .cloned()
                .collect();

            for sample in recent_samples {
                let centered = &sample - mean;
                // Inline the update to avoid borrow checker issues
                if let Some(ref mut components) = self.components {
                    let y = components.dot(&centered);
                    let g: Array1<Float> = y.mapv(|val| val.tanh());
                    let g_prime: Array1<Float> = y.mapv(|val| 1.0 - val.tanh().powi(2));

                    let outer_product = g
                        .insert_axis(Axis(1))
                        .dot(&centered.clone().insert_axis(Axis(0)));
                    let diagonal = Array2::from_diag(&g_prime);

                    let gradient = outer_product - diagonal.dot(&*components);
                    *components += &(gradient * self.learning_rate);
                }
            }

            // Orthogonalize components
            if let Some(ref mut components) = self.components {
                Self::orthogonalize_components_static(components)?;
            }

            // Detect concept drift
            if self.config.detect_drift {
                if let Some(ref components) = self.components {
                    let components_clone = components.clone();
                    self.detect_concept_drift(&components_clone)?;
                }
            }
        }

        Ok(())
    }

    /// Online ICA update using natural gradient
    #[allow(dead_code)]
    fn online_ica_update(
        &mut self,
        components: &mut Array2<Float>,
        x: &Array1<Float>,
    ) -> Result<()> {
        // Project data
        let y = components.dot(x);

        // Compute nonlinearity (tanh)
        let g: Array1<Float> = y.mapv(|val| val.tanh());
        let g_prime: Array1<Float> = y.mapv(|val| 1.0 - val.tanh().powi(2));

        // Natural gradient update
        let outer_product = g.insert_axis(Axis(1)).dot(&x.clone().insert_axis(Axis(0)));
        let _identity = Array2::<Float>::eye(self.n_components);
        let diagonal = Array2::from_diag(&g_prime);

        let gradient = outer_product - diagonal.dot(components);

        // Update components with learning rate
        *components += &(gradient * self.learning_rate);

        // Orthogonalize components
        self.orthogonalize_components(components)?;

        Ok(())
    }

    /// Orthogonalize components using Gram-Schmidt
    #[allow(dead_code)]
    fn orthogonalize_components(&self, components: &mut Array2<Float>) -> Result<()> {
        Self::orthogonalize_components_static(components)
    }

    /// Static version of orthogonalize_components for borrow checker
    fn orthogonalize_components_static(components: &mut Array2<Float>) -> Result<()> {
        for i in 0..components.nrows() {
            // Normalize current component
            let norm = components.row(i).dot(&components.row(i)).sqrt();
            if norm > 1e-10 {
                components.row_mut(i).mapv_inplace(|x| x / norm);
            }

            // Orthogonalize against previous components
            for j in 0..i {
                let projection = components.row(i).dot(&components.row(j));
                let row_j = components.row(j).to_owned();
                Zip::from(components.row_mut(i))
                    .and(&row_j)
                    .for_each(|a, &b| *a -= projection * b);
            }
        }

        Ok(())
    }

    /// Detect concept drift in the data
    fn detect_concept_drift(&mut self, new_components: &Array2<Float>) -> Result<()> {
        if let Some(ref prev_components) = self.previous_components {
            let similarity = self.compute_subspace_similarity(prev_components, new_components)?;

            if similarity < (1.0 - self.config.drift_threshold) {
                self.drift_detected = true;
                println!("Concept drift detected in ICA! Similarity: {similarity:.4}");
            } else {
                self.drift_detected = false;
            }
        }

        self.previous_components = Some(new_components.clone());
        Ok(())
    }

    /// Compute similarity between two subspaces
    fn compute_subspace_similarity(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Float> {
        let ab = a.dot(&b.t());
        let (_u, s, _vt) = ab
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidOperation(format!("SVD failed: {:?}", e)))?;

        let similarity = s.iter().map(|&x| x.powi(2)).sum::<Float>() / s.len() as Float;
        Ok(similarity)
    }

    /// Transform new data using current components
    pub fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if let (Some(ref components), Some(ref mean)) = (&self.components, &self.mean) {
            let centered = x - mean;
            Ok(components.dot(&centered.t()).t().to_owned())
        } else {
            Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            })
        }
    }

    /// Get the current components
    pub fn components(&self) -> Option<&Array2<Float>> {
        self.components.as_ref()
    }

    /// Check if concept drift was detected
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Get number of samples seen
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Transform a single sample for real-time signal processing
    pub fn transform_sample(&self, sample: &Array1<Float>) -> Result<Array1<Float>> {
        if let (Some(ref components), Some(ref mean)) = (&self.components, &self.mean) {
            let centered = sample - mean;
            Ok(components.dot(&centered))
        } else {
            Err(SklearsError::NotFitted {
                operation: "transform_sample".to_string(),
            })
        }
    }

    /// Get the mixing matrix for source reconstruction
    pub fn mixing_matrix(&self) -> Option<&Array2<Float>> {
        self.mixing_matrix.as_ref()
    }

    /// Reconstruct the original sources from mixed signal
    pub fn separate_sources(&self, mixed_sample: &Array1<Float>) -> Result<Array1<Float>> {
        if let (Some(ref components), Some(ref mean)) = (&self.components, &self.mean) {
            let centered = mixed_sample - mean;
            Ok(components.dot(&centered))
        } else {
            Err(SklearsError::NotFitted {
                operation: "separate_sources".to_string(),
            })
        }
    }

    /// Set learning rate for adaptive tuning
    pub fn set_learning_rate(&mut self, learning_rate: Float) {
        self.learning_rate = learning_rate;
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> Float {
        self.learning_rate
    }

    /// Force immediate update for real-time scenarios
    pub fn force_update(&mut self) -> Result<()> {
        if self.n_samples_seen >= self.config.min_samples {
            self.update_decomposition()
        } else {
            Err(SklearsError::InvalidParameter {
                name: "min_samples".to_string(),
                reason: "Not enough samples for decomposition update".to_string(),
            })
        }
    }

    /// Reset for new signal processing session
    pub fn reset(&mut self) {
        self.components = None;
        self.mixing_matrix = None;
        self.mean = None;
        self.n_samples_seen = 0;
        self.data_buffer.clear();
        self.previous_components = None;
        self.drift_detected = false;
    }
}

/// Adaptive decomposition method that can switch between algorithms
#[derive(Debug, Clone)]
pub struct AdaptiveDecomposition {
    #[allow(dead_code)]
    config: StreamingConfig,
    // Current active algorithm
    active_algorithm: String,
    // Algorithm instances
    streaming_pca: Option<StreamingPCA>,
    streaming_ica: Option<StreamingICA>,
    // Performance metrics
    reconstruction_errors: VecDeque<Float>,
    adaptation_threshold: Float,
}

impl AdaptiveDecomposition {
    /// Create a new adaptive decomposition
    pub fn new(n_components: usize) -> Self {
        Self {
            config: StreamingConfig::default(),
            active_algorithm: "PCA".to_string(),
            streaming_pca: Some(StreamingPCA::new(n_components)),
            streaming_ica: Some(StreamingICA::new(n_components)),
            reconstruction_errors: VecDeque::new(),
            adaptation_threshold: 0.05,
        }
    }

    /// Update with new data sample
    pub fn partial_fit(&mut self, sample: &Array1<Float>) -> Result<()> {
        // Update both algorithms
        if let Some(ref mut pca) = self.streaming_pca {
            pca.partial_fit(sample)?;
        }
        if let Some(ref mut ica) = self.streaming_ica {
            ica.partial_fit(sample)?;
        }

        // Evaluate performance and adapt if needed
        self.evaluate_and_adapt(sample)?;

        Ok(())
    }

    /// Evaluate performance and adapt algorithm if needed
    fn evaluate_and_adapt(&mut self, sample: &Array1<Float>) -> Result<()> {
        // Compute reconstruction errors for both algorithms
        let pca_error = self.compute_reconstruction_error_pca(sample)?;
        let ica_error = self.compute_reconstruction_error_ica(sample)?;

        // Store errors
        self.reconstruction_errors
            .push_back(pca_error.min(ica_error));
        if self.reconstruction_errors.len() > 100 {
            self.reconstruction_errors.pop_front();
        }

        // Decide which algorithm to use
        if pca_error < ica_error - self.adaptation_threshold {
            if self.active_algorithm != "PCA" {
                println!("Switching to PCA (error: {pca_error:.4})");
                self.active_algorithm = "PCA".to_string();
            }
        } else if ica_error < pca_error - self.adaptation_threshold
            && self.active_algorithm != "ICA"
        {
            println!("Switching to ICA (error: {ica_error:.4})");
            self.active_algorithm = "ICA".to_string();
        }

        Ok(())
    }

    /// Compute reconstruction error for PCA
    fn compute_reconstruction_error_pca(&self, sample: &Array1<Float>) -> Result<Float> {
        if let Some(ref pca) = self.streaming_pca {
            if let (Some(components), Some(mean)) = (pca.components(), pca.mean.as_ref()) {
                let centered = sample - mean;
                let transformed = components.dot(&centered);
                let reconstructed = components.t().dot(&transformed) + mean;
                let error = (sample - &reconstructed).mapv(|x| x.powi(2)).sum().sqrt();
                return Ok(error);
            }
        }
        Ok(Float::INFINITY)
    }

    /// Compute reconstruction error for ICA
    fn compute_reconstruction_error_ica(&self, sample: &Array1<Float>) -> Result<Float> {
        if let Some(ref ica) = self.streaming_ica {
            if let (Some(components), Some(mean)) = (ica.components(), ica.mean.as_ref()) {
                let centered = sample - mean;
                let transformed = components.dot(&centered);
                let reconstructed = components.t().dot(&transformed) + mean;
                let error = (sample - &reconstructed).mapv(|x| x.powi(2)).sum().sqrt();
                return Ok(error);
            }
        }
        Ok(Float::INFINITY)
    }

    /// Get the active algorithm name
    pub fn active_algorithm(&self) -> &str {
        &self.active_algorithm
    }

    /// Get average reconstruction error
    pub fn average_reconstruction_error(&self) -> Float {
        if self.reconstruction_errors.is_empty() {
            0.0
        } else {
            self.reconstruction_errors.iter().sum::<Float>()
                / self.reconstruction_errors.len() as Float
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_pca_basic() {
        let mut pca = StreamingPCA::new(2);

        // Add some samples
        for i in 0..100 {
            let sample = Array1::from_vec(vec![i as Float, (i * 2) as Float, (i * 3) as Float]);
            pca.partial_fit(&sample).unwrap();
        }

        assert!(pca.n_samples_seen() == 100);
        assert!(pca.components().is_some());
    }

    #[test]
    fn test_streaming_ica_basic() {
        let mut ica = StreamingICA::new(2);

        // Add some samples
        for i in 0..100 {
            let sample = Array1::from_vec(vec![i as Float, (i * 2) as Float]);
            ica.partial_fit(&sample).unwrap();
        }

        assert!(ica.n_samples_seen() == 100);
        assert!(ica.components().is_some());
    }

    #[test]
    fn test_adaptive_decomposition() {
        let mut adaptive = AdaptiveDecomposition::new(2);

        // Add some samples
        for i in 0..100 {
            let sample = Array1::from_vec(vec![i as Float, (i * 2) as Float]);
            adaptive.partial_fit(&sample).unwrap();
        }

        assert!(adaptive.active_algorithm() == "PCA" || adaptive.active_algorithm() == "ICA");
    }
}
