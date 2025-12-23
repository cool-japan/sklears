//! Incremental Principal Component Analysis implementation

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::svd;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Incremental PCA
#[derive(Debug, Clone)]
pub struct IncrementalPcaConfig {
    /// Number of components to keep
    pub n_components: Option<usize>,
    /// Whether to whiten the components
    pub whiten: bool,
    /// Whether to copy the input data
    pub copy: bool,
    /// Batch size for processing
    pub batch_size: Option<usize>,
}

impl Default for IncrementalPcaConfig {
    fn default() -> Self {
        Self {
            n_components: None,
            whiten: false,
            copy: true,
            batch_size: None,
        }
    }
}

/// Incremental Principal Component Analysis
///
/// Linear dimensionality reduction using incremental SVD. This algorithm can
/// process data in batches, making it suitable for large datasets that don't
/// fit in memory.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_decomposition::IncrementalPCA;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{array, s};
///
/// let x = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
/// ];
///
/// let mut pca = IncrementalPCA::new()
///     .n_components(2);
///
/// // Fit incrementally
/// pca = pca.partial_fit(&x.slice(s![0..2, ..]).to_owned(), &())?;
/// pca = pca.partial_fit(&x.slice(s![2..3, ..]).to_owned(), &())?;
///
/// // Convert to trained state and transform
/// let trained_pca = pca.into_trained()?;
/// let x_transformed = trained_pca.transform(&x)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalPCA<State = Untrained> {
    config: IncrementalPcaConfig,
    state: PhantomData<State>,
    // Fitted parameters
    components_: Option<Array2<Float>>,
    explained_variance_: Option<Array1<Float>>,
    explained_variance_ratio_: Option<Array1<Float>>,
    singular_values_: Option<Array1<Float>>,
    mean_: Option<Array1<Float>>,
    var_: Option<Array1<Float>>,
    n_components_: Option<usize>,
    n_features_in_: Option<usize>,
    n_samples_seen_: usize,
    // Incremental computation state
    sum_: Option<Array1<Float>>,
    sum_squared_: Option<Array1<Float>>,
}

impl IncrementalPCA<Untrained> {
    /// Create a new Incremental PCA
    pub fn new() -> Self {
        Self {
            config: IncrementalPcaConfig::default(),
            state: PhantomData,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            singular_values_: None,
            mean_: None,
            var_: None,
            n_components_: None,
            n_features_in_: None,
            n_samples_seen_: 0,
            sum_: None,
            sum_squared_: None,
        }
    }

    /// Set the number of components to keep
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = Some(n_components);
        self
    }

    /// Set whether to whiten the components
    pub fn whiten(mut self, whiten: bool) -> Self {
        self.config.whiten = whiten;
        self
    }

    /// Set whether to copy the input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.config.copy = copy;
        self
    }

    /// Set the batch size for processing
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = Some(batch_size);
        self
    }

    /// Initialize the incremental state
    fn initialize(&mut self, x: &Array2<Float>) -> Result<()> {
        let (_, n_features) = x.dim();

        if self.n_features_in_.is_none() {
            self.n_features_in_ = Some(n_features);
            self.sum_ = Some(Array1::zeros(n_features));
            self.sum_squared_ = Some(Array1::zeros(n_features));
        } else if self.n_features_in_.unwrap() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap(),
                actual: n_features,
            });
        }

        Ok(())
    }

    /// Update running statistics with new batch
    fn update_statistics(&mut self, x: &Array2<Float>) {
        let (n_samples, _) = x.dim();

        // Update running sums
        let batch_sum = x.sum_axis(Axis(0));
        let batch_sum_squared = x.mapv(|v| v * v).sum_axis(Axis(0));

        if let (Some(sum), Some(sum_squared)) = (&mut self.sum_, &mut self.sum_squared_) {
            *sum += &batch_sum;
            *sum_squared += &batch_sum_squared;
        }

        self.n_samples_seen_ += n_samples;
    }

    /// Compute mean and variance from running statistics
    fn compute_mean_and_variance(&self) -> (Array1<Float>, Array1<Float>) {
        let sum = self.sum_.as_ref().unwrap();
        let sum_squared = self.sum_squared_.as_ref().unwrap();
        let n_samples = self.n_samples_seen_ as Float;

        // Mean = sum / n_samples
        let mean = sum / n_samples;

        // Variance = (sum_squared / n_samples) - mean^2
        let variance = sum_squared / n_samples - &mean * &mean;

        (mean, variance)
    }

    /// Perform incremental SVD update
    fn update_components(&mut self, x: &Array2<Float>) -> Result<()> {
        let (mean, variance) = self.compute_mean_and_variance();
        self.mean_ = Some(mean.clone());
        self.var_ = Some(variance);

        // Center the current batch
        let x_centered = x - &mean.clone().insert_axis(Axis(0));

        // For incremental PCA, we need to update the SVD incrementally
        // This is a simplified version - in practice, you'd use more sophisticated
        // incremental SVD algorithms like the one in scikit-learn

        let n_components = self
            .config
            .n_components
            .unwrap_or_else(|| x.dim().1.min(self.n_samples_seen_));

        // Perform SVD on centered data
        let (n_samples, n_features) = x_centered.dim();

        // Convert to nalgebra for SVD
        let matrix =
            nalgebra::DMatrix::from_iterator(n_samples, n_features, x_centered.iter().cloned());

        let svd = matrix.svd(true, true);

        // Extract components and singular values
        let vt = svd.v_t.ok_or_else(|| {
            SklearsError::InvalidInput("SVD failed to compute V^T matrix".to_string())
        })?;
        let singular_values = svd.singular_values;

        // Take first n_components
        let n_comp = n_components.min(singular_values.len());

        // Components are rows of V^T
        let mut components = Array2::<Float>::zeros((n_comp, n_features));
        for i in 0..n_comp {
            for j in 0..n_features {
                components[[i, j]] = vt[(i, j)];
            }
        }

        // Singular values
        let sing_vals = Array1::from_iter(singular_values.iter().take(n_comp).cloned());

        // Calculate explained variance
        let n_samples_f = self.n_samples_seen_ as Float;
        let explained_variance = sing_vals.mapv(|s| s * s / (n_samples_f - 1.0));

        let total_variance = explained_variance.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|var| var / total_variance)
        } else {
            Array1::zeros(explained_variance.len())
        };

        self.components_ = Some(components);
        self.singular_values_ = Some(sing_vals);
        self.explained_variance_ = Some(explained_variance);
        self.explained_variance_ratio_ = Some(explained_variance_ratio);
        self.n_components_ = Some(n_comp);

        Ok(())
    }

    /// Partial fit on a batch of data
    pub fn partial_fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self> {
        let (n_samples, _) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit IncrementalPCA on empty batch".to_string(),
            ));
        }

        // Initialize if this is the first batch
        self.initialize(x)?;

        // Update running statistics
        self.update_statistics(x);

        // Update components
        self.update_components(x)?;

        Ok(self)
    }

    /// Convert to trained state after fitting
    pub fn into_trained(self) -> Result<IncrementalPCA<Trained>> {
        if self.n_samples_seen_ == 0 {
            return Err(SklearsError::InvalidInput(
                "IncrementalPCA has not been fitted yet".to_string(),
            ));
        }

        Ok(IncrementalPCA {
            config: self.config,
            state: PhantomData,
            components_: self.components_,
            explained_variance_: self.explained_variance_,
            explained_variance_ratio_: self.explained_variance_ratio_,
            singular_values_: self.singular_values_,
            mean_: self.mean_,
            var_: self.var_,
            n_components_: self.n_components_,
            n_features_in_: self.n_features_in_,
            n_samples_seen_: self.n_samples_seen_,
            sum_: self.sum_,
            sum_squared_: self.sum_squared_,
        })
    }
}

impl IncrementalPCA<Trained> {
    /// Get the principal components
    pub fn components(&self) -> &Array2<Float> {
        self.components_
            .as_ref()
            .expect("IncrementalPCA should be fitted")
    }

    /// Get the explained variance for each component
    pub fn explained_variance(&self) -> &Array1<Float> {
        self.explained_variance_
            .as_ref()
            .expect("IncrementalPCA should be fitted")
    }

    /// Get the explained variance ratio for each component
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        self.explained_variance_ratio_
            .as_ref()
            .expect("IncrementalPCA should be fitted")
    }

    /// Get the singular values
    pub fn singular_values(&self) -> &Array1<Float> {
        self.singular_values_
            .as_ref()
            .expect("IncrementalPCA should be fitted")
    }

    /// Get the mean of the training data
    pub fn mean(&self) -> &Array1<Float> {
        self.mean_
            .as_ref()
            .expect("IncrementalPCA should be fitted")
    }

    /// Get the variance of the training data
    pub fn var(&self) -> &Array1<Float> {
        self.var_.as_ref().expect("IncrementalPCA should be fitted")
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components_.expect("IncrementalPCA should be fitted")
    }

    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("IncrementalPCA should be fitted")
    }

    /// Get the number of samples seen during fitting
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen_
    }

    /// Transform data back to original space
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_components_transformed) = x.dim();

        if n_components_transformed != self.n_components() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_components(),
                actual: n_components_transformed,
            });
        }

        let components = self.components();
        let mean = self.mean();

        // X_original = X_transformed @ components + mean
        let reconstructed = x.dot(components) + &mean.clone().insert_axis(Axis(0));

        Ok(reconstructed)
    }

    /// Continue fitting with additional data (converts back to untrained temporarily)
    pub fn partial_fit_more(self, x: &Array2<Float>, _y: &()) -> Result<IncrementalPCA<Trained>> {
        // Convert back to untrained state
        let untrained = IncrementalPCA {
            config: self.config,
            state: PhantomData,
            components_: self.components_,
            explained_variance_: self.explained_variance_,
            explained_variance_ratio_: self.explained_variance_ratio_,
            singular_values_: self.singular_values_,
            mean_: self.mean_,
            var_: self.var_,
            n_components_: self.n_components_,
            n_features_in_: self.n_features_in_,
            n_samples_seen_: self.n_samples_seen_,
            sum_: self.sum_,
            sum_squared_: self.sum_squared_,
        };

        // Partial fit on new data
        let updated = untrained.partial_fit(x, _y)?;
        updated.into_trained()
    }
}

impl IncrementalPCA<Untrained> {
    /// Fit using adaptive mini-batch processing
    ///
    /// Automatically determines optimal batch size based on data characteristics
    /// and available memory, then processes the data in mini-batches.
    pub fn fit_minibatch_adaptive(
        mut self,
        x: &Array2<Float>,
        _y: &(),
        memory_limit_mb: Option<usize>,
    ) -> Result<IncrementalPCA<Trained>> {
        let (n_samples, n_features) = x.dim();

        // Calculate adaptive batch size
        let batch_size = self.calculate_adaptive_batch_size(n_samples, n_features, memory_limit_mb);

        // Process data in mini-batches
        let mut current_pos = 0;
        while current_pos < n_samples {
            let end_pos = (current_pos + batch_size).min(n_samples);
            let batch = x
                .slice(scirs2_core::ndarray::s![current_pos..end_pos, ..])
                .to_owned();

            self = self.partial_fit(&batch, _y)?;
            current_pos = end_pos;
        }

        self.into_trained()
    }

    /// Calculate optimal batch size based on data characteristics and memory constraints
    fn calculate_adaptive_batch_size(
        &self,
        n_samples: usize,
        n_features: usize,
        memory_limit_mb: Option<usize>,
    ) -> usize {
        // Default memory limit: 100 MB
        let memory_limit_bytes = memory_limit_mb.unwrap_or(100) * 1024 * 1024;

        // Estimate memory per sample (Float = 8 bytes typically)
        let bytes_per_sample = n_features * 8;

        // Calculate max samples that fit in memory limit
        let max_samples_from_memory = memory_limit_bytes / bytes_per_sample;

        // Use heuristics for optimal batch size
        let min_batch_size = 10;
        let max_batch_size = 1000;

        // Adaptive batch size based on data size and memory
        let adaptive_size = if n_samples <= 100 {
            n_samples // Use all data if small
        } else if n_samples <= 1000 {
            n_samples / 4 // Use quarter for medium data
        } else {
            // For large data, balance between memory and efficiency
            let efficiency_size = (n_samples as f64).sqrt() as usize;
            efficiency_size.min(max_samples_from_memory)
        };

        // Apply constraints
        adaptive_size
            .max(min_batch_size)
            .min(max_batch_size)
            .min(n_samples)
    }

    /// Process large dataset with streaming mini-batch approach
    ///
    /// This method is designed for very large datasets that don't fit in memory.
    /// It expects an iterator that yields mini-batches.
    pub fn fit_streaming<I>(
        mut self,
        batch_iterator: I,
        forgetting_factor: Option<Float>,
    ) -> Result<IncrementalPCA<Trained>>
    where
        I: Iterator<Item = Array2<Float>>,
    {
        let forgetting = forgetting_factor.unwrap_or(1.0);

        for (batch_idx, batch) in batch_iterator.enumerate() {
            // Apply forgetting factor to reduce influence of old data
            if forgetting < 1.0 && batch_idx > 0 {
                self = self.apply_forgetting_factor(forgetting)?;
            }

            self = self.partial_fit(&batch, &())?;
        }

        self.into_trained()
    }

    /// Apply forgetting factor to reduce influence of historical data
    fn apply_forgetting_factor(mut self, factor: Float) -> Result<Self> {
        if let (Some(ref mut sum), Some(ref mut sum_squared)) =
            (&mut self.sum_, &mut self.sum_squared_)
        {
            *sum *= factor;
            *sum_squared *= factor;
            self.n_samples_seen_ = (self.n_samples_seen_ as Float * factor) as usize;
        }
        Ok(self)
    }

    /// Online PCA with exponential moving averages
    ///
    /// Updates the PCA model using exponential decay for older samples,
    /// making it suitable for time-series data where recent observations
    /// are more important.
    pub fn fit_online_ema(
        mut self,
        x: &Array2<Float>,
        _y: &(),
        decay_rate: Float,
    ) -> Result<IncrementalPCA<Trained>> {
        let (n_samples, _) = x.dim();

        // Process each sample individually with exponential decay
        for i in 0..n_samples {
            let sample = x.slice(scirs2_core::ndarray::s![i..i + 1, ..]).to_owned();

            // Apply decay to existing statistics before adding new sample
            if i > 0 {
                self = self.apply_exponential_decay(decay_rate)?;
            }

            self = self.partial_fit(&sample, _y)?;
        }

        self.into_trained()
    }

    /// Apply exponential decay to running statistics
    fn apply_exponential_decay(mut self, decay_rate: Float) -> Result<Self> {
        if let (Some(ref mut sum), Some(ref mut sum_squared)) =
            (&mut self.sum_, &mut self.sum_squared_)
        {
            *sum *= decay_rate;
            *sum_squared *= decay_rate;

            // Adjust sample count to reflect decay
            self.n_samples_seen_ = (self.n_samples_seen_ as Float * decay_rate) as usize;
        }
        Ok(self)
    }

    /// Batch processing with early stopping based on convergence
    ///
    /// Monitors the convergence of principal components and stops early
    /// if the components have stabilized, saving computation time.
    pub fn fit_early_stopping(
        mut self,
        x: &Array2<Float>,
        _y: &(),
        batch_size: Option<usize>,
        convergence_tol: Float,
        patience: usize,
    ) -> Result<(IncrementalPCA<Trained>, usize)> {
        let (n_samples, n_features) = x.dim();
        let actual_batch_size =
            batch_size.unwrap_or(self.calculate_adaptive_batch_size(n_samples, n_features, None));

        let mut prev_components: Option<Array2<Float>> = None;
        let mut no_improvement_count = 0;
        let mut batches_processed = 0;

        let mut current_pos = 0;
        while current_pos < n_samples {
            let end_pos = (current_pos + actual_batch_size).min(n_samples);
            let batch = x
                .slice(scirs2_core::ndarray::s![current_pos..end_pos, ..])
                .to_owned();

            self = self.partial_fit(&batch, _y)?;
            batches_processed += 1;

            // Check convergence after processing a batch
            if let Some(ref current_components) = self.components_ {
                if let Some(ref prev) = prev_components {
                    // Calculate the maximum change in components
                    let component_diff =
                        self.compute_component_difference(current_components, prev);

                    if component_diff < convergence_tol {
                        no_improvement_count += 1;
                        if no_improvement_count >= patience {
                            break; // Early stopping
                        }
                    } else {
                        no_improvement_count = 0; // Reset counter
                    }
                }
                prev_components = Some(current_components.clone());
            }

            current_pos = end_pos;
        }

        let trained = self.into_trained()?;
        Ok((trained, batches_processed))
    }

    /// Compute the maximum difference between component matrices
    fn compute_component_difference(
        &self,
        current: &Array2<Float>,
        previous: &Array2<Float>,
    ) -> Float {
        if current.dim() != previous.dim() {
            return Float::INFINITY; // Dimensions changed, no convergence
        }

        let diff = current - previous;
        diff.iter()
            .map(|x| x.abs())
            .fold(0.0f64, |acc, x| acc.max(x))
    }

    /// Memory-efficient batch processing with progress monitoring
    ///
    /// Provides detailed progress information and memory usage optimization
    /// for very large datasets.
    pub fn fit_memory_efficient<F>(
        mut self,
        x: &Array2<Float>,
        _y: &(),
        progress_callback: Option<F>,
    ) -> Result<IncrementalPCA<Trained>>
    where
        F: Fn(usize, usize, &IncrementalPCA<Untrained>),
    {
        let (n_samples, n_features) = x.dim();

        // Use conservative batch size for memory efficiency
        let batch_size = self.calculate_adaptive_batch_size(n_samples, n_features, Some(50)); // 50MB limit

        let total_batches = (n_samples + batch_size - 1) / batch_size;
        let mut current_pos = 0;
        let mut batch_count = 0;

        while current_pos < n_samples {
            let end_pos = (current_pos + batch_size).min(n_samples);
            let batch = x
                .slice(scirs2_core::ndarray::s![current_pos..end_pos, ..])
                .to_owned();

            self = self.partial_fit(&batch, _y)?;
            batch_count += 1;

            // Call progress callback if provided
            if let Some(ref callback) = progress_callback {
                callback(batch_count, total_batches, &self);
            }

            current_pos = end_pos;
        }

        self.into_trained()
    }
}

impl Default for IncrementalPCA<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for IncrementalPCA<Untrained> {
    type Fitted = IncrementalPCA<Trained>;

    fn fit(self, x: &Array2<Float>, y: &()) -> Result<Self::Fitted> {
        let updated = self.partial_fit(x, y)?;
        updated.into_trained()
    }
}

impl Transform<Array2<Float>, Array2<Float>> for IncrementalPCA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let components = self.components();
        let mean = self.mean();

        // Center the data
        let x_centered = x - &mean.clone().insert_axis(Axis(0));

        // Project onto principal components
        let x_transformed = x_centered.dot(&components.t());

        // Apply whitening if requested
        if self.config.whiten {
            let singular_values = self.singular_values();
            let sqrt_n_samples = ((self.n_samples_seen_ - 1) as Float).sqrt();

            let mut result = x_transformed;
            for i in 0..self.n_components() {
                if singular_values[i] > 1e-12 {
                    result
                        .column_mut(i)
                        .mapv_inplace(|val| val * sqrt_n_samples / singular_values[i]);
                }
            }
            Ok(result)
        } else {
            Ok(x_transformed)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_incremental_pca_basic() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let pca = IncrementalPCA::new().n_components(2).fit(&x, &()).unwrap();

        // Check fitted parameters
        assert_eq!(pca.n_components(), 2);
        assert_eq!(pca.n_features_in(), 3);
        assert_eq!(pca.n_samples_seen(), 3);
        assert_eq!(pca.components().dim(), (2, 3));

        // Transform data
        let x_transformed = pca.transform(&x).unwrap();
        assert_eq!(x_transformed.dim(), (3, 2));
    }

    #[test]
    fn test_incremental_pca_partial_fit() {
        let x1 = array![[1.0, 2.0], [2.0, 4.0],];

        let x2 = array![[3.0, 6.0], [4.0, 8.0],];

        let mut pca = IncrementalPCA::new().n_components(1);

        // Fit incrementally
        pca = pca.partial_fit(&x1, &()).unwrap();
        assert_eq!(pca.n_samples_seen_, 2);

        pca = pca.partial_fit(&x2, &()).unwrap();
        assert_eq!(pca.n_samples_seen_, 4);

        let trained_pca = pca.into_trained().unwrap();
        assert_eq!(trained_pca.n_samples_seen(), 4);

        // Transform combined data
        let x_all = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0],];

        let x_transformed = trained_pca.transform(&x_all).unwrap();
        assert_eq!(x_transformed.dim(), (4, 1));
    }

    #[test]
    fn test_incremental_pca_statistics() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let pca = IncrementalPCA::new().fit(&x, &()).unwrap();

        // Check mean calculation
        let expected_mean = array![3.0, 4.0];
        assert_abs_diff_eq!(pca.mean(), &expected_mean, epsilon = 1e-10);

        // Check variance is non-negative
        for &var in pca.var().iter() {
            assert!(var >= 0.0);
        }
    }

    #[test]
    fn test_incremental_pca_inverse_transform() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let pca = IncrementalPCA::new().n_components(2).fit(&x, &()).unwrap();

        let x_transformed = pca.transform(&x).unwrap();
        let x_reconstructed = pca.inverse_transform(&x_transformed).unwrap();

        assert_eq!(x_reconstructed.dim(), (3, 3));
        // Note: Reconstruction won't be perfect with fewer components
    }

    #[test]
    fn test_incremental_pca_empty_batch() {
        let x = Array2::<Float>::zeros((0, 3));
        let result = IncrementalPCA::new().partial_fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_pca_feature_mismatch() {
        let x1 = array![[1.0, 2.0, 3.0]];
        let x2 = array![[4.0, 5.0]]; // Different number of features

        let pca = IncrementalPCA::new().partial_fit(&x1, &()).unwrap();
        let result = pca.partial_fit(&x2, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_pca_partial_fit_more() {
        let x1 = array![[1.0, 2.0], [2.0, 4.0],];

        let x2 = array![[3.0, 6.0],];

        let pca = IncrementalPCA::new().fit(&x1, &()).unwrap();

        assert_eq!(pca.n_samples_seen(), 2);

        let updated_pca = pca.partial_fit_more(&x2, &()).unwrap();
        assert_eq!(updated_pca.n_samples_seen(), 3);
    }
}
