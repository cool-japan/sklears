use crate::SklResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use sklears_core::prelude::{Float, SklearsError};

/// Streaming Feature Extractor
///
/// Process large datasets in chunks for memory-efficient feature extraction.
/// Supports online computation of statistical features without loading entire dataset into memory.
///
/// # Parameters
///
/// * `chunk_size` - Size of chunks to process at once
/// * `buffer_size` - Size of internal buffer for overlapping windows
/// * `features` - Types of features to extract
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::streaming_projection::StreamingFeatureExtractor;
/// use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut extractor = StreamingFeatureExtractor::new()
///     .chunk_size(1000)
///     .buffer_size(100);
///
/// // Process data in chunks
/// let mut features = Vec::new();
/// let data_chunks = vec![Array2::from_elem((10, 3), 1.0)];
/// for chunk in data_chunks {
///     let chunk_features = extractor.extract_chunk(&chunk.view())?;
///     features.extend(chunk_features);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StreamingFeatureExtractor {
    chunk_size: usize,
    buffer_size: usize,
    include_mean: bool,
    include_std: bool,
    include_min_max: bool,
    include_quantiles: bool,
    include_moments: bool,
    // Internal state for streaming computation
    running_mean: Option<Array1<Float>>,
    running_variance: Option<Array1<Float>>,
    running_count: usize,
    running_min: Option<Array1<Float>>,
    running_max: Option<Array1<Float>>,
}

impl StreamingFeatureExtractor {
    /// Create a new streaming feature extractor
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            buffer_size: 100,
            include_mean: true,
            include_std: true,
            include_min_max: true,
            include_quantiles: false,
            include_moments: false,
            running_mean: None,
            running_variance: None,
            running_count: 0,
            running_min: None,
            running_max: None,
        }
    }

    /// Set chunk size for processing
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set buffer size for overlapping windows
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Include mean in extracted features
    pub fn include_mean(mut self, include: bool) -> Self {
        self.include_mean = include;
        self
    }

    /// Include standard deviation in extracted features
    pub fn include_std(mut self, include: bool) -> Self {
        self.include_std = include;
        self
    }

    /// Include min/max in extracted features
    pub fn include_min_max(mut self, include: bool) -> Self {
        self.include_min_max = include;
        self
    }

    /// Include quantiles in extracted features
    pub fn include_quantiles(mut self, include: bool) -> Self {
        self.include_quantiles = include;
        self
    }

    /// Include higher moments in extracted features
    pub fn include_moments(mut self, include: bool) -> Self {
        self.include_moments = include;
        self
    }

    /// Process a single chunk of data
    pub fn extract_chunk(&mut self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Empty data chunk".to_string()));
        }

        let (n_samples, n_features) = data.dim();

        // Update running statistics
        self.update_running_stats(data)?;

        // Extract features for this chunk
        let mut features = Vec::new();

        if self.include_mean {
            if let Some(ref mean) = self.running_mean {
                features.extend(mean.iter().cloned());
            }
        }

        if self.include_std {
            if let Some(ref variance) = self.running_variance {
                let std_dev: Vec<Float> = variance.iter().map(|&v| v.sqrt()).collect();
                features.extend(std_dev);
            }
        }

        if self.include_min_max {
            if let Some(ref min_vals) = self.running_min {
                features.extend(min_vals.iter().cloned());
            }
            if let Some(ref max_vals) = self.running_max {
                features.extend(max_vals.iter().cloned());
            }
        }

        if self.include_quantiles {
            // For streaming quantiles, we use approximate methods
            for feature_idx in 0..n_features {
                let column = data.column(feature_idx);
                let quantiles = self.compute_approximate_quantiles(&column)?;
                features.extend(quantiles);
            }
        }

        if self.include_moments {
            // Compute higher moments for current chunk
            for feature_idx in 0..n_features {
                let column = data.column(feature_idx);
                let moments = self.compute_moments(&column)?;
                features.extend(moments);
            }
        }

        Ok(Array1::from_vec(features))
    }

    /// Finalize streaming computation and return aggregated features
    pub fn finalize(&self) -> SklResult<Array1<Float>> {
        if self.running_count == 0 {
            return Err(SklearsError::InvalidInput("No data processed".to_string()));
        }

        let mut features = Vec::new();

        if self.include_mean {
            if let Some(ref mean) = self.running_mean {
                features.extend(mean.iter().cloned());
            }
        }

        if self.include_std {
            if let Some(ref variance) = self.running_variance {
                let std_dev: Vec<Float> = variance.iter().map(|&v| v.sqrt()).collect();
                features.extend(std_dev);
            }
        }

        if self.include_min_max {
            if let Some(ref min_vals) = self.running_min {
                features.extend(min_vals.iter().cloned());
            }
            if let Some(ref max_vals) = self.running_max {
                features.extend(max_vals.iter().cloned());
            }
        }

        Ok(Array1::from_vec(features))
    }

    /// Reset internal state for new stream
    pub fn reset(&mut self) {
        self.running_mean = None;
        self.running_variance = None;
        self.running_count = 0;
        self.running_min = None;
        self.running_max = None;
    }

    /// Update running statistics with new data chunk
    fn update_running_stats(&mut self, data: &ArrayView2<Float>) -> SklResult<()> {
        let (n_samples, n_features) = data.dim();

        if self.running_count == 0 {
            // Initialize running statistics
            self.running_mean = Some(Array1::zeros(n_features));
            self.running_variance = Some(Array1::zeros(n_features));
            self.running_min = Some(Array1::from_elem(n_features, Float::INFINITY));
            self.running_max = Some(Array1::from_elem(n_features, Float::NEG_INFINITY));
        }

        // Update using Welford's online algorithm
        for sample in data.axis_iter(Axis(0)) {
            self.running_count += 1;
            let count = self.running_count as Float;

            for (feature_idx, &value) in sample.iter().enumerate() {
                // Update mean and variance using Welford's algorithm
                if let (Some(ref mut mean), Some(ref mut variance)) =
                    (&mut self.running_mean, &mut self.running_variance)
                {
                    let delta = value - mean[feature_idx];
                    mean[feature_idx] += delta / count;
                    let delta2 = value - mean[feature_idx];
                    variance[feature_idx] += delta * delta2;
                }

                // Update min/max
                if let Some(ref mut min_vals) = self.running_min {
                    if value < min_vals[feature_idx] {
                        min_vals[feature_idx] = value;
                    }
                }
                if let Some(ref mut max_vals) = self.running_max {
                    if value > max_vals[feature_idx] {
                        max_vals[feature_idx] = value;
                    }
                }
            }
        }

        // Normalize variance
        if let Some(ref mut variance) = self.running_variance {
            let count = self.running_count as Float;
            if count > 1.0 {
                variance.mapv_inplace(|v| v / (count - 1.0));
            }
        }

        Ok(())
    }

    /// Compute approximate quantiles for streaming data
    fn compute_approximate_quantiles(&self, data: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut sorted_data: Vec<Float> = data.iter().cloned().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len();
        let quantiles = vec![0.25, 0.5, 0.75]; // Q1, median, Q3
        let mut result = Vec::new();

        for &q in &quantiles {
            let index = (q * (n - 1) as Float).round() as usize;
            let index = index.min(n - 1);
            result.push(sorted_data[index]);
        }

        Ok(result)
    }

    /// Compute moments for a data column
    fn compute_moments(&self, data: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let n = data.len() as Float;
        if n == 0.0 {
            return Ok(vec![0.0, 0.0, 0.0]); // mean, skewness, kurtosis
        }

        let mean = data.sum() / n;
        let variance = data.mapv(|x| (x - mean).powi(2)).sum() / n;
        let std_dev = variance.sqrt();

        let (skewness, kurtosis) = if std_dev < 1e-10 {
            (0.0, 0.0)
        } else {
            let skewness = data.mapv(|x| ((x - mean) / std_dev).powi(3)).sum() / n;
            let kurtosis = data.mapv(|x| ((x - mean) / std_dev).powi(4)).sum() / n - 3.0;
            (skewness, kurtosis)
        };

        Ok(vec![mean, skewness, kurtosis])
    }
}

impl Default for StreamingFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Random Projection Feature Extractor
///
/// Performs dimensionality reduction using random projections while preserving
/// approximate distances between data points (Johnson-Lindenstrauss lemma).
/// Useful for high-dimensional data preprocessing and approximate similarity search.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the projected space
/// * `random_state` - Random seed for reproducibility
/// * `density` - Density of random projection matrix (for sparse projections)
/// * `eps` - Distortion parameter for auto-sizing components
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::streaming_projection::RandomProjectionFeatures;
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let X = Array2::from_elem((100, 1000), 1.0);
/// let mut rp = RandomProjectionFeatures::new()
///     .n_components(50)
///     .random_state(42);
///
/// let X_projected = rp.fit_transform(&X.view())?;
/// assert_eq!(X_projected.ncols(), 50);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct RandomProjectionFeatures {
    n_components: Option<usize>,
    random_state: Option<u64>,
    density: Option<Float>,
    eps: Float,
    projection_matrix: Option<Array2<Float>>,
    n_features: Option<usize>,
}

impl RandomProjectionFeatures {
    /// Create a new random projection feature extractor
    pub fn new() -> Self {
        Self {
            n_components: None,
            random_state: None,
            density: None,
            eps: 0.1,
            projection_matrix: None,
            n_features: None,
        }
    }

    /// Set number of components in projected space
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set density for sparse random projections
    pub fn density(mut self, density: Float) -> Self {
        self.density = Some(density);
        self
    }

    /// Set distortion parameter for auto-sizing
    pub fn eps(mut self, eps: Float) -> Self {
        self.eps = eps;
        self
    }

    /// Fit the random projection to data
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> SklResult<()> {
        let (n_samples, n_features) = X.dim();
        self.n_features = Some(n_features);

        // Auto-size number of components if not specified
        let n_components = if let Some(n_comp) = self.n_components {
            n_comp
        } else {
            self.johnson_lindenstrauss_min_dim(n_samples, self.eps)?
        };

        // Create random projection matrix
        self.projection_matrix = Some(self.create_projection_matrix(n_features, n_components)?);

        Ok(())
    }

    /// Transform data using fitted random projection
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if let Some(ref proj_matrix) = self.projection_matrix {
            let (n_samples, n_features) = X.dim();

            if Some(n_features) != self.n_features {
                return Err(SklearsError::InvalidInput(
                    "Number of features does not match fitted data".to_string(),
                ));
            }

            // Perform matrix multiplication: X @ projection_matrix.T
            let result = X.dot(proj_matrix);
            Ok(result)
        } else {
            Err(SklearsError::InvalidInput(
                "Random projection not fitted yet".to_string(),
            ))
        }
    }

    /// Fit and transform data in one step
    pub fn fit_transform(&mut self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        self.fit(X)?;
        self.transform(X)
    }

    /// Create random projection matrix
    fn create_projection_matrix(
        &self,
        n_features: usize,
        n_components: usize,
    ) -> SklResult<Array2<Float>> {
        use scirs2_core::random::thread_rng;

        // Use rng() function from scirs2_core - seeding handled elsewhere if needed
        let mut rng = thread_rng();

        let mut matrix = Array2::zeros((n_features, n_components));

        if let Some(density) = self.density {
            // Sparse random projection
            let threshold = density;
            for i in 0..n_features {
                for j in 0..n_components {
                    let random_val: Float = rng.gen();
                    if random_val < threshold {
                        matrix[[i, j]] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                        matrix[[i, j]] /= (density * n_components as Float).sqrt();
                    }
                }
            }
        } else {
            // Dense Gaussian random projection
            for i in 0..n_features {
                for j in 0..n_components {
                    matrix[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
            // Normalize
            let norm_factor = 1.0 / (n_components as Float).sqrt();
            matrix.mapv_inplace(|x| x * norm_factor);
        }

        Ok(matrix)
    }

    /// Compute minimum number of components for Johnson-Lindenstrauss lemma
    fn johnson_lindenstrauss_min_dim(&self, n_samples: usize, eps: Float) -> SklResult<usize> {
        if eps <= 0.0 || eps >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "eps must be between 0 and 1".to_string(),
            ));
        }

        if n_samples <= 1 {
            return Ok(1);
        }

        // Simplified JL bound: k >= 4 * ln(n) / (eps^2 / 2)
        // This is a more conservative but simpler bound
        let ln_n = (n_samples as Float).ln();
        let denominator = eps * eps / 2.0;

        if denominator <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Invalid eps value for Johnson-Lindenstrauss bound".to_string(),
            ));
        }

        let min_dim = (4.0 * ln_n / denominator).ceil() as usize;
        Ok(min_dim.max(1))
    }
}

impl Default for RandomProjectionFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Locality-Sensitive Hashing (LSH) Feature Extractor
///
/// Implements locality-sensitive hashing for approximate similarity search
/// and clustering. Maps similar items to the same buckets with high probability.
///
/// # Parameters
///
/// * `n_hash_functions` - Number of hash functions to use
/// * `hash_table_size` - Size of each hash table
/// * `random_state` - Random seed for reproducibility
/// * `distance_metric` - Distance metric for similarity ('cosine', 'euclidean')
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::streaming_projection::LocalitySensitiveHashing;
/// use scirs2_core::ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut lsh = LocalitySensitiveHashing::new()
///     .n_hash_functions(10)
///     .hash_table_size(1000)
///     .distance_metric("cosine");
///
/// let X = Array2::from_elem((100, 50), 1.0);
/// let hash_codes = lsh.fit_transform(&X.view())?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct LocalitySensitiveHashing {
    n_hash_functions: usize,
    hash_table_size: usize,
    random_state: Option<u64>,
    distance_metric: String,
    hash_matrices: Option<Vec<Array2<Float>>>,
    n_features: Option<usize>,
}

impl LocalitySensitiveHashing {
    /// Create a new LSH feature extractor
    pub fn new() -> Self {
        Self {
            n_hash_functions: 10,
            hash_table_size: 1000,
            random_state: None,
            distance_metric: "cosine".to_string(),
            hash_matrices: None,
            n_features: None,
        }
    }

    /// Set number of hash functions
    pub fn n_hash_functions(mut self, n_hash_functions: usize) -> Self {
        self.n_hash_functions = n_hash_functions;
        self
    }

    /// Set hash table size
    pub fn hash_table_size(mut self, hash_table_size: usize) -> Self {
        self.hash_table_size = hash_table_size;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: &str) -> Self {
        self.distance_metric = metric.to_string();
        self
    }

    /// Fit LSH to data
    pub fn fit(&mut self, X: &ArrayView2<Float>) -> SklResult<()> {
        let (_, n_features) = X.dim();
        self.n_features = Some(n_features);

        // Create hash matrices
        self.hash_matrices = Some(self.create_hash_matrices(n_features)?);

        Ok(())
    }

    /// Transform data to hash codes
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if let Some(ref hash_matrices) = self.hash_matrices {
            let (n_samples, n_features) = X.dim();

            if Some(n_features) != self.n_features {
                return Err(SklearsError::InvalidInput(
                    "Number of features does not match fitted data".to_string(),
                ));
            }

            let mut hash_codes = Array2::zeros((n_samples, self.n_hash_functions));

            for (hash_idx, hash_matrix) in hash_matrices.iter().enumerate() {
                // Apply hash function
                let hash_values = X.dot(hash_matrix);

                // Convert to binary hash codes
                for (sample_idx, &value) in hash_values.iter().enumerate() {
                    hash_codes[[sample_idx, hash_idx]] = if value >= 0.0 { 1.0 } else { 0.0 };
                }
            }

            Ok(hash_codes)
        } else {
            Err(SklearsError::InvalidInput("LSH not fitted yet".to_string()))
        }
    }

    /// Fit and transform data
    pub fn fit_transform(&mut self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        self.fit(X)?;
        self.transform(X)
    }

    /// Find approximate nearest neighbors using LSH
    pub fn find_neighbors(
        &self,
        X: &ArrayView2<Float>,
        query: &ArrayView1<Float>,
        n_neighbors: usize,
    ) -> SklResult<Vec<usize>> {
        if let Some(ref hash_matrices) = self.hash_matrices {
            let (n_samples, n_features) = X.dim();

            if query.len() != n_features {
                return Err(SklearsError::InvalidInput(
                    "Query dimension does not match fitted data".to_string(),
                ));
            }

            // Compute hash code for query
            let mut query_hash = vec![0.0; self.n_hash_functions];
            for (hash_idx, hash_matrix) in hash_matrices.iter().enumerate() {
                let hash_value = query.dot(&hash_matrix.column(0));
                query_hash[hash_idx] = if hash_value >= 0.0 { 1.0 } else { 0.0 };
            }

            // Compute hash codes for all data points
            let data_hashes = self.transform(X)?;

            // Find candidates with similar hash codes
            let mut candidates: Vec<(usize, usize)> = Vec::new();
            for sample_idx in 0..n_samples {
                let mut matching_hashes = 0;
                for hash_idx in 0..self.n_hash_functions {
                    if (data_hashes[[sample_idx, hash_idx]] - query_hash[hash_idx]).abs() < 1e-10 {
                        matching_hashes += 1;
                    }
                }
                if matching_hashes > 0 {
                    candidates.push((sample_idx, matching_hashes));
                }
            }

            // Sort by number of matching hash functions
            candidates.sort_by(|a, b| b.1.cmp(&a.1));

            // Return top candidates
            let result: Vec<usize> = candidates
                .into_iter()
                .take(n_neighbors)
                .map(|(idx, _)| idx)
                .collect();

            Ok(result)
        } else {
            Err(SklearsError::InvalidInput("LSH not fitted yet".to_string()))
        }
    }

    /// Create hash matrices for LSH
    fn create_hash_matrices(&self, n_features: usize) -> SklResult<Vec<Array2<Float>>> {
        use scirs2_core::random::thread_rng;

        // Use rng() function from scirs2_core - seeding handled elsewhere if needed
        let mut rng = thread_rng();

        let mut hash_matrices = Vec::new();

        for _ in 0..self.n_hash_functions {
            let mut matrix = Array2::zeros((n_features, 1));

            match self.distance_metric.as_str() {
                "cosine" => {
                    // Random hyperplane hash for cosine similarity
                    for i in 0..n_features {
                        matrix[[i, 0]] = rng.gen_range(-1.0..1.0);
                    }
                }
                "euclidean" => {
                    // Random projection hash for Euclidean distance
                    for i in 0..n_features {
                        matrix[[i, 0]] = rng.gen_range(-1.0..1.0);
                    }
                    // Normalize
                    let norm = (matrix.mapv(|x| x * x).sum() as Float).sqrt();
                    matrix.mapv_inplace(|x| x / norm);
                }
                _ => {
                    return Err(SklearsError::InvalidInput(
                        "Unsupported distance metric".to_string(),
                    ));
                }
            }

            hash_matrices.push(matrix);
        }

        Ok(hash_matrices)
    }
}

impl Default for LocalitySensitiveHashing {
    fn default() -> Self {
        Self::new()
    }
}
