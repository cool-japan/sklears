//! Minimum Covariance Determinant (MCD) Estimator

use crate::utils::{matrix_determinant, matrix_inverse};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Minimum Covariance Determinant (MCD) Estimator
///
/// Robust covariance estimator that finds the subset of observations
/// that minimizes the determinant of the covariance matrix.
/// This is useful for outlier detection.
///
/// # Parameters
///
/// * `support_fraction` - Fraction of points to include in the support
/// * `random_state` - Random state for reproducibility
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::MinCovDet;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [100.0, 100.0]]; // Last point is outlier
///
/// let estimator = MinCovDet::new().support_fraction(0.75);
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// ```
#[derive(Debug, Clone)]
pub struct MinCovDet<S = Untrained> {
    state: S,
    support_fraction: Option<f64>,
    random_state: Option<u64>,
    store_precision: bool,
    assume_centered: bool,
    n_trials: usize,
    use_fast_mcd: bool,
}

/// Trained state for MinCovDet
#[derive(Debug, Clone)]
pub struct MinCovDetTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The support mask indicating which samples were used
    pub support: Array1<bool>,
    /// The Mahalanobis distances for each sample
    pub dist: Array1<f64>,
}

impl MinCovDet<Untrained> {
    /// Create a new MinCovDet instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            support_fraction: None,
            random_state: None,
            store_precision: true,
            assume_centered: false,
            n_trials: 500,
            use_fast_mcd: true,
        }
    }

    /// Set the support fraction
    pub fn support_fraction(mut self, support_fraction: f64) -> Self {
        self.support_fraction = Some(support_fraction.clamp(0.1, 1.0));
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Enable FastMCD algorithm (default is true)
    pub fn use_fast_mcd(mut self, use_fast: bool) -> Self {
        self.use_fast_mcd = use_fast;
        if use_fast {
            self.n_trials = 500; // FastMCD uses more trials
        } else {
            self.n_trials = 50; // Basic MCD uses fewer trials
        }
        self
    }

    /// Set number of trials for subset selection
    pub fn n_trials(mut self, n_trials: usize) -> Self {
        self.n_trials = n_trials;
        self
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }
}

impl Default for MinCovDet<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MinCovDet<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MinCovDet<Untrained> {
    type Fitted = MinCovDet<MinCovDetTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < n_features + 1 {
            return Err(SklearsError::InvalidInput(
                "Need more samples than features".to_string(),
            ));
        }

        // Determine support size
        let support_fraction = self
            .support_fraction
            .unwrap_or(((n_samples + n_features + 1) as f64 / 2.0) / n_samples as f64);
        let h = ((support_fraction * n_samples as f64).ceil() as usize).min(n_samples);

        // Choose algorithm based on use_fast_mcd flag
        let (best_indices, best_det) = if self.use_fast_mcd {
            fast_mcd_algorithm(
                &x,
                h,
                self.n_trials,
                self.assume_centered,
                self.random_state,
            )?
        } else {
            basic_mcd_algorithm(&x, h, self.n_trials.min(50), self.assume_centered)?
        };

        if best_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Could not find valid subset".to_string(),
            ));
        }

        // Compute final covariance using best subset
        let subset_data: Vec<Vec<f64>> = best_indices.iter().map(|&i| x.row(i).to_vec()).collect();

        let (covariance, mean) = compute_subset_covariance(&subset_data, self.assume_centered)?;

        // Create support mask
        let mut support = Array1::from_elem(n_samples, false);
        for &idx in &best_indices {
            support[idx] = true;
        }

        // Compute precision matrix if requested
        let precision = if self.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(MinCovDet {
            state: MinCovDetTrained {
                covariance,
                precision,
                location: mean,
                support,
                dist: Array1::zeros(n_samples), // Will be computed on demand
            },
            support_fraction: self.support_fraction,
            random_state: self.random_state,
            store_precision: self.store_precision,
            assume_centered: self.assume_centered,
            n_trials: self.n_trials,
            use_fast_mcd: self.use_fast_mcd,
        })
    }
}

impl MinCovDet<MinCovDetTrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the location (mean)
    pub fn get_location(&self) -> &Array1<f64> {
        &self.state.location
    }

    /// Get the support mask (which samples were used)
    pub fn get_support(&self) -> &Array1<bool> {
        &self.state.support
    }

    /// Check if samples are inliers based on distance threshold
    pub fn is_inlier(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<bool>> {
        let distances = self.mahalanobis_distance(x)?;
        let threshold = 2.5; // Common threshold for outlier detection
        Ok(distances.mapv(|d| d <= threshold))
    }

    /// Compute Mahalanobis distance
    pub fn mahalanobis_distance(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let x = *x;
        let precision = self.state.precision.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Precision matrix not computed".to_string())
        })?;

        let mut distances = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let centered = &sample - &self.state.location;
            let temp = precision.dot(&centered);
            distances[i] = centered.dot(&temp).sqrt();
        }

        Ok(distances)
    }
}

/// Compute covariance for a subset of data
fn compute_subset_covariance(
    subset_data: &[Vec<f64>],
    assume_centered: bool,
) -> SklResult<(Array2<f64>, Array1<f64>)> {
    if subset_data.is_empty() {
        return Err(SklearsError::InvalidInput("Empty subset".to_string()));
    }

    let n_samples = subset_data.len();
    let n_features = subset_data[0].len();

    // Compute mean
    let mean = if assume_centered {
        Array1::zeros(n_features)
    } else {
        let mut sum = Array1::zeros(n_features);
        for sample in subset_data {
            for (j, &val) in sample.iter().enumerate() {
                sum[j] += val;
            }
        }
        sum / n_samples as f64
    };

    // Compute covariance
    let mut cov = Array2::zeros((n_features, n_features));

    for sample in subset_data {
        for i in 0..n_features {
            for j in 0..n_features {
                let ci = sample[i] - mean[i];
                let cj = sample[j] - mean[j];
                cov[[i, j]] += ci * cj;
            }
        }
    }

    cov /= (n_samples - 1) as f64;

    Ok((cov, mean))
}

/// Basic MCD algorithm - simple random subset selection
fn basic_mcd_algorithm(
    x: &ArrayView2<Float>,
    h: usize,
    n_trials: usize,
    assume_centered: bool,
) -> SklResult<(Vec<usize>, f64)> {
    let n_samples = x.nrows();
    let mut best_det = f64::INFINITY;
    let mut best_indices = Vec::new();

    for trial in 0..n_trials {
        // Select random subset
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Simple deterministic shuffling based on trial number
        for i in 0..h {
            let swap_idx = (i + trial * 7) % (n_samples - i) + i;
            indices.swap(i, swap_idx);
        }

        let subset_indices = &indices[..h];

        // Compute covariance for this subset
        let subset_data: Vec<Vec<f64>> =
            subset_indices.iter().map(|&i| x.row(i).to_vec()).collect();

        if let Ok((cov, _)) = compute_subset_covariance(&subset_data, assume_centered) {
            let det = matrix_determinant(&cov);
            if det > 0.0 && det < best_det {
                best_det = det;
                best_indices = subset_indices.to_vec();
            }
        }
    }

    Ok((best_indices, best_det))
}

/// FastMCD algorithm with concentration steps
fn fast_mcd_algorithm(
    x: &ArrayView2<Float>,
    h: usize,
    n_trials: usize,
    assume_centered: bool,
    random_state: Option<u64>,
) -> SklResult<(Vec<usize>, f64)> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 * h {
        return basic_mcd_algorithm(x, h, n_trials, assume_centered);
    }

    let mut best_det = f64::INFINITY;
    let mut best_indices = Vec::new();

    // Phase 1: Multiple random starts with concentration steps
    for trial in 0..n_trials {
        // Initial random subset
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Better pseudo-random shuffling using trial and optional seed
        let seed = random_state.unwrap_or(0) + trial as u64;
        for i in 0..h {
            let rng_val = (seed
                .wrapping_mul(1103515245)
                .wrapping_add(12345)
                .wrapping_add(i as u64 * 17))
                % (n_samples - i) as u64;
            let swap_idx = i + rng_val as usize;
            indices.swap(i, swap_idx);
        }

        let mut current_indices = indices[..h].to_vec();

        // Concentration steps (C-steps)
        let max_c_steps = 10;
        let mut converged = false;

        for _c_step in 0..max_c_steps {
            // Compute covariance and mean for current subset
            let subset_data: Vec<Vec<f64>> =
                current_indices.iter().map(|&i| x.row(i).to_vec()).collect();

            let (cov, mean) = match compute_subset_covariance(&subset_data, assume_centered) {
                Ok(result) => result,
                Err(_) => break, // Skip this trial if covariance computation fails
            };

            // Compute precision matrix for distance calculation
            let precision = match matrix_inverse(&cov) {
                Ok(prec) => prec,
                Err(_) => break, // Skip this trial if matrix is singular
            };

            // Compute Mahalanobis distances for all samples
            let mut distances: Vec<(f64, usize)> = Vec::new();

            for (idx, sample) in x.axis_iter(Axis(0)).enumerate() {
                let centered = &sample - &mean;
                let temp = precision.dot(&centered);
                let dist_sq = centered.dot(&temp);
                if dist_sq >= 0.0 {
                    distances.push((dist_sq, idx));
                }
            }

            if distances.len() < h {
                break; // Not enough valid distances
            }

            // Sort by distance and select h closest points
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let new_indices: Vec<usize> = distances[..h].iter().map(|(_, idx)| *idx).collect();

            // Check for convergence
            if new_indices == current_indices {
                converged = true;
                break;
            }

            current_indices = new_indices;
        }

        // Evaluate final subset
        {
            // Always evaluate, even if not fully converged
            let subset_data: Vec<Vec<f64>> =
                current_indices.iter().map(|&i| x.row(i).to_vec()).collect();

            if let Ok((cov, _)) = compute_subset_covariance(&subset_data, assume_centered) {
                let det = matrix_determinant(&cov);
                if det > 0.0 && det < best_det {
                    best_det = det;
                    best_indices = current_indices;
                }
            }
        }
    }

    Ok((best_indices, best_det))
}
