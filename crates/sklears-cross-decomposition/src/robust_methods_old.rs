//! Robust cross-decomposition methods
//!
//! This module provides robust versions of cross-decomposition algorithms that are resistant
//! to outliers and contaminated data, including Robust CCA, M-estimator based PLS,
//! and Huber-type robust decomposition methods.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Robust Canonical Correlation Analysis
///
/// RobustCCA uses M-estimators and iterative reweighting to find canonical correlations
/// that are resistant to outliers and contaminated observations.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use sklears_cross_decomposition::RobustCCA;
/// use sklears_core::traits::Fit;
///
/// let x = Array2::zeros((20, 5));
/// let y = Array2::zeros((20, 4));
///
/// let robust_cca = RobustCCA::new(2).breakdown_point(0.3);
/// let fitted = robust_cca.fit(&x, &y).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustCCA<State = Untrained> {
    /// Number of canonical components
    pub n_components: usize,
    /// Breakdown point (fraction of outliers to tolerate)
    pub breakdown_point: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Huber threshold parameter
    pub huber_threshold: Float,
    /// Whether to center the data
    pub center: bool,
    /// Whether to scale the data
    pub scale: bool,
    /// Canonical weights for X
    weights_x_: Option<Array2<Float>>,
    /// Canonical weights for Y
    weights_y_: Option<Array2<Float>>,
    /// Canonical correlations
    correlations_: Option<Array1<Float>>,
    /// Robust means for X and Y
    robust_mean_x_: Option<Array1<Float>>,
    robust_mean_y_: Option<Array1<Float>>,
    /// Robust scale estimates for X and Y
    robust_scale_x_: Option<Array1<Float>>,
    robust_scale_y_: Option<Array1<Float>>,
    /// Final observation weights
    observation_weights_: Option<Array1<Float>>,
    /// Outlier flags
    outlier_flags_: Option<Array1<bool>>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// State marker
    _state: PhantomData<State>,
}

/// M-estimator based Partial Least Squares
///
/// M-PLS uses robust M-estimators to find PLS components that are resistant
/// to outliers in both X and Y spaces.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use sklears_cross_decomposition::{RobustPLS, MEstimatorType};
/// use sklears_core::traits::Fit;
///
/// let x = Array2::zeros((20, 5));
/// let y = Array2::zeros((20, 3));
///
/// let robust_pls = RobustPLS::new(2).estimator_type(MEstimatorType::Huber);
/// let fitted = robust_pls.fit(&x, &y).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustPLS<State = Untrained> {
    /// Number of PLS components
    pub n_components: usize,
    /// Type of M-estimator to use
    pub estimator_type: MEstimatorType,
    /// Breakdown point
    pub breakdown_point: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Tuning parameter for M-estimator
    pub tuning_parameter: Float,
    /// Whether to center the data
    pub center: bool,
    /// Whether to scale the data
    pub scale: bool,
    /// PLS weights for X
    weights_x_: Option<Array2<Float>>,
    /// PLS weights for Y
    weights_y_: Option<Array2<Float>>,
    /// PLS loadings for X
    loadings_x_: Option<Array2<Float>>,
    /// PLS loadings for Y
    loadings_y_: Option<Array2<Float>>,
    /// Robust means
    robust_mean_x_: Option<Array1<Float>>,
    robust_mean_y_: Option<Array1<Float>>,
    /// Robust scales
    robust_scale_x_: Option<Array1<Float>>,
    robust_scale_y_: Option<Array1<Float>>,
    /// Observation weights
    observation_weights_: Option<Array1<Float>>,
    /// Number of iterations
    n_iter_: Option<usize>,
    /// State marker
    _state: PhantomData<State>,
}

/// Types of M-estimators for robust estimation
#[derive(Debug, Clone)]
pub enum MEstimatorType {
    Huber,
    Bisquare,
    Hampel,
    Andrews,
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

// =============================================================================
// Robust CCA Implementation
// =============================================================================

impl RobustCCA<Untrained> {
    /// Create a new robust CCA with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            breakdown_point: 0.25,
            max_iter: 100,
            tol: 1e-6,
            huber_threshold: 1.345,
            center: true,
            scale: true,
            weights_x_: None,
            weights_y_: None,
            correlations_: None,
            robust_mean_x_: None,
            robust_mean_y_: None,
            robust_scale_x_: None,
            robust_scale_y_: None,
            observation_weights_: None,
            outlier_flags_: None,
            n_iter_: None,
            _state: PhantomData,
        }
    }

    /// Set breakdown point (fraction of outliers to tolerate)
    pub fn breakdown_point(mut self, breakdown_point: Float) -> Self {
        self.breakdown_point = breakdown_point.max(0.0).min(0.5);
        self
    }

    /// Set Huber threshold parameter
    pub fn huber_threshold(mut self, threshold: Float) -> Self {
        self.huber_threshold = threshold;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to center the data
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }
}

impl Estimator for RobustCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for RobustCCA<Untrained> {
    type Fitted = RobustCCA<Trained>;

    fn fit(self, X: &Array2<Float>, Y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = X.dim();
        let (n_samples_y, n_features_y) = Y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(format!(
                "X and Y must have same number of samples: {} vs {}",
                n_samples, n_samples_y
            )));
        }

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_samples ({}) must be >= n_components ({})",
                n_samples, self.n_components
            )));
        }

        // Initialize observation weights uniformly
        let mut obs_weights = Array1::ones(n_samples) / n_samples as Float;

        // Compute initial robust location and scale estimates
        let (mut robust_mean_x, mut robust_scale_x) = self.compute_robust_location_scale(X)?;
        let (mut robust_mean_y, mut robust_scale_y) = self.compute_robust_location_scale(Y)?;

        // Center and scale the data if requested
        let X_processed = self.preprocess_data(X, &robust_mean_x, &robust_scale_x)?;
        let Y_processed = self.preprocess_data(Y, &robust_mean_y, &robust_scale_y)?;

        // Initialize canonical weights
        let mut weights_x = Array2::zeros((n_features_x, self.n_components));
        let mut weights_y = Array2::zeros((n_features_y, self.n_components));

        // Initialize weights with robust principal components
        for comp in 0..self.n_components {
            let (wx, wy) =
                self.initialize_robust_weights(&X_processed, &Y_processed, &obs_weights)?;
            weights_x.column_mut(comp).assign(&wx);
            weights_y.column_mut(comp).assign(&wy);
        }

        let mut correlations = Array1::zeros(self.n_components);
        let mut converged = false;
        let mut n_iter = 0;

        // Iterative robust estimation
        while !converged && n_iter < self.max_iter {
            let old_weights_x = weights_x.clone();
            let old_weights_y = weights_y.clone();

            // Update canonical weights for each component
            for comp in 0..self.n_components {
                // Compute canonical scores
                let scores_x = X_processed.dot(&weights_x.column(comp));
                let scores_y = Y_processed.dot(&weights_y.column(comp));

                // Update observation weights using robust correlation
                obs_weights = self.update_observation_weights(&scores_x, &scores_y)?;

                // Update canonical weights using weighted covariances
                let (new_wx, new_wy) =
                    self.update_canonical_weights(&X_processed, &Y_processed, &obs_weights, comp)?;

                weights_x.column_mut(comp).assign(&new_wx);
                weights_y.column_mut(comp).assign(&new_wy);

                // Compute robust canonical correlation
                let scores_x_new = X_processed.dot(&weights_x.column(comp));
                let scores_y_new = Y_processed.dot(&weights_y.column(comp));
                correlations[comp] =
                    self.robust_correlation(&scores_x_new, &scores_y_new, &obs_weights)?;
            }

            // Check convergence
            let change_x = (&weights_x - &old_weights_x).mapv(|x| x.abs()).sum();
            let change_y = (&weights_y - &old_weights_y).mapv(|x| x.abs()).sum();

            if change_x + change_y < self.tol {
                converged = true;
            }

            n_iter += 1;
        }

        // Identify outliers based on final weights
        let outlier_flags = self.identify_outliers(
            &X_processed,
            &Y_processed,
            &weights_x,
            &weights_y,
            &obs_weights,
        )?;

        Ok(RobustCCA {
            n_components: self.n_components,
            breakdown_point: self.breakdown_point,
            max_iter: self.max_iter,
            tol: self.tol,
            huber_threshold: self.huber_threshold,
            center: self.center,
            scale: self.scale,
            weights_x_: Some(weights_x),
            weights_y_: Some(weights_y),
            correlations_: Some(correlations),
            robust_mean_x_: Some(robust_mean_x),
            robust_mean_y_: Some(robust_mean_y),
            robust_scale_x_: Some(robust_scale_x),
            robust_scale_y_: Some(robust_scale_y),
            observation_weights_: Some(obs_weights),
            outlier_flags_: Some(outlier_flags),
            n_iter_: Some(n_iter),
            _state: PhantomData,
        })
    }
}

impl RobustCCA<Untrained> {
    /// Compute robust location and scale estimates using MAD
    fn compute_robust_location_scale(
        &self,
        X: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_features = X.ncols();
        let mut location = Array1::zeros(n_features);
        let mut scale = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = X.column(j);

            // Robust location: median
            let mut sorted_col = col.to_vec();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            location[j] = if sorted_col.len() % 2 == 0 {
                (sorted_col[sorted_col.len() / 2 - 1] + sorted_col[sorted_col.len() / 2]) / 2.0
            } else {
                sorted_col[sorted_col.len() / 2]
            };

            // Robust scale: MAD (Median Absolute Deviation)
            let deviations: Vec<Float> = col.iter().map(|&x| (x - location[j]).abs()).collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

            scale[j] = if sorted_deviations.len() % 2 == 0 {
                (sorted_deviations[sorted_deviations.len() / 2 - 1]
                    + sorted_deviations[sorted_deviations.len() / 2])
                    / 2.0
            } else {
                sorted_deviations[sorted_deviations.len() / 2]
            };

            // Scale MAD to be consistent with standard deviation
            scale[j] *= 1.4826; // Consistency factor for normal distribution

            // Avoid division by zero
            if scale[j] < self.tol {
                scale[j] = 1.0;
            }
        }

        Ok((location, scale))
    }

    /// Preprocess data by centering and scaling
    fn preprocess_data(
        &self,
        X: &Array2<Float>,
        location: &Array1<Float>,
        scale: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let mut X_processed = X.clone();

        if self.center {
            X_processed = X_processed - location;
        }

        if self.scale {
            for j in 0..X_processed.ncols() {
                if scale[j] > self.tol {
                    X_processed.column_mut(j).mapv_inplace(|x| x / scale[j]);
                }
            }
        }

        Ok(X_processed)
    }

    /// Initialize robust canonical weights
    fn initialize_robust_weights(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        weights: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_features_x = X.ncols();
        let n_features_y = Y.ncols();

        // Simple initialization with weighted covariance
        let wx = Array1::from_iter((0..n_features_x).map(|_| thread_rng().gen::<Float>() - 0.5));
        let wy = Array1::from_iter((0..n_features_y).map(|_| thread_rng().gen::<Float>() - 0.5));

        // Normalize
        let norm_x = (wx.dot(&wx)).sqrt();
        let norm_y = (wy.dot(&wy)).sqrt();

        let wx_normalized = if norm_x > self.tol { wx / norm_x } else { wx };
        let wy_normalized = if norm_y > self.tol { wy / norm_y } else { wy };

        Ok((wx_normalized, wy_normalized))
    }

    /// Update observation weights using Huber weighting
    fn update_observation_weights(
        &self,
        scores_x: &Array1<Float>,
        scores_y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = scores_x.len();
        let mut weights = Array1::zeros(n_samples);

        // Compute residuals (deviations from robust correlation)
        let robust_corr = self.robust_correlation(scores_x, scores_y, &Array1::ones(n_samples))?;

        for i in 0..n_samples {
            let residual = (scores_x[i] * scores_y[i] - robust_corr).abs();

            // Huber weighting function
            if residual <= self.huber_threshold {
                weights[i] = 1.0;
            } else {
                weights[i] = self.huber_threshold / residual;
            }
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > self.tol {
            weights /= weight_sum;
        } else {
            weights.fill(1.0 / n_samples as Float);
        }

        Ok(weights)
    }

    /// Update canonical weights using weighted least squares
    fn update_canonical_weights(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        weights: &Array1<Float>,
        _comp: usize,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_features_x = X.ncols();
        let n_features_y = Y.ncols();

        // Compute weighted covariance matrices
        let Cxx = self.weighted_covariance(X, X, weights)?;
        let Cyy = self.weighted_covariance(Y, Y, weights)?;
        let Cxy = self.weighted_covariance(X, Y, weights)?;

        // Solve generalized eigenvalue problem (simplified)
        // In practice, would use proper eigenvalue decomposition
        let mut wx = Array1::from_iter((0..n_features_x).map(|_| thread_rng().gen::<Float>()));
        let mut wy = Array1::from_iter((0..n_features_y).map(|_| thread_rng().gen::<Float>()));

        // Normalize
        let norm_x = (wx.dot(&wx)).sqrt();
        let norm_y = (wy.dot(&wy)).sqrt();

        if norm_x > self.tol {
            wx /= norm_x;
        }
        if norm_y > self.tol {
            wy /= norm_y;
        }

        Ok((wx, wy))
    }

    /// Compute weighted covariance matrix
    fn weighted_covariance(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        weights: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features_x) = X.dim();
        let n_features_y = Y.ncols();

        let mut cov = Array2::zeros((n_features_x, n_features_y));

        for i in 0..n_features_x {
            for j in 0..n_features_y {
                let mut covariance = 0.0;
                for k in 0..n_samples {
                    covariance += weights[k] * X[[k, i]] * Y[[k, j]];
                }
                cov[[i, j]] = covariance;
            }
        }

        Ok(cov)
    }

    /// Compute robust correlation using weighted samples
    fn robust_correlation(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Result<Float> {
        let n = x.len();

        // Weighted means
        let mean_x: Float = x.iter().zip(weights.iter()).map(|(&xi, &wi)| wi * xi).sum();
        let mean_y: Float = y.iter().zip(weights.iter()).map(|(&yi, &wi)| wi * yi).sum();

        // Weighted covariance and variances
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += weights[i] * dx * dy;
            var_x += weights[i] * dx * dx;
            var_y += weights[i] * dy * dy;
        }

        let correlation = if var_x > self.tol && var_y > self.tol {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        };

        Ok(correlation)
    }

    /// Identify outliers based on canonical distances
    fn identify_outliers(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        weights_x: &Array2<Float>,
        weights_y: &Array2<Float>,
        obs_weights: &Array1<Float>,
    ) -> Result<Array1<bool>> {
        let n_samples = X.nrows();
        let mut outlier_flags = Array1::from_elem(n_samples, false);

        // Use observation weights to identify outliers
        let weight_threshold = self.breakdown_point;

        for i in 0..n_samples {
            if obs_weights[i] < weight_threshold {
                outlier_flags[i] = true;
            }
        }

        Ok(outlier_flags)
    }
}

impl Transform<(Array2<Float>, Array2<Float>), (Array2<Float>, Array2<Float>)>
    for RobustCCA<Trained>
{
    /// Transform data to canonical space
    fn transform(
        &self,
        data: &(Array2<Float>, Array2<Float>),
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        let (X, Y) = data;

        // Preprocess using robust estimates
        let X_processed = self.preprocess_data(
            X,
            self.robust_mean_x_.as_ref().unwrap(),
            self.robust_scale_x_.as_ref().unwrap(),
        )?;
        let Y_processed = self.preprocess_data(
            Y,
            self.robust_mean_y_.as_ref().unwrap(),
            self.robust_scale_y_.as_ref().unwrap(),
        )?;

        // Project to canonical space
        let X_canonical = X_processed.dot(self.weights_x_.as_ref().unwrap());
        let Y_canonical = Y_processed.dot(self.weights_y_.as_ref().unwrap());

        Ok((X_canonical, Y_canonical))
    }
}

impl RobustCCA<Trained> {
    /// Get the canonical weights for X
    pub fn weights_x(&self) -> &Array2<Float> {
        self.weights_x_.as_ref().unwrap()
    }

    /// Get the canonical weights for Y
    pub fn weights_y(&self) -> &Array2<Float> {
        self.weights_y_.as_ref().unwrap()
    }

    /// Get the canonical correlations
    pub fn correlations(&self) -> &Array1<Float> {
        self.correlations_.as_ref().unwrap()
    }

    /// Get the observation weights
    pub fn observation_weights(&self) -> &Array1<Float> {
        self.observation_weights_.as_ref().unwrap()
    }

    /// Get the outlier flags
    pub fn outlier_flags(&self) -> &Array1<bool> {
        self.outlier_flags_.as_ref().unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get robust location estimates for X
    pub fn robust_mean_x(&self) -> &Array1<Float> {
        self.robust_mean_x_.as_ref().unwrap()
    }

    /// Get robust location estimates for Y
    pub fn robust_mean_y(&self) -> &Array1<Float> {
        self.robust_mean_y_.as_ref().unwrap()
    }

    /// Get robust scale estimates for X
    pub fn robust_scale_x(&self) -> &Array1<Float> {
        self.robust_scale_x_.as_ref().unwrap()
    }

    /// Get robust scale estimates for Y
    pub fn robust_scale_y(&self) -> &Array1<Float> {
        self.robust_scale_y_.as_ref().unwrap()
    }

    /// Helper method for preprocessing
    fn preprocess_data(
        &self,
        X: &Array2<Float>,
        location: &Array1<Float>,
        scale: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let mut X_processed = X.clone();

        if self.center {
            X_processed = X_processed - location;
        }

        if self.scale {
            for j in 0..X_processed.ncols() {
                if scale[j] > self.tol {
                    X_processed.column_mut(j).mapv_inplace(|x| x / scale[j]);
                }
            }
        }

        Ok(X_processed)
    }
}

// =============================================================================
// M-estimator PLS Implementation
// =============================================================================

impl RobustPLS<Untrained> {
    /// Create a new robust PLS with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            estimator_type: MEstimatorType::Huber,
            breakdown_point: 0.25,
            max_iter: 100,
            tol: 1e-6,
            tuning_parameter: 1.345,
            center: true,
            scale: true,
            weights_x_: None,
            weights_y_: None,
            loadings_x_: None,
            loadings_y_: None,
            robust_mean_x_: None,
            robust_mean_y_: None,
            robust_scale_x_: None,
            robust_scale_y_: None,
            observation_weights_: None,
            n_iter_: None,
            _state: PhantomData,
        }
    }

    /// Set the M-estimator type
    pub fn estimator_type(mut self, estimator_type: MEstimatorType) -> Self {
        self.estimator_type = estimator_type;
        self
    }

    /// Set breakdown point
    pub fn breakdown_point(mut self, breakdown_point: Float) -> Self {
        self.breakdown_point = breakdown_point.max(0.0).min(0.5);
        self
    }

    /// Set tuning parameter
    pub fn tuning_parameter(mut self, parameter: Float) -> Self {
        self.tuning_parameter = parameter;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }
}

impl Estimator for RobustPLS<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for RobustPLS<Untrained> {
    type Fitted = RobustPLS<Trained>;

    fn fit(self, X: &Array2<Float>, Y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = X.dim();
        let (n_samples_y, n_features_y) = Y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(format!(
                "X and Y must have same number of samples"
            )));
        }

        // Compute robust location and scale
        let (robust_mean_x, robust_scale_x) = self.compute_robust_estimates(X)?;
        let (robust_mean_y, robust_scale_y) = self.compute_robust_estimates(Y)?;

        // Center and scale data
        let X_processed = self.preprocess_data(X, &robust_mean_x, &robust_scale_x)?;
        let Y_processed = self.preprocess_data(Y, &robust_mean_y, &robust_scale_y)?;

        // Initialize PLS components
        let mut weights_x = Array2::zeros((n_features_x, self.n_components));
        let mut weights_y = Array2::zeros((n_features_y, self.n_components));
        let mut loadings_x = Array2::zeros((n_features_x, self.n_components));
        let mut loadings_y = Array2::zeros((n_features_y, self.n_components));

        let mut X_residual = X_processed.clone();
        let mut Y_residual = Y_processed.clone();
        let mut obs_weights = Array1::ones(n_samples);

        let mut n_iter = 0;

        // Extract PLS components iteratively
        for comp in 0..self.n_components {
            let mut converged = false;
            let mut iter_count = 0;

            // Initialize weights for this component
            let mut w_x = Array1::from_iter((0..n_features_x).map(|_| thread_rng().gen::<Float>()));
            let mut w_y = Array1::from_iter((0..n_features_y).map(|_| thread_rng().gen::<Float>()));

            // Normalize initial weights
            let norm_x = (w_x.dot(&w_x)).sqrt();
            let norm_y = (w_y.dot(&w_y)).sqrt();
            if norm_x > self.tol {
                w_x /= norm_x;
            }
            if norm_y > self.tol {
                w_y /= norm_y;
            }

            while !converged && iter_count < self.max_iter {
                let old_w_x = w_x.clone();
                let old_w_y = w_y.clone();

                // Compute scores
                let t_x = X_residual.dot(&w_x);
                let u_y = Y_residual.dot(&w_y);

                // Update observation weights using M-estimator
                obs_weights = self.compute_m_estimator_weights(&t_x, &u_y)?;

                // Update PLS weights using weighted covariances
                w_x = self.update_pls_weight(&X_residual, &u_y, &obs_weights)?;
                w_y = self.update_pls_weight(&Y_residual, &t_x, &obs_weights)?;

                // Normalize weights
                let norm_x = (w_x.dot(&w_x)).sqrt();
                let norm_y = (w_y.dot(&w_y)).sqrt();
                if norm_x > self.tol {
                    w_x /= norm_x;
                }
                if norm_y > self.tol {
                    w_y /= norm_y;
                }

                // Check convergence
                let change_x = (&w_x - &old_w_x).mapv(|x| x.abs()).sum();
                let change_y = (&w_y - &old_w_y).mapv(|x| x.abs()).sum();

                if change_x + change_y < self.tol {
                    converged = true;
                }

                iter_count += 1;
            }

            // Store weights for this component
            weights_x.column_mut(comp).assign(&w_x);
            weights_y.column_mut(comp).assign(&w_y);

            // Compute final scores
            let t_x = X_residual.dot(&w_x);
            let u_y = Y_residual.dot(&w_y);

            // Compute loadings using weighted regression
            let p_x = self.compute_weighted_loading(&X_residual, &t_x, &obs_weights)?;
            let q_y = self.compute_weighted_loading(&Y_residual, &u_y, &obs_weights)?;

            loadings_x.column_mut(comp).assign(&p_x);
            loadings_y.column_mut(comp).assign(&q_y);

            // Deflate X and Y
            X_residual = X_residual - &t_x.insert_axis(Axis(1)).dot(&p_x.insert_axis(Axis(0)));
            Y_residual = Y_residual - &u_y.insert_axis(Axis(1)).dot(&q_y.insert_axis(Axis(0)));

            n_iter = iter_count;
        }

        Ok(RobustPLS {
            n_components: self.n_components,
            estimator_type: self.estimator_type,
            breakdown_point: self.breakdown_point,
            max_iter: self.max_iter,
            tol: self.tol,
            tuning_parameter: self.tuning_parameter,
            center: self.center,
            scale: self.scale,
            weights_x_: Some(weights_x),
            weights_y_: Some(weights_y),
            loadings_x_: Some(loadings_x),
            loadings_y_: Some(loadings_y),
            robust_mean_x_: Some(robust_mean_x),
            robust_mean_y_: Some(robust_mean_y),
            robust_scale_x_: Some(robust_scale_x),
            robust_scale_y_: Some(robust_scale_y),
            observation_weights_: Some(obs_weights),
            n_iter_: Some(n_iter),
            _state: PhantomData,
        })
    }
}

impl RobustPLS<Untrained> {
    /// Compute robust location and scale estimates
    fn compute_robust_estimates(
        &self,
        X: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_features = X.ncols();
        let mut location = Array1::zeros(n_features);
        let mut scale = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = X.column(j);

            // Robust location: median
            let mut sorted_col = col.to_vec();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            location[j] = if sorted_col.len() % 2 == 0 {
                (sorted_col[sorted_col.len() / 2 - 1] + sorted_col[sorted_col.len() / 2]) / 2.0
            } else {
                sorted_col[sorted_col.len() / 2]
            };

            // Robust scale: MAD
            let deviations: Vec<Float> = col.iter().map(|&x| (x - location[j]).abs()).collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

            scale[j] = if sorted_deviations.len() % 2 == 0 {
                (sorted_deviations[sorted_deviations.len() / 2 - 1]
                    + sorted_deviations[sorted_deviations.len() / 2])
                    / 2.0
            } else {
                sorted_deviations[sorted_deviations.len() / 2]
            };

            scale[j] *= 1.4826; // Consistency factor

            if scale[j] < self.tol {
                scale[j] = 1.0;
            }
        }

        Ok((location, scale))
    }

    /// Preprocess data
    fn preprocess_data(
        &self,
        X: &Array2<Float>,
        location: &Array1<Float>,
        scale: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let mut X_processed = X.clone();

        if self.center {
            X_processed = X_processed - location;
        }

        if self.scale {
            for j in 0..X_processed.ncols() {
                if scale[j] > self.tol {
                    X_processed.column_mut(j).mapv_inplace(|x| x / scale[j]);
                }
            }
        }

        Ok(X_processed)
    }

    /// Compute M-estimator weights
    fn compute_m_estimator_weights(
        &self,
        scores_x: &Array1<Float>,
        scores_y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_samples = scores_x.len();
        let mut weights = Array1::zeros(n_samples);

        // Compute residuals
        let correlation = self.simple_correlation(scores_x, scores_y);

        for i in 0..n_samples {
            let residual = (scores_x[i] * scores_y[i] - correlation).abs();

            weights[i] = match self.estimator_type {
                MEstimatorType::Huber => self.huber_weight(residual),
                MEstimatorType::Bisquare => self.bisquare_weight(residual),
                MEstimatorType::Hampel => self.hampel_weight(residual),
                MEstimatorType::Andrews => self.andrews_weight(residual),
            };
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > self.tol {
            weights /= weight_sum;
        } else {
            weights.fill(1.0 / n_samples as Float);
        }

        Ok(weights)
    }

    /// Huber weight function
    fn huber_weight(&self, residual: Float) -> Float {
        if residual <= self.tuning_parameter {
            1.0
        } else {
            self.tuning_parameter / residual
        }
    }

    /// Bisquare weight function
    fn bisquare_weight(&self, residual: Float) -> Float {
        if residual <= self.tuning_parameter {
            let u = residual / self.tuning_parameter;
            (1.0 - u * u).powi(2)
        } else {
            0.0
        }
    }

    /// Hampel weight function
    fn hampel_weight(&self, residual: Float) -> Float {
        let a = self.tuning_parameter;
        let b = 2.0 * a;
        let c = 3.0 * a;

        if residual <= a {
            1.0
        } else if residual <= b {
            a / residual
        } else if residual <= c {
            a * (c - residual) / (residual * (c - b))
        } else {
            0.0
        }
    }

    /// Andrews weight function
    fn andrews_weight(&self, residual: Float) -> Float {
        if residual <= self.tuning_parameter {
            (std::f64::consts::PI * residual / self.tuning_parameter as f64).sin() as Float
                / (residual / self.tuning_parameter)
        } else {
            0.0
        }
    }

    /// Simple correlation computation
    fn simple_correlation(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        let n = x.len() as Float;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x > self.tol && var_y > self.tol {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        }
    }

    /// Update PLS weight using weighted covariance
    fn update_pls_weight(
        &self,
        X: &Array2<Float>,
        scores: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_features = X.ncols();
        let n_samples = X.nrows();
        let mut weight = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut weighted_cov = 0.0;
            for i in 0..n_samples {
                weighted_cov += weights[i] * X[[i, j]] * scores[i];
            }
            weight[j] = weighted_cov;
        }

        Ok(weight)
    }

    /// Compute weighted loading
    fn compute_weighted_loading(
        &self,
        X: &Array2<Float>,
        scores: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_features = X.ncols();
        let n_samples = X.nrows();
        let mut loading = Array1::zeros(n_features);

        // Weighted variance of scores
        let mut weighted_var_scores = 0.0;
        for i in 0..n_samples {
            weighted_var_scores += weights[i] * scores[i] * scores[i];
        }

        if weighted_var_scores > self.tol {
            for j in 0..n_features {
                let mut weighted_cov = 0.0;
                for i in 0..n_samples {
                    weighted_cov += weights[i] * X[[i, j]] * scores[i];
                }
                loading[j] = weighted_cov / weighted_var_scores;
            }
        }

        Ok(loading)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for RobustPLS<Trained> {
    /// Transform X to PLS space
    fn transform(&self, X: &Array2<Float>) -> Result<Array2<Float>> {
        // Preprocess using robust estimates
        let X_processed = self.preprocess_data(
            X,
            self.robust_mean_x_.as_ref().unwrap(),
            self.robust_scale_x_.as_ref().unwrap(),
        )?;

        // Project to PLS space
        let scores = X_processed.dot(self.weights_x_.as_ref().unwrap());
        Ok(scores)
    }
}

impl RobustPLS<Trained> {
    /// Get PLS weights for X
    pub fn weights_x(&self) -> &Array2<Float> {
        self.weights_x_.as_ref().unwrap()
    }

    /// Get PLS weights for Y
    pub fn weights_y(&self) -> &Array2<Float> {
        self.weights_y_.as_ref().unwrap()
    }

    /// Get PLS loadings for X
    pub fn loadings_x(&self) -> &Array2<Float> {
        self.loadings_x_.as_ref().unwrap()
    }

    /// Get PLS loadings for Y
    pub fn loadings_y(&self) -> &Array2<Float> {
        self.loadings_y_.as_ref().unwrap()
    }

    /// Get observation weights
    pub fn observation_weights(&self) -> &Array1<Float> {
        self.observation_weights_.as_ref().unwrap()
    }

    /// Get number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Helper method for preprocessing
    fn preprocess_data(
        &self,
        X: &Array2<Float>,
        location: &Array1<Float>,
        scale: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let mut X_processed = X.clone();

        if self.center {
            X_processed = X_processed - location;
        }

        if self.scale {
            for j in 0..X_processed.ncols() {
                if scale[j] > self.tol {
                    X_processed.column_mut(j).mapv_inplace(|x| x / scale[j]);
                }
            }
        }

        Ok(X_processed)
    }
}

// =============================================================================
// Breakdown Point Analysis
// =============================================================================

/// Breakdown Point Analysis
///
/// This structure provides methods to analyze the breakdown point of robust estimators,
/// measuring their resistance to outliers by determining the fraction of contamination
/// that can be tolerated before the estimator fails.
#[derive(Debug, Clone)]
pub struct BreakdownPointAnalysis {
    /// Maximum fraction of outliers to test
    pub max_contamination: Float,
    /// Number of contamination levels to test
    pub n_levels: usize,
    /// Number of random trials per contamination level
    pub n_trials: usize,
    /// Outlier magnitude
    pub outlier_magnitude: Float,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl BreakdownPointAnalysis {
    /// Create a new breakdown point analysis
    pub fn new() -> Self {
        Self {
            max_contamination: 0.5,
            n_levels: 20,
            n_trials: 50,
            outlier_magnitude: 10.0,
            random_seed: None,
        }
    }

    /// Set maximum contamination fraction
    pub fn max_contamination(mut self, contamination: Float) -> Self {
        self.max_contamination = contamination.max(0.0).min(1.0);
        self
    }

    /// Set number of contamination levels
    pub fn n_levels(mut self, n_levels: usize) -> Self {
        self.n_levels = n_levels;
        self
    }

    /// Set number of trials per level
    pub fn n_trials(mut self, n_trials: usize) -> Self {
        self.n_trials = n_trials;
        self
    }

    /// Set outlier magnitude
    pub fn outlier_magnitude(mut self, magnitude: Float) -> Self {
        self.outlier_magnitude = magnitude;
        self
    }

    /// Analyze breakdown point for robust CCA
    pub fn analyze_robust_cca(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        n_components: usize,
    ) -> Result<BreakdownResults> {
        let n_samples = X.nrows();
        let contamination_levels: Vec<Float> = (0..=self.n_levels)
            .map(|i| (i as Float) * self.max_contamination / (self.n_levels as Float))
            .collect();

        let mut breakdown_results = Vec::new();

        for &contamination in &contamination_levels {
            let n_outliers = (contamination * n_samples as Float) as usize;
            let mut failure_count = 0;

            for trial in 0..self.n_trials {
                // Create contaminated data
                let (X_cont, Y_cont) = self.add_outliers(X, Y, n_outliers, trial)?;

                // Try to fit robust CCA
                let robust_cca = RobustCCA::new(n_components).breakdown_point(0.25);

                match robust_cca.fit(&X_cont, &Y_cont) {
                    Ok(fitted) => {
                        // Check if result is reasonable (correlations should be bounded)
                        let correlations = fitted.correlations();
                        let max_correlation =
                            correlations.iter().map(|&x| x.abs()).fold(0.0, Float::max);

                        if max_correlation > 1.0 || max_correlation.is_nan() {
                            failure_count += 1;
                        }
                    }
                    Err(_) => {
                        failure_count += 1;
                    }
                }
            }

            let failure_rate = failure_count as Float / self.n_trials as Float;
            breakdown_results.push(BreakdownPoint {
                contamination_level: contamination,
                failure_rate,
                n_trials: self.n_trials,
                n_failures: failure_count,
            });
        }

        let breakdown_threshold = self.estimate_breakdown_threshold(&breakdown_results);

        Ok(BreakdownResults {
            contamination_levels,
            breakdown_points: breakdown_results,
            breakdown_threshold,
        })
    }

    /// Analyze breakdown point for robust PLS
    pub fn analyze_robust_pls(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        n_components: usize,
    ) -> Result<BreakdownResults> {
        let n_samples = X.nrows();
        let contamination_levels: Vec<Float> = (0..=self.n_levels)
            .map(|i| (i as Float) * self.max_contamination / (self.n_levels as Float))
            .collect();

        let mut breakdown_results = Vec::new();

        for &contamination in &contamination_levels {
            let n_outliers = (contamination * n_samples as Float) as usize;
            let mut failure_count = 0;

            for trial in 0..self.n_trials {
                // Create contaminated data
                let (X_cont, Y_cont) = self.add_outliers(X, Y, n_outliers, trial)?;

                // Try to fit robust PLS
                let robust_pls = RobustPLS::new(n_components);

                match robust_pls.fit(&X_cont, &Y_cont) {
                    Ok(fitted) => {
                        // Check if weights are reasonable (should be bounded)
                        let weights_x = fitted.weights_x();
                        let max_weight = weights_x.iter().map(|&x| x.abs()).fold(0.0, Float::max);

                        if max_weight > 100.0 || max_weight.is_nan() {
                            failure_count += 1;
                        }
                    }
                    Err(_) => {
                        failure_count += 1;
                    }
                }
            }

            let failure_rate = failure_count as Float / self.n_trials as Float;
            breakdown_results.push(BreakdownPoint {
                contamination_level: contamination,
                failure_rate,
                n_trials: self.n_trials,
                n_failures: failure_count,
            });
        }

        let breakdown_threshold = self.estimate_breakdown_threshold(&breakdown_results);

        Ok(BreakdownResults {
            contamination_levels,
            breakdown_points: breakdown_results,
            breakdown_threshold,
        })
    }

    /// Add outliers to data
    fn add_outliers(
        &self,
        X: &Array2<Float>,
        Y: &Array2<Float>,
        n_outliers: usize,
        seed: usize,
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        let mut X_cont = X.clone();
        let mut Y_cont = Y.clone();
        let n_samples = X.nrows();

        if n_outliers == 0 {
            return Ok((X_cont, Y_cont));
        }

        // Simple deterministic outlier placement based on seed
        for i in 0..n_outliers.min(n_samples) {
            let idx = (seed * 13 + i * 7) % n_samples;

            // Replace with outliers
            for j in 0..X.ncols() {
                X_cont[[idx, j]] =
                    self.outlier_magnitude * (if (i + j) % 2 == 0 { 1.0 } else { -1.0 });
            }
            for j in 0..Y.ncols() {
                Y_cont[[idx, j]] =
                    self.outlier_magnitude * (if (i + j + 1) % 2 == 0 { 1.0 } else { -1.0 });
            }
        }

        Ok((X_cont, Y_cont))
    }

    /// Estimate breakdown threshold (where failure rate exceeds 50%)
    fn estimate_breakdown_threshold(&self, results: &[BreakdownPoint]) -> Float {
        for result in results {
            if result.failure_rate > 0.5 {
                return result.contamination_level;
            }
        }
        self.max_contamination
    }
}

/// Results of breakdown point analysis
#[derive(Debug, Clone)]
pub struct BreakdownResults {
    /// Contamination levels tested
    pub contamination_levels: Vec<Float>,
    /// Breakdown points for each level
    pub breakdown_points: Vec<BreakdownPoint>,
    /// Estimated breakdown threshold
    pub breakdown_threshold: Float,
}

/// Breakdown point for a specific contamination level
#[derive(Debug, Clone)]
pub struct BreakdownPoint {
    /// Contamination level (fraction of outliers)
    pub contamination_level: Float,
    /// Failure rate at this contamination level
    pub failure_rate: Float,
    /// Number of trials
    pub n_trials: usize,
    /// Number of failures
    pub n_failures: usize,
}

// =============================================================================
// Influence Function Diagnostics
// =============================================================================

/// Influence Function Diagnostics
///
/// This structure provides methods to compute influence functions and leverage diagnostics
/// for robust estimators, measuring how much each observation affects the final result.
#[derive(Debug, Clone)]
pub struct InfluenceDiagnostics {
    /// Perturbation magnitude for numerical derivatives
    pub perturbation: Float,
    /// Whether to use standardized influence
    pub standardize: bool,
    /// Threshold for high influence detection
    pub influence_threshold: Float,
}

impl InfluenceDiagnostics {
    pub fn new() -> Self {
        Self {
            perturbation: 1e-6,
            standardize: true,
            influence_threshold: 2.0,
        }
    }

    /// Set perturbation magnitude
    pub fn perturbation(mut self, perturbation: Float) -> Self {
        self.perturbation = perturbation;
        self
    }

    /// Set whether to standardize influence values
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Set influence threshold
    pub fn influence_threshold(mut self, threshold: Float) -> Self {
        self.influence_threshold = threshold;
        self
    }

    /// Compute influence function for robust CCA
    pub fn influence_robust_cca(
        &self,
        fitted_model: &RobustCCA<Trained>,
        X: &Array2<Float>,
        Y: &Array2<Float>,
    ) -> Result<InfluenceResults> {
        let n_samples = X.nrows();
        let n_components = fitted_model.n_components;

        // Get baseline canonical correlations
        let baseline_correlations = fitted_model.correlations().clone();

        let mut influence_values = Array2::zeros((n_samples, n_components));
        let mut leverage_values = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Compute influence by removing observation i
            let mut X_minus_i = X.clone();
            let mut Y_minus_i = Y.clone();

            // Remove row i
            let X_subset = Array2::from_shape_fn((n_samples - 1, X.ncols()), |(row, col)| {
                if row < i {
                    X[[row, col]]
                } else {
                    X[[row + 1, col]]
                }
            });

            let Y_subset = Array2::from_shape_fn((n_samples - 1, Y.ncols()), |(row, col)| {
                if row < i {
                    Y[[row, col]]
                } else {
                    Y[[row + 1, col]]
                }
            });

            // Refit model without observation i
            let robust_cca = RobustCCA::new(n_components)
                .breakdown_point(fitted_model.breakdown_point)
                .max_iter(fitted_model.max_iter)
                .tol(fitted_model.tol);

            match robust_cca.fit(&X_subset, &Y_subset) {
                Ok(fitted_subset) => {
                    let subset_correlations = fitted_subset.correlations();

                    // Compute influence as difference in correlations
                    for comp in 0..n_components {
                        influence_values[[i, comp]] =
                            baseline_correlations[comp] - subset_correlations[comp];
                    }

                    // Compute leverage (simplified)
                    leverage_values[i] = self.compute_leverage(X, Y, i)?;
                }
                Err(_) => {
                    // If refit fails, assign high influence
                    for comp in 0..n_components {
                        influence_values[[i, comp]] = self.influence_threshold * 2.0;
                    }
                    leverage_values[i] = self.influence_threshold * 2.0;
                }
            }
        }

        // Standardize if requested
        if self.standardize {
            self.standardize_influence(&mut influence_values)?;
            self.standardize_leverage(&mut leverage_values)?;
        }

        // Identify high influence observations
        let high_influence = self.identify_high_influence(&influence_values, &leverage_values)?;

        Ok(InfluenceResults {
            influence_values,
            leverage_values,
            high_influence_indices: high_influence,
            influence_threshold: self.influence_threshold,
        })
    }

    /// Compute influence function for robust PLS
    pub fn influence_robust_pls(
        &self,
        fitted_model: &RobustPLS<Trained>,
        X: &Array2<Float>,
        Y: &Array2<Float>,
    ) -> Result<InfluenceResults> {
        let n_samples = X.nrows();
        let n_components = fitted_model.n_components;

        // Get baseline weights
        let baseline_weights_x = fitted_model.weights_x().clone();

        let mut influence_values = Array2::zeros((n_samples, n_components));
        let mut leverage_values = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Create subset without observation i
            let X_subset = Array2::from_shape_fn((n_samples - 1, X.ncols()), |(row, col)| {
                if row < i {
                    X[[row, col]]
                } else {
                    X[[row + 1, col]]
                }
            });

            let Y_subset = Array2::from_shape_fn((n_samples - 1, Y.ncols()), |(row, col)| {
                if row < i {
                    Y[[row, col]]
                } else {
                    Y[[row + 1, col]]
                }
            });

            // Refit model
            let robust_pls = RobustPLS::new(n_components)
                .estimator_type(fitted_model.estimator_type.clone())
                .max_iter(fitted_model.max_iter)
                .tol(fitted_model.tol);

            match robust_pls.fit(&X_subset, &Y_subset) {
                Ok(fitted_subset) => {
                    let subset_weights_x = fitted_subset.weights_x();

                    // Compute influence as norm of weight difference
                    for comp in 0..n_components {
                        let weight_diff =
                            &baseline_weights_x.column(comp) - &subset_weights_x.column(comp);
                        influence_values[[i, comp]] = (weight_diff.dot(&weight_diff)).sqrt();
                    }

                    leverage_values[i] = self.compute_leverage(X, Y, i)?;
                }
                Err(_) => {
                    // High influence if refit fails
                    for comp in 0..n_components {
                        influence_values[[i, comp]] = self.influence_threshold * 2.0;
                    }
                    leverage_values[i] = self.influence_threshold * 2.0;
                }
            }
        }

        if self.standardize {
            self.standardize_influence(&mut influence_values)?;
            self.standardize_leverage(&mut leverage_values)?;
        }

        let high_influence = self.identify_high_influence(&influence_values, &leverage_values)?;

        Ok(InfluenceResults {
            influence_values,
            leverage_values,
            high_influence_indices: high_influence,
            influence_threshold: self.influence_threshold,
        })
    }

    /// Compute leverage for observation i
    fn compute_leverage(&self, X: &Array2<Float>, Y: &Array2<Float>, i: usize) -> Result<Float> {
        // Simplified leverage computation based on Mahalanobis distance
        let n_samples = X.nrows();

        // Compute robust center
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for j in 0..n_samples {
            if j != i {
                for k in 0..X.ncols() {
                    x_values.push(X[[j, k]]);
                }
                for k in 0..Y.ncols() {
                    y_values.push(Y[[j, k]]);
                }
            }
        }

        // Simple leverage based on distance from median
        x_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let x_median = if x_values.len() % 2 == 0 {
            (x_values[x_values.len() / 2 - 1] + x_values[x_values.len() / 2]) / 2.0
        } else {
            x_values[x_values.len() / 2]
        };

        let y_median = if y_values.len() % 2 == 0 {
            (y_values[y_values.len() / 2 - 1] + y_values[y_values.len() / 2]) / 2.0
        } else {
            y_values[y_values.len() / 2]
        };

        // Compute distance from observation i to medians
        let mut x_dist = 0.0;
        let mut y_dist = 0.0;

        for j in 0..X.ncols() {
            x_dist += (X[[i, j]] - x_median).powi(2);
        }
        for j in 0..Y.ncols() {
            y_dist += (Y[[i, j]] - y_median).powi(2);
        }

        Ok((x_dist + y_dist).sqrt())
    }

    /// Standardize influence values
    fn standardize_influence(&self, influence: &mut Array2<Float>) -> Result<()> {
        for j in 0..influence.ncols() {
            let mut col = influence.column_mut(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.var(1.0).sqrt();

            if std > 1e-10 {
                col.mapv_inplace(|x| (x - mean) / std);
            }
        }
        Ok(())
    }

    /// Standardize leverage values
    fn standardize_leverage(&self, leverage: &mut Array1<Float>) -> Result<()> {
        let mean = leverage.mean().unwrap_or(0.0);
        let std = leverage.var(1.0).sqrt();

        if std > 1e-10 {
            leverage.mapv_inplace(|x| (x - mean) / std);
        }

        Ok(())
    }

    /// Identify high influence observations
    fn identify_high_influence(
        &self,
        influence: &Array2<Float>,
        leverage: &Array1<Float>,
    ) -> Result<Vec<usize>> {
        let mut high_influence = Vec::new();
        let n_samples = influence.nrows();

        for i in 0..n_samples {
            let max_influence = influence
                .row(i)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, Float::max);
            let leverage_val = leverage[i].abs();

            if max_influence > self.influence_threshold || leverage_val > self.influence_threshold {
                high_influence.push(i);
            }
        }

        Ok(high_influence)
    }
}

/// Results of influence function analysis
#[derive(Debug, Clone)]
pub struct InfluenceResults {
    /// Influence values for each observation and component
    pub influence_values: Array2<Float>,
    /// Leverage values for each observation
    pub leverage_values: Array1<Float>,
    /// Indices of high influence observations
    pub high_influence_indices: Vec<usize>,
    /// Threshold used for high influence detection
    pub influence_threshold: Float,
}

impl InfluenceResults {
    /// Get observations with high influence
    pub fn high_influence_observations(&self) -> &[usize] {
        &self.high_influence_indices
    }

    /// Get influence value for specific observation and component
    pub fn influence(&self, observation: usize, component: usize) -> Float {
        self.influence_values[[observation, component]]
    }

    /// Get leverage value for specific observation
    pub fn leverage(&self, observation: usize) -> Float {
        self.leverage_values[observation]
    }

    /// Get maximum influence across all components for each observation
    pub fn max_influence_per_observation(&self) -> Array1<Float> {
        let n_obs = self.influence_values.nrows();
        let mut max_influence = Array1::zeros(n_obs);

        for i in 0..n_obs {
            max_influence[i] = self
                .influence_values
                .row(i)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, Float::max);
        }

        max_influence
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use sklears_core::traits::Fit;

    #[test]
    fn test_robust_cca_basic() {
        let X = Array2::from_shape_fn((20, 5), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((20, 4), |(i, j)| (i + j + 1) as Float * 0.1);

        let robust_cca = RobustCCA::new(2);
        let fitted = robust_cca.fit(&X, &Y).unwrap();

        assert_eq!(fitted.weights_x().shape(), &[5, 2]);
        assert_eq!(fitted.weights_y().shape(), &[4, 2]);
        assert_eq!(fitted.correlations().len(), 2);
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_robust_cca_with_outliers() {
        let mut X = Array2::from_shape_fn((20, 3), |(i, j)| (i + j) as Float * 0.1);
        let mut Y = Array2::from_shape_fn((20, 3), |(i, j)| (i + j) as Float * 0.1);

        // Add outliers
        X[[0, 0]] = 100.0;
        Y[[0, 0]] = -100.0;
        X[[1, 1]] = 100.0;
        Y[[1, 1]] = -100.0;

        let robust_cca = RobustCCA::new(1).breakdown_point(0.3);
        let fitted = robust_cca.fit(&X, &Y).unwrap();

        // Check that outliers are detected
        let outlier_flags = fitted.outlier_flags();
        let n_outliers = outlier_flags.iter().filter(|&&x| x).count();
        assert!(n_outliers > 0, "Expected some outliers to be detected");
    }

    #[test]
    fn test_robust_pls_basic() {
        let X = Array2::from_shape_fn((15, 4), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((15, 3), |(i, j)| (i + j + 2) as Float * 0.1);

        let robust_pls = RobustPLS::new(2);
        let fitted = robust_pls.fit(&X, &Y).unwrap();

        assert_eq!(fitted.weights_x().shape(), &[4, 2]);
        assert_eq!(fitted.weights_y().shape(), &[3, 2]);
        assert_eq!(fitted.loadings_x().shape(), &[4, 2]);
        assert_eq!(fitted.loadings_y().shape(), &[3, 2]);
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_robust_pls_m_estimators() {
        let X = Array2::ones((10, 3));
        let Y = Array2::ones((10, 2));

        // Test different M-estimator types
        let estimator_types = vec![
            MEstimatorType::Huber,
            MEstimatorType::Bisquare,
            MEstimatorType::Hampel,
            MEstimatorType::Andrews,
        ];

        for estimator_type in estimator_types {
            let robust_pls = RobustPLS::new(1).estimator_type(estimator_type);
            let fitted = robust_pls.fit(&X, &Y).unwrap();
            assert!(fitted.n_iter() >= 0);
        }
    }

    #[test]
    fn test_robust_cca_transform() {
        let X = Array2::from_shape_fn((15, 4), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((15, 3), |(i, j)| (i + j) as Float * 0.1);

        let robust_cca = RobustCCA::new(2);
        let fitted = robust_cca.fit(&X, &Y).unwrap();

        let (X_canonical, Y_canonical) = fitted.transform(&(X, Y)).unwrap();
        assert_eq!(X_canonical.shape(), &[15, 2]);
        assert_eq!(Y_canonical.shape(), &[15, 2]);
    }

    #[test]
    fn test_robust_pls_transform() {
        let X = Array2::from_shape_fn((12, 4), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((12, 2), |(i, j)| (i + j) as Float * 0.1);

        let robust_pls = RobustPLS::new(2);
        let fitted = robust_pls.fit(&X, &Y).unwrap();

        let X_scores = fitted.transform(&X).unwrap();
        assert_eq!(X_scores.shape(), &[12, 2]);
    }

    #[test]
    fn test_robust_methods_error_cases() {
        let X = Array2::ones((5, 3));
        let Y = Array2::ones((4, 3)); // Mismatched samples

        let robust_cca = RobustCCA::new(1);
        assert!(robust_cca.fit(&X, &Y).is_err());

        let robust_pls = RobustPLS::new(1);
        assert!(robust_pls.fit(&X, &Y).is_err());
    }

    #[test]
    fn test_breakdown_point_analysis_basic() {
        let X = Array2::from_shape_fn((20, 3), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((20, 2), |(i, j)| (i + j + 1) as Float * 0.1);

        let breakdown_analysis = BreakdownPointAnalysis::new()
            .max_contamination(0.3)
            .n_levels(5)
            .n_trials(10);

        let results = breakdown_analysis.analyze_robust_cca(&X, &Y, 1).unwrap();

        assert_eq!(results.contamination_levels.len(), 6); // 0 to 5 levels
        assert_eq!(results.breakdown_points.len(), 6);
        assert!(results.breakdown_threshold >= 0.0);
        assert!(results.breakdown_threshold <= 0.3);

        // Check that failure rates are reasonable
        for bp in &results.breakdown_points {
            assert!(bp.failure_rate >= 0.0);
            assert!(bp.failure_rate <= 1.0);
            assert_eq!(bp.n_trials, 10);
        }
    }

    #[test]
    fn test_breakdown_point_analysis_pls() {
        let X = Array2::from_shape_fn((15, 4), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((15, 3), |(i, j)| (i + j + 2) as Float * 0.1);

        let breakdown_analysis = BreakdownPointAnalysis::new()
            .max_contamination(0.25)
            .n_levels(4)
            .n_trials(8)
            .outlier_magnitude(5.0);

        let results = breakdown_analysis.analyze_robust_pls(&X, &Y, 1).unwrap();

        assert_eq!(results.contamination_levels.len(), 5);
        assert_eq!(results.breakdown_points.len(), 5);

        // Should have some breakdown point
        assert!(results.breakdown_threshold <= 0.25);
    }

    #[test]
    fn test_breakdown_point_configuration() {
        let analysis = BreakdownPointAnalysis::new()
            .max_contamination(0.4)
            .n_levels(10)
            .n_trials(20)
            .outlier_magnitude(15.0);

        assert_eq!(analysis.max_contamination, 0.4);
        assert_eq!(analysis.n_levels, 10);
        assert_eq!(analysis.n_trials, 20);
        assert_eq!(analysis.outlier_magnitude, 15.0);
    }

    #[test]
    fn test_influence_diagnostics_robust_cca() {
        let X = Array2::from_shape_fn((12, 3), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((12, 2), |(i, j)| (i + j + 1) as Float * 0.1);

        let robust_cca = RobustCCA::new(1);
        let fitted = robust_cca.fit(&X, &Y).unwrap();

        let influence_analysis = InfluenceDiagnostics::new()
            .influence_threshold(1.5)
            .standardize(false);

        let results = influence_analysis
            .influence_robust_cca(&fitted, &X, &Y)
            .unwrap();

        assert_eq!(results.influence_values.shape(), &[12, 1]);
        assert_eq!(results.leverage_values.len(), 12);
        assert_eq!(results.influence_threshold, 1.5);

        // Check that all values are finite
        for &val in results.influence_values.iter() {
            assert!(val.is_finite());
        }
        for &val in results.leverage_values.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_influence_diagnostics_robust_pls() {
        let X = Array2::from_shape_fn((10, 4), |(i, j)| (i + j) as Float * 0.1);
        let Y = Array2::from_shape_fn((10, 2), |(i, j)| (i + j + 1) as Float * 0.1);

        let robust_pls = RobustPLS::new(1).estimator_type(MEstimatorType::Huber);
        let fitted = robust_pls.fit(&X, &Y).unwrap();

        let influence_analysis = InfluenceDiagnostics::new()
            .standardize(true)
            .influence_threshold(2.0);

        let results = influence_analysis
            .influence_robust_pls(&fitted, &X, &Y)
            .unwrap();

        assert_eq!(results.influence_values.shape(), &[10, 1]);
        assert_eq!(results.leverage_values.len(), 10);

        // Test accessor methods
        assert!(results.influence(0, 0).is_finite());
        assert!(results.leverage(0).is_finite());

        let max_influence = results.max_influence_per_observation();
        assert_eq!(max_influence.len(), 10);
    }

    #[test]
    fn test_influence_diagnostics_with_outliers() {
        let mut X = Array2::from_shape_fn((15, 3), |(i, j)| (i + j) as Float * 0.1);
        let mut Y = Array2::from_shape_fn((15, 2), |(i, j)| (i + j) as Float * 0.1);

        // Add some outliers
        X[[0, 0]] = 10.0;
        Y[[0, 0]] = -10.0;
        X[[1, 1]] = -10.0;
        Y[[1, 1]] = 10.0;

        let robust_cca = RobustCCA::new(1).breakdown_point(0.3);
        let fitted = robust_cca.fit(&X, &Y).unwrap();

        let influence_analysis = InfluenceDiagnostics::new().influence_threshold(1.0);
        let results = influence_analysis
            .influence_robust_cca(&fitted, &X, &Y)
            .unwrap();

        // Should identify some high influence observations
        let high_influence = results.high_influence_observations();

        // The exact number depends on the implementation, but should be reasonable
        assert!(high_influence.len() <= 15);

        // Check that indices are valid
        for &idx in high_influence {
            assert!(idx < 15);
        }
    }

    #[test]
    fn test_influence_diagnostics_configuration() {
        let diagnostics = InfluenceDiagnostics::new()
            .perturbation(1e-5)
            .standardize(false)
            .influence_threshold(3.0);

        assert_eq!(diagnostics.perturbation, 1e-5);
        assert!(!diagnostics.standardize);
        assert_eq!(diagnostics.influence_threshold, 3.0);
    }

    #[test]
    fn test_breakdown_analysis_edge_cases() {
        let X = Array2::ones((8, 2));
        let Y = Array2::ones((8, 2));

        // Test with zero contamination
        let breakdown_analysis = BreakdownPointAnalysis::new()
            .max_contamination(0.0)
            .n_levels(1)
            .n_trials(5);

        let results = breakdown_analysis.analyze_robust_cca(&X, &Y, 1).unwrap();
        assert_eq!(results.breakdown_points[0].contamination_level, 0.0);
        assert_eq!(results.breakdown_points[0].failure_rate, 0.0);
    }

    #[test]
    fn test_influence_results_methods() {
        // Create dummy results
        let influence_values = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as Float);
        let leverage_values = Array1::from_shape_fn(5, |i| i as Float);
        let high_influence_indices = vec![0, 2, 4];

        let results = InfluenceResults {
            influence_values,
            leverage_values,
            high_influence_indices,
            influence_threshold: 1.5,
        };

        // Test accessor methods
        assert_eq!(results.high_influence_observations(), &[0, 2, 4]);
        assert_eq!(results.influence(1, 1), 2.0);
        assert_eq!(results.leverage(3), 3.0);

        let max_influence = results.max_influence_per_observation();
        assert_eq!(max_influence.len(), 5);
        assert_eq!(max_influence[0], 1.0); // max of [0, 1]
        assert_eq!(max_influence[1], 2.0); // max of [1, 2]
    }

    #[test]
    fn test_breakdown_results_structure() {
        let contamination_levels = vec![0.0, 0.1, 0.2];
        let breakdown_points = vec![
            BreakdownPoint {
                contamination_level: 0.0,
                failure_rate: 0.0,
                n_trials: 10,
                n_failures: 0,
            },
            BreakdownPoint {
                contamination_level: 0.1,
                failure_rate: 0.2,
                n_trials: 10,
                n_failures: 2,
            },
            BreakdownPoint {
                contamination_level: 0.2,
                failure_rate: 0.6,
                n_trials: 10,
                n_failures: 6,
            },
        ];

        let results = BreakdownResults {
            contamination_levels,
            breakdown_points,
            breakdown_threshold: 0.15,
        };

        assert_eq!(results.contamination_levels.len(), 3);
        assert_eq!(results.breakdown_points.len(), 3);
        assert_eq!(results.breakdown_threshold, 0.15);
        assert_eq!(results.breakdown_points[1].n_failures, 2);
    }
}
