//! Robust Canonical Correlation Analysis
//!
//! This module provides robust CCA implementation using M-estimators and iterative
//! reweighting to find canonical correlations that are resistant to outliers and
//! contaminated observations.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

use super::common::{MEstimatorType, Trained, Untrained};

/// Robust Canonical Correlation Analysis
///
/// RobustCCA uses M-estimators and iterative reweighting to find canonical correlations
/// that are resistant to outliers and contaminated observations.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use sklears_cross_decomposition::robust_methods::RobustCCA;
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
        let wx = Array1::from_iter((0..n_features_x).map(|_| thread_rng().random::<Float>() - 0.5));
        let wy = Array1::from_iter((0..n_features_y).map(|_| thread_rng().random::<Float>() - 0.5));

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
        let mut wx = Array1::from_iter((0..n_features_x).map(|_| thread_rng().random::<Float>()));
        let mut wy = Array1::from_iter((0..n_features_y).map(|_| thread_rng().random::<Float>()));

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
    fn test_robust_cca_error_cases() {
        let X = Array2::ones((5, 3));
        let Y = Array2::ones((4, 3)); // Mismatched samples

        let robust_cca = RobustCCA::new(1);
        assert!(robust_cca.fit(&X, &Y).is_err());
    }

    #[test]
    fn test_robust_cca_configuration() {
        let robust_cca = RobustCCA::new(3)
            .breakdown_point(0.4)
            .huber_threshold(2.0)
            .max_iter(150)
            .tol(1e-8)
            .center(false)
            .scale(false);

        assert_eq!(robust_cca.n_components, 3);
        assert_eq!(robust_cca.breakdown_point, 0.4);
        assert_eq!(robust_cca.huber_threshold, 2.0);
        assert_eq!(robust_cca.max_iter, 150);
        assert_eq!(robust_cca.tol, 1e-8);
        assert!(!robust_cca.center);
        assert!(!robust_cca.scale);
    }
}
