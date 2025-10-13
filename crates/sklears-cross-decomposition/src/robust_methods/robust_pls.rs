//! Robust Partial Least Squares (M-PLS) implementation
//!
//! This module provides the RobustPLS struct and its implementation using M-estimators
//! to create PLS models that are resistant to outliers in both X and Y spaces.

use super::common::{MEstimatorType, Trained, Untrained};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

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

// =============================================================================
// M-estimator PLS Implementation
// =============================================================================

impl RobustPLS<Untrained> {
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
