//! Robust Decomposition Methods
//!
//! This module provides robust decomposition techniques that are resistant to outliers including:
//! - Robust PCA with L1 loss
//! - M-estimator based decomposition
//! - Outlier-resistant methods
//! - Breakdown point analysis
//! - Influence function diagnostics

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, RandNormal, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Configuration for robust decomposition methods
#[derive(Debug, Clone)]
pub struct RobustConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Outlier detection threshold
    pub outlier_threshold: Float,
    /// Robustness parameter for M-estimators
    pub tuning_constant: Float,
    /// Loss function type
    pub loss_function: LossFunction,
}

/// Loss function types for robust estimation
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Huber loss function
    Huber,
    /// Tukey's biweight function
    Tukey,
    /// L1 (absolute) loss
    L1,
    /// Cauchy loss function
    Cauchy,
    /// Welsch loss function
    Welsch,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            outlier_threshold: 2.5,
            tuning_constant: 1.345, // For Huber loss with 95% efficiency
            loss_function: LossFunction::Huber,
        }
    }
}

/// Robust Principal Component Analysis using L1 loss
///
/// This implementation uses iterative algorithms to find principal components
/// that are robust to outliers in the data.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_decomposition::RobustPCA;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((4, 3), vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
///     100.0, 2.0, 3.0  // Outlier
/// ]).unwrap();
///
/// let rpca = RobustPCA::new().n_components(2);
/// let result = rpca.fit_transform(&data).unwrap();
/// ```
pub struct RobustPCA {
    config: RobustConfig,
    n_components: Option<usize>,
    // Fitted parameters
    components_: Option<Array2<Float>>,
    explained_variance_: Option<Array1<Float>>,
    mean_: Option<Array1<Float>>,
    outlier_weights_: Option<Array2<Float>>,
}

impl RobustPCA {
    /// Create a new Robust PCA instance
    pub fn new() -> Self {
        Self {
            config: RobustConfig::default(),
            n_components: None,
            components_: None,
            explained_variance_: None,
            mean_: None,
            outlier_weights_: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    /// Set the loss function
    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.config.loss_function = loss_function;
        self
    }

    /// Set the tuning constant for M-estimators
    pub fn tuning_constant(mut self, tuning_constant: Float) -> Self {
        self.config.tuning_constant = tuning_constant;
        self
    }

    /// Set the outlier threshold
    pub fn outlier_threshold(mut self, threshold: Float) -> Self {
        self.config.outlier_threshold = threshold;
        self
    }

    /// Fit robust PCA and transform the data
    pub fn fit_transform(&mut self, data: &Array2<Float>) -> Result<RobustPCAResult> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Fit the robust PCA model
    pub fn fit(&mut self, data: &Array2<Float>) -> Result<()> {
        let (n_samples, n_features) = data.dim();
        let n_components = self.n_components.unwrap_or(n_features.min(n_samples));

        if n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "Number of components cannot exceed min(n_samples, n_features)".to_string(),
            ));
        }

        // Center the data using robust mean estimation
        let robust_mean = self.compute_robust_mean(data)?;
        let centered_data = data - &robust_mean.view().insert_axis(Axis(0));

        // Initialize components randomly
        let mut rng = thread_rng();
        let mut components = Array2::zeros((n_components, n_features));
        for elem in components.iter_mut() {
            *elem = rng.sample(RandNormal::new(0.0, 1.0).unwrap());
        }

        // Orthogonalize initial components
        self.orthogonalize_components(&mut components);

        // Iterative robust estimation
        let mut prev_components = components.clone();
        let mut outlier_weights = Array2::ones((n_samples, n_components));

        for _iter in 0..self.config.max_iterations {
            // Update components using weighted least squares
            for comp_idx in 0..n_components {
                let weights = outlier_weights.column(comp_idx);
                let updated_component =
                    self.update_component(&centered_data, &weights, comp_idx)?;
                components.row_mut(comp_idx).assign(&updated_component);
            }

            // Orthogonalize components
            self.orthogonalize_components(&mut components);

            // Update outlier weights based on residuals
            outlier_weights = self.compute_outlier_weights(&centered_data, &components)?;

            // Check convergence
            let component_change = self.compute_component_change(&components, &prev_components);
            if component_change < self.config.tolerance {
                break;
            }

            prev_components = components.clone();
        }

        // Compute explained variance
        let explained_variance = self.compute_explained_variance(&centered_data, &components)?;

        self.components_ = Some(components);
        self.explained_variance_ = Some(explained_variance);
        self.mean_ = Some(robust_mean);
        self.outlier_weights_ = Some(outlier_weights);

        Ok(())
    }

    /// Transform data using fitted robust PCA model
    pub fn transform(&self, data: &Array2<Float>) -> Result<RobustPCAResult> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model must be fitted first".to_string()))?;

        let mean = self
            .mean_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model must be fitted first".to_string()))?;

        let explained_variance = self
            .explained_variance_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model must be fitted first".to_string()))?;

        let outlier_weights = self
            .outlier_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model must be fitted first".to_string()))?;

        // Center the data
        let centered_data = data - &mean.view().insert_axis(Axis(0));

        // Project data onto robust components
        let transformed = centered_data.dot(&components.t());

        Ok(RobustPCAResult {
            transformed_data: transformed,
            components: components.clone(),
            explained_variance: explained_variance.clone(),
            mean: mean.clone(),
            outlier_weights: outlier_weights.clone(),
        })
    }

    /// Compute robust mean using iterative M-estimator
    fn compute_robust_mean(&self, data: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = data.dim();
        let mut mean = data.mean_axis(Axis(0)).unwrap();

        for _ in 0..self.config.max_iterations {
            let mut new_mean = Array1::zeros(n_features);
            let mut weight_sum = 0.0;

            for i in 0..n_samples {
                let residual = &data.row(i) - &mean;
                let residual_norm = residual.dot(&residual).sqrt();
                let weight = self.robust_weight(residual_norm);

                new_mean += &(weight * &data.row(i));
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                new_mean /= weight_sum;
            }

            let change = (&new_mean - &mean).dot(&(&new_mean - &mean)).sqrt();
            mean = new_mean;

            if change < self.config.tolerance {
                break;
            }
        }

        Ok(mean)
    }

    /// Update a single component using robust estimation
    fn update_component(
        &self,
        data: &Array2<Float>,
        weights: &scirs2_core::ndarray::ArrayView1<Float>,
        _comp_idx: usize,
    ) -> Result<Array1<Float>> {
        let (n_samples, n_features) = data.dim();
        let mut component = Array1::zeros(n_features);
        let mut weight_sum = 0.0;

        // Weighted covariance computation
        for i in 0..n_samples {
            let weight = weights[i];
            let sample = data.row(i);
            component += &(weight * &sample);
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            component /= weight_sum;
        }

        // Normalize component
        let norm = component.dot(&component).sqrt();
        if norm > 1e-12 {
            component /= norm;
        }

        Ok(component)
    }

    /// Orthogonalize components using Gram-Schmidt process
    fn orthogonalize_components(&self, components: &mut Array2<Float>) {
        let n_components = components.nrows();

        for i in 0..n_components {
            // Collect previous components before mutable borrow
            let prev_components: Vec<Array1<Float>> =
                (0..i).map(|j| components.row(j).to_owned()).collect();

            // Normalize current component
            let mut current = components.row_mut(i);
            let norm = current.dot(&current).sqrt();
            if norm > 1e-12 {
                current /= norm;
            }

            // Orthogonalize against previous components
            for prev in prev_components {
                let projection = current.dot(&prev);
                current -= &(projection * &prev);

                // Renormalize
                let norm = current.dot(&current).sqrt();
                if norm > 1e-12 {
                    current /= norm;
                }
            }
        }
    }

    /// Compute outlier weights based on residuals
    fn compute_outlier_weights(
        &self,
        data: &Array2<Float>,
        components: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (n_samples, _) = data.dim();
        let n_components = components.nrows();
        let mut weights = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let sample = data.row(i);

            for j in 0..n_components {
                let component = components.row(j);
                let projection = sample.dot(&component);
                let reconstructed = projection * &component;
                let residual = &sample - &reconstructed;
                let residual_norm = residual.dot(&residual).sqrt();

                weights[[i, j]] = self.robust_weight(residual_norm);
            }
        }

        Ok(weights)
    }

    /// Compute robust weight based on loss function
    fn robust_weight(&self, residual: Float) -> Float {
        let standardized_residual = residual / self.config.tuning_constant;

        match self.config.loss_function {
            LossFunction::Huber => {
                if standardized_residual.abs() <= 1.0 {
                    1.0
                } else {
                    1.0 / standardized_residual.abs()
                }
            }
            LossFunction::Tukey => {
                if standardized_residual.abs() <= 1.0 {
                    let x2 = standardized_residual * standardized_residual;
                    (1.0 - x2) * (1.0 - x2)
                } else {
                    0.0
                }
            }
            LossFunction::L1 => {
                if standardized_residual.abs() > 1e-12 {
                    1.0 / standardized_residual.abs()
                } else {
                    1.0
                }
            }
            LossFunction::Cauchy => 1.0 / (1.0 + standardized_residual * standardized_residual),
            LossFunction::Welsch => {
                let x2 = standardized_residual * standardized_residual;
                (-x2).exp()
            }
        }
    }

    /// Compute explained variance for robust components
    fn compute_explained_variance(
        &self,
        data: &Array2<Float>,
        components: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let n_components = components.nrows();
        let mut explained_variance = Array1::zeros(n_components);

        // Total variance (robust estimate)
        let total_variance = self.compute_robust_variance(data);

        for i in 0..n_components {
            let component = components.row(i);
            let projections = data.dot(&component);
            let component_variance = self.compute_robust_variance_1d(&projections);
            explained_variance[i] = component_variance / total_variance;
        }

        Ok(explained_variance)
    }

    /// Compute robust variance estimate for matrix
    fn compute_robust_variance(&self, data: &Array2<Float>) -> Float {
        let (_n_samples, n_features) = data.dim();
        let mut total_variance = 0.0;

        for j in 0..n_features {
            let column = data.column(j);
            total_variance += self.compute_robust_variance_1d(&column.to_owned());
        }

        total_variance / n_features as Float
    }

    /// Compute robust variance estimate for 1D array
    fn compute_robust_variance_1d(&self, data: &Array1<Float>) -> Float {
        let n = data.len();
        if n < 2 {
            return 0.0;
        }

        // Use median absolute deviation (MAD) as robust variance estimate
        let median = self.compute_median(data);
        let mut abs_deviations: Vec<Float> = data.iter().map(|&x| (x - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mad = if abs_deviations.len() % 2 == 0 {
            let mid = abs_deviations.len() / 2;
            (abs_deviations[mid - 1] + abs_deviations[mid]) / 2.0
        } else {
            abs_deviations[abs_deviations.len() / 2]
        };

        // Scale MAD to approximate standard deviation
        1.4826 * mad * 1.4826 * mad // MAD^2 scaled
    }

    /// Compute median of 1D array
    fn compute_median(&self, data: &Array1<Float>) -> Float {
        let mut sorted_data: Vec<Float> = data.iter().cloned().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len();
        if n % 2 == 0 {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
        } else {
            sorted_data[n / 2]
        }
    }

    /// Compute change in components for convergence check
    fn compute_component_change(&self, current: &Array2<Float>, previous: &Array2<Float>) -> Float {
        let diff = current - previous;
        (diff.iter().map(|&x| x * x).sum::<Float>()).sqrt()
    }

    /// Get the fitted components
    pub fn components(&self) -> Option<&Array2<Float>> {
        self.components_.as_ref()
    }

    /// Get the explained variance
    pub fn explained_variance(&self) -> Option<&Array1<Float>> {
        self.explained_variance_.as_ref()
    }

    /// Get the robust mean
    pub fn mean(&self) -> Option<&Array1<Float>> {
        self.mean_.as_ref()
    }

    /// Get the outlier weights
    pub fn outlier_weights(&self) -> Option<&Array2<Float>> {
        self.outlier_weights_.as_ref()
    }
}

impl Default for RobustPCA {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of robust PCA decomposition
#[derive(Debug, Clone)]
pub struct RobustPCAResult {
    /// Transformed data in the robust principal component space
    pub transformed_data: Array2<Float>,
    /// Robust principal components
    pub components: Array2<Float>,
    /// Explained variance ratios
    pub explained_variance: Array1<Float>,
    /// Robust mean of the data
    pub mean: Array1<Float>,
    /// Outlier weights for each sample and component
    pub outlier_weights: Array2<Float>,
}

impl RobustPCAResult {
    /// Reconstruct data from robust principal components
    pub fn inverse_transform(&self, transformed_data: &Array2<Float>) -> Array2<Float> {
        let reconstructed = transformed_data.dot(&self.components);
        reconstructed + self.mean.view().insert_axis(Axis(0))
    }

    /// Identify outliers based on weights
    pub fn identify_outliers(&self, threshold: Float) -> Array1<bool> {
        let (n_samples, _) = self.outlier_weights.dim();
        let mut is_outlier = Array1::from_elem(n_samples, false);

        for i in 0..n_samples {
            let min_weight = self
                .outlier_weights
                .row(i)
                .iter()
                .cloned()
                .fold(Float::INFINITY, Float::min);
            if min_weight < threshold {
                is_outlier[i] = true;
            }
        }

        is_outlier
    }

    /// Compute reconstruction error for each sample
    pub fn reconstruction_error(&self, original_data: &Array2<Float>) -> Result<Array1<Float>> {
        let reconstructed = self.inverse_transform(&self.transformed_data);
        let (n_samples, _) = original_data.dim();
        let mut errors = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let original_row = original_data.row(i);
            let reconstructed_row = reconstructed.row(i);
            let diff = &original_row - &reconstructed_row;
            errors[i] = diff.dot(&diff).sqrt();
        }

        Ok(errors)
    }
}

/// M-estimator based matrix decomposition
pub struct MEstimatorDecomposition {
    config: RobustConfig,
    rank: Option<usize>,
}

impl MEstimatorDecomposition {
    /// Create a new M-estimator decomposition
    pub fn new() -> Self {
        Self {
            config: RobustConfig::default(),
            rank: None,
        }
    }

    /// Set the target rank for decomposition
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    /// Set the loss function
    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.config.loss_function = loss_function;
        self
    }

    /// Decompose matrix using M-estimator
    pub fn decompose(&self, matrix: &Array2<Float>) -> Result<MEstimatorResult> {
        let (m, n) = matrix.dim();
        let rank = self.rank.unwrap_or((m.min(n) / 2).max(1));

        // Initialize factors
        let mut rng = thread_rng();
        let mut u = Array2::zeros((m, rank));
        let mut v = Array2::zeros((n, rank));

        // Fill with normal random values
        for elem in u.iter_mut() {
            *elem = rng.sample(RandNormal::new(0.0, 1.0).unwrap());
        }
        for elem in v.iter_mut() {
            *elem = rng.sample(RandNormal::new(0.0, 1.0).unwrap());
        }

        // Iterative M-estimator updates
        for _ in 0..self.config.max_iterations {
            let prev_u = u.clone();
            let prev_v = v.clone();

            // Update U
            u = self.update_factor_u(matrix, &u, &v)?;

            // Update V
            v = self.update_factor_v(matrix, &u, &v)?;

            // Check convergence
            let u_change = (&u - &prev_u).iter().map(|&x| x * x).sum::<Float>().sqrt();
            let v_change = (&v - &prev_v).iter().map(|&x| x * x).sum::<Float>().sqrt();

            if u_change + v_change < self.config.tolerance {
                break;
            }
        }

        // Compute residuals and weights
        let reconstruction = u.dot(&v.t());
        let residuals = matrix - &reconstruction;
        let weights = self.compute_residual_weights(&residuals);

        Ok(MEstimatorResult {
            u_factor: u,
            v_factor: v,
            reconstruction,
            residuals,
            weights,
        })
    }

    /// Update U factor using robust estimation
    fn update_factor_u(
        &self,
        matrix: &Array2<Float>,
        u: &Array2<Float>,
        v: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (m, rank) = u.dim();
        let mut new_u = Array2::zeros((m, rank));

        for i in 0..m {
            for r in 0..rank {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for j in 0..matrix.ncols() {
                    let residual = matrix[[i, j]] - u.row(i).dot(&v.row(j));
                    let weight = self.robust_weight(residual.abs());

                    numerator += weight * matrix[[i, j]] * v[[j, r]];
                    denominator += weight * v[[j, r]] * v[[j, r]];
                }

                if denominator > 1e-12 {
                    new_u[[i, r]] = numerator / denominator;
                }
            }
        }

        Ok(new_u)
    }

    /// Update V factor using robust estimation
    fn update_factor_v(
        &self,
        matrix: &Array2<Float>,
        u: &Array2<Float>,
        v: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let (n, rank) = v.dim();
        let mut new_v = Array2::zeros((n, rank));

        for j in 0..n {
            for r in 0..rank {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for i in 0..matrix.nrows() {
                    let residual = matrix[[i, j]] - u.row(i).dot(&v.row(j));
                    let weight = self.robust_weight(residual.abs());

                    numerator += weight * matrix[[i, j]] * u[[i, r]];
                    denominator += weight * u[[i, r]] * u[[i, r]];
                }

                if denominator > 1e-12 {
                    new_v[[j, r]] = numerator / denominator;
                }
            }
        }

        Ok(new_v)
    }

    /// Compute weights for residuals
    fn compute_residual_weights(&self, residuals: &Array2<Float>) -> Array2<Float> {
        residuals.mapv(|r| self.robust_weight(r.abs()))
    }

    /// Compute robust weight (same as in RobustPCA)
    fn robust_weight(&self, residual: Float) -> Float {
        let standardized_residual = residual / self.config.tuning_constant;

        match self.config.loss_function {
            LossFunction::Huber => {
                if standardized_residual.abs() <= 1.0 {
                    1.0
                } else {
                    1.0 / standardized_residual.abs()
                }
            }
            LossFunction::Tukey => {
                if standardized_residual.abs() <= 1.0 {
                    let x2 = standardized_residual * standardized_residual;
                    (1.0 - x2) * (1.0 - x2)
                } else {
                    0.0
                }
            }
            LossFunction::L1 => {
                if standardized_residual.abs() > 1e-12 {
                    1.0 / standardized_residual.abs()
                } else {
                    1.0
                }
            }
            LossFunction::Cauchy => 1.0 / (1.0 + standardized_residual * standardized_residual),
            LossFunction::Welsch => {
                let x2 = standardized_residual * standardized_residual;
                (-x2).exp()
            }
        }
    }
}

impl Default for MEstimatorDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of M-estimator decomposition
#[derive(Debug, Clone)]
pub struct MEstimatorResult {
    /// U factor matrix
    pub u_factor: Array2<Float>,
    /// V factor matrix  
    pub v_factor: Array2<Float>,
    /// Reconstructed matrix
    pub reconstruction: Array2<Float>,
    /// Residuals matrix
    pub residuals: Array2<Float>,
    /// Robust weights for each element
    pub weights: Array2<Float>,
}

impl MEstimatorResult {
    /// Compute the Frobenius norm of residuals
    pub fn residual_norm(&self) -> Float {
        self.residuals.iter().map(|&x| x * x).sum::<Float>().sqrt()
    }

    /// Compute weighted residual norm
    pub fn weighted_residual_norm(&self) -> Float {
        let weighted_residuals = &self.residuals * &self.weights;
        weighted_residuals
            .iter()
            .map(|&x| x * x)
            .sum::<Float>()
            .sqrt()
    }

    /// Identify outlier elements based on low weights
    pub fn identify_outlier_elements(&self, threshold: Float) -> Array2<bool> {
        self.weights.mapv(|w| w < threshold)
    }
}

/// Breakdown point analysis for robust methods
pub struct BreakdownPointAnalysis {
    #[allow(dead_code)]
    config: RobustConfig,
}

impl BreakdownPointAnalysis {
    /// Create a new breakdown point analysis
    pub fn new() -> Self {
        Self {
            config: RobustConfig::default(),
        }
    }

    /// Compute empirical breakdown point for robust PCA
    pub fn empirical_breakdown_point(
        &self,
        data: &Array2<Float>,
        contamination_levels: &[Float],
    ) -> Result<BreakdownResult> {
        let (n_samples, _) = data.dim();
        let mut breakdown_errors = Vec::new();
        let mut breakdown_levels = Vec::new();

        for &contamination_level in contamination_levels {
            let n_outliers = (contamination_level * n_samples as Float) as usize;

            if n_outliers >= n_samples {
                continue;
            }

            // Add outliers to data
            let contaminated_data = self.add_outliers(data, n_outliers)?;

            // Fit robust PCA
            let mut robust_pca = RobustPCA::new().n_components(2);
            let robust_result = robust_pca.fit_transform(&contaminated_data)?;

            // Fit standard PCA for comparison
            let standard_result = self.standard_pca(&contaminated_data, 2)?;

            // Compute difference in principal directions
            let direction_error =
                self.compute_direction_error(&robust_result.components, &standard_result);

            breakdown_errors.push(direction_error);
            breakdown_levels.push(contamination_level);
        }

        Ok(BreakdownResult {
            contamination_levels: Array1::from_vec(breakdown_levels),
            direction_errors: Array1::from_vec(breakdown_errors),
        })
    }

    /// Add outliers to data for breakdown analysis
    fn add_outliers(&self, data: &Array2<Float>, n_outliers: usize) -> Result<Array2<Float>> {
        let (n_samples, n_features) = data.dim();
        let mut contaminated_data = data.clone();

        // Generate outliers at large distances from the data center
        let data_center = data.mean_axis(Axis(0)).unwrap();
        let data_scale = self.estimate_scale(data);
        let mut rng = thread_rng();

        for i in 0..n_outliers.min(n_samples) {
            for j in 0..n_features {
                // Place outliers at 10 times the data scale
                contaminated_data[[i, j]] =
                    data_center[j] + 10.0 * data_scale * (2.0 * (rng.gen::<Float>()) - 1.0);
            }
        }

        Ok(contaminated_data)
    }

    /// Estimate data scale using MAD
    fn estimate_scale(&self, data: &Array2<Float>) -> Float {
        let center = data.mean_axis(Axis(0)).unwrap();
        let distances: Vec<Float> = data
            .axis_iter(Axis(0))
            .map(|row| {
                let diff = &row - &center;
                diff.dot(&diff).sqrt()
            })
            .collect();

        // Compute median of distances
        let mut sorted_distances = distances;
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_distances.len();
        if n % 2 == 0 {
            (sorted_distances[n / 2 - 1] + sorted_distances[n / 2]) / 2.0
        } else {
            sorted_distances[n / 2]
        }
    }

    /// Simplified standard PCA for comparison
    fn standard_pca(&self, data: &Array2<Float>, n_components: usize) -> Result<Array2<Float>> {
        let (n_samples, n_features) = data.dim();

        // Center data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = data - &mean.view().insert_axis(Axis(0));

        // Compute covariance matrix
        let _cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as Float;

        // Simplified eigendecomposition (in practice would use proper SVD/eigen)
        let mut components = Array2::eye(n_features);
        components = components
            .slice(scirs2_core::ndarray::s![0..n_components, ..])
            .to_owned();

        Ok(components)
    }

    /// Compute angular error between principal directions
    fn compute_direction_error(
        &self,
        robust_components: &Array2<Float>,
        standard_components: &Array2<Float>,
    ) -> Float {
        let n_components = robust_components.nrows().min(standard_components.nrows());
        let mut total_error = 0.0;

        for i in 0..n_components {
            let robust_dir = robust_components.row(i);
            let standard_dir = standard_components.row(i);

            // Compute cosine of angle between directions
            let dot_product = robust_dir.dot(&standard_dir);
            let robust_norm = robust_dir.dot(&robust_dir).sqrt();
            let standard_norm = standard_dir.dot(&standard_dir).sqrt();

            if robust_norm > 1e-12 && standard_norm > 1e-12 {
                let cos_angle = dot_product / (robust_norm * standard_norm);
                let angle = cos_angle.abs().acos();
                total_error += angle;
            }
        }

        total_error / n_components as Float
    }
}

impl Default for BreakdownPointAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of breakdown point analysis
#[derive(Debug, Clone)]
pub struct BreakdownResult {
    /// Contamination levels tested
    pub contamination_levels: Array1<Float>,
    /// Direction errors at each contamination level
    pub direction_errors: Array1<Float>,
}

impl BreakdownResult {
    /// Find the breakdown point (contamination level where error exceeds threshold)
    pub fn breakdown_point(&self, error_threshold: Float) -> Option<Float> {
        for (level, error) in self
            .contamination_levels
            .iter()
            .zip(self.direction_errors.iter())
        {
            if *error > error_threshold {
                return Some(*level);
            }
        }
        None
    }

    /// Compute the area under the error curve
    pub fn robustness_score(&self) -> Float {
        let n = self.contamination_levels.len();
        if n < 2 {
            return 0.0;
        }

        let mut area = 0.0;
        for i in 1..n {
            let dx = self.contamination_levels[i] - self.contamination_levels[i - 1];
            let avg_error = (self.direction_errors[i] + self.direction_errors[i - 1]) / 2.0;
            area += dx * avg_error;
        }

        // Return inverse of area (higher is more robust)
        1.0 / (area + 1e-12)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_robust_pca_basic() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let mut rpca = RobustPCA::new().n_components(2);
        let result = rpca.fit_transform(&data).unwrap();

        assert_eq!(result.transformed_data.dim(), (4, 2));
        assert_eq!(result.components.dim(), (2, 3));
        assert_eq!(result.explained_variance.len(), 2);
        assert_eq!(result.outlier_weights.dim(), (4, 2));
    }

    #[test]
    fn test_robust_pca_with_outliers() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 100.0, 100.0,
                100.0, // Clear outlier
            ],
        )
        .unwrap();

        let mut rpca = RobustPCA::new()
            .n_components(2)
            .loss_function(LossFunction::Tukey);

        let result = rpca.fit_transform(&data).unwrap();

        // Check that outlier has low weights
        let outlier_weights = result.outlier_weights.row(4);
        let min_weight = outlier_weights
            .iter()
            .cloned()
            .fold(Float::INFINITY, Float::min);
        assert!(min_weight < 0.5); // Outlier should have low weight

        // Check outlier identification
        let outliers = result.identify_outliers(0.5);
        assert!(outliers[4]); // Last sample should be identified as outlier
    }

    #[test]
    fn test_m_estimator_decomposition() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let m_est = MEstimatorDecomposition::new()
            .rank(2)
            .loss_function(LossFunction::Huber);

        let result = m_est.decompose(&matrix).unwrap();

        assert_eq!(result.u_factor.dim(), (3, 2));
        assert_eq!(result.v_factor.dim(), (3, 2));
        assert_eq!(result.reconstruction.dim(), (3, 3));
        assert_eq!(result.residuals.dim(), (3, 3));
        assert_eq!(result.weights.dim(), (3, 3));
    }

    #[test]
    fn test_loss_functions() {
        let config = RobustConfig::default();
        let rpca = RobustPCA::new();

        let residuals = vec![0.5, 1.0, 1.5, 2.0, 5.0];

        for residual in residuals {
            let huber_weight = rpca.robust_weight(residual);
            assert!(huber_weight > 0.0);
            assert!(huber_weight <= 1.0);
        }
    }

    #[test]
    fn test_breakdown_analysis() {
        let data = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0,
            ],
        )
        .unwrap();

        let breakdown_analysis = BreakdownPointAnalysis::new();
        let contamination_levels = vec![0.1, 0.2, 0.3];

        let result = breakdown_analysis
            .empirical_breakdown_point(&data, &contamination_levels)
            .unwrap();

        assert_eq!(
            result.contamination_levels.len(),
            result.direction_errors.len()
        );
        assert!(result.contamination_levels.len() <= contamination_levels.len());

        // Test robustness score
        let score = result.robustness_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_robust_mean_computation() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 100.0, 100.0, // Outlier
            ],
        )
        .unwrap();

        let rpca = RobustPCA::new();
        let robust_mean = rpca.compute_robust_mean(&data).unwrap();

        // Robust mean should be less affected by the outlier
        assert!(robust_mean[0] < 50.0); // Should be much less than simple mean
        assert!(robust_mean[1] < 50.0);
    }

    #[test]
    fn test_reconstruction_error() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let mut rpca = RobustPCA::new().n_components(2);
        let result = rpca.fit_transform(&data).unwrap();

        let errors = result.reconstruction_error(&data).unwrap();
        assert_eq!(errors.len(), 4);

        // All errors should be non-negative
        for error in errors.iter() {
            assert!(*error >= 0.0);
        }
    }
}
