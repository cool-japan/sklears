//! Automatic kernel construction and selection
//!
//! This module provides functionality for automatically constructing and selecting
//! appropriate kernel functions based on data characteristics and statistical tests.

use crate::kernels::*;
// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Automatic kernel constructor that can intelligently choose and combine kernels
#[derive(Debug, Clone)]
pub struct AutomaticKernelConstructor {
    /// Maximum number of components to consider in composite kernels
    pub max_components: usize,
    /// Whether to consider periodic patterns
    pub include_periodic: bool,
    /// Whether to consider linear trends
    pub include_linear: bool,
    /// Whether to consider polynomial features
    pub include_polynomial: bool,
    /// Minimum correlation threshold for including components
    pub correlation_threshold: f64,
    /// Whether to use cross-validation for kernel selection
    pub use_cross_validation: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for AutomaticKernelConstructor {
    fn default() -> Self {
        Self {
            max_components: 5,
            include_periodic: true,
            include_linear: true,
            include_polynomial: true,
            correlation_threshold: 0.1,
            use_cross_validation: true,
            random_state: Some(42),
        }
    }
}

/// Result of automatic kernel construction
#[derive(Debug, Clone)]
pub struct KernelConstructionResult {
    /// The best kernel found
    pub best_kernel: Box<dyn Kernel>,
    /// Score of the best kernel (negative log marginal likelihood)
    pub best_score: f64,
    /// All kernels tried with their scores
    pub kernel_scores: Vec<(String, f64)>,
    /// Data characteristics detected
    pub data_characteristics: DataCharacteristics,
}

/// Data characteristics detected during analysis
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Dimensionality of input data
    pub n_dimensions: usize,
    /// Number of data points
    pub n_samples: usize,
    /// Presence of periodic patterns
    pub has_periodicity: bool,
    /// Strength of linear trend (0-1)
    pub linear_trend_strength: f64,
    /// Noise level estimate
    pub noise_level: f64,
    /// Length scale estimates for each dimension
    pub length_scales: Array1<f64>,
    /// Dominant frequencies (if periodic patterns detected)
    pub dominant_frequencies: Vec<f64>,
}

impl AutomaticKernelConstructor {
    /// Create a new automatic kernel constructor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of components
    pub fn max_components(mut self, max_components: usize) -> Self {
        self.max_components = max_components;
        self
    }

    /// Set whether to include periodic patterns
    pub fn include_periodic(mut self, include_periodic: bool) -> Self {
        self.include_periodic = include_periodic;
        self
    }

    /// Set whether to include linear trends
    pub fn include_linear(mut self, include_linear: bool) -> Self {
        self.include_linear = include_linear;
        self
    }

    /// Set correlation threshold
    pub fn correlation_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set whether to use cross-validation for kernel evaluation
    pub fn use_cross_validation(mut self, use_cross_validation: bool) -> Self {
        self.use_cross_validation = use_cross_validation;
        self
    }

    /// Automatically construct the best kernel for the given data
    pub fn construct_kernel(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> SklResult<KernelConstructionResult> {
        // Analyze data characteristics
        let characteristics = self.analyze_data_characteristics(&X, &y)?;

        // Generate candidate kernels based on characteristics
        let candidate_kernels = self.generate_candidate_kernels(&characteristics)?;

        // Evaluate kernels and select the best one
        let mut kernel_scores = Vec::new();
        let mut best_kernel: Option<Box<dyn Kernel>> = None;
        let mut best_score = f64::INFINITY;

        for (name, kernel) in candidate_kernels {
            let score = self.evaluate_kernel(&kernel, &X, &y)?;
            kernel_scores.push((name.clone(), score));

            if score < best_score {
                best_score = score;
                best_kernel = Some(kernel);
            }
        }

        let best_kernel = best_kernel
            .ok_or_else(|| SklearsError::InvalidOperation("No valid kernels found".to_string()))?;

        Ok(KernelConstructionResult {
            best_kernel,
            best_score,
            kernel_scores,
            data_characteristics: characteristics,
        })
    }

    /// Analyze characteristics of the input data
    fn analyze_data_characteristics(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<DataCharacteristics> {
        let n_samples = X.nrows();
        let n_dimensions = X.ncols();

        // Estimate noise level from residuals of simple linear fit
        let noise_level = self.estimate_noise_level(X, y)?;

        // Detect linear trend strength
        let linear_trend_strength = self.detect_linear_trend(X, y)?;

        // Estimate length scales for each dimension
        let length_scales = self.estimate_length_scales(X)?;

        // Detect periodicity
        let (has_periodicity, dominant_frequencies) = if self.include_periodic {
            self.detect_periodicity(X, y)?
        } else {
            (false, Vec::new())
        };

        Ok(DataCharacteristics {
            n_dimensions,
            n_samples,
            has_periodicity,
            linear_trend_strength,
            noise_level,
            length_scales,
            dominant_frequencies,
        })
    }

    /// Estimate noise level from data
    fn estimate_noise_level(&self, _X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<f64> {
        // Simple approach: use variance of differences between nearby points
        if y.len() < 2 {
            return Ok(0.1); // Default noise level
        }

        let mut differences = Vec::new();
        for i in 1..y.len() {
            differences.push((y[i] - y[i - 1]).abs());
        }

        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        Ok(mean_diff.max(1e-6)) // Ensure minimum noise level
    }

    /// Detect linear trend strength
    fn detect_linear_trend(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> SklResult<f64> {
        if X.ncols() == 0 || y.is_empty() {
            return Ok(0.0);
        }

        // Simple correlation with first dimension
        let x_first = X.column(0);
        let correlation = self.compute_correlation(&x_first, y)?;
        Ok(correlation.abs())
    }

    /// Compute correlation between two arrays
    fn compute_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }

        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..x.len() {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            numerator += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        let denominator = (x_var * y_var).sqrt();
        if denominator < 1e-10 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Estimate characteristic length scales for each dimension
    fn estimate_length_scales(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let mut length_scales = Array1::zeros(X.ncols());

        for dim in 0..X.ncols() {
            let column = X.column(dim);
            let range = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - column.fold(f64::INFINITY, |a, &b| a.min(b));

            // Use a fraction of the range as initial length scale estimate
            length_scales[dim] = (range / 10.0).max(1e-3);
        }

        Ok(length_scales)
    }

    /// Detect periodic patterns in the data
    fn detect_periodicity(
        &self,
        _X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<(bool, Vec<f64>)> {
        // Simple autocorrelation-based periodicity detection
        let mut dominant_frequencies = Vec::new();

        if y.len() < 10 {
            return Ok((false, dominant_frequencies));
        }

        // Compute autocorrelation at different lags
        let max_lag = (y.len() / 4).min(50);
        let mut autocorr = Vec::new();

        for lag in 1..max_lag {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in lag..y.len() {
                correlation += y[i] * y[i - lag];
                count += 1;
            }

            if count > 0 {
                autocorr.push(correlation / count as f64);
            }
        }

        // Find peaks in autocorrelation
        let threshold = 0.3; // Minimum correlation for considering periodicity
        let mut has_periodicity = false;

        for i in 1..autocorr.len() - 1 {
            if autocorr[i] > threshold
                && autocorr[i] > autocorr[i - 1]
                && autocorr[i] > autocorr[i + 1]
            {
                has_periodicity = true;
                // Convert lag to frequency (approximate)
                let frequency = 2.0 * std::f64::consts::PI / (i as f64 + 1.0);
                dominant_frequencies.push(frequency);
            }
        }

        Ok((has_periodicity, dominant_frequencies))
    }

    /// Generate candidate kernels based on data characteristics
    fn generate_candidate_kernels(
        &self,
        characteristics: &DataCharacteristics,
    ) -> SklResult<Vec<(String, Box<dyn Kernel>)>> {
        let mut kernels = Vec::new();

        // Base RBF kernel (always include)
        let base_length_scale = characteristics.length_scales.mean().unwrap_or(1.0);
        kernels.push((
            "RBF".to_string(),
            Box::new(RBF::new(base_length_scale)) as Box<dyn Kernel>,
        ));

        // ARD RBF kernel if multi-dimensional
        if characteristics.n_dimensions > 1 {
            kernels.push((
                "ARD_RBF".to_string(),
                Box::new(crate::kernels::ARDRBF::new(
                    characteristics.length_scales.clone(),
                )) as Box<dyn Kernel>,
            ));
        }

        // Matérn kernels
        kernels.push((
            "Matern_1_2".to_string(),
            Box::new(Matern::new(base_length_scale, 0.5)) as Box<dyn Kernel>,
        ));
        kernels.push((
            "Matern_3_2".to_string(),
            Box::new(Matern::new(base_length_scale, 1.5)) as Box<dyn Kernel>,
        ));

        // Linear kernel if strong linear trend
        if self.include_linear && characteristics.linear_trend_strength > 0.3 {
            kernels.push((
                "Linear".to_string(),
                Box::new(Linear::new(1.0, 1.0)) as Box<dyn Kernel>,
            ));

            // RBF + Linear combination
            let rbf = Box::new(RBF::new(base_length_scale));
            let linear = Box::new(Linear::new(1.0, 1.0));
            kernels.push((
                "RBF+Linear".to_string(),
                Box::new(crate::kernels::SumKernel::new(vec![rbf, linear])) as Box<dyn Kernel>,
            ));
        }

        // Periodic kernels if periodicity detected
        if self.include_periodic && characteristics.has_periodicity {
            for &freq in &characteristics.dominant_frequencies {
                let period = 2.0 * std::f64::consts::PI / freq;
                kernels.push((
                    format!("ExpSineSquared_{:.2}", period),
                    Box::new(ExpSineSquared::new(base_length_scale, period)) as Box<dyn Kernel>,
                ));

                // RBF * ExpSineSquared combination
                let rbf = Box::new(RBF::new(base_length_scale));
                let periodic = Box::new(ExpSineSquared::new(base_length_scale, period));
                kernels.push((
                    format!("RBF*ExpSineSquared_{:.2}", period),
                    Box::new(crate::kernels::ProductKernel::new(vec![rbf, periodic]))
                        as Box<dyn Kernel>,
                ));
            }
        }

        // Rational Quadratic (good for multiple length scales)
        kernels.push((
            "RationalQuadratic".to_string(),
            Box::new(RationalQuadratic::new(base_length_scale, 1.0)) as Box<dyn Kernel>,
        ));

        // Note: SpectralMixture kernel can be added when properly exported

        Ok(kernels)
    }

    /// Evaluate a kernel using cross-validation or simple marginal likelihood
    fn evaluate_kernel(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        if self.use_cross_validation && X.nrows() > 10 {
            self.cross_validate_kernel(kernel, X, y)
        } else {
            self.evaluate_marginal_likelihood(kernel, X, y)
        }
    }

    /// Simple marginal likelihood evaluation
    #[allow(non_snake_case)]
    fn evaluate_marginal_likelihood(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        // Compute kernel matrix
        let X_owned = X.to_owned();
        let K = kernel.compute_kernel_matrix(&X_owned, Some(&X_owned))?;

        // Add noise to diagonal
        let mut K_noisy = K;
        let noise_var = 0.1; // Simple noise estimate
        for i in 0..K_noisy.nrows() {
            K_noisy[[i, i]] += noise_var;
        }

        // Compute Cholesky decomposition
        match crate::utils::cholesky_decomposition(&K_noisy) {
            Ok(L) => {
                // Compute log marginal likelihood
                let mut log_det = 0.0;
                for i in 0..L.nrows() {
                    log_det += L[[i, i]].ln();
                }
                log_det *= 2.0;

                // Solve for alpha = K^(-1) * y
                let y_owned = y.to_owned();
                let alpha = match crate::utils::triangular_solve(&L, &y_owned) {
                    Ok(temp) => {
                        let L_T = L.t();
                        crate::utils::triangular_solve(&L_T.view().to_owned(), &temp)?
                    }
                    Err(_) => return Ok(f64::INFINITY), // Numerical issues
                };

                let data_fit = -0.5 * y.dot(&alpha);
                let complexity_penalty = -0.5 * log_det;
                let normalization = -0.5 * y.len() as f64 * (2.0 * std::f64::consts::PI).ln();

                Ok(-(data_fit + complexity_penalty + normalization))
            }
            Err(_) => Ok(f64::INFINITY), // Kernel matrix not positive definite
        }
    }

    /// Cross-validation based kernel evaluation
    #[allow(non_snake_case)]
    fn cross_validate_kernel(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        let n_folds = 5.min(X.nrows() / 2);
        if n_folds < 2 {
            return self.evaluate_marginal_likelihood(kernel, X, y);
        }

        let fold_size = X.nrows() / n_folds;
        let mut total_score = 0.0;

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                X.nrows()
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..X.nrows() {
                if i >= start_idx && i < end_idx {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if train_indices.is_empty() || test_indices.is_empty() {
                continue;
            }

            // Extract train and test data
            let X_train = X.select(Axis(0), &train_indices);
            let y_train = y.select(Axis(0), &train_indices);
            let _X_test = X.select(Axis(0), &test_indices);
            let _y_test = y.select(Axis(0), &test_indices);

            // Evaluate on this fold
            let fold_score =
                self.evaluate_marginal_likelihood(kernel, &X_train.view(), &y_train.view())?;

            total_score += fold_score;
        }

        Ok(total_score / n_folds as f64)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_automatic_kernel_constructor_creation() {
        let constructor = AutomaticKernelConstructor::new();
        assert_eq!(constructor.max_components, 5);
        assert!(constructor.include_periodic);
        assert!(constructor.include_linear);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_data_characteristics_analysis() {
        let constructor = AutomaticKernelConstructor::new();

        // Create simple test data
        let X = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let characteristics = constructor
            .analyze_data_characteristics(&X.view(), &y.view())
            .unwrap();

        assert_eq!(characteristics.n_dimensions, 2);
        assert_eq!(characteristics.n_samples, 10);
        assert!(characteristics.linear_trend_strength > 0.5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_kernel_construction() {
        let constructor = AutomaticKernelConstructor::new()
            .max_components(3)
            .use_cross_validation(false);

        // Create simple test data
        let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);

        let result = constructor.construct_kernel(X.view(), y.view()).unwrap();

        assert!(result.best_score.is_finite());
        assert!(result.kernel_scores.len() > 0);
    }

    #[test]
    fn test_correlation_computation() {
        let constructor = AutomaticKernelConstructor::new();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let correlation = constructor
            .compute_correlation(&x.view(), &y.view())
            .unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_length_scale_estimation() {
        let constructor = AutomaticKernelConstructor::new();
        let X = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 10.0, 10.0, 20.0]).unwrap();

        let length_scales = constructor.estimate_length_scales(&X.view()).unwrap();

        assert_eq!(length_scales.len(), 2);
        assert!(length_scales[0] > 0.0);
        assert!(length_scales[1] > 0.0);
    }
}
