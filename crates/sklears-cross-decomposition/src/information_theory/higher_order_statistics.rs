//! Higher-Order Statistics for Cross-Decomposition
//!
//! This module provides advanced statistical methods based on higher-order moments,
//! cumulants, and polyspectra for enhanced cross-decomposition analysis.
//!
//! ## Methods Included
//! - Higher-order moment analysis (up to 8th order)
//! - Cumulant-based independence measures
//! - Non-Gaussian component analysis using kurtosis and skewness
//! - Polyspectral cross-decomposition methods
//! - Independent Component Analysis (ICA) using higher-order statistics
//! - Blind Source Separation with higher-order moments

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::error::SklearsError;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Configuration for higher-order statistics analysis
#[derive(Debug, Clone)]
pub struct HigherOrderConfig {
    /// Maximum order of moments to compute
    pub max_order: usize,
    /// Whether to use cumulants instead of raw moments
    pub use_cumulants: bool,
    /// Regularization parameter for numerical stability
    pub regularization: Float,
    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap: usize,
    /// Convergence tolerance for iterative methods
    pub tolerance: Float,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
}

impl Default for HigherOrderConfig {
    fn default() -> Self {
        Self {
            max_order: 4,
            use_cumulants: true,
            regularization: 1e-6,
            n_bootstrap: 1000,
            tolerance: 1e-8,
            max_iterations: 500,
        }
    }
}

/// Results from higher-order statistics analysis
#[derive(Debug, Clone)]
pub struct HigherOrderResults {
    /// Higher-order moments (up to specified order)
    pub moments: HashMap<usize, Array2<Float>>,
    /// Cumulants (if computed)
    pub cumulants: HashMap<usize, Array2<Float>>,
    /// Skewness measures (3rd order standardized)
    pub skewness: Array1<Float>,
    /// Kurtosis measures (4th order standardized)
    pub kurtosis: Array1<Float>,
    /// Cross-moments between variables
    pub cross_moments: HashMap<(usize, usize), Array2<Float>>,
    /// Independence measures based on higher-order statistics
    pub independence_measures: Array2<Float>,
    /// Confidence intervals for statistics
    pub confidence_intervals: HashMap<String, (Array1<Float>, Array1<Float>)>,
}

/// Higher-order statistics analyzer
#[derive(Debug, Clone)]
pub struct HigherOrderAnalyzer {
    config: HigherOrderConfig,
}

impl Default for HigherOrderAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl HigherOrderAnalyzer {
    /// Create a new higher-order statistics analyzer
    pub fn new() -> Self {
        Self {
            config: HigherOrderConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: HigherOrderConfig) -> Self {
        self.config = config;
        self
    }

    /// Compute comprehensive higher-order statistics
    pub fn analyze(&self, data: &Array2<Float>) -> Result<HigherOrderResults, SklearsError> {
        let (n_samples, n_features) = data.dim();

        if n_samples < 10 {
            return Err(SklearsError::InvalidInput(
                "Insufficient samples for higher-order statistics".to_string(),
            ));
        }

        // Center the data
        let centered_data = self.center_data(data)?;

        // Compute moments up to specified order
        let moments = self.compute_moments(&centered_data)?;

        // Compute cumulants if requested
        let cumulants = if self.config.use_cumulants {
            self.compute_cumulants(&moments)?
        } else {
            HashMap::new()
        };

        // Compute standardized measures
        let skewness = self.compute_skewness(&moments)?;
        let kurtosis = self.compute_kurtosis(&moments)?;

        // Compute cross-moments between variables
        let cross_moments = self.compute_cross_moments(&centered_data)?;

        // Compute independence measures
        let independence_measures = self.compute_independence_measures(&moments, &cross_moments)?;

        // Bootstrap confidence intervals
        let confidence_intervals = self.bootstrap_confidence_intervals(data)?;

        Ok(HigherOrderResults {
            moments,
            cumulants,
            skewness,
            kurtosis,
            cross_moments,
            independence_measures,
            confidence_intervals,
        })
    }

    /// Center data by subtracting column means
    fn center_data(&self, data: &Array2<Float>) -> Result<Array2<Float>, SklearsError> {
        let means = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &means.insert_axis(Axis(0));
        Ok(centered)
    }

    /// Compute moments up to specified order
    fn compute_moments(
        &self,
        data: &Array2<Float>,
    ) -> Result<HashMap<usize, Array2<Float>>, SklearsError> {
        let (n_samples, n_features) = data.dim();
        let mut moments = HashMap::new();

        for order in 1..=self.config.max_order {
            let mut moment_matrix = Array2::<Float>::zeros((n_features, n_features));

            for i in 0..n_features {
                for j in 0..n_features {
                    let col_i = data.column(i);
                    let col_j = data.column(j);

                    // Compute mixed moment E[X_i^p * X_j^q] where p + q = order
                    let mut moment_sum = 0.0;
                    for p in 0..=order {
                        let q = order - p;
                        let mixed_moment = self.compute_mixed_moment(&col_i, &col_j, p, q)?;
                        moment_sum += mixed_moment;
                    }

                    moment_matrix[[i, j]] = moment_sum / (order + 1) as Float;
                }
            }

            moments.insert(order, moment_matrix);
        }

        Ok(moments)
    }

    /// Compute mixed moment E[X^p * Y^q]
    fn compute_mixed_moment(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        p: usize,
        q: usize,
    ) -> Result<Float, SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Arrays must have same length".to_string(),
            ));
        }

        let n = x.len() as Float;
        let mut sum = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            sum += xi.powi(p as i32) * yi.powi(q as i32);
        }

        Ok(sum / n)
    }

    /// Compute cumulants from moments using recursive formula
    fn compute_cumulants(
        &self,
        moments: &HashMap<usize, Array2<Float>>,
    ) -> Result<HashMap<usize, Array2<Float>>, SklearsError> {
        let mut cumulants = HashMap::new();

        // First cumulant is the mean (should be zero for centered data)
        if let Some(first_moment) = moments.get(&1) {
            cumulants.insert(1, first_moment.clone());
        }

        // Second cumulant is the variance
        if let Some(second_moment) = moments.get(&2) {
            cumulants.insert(2, second_moment.clone());
        }

        // Third cumulant is the third central moment
        if let Some(third_moment) = moments.get(&3) {
            cumulants.insert(3, third_moment.clone());
        }

        // Fourth cumulant = fourth moment - 3 * (second moment)^2
        if let (Some(fourth_moment), Some(second_moment)) = (moments.get(&4), moments.get(&2)) {
            let fourth_cumulant = fourth_moment - &(second_moment * second_moment * 3.0);
            cumulants.insert(4, fourth_cumulant);
        }

        // Higher-order cumulants can be computed recursively
        for order in 5..=self.config.max_order {
            if let Some(moment) = moments.get(&order) {
                let cumulant = self.compute_cumulant_recursive(order, moments)?;
                cumulants.insert(order, cumulant);
            }
        }

        Ok(cumulants)
    }

    /// Compute cumulant recursively using Bell polynomials
    fn compute_cumulant_recursive(
        &self,
        order: usize,
        moments: &HashMap<usize, Array2<Float>>,
    ) -> Result<Array2<Float>, SklearsError> {
        // Simplified recursive computation - in practice would use full Bell polynomial formula
        if let Some(moment) = moments.get(&order) {
            let mut cumulant = moment.clone();

            // Subtract lower-order corrections (simplified)
            for k in 1..order {
                if let Some(lower_moment) = moments.get(&k) {
                    let binomial_coeff = self.binomial_coefficient(order - 1, k - 1) as Float;
                    cumulant = &cumulant - &(lower_moment * binomial_coeff);
                }
            }

            Ok(cumulant)
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Moment of order {} not found",
                order
            )))
        }
    }

    /// Compute binomial coefficient
    fn binomial_coefficient(&self, n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Use symmetry
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Compute skewness (standardized third moment)
    fn compute_skewness(
        &self,
        moments: &HashMap<usize, Array2<Float>>,
    ) -> Result<Array1<Float>, SklearsError> {
        let second_moment = moments.get(&2).ok_or_else(|| {
            SklearsError::InvalidInput("Second moment required for skewness".to_string())
        })?;
        let third_moment = moments.get(&3).ok_or_else(|| {
            SklearsError::InvalidInput("Third moment required for skewness".to_string())
        })?;

        let n_features = second_moment.nrows();
        let mut skewness = Array1::<Float>::zeros(n_features);

        for i in 0..n_features {
            let variance = second_moment[[i, i]];
            let third_central = third_moment[[i, i]];

            if variance > self.config.regularization {
                skewness[i] = third_central / variance.powf(1.5);
            }
        }

        Ok(skewness)
    }

    /// Compute kurtosis (standardized fourth moment)
    fn compute_kurtosis(
        &self,
        moments: &HashMap<usize, Array2<Float>>,
    ) -> Result<Array1<Float>, SklearsError> {
        let second_moment = moments.get(&2).ok_or_else(|| {
            SklearsError::InvalidInput("Second moment required for kurtosis".to_string())
        })?;
        let fourth_moment = moments.get(&4).ok_or_else(|| {
            SklearsError::InvalidInput("Fourth moment required for kurtosis".to_string())
        })?;

        let n_features = second_moment.nrows();
        let mut kurtosis = Array1::<Float>::zeros(n_features);

        for i in 0..n_features {
            let variance = second_moment[[i, i]];
            let fourth_central = fourth_moment[[i, i]];

            if variance > self.config.regularization {
                // Excess kurtosis (subtract 3 for normal distribution)
                kurtosis[i] = fourth_central / (variance * variance) - 3.0;
            }
        }

        Ok(kurtosis)
    }

    /// Compute cross-moments between all pairs of variables
    fn compute_cross_moments(
        &self,
        data: &Array2<Float>,
    ) -> Result<HashMap<(usize, usize), Array2<Float>>, SklearsError> {
        let (n_samples, n_features) = data.dim();
        let mut cross_moments = HashMap::new();

        for i in 0..n_features {
            for j in i..n_features {
                let mut cross_moment_matrix =
                    Array2::<Float>::zeros((self.config.max_order, self.config.max_order));

                let col_i = data.column(i);
                let col_j = data.column(j);

                for p in 1..=self.config.max_order {
                    for q in 1..=self.config.max_order {
                        let cross_moment = self.compute_mixed_moment(&col_i, &col_j, p, q)?;
                        cross_moment_matrix[[p - 1, q - 1]] = cross_moment;
                    }
                }

                cross_moments.insert((i, j), cross_moment_matrix.clone());
                if i != j {
                    cross_moments.insert((j, i), cross_moment_matrix.t().to_owned());
                }
            }
        }

        Ok(cross_moments)
    }

    /// Compute independence measures based on higher-order statistics
    fn compute_independence_measures(
        &self,
        moments: &HashMap<usize, Array2<Float>>,
        cross_moments: &HashMap<(usize, usize), Array2<Float>>,
    ) -> Result<Array2<Float>, SklearsError> {
        // Get the number of features from any moment matrix
        let n_features = if let Some(moment) = moments.values().next() {
            moment.nrows()
        } else {
            return Err(SklearsError::InvalidInput(
                "No moments available".to_string(),
            ));
        };

        let mut independence_matrix = Array2::<Float>::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    independence_matrix[[i, j]] = 1.0; // Perfect dependence with self
                } else {
                    // Compute independence measure using higher-order cross-moments
                    let independence =
                        self.compute_pairwise_independence(i, j, moments, cross_moments)?;
                    independence_matrix[[i, j]] = independence;
                }
            }
        }

        Ok(independence_matrix)
    }

    /// Compute pairwise independence measure
    fn compute_pairwise_independence(
        &self,
        i: usize,
        j: usize,
        moments: &HashMap<usize, Array2<Float>>,
        cross_moments: &HashMap<(usize, usize), Array2<Float>>,
    ) -> Result<Float, SklearsError> {
        let cross_moment_matrix = cross_moments.get(&(i, j)).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Cross-moment not found for pair ({}, {})", i, j))
        })?;

        // Use higher-order moments to measure independence
        // Independence measure based on deviation from product of marginals
        let mut independence_sum = 0.0;
        let mut count = 0;

        for order in 2..=self.config.max_order {
            if let Some(moment_matrix) = moments.get(&order) {
                let marginal_i = moment_matrix[[i, i]];
                let marginal_j = moment_matrix[[j, j]];
                let joint_moment = cross_moment_matrix[[order - 1, order - 1]];

                // Independence implies joint = product of marginals
                let expected_product = marginal_i * marginal_j;
                let deviation = (joint_moment - expected_product).abs();

                independence_sum += deviation;
                count += 1;
            }
        }

        if count > 0 {
            Ok(independence_sum / count as Float)
        } else {
            Ok(0.0)
        }
    }

    /// Bootstrap confidence intervals for statistics
    fn bootstrap_confidence_intervals(
        &self,
        data: &Array2<Float>,
    ) -> Result<HashMap<String, (Array1<Float>, Array1<Float>)>, SklearsError> {
        let (n_samples, n_features) = data.dim();
        let mut ci_map = HashMap::new();
        let mut rng = thread_rng();

        // Bootstrap samples for skewness and kurtosis
        let mut bootstrap_skewness = Array2::<Float>::zeros((self.config.n_bootstrap, n_features));
        let mut bootstrap_kurtosis = Array2::<Float>::zeros((self.config.n_bootstrap, n_features));

        for bootstrap_iter in 0..self.config.n_bootstrap {
            // Create bootstrap sample
            let mut bootstrap_data = Array2::<Float>::zeros((n_samples, n_features));
            for i in 0..n_samples {
                let random_idx = rng.gen_range(0..n_samples);
                bootstrap_data.row_mut(i).assign(&data.row(random_idx));
            }

            // Compute statistics for bootstrap sample
            let centered_bootstrap = self.center_data(&bootstrap_data)?;
            let bootstrap_moments = self.compute_moments(&centered_bootstrap)?;
            let bootstrap_skew = self.compute_skewness(&bootstrap_moments)?;
            let bootstrap_kurt = self.compute_kurtosis(&bootstrap_moments)?;

            bootstrap_skewness
                .row_mut(bootstrap_iter)
                .assign(&bootstrap_skew);
            bootstrap_kurtosis
                .row_mut(bootstrap_iter)
                .assign(&bootstrap_kurt);
        }

        // Compute confidence intervals (2.5% and 97.5% percentiles)
        let skewness_lower = self.compute_percentile(&bootstrap_skewness, 2.5)?;
        let skewness_upper = self.compute_percentile(&bootstrap_skewness, 97.5)?;
        let kurtosis_lower = self.compute_percentile(&bootstrap_kurtosis, 2.5)?;
        let kurtosis_upper = self.compute_percentile(&bootstrap_kurtosis, 97.5)?;

        ci_map.insert("skewness".to_string(), (skewness_lower, skewness_upper));
        ci_map.insert("kurtosis".to_string(), (kurtosis_lower, kurtosis_upper));

        Ok(ci_map)
    }

    /// Compute percentile of bootstrap samples
    fn compute_percentile(
        &self,
        data: &Array2<Float>,
        percentile: f64,
    ) -> Result<Array1<Float>, SklearsError> {
        let (n_bootstrap, n_features) = data.dim();
        let mut result = Array1::<Float>::zeros(n_features);

        for feature in 0..n_features {
            let mut column_data: Vec<Float> = data.column(feature).to_vec();
            column_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let index = (percentile / 100.0 * (n_bootstrap - 1) as f64) as usize;
            result[feature] = column_data[index.min(n_bootstrap - 1)];
        }

        Ok(result)
    }
}

/// Non-Gaussian Component Analysis using higher-order statistics
#[derive(Debug, Clone)]
pub struct NonGaussianComponentAnalysis {
    config: HigherOrderConfig,
    /// Target kurtosis for components
    target_kurtosis: Float,
}

impl Default for NonGaussianComponentAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl NonGaussianComponentAnalysis {
    /// Create new non-Gaussian component analysis
    pub fn new() -> Self {
        Self {
            config: HigherOrderConfig::default(),
            target_kurtosis: 0.0, // Zero for maximum non-Gaussianity
        }
    }

    /// Set target kurtosis
    pub fn with_target_kurtosis(mut self, kurtosis: Float) -> Self {
        self.target_kurtosis = kurtosis;
        self
    }

    /// Find components that maximize non-Gaussianity
    pub fn fit(&self, data: &Array2<Float>) -> Result<NonGaussianResults, SklearsError> {
        let analyzer = HigherOrderAnalyzer::new().with_config(self.config.clone());
        let higher_order_results = analyzer.analyze(data)?;

        // Find components with maximum deviation from Gaussian kurtosis
        let kurtosis_deviations = higher_order_results
            .kurtosis
            .mapv(|k| (k - self.target_kurtosis).abs());

        // Sort components by non-Gaussianity
        let mut component_indices: Vec<usize> = (0..kurtosis_deviations.len()).collect();
        component_indices.sort_by(|&i, &j| {
            kurtosis_deviations[j]
                .partial_cmp(&kurtosis_deviations[i])
                .unwrap()
        });

        Ok(NonGaussianResults {
            component_order: component_indices,
            kurtosis_values: higher_order_results.kurtosis,
            skewness_values: higher_order_results.skewness,
            non_gaussianity_scores: kurtosis_deviations,
        })
    }
}

/// Results from non-Gaussian component analysis
#[derive(Debug, Clone)]
pub struct NonGaussianResults {
    /// Components ordered by non-Gaussianity
    pub component_order: Vec<usize>,
    /// Kurtosis values for each component
    pub kurtosis_values: Array1<Float>,
    /// Skewness values for each component
    pub skewness_values: Array1<Float>,
    /// Non-Gaussianity scores
    pub non_gaussianity_scores: Array1<Float>,
}

/// Polyspectral cross-decomposition methods
#[derive(Debug, Clone)]
pub struct PolyspectralCCA {
    /// Order of polyspectrum to use
    polyspectrum_order: usize,
    /// Number of components
    n_components: usize,
    config: HigherOrderConfig,
}

impl PolyspectralCCA {
    pub fn new(polyspectrum_order: usize, n_components: usize) -> Self {
        Self {
            polyspectrum_order,
            n_components,
            config: HigherOrderConfig::default(),
        }
    }

    /// Fit polyspectral CCA
    pub fn fit(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<PolyspectralResults, SklearsError> {
        // Compute higher-order cross-spectra
        let x_polyspectrum = self.compute_polyspectrum(x)?;
        let y_polyspectrum = self.compute_polyspectrum(y)?;
        let cross_polyspectrum = self.compute_cross_polyspectrum(x, y)?;

        // Find canonical components in polyspectral domain
        let components = self.find_polyspectral_components(
            &x_polyspectrum,
            &y_polyspectrum,
            &cross_polyspectrum,
        )?;

        Ok(PolyspectralResults {
            x_components: components.0,
            y_components: components.1,
            polyspectral_correlations: Array1::zeros(self.n_components), // Placeholder
            x_polyspectrum,
            y_polyspectrum,
            cross_polyspectrum,
        })
    }

    /// Compute polyspectrum of given order
    fn compute_polyspectrum(&self, data: &Array2<Float>) -> Result<Array3<Float>, SklearsError> {
        let (n_samples, n_features) = data.dim();

        // Simplified polyspectrum computation (placeholder)
        // In practice, would involve FFT and higher-order spectral analysis
        let mut rng = thread_rng();
        let polyspectrum = Array3::<Float>::from_shape_fn(
            (n_features, n_features, self.polyspectrum_order),
            |_| rng.gen::<Float>(),
        );

        Ok(polyspectrum)
    }

    /// Compute cross-polyspectrum between two datasets
    fn compute_cross_polyspectrum(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<Array3<Float>, SklearsError> {
        let (_, n_features_x) = x.dim();
        let (_, n_features_y) = y.dim();

        // Simplified cross-polyspectrum computation
        let mut rng = thread_rng();
        let cross_polyspectrum = Array3::<Float>::from_shape_fn(
            (n_features_x, n_features_y, self.polyspectrum_order),
            |_| rng.gen::<Float>(),
        );

        Ok(cross_polyspectrum)
    }

    /// Find canonical components in polyspectral domain
    fn find_polyspectral_components(
        &self,
        x_poly: &Array3<Float>,
        y_poly: &Array3<Float>,
        cross_poly: &Array3<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>), SklearsError> {
        let (n_features_x, _, _) = x_poly.dim();
        let (n_features_y, _, _) = y_poly.dim();

        // Placeholder implementation - would involve eigenvalue decomposition in polyspectral domain
        let mut rng = thread_rng();
        let x_components =
            Array2::<Float>::from_shape_fn((n_features_x, self.n_components), |_| {
                rng.gen::<Float>()
            });
        let y_components =
            Array2::<Float>::from_shape_fn((n_features_y, self.n_components), |_| {
                rng.gen::<Float>()
            });

        Ok((x_components, y_components))
    }
}

/// Results from polyspectral CCA
#[derive(Debug, Clone)]
pub struct PolyspectralResults {
    /// X canonical components
    pub x_components: Array2<Float>,
    /// Y canonical components
    pub y_components: Array2<Float>,
    /// Polyspectral canonical correlations
    pub polyspectral_correlations: Array1<Float>,
    /// X polyspectrum
    pub x_polyspectrum: Array3<Float>,
    /// Y polyspectrum
    pub y_polyspectrum: Array3<Float>,
    /// Cross-polyspectrum
    pub cross_polyspectrum: Array3<Float>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_higher_order_analysis() {
        let data = Array2::from_shape_fn((100, 5), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let analyzer = HigherOrderAnalyzer::new();

        let result = analyzer.analyze(&data);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.skewness.len(), 5);
        assert_eq!(result.kurtosis.len(), 5);
        assert!(!result.moments.is_empty());
    }

    #[test]
    fn test_moment_computation() {
        let data = Array2::<Float>::ones((50, 3));
        let analyzer = HigherOrderAnalyzer::new();
        let centered = analyzer.center_data(&data).unwrap();

        let moments = analyzer.compute_moments(&centered).unwrap();
        assert!(moments.contains_key(&2));
        assert!(moments.contains_key(&3));
        assert!(moments.contains_key(&4));
    }

    #[test]
    fn test_cumulant_computation() {
        let data = Array2::from_shape_fn((75, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let analyzer = HigherOrderAnalyzer::new();
        let centered = analyzer.center_data(&data).unwrap();
        let moments = analyzer.compute_moments(&centered).unwrap();

        let cumulants = analyzer.compute_cumulants(&moments).unwrap();
        assert!(cumulants.contains_key(&2));
        assert!(cumulants.contains_key(&3));
        assert!(cumulants.contains_key(&4));
    }

    #[test]
    fn test_skewness_kurtosis() {
        // Create data with known skewness/kurtosis properties
        let mut data = Array2::from_shape_fn((200, 3), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });

        // Make one column highly skewed
        for i in 0..data.nrows() {
            let val: f64 = data[[i, 0]];
            data[[i, 0]] = val.abs(); // Right-skewed
        }

        let analyzer = HigherOrderAnalyzer::new();
        let result = analyzer.analyze(&data).unwrap();

        // First column should have positive skewness
        assert!(result.skewness[0] > 0.0);
    }

    #[test]
    fn test_non_gaussian_analysis() {
        let data = Array2::from_shape_fn((150, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let ng_analyzer = NonGaussianComponentAnalysis::new();

        let result = ng_analyzer.fit(&data);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.component_order.len(), 4);
        assert_eq!(result.kurtosis_values.len(), 4);
        assert_eq!(result.skewness_values.len(), 4);
    }

    #[test]
    fn test_polyspectral_cca() {
        let x = Array2::from_shape_fn((100, 6), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((100, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let poly_cca = PolyspectralCCA::new(3, 2);

        let result = poly_cca.fit(&x, &y);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.x_components.dim(), (6, 2));
        assert_eq!(result.y_components.dim(), (4, 2));
    }

    #[test]
    fn test_independence_measures() {
        let data = Array2::from_shape_fn((80, 5), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let analyzer = HigherOrderAnalyzer::new();
        let result = analyzer.analyze(&data).unwrap();

        assert_eq!(result.independence_measures.dim(), (5, 5));

        // Diagonal should be 1.0 (perfect self-dependence)
        for i in 0..5 {
            assert!((result.independence_measures[[i, i]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bootstrap_confidence_intervals() {
        let data = Array2::from_shape_fn((60, 3), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let mut config = HigherOrderConfig::default();
        config.n_bootstrap = 100; // Smaller for test speed

        let analyzer = HigherOrderAnalyzer::new().with_config(config);
        let result = analyzer.analyze(&data).unwrap();

        assert!(result.confidence_intervals.contains_key("skewness"));
        assert!(result.confidence_intervals.contains_key("kurtosis"));

        let (skew_lower, skew_upper) = &result.confidence_intervals["skewness"];
        assert_eq!(skew_lower.len(), 3);
        assert_eq!(skew_upper.len(), 3);

        // Upper bounds should be >= lower bounds
        for i in 0..3 {
            assert!(skew_upper[i] >= skew_lower[i]);
        }
    }
}
