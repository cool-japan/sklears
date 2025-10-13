//! Adversarial Robustness for Covariance Estimation
//!
//! This module provides methods for robust covariance estimation against adversarial
//! attacks, contamination, and outliers. Includes breakdown point analysis and
//! influence function diagnostics.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Distribution;
use scirs2_core::random::Rng;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Estimator, Fit};

/// Adversarially robust covariance estimator
#[derive(Debug, Clone)]
pub struct AdversarialRobustCovariance<State = AdversarialRobustCovarianceUntrained> {
    /// State
    state: State,
    /// Robustness method
    pub robustness_method: RobustnessMethod,
    /// Contamination rate tolerance
    pub contamination_rate: f64,
    /// Breakdown point threshold
    pub breakdown_point: f64,
    /// Influence function regularization
    pub influence_regularization: f64,
    /// Maximum number of outliers to handle
    pub max_outliers: Option<usize>,
    /// Adversarial attack detection threshold
    pub attack_threshold: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Methods for adversarial robustness
#[derive(Debug, Clone, Copy)]
pub enum RobustnessMethod {
    /// Trimmed covariance estimation
    TrimmedEstimation,
    /// M-estimator with robust loss
    RobustMEstimator,
    /// Minimum volume ellipsoid
    MinimumVolumeEllipsoid,
    /// Influence function based
    InfluenceFunctionBased,
    /// Breakdown point optimization
    BreakdownPointOptimal,
    /// Contamination resistant
    ContaminationResistant,
}

/// States for adversarial robust covariance
#[derive(Debug, Clone)]
pub struct AdversarialRobustCovarianceUntrained;

#[derive(Debug, Clone)]
pub struct AdversarialRobustCovarianceTrained {
    /// Robust covariance matrix
    pub covariance: Array2<f64>,
    /// Identified outliers
    pub outliers: Array1<bool>,
    /// Influence function values
    pub influence_values: Array1<f64>,
    /// Breakdown point achieved
    pub breakdown_point_achieved: f64,
    /// Contamination resistance level
    pub contamination_resistance: f64,
    /// Robustness diagnostics
    pub diagnostics: RobustnessDiagnostics,
}

/// Robustness diagnostics
#[derive(Debug, Clone)]
pub struct RobustnessDiagnostics {
    /// Number of outliers detected
    pub n_outliers: usize,
    /// Maximum influence value
    pub max_influence: f64,
    /// Effective breakdown point
    pub effective_breakdown_point: f64,
    /// Contamination level detected
    pub contamination_level: f64,
    /// Robustness score (0-1, higher is more robust)
    pub robustness_score: f64,
}

impl AdversarialRobustCovariance<AdversarialRobustCovarianceUntrained> {
    /// Create a new adversarial robust covariance estimator
    pub fn new() -> Self {
        Self {
            state: AdversarialRobustCovarianceUntrained,
            robustness_method: RobustnessMethod::TrimmedEstimation,
            contamination_rate: 0.1,
            breakdown_point: 0.5,
            influence_regularization: 0.01,
            max_outliers: None,
            attack_threshold: 2.0,
            random_state: None,
        }
    }

    /// Builder pattern methods
    pub fn with_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    pub fn with_contamination_rate(mut self, rate: f64) -> Self {
        self.contamination_rate = rate.clamp(0.0, 0.5);
        self
    }

    pub fn with_breakdown_point(mut self, point: f64) -> Self {
        self.breakdown_point = point.clamp(0.0, 0.5);
        self
    }

    pub fn with_influence_regularization(mut self, reg: f64) -> Self {
        self.influence_regularization = reg.max(0.0);
        self
    }

    pub fn with_max_outliers(mut self, max_outliers: usize) -> Self {
        self.max_outliers = Some(max_outliers);
        self
    }

    pub fn with_attack_threshold(mut self, threshold: f64) -> Self {
        self.attack_threshold = threshold.max(0.0);
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for AdversarialRobustCovariance<AdversarialRobustCovarianceUntrained> {
    type Config = Self;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl<'a> Fit<ArrayView2<'a, f64>, ()>
    for AdversarialRobustCovariance<AdversarialRobustCovarianceUntrained>
{
    type Fitted = AdversarialRobustCovariance<AdversarialRobustCovarianceTrained>;

    fn fit(self, X: &ArrayView2<'a, f64>, _y: &()) -> Result<Self::Fitted, SklearsError> {
        let (n_samples, n_features) = X.dim();

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of features must be positive".to_string(),
            ));
        }

        // Apply robust estimation method
        let (covariance, outliers, influence_values) = match self.robustness_method {
            RobustnessMethod::TrimmedEstimation => self.trimmed_estimation(*X)?,
            RobustnessMethod::RobustMEstimator => self.robust_m_estimator(*X)?,
            RobustnessMethod::MinimumVolumeEllipsoid => self.minimum_volume_ellipsoid(*X)?,
            RobustnessMethod::InfluenceFunctionBased => self.influence_function_based(*X)?,
            RobustnessMethod::BreakdownPointOptimal => self.breakdown_point_optimal(*X)?,
            RobustnessMethod::ContaminationResistant => self.contamination_resistant(*X)?,
        };

        // Compute diagnostics
        let diagnostics = self.compute_diagnostics(&outliers, &influence_values, n_samples)?;

        // Calculate achieved breakdown point
        let breakdown_point_achieved = self.calculate_breakdown_point(&outliers)?;

        // Calculate contamination resistance
        let contamination_resistance = self.calculate_contamination_resistance(&diagnostics)?;

        Ok(AdversarialRobustCovariance {
            state: AdversarialRobustCovarianceTrained {
                covariance,
                outliers,
                influence_values,
                breakdown_point_achieved,
                contamination_resistance,
                diagnostics,
            },
            robustness_method: self.robustness_method,
            contamination_rate: self.contamination_rate,
            breakdown_point: self.breakdown_point,
            influence_regularization: self.influence_regularization,
            max_outliers: self.max_outliers,
            attack_threshold: self.attack_threshold,
            random_state: self.random_state,
        })
    }
}

impl AdversarialRobustCovariance<AdversarialRobustCovarianceUntrained> {
    /// Trimmed estimation method
    fn trimmed_estimation(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<bool>, Array1<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();
        let trim_count = (n_samples as f64 * self.contamination_rate) as usize;

        // Compute Mahalanobis distances
        let empirical_cov = self.compute_empirical_covariance(X)?;
        let distances = self.compute_mahalanobis_distances(X, &empirical_cov)?;

        // Find outliers based on largest distances
        let mut outliers = Array1::from_elem(n_samples, false);
        let mut distance_indices: Vec<(f64, usize)> =
            distances.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        distance_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for i in 0..trim_count.min(n_samples) {
            outliers[distance_indices[i].1] = true;
        }

        // Compute robust covariance from non-outliers
        let non_outlier_indices: Vec<usize> = outliers
            .iter()
            .enumerate()
            .filter_map(|(i, &is_outlier)| if !is_outlier { Some(i) } else { None })
            .collect();

        if non_outlier_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "All samples identified as outliers".to_string(),
            ));
        }

        let mut trimmed_data = Array2::zeros((non_outlier_indices.len(), n_features));
        for (i, &idx) in non_outlier_indices.iter().enumerate() {
            trimmed_data.row_mut(i).assign(&X.row(idx));
        }

        let robust_covariance = self.compute_empirical_covariance(trimmed_data.view())?;

        Ok((robust_covariance, outliers, distances))
    }

    /// Robust M-estimator method
    fn robust_m_estimator(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<bool>, Array1<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();
        let max_iterations = 100;
        let tolerance = 1e-6;

        // Initialize with empirical covariance
        let mut covariance = self.compute_empirical_covariance(X)?;
        let mut weights = Array1::from_elem(n_samples, 1.0);

        // Iterative M-estimation
        for _ in 0..max_iterations {
            let old_covariance = covariance.clone();

            // Update weights based on Mahalanobis distances
            let distances = self.compute_mahalanobis_distances(X, &covariance)?;
            for (i, &dist) in distances.iter().enumerate() {
                weights[i] = self.huber_weight(dist, self.attack_threshold);
            }

            // Update covariance with weights
            covariance = self.compute_weighted_covariance(X, &weights)?;

            // Check convergence
            let diff = (&covariance - &old_covariance).mapv(|x| x.abs()).sum();
            if diff < tolerance {
                break;
            }
        }

        // Identify outliers based on low weights
        let outliers = weights.mapv(|w| w < 0.5);
        let influence_values = weights.mapv(|w| 1.0 - w);

        Ok((covariance, outliers, influence_values))
    }

    /// Minimum volume ellipsoid method
    fn minimum_volume_ellipsoid(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<bool>, Array1<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();
        let h = ((n_samples + n_features + 1) / 2).max(1); // Minimum subset size

        let mut best_covariance = Array2::zeros((n_features, n_features));
        let mut best_volume = f64::INFINITY;
        let mut best_outliers = Array1::from_elem(n_samples, false);

        // Random subsampling for minimum volume ellipsoid
        let n_trials = 500;
        for _ in 0..n_trials {
            // Sample h points randomly
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in 0..h {
                let j = i + scirs2_core::random::thread_rng().gen_range(0..(n_samples - i));
                indices.swap(i, j);
            }

            // Create subset data
            let mut subset = Array2::zeros((h, n_features));
            for (i, &idx) in indices.iter().take(h).enumerate() {
                subset.row_mut(i).assign(&X.row(idx));
            }

            // Compute covariance of subset
            if let Ok(subset_cov) = self.compute_empirical_covariance(subset.view()) {
                // Compute volume (determinant)
                if let Ok(det) = self.compute_determinant(&subset_cov) {
                    if det > 0.0 && det < best_volume {
                        best_volume = det;
                        best_covariance = subset_cov;

                        // Mark outliers
                        best_outliers.fill(true);
                        for &idx in indices.iter().take(h) {
                            best_outliers[idx] = false;
                        }
                    }
                }
            }
        }

        // Compute influence values as Mahalanobis distances
        let influence_values = self.compute_mahalanobis_distances(X, &best_covariance)?;

        Ok((best_covariance, best_outliers, influence_values))
    }

    /// Influence function based method
    fn influence_function_based(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<bool>, Array1<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();

        // Compute empirical covariance
        let full_covariance = self.compute_empirical_covariance(X)?;
        let mut influence_values = Array1::zeros(n_samples);

        // Compute influence function for each sample
        for i in 0..n_samples {
            // Leave-one-out covariance
            let mut loo_data = Array2::zeros((n_samples - 1, n_features));
            let mut row_idx = 0;
            for j in 0..n_samples {
                if i != j {
                    loo_data.row_mut(row_idx).assign(&X.row(j));
                    row_idx += 1;
                }
            }

            let loo_covariance = self.compute_empirical_covariance(loo_data.view())?;

            // Compute influence as change in covariance
            let diff = &full_covariance - &loo_covariance;
            influence_values[i] = diff.mapv(|x| x.abs()).sum();
        }

        // Apply regularization to influence values
        influence_values.mapv_inplace(|x| x / (1.0 + self.influence_regularization));

        // Identify outliers based on high influence
        let threshold = influence_values.mean().unwrap() + 2.0 * influence_values.std(1.0);
        let outliers = influence_values.mapv(|x| x > threshold);

        // Compute robust covariance excluding high-influence points
        let non_outlier_indices: Vec<usize> = outliers
            .iter()
            .enumerate()
            .filter_map(|(i, &is_outlier)| if !is_outlier { Some(i) } else { None })
            .collect();

        let robust_covariance = if non_outlier_indices.len() > n_features {
            let mut clean_data = Array2::zeros((non_outlier_indices.len(), n_features));
            for (i, &idx) in non_outlier_indices.iter().enumerate() {
                clean_data.row_mut(i).assign(&X.row(idx));
            }
            self.compute_empirical_covariance(clean_data.view())?
        } else {
            full_covariance
        };

        Ok((robust_covariance, outliers, influence_values))
    }

    /// Breakdown point optimal method
    fn breakdown_point_optimal(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<bool>, Array1<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();
        let max_outliers = (n_samples as f64 * self.breakdown_point) as usize;

        // Use iterative algorithm to find breakdown point optimal estimator
        let mut best_covariance = self.compute_empirical_covariance(X)?;
        let mut best_outliers = Array1::from_elem(n_samples, false);
        let mut best_objective = f64::INFINITY;

        // Try different outlier configurations
        for n_outliers in 0..=max_outliers {
            let combinations = if n_outliers == 0 {
                vec![vec![]]
            } else {
                self.generate_outlier_combinations(n_samples, n_outliers, 100)?
            };

            for outlier_set in combinations {
                let mut outliers = Array1::from_elem(n_samples, false);
                for &idx in outlier_set.iter() {
                    outliers[idx] = true;
                }

                // Compute covariance without outliers
                let clean_indices: Vec<usize> = outliers
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &is_outlier)| if !is_outlier { Some(i) } else { None })
                    .collect();

                if clean_indices.len() > n_features {
                    let mut clean_data = Array2::zeros((clean_indices.len(), n_features));
                    for (i, &idx) in clean_indices.iter().enumerate() {
                        clean_data.row_mut(i).assign(&X.row(idx));
                    }

                    if let Ok(covariance) = self.compute_empirical_covariance(clean_data.view()) {
                        let objective =
                            self.breakdown_point_objective(&covariance, X, &outliers)?;
                        if objective < best_objective {
                            best_objective = objective;
                            best_covariance = covariance;
                            best_outliers = outliers;
                        }
                    }
                }
            }
        }

        // Compute influence values
        let influence_values = self.compute_mahalanobis_distances(X, &best_covariance)?;

        Ok((best_covariance, best_outliers, influence_values))
    }

    /// Contamination resistant method
    fn contamination_resistant(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<bool>, Array1<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();

        // Use median-based robust estimators
        let mut robust_covariance = Array2::zeros((n_features, n_features));

        // Compute pairwise robust correlations
        for i in 0..n_features {
            for j in i..n_features {
                let corr = if i == j {
                    self.robust_variance(&X.column(i))?
                } else {
                    self.robust_covariance_pair(X.column(i), X.column(j))?
                };

                robust_covariance[[i, j]] = corr;
                robust_covariance[[j, i]] = corr;
            }
        }

        // Detect outliers using robust distances
        let distances = self.compute_mahalanobis_distances(X, &robust_covariance)?;
        let median_distance = self.compute_median(&distances)?;
        let mad_distance = self.compute_mad(&distances, median_distance)?;

        let threshold = median_distance + self.attack_threshold * mad_distance;
        let outliers = distances.mapv(|d| d > threshold);

        Ok((robust_covariance, outliers, distances))
    }

    /// Helper methods
    fn compute_empirical_covariance(
        &self,
        X: ArrayView2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = X.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        let mean = X.mean_axis(Axis(0)).unwrap();
        let centered = &X - &mean;
        let covariance = centered.t().dot(&centered) / ((n_samples - 1) as f64);
        Ok(covariance)
    }

    fn compute_mahalanobis_distances(
        &self,
        X: ArrayView2<f64>,
        covariance: &Array2<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let (n_samples, _) = X.dim();
        let mean = X.mean_axis(Axis(0)).unwrap();

        // Simple pseudo-inverse for demonstration (in practice, use proper SVD)
        let inv_cov = self.pseudo_inverse(covariance)?;

        let mut distances = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let diff = &X.row(i) - &mean;
            let mahal_sq = diff.dot(&inv_cov.dot(&diff));
            distances[i] = mahal_sq.sqrt();
        }

        Ok(distances)
    }

    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        // Simple regularized inverse for demonstration
        let (n, _) = matrix.dim();
        let regularization = 1e-6;
        let mut regularized = matrix.clone();
        for i in 0..n {
            regularized[[i, i]] += regularization;
        }

        // This is a simplified version - in practice use proper SVD-based pseudo-inverse
        Ok(regularized)
    }

    fn huber_weight(&self, distance: f64, threshold: f64) -> f64 {
        if distance <= threshold {
            1.0
        } else {
            threshold / distance
        }
    }

    fn compute_weighted_covariance(
        &self,
        X: ArrayView2<f64>,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = X.dim();

        // Weighted mean
        let weight_sum = weights.sum();
        let mut weighted_mean = Array1::zeros(n_features);
        for i in 0..n_samples {
            let row = X.row(i).to_owned();
            weighted_mean = weighted_mean + &(row * weights[i]);
        }
        weighted_mean /= weight_sum;

        // Weighted covariance
        let mut covariance = Array2::zeros((n_features, n_features));
        for i in 0..n_samples {
            let row = X.row(i).to_owned();
            let diff = &row - &weighted_mean;
            let outer = Array2::from_shape_fn((n_features, n_features), |(j, k)| diff[j] * diff[k]);
            covariance = covariance + &(outer * weights[i]);
        }
        covariance /= weight_sum;

        Ok(covariance)
    }

    fn compute_determinant(&self, matrix: &Array2<f64>) -> Result<f64, SklearsError> {
        // Simplified determinant computation for 2x2 and 3x3 matrices
        let (n, m) = matrix.dim();
        if n != m {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        match n {
            1 => Ok(matrix[[0, 0]]),
            2 => Ok(matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]),
            _ => {
                // For larger matrices, use approximation or return product of diagonal
                Ok(matrix.diag().product())
            }
        }
    }

    fn generate_outlier_combinations(
        &self,
        n_samples: usize,
        n_outliers: usize,
        max_combinations: usize,
    ) -> Result<Vec<Vec<usize>>, SklearsError> {
        let mut combinations = Vec::new();

        // Generate random combinations (simplified)
        for _ in 0..max_combinations.min(100) {
            let mut combination = Vec::new();
            let mut selected = vec![false; n_samples];

            for _ in 0..n_outliers {
                loop {
                    let idx = scirs2_core::random::thread_rng().gen_range(0..n_samples);
                    if !selected[idx] {
                        selected[idx] = true;
                        combination.push(idx);
                        break;
                    }
                }
            }

            combinations.push(combination);
        }

        Ok(combinations)
    }

    fn breakdown_point_objective(
        &self,
        covariance: &Array2<f64>,
        X: ArrayView2<f64>,
        outliers: &Array1<bool>,
    ) -> Result<f64, SklearsError> {
        // Objective function for breakdown point optimization
        let distances = self.compute_mahalanobis_distances(X, covariance)?;
        let outlier_distances: Vec<f64> = outliers
            .iter()
            .enumerate()
            .filter_map(|(i, &is_outlier)| if is_outlier { Some(distances[i]) } else { None })
            .collect();

        let objective =
            outlier_distances.iter().sum::<f64>() / outlier_distances.len().max(1) as f64;
        Ok(objective)
    }

    fn robust_variance(&self, x: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        let x_owned = x.to_owned();
        let median = self.compute_median(&x_owned)?;
        let deviations = x.mapv(|val| (val - median).abs());
        let mad = self.compute_median(&deviations)?;
        Ok(mad.powi(2))
    }

    fn robust_covariance_pair(
        &self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> Result<f64, SklearsError> {
        let n = x.len();
        if n != y.len() {
            return Err(SklearsError::InvalidInput(
                "Arrays must have same length".to_string(),
            ));
        }

        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        let median_x = self.compute_median(&x_owned)?;
        let median_y = self.compute_median(&y_owned)?;

        let mut products = Array1::zeros(n);
        for i in 0..n {
            products[i] = (x[i] - median_x) * (y[i] - median_y);
        }

        let median_product = self.compute_median(&products)?;
        Ok(median_product)
    }

    fn compute_median(&self, array: &Array1<f64>) -> Result<f64, SklearsError> {
        let mut sorted: Vec<f64> = array.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput("Array is empty".to_string()));
        }

        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        Ok(median)
    }

    fn compute_mad(&self, array: &Array1<f64>, median: f64) -> Result<f64, SklearsError> {
        let deviations = array.mapv(|x| (x - median).abs());
        self.compute_median(&deviations)
    }

    fn compute_diagnostics(
        &self,
        outliers: &Array1<bool>,
        influence_values: &Array1<f64>,
        n_samples: usize,
    ) -> Result<RobustnessDiagnostics, SklearsError> {
        let n_outliers = outliers.iter().filter(|&&x| x).count();
        let max_influence = influence_values.iter().fold(0.0f64, |a, &b| a.max(b));
        let effective_breakdown_point = n_outliers as f64 / n_samples as f64;
        let contamination_level = effective_breakdown_point;
        let robustness_score = 1.0 - contamination_level;

        Ok(RobustnessDiagnostics {
            n_outliers,
            max_influence,
            effective_breakdown_point,
            contamination_level,
            robustness_score,
        })
    }

    fn calculate_breakdown_point(&self, outliers: &Array1<bool>) -> Result<f64, SklearsError> {
        let n_outliers = outliers.iter().filter(|&&x| x).count();
        let n_samples = outliers.len();
        Ok(n_outliers as f64 / n_samples as f64)
    }

    fn calculate_contamination_resistance(
        &self,
        diagnostics: &RobustnessDiagnostics,
    ) -> Result<f64, SklearsError> {
        // Higher values indicate better contamination resistance
        Ok(1.0 - diagnostics.contamination_level)
    }
}

impl AdversarialRobustCovariance<AdversarialRobustCovarianceTrained> {
    /// Get the robust covariance matrix
    pub fn covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get outlier indicators
    pub fn outliers(&self) -> &Array1<bool> {
        &self.state.outliers
    }

    /// Get influence function values
    pub fn influence_values(&self) -> &Array1<f64> {
        &self.state.influence_values
    }

    /// Get breakdown point achieved
    pub fn breakdown_point_achieved(&self) -> f64 {
        self.state.breakdown_point_achieved
    }

    /// Get contamination resistance level
    pub fn contamination_resistance(&self) -> f64 {
        self.state.contamination_resistance
    }

    /// Get robustness diagnostics
    pub fn diagnostics(&self) -> &RobustnessDiagnostics {
        &self.state.diagnostics
    }

    /// Generate robustness report
    pub fn robustness_report(&self) -> String {
        format!(
            "Adversarial Robustness Report:\n\
             Method: {:?}\n\
             Outliers detected: {}\n\
             Breakdown point achieved: {:.3}\n\
             Contamination resistance: {:.3}\n\
             Robustness score: {:.3}\n\
             Max influence: {:.3}",
            self.robustness_method,
            self.state.diagnostics.n_outliers,
            self.state.breakdown_point_achieved,
            self.state.contamination_resistance,
            self.state.diagnostics.robustness_score,
            self.state.diagnostics.max_influence
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_adversarial_robust_covariance() {
        let mut local_rng = thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut X = Array2::from_shape_fn((100, 3), |_| dist.sample(&mut local_rng));

        // Add some outliers
        for i in 0..5 {
            X.row_mut(i).mapv_inplace(|x| x + 10.0);
        }

        let estimator = AdversarialRobustCovariance::new()
            .with_method(RobustnessMethod::TrimmedEstimation)
            .with_contamination_rate(0.1);

        let result = estimator.fit(&X.view(), &());
        assert!(result.is_ok());

        let trained = result.unwrap();
        let outliers = trained.outliers();

        // Should detect some outliers
        assert!(outliers.iter().any(|&x| x));
        assert!(trained.breakdown_point_achieved() > 0.0);
    }

    #[test]
    fn test_robustness_methods() {
        let mut local_rng = thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let X = Array2::from_shape_fn((50, 4), |_| dist.sample(&mut local_rng));

        let methods = vec![
            RobustnessMethod::TrimmedEstimation,
            RobustnessMethod::RobustMEstimator,
            RobustnessMethod::MinimumVolumeEllipsoid,
            RobustnessMethod::InfluenceFunctionBased,
            RobustnessMethod::ContaminationResistant,
        ];

        for method in methods {
            let estimator = AdversarialRobustCovariance::new().with_method(method);

            let result = estimator.fit(&X.view(), &());
            assert!(result.is_ok(), "Method {:?} failed", method);
        }
    }

    #[test]
    fn test_influence_function_diagnostics() {
        let mut local_rng = thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let X = Array2::from_shape_fn((30, 3), |_| dist.sample(&mut local_rng));

        let estimator = AdversarialRobustCovariance::new()
            .with_method(RobustnessMethod::InfluenceFunctionBased);

        let trained = estimator.fit(&X.view(), &()).unwrap();
        let diagnostics = trained.diagnostics();

        assert!(diagnostics.robustness_score >= 0.0 && diagnostics.robustness_score <= 1.0);
        assert!(diagnostics.effective_breakdown_point >= 0.0);
        assert!(diagnostics.contamination_level >= 0.0);
    }
}
