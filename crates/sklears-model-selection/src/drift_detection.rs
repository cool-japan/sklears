//! Data drift detection for validation and monitoring
//!
//! This module provides methods to detect distribution changes in data
//! that can affect model performance over time.

use numrs2::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::SliceRandomExt;
use sklears_core::error::{Result, SklearsError};
use std::cmp::Ordering;

/// Configuration for drift detection
#[derive(Debug, Clone)]
pub struct DriftDetectionConfig {
    /// Detection method to use
    pub detection_method: DriftDetectionMethod,
    /// Significance level for statistical tests
    pub alpha: f64,
    /// Window size for windowed detection methods
    pub window_size: usize,
    /// Warning threshold (fraction of alpha)
    pub warning_threshold: f64,
    /// Minimum samples required for detection
    pub min_samples: usize,
    /// Whether to detect multivariate drift
    pub multivariate: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for DriftDetectionConfig {
    fn default() -> Self {
        Self {
            detection_method: DriftDetectionMethod::KolmogorovSmirnov,
            alpha: 0.05,
            window_size: 100,
            warning_threshold: 0.5,
            min_samples: 30,
            multivariate: false,
            random_state: None,
        }
    }
}

/// Drift detection methods
#[derive(Debug, Clone)]
pub enum DriftDetectionMethod {
    /// Kolmogorov-Smirnov test for univariate drift
    KolmogorovSmirnov,
    /// Anderson-Darling test
    AndersonDarling,
    /// Mann-Whitney U test
    MannWhitney,
    /// Permutation test
    Permutation,
    /// Population Stability Index (PSI)
    PopulationStabilityIndex,
    /// Maximum Mean Discrepancy (MMD)
    MaximumMeanDiscrepancy,
    /// ADWIN (Adaptive Windowing)
    ADWIN,
    /// Page-Hinkley test
    PageHinkley,
    /// Drift Detection Method (DDM)
    DDM,
    /// Early Drift Detection Method (EDDM)
    EDDM,
}

/// Results from drift detection
#[derive(Debug, Clone)]
pub struct DriftDetectionResult {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Whether warning threshold was exceeded
    pub warning_detected: bool,
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value for statistical tests
    pub p_value: Option<f64>,
    /// Threshold used for detection
    pub threshold: f64,
    /// Drift score (higher = more drift)
    pub drift_score: f64,
    /// Per-feature drift scores
    pub feature_drift_scores: Option<Vec<f64>>,
    /// Detailed statistics
    pub statistics: DriftStatistics,
}

/// Detailed drift statistics
#[derive(Debug, Clone)]
pub struct DriftStatistics {
    /// Number of reference samples
    pub n_reference: usize,
    /// Number of current samples
    pub n_current: usize,
    /// Number of features analyzed
    pub n_features: usize,
    /// Drift magnitude estimate
    pub drift_magnitude: f64,
    /// Confidence in drift detection
    pub confidence: f64,
    /// Time since last drift (if applicable)
    pub time_since_drift: Option<usize>,
}

/// Drift detector for monitoring data distribution changes
#[derive(Debug, Clone)]
pub struct DriftDetector {
    config: DriftDetectionConfig,
    reference_data: Option<Array2<f64>>,
    current_window: Vec<Array1<f64>>,
    drift_history: Vec<DriftDetectionResult>,
    last_drift_time: Option<usize>,
}

impl DriftDetector {
    pub fn new(config: DriftDetectionConfig) -> Self {
        Self {
            config,
            reference_data: None,
            current_window: Vec::new(),
            drift_history: Vec::new(),
            last_drift_time: None,
        }
    }

    /// Set reference data for drift detection
    pub fn set_reference(&mut self, reference_data: Array2<f64>) {
        self.reference_data = Some(reference_data);
    }

    /// Detect drift in new data
    pub fn detect_drift(&mut self, current_data: &Array2<f64>) -> Result<DriftDetectionResult> {
        if self.reference_data.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "drift detection".to_string(),
            });
        }

        let reference = self.reference_data.as_ref().unwrap();

        if reference.ncols() != current_data.ncols() {
            return Err(SklearsError::InvalidInput(
                "Reference and current data must have same number of features".to_string(),
            ));
        }

        let result = match self.config.detection_method {
            DriftDetectionMethod::KolmogorovSmirnov => {
                self.kolmogorov_smirnov_test(reference, current_data)?
            }
            DriftDetectionMethod::AndersonDarling => {
                self.anderson_darling_test(reference, current_data)?
            }
            DriftDetectionMethod::MannWhitney => self.mann_whitney_test(reference, current_data)?,
            DriftDetectionMethod::Permutation => self.permutation_test(reference, current_data)?,
            DriftDetectionMethod::PopulationStabilityIndex => {
                self.population_stability_index(reference, current_data)?
            }
            DriftDetectionMethod::MaximumMeanDiscrepancy => {
                self.maximum_mean_discrepancy(reference, current_data)?
            }
            DriftDetectionMethod::ADWIN => self.adwin_test(current_data)?,
            DriftDetectionMethod::PageHinkley => self.page_hinkley_test(current_data)?,
            DriftDetectionMethod::DDM => self.ddm_test(current_data)?,
            DriftDetectionMethod::EDDM => self.eddm_test(current_data)?,
        };

        self.drift_history.push(result.clone());

        if result.drift_detected {
            self.last_drift_time = Some(self.drift_history.len());
        }

        Ok(result)
    }

    /// Kolmogorov-Smirnov test for drift detection
    fn kolmogorov_smirnov_test(
        &self,
        reference: &Array2<f64>,
        current: &Array2<f64>,
    ) -> Result<DriftDetectionResult> {
        let n_features = reference.ncols();
        let mut feature_scores = Vec::new();
        let mut max_statistic: f64 = 0.0;
        let mut min_p_value: f64 = 1.0;

        for feature_idx in 0..n_features {
            let ref_feature: Vec<f64> = (0..reference.nrows())
                .map(|i| reference[[i, feature_idx]])
                .collect();
            let cur_feature: Vec<f64> = (0..current.nrows())
                .map(|i| current[[i, feature_idx]])
                .collect();

            let (statistic, p_value) = self.ks_test(&ref_feature, &cur_feature);
            feature_scores.push(statistic);
            max_statistic = max_statistic.max(statistic);
            min_p_value = min_p_value.min(p_value);
        }

        let drift_detected = min_p_value < self.config.alpha;
        let warning_detected =
            min_p_value < self.config.alpha * (1.0 + self.config.warning_threshold);

        let statistics = DriftStatistics {
            n_reference: reference.nrows(),
            n_current: current.nrows(),
            n_features,
            drift_magnitude: max_statistic,
            confidence: 1.0 - min_p_value,
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: max_statistic,
            p_value: Some(min_p_value),
            threshold: self.config.alpha,
            drift_score: max_statistic,
            feature_drift_scores: Some(feature_scores),
            statistics,
        })
    }

    /// Anderson-Darling test for drift detection
    fn anderson_darling_test(
        &self,
        reference: &Array2<f64>,
        current: &Array2<f64>,
    ) -> Result<DriftDetectionResult> {
        // Simplified Anderson-Darling test implementation
        let n_features = reference.ncols();
        let mut feature_scores = Vec::new();
        let mut max_statistic: f64 = 0.0;

        for feature_idx in 0..n_features {
            let ref_feature: Vec<f64> = (0..reference.nrows())
                .map(|i| reference[[i, feature_idx]])
                .collect();
            let cur_feature: Vec<f64> = (0..current.nrows())
                .map(|i| current[[i, feature_idx]])
                .collect();

            let statistic = self.anderson_darling_statistic(&ref_feature, &cur_feature);
            feature_scores.push(statistic);
            max_statistic = max_statistic.max(statistic);
        }

        // Approximate threshold for Anderson-Darling
        let threshold = 2.492; // Critical value for alpha = 0.05
        let drift_detected = max_statistic > threshold;
        let warning_detected = max_statistic > threshold * self.config.warning_threshold;

        let statistics = DriftStatistics {
            n_reference: reference.nrows(),
            n_current: current.nrows(),
            n_features,
            drift_magnitude: max_statistic,
            confidence: if drift_detected { 0.95 } else { 0.5 },
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: max_statistic,
            p_value: None,
            threshold,
            drift_score: max_statistic,
            feature_drift_scores: Some(feature_scores),
            statistics,
        })
    }

    /// Mann-Whitney U test for drift detection
    fn mann_whitney_test(
        &self,
        reference: &Array2<f64>,
        current: &Array2<f64>,
    ) -> Result<DriftDetectionResult> {
        let n_features = reference.ncols();
        let mut feature_scores = Vec::new();
        let mut max_statistic: f64 = 0.0;
        let mut min_p_value: f64 = 1.0;

        for feature_idx in 0..n_features {
            let ref_feature: Vec<f64> = (0..reference.nrows())
                .map(|i| reference[[i, feature_idx]])
                .collect();
            let cur_feature: Vec<f64> = (0..current.nrows())
                .map(|i| current[[i, feature_idx]])
                .collect();

            let (u_statistic, p_value) = self.mann_whitney_u_test(&ref_feature, &cur_feature);
            let normalized_statistic = u_statistic / (ref_feature.len() * cur_feature.len()) as f64;

            feature_scores.push(normalized_statistic);
            max_statistic = max_statistic.max(normalized_statistic);
            min_p_value = min_p_value.min(p_value);
        }

        let drift_detected = min_p_value < self.config.alpha;
        let warning_detected =
            min_p_value < self.config.alpha * (1.0 + self.config.warning_threshold);

        let statistics = DriftStatistics {
            n_reference: reference.nrows(),
            n_current: current.nrows(),
            n_features,
            drift_magnitude: max_statistic,
            confidence: 1.0 - min_p_value,
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: max_statistic,
            p_value: Some(min_p_value),
            threshold: self.config.alpha,
            drift_score: max_statistic,
            feature_drift_scores: Some(feature_scores),
            statistics,
        })
    }

    /// Permutation test for drift detection
    fn permutation_test(
        &self,
        reference: &Array2<f64>,
        current: &Array2<f64>,
    ) -> Result<DriftDetectionResult> {
        let n_permutations = 1000;
        let observed_statistic = self.calculate_permutation_statistic(reference, current);

        let mut permutation_statistics = Vec::new();
        let combined_data = self.combine_data(reference, current);
        let n_ref = reference.nrows();

        for _ in 0..n_permutations {
            let (perm_ref, perm_cur) = self.random_permutation_split(&combined_data, n_ref);
            let perm_statistic = self.calculate_permutation_statistic(&perm_ref, &perm_cur);
            permutation_statistics.push(perm_statistic);
        }

        // Calculate p-value
        let extreme_count = permutation_statistics
            .iter()
            .filter(|&&stat| stat >= observed_statistic)
            .count();
        let p_value = extreme_count as f64 / n_permutations as f64;

        let drift_detected = p_value < self.config.alpha;
        let warning_detected = p_value < self.config.alpha * (1.0 + self.config.warning_threshold);

        let statistics = DriftStatistics {
            n_reference: reference.nrows(),
            n_current: current.nrows(),
            n_features: reference.ncols(),
            drift_magnitude: observed_statistic,
            confidence: 1.0 - p_value,
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: observed_statistic,
            p_value: Some(p_value),
            threshold: self.config.alpha,
            drift_score: observed_statistic,
            feature_drift_scores: None,
            statistics,
        })
    }

    /// Population Stability Index (PSI) for drift detection
    fn population_stability_index(
        &self,
        reference: &Array2<f64>,
        current: &Array2<f64>,
    ) -> Result<DriftDetectionResult> {
        let n_features = reference.ncols();
        let n_bins = 10;
        let mut feature_scores = Vec::new();
        let mut total_psi = 0.0;

        for feature_idx in 0..n_features {
            let ref_feature: Vec<f64> = (0..reference.nrows())
                .map(|i| reference[[i, feature_idx]])
                .collect();
            let cur_feature: Vec<f64> = (0..current.nrows())
                .map(|i| current[[i, feature_idx]])
                .collect();

            let psi = self.calculate_psi(&ref_feature, &cur_feature, n_bins);
            feature_scores.push(psi);
            total_psi += psi;
        }

        let avg_psi = total_psi / n_features as f64;

        // PSI thresholds: <0.1 (no drift), 0.1-0.2 (minor), >0.2 (major)
        let drift_detected = avg_psi > 0.2;
        let warning_detected = avg_psi > 0.1;

        let statistics = DriftStatistics {
            n_reference: reference.nrows(),
            n_current: current.nrows(),
            n_features,
            drift_magnitude: avg_psi,
            confidence: if drift_detected { 0.8 } else { 0.5 },
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: avg_psi,
            p_value: None,
            threshold: 0.2,
            drift_score: avg_psi,
            feature_drift_scores: Some(feature_scores),
            statistics,
        })
    }

    /// Maximum Mean Discrepancy (MMD) test
    fn maximum_mean_discrepancy(
        &self,
        reference: &Array2<f64>,
        current: &Array2<f64>,
    ) -> Result<DriftDetectionResult> {
        let mmd_statistic = self.calculate_mmd(reference, current);

        // Use permutation test to get p-value
        let n_permutations = 1000;
        let mut permutation_mmds = Vec::new();
        let combined_data = self.combine_data(reference, current);
        let n_ref = reference.nrows();

        for _ in 0..n_permutations {
            let (perm_ref, perm_cur) = self.random_permutation_split(&combined_data, n_ref);
            let perm_mmd = self.calculate_mmd(&perm_ref, &perm_cur);
            permutation_mmds.push(perm_mmd);
        }

        let extreme_count = permutation_mmds
            .iter()
            .filter(|&&mmd| mmd >= mmd_statistic)
            .count();
        let p_value = extreme_count as f64 / n_permutations as f64;

        let drift_detected = p_value < self.config.alpha;
        let warning_detected = p_value < self.config.alpha * (1.0 + self.config.warning_threshold);

        let statistics = DriftStatistics {
            n_reference: reference.nrows(),
            n_current: current.nrows(),
            n_features: reference.ncols(),
            drift_magnitude: mmd_statistic,
            confidence: 1.0 - p_value,
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: mmd_statistic,
            p_value: Some(p_value),
            threshold: self.config.alpha,
            drift_score: mmd_statistic,
            feature_drift_scores: None,
            statistics,
        })
    }

    /// ADWIN (Adaptive Windowing) drift detection
    fn adwin_test(&mut self, current: &Array2<f64>) -> Result<DriftDetectionResult> {
        // Simplified ADWIN implementation
        // In practice, this would maintain adaptive windows

        let n_samples = current.nrows();
        let avg_performance = self.calculate_average_performance(current);

        // Add to current window
        for i in 0..n_samples {
            let sample = current.row(i).to_owned();
            self.current_window.push(sample);
        }

        // Keep window size manageable
        if self.current_window.len() > self.config.window_size * 2 {
            let excess = self.current_window.len() - self.config.window_size;
            self.current_window.drain(0..excess);
        }

        let drift_detected =
            self.current_window.len() >= self.config.min_samples && avg_performance < 0.5; // Simplified threshold

        let statistics = DriftStatistics {
            n_reference: 0,
            n_current: current.nrows(),
            n_features: current.ncols(),
            drift_magnitude: 1.0 - avg_performance,
            confidence: if drift_detected { 0.8 } else { 0.5 },
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected: avg_performance < 0.7,
            test_statistic: 1.0 - avg_performance,
            p_value: None,
            threshold: 0.5,
            drift_score: 1.0 - avg_performance,
            feature_drift_scores: None,
            statistics,
        })
    }

    /// Page-Hinkley test for drift detection
    fn page_hinkley_test(&self, current: &Array2<f64>) -> Result<DriftDetectionResult> {
        // Simplified Page-Hinkley test
        let avg_performance = self.calculate_average_performance(current);
        let threshold = 3.0; // Typical threshold

        let cumulative_sum = (0.5 - avg_performance) * current.nrows() as f64;
        let drift_detected = cumulative_sum.abs() > threshold;

        let statistics = DriftStatistics {
            n_reference: 0,
            n_current: current.nrows(),
            n_features: current.ncols(),
            drift_magnitude: cumulative_sum.abs(),
            confidence: if drift_detected { 0.8 } else { 0.5 },
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected: cumulative_sum.abs() > threshold * 0.7,
            test_statistic: cumulative_sum.abs(),
            p_value: None,
            threshold,
            drift_score: cumulative_sum.abs(),
            feature_drift_scores: None,
            statistics,
        })
    }

    /// DDM (Drift Detection Method)
    fn ddm_test(&self, current: &Array2<f64>) -> Result<DriftDetectionResult> {
        // Simplified DDM implementation
        let error_rate = 1.0 - self.calculate_average_performance(current);
        let std_error = (error_rate * (1.0 - error_rate) / current.nrows() as f64).sqrt();

        let warning_threshold = error_rate + 2.0 * std_error;
        let drift_threshold = error_rate + 3.0 * std_error;

        let drift_detected = error_rate > drift_threshold;
        let warning_detected = error_rate > warning_threshold;

        let statistics = DriftStatistics {
            n_reference: 0,
            n_current: current.nrows(),
            n_features: current.ncols(),
            drift_magnitude: error_rate,
            confidence: if drift_detected {
                0.99
            } else if warning_detected {
                0.95
            } else {
                0.5
            },
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected,
            test_statistic: error_rate,
            p_value: None,
            threshold: drift_threshold,
            drift_score: error_rate,
            feature_drift_scores: None,
            statistics,
        })
    }

    /// EDDM (Early Drift Detection Method)
    fn eddm_test(&self, current: &Array2<f64>) -> Result<DriftDetectionResult> {
        // Simplified EDDM implementation
        let avg_performance = self.calculate_average_performance(current);
        let distance_between_errors = 1.0 / (1.0 - avg_performance + 1e-8);

        let threshold = 0.95;
        let drift_detected = avg_performance < threshold;

        let statistics = DriftStatistics {
            n_reference: 0,
            n_current: current.nrows(),
            n_features: current.ncols(),
            drift_magnitude: 1.0 - avg_performance,
            confidence: if drift_detected { 0.8 } else { 0.5 },
            time_since_drift: self.time_since_drift(),
        };

        Ok(DriftDetectionResult {
            drift_detected,
            warning_detected: avg_performance < 0.98,
            test_statistic: 1.0 - avg_performance,
            p_value: None,
            threshold: 1.0 - threshold,
            drift_score: 1.0 - avg_performance,
            feature_drift_scores: None,
            statistics,
        })
    }

    // Helper methods

    /// Kolmogorov-Smirnov test implementation
    fn ks_test(&self, sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
        let mut combined: Vec<(f64, usize)> = sample1.iter().map(|&x| (x, 0)).collect();
        combined.extend(sample2.iter().map(|&x| (x, 1)));
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        let mut cdf1 = 0.0;
        let mut cdf2 = 0.0;
        let mut max_diff: f64 = 0.0;

        for (_, group) in combined {
            if group == 0 {
                cdf1 += 1.0 / n1;
            } else {
                cdf2 += 1.0 / n2;
            }
            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }

        // Approximate p-value calculation
        let ks_statistic = max_diff;
        let en = (n1 * n2 / (n1 + n2)).sqrt();
        let lambda = en * ks_statistic;
        let p_value = 2.0 * (-2.0 * lambda * lambda).exp();

        (ks_statistic, p_value.max(0.0).min(1.0))
    }

    /// Anderson-Darling statistic calculation
    fn anderson_darling_statistic(&self, sample1: &[f64], sample2: &[f64]) -> f64 {
        // Simplified implementation
        let mut combined: Vec<f64> = sample1.iter().chain(sample2.iter()).cloned().collect();
        combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        let n = n1 + n2;

        let mut h = 0.0;
        let mut prev_val = f64::NEG_INFINITY;
        let i = 0.0;

        for &val in &combined {
            if val != prev_val {
                let count1 = sample1.iter().filter(|&&x| x <= val).count() as f64;
                let count2 = sample2.iter().filter(|&&x| x <= val).count() as f64;

                let l = count1 + count2;
                if l > 0.0 && l < n {
                    h += (count1 / n1 - count2 / n2).powi(2) / (l * (n - l));
                }
                prev_val = val;
            }
        }

        n1 * n2 * h / n
    }

    /// Mann-Whitney U test implementation
    fn mann_whitney_u_test(&self, sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
        let mut combined: Vec<(f64, usize)> =
            sample1.iter().enumerate().map(|(i, &x)| (x, 0)).collect();
        combined.extend(sample2.iter().enumerate().map(|(i, &x)| (x, 1)));
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut rank_sum1 = 0.0;
        for (rank, (_, group)) in combined.iter().enumerate() {
            if *group == 0 {
                rank_sum1 += (rank + 1) as f64;
            }
        }

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        let u1 = rank_sum1 - n1 * (n1 + 1.0) / 2.0;
        let u2 = n1 * n2 - u1;
        let u_statistic = u1.min(u2);

        // Approximate p-value using normal approximation
        let mu = n1 * n2 / 2.0;
        let sigma = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
        let z = (u_statistic - mu).abs() / sigma;
        let p_value = 2.0 * (1.0 - self.normal_cdf(z));

        (u_statistic, p_value.max(0.0).min(1.0))
    }

    /// Calculate permutation test statistic
    fn calculate_permutation_statistic(
        &self,
        ref_data: &Array2<f64>,
        cur_data: &Array2<f64>,
    ) -> f64 {
        // Use mean difference as statistic
        let ref_mean = self.calculate_mean(ref_data);
        let cur_mean = self.calculate_mean(cur_data);
        (ref_mean - cur_mean).abs()
    }

    /// Combine two datasets
    fn combine_data(&self, data1: &Array2<f64>, data2: &Array2<f64>) -> Array2<f64> {
        let n_rows = data1.nrows() + data2.nrows();
        let n_cols = data1.ncols();
        let mut combined = Array2::zeros((n_rows, n_cols));

        // Copy data1
        for i in 0..data1.nrows() {
            for j in 0..n_cols {
                combined[[i, j]] = data1[[i, j]];
            }
        }

        // Copy data2
        for i in 0..data2.nrows() {
            for j in 0..n_cols {
                combined[[data1.nrows() + i, j]] = data2[[i, j]];
            }
        }

        combined
    }

    /// Random permutation split
    fn random_permutation_split(
        &self,
        data: &Array2<f64>,
        n_first: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};

        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut indices: Vec<usize> = (0..data.nrows()).collect();
        indices.shuffle(&mut rng);

        let n_cols = data.ncols();
        let mut first = Array2::zeros((n_first, n_cols));
        let mut second = Array2::zeros((data.nrows() - n_first, n_cols));

        for (i, &idx) in indices[..n_first].iter().enumerate() {
            for j in 0..n_cols {
                first[[i, j]] = data[[idx, j]];
            }
        }

        for (i, &idx) in indices[n_first..].iter().enumerate() {
            for j in 0..n_cols {
                second[[i, j]] = data[[idx, j]];
            }
        }

        (first, second)
    }

    /// Calculate Population Stability Index
    fn calculate_psi(&self, reference: &[f64], current: &[f64], n_bins: usize) -> f64 {
        // Calculate bin edges based on reference data
        let mut ref_sorted = reference.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let mut bin_edges = Vec::new();
        for i in 0..=n_bins {
            let quantile = i as f64 / n_bins as f64;
            let idx = ((ref_sorted.len() - 1) as f64 * quantile) as usize;
            bin_edges.push(ref_sorted[idx.min(ref_sorted.len() - 1)]);
        }

        // Calculate bin counts
        let ref_counts = self.calculate_bin_counts(reference, &bin_edges);
        let cur_counts = self.calculate_bin_counts(current, &bin_edges);

        // Calculate PSI
        let mut psi = 0.0;
        for i in 0..n_bins {
            let ref_prop = ref_counts[i] / reference.len() as f64;
            let cur_prop = cur_counts[i] / current.len() as f64;

            if ref_prop > 0.0 && cur_prop > 0.0 {
                psi += (cur_prop - ref_prop) * (cur_prop / ref_prop).ln();
            }
        }

        psi
    }

    /// Calculate bin counts
    fn calculate_bin_counts(&self, data: &[f64], bin_edges: &[f64]) -> Vec<f64> {
        let n_bins = bin_edges.len() - 1;
        let mut counts = vec![0.0; n_bins];

        for &value in data {
            for i in 0..n_bins {
                if value >= bin_edges[i] && value < bin_edges[i + 1] {
                    counts[i] += 1.0;
                    break;
                } else if i == n_bins - 1 && value >= bin_edges[i] {
                    counts[i] += 1.0;
                    break;
                }
            }
        }

        counts
    }

    /// Calculate Maximum Mean Discrepancy
    fn calculate_mmd(&self, data1: &Array2<f64>, data2: &Array2<f64>) -> f64 {
        // Simplified MMD with linear kernel
        let mean1 = self.calculate_mean(data1);
        let mean2 = self.calculate_mean(data2);
        (mean1 - mean2).abs()
    }

    /// Calculate mean of dataset
    fn calculate_mean(&self, data: &Array2<f64>) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                sum += data[[i, j]];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Calculate average performance (simplified)
    fn calculate_average_performance(&self, data: &Array2<f64>) -> f64 {
        // Simplified performance calculation
        let mean = self.calculate_mean(data);
        // Normalize to [0, 1] range (simplified)
        (mean + 1.0) / 2.0
    }

    /// Time since last drift
    fn time_since_drift(&self) -> Option<usize> {
        self.last_drift_time
            .map(|last| self.drift_history.len() - last)
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_drift_detection() {
        let config = DriftDetectionConfig::default();
        let mut detector = DriftDetector::new(config);

        // Create reference data
        let mut reference = Array2::zeros((100, 2));
        for i in 0..100 {
            reference[[i, 0]] = i as f64 / 100.0;
            reference[[i, 1]] = (i as f64 / 100.0).sin();
        }
        detector.set_reference(reference);

        // Create similar current data (no drift) - subsample from same distribution
        let mut current = Array2::zeros((50, 2));
        for i in 0..50 {
            let idx = i * 2; // Sample every other point from reference to avoid exact duplication
            current[[i, 0]] = idx as f64 / 100.0;
            current[[i, 1]] = (idx as f64 / 100.0).sin();
        }

        let result = detector.detect_drift(&current).unwrap();
        assert!(
            !result.drift_detected,
            "Should not detect drift in similar data"
        );
    }

    #[test]
    fn test_psi_drift_detection() {
        let config = DriftDetectionConfig {
            detection_method: DriftDetectionMethod::PopulationStabilityIndex,
            ..Default::default()
        };
        let mut detector = DriftDetector::new(config);

        // Create reference data
        let mut reference = Array2::zeros((100, 1));
        for i in 0..100 {
            reference[[i, 0]] = i as f64 / 100.0;
        }
        detector.set_reference(reference);

        // Create shifted current data (drift)
        let mut current = Array2::zeros((50, 1));
        for i in 0..50 {
            current[[i, 0]] = (i as f64 / 50.0) + 0.5; // Shifted distribution
        }

        let result = detector.detect_drift(&current).unwrap();
        // PSI should detect this shift
        assert!(result.drift_score > 0.1, "Should detect distribution shift");
    }
}
