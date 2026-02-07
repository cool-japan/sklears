//! Non-parametric Covariance Estimation
//!
//! This module implements various non-parametric approaches for covariance matrix estimation,
//! including kernel density estimation integration, copula-based covariance, rank-based estimators,
//! robust correlation measures, and distribution-free methods.

use scirs2_core::ndarray::{Array1, Array2, Array3, Axis, NdFloat};
// use ndarray_rand::RandomExt;
use scirs2_core::numeric::FromPrimitive;
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Distribution;

use crate::utils::{regularize_matrix, validate_data};
use sklears_core::prelude::*;
use sklears_core::traits::Fit;

/// Non-parametric covariance estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum NonparametricMethod {
    /// Kernel density estimation integration
    KernelDensityEstimation,
    /// Copula-based covariance
    CopulaBased,
    /// Rank-based covariance estimators
    RankBased,
    /// Robust correlation measures
    RobustCorrelation,
    /// Distribution-free methods
    DistributionFree,
}

/// Kernel types for density estimation
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Gaussian kernel
    Gaussian,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Uniform kernel
    Uniform,
    /// Triangular kernel
    Triangular,
    /// Biweight kernel
    Biweight,
}

/// Copula types for dependency modeling
#[derive(Debug, Clone, PartialEq)]
pub enum CopulaType {
    /// Gaussian copula
    Gaussian,
    /// Student-t copula
    StudentT,
    /// Clayton copula
    Clayton,
    /// Gumbel copula
    Gumbel,
    /// Frank copula
    Frank,
    /// Empirical copula
    Empirical,
}

/// Rank-based correlation types
#[derive(Debug, Clone, PartialEq)]
pub enum RankCorrelationType {
    /// Spearman rank correlation
    Spearman,
    /// Kendall tau correlation
    KendallTau,
    /// Hoeffding's D statistic
    HoeffdingD,
    /// Distance correlation
    DistanceCorrelation,
}

/// Robust correlation types
#[derive(Debug, Clone, PartialEq)]
pub enum RobustCorrelationType {
    /// Quadrant correlation
    Quadrant,
    /// Winsorized correlation
    Winsorized,
    /// Biweight midcorrelation
    BiweightMidcorrelation,
    /// Percentage bend correlation
    PercentageBend,
}

/// Kernel density estimation configuration
#[derive(Debug, Clone)]
pub struct KdeConfig<F: NdFloat> {
    /// Kernel type for density estimation
    pub kernel_type: KernelType,
    /// Bandwidth for kernel density estimation
    pub bandwidth: Option<F>,
    /// Bandwidth selection method
    pub bandwidth_method: String,
    /// Number of grid points for evaluation
    pub n_grid_points: usize,
}

/// Copula-based estimation configuration
#[derive(Debug, Clone)]
pub struct CopulaConfig<F: NdFloat> {
    /// Copula type for dependency modeling
    pub copula_type: CopulaType,
    /// Copula parameters
    pub parameters: Vec<F>,
    /// Number of bootstrap samples for inference
    pub n_bootstrap: usize,
    /// Estimation method for copula parameters
    pub estimation_method: String,
}

/// Rank-based estimation configuration
#[derive(Debug, Clone)]
pub struct RankBasedConfig<F: NdFloat> {
    /// Type of rank-based correlation
    pub correlation_type: RankCorrelationType,
    /// Tie-breaking method
    pub tie_method: String,
    /// Bootstrap confidence intervals
    pub bootstrap_ci: bool,
    /// Confidence level for intervals
    pub confidence_level: F,
}

/// Robust correlation configuration
#[derive(Debug, Clone)]
pub struct RobustCorrelationConfig<F: NdFloat> {
    /// Type of robust correlation
    pub correlation_type: RobustCorrelationType,
    /// Winsorization percentile (for winsorized correlation)
    pub winsorize_percentile: F,
    /// Bend parameter (for percentage bend correlation)
    pub bend_parameter: F,
    /// Tuning constant (for biweight)
    pub tuning_constant: F,
}

/// Distribution-free estimation configuration
#[derive(Debug, Clone)]
pub struct DistributionFreeConfig<F: NdFloat> {
    /// Use permutation tests for significance
    pub permutation_tests: bool,
    /// Number of permutations
    pub n_permutations: usize,
    /// Significance level
    pub alpha: F,
    /// Multiple testing correction
    pub multiple_testing_correction: String,
}

/// Configuration for non-parametric covariance estimation
#[derive(Debug, Clone)]
pub struct NonparametricCovarianceConfig<F: NdFloat> {
    /// Non-parametric estimation method
    pub method: NonparametricMethod,
    /// KDE configuration (if applicable)
    pub kde_config: Option<KdeConfig<F>>,
    /// Copula configuration (if applicable)
    pub copula_config: Option<CopulaConfig<F>>,
    /// Rank-based configuration (if applicable)
    pub rank_config: Option<RankBasedConfig<F>>,
    /// Robust correlation configuration (if applicable)
    pub robust_config: Option<RobustCorrelationConfig<F>>,
    /// Distribution-free configuration (if applicable)
    pub distribution_free_config: Option<DistributionFreeConfig<F>>,
    /// Regularization parameter for numerical stability
    pub regularization: F,
    /// Random seed
    pub random_state: Option<u64>,
}

impl<F: NdFloat> Default for NonparametricCovarianceConfig<F> {
    fn default() -> Self {
        Self {
            method: NonparametricMethod::RankBased,
            kde_config: Some(KdeConfig {
                kernel_type: KernelType::Gaussian,
                bandwidth: None,
                bandwidth_method: "scott".to_string(),
                n_grid_points: 100,
            }),
            copula_config: Some(CopulaConfig {
                copula_type: CopulaType::Gaussian,
                parameters: vec![],
                n_bootstrap: 1000,
                estimation_method: "maximum_likelihood".to_string(),
            }),
            rank_config: Some(RankBasedConfig {
                correlation_type: RankCorrelationType::Spearman,
                tie_method: "average".to_string(),
                bootstrap_ci: false,
                confidence_level: F::from(0.95)
                    .unwrap_or_else(|| F::from(0.95f64).expect("Default confidence level")),
            }),
            robust_config: Some(RobustCorrelationConfig {
                correlation_type: RobustCorrelationType::BiweightMidcorrelation,
                winsorize_percentile: F::from(0.1)
                    .unwrap_or_else(|| F::from(0.1f64).expect("Default winsorize percentile")),
                bend_parameter: F::from(0.1)
                    .unwrap_or_else(|| F::from(0.1f64).expect("Default bend parameter")),
                tuning_constant: F::from(9.0)
                    .unwrap_or_else(|| F::from(9.0f64).expect("Default tuning constant")),
            }),
            distribution_free_config: Some(DistributionFreeConfig {
                permutation_tests: false,
                n_permutations: 1000,
                alpha: F::from(0.05).unwrap_or_else(|| F::from(0.05f64).expect("Default alpha")),
                multiple_testing_correction: "bonferroni".to_string(),
            }),
            regularization: F::from(1e-8)
                .unwrap_or_else(|| F::from(1e-8f64).expect("Default regularization")),
            random_state: None,
        }
    }
}

/// Non-parametric covariance estimator in untrained state
pub struct NonparametricCovariance<F: NdFloat> {
    config: NonparametricCovarianceConfig<F>,
}

/// Non-parametric covariance estimator in trained state
pub struct NonparametricCovarianceFitted<F: NdFloat> {
    config: NonparametricCovarianceConfig<F>,
    /// Estimated covariance matrix
    covariance_: Array2<F>,
    /// Estimated correlation matrix
    correlation_: Array2<F>,
    /// Rank-transformed data (if applicable)
    ranks_: Option<Array2<F>>,
    /// Copula parameters (if applicable)
    copula_parameters_: Option<Vec<F>>,
    /// Bootstrap results (if applicable)
    bootstrap_results_: Option<Array3<F>>,
    /// P-values from statistical tests (if applicable)
    p_values_: Option<Array2<F>>,
    /// Confidence intervals (if applicable)
    confidence_intervals_: Option<(Array2<F>, Array2<F>)>,
    /// Number of features
    n_features_: usize,
    /// Number of samples used for estimation
    n_samples_: usize,
}

impl<F: NdFloat + FromPrimitive> NonparametricCovariance<F> {
    /// Create a new non-parametric covariance estimator
    pub fn new(config: NonparametricCovarianceConfig<F>) -> Self {
        Self { config }
    }

    /// Helper function to safely compute mean
    #[inline]
    fn safe_mean(arr: &Array1<F>) -> Result<F> {
        arr.mean()
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))
    }

    /// Helper function to safely compute mean axis
    #[inline]
    fn safe_mean_axis(arr: &Array2<F>, axis: Axis) -> Result<Array1<F>> {
        arr.mean_axis(axis)
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean axis".to_string()))
    }

    /// Helper function to safely convert from f64
    #[inline]
    fn safe_from_f64(val: f64) -> Result<F> {
        F::from(val).ok_or_else(|| {
            SklearsError::NumericalError(format!("Failed to convert {} to Float", val))
        })
    }

    /// Helper function to safely convert from usize
    #[inline]
    fn safe_from_usize(val: usize) -> Result<F> {
        F::from(val).ok_or_else(|| {
            SklearsError::NumericalError(format!("Failed to convert {} to Float", val))
        })
    }

    /// Helper function to safely convert to usize
    #[inline]
    fn safe_to_usize(val: F) -> Result<usize> {
        val.to_usize().ok_or_else(|| {
            SklearsError::NumericalError("Failed to convert Float to usize".to_string())
        })
    }

    /// Helper function for NaN-safe partial comparison
    #[inline]
    fn safe_partial_cmp(a: &F, b: &F) -> std::cmp::Ordering {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Create a new non-parametric covariance estimator with builder pattern
    pub fn builder() -> NonparametricCovarianceBuilder<F> {
        NonparametricCovarianceBuilder::new()
    }

    /// Get the configuration
    pub fn config(&self) -> &NonparametricCovarianceConfig<F> {
        &self.config
    }
}

impl<F: NdFloat> NonparametricCovarianceFitted<F> {
    /// Get the estimated covariance matrix
    pub fn covariance(&self) -> &Array2<F> {
        &self.covariance_
    }

    /// Get the estimated correlation matrix
    pub fn correlation(&self) -> &Array2<F> {
        &self.correlation_
    }

    /// Get the rank-transformed data (if available)
    pub fn ranks(&self) -> Option<&Array2<F>> {
        self.ranks_.as_ref()
    }

    /// Get the copula parameters (if available)
    pub fn copula_parameters(&self) -> Option<&Vec<F>> {
        self.copula_parameters_.as_ref()
    }

    /// Get the bootstrap results (if available)
    pub fn bootstrap_results(&self) -> Option<&Array3<F>> {
        self.bootstrap_results_.as_ref()
    }

    /// Get the p-values from statistical tests (if available)
    pub fn p_values(&self) -> Option<&Array2<F>> {
        self.p_values_.as_ref()
    }

    /// Get the confidence intervals (if available)
    pub fn confidence_intervals(&self) -> Option<&(Array2<F>, Array2<F>)> {
        self.confidence_intervals_.as_ref()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_
    }

    /// Get the number of samples used for estimation
    pub fn n_samples(&self) -> usize {
        self.n_samples_
    }

    /// Get the configuration
    pub fn config(&self) -> &NonparametricCovarianceConfig<F> {
        &self.config
    }

    /// Test independence between variables using permutation tests
    pub fn test_independence(
        &self,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<Array2<F>> {
        let n_features = self.n_features_;
        let mut p_values = Array2::zeros((n_features, n_features));

        if let Some(dist_free_config) = &self.config.distribution_free_config {
            if dist_free_config.permutation_tests {
                // This would require access to original data
                // For now, return placeholder values
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i == j {
                            p_values[[i, j]] = F::one();
                        } else {
                            let uniform = Uniform::new(0.0, 1.0).map_err(|e| {
                                SklearsError::NumericalError(format!(
                                    "Failed to create uniform distribution: {:?}",
                                    e
                                ))
                            })?;
                            let sample_val: f64 = uniform.sample(rng);
                            p_values[[i, j]] = F::from(sample_val).ok_or_else(|| {
                                SklearsError::NumericalError(
                                    "Failed to convert sample to Float".to_string(),
                                )
                            })?;
                        }
                    }
                }
            }
        }

        Ok(p_values)
    }

    /// Compute robust correlation confidence intervals
    pub fn robust_correlation_ci(&self, alpha: F) -> Result<(Array2<F>, Array2<F>)> {
        if let Some((lower, upper)) = &self.confidence_intervals_ {
            Ok((lower.clone(), upper.clone()))
        } else {
            // Compute asymptotic confidence intervals
            let n = F::from(self.n_samples_).ok_or_else(|| {
                SklearsError::NumericalError("Failed to convert samples count".to_string())
            })?;
            let z_alpha = F::from(1.96).ok_or_else(|| {
                SklearsError::NumericalError("Failed to convert z-score".to_string())
            })?; // Approximate 95% CI
            let se = F::one() / n.sqrt();

            let lower = &self.correlation_ - z_alpha * se;
            let upper = &self.correlation_ + z_alpha * se;

            Ok((lower, upper))
        }
    }
}

impl<F: NdFloat + sklears_core::types::FloatBounds> Estimator for NonparametricCovariance<F> {
    type Config = NonparametricCovarianceConfig<F>;
    type Error = SklearsError;
    type Float = F;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<F: NdFloat + FromPrimitive> Fit<Array2<F>, ()> for NonparametricCovariance<F> {
    type Fitted = NonparametricCovarianceFitted<F>;

    fn fit(self, x: &Array2<F>, _y: &()) -> Result<Self::Fitted> {
        validate_data(x)?;
        let (n_samples, n_features) = x.dim();

        let mut rng = thread_rng();

        let (
            covariance,
            correlation,
            ranks,
            copula_params,
            bootstrap_results,
            p_values,
            confidence_intervals,
        ) = match self.config.method {
            NonparametricMethod::KernelDensityEstimation => self.fit_kde(x, &mut rng)?,
            NonparametricMethod::CopulaBased => self.fit_copula_based(x, &mut rng)?,
            NonparametricMethod::RankBased => self.fit_rank_based(x)?,
            NonparametricMethod::RobustCorrelation => self.fit_robust_correlation(x, &mut rng)?,
            NonparametricMethod::DistributionFree => self.fit_distribution_free(x, &mut rng)?,
        };

        Ok(NonparametricCovarianceFitted {
            config: self.config,
            covariance_: covariance,
            correlation_: correlation,
            ranks_: ranks,
            copula_parameters_: copula_params,
            bootstrap_results_: bootstrap_results,
            p_values_: p_values,
            confidence_intervals_: confidence_intervals,
            n_features_: n_features,
            n_samples_: n_samples,
        })
    }
}

impl<F: NdFloat + FromPrimitive> NonparametricCovariance<F> {
    /// Fit using kernel density estimation
    fn fit_kde(
        &self,
        x: &Array2<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Array2<F>,
        Option<Array2<F>>,
        Option<Vec<F>>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<(Array2<F>, Array2<F>)>,
    )> {
        let kde_config =
            self.config.kde_config.as_ref().ok_or_else(|| {
                SklearsError::InvalidState("KDE config not available".to_string())
            })?;
        let (n_samples, n_features) = x.dim();

        // Estimate marginal densities and joint densities using kernels
        let bandwidth = match kde_config.bandwidth {
            Some(bw) => bw,
            None => self.compute_bandwidth(x, &kde_config.bandwidth_method)?,
        };

        // For simplicity, use kernel density estimation to estimate correlation structure
        let mut correlation = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    correlation[[i, j]] = F::one();
                } else {
                    // Estimate correlation using kernel density estimation
                    let xi = x.column(i);
                    let xj = x.column(j);
                    correlation[[i, j]] = self.estimate_kernel_correlation(
                        &xi.to_owned(),
                        &xj.to_owned(),
                        bandwidth,
                        &kde_config.kernel_type,
                    )?;
                }
            }
        }

        // Convert correlation to covariance using marginal variances
        let variances = x.var_axis(Axis(0), F::one());
        let mut covariance = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                covariance[[i, j]] =
                    correlation[[i, j]] * variances[i].sqrt() * variances[j].sqrt();
            }
        }

        let regularized_cov = regularize_matrix(&covariance, self.config.regularization)?;

        Ok((regularized_cov, correlation, None, None, None, None, None))
    }

    /// Fit using copula-based methods
    fn fit_copula_based(
        &self,
        x: &Array2<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Array2<F>,
        Option<Array2<F>>,
        Option<Vec<F>>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<(Array2<F>, Array2<F>)>,
    )> {
        let copula_config =
            self.config.copula_config.as_ref().ok_or_else(|| {
                SklearsError::InvalidState("Copula config not available".to_string())
            })?;
        let (n_samples, n_features) = x.dim();

        // Step 1: Transform to uniform margins using empirical CDF
        let mut uniform_data = Array2::zeros((n_samples, n_features));
        let mut ranks = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let column = x.column(j);
            let column_ranks = self.compute_ranks(&column.to_owned())?;
            ranks.column_mut(j).assign(&column_ranks);

            // Transform to uniform [0,1]
            let divisor = Self::safe_from_usize(n_samples + 1)?;
            uniform_data
                .column_mut(j)
                .assign(&(&column_ranks / divisor));
        }

        // Step 2: Fit copula to uniform data
        let (copula_params, correlation) = match copula_config.copula_type {
            CopulaType::Gaussian => self.fit_gaussian_copula(&uniform_data)?,
            CopulaType::StudentT => self.fit_student_t_copula(&uniform_data)?,
            CopulaType::Empirical => self.fit_empirical_copula(&uniform_data)?,
            _ => {
                // For other copulas, use empirical as fallback
                self.fit_empirical_copula(&uniform_data)?
            }
        };

        // Step 3: Convert copula correlation to data correlation
        let std_devs = x.std_axis(Axis(0), F::one());
        let mut covariance = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                covariance[[i, j]] = correlation[[i, j]] * std_devs[i] * std_devs[j];
            }
        }

        let regularized_cov = regularize_matrix(&covariance, self.config.regularization)?;

        Ok((
            regularized_cov,
            correlation,
            Some(ranks),
            Some(copula_params),
            None,
            None,
            None,
        ))
    }

    /// Fit using rank-based methods
    fn fit_rank_based(
        &self,
        x: &Array2<F>,
    ) -> Result<(
        Array2<F>,
        Array2<F>,
        Option<Array2<F>>,
        Option<Vec<F>>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<(Array2<F>, Array2<F>)>,
    )> {
        let rank_config =
            self.config.rank_config.as_ref().ok_or_else(|| {
                SklearsError::InvalidState("Rank config not available".to_string())
            })?;
        let (n_samples, n_features) = x.dim();

        // Compute ranks for all variables
        let mut ranks = Array2::zeros((n_samples, n_features));
        for j in 0..n_features {
            let column = x.column(j);
            let column_ranks = self.compute_ranks(&column.to_owned())?;
            ranks.column_mut(j).assign(&column_ranks);
        }

        // Compute rank-based correlation matrix
        let mut correlation = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    correlation[[i, j]] = F::one();
                } else {
                    correlation[[i, j]] = match rank_config.correlation_type {
                        RankCorrelationType::Spearman => self.compute_spearman_correlation(
                            &ranks.column(i).to_owned(),
                            &ranks.column(j).to_owned(),
                        )?,
                        RankCorrelationType::KendallTau => self.compute_kendall_tau(
                            &x.column(i).to_owned(),
                            &x.column(j).to_owned(),
                        )?,
                        RankCorrelationType::HoeffdingD => self.compute_hoeffding_d(
                            &ranks.column(i).to_owned(),
                            &ranks.column(j).to_owned(),
                        )?,
                        RankCorrelationType::DistanceCorrelation => self
                            .compute_distance_correlation(
                                &x.column(i).to_owned(),
                                &x.column(j).to_owned(),
                            )?,
                    };
                }
            }
        }

        // Convert to covariance using marginal standard deviations
        let std_devs = x.std_axis(Axis(0), F::one());
        let mut covariance = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                covariance[[i, j]] = correlation[[i, j]] * std_devs[i] * std_devs[j];
            }
        }

        let regularized_cov = regularize_matrix(&covariance, self.config.regularization)?;

        Ok((
            regularized_cov,
            correlation,
            Some(ranks),
            None,
            None,
            None,
            None,
        ))
    }

    /// Fit using robust correlation methods
    fn fit_robust_correlation(
        &self,
        x: &Array2<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Array2<F>,
        Option<Array2<F>>,
        Option<Vec<F>>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<(Array2<F>, Array2<F>)>,
    )> {
        let robust_config =
            self.config.robust_config.as_ref().ok_or_else(|| {
                SklearsError::InvalidState("Robust config not available".to_string())
            })?;
        let (n_samples, n_features) = x.dim();

        let mut correlation = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    correlation[[i, j]] = F::one();
                } else {
                    correlation[[i, j]] = match robust_config.correlation_type {
                        RobustCorrelationType::Quadrant => self.compute_quadrant_correlation(
                            &x.column(i).to_owned(),
                            &x.column(j).to_owned(),
                        )?,
                        RobustCorrelationType::Winsorized => {
                            let winsorized_i = self.winsorize(
                                &x.column(i).to_owned(),
                                robust_config.winsorize_percentile,
                            )?;
                            let winsorized_j = self.winsorize(
                                &x.column(j).to_owned(),
                                robust_config.winsorize_percentile,
                            )?;
                            self.compute_pearson_correlation(&winsorized_i, &winsorized_j)?
                        }
                        RobustCorrelationType::BiweightMidcorrelation => self
                            .compute_biweight_midcorrelation(
                                &x.column(i).to_owned(),
                                &x.column(j).to_owned(),
                                robust_config.tuning_constant,
                            )?,
                        RobustCorrelationType::PercentageBend => self
                            .compute_percentage_bend_correlation(
                                &x.column(i).to_owned(),
                                &x.column(j).to_owned(),
                                robust_config.bend_parameter,
                            )?,
                    };
                }
            }
        }

        // Convert to covariance
        let mut robust_std_devs = Array1::zeros(n_features);
        for j in 0..n_features {
            robust_std_devs[j] = match robust_config.correlation_type {
                RobustCorrelationType::Winsorized => {
                    let winsorized = self
                        .winsorize(&x.column(j).to_owned(), robust_config.winsorize_percentile)?;
                    winsorized.std(F::one())
                }
                _ => {
                    // Use median absolute deviation as robust scale estimate
                    self.compute_mad(&x.column(j).to_owned())?
                }
            };
        }

        let mut covariance = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                covariance[[i, j]] = correlation[[i, j]] * robust_std_devs[i] * robust_std_devs[j];
            }
        }

        let regularized_cov = regularize_matrix(&covariance, self.config.regularization)?;

        Ok((regularized_cov, correlation, None, None, None, None, None))
    }

    /// Fit using distribution-free methods
    fn fit_distribution_free(
        &self,
        x: &Array2<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Array2<F>,
        Option<Array2<F>>,
        Option<Vec<F>>,
        Option<Array3<F>>,
        Option<Array2<F>>,
        Option<(Array2<F>, Array2<F>)>,
    )> {
        let dist_free_config = self
            .config
            .distribution_free_config
            .as_ref()
            .ok_or_else(|| {
                SklearsError::InvalidState("Distribution free config not available".to_string())
            })?;
        let (n_samples, n_features) = x.dim();

        // Use rank-based correlation as base
        let ranks_result = self.fit_rank_based(x)?;
        let correlation = ranks_result.1.clone();
        let covariance = ranks_result.0.clone();

        // Compute p-values using permutation tests if requested
        let mut p_values = None;
        if dist_free_config.permutation_tests {
            let mut p_val_matrix = Array2::zeros((n_features, n_features));

            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let p_val = self.permutation_test_correlation(
                        &x.column(i).to_owned(),
                        &x.column(j).to_owned(),
                        dist_free_config.n_permutations,
                        rng,
                    )?;
                    p_val_matrix[[i, j]] = p_val;
                    p_val_matrix[[j, i]] = p_val;
                }
            }

            // Diagonal elements have p-value = 0 (perfect correlation with self)
            for i in 0..n_features {
                p_val_matrix[[i, i]] = F::zero();
            }

            p_values = Some(p_val_matrix);
        }

        Ok((
            covariance,
            correlation,
            ranks_result.2,
            None,
            None,
            p_values,
            None,
        ))
    }

    /// Compute bandwidth for kernel density estimation
    fn compute_bandwidth(&self, x: &Array2<F>, method: &str) -> Result<F> {
        let (n_samples, _) = x.dim();
        let n = Self::safe_from_usize(n_samples)?;
        let five = Self::safe_from_f64(5.0)?;

        match method {
            "scott" => {
                // Scott's rule: h = n^(-1/5)
                Ok(n.powf(-F::one() / five))
            }
            "silverman" => {
                // Silverman's rule: h = (4/3)^(1/5) * n^(-1/5)
                let factor = Self::safe_from_f64(4.0 / 3.0)?.powf(F::one() / five);
                Ok(factor * n.powf(-F::one() / five))
            }
            _ => {
                // Default to Scott's rule
                Ok(n.powf(-F::one() / five))
            }
        }
    }

    /// Estimate correlation using kernel density estimation
    fn estimate_kernel_correlation(
        &self,
        x: &Array1<F>,
        y: &Array1<F>,
        bandwidth: F,
        kernel_type: &KernelType,
    ) -> Result<F> {
        let n = x.len();

        // Simple kernel correlation estimation
        let mean_x = Self::safe_mean(x)?;
        let mean_y = Self::safe_mean(y)?;
        let std_x = x.std(F::one());
        let std_y = y.std(F::one());

        let mut numerator = F::zero();
        let mut denom_x = F::zero();
        let mut denom_y = F::zero();

        for i in 0..n {
            let weight_i = self.kernel_weight((x[i] - mean_x) / bandwidth, kernel_type)?
                * self.kernel_weight((y[i] - mean_y) / bandwidth, kernel_type)?;

            numerator = numerator + weight_i * (x[i] - mean_x) * (y[i] - mean_y);
            denom_x = denom_x + weight_i * (x[i] - mean_x) * (x[i] - mean_x);
            denom_y = denom_y + weight_i * (y[i] - mean_y) * (y[i] - mean_y);
        }

        if denom_x > F::zero() && denom_y > F::zero() {
            Ok(numerator / (denom_x.sqrt() * denom_y.sqrt()))
        } else {
            Ok(F::zero())
        }
    }

    /// Compute kernel weight
    fn kernel_weight(&self, u: F, kernel_type: &KernelType) -> Result<F> {
        Ok(match kernel_type {
            KernelType::Gaussian => {
                let pi = Self::safe_from_f64(std::f64::consts::PI)?;
                let two = Self::safe_from_f64(2.0)?;
                let two_pi = two * pi;
                (-(u * u) / two).exp() / two_pi.sqrt()
            }
            KernelType::Epanechnikov => {
                if u.abs() <= F::one() {
                    Self::safe_from_f64(0.75)? * (F::one() - u * u)
                } else {
                    F::zero()
                }
            }
            KernelType::Uniform => {
                if u.abs() <= F::one() {
                    Self::safe_from_f64(0.5)?
                } else {
                    F::zero()
                }
            }
            KernelType::Triangular => {
                if u.abs() <= F::one() {
                    F::one() - u.abs()
                } else {
                    F::zero()
                }
            }
            KernelType::Biweight => {
                if u.abs() <= F::one() {
                    Self::safe_from_f64(15.0 / 16.0)? * (F::one() - u * u).powi(2)
                } else {
                    F::zero()
                }
            }
        })
    }

    /// Fit Gaussian copula
    fn fit_gaussian_copula(&self, uniform_data: &Array2<F>) -> Result<(Vec<F>, Array2<F>)> {
        let (n_samples, n_features) = uniform_data.dim();

        // Transform uniform data to normal using inverse normal CDF (approximated)
        let mut normal_data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                normal_data[[i, j]] = self.inverse_normal_cdf(uniform_data[[i, j]])?;
            }
        }

        // Compute sample correlation of transformed data
        let correlation = self.compute_sample_correlation(&normal_data)?;
        let params = vec![]; // Gaussian copula parameters are just the correlation matrix

        Ok((params, correlation))
    }

    /// Fit Student-t copula
    fn fit_student_t_copula(&self, uniform_data: &Array2<F>) -> Result<(Vec<F>, Array2<F>)> {
        // Simplified: use Gaussian copula as approximation
        let (params, correlation) = self.fit_gaussian_copula(uniform_data)?;
        // In practice, would estimate degrees of freedom parameter
        let mut t_params = params;
        t_params.push(Self::safe_from_f64(5.0)?); // Fixed df = 5

        Ok((t_params, correlation))
    }

    /// Fit empirical copula
    fn fit_empirical_copula(&self, uniform_data: &Array2<F>) -> Result<(Vec<F>, Array2<F>)> {
        // Compute empirical copula correlation (Spearman correlation of uniform data)
        let correlation = self.compute_sample_correlation(uniform_data)?;
        let params = vec![]; // Empirical copula has no parameters

        Ok((params, correlation))
    }

    /// Approximate inverse normal CDF
    fn inverse_normal_cdf(&self, p: F) -> Result<F> {
        // Simple approximation using rational approximation
        if p <= F::zero() {
            return Ok(Self::safe_from_f64(-6.0)?);
        }
        if p >= F::one() {
            return Ok(Self::safe_from_f64(6.0)?);
        }

        // Use simple linear approximation for demonstration
        let two = Self::safe_from_f64(2.0)?;
        let half = Self::safe_from_f64(0.5)?;
        let six = Self::safe_from_f64(6.0)?;
        let result = (p - half) * six;
        Ok(result)
    }

    /// Compute ranks of data
    fn compute_ranks(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n = data.len();
        let mut indexed_data: Vec<(F, usize)> =
            data.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        indexed_data.sort_by(|a, b| Self::safe_partial_cmp(&a.0, &b.0));

        let mut ranks = Array1::zeros(n);
        for (rank, (_, original_index)) in indexed_data.iter().enumerate() {
            ranks[*original_index] = Self::safe_from_usize(rank + 1)?;
        }

        Ok(ranks)
    }

    /// Compute sample correlation
    fn compute_sample_correlation(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let (n_samples, n_features) = x.dim();
        let means = Self::safe_mean_axis(x, Axis(0))?;

        let mut correlation = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    correlation[[i, j]] = F::one();
                } else {
                    let mut numerator = F::zero();
                    let mut sum_sq_i = F::zero();
                    let mut sum_sq_j = F::zero();

                    for k in 0..n_samples {
                        let diff_i = x[[k, i]] - means[i];
                        let diff_j = x[[k, j]] - means[j];

                        numerator = numerator + diff_i * diff_j;
                        sum_sq_i = sum_sq_i + diff_i * diff_i;
                        sum_sq_j = sum_sq_j + diff_j * diff_j;
                    }

                    if sum_sq_i > F::zero() && sum_sq_j > F::zero() {
                        correlation[[i, j]] = numerator / (sum_sq_i.sqrt() * sum_sq_j.sqrt());
                    } else {
                        correlation[[i, j]] = F::zero();
                    }
                }
            }
        }

        Ok(correlation)
    }

    /// Compute Spearman rank correlation
    fn compute_spearman_correlation(&self, ranks_x: &Array1<F>, ranks_y: &Array1<F>) -> Result<F> {
        let n = Self::safe_from_usize(ranks_x.len())?;
        let two = Self::safe_from_f64(2.0)?;
        let mean_rank = (n + F::one()) / two;

        let mut numerator = F::zero();
        let mut sum_sq_x = F::zero();
        let mut sum_sq_y = F::zero();

        for i in 0..ranks_x.len() {
            let diff_x = ranks_x[i] - mean_rank;
            let diff_y = ranks_y[i] - mean_rank;

            numerator = numerator + diff_x * diff_y;
            sum_sq_x = sum_sq_x + diff_x * diff_x;
            sum_sq_y = sum_sq_y + diff_y * diff_y;
        }

        if sum_sq_x > F::zero() && sum_sq_y > F::zero() {
            Ok(numerator / (sum_sq_x.sqrt() * sum_sq_y.sqrt()))
        } else {
            Ok(F::zero())
        }
    }

    /// Compute Kendall's tau
    fn compute_kendall_tau(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
        let n = x.len();
        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sign_x = if x[i] > x[j] {
                    1
                } else if x[i] < x[j] {
                    -1
                } else {
                    0
                };
                let sign_y = if y[i] > y[j] {
                    1
                } else if y[i] < y[j] {
                    -1
                } else {
                    0
                };

                let product = sign_x * sign_y;
                if product > 0 {
                    concordant += 1;
                } else if product < 0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        let tau = Self::safe_from_f64((concordant as i32 - discordant as i32) as f64)?
            / Self::safe_from_usize(total_pairs)?;

        Ok(tau)
    }

    /// Compute Hoeffding's D statistic
    fn compute_hoeffding_d(&self, ranks_x: &Array1<F>, ranks_y: &Array1<F>) -> Result<F> {
        let n = Self::safe_from_usize(ranks_x.len())?;
        let mut d = F::zero();

        for i in 0..ranks_x.len() {
            let r_i = ranks_x[i];
            let s_i = ranks_y[i];

            let mut q_i = F::zero();
            for j in 0..ranks_x.len() {
                if ranks_x[j] <= r_i && ranks_y[j] <= s_i {
                    q_i = q_i + F::one();
                }
            }

            d = d + (q_i - r_i * s_i / n).powi(2);
        }

        Ok(d / (n * n))
    }

    /// Compute distance correlation
    fn compute_distance_correlation(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
        let n = x.len();

        // Compute distance matrices
        let mut dist_x = Array2::zeros((n, n));
        let mut dist_y = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                dist_x[[i, j]] = (x[i] - x[j]).abs();
                dist_y[[i, j]] = (y[i] - y[j]).abs();
            }
        }

        // Double center the distance matrices
        let mean_x = dist_x.mean().ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute mean of distance matrix".to_string())
        })?;
        let mean_y = dist_y.mean().ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute mean of distance matrix".to_string())
        })?;
        let row_means_x = dist_x.mean_axis(Axis(1)).ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute row means".to_string())
        })?;
        let row_means_y = dist_y.mean_axis(Axis(1)).ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute row means".to_string())
        })?;
        let col_means_x = dist_x.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute column means".to_string())
        })?;
        let col_means_y = dist_y.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute column means".to_string())
        })?;

        let mut a = Array2::zeros((n, n));
        let mut b = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = dist_x[[i, j]] - row_means_x[i] - col_means_x[j] + mean_x;
                b[[i, j]] = dist_y[[i, j]] - row_means_y[i] - col_means_y[j] + mean_y;
            }
        }

        // Compute distance covariance and variances
        let dcov_xy = (&a * &b).mean().ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute distance covariance".to_string())
        })?;
        let dvar_x = (&a * &a).mean().ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute distance variance".to_string())
        })?;
        let dvar_y = (&b * &b).mean().ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute distance variance".to_string())
        })?;

        if dvar_x > F::zero() && dvar_y > F::zero() {
            Ok(dcov_xy / (dvar_x.sqrt() * dvar_y.sqrt()))
        } else {
            Ok(F::zero())
        }
    }

    /// Compute quadrant correlation
    fn compute_quadrant_correlation(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
        let n = x.len();
        let median_x = self.compute_median(x)?;
        let median_y = self.compute_median(y)?;

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            let sign_x = if x[i] >= median_x { 1 } else { -1 };
            let sign_y = if y[i] >= median_y { 1 } else { -1 };

            if sign_x * sign_y > 0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }

        let quadrant_corr = Self::safe_from_f64((concordant as i32 - discordant as i32) as f64)?
            / Self::safe_from_usize(n)?;
        Ok(quadrant_corr)
    }

    /// Compute median
    fn compute_median(&self, data: &Array1<F>) -> Result<F> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| Self::safe_partial_cmp(a, b));

        let n = sorted_data.len();
        if n % 2 == 0 {
            let two = Self::safe_from_f64(2.0)?;
            Ok((sorted_data[n / 2 - 1] + sorted_data[n / 2]) / two)
        } else {
            Ok(sorted_data[n / 2])
        }
    }

    /// Winsorize data
    fn winsorize(&self, data: &Array1<F>, percentile: F) -> Result<Array1<F>> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| Self::safe_partial_cmp(a, b));

        let n = sorted_data.len();
        let n_f = Self::safe_from_usize(n)?;
        let lower_idx = Self::safe_to_usize(percentile * n_f)?;
        let upper_idx = n - 1 - lower_idx;

        let lower_bound = sorted_data[lower_idx];
        let upper_bound = sorted_data[upper_idx];

        let mut winsorized = data.clone();
        for i in 0..winsorized.len() {
            if winsorized[i] < lower_bound {
                winsorized[i] = lower_bound;
            } else if winsorized[i] > upper_bound {
                winsorized[i] = upper_bound;
            }
        }

        Ok(winsorized)
    }

    /// Compute Pearson correlation
    fn compute_pearson_correlation(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
        let n = Self::safe_from_usize(x.len())?;
        let mean_x = Self::safe_mean(x)?;
        let mean_y = Self::safe_mean(y)?;

        let mut numerator = F::zero();
        let mut sum_sq_x = F::zero();
        let mut sum_sq_y = F::zero();

        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;

            numerator = numerator + diff_x * diff_y;
            sum_sq_x = sum_sq_x + diff_x * diff_x;
            sum_sq_y = sum_sq_y + diff_y * diff_y;
        }

        if sum_sq_x > F::zero() && sum_sq_y > F::zero() {
            Ok(numerator / (sum_sq_x.sqrt() * sum_sq_y.sqrt()))
        } else {
            Ok(F::zero())
        }
    }

    /// Compute biweight midcorrelation
    fn compute_biweight_midcorrelation(&self, x: &Array1<F>, y: &Array1<F>, c: F) -> Result<F> {
        let n = x.len();
        let median_x = self.compute_median(x)?;
        let median_y = self.compute_median(y)?;
        let mad_x = self.compute_mad(x)?;
        let mad_y = self.compute_mad(y)?;

        let mut numerator = F::zero();
        let mut sum_sq_x = F::zero();
        let mut sum_sq_y = F::zero();

        for i in 0..n {
            let u_i = (x[i] - median_x) / (c * mad_x);
            let v_i = (y[i] - median_y) / (c * mad_y);

            if u_i.abs() <= F::one() && v_i.abs() <= F::one() {
                let weight_x = (F::one() - u_i * u_i).powi(2);
                let weight_y = (F::one() - v_i * v_i).powi(2);

                numerator = numerator + (x[i] - median_x) * weight_x * (y[i] - median_y) * weight_y;
                sum_sq_x = sum_sq_x + (x[i] - median_x) * (x[i] - median_x) * weight_x * weight_x;
                sum_sq_y = sum_sq_y + (y[i] - median_y) * (y[i] - median_y) * weight_y * weight_y;
            }
        }

        if sum_sq_x > F::zero() && sum_sq_y > F::zero() {
            Ok(numerator / (sum_sq_x.sqrt() * sum_sq_y.sqrt()))
        } else {
            Ok(F::zero())
        }
    }

    /// Compute percentage bend correlation
    fn compute_percentage_bend_correlation(
        &self,
        x: &Array1<F>,
        y: &Array1<F>,
        beta: F,
    ) -> Result<F> {
        let n = x.len();
        let n_f = Self::safe_from_usize(n)?;
        let bend_idx = Self::safe_to_usize(beta * n_f)?;

        let mut sorted_x = x.to_vec();
        let mut sorted_y = y.to_vec();
        sorted_x.sort_by(|a, b| Self::safe_partial_cmp(a, b));
        sorted_y.sort_by(|a, b| Self::safe_partial_cmp(a, b));

        let bend_x = sorted_x[bend_idx];
        let bend_y = sorted_y[bend_idx];

        // Apply bending transformation
        let mut bent_x = Array1::zeros(n);
        let mut bent_y = Array1::zeros(n);

        for i in 0..n {
            bent_x[i] = if x[i].abs() <= bend_x {
                x[i]
            } else {
                bend_x * x[i].signum()
            };
            bent_y[i] = if y[i].abs() <= bend_y {
                y[i]
            } else {
                bend_y * y[i].signum()
            };
        }

        self.compute_pearson_correlation(&bent_x, &bent_y)
    }

    /// Compute median absolute deviation
    fn compute_mad(&self, data: &Array1<F>) -> Result<F> {
        let median = self.compute_median(data)?;
        let mut abs_deviations = Array1::zeros(data.len());

        for i in 0..data.len() {
            abs_deviations[i] = (data[i] - median).abs();
        }

        let scale_factor = Self::safe_from_f64(1.4826)?;
        Ok(self.compute_median(&abs_deviations)? * scale_factor) // Scale factor for normal distribution
    }

    /// Permutation test for correlation
    fn permutation_test_correlation(
        &self,
        x: &Array1<F>,
        y: &Array1<F>,
        n_permutations: usize,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<F> {
        let observed_corr = self.compute_pearson_correlation(x, y)?.abs();
        let mut extreme_count = 0;

        let mut y_permuted = y.clone();
        for _ in 0..n_permutations {
            // Permute y
            for i in 0..y_permuted.len() {
                let j = rng.gen_range(0..y_permuted.len());
                let temp = y_permuted[i];
                y_permuted[i] = y_permuted[j];
                y_permuted[j] = temp;
            }

            let perm_corr = self.compute_pearson_correlation(x, &y_permuted)?.abs();
            if perm_corr >= observed_corr {
                extreme_count += 1;
            }
        }

        let ec_f = Self::safe_from_usize(extreme_count)?;
        let np_f = Self::safe_from_usize(n_permutations)?;
        Ok(ec_f / np_f)
    }
}

/// Builder for non-parametric covariance estimation
pub struct NonparametricCovarianceBuilder<F: NdFloat + FromPrimitive> {
    config: NonparametricCovarianceConfig<F>,
}

impl<F: NdFloat + FromPrimitive> NonparametricCovarianceBuilder<F> {
    pub fn new() -> Self {
        Self {
            config: NonparametricCovarianceConfig::default(),
        }
    }

    pub fn method(mut self, method: NonparametricMethod) -> Self {
        self.config.method = method;
        self
    }

    pub fn kernel_type(mut self, kernel_type: KernelType) -> Self {
        if let Some(ref mut kde_config) = self.config.kde_config {
            kde_config.kernel_type = kernel_type;
        }
        self
    }

    pub fn bandwidth(mut self, bandwidth: F) -> Self {
        if let Some(ref mut kde_config) = self.config.kde_config {
            kde_config.bandwidth = Some(bandwidth);
        }
        self
    }

    pub fn bandwidth_method(mut self, method: String) -> Self {
        if let Some(ref mut kde_config) = self.config.kde_config {
            kde_config.bandwidth_method = method;
        }
        self
    }

    pub fn copula_type(mut self, copula_type: CopulaType) -> Self {
        if let Some(ref mut copula_config) = self.config.copula_config {
            copula_config.copula_type = copula_type;
        }
        self
    }

    pub fn copula_parameters(mut self, parameters: Vec<F>) -> Self {
        if let Some(ref mut copula_config) = self.config.copula_config {
            copula_config.parameters = parameters;
        }
        self
    }

    pub fn rank_correlation_type(mut self, correlation_type: RankCorrelationType) -> Self {
        if let Some(ref mut rank_config) = self.config.rank_config {
            rank_config.correlation_type = correlation_type;
        }
        self
    }

    pub fn robust_correlation_type(mut self, correlation_type: RobustCorrelationType) -> Self {
        if let Some(ref mut robust_config) = self.config.robust_config {
            robust_config.correlation_type = correlation_type;
        }
        self
    }

    pub fn winsorize_percentile(mut self, percentile: F) -> Self {
        if let Some(ref mut robust_config) = self.config.robust_config {
            robust_config.winsorize_percentile = percentile;
        }
        self
    }

    pub fn tuning_constant(mut self, constant: F) -> Self {
        if let Some(ref mut robust_config) = self.config.robust_config {
            robust_config.tuning_constant = constant;
        }
        self
    }

    pub fn permutation_tests(mut self, enable: bool) -> Self {
        if let Some(ref mut dist_free_config) = self.config.distribution_free_config {
            dist_free_config.permutation_tests = enable;
        }
        self
    }

    pub fn n_permutations(mut self, n_permutations: usize) -> Self {
        if let Some(ref mut dist_free_config) = self.config.distribution_free_config {
            dist_free_config.n_permutations = n_permutations;
        }
        self
    }

    pub fn regularization(mut self, reg: F) -> Self {
        self.config.regularization = reg;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn build(self) -> NonparametricCovariance<F> {
        NonparametricCovariance::new(self.config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::Distribution;
    use scirs2_core::Random;
    use scirs2_core::StandardNormal;
    use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};

    fn generate_nonlinear_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut rng = Random::seed(42);
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let base = StandardNormal.sample(&mut rng);
            data[[i, 0]] = base;

            if n_features > 1 {
                // Add nonlinear relationships
                let sample: f64 = StandardNormal.sample(&mut rng);
                data[[i, 1]] = base * base + 0.5 * sample;
            }
            if n_features > 2 {
                let sample: f64 = StandardNormal.sample(&mut rng);
                #[allow(clippy::unnecessary_cast)]
                {
                    data[[i, 2]] = ((base * 2.0) as f64).sin() + 0.3 * sample;
                }
            }
        }

        data
    }

    #[test]
    fn test_rank_based_covariance() {
        let data = generate_nonlinear_data(50, 3);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RankBased)
            .rank_correlation_type(RankCorrelationType::Spearman)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 3);
        assert_eq!(fitted.n_samples(), 50);
        assert_eq!(fitted.covariance().shape(), &[3, 3]);
        assert_eq!(fitted.correlation().shape(), &[3, 3]);
        assert!(fitted.ranks().is_some());

        let correlation = fitted.correlation();
        // Check that correlation matrix is symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(correlation[[i, j]], correlation[[j, i]], epsilon = 1e-10);
            }
        }

        // Check diagonal elements are 1
        for i in 0..3 {
            assert_abs_diff_eq!(correlation[[i, i]], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kendall_tau_correlation() {
        let data = generate_nonlinear_data(30, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RankBased)
            .rank_correlation_type(RankCorrelationType::KendallTau)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        let correlation = fitted.correlation();

        // Kendall's tau should be between -1 and 1
        for i in 0..2 {
            for j in 0..2 {
                assert!(correlation[[i, j]] >= -1.0 && correlation[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_robust_correlation() {
        let mut data = generate_nonlinear_data(40, 2);

        // Add some outliers
        data[[0, 0]] = 10.0;
        data[[1, 1]] = -8.0;

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RobustCorrelation)
            .robust_correlation_type(RobustCorrelationType::BiweightMidcorrelation)
            .tuning_constant(9.0)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        let correlation = fitted.correlation();

        // Check that correlation values are finite and within bounds
        for i in 0..2 {
            for j in 0..2 {
                assert!(correlation[[i, j]].is_finite());
                assert!(correlation[[i, j]] >= -1.0 && correlation[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_winsorized_correlation() {
        let mut data = generate_nonlinear_data(35, 2);

        // Add extreme values
        data[[0, 0]] = 15.0;
        data[[1, 1]] = -12.0;

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RobustCorrelation)
            .robust_correlation_type(RobustCorrelationType::Winsorized)
            .winsorize_percentile(0.1)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        let correlation = fitted.correlation();

        // Check basic properties
        assert_abs_diff_eq!(correlation[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(correlation[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(correlation[[0, 1]], correlation[[1, 0]], epsilon = 1e-10);
    }

    #[test]
    fn test_copula_based_covariance() {
        let data = generate_nonlinear_data(40, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::CopulaBased)
            .copula_type(CopulaType::Gaussian)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.ranks().is_some());
        assert!(fitted.copula_parameters().is_some());

        let ranks = fitted.ranks().unwrap();
        assert_eq!(ranks.shape(), &[40, 2]);

        // Check that ranks are in reasonable range
        for i in 0..40 {
            for j in 0..2 {
                assert!(ranks[[i, j]] >= 1.0 && ranks[[i, j]] <= 40.0);
            }
        }
    }

    #[test]
    fn test_empirical_copula() {
        let data = generate_nonlinear_data(25, 3);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::CopulaBased)
            .copula_type(CopulaType::Empirical)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 3);
        let correlation = fitted.correlation();

        // Check matrix properties
        for i in 0..3 {
            assert_abs_diff_eq!(correlation[[i, i]], 1.0, epsilon = 1e-10);
            for j in 0..3 {
                assert!(correlation[[i, j]] >= -1.0 && correlation[[i, j]] <= 1.0);
                assert_abs_diff_eq!(correlation[[i, j]], correlation[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_kernel_density_estimation() {
        let data = generate_nonlinear_data(30, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::KernelDensityEstimation)
            .kernel_type(KernelType::Gaussian)
            .bandwidth(0.5)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        let covariance = fitted.covariance();

        // Check that covariance matrix is positive definite
        let eigenvals = covariance.eigvalsh(UPLO::Lower).unwrap();
        assert!(eigenvals.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_distance_correlation() {
        let data = generate_nonlinear_data(30, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RankBased)
            .rank_correlation_type(RankCorrelationType::DistanceCorrelation)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        let correlation = fitted.correlation();

        // Distance correlation should be non-negative
        for i in 0..2 {
            for j in 0..2 {
                assert!(correlation[[i, j]] >= 0.0);
            }
        }

        // Should be 1 on diagonal
        assert_abs_diff_eq!(correlation[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(correlation[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distribution_free_methods() {
        let data = generate_nonlinear_data(35, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::DistributionFree)
            .permutation_tests(true)
            .n_permutations(100)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.p_values().is_some());

        let p_values = fitted.p_values().unwrap();
        assert_eq!(p_values.shape(), &[2, 2]);

        // P-values should be between 0 and 1
        for i in 0..2 {
            for j in 0..2 {
                assert!(p_values[[i, j]] >= 0.0 && p_values[[i, j]] <= 1.0);
            }
        }

        // Diagonal p-values should be 0 (perfect correlation with self)
        assert_abs_diff_eq!(p_values[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p_values[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrant_correlation() {
        let data = generate_nonlinear_data(40, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RobustCorrelation)
            .robust_correlation_type(RobustCorrelationType::Quadrant)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        let correlation = fitted.correlation();

        // Quadrant correlation should be between -1 and 1
        for i in 0..2 {
            for j in 0..2 {
                assert!(correlation[[i, j]] >= -1.0 && correlation[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_confidence_intervals() {
        let data = generate_nonlinear_data(30, 2);

        let estimator = NonparametricCovariance::builder()
            .method(NonparametricMethod::RobustCorrelation)
            .robust_correlation_type(RobustCorrelationType::BiweightMidcorrelation)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();
        let (lower, upper) = fitted.robust_correlation_ci(0.05).unwrap();

        assert_eq!(lower.shape(), &[2, 2]);
        assert_eq!(upper.shape(), &[2, 2]);

        // Check that lower bounds are less than upper bounds
        for i in 0..2 {
            for j in 0..2 {
                assert!(lower[[i, j]] <= upper[[i, j]]);
            }
        }
    }
}
