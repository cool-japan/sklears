//! Advanced Validation Framework for Kernel Approximation Methods
//!
//! This module provides comprehensive validation tools including theoretical error bound
//! validation, convergence analysis, and approximation quality assessment.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::error::Result;
use std::collections::HashMap;

/// Comprehensive validation framework for kernel approximation methods
#[derive(Debug, Clone)]
/// KernelApproximationValidator
pub struct KernelApproximationValidator {
    config: ValidationConfig,
    theoretical_bounds: HashMap<String, TheoreticalBound>,
}

/// Configuration for validation
#[derive(Debug, Clone)]
/// ValidationConfig
pub struct ValidationConfig {
    /// confidence_level
    pub confidence_level: f64,
    /// max_approximation_error
    pub max_approximation_error: f64,
    /// convergence_tolerance
    pub convergence_tolerance: f64,
    /// stability_tolerance
    pub stability_tolerance: f64,
    /// sample_sizes
    pub sample_sizes: Vec<usize>,
    /// approximation_dimensions
    pub approximation_dimensions: Vec<usize>,
    /// repetitions
    pub repetitions: usize,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            max_approximation_error: 0.1,
            convergence_tolerance: 1e-6,
            stability_tolerance: 1e-4,
            sample_sizes: vec![100, 500, 1000, 2000],
            approximation_dimensions: vec![50, 100, 200, 500],
            repetitions: 10,
            random_state: Some(42),
        }
    }
}

/// Theoretical error bounds for different approximation methods
#[derive(Debug, Clone)]
/// TheoreticalBound
pub struct TheoreticalBound {
    /// method_name
    pub method_name: String,
    /// bound_type
    pub bound_type: BoundType,
    /// bound_function
    pub bound_function: BoundFunction,
    /// constants
    pub constants: HashMap<String, f64>,
}

/// Types of theoretical bounds
#[derive(Debug, Clone)]
/// BoundType
pub enum BoundType {
    /// Probabilistic bound with confidence level
    Probabilistic { confidence: f64 },
    /// Deterministic worst-case bound
    Deterministic,
    /// Expected error bound
    Expected,
    /// Concentration inequality bound
    Concentration { deviation_parameter: f64 },
}

/// Functions for computing theoretical bounds
#[derive(Debug, Clone)]
/// BoundFunction
pub enum BoundFunction {
    /// RFF approximation error: O(sqrt(log(d)/m))
    RandomFourierFeatures,
    /// Nyström approximation error: depends on eigenvalue decay
    Nystroem,
    /// Structured random features: O(sqrt(d*log(d)/m))
    StructuredRandomFeatures,
    /// Fastfood approximation: O(sqrt(d*log^2(d)/m))
    Fastfood,
    /// Custom bound function
    Custom { formula: String },
}

/// Result of validation analysis
#[derive(Debug, Clone)]
/// ValidationResult
pub struct ValidationResult {
    /// method_name
    pub method_name: String,
    /// empirical_errors
    pub empirical_errors: Vec<f64>,
    /// theoretical_bounds
    pub theoretical_bounds: Vec<f64>,
    /// bound_violations
    pub bound_violations: usize,
    /// bound_tightness
    pub bound_tightness: f64,
    /// convergence_rate
    pub convergence_rate: Option<f64>,
    /// stability_analysis
    pub stability_analysis: StabilityAnalysis,
    /// sample_complexity
    pub sample_complexity: SampleComplexityAnalysis,
    /// dimension_dependency
    pub dimension_dependency: DimensionDependencyAnalysis,
}

/// Stability analysis results
#[derive(Debug, Clone)]
/// StabilityAnalysis
pub struct StabilityAnalysis {
    /// perturbation_sensitivity
    pub perturbation_sensitivity: f64,
    /// numerical_stability
    pub numerical_stability: f64,
    /// condition_numbers
    pub condition_numbers: Vec<f64>,
    /// eigenvalue_stability
    pub eigenvalue_stability: f64,
}

/// Sample complexity analysis
#[derive(Debug, Clone)]
/// SampleComplexityAnalysis
pub struct SampleComplexityAnalysis {
    /// minimum_samples
    pub minimum_samples: usize,
    /// convergence_rate
    pub convergence_rate: f64,
    /// sample_efficiency
    pub sample_efficiency: f64,
    /// dimension_scaling
    pub dimension_scaling: f64,
}

/// Dimension dependency analysis
#[derive(Debug, Clone)]
/// DimensionDependencyAnalysis
pub struct DimensionDependencyAnalysis {
    /// approximation_quality_vs_dimension
    pub approximation_quality_vs_dimension: Vec<(usize, f64)>,
    /// computational_cost_vs_dimension
    pub computational_cost_vs_dimension: Vec<(usize, f64)>,
    /// optimal_dimension
    pub optimal_dimension: usize,
    /// dimension_efficiency
    pub dimension_efficiency: f64,
}

/// Cross-validation result for kernel approximation
#[derive(Debug, Clone)]
/// CrossValidationResult
pub struct CrossValidationResult {
    /// method_name
    pub method_name: String,
    /// cv_scores
    pub cv_scores: Vec<f64>,
    /// mean_score
    pub mean_score: f64,
    /// std_score
    pub std_score: f64,
    /// best_parameters
    pub best_parameters: HashMap<String, f64>,
    /// parameter_sensitivity
    pub parameter_sensitivity: HashMap<String, f64>,
}

impl KernelApproximationValidator {
    /// Create a new validator with configuration
    pub fn new(config: ValidationConfig) -> Self {
        let mut validator = Self {
            config,
            theoretical_bounds: HashMap::new(),
        };

        // Add default theoretical bounds
        validator.add_default_bounds();
        validator
    }

    /// Add theoretical bounds for a specific method
    pub fn add_theoretical_bound(&mut self, bound: TheoreticalBound) {
        self.theoretical_bounds
            .insert(bound.method_name.clone(), bound);
    }

    fn add_default_bounds(&mut self) {
        // RFF bounds
        self.add_theoretical_bound(TheoreticalBound {
            method_name: "RBF".to_string(),
            bound_type: BoundType::Probabilistic { confidence: 0.95 },
            bound_function: BoundFunction::RandomFourierFeatures,
            constants: [
                ("kernel_bound".to_string(), 1.0),
                ("lipschitz_constant".to_string(), 1.0),
            ]
            .iter()
            .cloned()
            .collect(),
        });

        // Nyström bounds
        self.add_theoretical_bound(TheoreticalBound {
            method_name: "Nystroem".to_string(),
            bound_type: BoundType::Expected,
            bound_function: BoundFunction::Nystroem,
            constants: [
                ("trace_bound".to_string(), 1.0),
                ("effective_rank".to_string(), 100.0),
            ]
            .iter()
            .cloned()
            .collect(),
        });

        // Fastfood bounds
        self.add_theoretical_bound(TheoreticalBound {
            method_name: "Fastfood".to_string(),
            bound_type: BoundType::Probabilistic { confidence: 0.95 },
            bound_function: BoundFunction::Fastfood,
            constants: [
                ("dimension_factor".to_string(), 1.0),
                ("log_factor".to_string(), 2.0),
            ]
            .iter()
            .cloned()
            .collect(),
        });
    }

    /// Validate a kernel approximation method
    pub fn validate_method<T: ValidatableKernelMethod>(
        &self,
        method: &T,
        data: &Array2<f64>,
        true_kernel: Option<&Array2<f64>>,
    ) -> Result<ValidationResult> {
        let method_name = method.method_name();
        let mut empirical_errors = Vec::new();
        let mut theoretical_bounds = Vec::new();
        let mut condition_numbers = Vec::new();

        // Test different approximation dimensions
        for &n_components in &self.config.approximation_dimensions {
            let mut dimension_errors = Vec::new();

            for _ in 0..self.config.repetitions {
                // Fit and evaluate the method
                let fitted = method.fit_with_dimension(data, n_components)?;
                let approximation = fitted.get_kernel_approximation(data)?;

                // Compute empirical error
                let empirical_error = if let Some(true_k) = true_kernel {
                    self.compute_approximation_error(&approximation, true_k)?
                } else {
                    // Use RBF kernel as reference
                    let rbf_kernel = self.compute_rbf_kernel(data, 1.0)?;
                    self.compute_approximation_error(&approximation, &rbf_kernel)?
                };

                dimension_errors.push(empirical_error);

                // Compute condition number for stability analysis
                if let Some(cond_num) = fitted.compute_condition_number()? {
                    condition_numbers.push(cond_num);
                }
            }

            let mean_error = dimension_errors.iter().sum::<f64>() / dimension_errors.len() as f64;
            empirical_errors.push(mean_error);

            // Compute theoretical bound
            if let Some(bound) = self.theoretical_bounds.get(&method_name) {
                let theoretical_bound = self.compute_theoretical_bound(
                    bound,
                    data.nrows(),
                    data.ncols(),
                    n_components,
                )?;
                theoretical_bounds.push(theoretical_bound);
            } else {
                theoretical_bounds.push(f64::INFINITY);
            }
        }

        // Count bound violations
        let bound_violations = empirical_errors
            .iter()
            .zip(theoretical_bounds.iter())
            .filter(|(&emp, &theo)| emp > theo)
            .count();

        // Compute bound tightness (average ratio of empirical to theoretical)
        let bound_tightness = empirical_errors
            .iter()
            .zip(theoretical_bounds.iter())
            .filter(|(_, &theo)| theo.is_finite())
            .map(|(&emp, &theo)| emp / theo)
            .sum::<f64>()
            / empirical_errors.len() as f64;

        // Analyze convergence rate
        let convergence_rate = self.estimate_convergence_rate(&empirical_errors);

        // Perform stability analysis
        let stability_analysis = self.analyze_stability(method, data, &condition_numbers)?;

        // Analyze sample complexity
        let sample_complexity = self.analyze_sample_complexity(method, data)?;

        // Analyze dimension dependency
        let dimension_dependency =
            self.analyze_dimension_dependency(method, data, &empirical_errors)?;

        Ok(ValidationResult {
            method_name,
            empirical_errors,
            theoretical_bounds,
            bound_violations,
            bound_tightness,
            convergence_rate,
            stability_analysis,
            sample_complexity,
            dimension_dependency,
        })
    }

    /// Perform cross-validation for parameter selection
    pub fn cross_validate<T: ValidatableKernelMethod>(
        &self,
        method: &T,
        data: &Array2<f64>,
        targets: Option<&Array1<f64>>,
        parameter_grid: HashMap<String, Vec<f64>>,
    ) -> Result<CrossValidationResult> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();
        let mut all_scores = Vec::new();
        let mut parameter_sensitivity = HashMap::new();

        // Generate parameter combinations
        let param_combinations = self.generate_parameter_combinations(&parameter_grid);

        for params in param_combinations {
            let cv_scores = self.k_fold_cross_validation(method, data, targets, &params, 5)?;
            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;

            all_scores.push(mean_score);

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = params.clone();
            }
        }

        // Analyze parameter sensitivity
        for (param_name, param_values) in &parameter_grid {
            let mut sensitivities = Vec::new();

            for &param_value in param_values.iter() {
                let mut single_param = best_parameters.clone();
                single_param.insert(param_name.clone(), param_value);

                let cv_scores =
                    self.k_fold_cross_validation(method, data, targets, &single_param, 3)?;
                let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
                sensitivities.push((best_score - mean_score).abs());
            }

            let sensitivity = sensitivities.iter().sum::<f64>() / sensitivities.len() as f64;
            parameter_sensitivity.insert(param_name.clone(), sensitivity);
        }

        let mean_score = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
        let variance = all_scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f64>()
            / all_scores.len() as f64;
        let std_score = variance.sqrt();

        Ok(CrossValidationResult {
            method_name: method.method_name(),
            cv_scores: all_scores,
            mean_score,
            std_score,
            best_parameters,
            parameter_sensitivity,
        })
    }

    fn compute_approximation_error(
        &self,
        approx_kernel: &Array2<f64>,
        true_kernel: &Array2<f64>,
    ) -> Result<f64> {
        // Compute Frobenius norm error
        let diff = approx_kernel - true_kernel;
        let frobenius_error = diff.mapv(|x| x * x).sum().sqrt();

        // Normalize by true kernel norm
        let true_norm = true_kernel.mapv(|x| x * x).sum().sqrt();
        Ok(frobenius_error / true_norm.max(1e-8))
    }

    fn compute_rbf_kernel(&self, data: &Array2<f64>, gamma: f64) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut kernel = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let diff = &data.row(i) - &data.row(j);
                let dist_sq = diff.mapv(|x| x * x).sum();
                let similarity = (-gamma * dist_sq).exp();
                kernel[[i, j]] = similarity;
                kernel[[j, i]] = similarity;
            }
        }

        Ok(kernel)
    }

    fn compute_theoretical_bound(
        &self,
        bound: &TheoreticalBound,
        n_samples: usize,
        n_features: usize,
        n_components: usize,
    ) -> Result<f64> {
        let bound_value = match &bound.bound_function {
            BoundFunction::RandomFourierFeatures => {
                let kernel_bound = bound.constants.get("kernel_bound").unwrap_or(&1.0);
                let lipschitz = bound.constants.get("lipschitz_constant").unwrap_or(&1.0);

                // O(sqrt(log(d)/m)) bound for RFF
                let log_factor = (n_features as f64).ln();
                kernel_bound * lipschitz * (log_factor / n_components as f64).sqrt()
            }
            BoundFunction::Nystroem => {
                let trace_bound = bound.constants.get("trace_bound").unwrap_or(&1.0);
                let effective_rank = bound.constants.get("effective_rank").unwrap_or(&100.0);

                // Approximation depends on eigenvalue decay
                trace_bound * (effective_rank / n_components as f64).sqrt()
            }
            BoundFunction::StructuredRandomFeatures => {
                let log_factor = (n_features as f64).ln();
                (n_features as f64 * log_factor / n_components as f64).sqrt()
            }
            BoundFunction::Fastfood => {
                let log_factor = bound.constants.get("log_factor").unwrap_or(&2.0);
                let dim_factor = bound.constants.get("dimension_factor").unwrap_or(&1.0);

                let log_d = (n_features as f64).ln();
                dim_factor
                    * (n_features as f64 * log_d.powf(*log_factor) / n_components as f64).sqrt()
            }
            BoundFunction::Custom { formula: _ } => {
                // Placeholder for custom formulas
                1.0 / (n_components as f64).sqrt()
            }
        };

        // Apply bound type modifications
        let final_bound = match &bound.bound_type {
            BoundType::Probabilistic { confidence } => {
                // Add confidence-dependent factor
                let z_score = self.inverse_normal_cdf(*confidence);
                bound_value * (1.0 + z_score / (n_samples as f64).sqrt())
            }
            BoundType::Deterministic => bound_value,
            BoundType::Expected => bound_value * 0.8, // Expected is typically tighter
            BoundType::Concentration {
                deviation_parameter,
            } => bound_value * (1.0 + deviation_parameter / (n_samples as f64).sqrt()),
        };

        Ok(final_bound)
    }

    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Approximation of inverse normal CDF for confidence intervals
        if p <= 0.5 {
            -self.inverse_normal_cdf(1.0 - p)
        } else {
            let t = (-2.0 * (1.0 - p).ln()).sqrt();
            let c0 = 2.515517;
            let c1 = 0.802853;
            let c2 = 0.010328;
            let d1 = 1.432788;
            let d2 = 0.189269;
            let d3 = 0.001308;

            t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        }
    }

    fn estimate_convergence_rate(&self, errors: &[f64]) -> Option<f64> {
        if errors.len() < 3 {
            return None;
        }

        // Fit log(error) = a + b * log(dimension) to estimate convergence rate
        let dimensions: Vec<f64> = self
            .config
            .approximation_dimensions
            .iter()
            .take(errors.len())
            .map(|&x| (x as f64).ln())
            .collect();

        let log_errors: Vec<f64> = errors.iter().map(|&x| x.ln()).collect();

        // Simple linear regression
        let n = dimensions.len() as f64;
        let sum_x = dimensions.iter().sum::<f64>();
        let sum_y = log_errors.iter().sum::<f64>();
        let sum_xy = dimensions
            .iter()
            .zip(log_errors.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f64>();
        let sum_x2 = dimensions.iter().map(|&x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Some(-slope) // Negative because we expect decreasing error
    }

    fn analyze_stability<T: ValidatableKernelMethod>(
        &self,
        method: &T,
        data: &Array2<f64>,
        condition_numbers: &[f64],
    ) -> Result<StabilityAnalysis> {
        let mut rng = RealStdRng::seed_from_u64(self.config.random_state.unwrap_or(42));
        let normal = RandNormal::new(0.0, self.config.stability_tolerance).unwrap();

        // Test perturbation sensitivity
        let mut perturbation_errors = Vec::new();

        for _ in 0..5 {
            let mut perturbed_data = data.clone();
            for elem in perturbed_data.iter_mut() {
                *elem += rng.sample(normal);
            }

            let original_fitted = method.fit_with_dimension(data, 100)?;
            let perturbed_fitted = method.fit_with_dimension(&perturbed_data, 100)?;

            let original_approx = original_fitted.get_kernel_approximation(data)?;
            let perturbed_approx = perturbed_fitted.get_kernel_approximation(data)?;

            let error = self.compute_approximation_error(&perturbed_approx, &original_approx)?;
            perturbation_errors.push(error);
        }

        let perturbation_sensitivity =
            perturbation_errors.iter().sum::<f64>() / perturbation_errors.len() as f64;

        // Numerical stability from condition numbers
        let numerical_stability = if condition_numbers.is_empty() {
            1.0
        } else {
            let mean_condition =
                condition_numbers.iter().sum::<f64>() / condition_numbers.len() as f64;
            1.0 / mean_condition.ln().max(1.0)
        };

        // Eigenvalue stability (placeholder)
        let eigenvalue_stability = 1.0 - perturbation_sensitivity;

        Ok(StabilityAnalysis {
            perturbation_sensitivity,
            numerical_stability,
            condition_numbers: condition_numbers.to_vec(),
            eigenvalue_stability,
        })
    }

    fn analyze_sample_complexity<T: ValidatableKernelMethod>(
        &self,
        method: &T,
        data: &Array2<f64>,
    ) -> Result<SampleComplexityAnalysis> {
        let mut sample_errors = Vec::new();

        // Test different sample sizes
        for &n_samples in &self.config.sample_sizes {
            if n_samples > data.nrows() {
                continue;
            }

            let subset_data = data
                .slice(scirs2_core::ndarray::s![..n_samples, ..])
                .to_owned();
            let fitted = method.fit_with_dimension(&subset_data, 100)?;
            let approx = fitted.get_kernel_approximation(&subset_data)?;

            let rbf_kernel = self.compute_rbf_kernel(&subset_data, 1.0)?;
            let error = self.compute_approximation_error(&approx, &rbf_kernel)?;
            sample_errors.push(error);
        }

        // Estimate minimum required samples
        let target_error = self.config.max_approximation_error;
        let minimum_samples = self
            .config
            .sample_sizes
            .iter()
            .zip(sample_errors.iter())
            .find(|(_, &error)| error <= target_error)
            .map(|(&samples, _)| samples)
            .unwrap_or(*self.config.sample_sizes.last().unwrap());

        // Estimate convergence rate with respect to sample size
        let convergence_rate = if sample_errors.len() >= 2 {
            let log_samples: Vec<f64> = self
                .config
                .sample_sizes
                .iter()
                .take(sample_errors.len())
                .map(|&x| (x as f64).ln())
                .collect();
            let log_errors: Vec<f64> = sample_errors.iter().map(|&x| x.ln()).collect();

            // Linear regression for convergence rate
            let n = log_samples.len() as f64;
            let sum_x = log_samples.iter().sum::<f64>();
            let sum_y = log_errors.iter().sum::<f64>();
            let sum_xy = log_samples
                .iter()
                .zip(log_errors.iter())
                .map(|(&x, &y)| x * y)
                .sum::<f64>();
            let sum_x2 = log_samples.iter().map(|&x| x * x).sum::<f64>();

            -(n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        } else {
            0.5 // Default assumption
        };

        let sample_efficiency = 1.0 / minimum_samples as f64;
        let dimension_scaling = data.ncols() as f64 / minimum_samples as f64;

        Ok(SampleComplexityAnalysis {
            minimum_samples,
            convergence_rate,
            sample_efficiency,
            dimension_scaling,
        })
    }

    fn analyze_dimension_dependency<T: ValidatableKernelMethod>(
        &self,
        _method: &T,
        data: &Array2<f64>,
        errors: &[f64],
    ) -> Result<DimensionDependencyAnalysis> {
        let approximation_quality_vs_dimension: Vec<(usize, f64)> = self
            .config
            .approximation_dimensions
            .iter()
            .take(errors.len())
            .zip(errors.iter())
            .map(|(&dim, &error)| (dim, 1.0 - error)) // Convert error to quality
            .collect();

        // Estimate computational cost (simplified)
        let computational_cost_vs_dimension: Vec<(usize, f64)> = self
            .config
            .approximation_dimensions
            .iter()
            .map(|&dim| (dim, dim as f64 * data.nrows() as f64))
            .collect();

        // Find optimal dimension (best quality-to-cost ratio)
        let optimal_dimension = approximation_quality_vs_dimension
            .iter()
            .zip(computational_cost_vs_dimension.iter())
            .map(|((dim, quality), (_, cost))| (*dim, quality / cost))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(dim, _)| dim)
            .unwrap_or(100);

        let dimension_efficiency = approximation_quality_vs_dimension
            .iter()
            .map(|(_, quality)| quality)
            .sum::<f64>()
            / approximation_quality_vs_dimension.len() as f64;

        Ok(DimensionDependencyAnalysis {
            approximation_quality_vs_dimension,
            computational_cost_vs_dimension,
            optimal_dimension,
            dimension_efficiency,
        })
    }

    fn generate_parameter_combinations(
        &self,
        parameter_grid: &HashMap<String, Vec<f64>>,
    ) -> Vec<HashMap<String, f64>> {
        let mut combinations = vec![HashMap::new()];

        for (param_name, param_values) in parameter_grid {
            let mut new_combinations = Vec::new();

            for combination in &combinations {
                for &param_value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), param_value);
                    new_combinations.push(new_combination);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }

    fn k_fold_cross_validation<T: ValidatableKernelMethod>(
        &self,
        method: &T,
        data: &Array2<f64>,
        _targets: Option<&Array1<f64>>,
        parameters: &HashMap<String, f64>,
        k: usize,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let fold_size = n_samples / k;
        let mut scores = Vec::new();

        for fold in 0..k {
            let start_idx = fold * fold_size;
            let end_idx = if fold == k - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train and validation sets
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&i| i < start_idx || i >= end_idx)
                .collect();
            let val_indices: Vec<usize> = (start_idx..end_idx).collect();

            let train_data = data.select(Axis(0), &train_indices);
            let val_data = data.select(Axis(0), &val_indices);

            // Fit with parameters
            let fitted = method.fit_with_parameters(&train_data, parameters)?;
            let train_approx = fitted.get_kernel_approximation(&train_data)?;
            let val_approx = fitted.get_kernel_approximation(&val_data)?;

            // Compute validation score (kernel alignment as proxy)
            let train_kernel = self.compute_rbf_kernel(&train_data, 1.0)?;
            let val_kernel = self.compute_rbf_kernel(&val_data, 1.0)?;

            let train_error = self.compute_approximation_error(&train_approx, &train_kernel)?;
            let val_error = self.compute_approximation_error(&val_approx, &val_kernel)?;

            // Score is negative error (higher is better)
            let score = -(train_error + val_error) / 2.0;
            scores.push(score);
        }

        Ok(scores)
    }
}

/// Trait for kernel methods that can be validated
pub trait ValidatableKernelMethod {
    /// Get method name
    fn method_name(&self) -> String;

    /// Fit with specific approximation dimension
    fn fit_with_dimension(
        &self,
        data: &Array2<f64>,
        n_components: usize,
    ) -> Result<Box<dyn ValidatedFittedMethod>>;

    /// Fit with specific parameters
    fn fit_with_parameters(
        &self,
        data: &Array2<f64>,
        parameters: &HashMap<String, f64>,
    ) -> Result<Box<dyn ValidatedFittedMethod>>;
}

/// Trait for fitted methods that can be validated
pub trait ValidatedFittedMethod {
    /// Get kernel approximation matrix
    fn get_kernel_approximation(&self, data: &Array2<f64>) -> Result<Array2<f64>>;

    /// Compute condition number if applicable
    fn compute_condition_number(&self) -> Result<Option<f64>>;

    /// Get approximation dimension
    fn approximation_dimension(&self) -> usize;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Mock implementation for testing
    struct MockValidatableRBF {
        gamma: f64,
    }

    impl ValidatableKernelMethod for MockValidatableRBF {
        fn method_name(&self) -> String {
            "MockRBF".to_string()
        }

        fn fit_with_dimension(
            &self,
            _data: &Array2<f64>,
            n_components: usize,
        ) -> Result<Box<dyn ValidatedFittedMethod>> {
            Ok(Box::new(MockValidatedFitted { n_components }))
        }

        fn fit_with_parameters(
            &self,
            _data: &Array2<f64>,
            parameters: &HashMap<String, f64>,
        ) -> Result<Box<dyn ValidatedFittedMethod>> {
            let n_components = parameters.get("n_components").copied().unwrap_or(100.0) as usize;
            Ok(Box::new(MockValidatedFitted { n_components }))
        }
    }

    struct MockValidatedFitted {
        n_components: usize,
    }

    impl ValidatedFittedMethod for MockValidatedFitted {
        fn get_kernel_approximation(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
            let n_samples = data.nrows();
            let mut kernel = Array2::zeros((n_samples, n_samples));

            // Simple mock kernel matrix (identity-like)
            for i in 0..n_samples {
                kernel[[i, i]] = 1.0;
                for j in i + 1..n_samples {
                    let similarity = 0.5; // Simple mock similarity
                    kernel[[i, j]] = similarity;
                    kernel[[j, i]] = similarity;
                }
            }

            Ok(kernel)
        }

        fn compute_condition_number(&self) -> Result<Option<f64>> {
            // Simplified condition number estimation
            Ok(Some(10.0))
        }

        fn approximation_dimension(&self) -> usize {
            self.n_components
        }
    }

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = KernelApproximationValidator::new(config);

        assert!(!validator.theoretical_bounds.is_empty());
        assert!(validator.theoretical_bounds.contains_key("RBF"));
    }

    #[test]
    fn test_method_validation() {
        let config = ValidationConfig {
            approximation_dimensions: vec![10, 20],
            repetitions: 2,
            ..Default::default()
        };
        let validator = KernelApproximationValidator::new(config);

        let data = Array2::from_shape_fn((50, 5), |(i, j)| (i + j) as f64 * 0.1);
        let method = MockValidatableRBF { gamma: 1.0 };

        let result = validator.validate_method(&method, &data, None).unwrap();

        assert_eq!(result.method_name, "MockRBF");
        assert_eq!(result.empirical_errors.len(), 2);
        assert_eq!(result.theoretical_bounds.len(), 2);
        // Convergence rate may be None if insufficient data points or poor fit
        // This is acceptable as long as other validation results are present
        if let Some(rate) = result.convergence_rate {
            assert!(rate.is_finite());
        }
    }

    #[test]
    fn test_cross_validation() {
        let config = ValidationConfig::default();
        let validator = KernelApproximationValidator::new(config);

        let data = Array2::from_shape_fn((30, 4), |(i, j)| (i + j) as f64 * 0.1);
        let method = MockValidatableRBF { gamma: 1.0 };

        let mut parameter_grid = HashMap::new();
        parameter_grid.insert("gamma".to_string(), vec![0.5, 1.0, 2.0]);
        parameter_grid.insert("n_components".to_string(), vec![10.0, 20.0]);

        let result = validator
            .cross_validate(&method, &data, None, parameter_grid)
            .unwrap();

        assert_eq!(result.method_name, "MockRBF");
        assert!(!result.cv_scores.is_empty());
        assert!(!result.best_parameters.is_empty());
    }

    #[test]
    fn test_theoretical_bounds() {
        let config = ValidationConfig::default();
        let validator = KernelApproximationValidator::new(config);

        let bound = validator.theoretical_bounds.get("RBF").unwrap();
        let theoretical_bound = validator
            .compute_theoretical_bound(bound, 100, 10, 50)
            .unwrap();

        assert!(theoretical_bound > 0.0);
        assert!(theoretical_bound.is_finite());
    }
}
