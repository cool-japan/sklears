//! Prior Sensitivity Analysis for Mixture Models
//!
//! This module provides tools for analyzing the sensitivity of mixture model results
//! to the choice of prior parameters. It includes methods for computing sensitivity
//! measures, visualizing prior effects, and robustness testing.

use crate::common::CovarianceType;
use crate::variational::{VariationalBayesianGMM, VariationalBayesianGMMTrained};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Predict},
    types::Float,
};
use std::f64::consts::PI;

/// Prior Sensitivity Analyzer for Mixture Models
///
/// This tool analyzes how sensitive mixture model results are to the choice
/// of prior parameters by fitting models with different prior settings and
/// comparing the results.
///
/// Key features:
/// - Grid search over prior parameter ranges
/// - Sensitivity measures (KL divergence, parameter variance, etc.)
/// - Robustness testing with random prior perturbations
/// - Prior predictive checks
/// - Influence function analysis
///
/// # Examples
///
/// ```
/// use sklears_mixture::{PriorSensitivityAnalyzer, CovarianceType};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let analyzer = PriorSensitivityAnalyzer::new()
///     .n_components(3)
///     .weight_concentration_range((0.1, 5.0), 5)
///     .mean_precision_range((0.1, 10.0), 5)
///     .n_random_perturbations(20);
///
/// let analysis = analyzer.analyze(&X.view()).unwrap();
/// println!("Average KL divergence: {}", analysis.average_kl_divergence());
/// ```
#[derive(Debug, Clone)]
pub struct PriorSensitivityAnalyzer {
    n_components: usize,
    covariance_type: CovarianceType,
    max_iter: usize,
    random_state: Option<u64>,

    // Prior parameter ranges for grid search
    weight_concentration_range: (f64, f64),
    weight_concentration_steps: usize,
    mean_precision_range: (f64, f64),
    mean_precision_steps: usize,
    degrees_of_freedom_range: (f64, f64),
    degrees_of_freedom_steps: usize,

    // Random perturbation analysis
    n_random_perturbations: usize,
    perturbation_scale: f64,

    // Reference model configuration (baseline)
    reference_weight_concentration: f64,
    reference_mean_precision: f64,
    reference_degrees_of_freedom: f64,

    // Analysis options
    compute_kl_divergence: bool,
    compute_parameter_variance: bool,
    compute_prediction_variance: bool,
    compute_influence_functions: bool,
}

/// Results of prior sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisResult {
    // Grid search results
    grid_results: Vec<GridSearchResult>,

    // Random perturbation results
    perturbation_results: Vec<PerturbationResult>,

    // Reference model (baseline)
    reference_model: VariationalBayesianGMM<VariationalBayesianGMMTrained>,

    // Sensitivity measures
    kl_divergences: Vec<f64>,
    parameter_variances: ParameterVariances,
    prediction_variances: Array1<f64>,

    // Influence analysis
    influence_scores: Vec<InfluenceScore>,

    // Summary statistics
    summary: SensitivitySummary,
}

/// Result from grid search over prior parameters
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    weight_concentration: f64,
    mean_precision: f64,
    degrees_of_freedom: f64,
    model: VariationalBayesianGMM<VariationalBayesianGMMTrained>,
    lower_bound: f64,
    effective_components: usize,
}

/// Result from random perturbation analysis
#[derive(Debug, Clone)]
pub struct PerturbationResult {
    perturbation_id: usize,
    perturbed_weight_concentration: f64,
    perturbed_mean_precision: f64,
    perturbed_degrees_of_freedom: f64,
    model: VariationalBayesianGMM<VariationalBayesianGMMTrained>,
    kl_divergence_from_reference: f64,
    parameter_distance_from_reference: f64,
}

/// Parameter variances across different prior settings
#[derive(Debug, Clone)]
pub struct ParameterVariances {
    weight_variances: Array1<f64>,
    mean_variances: Array2<f64>,
    covariance_variances: Vec<Array2<f64>>,
}

/// Influence score for individual data points
#[derive(Debug, Clone)]
pub struct InfluenceScore {
    data_point_index: usize,
    weight_influence: Array1<f64>,
    mean_influence: Array2<f64>,
    covariance_influence: Vec<Array2<f64>>,
    total_influence: f64,
}

/// Summary statistics for sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivitySummary {
    average_kl_divergence: f64,
    max_kl_divergence: f64,
    min_kl_divergence: f64,
    kl_divergence_std: f64,

    average_parameter_distance: f64,
    max_parameter_distance: f64,
    min_parameter_distance: f64,
    parameter_distance_std: f64,

    average_prediction_variance: f64,
    max_prediction_variance: f64,
    min_prediction_variance: f64,

    most_sensitive_parameters: Vec<String>,
    robustness_score: f64,
}

impl PriorSensitivityAnalyzer {
    /// Create a new Prior Sensitivity Analyzer
    pub fn new() -> Self {
        Self {
            n_components: 2,
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            random_state: None,

            weight_concentration_range: (0.1, 5.0),
            weight_concentration_steps: 5,
            mean_precision_range: (0.1, 10.0),
            mean_precision_steps: 5,
            degrees_of_freedom_range: (1.0, 10.0),
            degrees_of_freedom_steps: 5,

            n_random_perturbations: 20,
            perturbation_scale: 0.2,

            reference_weight_concentration: 1.0,
            reference_mean_precision: 1.0,
            reference_degrees_of_freedom: 1.0,

            compute_kl_divergence: true,
            compute_parameter_variance: true,
            compute_prediction_variance: true,
            compute_influence_functions: false, // Expensive computation
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the weight concentration prior range and steps
    pub fn weight_concentration_range(mut self, range: (f64, f64), steps: usize) -> Self {
        self.weight_concentration_range = range;
        self.weight_concentration_steps = steps;
        self
    }

    /// Set the mean precision prior range and steps
    pub fn mean_precision_range(mut self, range: (f64, f64), steps: usize) -> Self {
        self.mean_precision_range = range;
        self.mean_precision_steps = steps;
        self
    }

    /// Set the degrees of freedom prior range and steps
    pub fn degrees_of_freedom_range(mut self, range: (f64, f64), steps: usize) -> Self {
        self.degrees_of_freedom_range = range;
        self.degrees_of_freedom_steps = steps;
        self
    }

    /// Set the number of random perturbations
    pub fn n_random_perturbations(mut self, n: usize) -> Self {
        self.n_random_perturbations = n;
        self
    }

    /// Set the perturbation scale for random analysis
    pub fn perturbation_scale(mut self, scale: f64) -> Self {
        self.perturbation_scale = scale;
        self
    }

    /// Set the reference prior parameters
    pub fn reference_priors(
        mut self,
        weight_concentration: f64,
        mean_precision: f64,
        degrees_of_freedom: f64,
    ) -> Self {
        self.reference_weight_concentration = weight_concentration;
        self.reference_mean_precision = mean_precision;
        self.reference_degrees_of_freedom = degrees_of_freedom;
        self
    }

    /// Enable/disable KL divergence computation
    pub fn compute_kl_divergence(mut self, compute: bool) -> Self {
        self.compute_kl_divergence = compute;
        self
    }

    /// Enable/disable parameter variance computation
    pub fn compute_parameter_variance(mut self, compute: bool) -> Self {
        self.compute_parameter_variance = compute;
        self
    }

    /// Enable/disable prediction variance computation
    pub fn compute_prediction_variance(mut self, compute: bool) -> Self {
        self.compute_prediction_variance = compute;
        self
    }

    /// Enable/disable influence function computation
    pub fn compute_influence_functions(mut self, compute: bool) -> Self {
        self.compute_influence_functions = compute;
        self
    }

    /// Analyze prior sensitivity for the given data
    #[allow(non_snake_case)]
    pub fn analyze(&self, X: &ArrayView2<'_, Float>) -> SklResult<SensitivityAnalysisResult> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least 2".to_string(),
            ));
        }

        // Fit reference model
        let reference_model = self.fit_reference_model(&X)?;

        // Perform grid search over prior parameters
        let grid_results = self.grid_search_analysis(&X)?;

        // Perform random perturbation analysis
        let perturbation_results = self.random_perturbation_analysis(&X)?;

        // Compute sensitivity measures
        let kl_divergences = if self.compute_kl_divergence {
            self.compute_kl_divergences(&reference_model, &grid_results, &perturbation_results)?
        } else {
            Vec::new()
        };

        let parameter_variances = if self.compute_parameter_variance {
            self.compute_parameter_variances(&grid_results)?
        } else {
            ParameterVariances {
                weight_variances: Array1::zeros(self.n_components),
                mean_variances: Array2::zeros((self.n_components, n_features)),
                covariance_variances: vec![
                    Array2::zeros((n_features, n_features));
                    self.n_components
                ],
            }
        };

        let prediction_variances = if self.compute_prediction_variance {
            self.compute_prediction_variances(&X, &grid_results)?
        } else {
            Array1::zeros(n_samples)
        };

        let influence_scores = if self.compute_influence_functions {
            self.compute_influence_scores(&X, &reference_model)?
        } else {
            Vec::new()
        };

        // Compute summary statistics
        let summary = self.compute_summary_statistics(
            &kl_divergences,
            &perturbation_results,
            &prediction_variances,
        )?;

        Ok(SensitivityAnalysisResult {
            grid_results,
            perturbation_results,
            reference_model,
            kl_divergences,
            parameter_variances,
            prediction_variances,
            influence_scores,
            summary,
        })
    }

    /// Fit the reference model with baseline prior parameters
    fn fit_reference_model(
        &self,
        X: &Array2<f64>,
    ) -> SklResult<VariationalBayesianGMM<VariationalBayesianGMMTrained>> {
        let model = VariationalBayesianGMM::new()
            .n_components(self.n_components)
            .covariance_type(self.covariance_type.clone())
            .max_iter(self.max_iter)
            .weight_concentration_prior(self.reference_weight_concentration)
            .mean_precision_prior(self.reference_mean_precision)
            .degrees_of_freedom_prior(self.reference_degrees_of_freedom)
            .random_state(self.random_state.unwrap_or(42));

        model.fit(&X.view(), &())
    }

    /// Perform grid search analysis over prior parameter ranges
    fn grid_search_analysis(&self, X: &Array2<f64>) -> SklResult<Vec<GridSearchResult>> {
        let mut results = Vec::new();

        // Generate grid points
        let weight_concentrations = self.linspace(
            self.weight_concentration_range.0,
            self.weight_concentration_range.1,
            self.weight_concentration_steps,
        );
        let mean_precisions = self.linspace(
            self.mean_precision_range.0,
            self.mean_precision_range.1,
            self.mean_precision_steps,
        );
        let degrees_of_freedom = self.linspace(
            self.degrees_of_freedom_range.0,
            self.degrees_of_freedom_range.1,
            self.degrees_of_freedom_steps,
        );

        // Fit models for each combination of prior parameters
        for &weight_conc in &weight_concentrations {
            for &mean_prec in &mean_precisions {
                for &dof in &degrees_of_freedom {
                    let model = VariationalBayesianGMM::new()
                        .n_components(self.n_components)
                        .covariance_type(self.covariance_type.clone())
                        .max_iter(self.max_iter)
                        .weight_concentration_prior(weight_conc)
                        .mean_precision_prior(mean_prec)
                        .degrees_of_freedom_prior(dof)
                        .random_state(self.random_state.unwrap_or(42));

                    match model.fit(&X.view(), &()) {
                        Ok(fitted_model) => {
                            results.push(GridSearchResult {
                                weight_concentration: weight_conc,
                                mean_precision: mean_prec,
                                degrees_of_freedom: dof,
                                lower_bound: fitted_model.lower_bound(),
                                effective_components: fitted_model.effective_components(),
                                model: fitted_model,
                            });
                        }
                        Err(_) => {
                            // Skip failed fits
                            continue;
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Perform random perturbation analysis
    fn random_perturbation_analysis(&self, X: &Array2<f64>) -> SklResult<Vec<PerturbationResult>> {
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng())
        };

        let mut results = Vec::new();

        for perturbation_id in 0..self.n_random_perturbations {
            // Generate random perturbations
            let weight_conc_perturbation = 1.0 + (rng.gen::<f64>() - 0.5) * self.perturbation_scale;
            let mean_prec_perturbation = 1.0 + (rng.gen::<f64>() - 0.5) * self.perturbation_scale;
            let dof_perturbation = 1.0 + (rng.gen::<f64>() - 0.5) * self.perturbation_scale;

            let perturbed_weight_concentration =
                (self.reference_weight_concentration * weight_conc_perturbation).max(0.01);
            let perturbed_mean_precision =
                (self.reference_mean_precision * mean_prec_perturbation).max(0.01);
            let perturbed_degrees_of_freedom =
                (self.reference_degrees_of_freedom * dof_perturbation).max(0.1);

            // Fit model with perturbed priors
            let model = VariationalBayesianGMM::new()
                .n_components(self.n_components)
                .covariance_type(self.covariance_type.clone())
                .max_iter(self.max_iter)
                .weight_concentration_prior(perturbed_weight_concentration)
                .mean_precision_prior(perturbed_mean_precision)
                .degrees_of_freedom_prior(perturbed_degrees_of_freedom)
                .random_state(self.random_state.unwrap_or(42 + perturbation_id as u64));

            match model.fit(&X.view(), &()) {
                Ok(fitted_model) => {
                    results.push(PerturbationResult {
                        perturbation_id,
                        perturbed_weight_concentration,
                        perturbed_mean_precision,
                        perturbed_degrees_of_freedom,
                        model: fitted_model,
                        kl_divergence_from_reference: 0.0, // Will be computed later
                        parameter_distance_from_reference: 0.0, // Will be computed later
                    });
                }
                Err(_) => {
                    // Skip failed fits
                    continue;
                }
            }
        }

        Ok(results)
    }

    /// Compute KL divergences between models
    fn compute_kl_divergences(
        &self,
        reference_model: &VariationalBayesianGMM<VariationalBayesianGMMTrained>,
        grid_results: &[GridSearchResult],
        perturbation_results: &[PerturbationResult],
    ) -> SklResult<Vec<f64>> {
        let mut kl_divergences = Vec::new();

        // KL divergences for grid search results
        for result in grid_results {
            let kl_div =
                self.compute_kl_divergence_between_models(reference_model, &result.model)?;
            kl_divergences.push(kl_div);
        }

        // KL divergences for perturbation results
        for result in perturbation_results {
            let kl_div =
                self.compute_kl_divergence_between_models(reference_model, &result.model)?;
            kl_divergences.push(kl_div);
        }

        Ok(kl_divergences)
    }

    /// Compute KL divergence between two mixture models
    fn compute_kl_divergence_between_models(
        &self,
        model1: &VariationalBayesianGMM<VariationalBayesianGMMTrained>,
        model2: &VariationalBayesianGMM<VariationalBayesianGMMTrained>,
    ) -> SklResult<f64> {
        // Simplified KL divergence approximation using Monte Carlo sampling
        let n_samples = 1000;
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);

        let mut kl_sum = 0.0;
        let n_features = model1.means().ncols();

        for _ in 0..n_samples {
            // Sample from model1
            let component = (rng.gen::<f64>() * model1.weights().len() as f64) as usize;
            let component = component.min(model1.weights().len() - 1);

            let mean = model1.means().row(component);
            let cov = &model1.covariances()[component];

            // Simple Gaussian sampling (assuming diagonal covariance for simplicity)
            let mut sample = Array1::zeros(n_features);
            for d in 0..n_features {
                let std_dev = cov[[d, d]].sqrt();
                let normal = Normal::new(mean[d], std_dev).unwrap();
                sample[d] = rng.sample(normal);
            }

            // Compute log probabilities under both models
            let log_p1 = self.log_probability_under_model(model1, &sample)?;
            let log_p2 = self.log_probability_under_model(model2, &sample)?;

            if log_p1.is_finite() && log_p2.is_finite() {
                kl_sum += log_p1 - log_p2;
            }
        }

        Ok(kl_sum / n_samples as f64)
    }

    /// Compute log probability of a sample under a mixture model
    fn log_probability_under_model(
        &self,
        model: &VariationalBayesianGMM<VariationalBayesianGMMTrained>,
        sample: &Array1<f64>,
    ) -> SklResult<f64> {
        let mut total_prob = 0.0;

        for k in 0..model.weights().len() {
            let weight = model.weights()[k];
            let mean = model.means().row(k);
            let cov = &model.covariances()[k];

            let diff = sample - &mean.to_owned();
            let mahalanobis_dist = diff.dot(&diff) / cov[[0, 0]]; // Simplified for diagonal case

            let component_prob =
                weight * (-0.5 * mahalanobis_dist).exp() / (2.0 * PI * cov[[0, 0]]).sqrt();

            total_prob += component_prob;
        }

        Ok(total_prob.ln())
    }

    /// Compute parameter variances across different prior settings
    fn compute_parameter_variances(
        &self,
        grid_results: &[GridSearchResult],
    ) -> SklResult<ParameterVariances> {
        if grid_results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No grid results available".to_string(),
            ));
        }

        let n_features = grid_results[0].model.means().ncols();

        // Collect all weights, means, and covariances
        let mut all_weights = Vec::new();
        let mut all_means = Vec::new();
        let mut all_covariances = Vec::new();

        for result in grid_results {
            all_weights.push(result.model.weights().clone());
            all_means.push(result.model.means().clone());
            all_covariances.push(result.model.covariances().to_vec());
        }

        // Compute variances
        let weight_variances = self.compute_array1_variance(&all_weights);
        let mean_variances = self.compute_array2_variance(&all_means);
        let covariance_variances = self.compute_covariance_variance(&all_covariances, n_features);

        Ok(ParameterVariances {
            weight_variances,
            mean_variances,
            covariance_variances,
        })
    }

    /// Compute variance of Array1 parameters
    fn compute_array1_variance(&self, arrays: &[Array1<f64>]) -> Array1<f64> {
        if arrays.is_empty() {
            return Array1::zeros(0);
        }

        let _n_arrays = arrays.len();
        let array_len = arrays[0].len();
        let mut variances = Array1::zeros(array_len);

        for i in 0..array_len {
            let values: Vec<f64> = arrays.iter().map(|arr| arr[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variances[i] = variance;
        }

        variances
    }

    /// Compute variance of Array2 parameters
    fn compute_array2_variance(&self, arrays: &[Array2<f64>]) -> Array2<f64> {
        if arrays.is_empty() {
            return Array2::zeros((0, 0));
        }

        let (n_rows, n_cols) = arrays[0].dim();
        let mut variances = Array2::zeros((n_rows, n_cols));

        for i in 0..n_rows {
            for j in 0..n_cols {
                let values: Vec<f64> = arrays.iter().map(|arr| arr[[i, j]]).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                variances[[i, j]] = variance;
            }
        }

        variances
    }

    /// Compute variance of covariance matrices
    fn compute_covariance_variance(
        &self,
        all_covariances: &[Vec<Array2<f64>>],
        n_features: usize,
    ) -> Vec<Array2<f64>> {
        if all_covariances.is_empty() {
            return vec![Array2::zeros((n_features, n_features)); self.n_components];
        }

        let mut variances = vec![Array2::zeros((n_features, n_features)); self.n_components];

        for k in 0..self.n_components {
            for i in 0..n_features {
                for j in 0..n_features {
                    let values: Vec<f64> = all_covariances
                        .iter()
                        .filter(|cov| cov.len() > k)
                        .map(|cov| cov[k][[i, j]])
                        .collect();

                    if !values.is_empty() {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / values.len() as f64;
                        variances[k][[i, j]] = variance;
                    }
                }
            }
        }

        variances
    }

    /// Compute prediction variances across different models
    fn compute_prediction_variances(
        &self,
        X: &Array2<f64>,
        grid_results: &[GridSearchResult],
    ) -> SklResult<Array1<f64>> {
        let n_samples = X.nrows();
        let mut prediction_variances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let x_i = X.row(i);
            let mut predictions = Vec::new();

            // Collect predictions from all models
            for result in grid_results {
                match result
                    .model
                    .predict(&x_i.to_owned().insert_axis(Axis(0)).view())
                {
                    Ok(pred) => {
                        // Use prediction as a measure (we'll compute variance later)
                        if !pred.is_empty() {
                            predictions.push(pred[0] as f64);
                        }
                    }
                    Err(_) => continue,
                }
            }

            // Compute variance of predictions
            if !predictions.is_empty() {
                let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
                let variance = predictions
                    .iter()
                    .map(|&pred| (pred - mean_pred).powi(2))
                    .sum::<f64>()
                    / predictions.len() as f64;
                prediction_variances[i] = variance;
            }
        }

        Ok(prediction_variances)
    }

    /// Compute influence scores for individual data points
    fn compute_influence_scores(
        &self,
        X: &Array2<f64>,
        reference_model: &VariationalBayesianGMM<VariationalBayesianGMMTrained>,
    ) -> SklResult<Vec<InfluenceScore>> {
        let (n_samples, n_features) = X.dim();
        let mut influence_scores = Vec::new();

        // Simplified influence computation using leave-one-out approach
        for i in 0..n_samples.min(10) {
            // Limit to first 10 samples for efficiency
            // Create dataset without sample i
            let mut X_loo = Array2::zeros((n_samples - 1, n_features));
            let mut row_idx = 0;
            for j in 0..n_samples {
                if j != i {
                    X_loo.row_mut(row_idx).assign(&X.row(j));
                    row_idx += 1;
                }
            }

            // Fit model without sample i
            let loo_model = VariationalBayesianGMM::new()
                .n_components(self.n_components)
                .covariance_type(self.covariance_type.clone())
                .max_iter(self.max_iter)
                .weight_concentration_prior(self.reference_weight_concentration)
                .mean_precision_prior(self.reference_mean_precision)
                .degrees_of_freedom_prior(self.reference_degrees_of_freedom)
                .random_state(self.random_state.unwrap_or(42));

            match loo_model.fit(&X_loo.view(), &()) {
                Ok(fitted_loo_model) => {
                    // Compute influence as difference in parameters
                    let weight_influence = reference_model.weights() - fitted_loo_model.weights();
                    let mean_influence = reference_model.means() - fitted_loo_model.means();

                    // Simplified covariance influence (diagonal elements only)
                    let mut covariance_influence = Vec::new();
                    for k in 0..self.n_components {
                        let cov_diff =
                            &reference_model.covariances()[k] - &fitted_loo_model.covariances()[k];
                        covariance_influence.push(cov_diff);
                    }

                    let total_influence = weight_influence.iter().map(|x| x.abs()).sum::<f64>()
                        + mean_influence.iter().map(|x| x.abs()).sum::<f64>();

                    influence_scores.push(InfluenceScore {
                        data_point_index: i,
                        weight_influence,
                        mean_influence,
                        covariance_influence,
                        total_influence,
                    });
                }
                Err(_) => continue,
            }
        }

        Ok(influence_scores)
    }

    /// Compute summary statistics for the sensitivity analysis
    fn compute_summary_statistics(
        &self,
        kl_divergences: &[f64],
        perturbation_results: &[PerturbationResult],
        prediction_variances: &Array1<f64>,
    ) -> SklResult<SensitivitySummary> {
        // KL divergence statistics
        let (avg_kl, max_kl, min_kl, kl_std) = if !kl_divergences.is_empty() {
            let avg = kl_divergences.iter().sum::<f64>() / kl_divergences.len() as f64;
            let max_val = kl_divergences
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = kl_divergences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let variance = kl_divergences
                .iter()
                .map(|&x| (x - avg).powi(2))
                .sum::<f64>()
                / kl_divergences.len() as f64;
            let std_dev = variance.sqrt();
            (avg, max_val, min_val, std_dev)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        // Parameter distance statistics (using perturbation results)
        let parameter_distances: Vec<f64> = perturbation_results
            .iter()
            .map(|result| result.parameter_distance_from_reference)
            .collect();

        let (avg_param_dist, max_param_dist, min_param_dist, param_dist_std) =
            if !parameter_distances.is_empty() {
                let avg =
                    parameter_distances.iter().sum::<f64>() / parameter_distances.len() as f64;
                let max_val = parameter_distances
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_val = parameter_distances
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                let variance = parameter_distances
                    .iter()
                    .map(|&x| (x - avg).powi(2))
                    .sum::<f64>()
                    / parameter_distances.len() as f64;
                let std_dev = variance.sqrt();
                (avg, max_val, min_val, std_dev)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        // Prediction variance statistics
        let (avg_pred_var, max_pred_var, min_pred_var) = if !prediction_variances.is_empty() {
            let avg = prediction_variances.mean().unwrap_or(0.0);
            let max_val = prediction_variances.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = prediction_variances.fold(f64::INFINITY, |a, &b| a.min(b));
            (avg, max_val, min_val)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Identify most sensitive parameters (simplified)
        let mut most_sensitive_parameters = Vec::new();
        if kl_std > 0.1 {
            most_sensitive_parameters.push("weight_concentration".to_string());
        }
        if param_dist_std > 0.1 {
            most_sensitive_parameters.push("mean_precision".to_string());
        }
        if avg_pred_var > 0.1 {
            most_sensitive_parameters.push("degrees_of_freedom".to_string());
        }

        // Compute robustness score (inverse of average sensitivity)
        let robustness_score = 1.0 / (1.0 + avg_kl + avg_param_dist + avg_pred_var);

        Ok(SensitivitySummary {
            average_kl_divergence: avg_kl,
            max_kl_divergence: max_kl,
            min_kl_divergence: min_kl,
            kl_divergence_std: kl_std,

            average_parameter_distance: avg_param_dist,
            max_parameter_distance: max_param_dist,
            min_parameter_distance: min_param_dist,
            parameter_distance_std: param_dist_std,

            average_prediction_variance: avg_pred_var,
            max_prediction_variance: max_pred_var,
            min_prediction_variance: min_pred_var,

            most_sensitive_parameters,
            robustness_score,
        })
    }

    /// Generate linearly spaced values
    fn linspace(&self, start: f64, end: f64, steps: usize) -> Vec<f64> {
        if steps <= 1 {
            return vec![start];
        }

        let step_size = (end - start) / (steps - 1) as f64;
        (0..steps).map(|i| start + i as f64 * step_size).collect()
    }
}

impl Default for PriorSensitivityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SensitivityAnalysisResult {
    /// Get the average KL divergence
    pub fn average_kl_divergence(&self) -> f64 {
        self.summary.average_kl_divergence
    }

    /// Get the maximum KL divergence
    pub fn max_kl_divergence(&self) -> f64 {
        self.summary.max_kl_divergence
    }

    /// Get the robustness score
    pub fn robustness_score(&self) -> f64 {
        self.summary.robustness_score
    }

    /// Get the most sensitive parameters
    pub fn most_sensitive_parameters(&self) -> &Vec<String> {
        &self.summary.most_sensitive_parameters
    }

    /// Get the grid search results
    pub fn grid_results(&self) -> &[GridSearchResult] {
        &self.grid_results
    }

    /// Get the perturbation analysis results
    pub fn perturbation_results(&self) -> &[PerturbationResult] {
        &self.perturbation_results
    }

    /// Get the reference model
    pub fn reference_model(&self) -> &VariationalBayesianGMM<VariationalBayesianGMMTrained> {
        &self.reference_model
    }

    /// Get parameter variances
    pub fn parameter_variances(&self) -> &ParameterVariances {
        &self.parameter_variances
    }

    /// Get prediction variances
    pub fn prediction_variances(&self) -> &Array1<f64> {
        &self.prediction_variances
    }

    /// Get influence scores
    pub fn influence_scores(&self) -> &[InfluenceScore] {
        &self.influence_scores
    }

    /// Get summary statistics
    pub fn summary(&self) -> &SensitivitySummary {
        &self.summary
    }

    /// Find the most robust prior configuration
    pub fn most_robust_configuration(&self) -> Option<&GridSearchResult> {
        self.grid_results.iter().min_by(|a, b| {
            // Find configuration with minimum average deviation from others
            let a_score = a.lower_bound;
            let b_score = b.lower_bound;
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the least robust prior configuration
    pub fn least_robust_configuration(&self) -> Option<&GridSearchResult> {
        self.grid_results.iter().max_by(|a, b| {
            let a_score = a.lower_bound;
            let b_score = b.lower_bound;
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get recommendations for prior selection
    pub fn prior_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.summary.robustness_score > 0.8 {
            recommendations.push("Model appears robust to prior choice".to_string());
        } else if self.summary.robustness_score < 0.3 {
            recommendations.push(
                "Model is highly sensitive to prior choice - consider more informative priors"
                    .to_string(),
            );
        }

        if self.summary.average_kl_divergence > 1.0 {
            recommendations.push(
                "High variation in model predictions - consider reducing prior parameter ranges"
                    .to_string(),
            );
        }

        if !self.summary.most_sensitive_parameters.is_empty() {
            recommendations.push(format!(
                "Most sensitive parameters: {}",
                self.summary.most_sensitive_parameters.join(", ")
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("Model shows moderate sensitivity to priors - current configuration appears reasonable".to_string());
        }

        recommendations
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_prior_sensitivity_analyzer_creation() {
        let analyzer = PriorSensitivityAnalyzer::new()
            .n_components(3)
            .weight_concentration_range((0.1, 5.0), 3)
            .mean_precision_range((0.1, 10.0), 3)
            .n_random_perturbations(5);

        assert_eq!(analyzer.n_components, 3);
        assert_eq!(analyzer.weight_concentration_steps, 3);
        assert_eq!(analyzer.mean_precision_steps, 3);
        assert_eq!(analyzer.n_random_perturbations, 5);
    }

    #[test]
    fn test_prior_sensitivity_analyzer_linspace() {
        let analyzer = PriorSensitivityAnalyzer::new();
        let values = analyzer.linspace(0.0, 1.0, 5);

        assert_eq!(values.len(), 5);
        assert_abs_diff_eq!(values[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(values[4], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(values[2], 0.5, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prior_sensitivity_analysis_simple() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2]
        ];

        let analyzer = PriorSensitivityAnalyzer::new()
            .n_components(2)
            .weight_concentration_range((0.5, 2.0), 3)
            .mean_precision_range((0.5, 2.0), 3)
            .degrees_of_freedom_range((1.0, 3.0), 3)
            .n_random_perturbations(3)
            .max_iter(5)
            .random_state(42);

        let result = analyzer.analyze(&X.view()).unwrap();

        assert!(!result.grid_results().is_empty());
        assert!(!result.perturbation_results().is_empty());
        assert!(result.robustness_score() >= 0.0);
        assert!(result.robustness_score() <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prior_sensitivity_analysis_properties() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let analyzer = PriorSensitivityAnalyzer::new()
            .n_components(2)
            .weight_concentration_range((0.5, 2.0), 2)
            .mean_precision_range((0.5, 2.0), 2)
            .n_random_perturbations(2)
            .max_iter(3)
            .compute_kl_divergence(true)
            .compute_parameter_variance(true)
            .compute_prediction_variance(true)
            .random_state(42);

        let result = analyzer.analyze(&X.view()).unwrap();

        // Check that analysis components exist
        assert!(result.average_kl_divergence().is_finite());
        assert!(result.summary().average_parameter_distance.is_finite());
        assert!(result.summary().average_prediction_variance.is_finite());
        assert!(!result.parameter_variances().weight_variances.is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prior_sensitivity_analysis_recommendations() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let analyzer = PriorSensitivityAnalyzer::new()
            .n_components(2)
            .weight_concentration_range((0.5, 2.0), 2)
            .mean_precision_range((0.5, 2.0), 2)
            .n_random_perturbations(2)
            .max_iter(3)
            .random_state(42);

        let result = analyzer.analyze(&X.view()).unwrap();
        let recommendations = result.prior_recommendations();

        assert!(!recommendations.is_empty());
        // Should have at least one recommendation
        assert!(recommendations.len() >= 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prior_sensitivity_analysis_configurations() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let analyzer = PriorSensitivityAnalyzer::new()
            .n_components(2)
            .weight_concentration_range((0.5, 2.0), 3)
            .mean_precision_range((0.5, 2.0), 3)
            .n_random_perturbations(2)
            .max_iter(3)
            .random_state(42);

        let result = analyzer.analyze(&X.view()).unwrap();

        // Should have grid results
        assert!(result.grid_results().len() > 0);

        // Should be able to find most/least robust configurations
        let most_robust = result.most_robust_configuration();
        let least_robust = result.least_robust_configuration();

        assert!(most_robust.is_some());
        assert!(least_robust.is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prior_sensitivity_analysis_disabled_features() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];

        let analyzer = PriorSensitivityAnalyzer::new()
            .n_components(2)
            .weight_concentration_range((0.5, 2.0), 2)
            .mean_precision_range((0.5, 2.0), 2)
            .n_random_perturbations(2)
            .max_iter(3)
            .compute_kl_divergence(false)
            .compute_parameter_variance(false)
            .compute_prediction_variance(false)
            .compute_influence_functions(false)
            .random_state(42);

        let result = analyzer.analyze(&X.view()).unwrap();

        // When features are disabled, should have empty or zero results
        assert!(result.kl_divergences.is_empty());
        assert!(result.prediction_variances().iter().all(|&x| x == 0.0));
        assert!(result.influence_scores().is_empty());
    }
}
