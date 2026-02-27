//! Clustering Stability Analysis
//!
//! This module provides comprehensive clustering stability analysis methods for evaluating
//! the robustness and consistency of clustering algorithms.
//!
//! # Stability Analysis Methods
//! - Subsample (Bootstrap) stability analysis
//! - Consensus clustering with multiple seeds
//! - Perturbation stability through noise injection
//! - Parameter sensitivity analysis across parameter ranges
//! - Cross-validation based stability analysis
//!
//! # Mathematical Background
//!
//! ## Subsample Stability
//! Evaluates stability by repeatedly subsampling the data and measuring consistency
//! of cluster assignments using Adjusted Rand Index (ARI) between overlapping samples.
//!
//! ## Consensus Stability
//! Runs the same algorithm multiple times with different random seeds and measures
//! the consistency of results using pairwise ARI comparisons.
//!
//! ## Perturbation Stability
//! Tests robustness by adding controlled Gaussian noise to the data and measuring
//! how clustering results degrade as noise levels increase.
//!
//! ## Parameter Sensitivity
//! Analyzes how sensitive clustering results are to parameter changes by testing
//! stability across a range of parameter values.

use std::collections::{HashMap, HashSet};

use scirs2_core::ndarray::Array2;
use scirs2_core::rand_prelude::{Distribution, SliceRandom};
use scirs2_core::random::{thread_rng, RandNormal, Rng};
use sklears_core::error::{Result, SklearsError};

use super::internal_validation::ClusteringValidator;
use super::validation_types::ValidationMetric;

/// Wrapper for f64 to use as HashMap key
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HashableFloat(f64);

impl std::cmp::Eq for HashableFloat {}

impl std::hash::Hash for HashableFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Convert to bits for consistent hashing, handle NaN as zero
        if self.0.is_nan() {
            0u64.hash(state)
        } else {
            self.0.to_bits().hash(state)
        }
    }
}

impl From<f64> for HashableFloat {
    fn from(f: f64) -> Self {
        HashableFloat(f)
    }
}

impl From<HashableFloat> for f64 {
    fn from(hf: HashableFloat) -> Self {
        hf.0
    }
}

impl HashableFloat {
    /// Returns the underlying floating point value.
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

/// Result of subsample stability analysis
#[derive(Debug, Clone)]
pub struct SubsampleStabilityResult {
    /// Mean stability across all pairwise comparisons
    pub mean_stability: f64,
    /// Standard deviation of stability scores
    pub std_stability: f64,
    /// Minimum stability observed
    pub min_stability: f64,
    /// Maximum stability observed
    pub max_stability: f64,
    /// Number of successful clustering trials
    pub n_successful_trials: usize,
    /// All pairwise stability scores
    pub pairwise_stabilities: Vec<f64>,
    /// Subsample ratio used
    pub subsample_ratio: f64,
}

/// Result of consensus clustering stability analysis
#[derive(Debug, Clone)]
pub struct ConsensusStabilityResult {
    /// Mean stability across all pairwise comparisons
    pub mean_stability: f64,
    /// Standard deviation of stability scores
    pub std_stability: f64,
    /// Minimum stability observed
    pub min_stability: f64,
    /// Maximum stability observed
    pub max_stability: f64,
    /// Mean cluster consistency score
    pub mean_cluster_consistency: f64,
    /// All pairwise stability scores
    pub pairwise_scores: Vec<f64>,
    /// Cluster consistency scores
    pub cluster_consistency_scores: Vec<f64>,
    /// Consensus matrix showing co-clustering probabilities
    pub consensus_matrix: Array2<f64>,
    /// Number of successful clustering runs
    pub n_successful_runs: usize,
    /// Seeds used for random initialization
    pub seeds_used: Vec<u64>,
}

/// Result of bootstrap stability analysis
#[derive(Debug, Clone)]
pub struct BootstrapStabilityResult {
    /// Mean bootstrap stability
    pub mean_stability: f64,
    /// Standard deviation of bootstrap stability
    pub std_stability: f64,
    /// Confidence interval for stability (if computed)
    pub confidence_interval: Option<(f64, f64)>,
    /// Individual bootstrap stability scores
    pub bootstrap_scores: Vec<f64>,
    /// Number of bootstrap samples
    pub n_bootstrap_samples: usize,
}

/// Result for a specific noise level in perturbation analysis
#[derive(Debug, Clone)]
pub struct NoiseStabilityResult {
    /// Noise level (standard deviation)
    pub noise_level: f64,
    /// Mean stability at this noise level
    pub mean_stability: f64,
    /// Standard deviation of stability
    pub std_stability: f64,
    /// Minimum stability observed
    pub min_stability: f64,
    /// Maximum stability observed
    pub max_stability: f64,
    /// Number of successful trials at this noise level
    pub n_successful_trials: usize,
    /// Individual trial stability scores
    pub trial_stabilities: Vec<f64>,
}

/// Result of perturbation stability analysis
#[derive(Debug, Clone)]
pub struct PerturbationStabilityResult {
    /// Baseline clustering without noise
    pub baseline_clustering: Vec<i32>,
    /// Results for each noise level tested
    pub noise_level_results: HashMap<HashableFloat, NoiseStabilityResult>,
    /// Stability degradation curve [(noise_level, mean_stability)]
    pub stability_degradation: Vec<(f64, f64)>,
    /// Estimated robustness threshold (noise level where stability drops significantly)
    pub robustness_threshold: Option<f64>,
    /// Noise levels that were tested
    pub tested_noise_levels: Vec<f64>,
}

/// Result of parameter sensitivity analysis
#[derive(Debug, Clone)]
pub struct ParameterSensitivityResult {
    /// Name of the parameter being varied
    pub parameter_name: String,
    /// Parameter values that were tested
    pub tested_parameters: Vec<f64>,
    /// Parameter values that produced successful clustering
    pub successful_parameters: Vec<f64>,
    /// Clustering results for each successful parameter value
    pub parameter_results: HashMap<HashableFloat, Vec<i32>>,
    /// Stability between adjacent parameter values
    pub adjacent_stabilities: Vec<f64>,
    /// Parameter pairs corresponding to adjacent stabilities
    pub parameter_pairs: Vec<(f64, f64)>,
    /// All pairwise stability scores
    pub all_pairwise_stabilities: Vec<f64>,
    /// Mean stability between adjacent parameters
    pub mean_adjacent_stability: f64,
    /// Overall mean stability across all pairs
    pub mean_overall_stability: f64,
    /// Standard deviation of adjacent stability
    pub std_adjacent_stability: f64,
    /// Parameter range where stability is consistently high
    pub stability_plateau: Option<(f64, f64)>,
}

/// Result of cross-validation stability analysis
#[derive(Debug, Clone)]
pub struct CrossValidationStabilityResult {
    /// Mean stability across all fold comparisons
    pub mean_stability: f64,
    /// Standard deviation of stability scores
    pub std_stability: f64,
    /// Minimum stability observed
    pub min_stability: f64,
    /// Maximum stability observed
    pub max_stability: f64,
    /// Mean stability per repeat
    pub mean_repeat_stability: f64,
    /// Individual fold stability scores
    pub fold_stabilities: Vec<f64>,
    /// Stability scores averaged per repeat
    pub repeat_stabilities: Vec<f64>,
    /// Number of folds used
    pub k_folds: usize,
    /// Number of repeats performed
    pub n_repeats: usize,
    /// Total number of successful fold comparisons
    pub n_successful_comparisons: usize,
}

/// Stability analysis methods for clustering evaluation
pub struct StabilityAnalyzer {
    /// Distance metric for validation
    metric: ValidationMetric,
    /// External validation methods for computing ARI, etc.
    external_validator: ClusteringValidator,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer
    pub fn new(metric: ValidationMetric) -> Self {
        Self {
            metric,
            external_validator: ClusteringValidator::euclidean(),
        }
    }

    /// Create analyzer with Euclidean distance
    pub fn euclidean() -> Self {
        Self::new(ValidationMetric::Euclidean)
    }

    /// Create analyzer with Manhattan distance
    pub fn manhattan() -> Self {
        Self::new(ValidationMetric::Manhattan)
    }

    /// Create analyzer with Cosine distance
    pub fn cosine() -> Self {
        Self::new(ValidationMetric::Cosine)
    }

    /// Compute cluster stability using subsampling approach
    ///
    /// This method evaluates clustering stability by repeatedly subsampling
    /// the data and measuring how consistently the same clusters are found.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `clustering_fn` - Clustering function that takes data, returns labels
    /// * `subsample_ratio` - Fraction of data to use in each subsample (0.0 to 1.0)
    /// * `n_trials` - Number of subsampling trials to perform
    ///
    /// # Returns
    /// SubsampleStabilityResult with stability metrics
    ///
    /// # Mathematical Details
    /// For each pair of subsamples i and j with overlap O:
    /// - Compute ARI(labels_i\[O\], labels_j\[O\]) for overlapping samples
    /// - Aggregate ARI scores to compute mean and variance
    pub fn subsample_stability<F>(
        &self,
        X: &Array2<f64>,
        clustering_fn: F,
        subsample_ratio: f64,
        n_trials: usize,
    ) -> Result<SubsampleStabilityResult>
    where
        F: Fn(&Array2<f64>) -> Result<Vec<i32>>,
    {
        if subsample_ratio <= 0.0 || subsample_ratio >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "Subsample ratio must be between 0 and 1".to_string(),
            ));
        }

        if n_trials < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 trials for subsample stability".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let n_samples = X.nrows();
        let subsample_size = ((n_samples as f64) * subsample_ratio) as usize;

        if subsample_size < 2 {
            return Err(SklearsError::InvalidInput(
                "Subsample size too small".to_string(),
            ));
        }

        let mut all_labelings = Vec::new();

        for _trial in 0..n_trials {
            // Create random subsample
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(subsample_size);
            indices.sort();

            // Extract subsample
            let mut subsample = Array2::<f64>::zeros((subsample_size, X.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                subsample.row_mut(i).assign(&X.row(idx));
            }

            // Perform clustering
            match clustering_fn(&subsample) {
                Ok(labels) => {
                    all_labelings.push((indices.clone(), labels));
                }
                Err(_) => {
                    // Skip failed clusterings
                    continue;
                }
            }
        }

        if all_labelings.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Not enough successful clustering trials".to_string(),
            ));
        }

        // Compute pairwise stability between all successful trials
        let mut pairwise_stabilities = Vec::new();

        for i in 0..all_labelings.len() {
            for j in (i + 1)..all_labelings.len() {
                let (indices_i, labels_i) = &all_labelings[i];
                let (indices_j, labels_j) = &all_labelings[j];

                // Find common indices between the two subsamples
                let common_indices = self.find_common_indices(indices_i, indices_j);

                if common_indices.len() > 1 {
                    // Map labels to common indices
                    let labels_i_common =
                        self.map_labels_to_common_indices(indices_i, labels_i, &common_indices);
                    let labels_j_common =
                        self.map_labels_to_common_indices(indices_j, labels_j, &common_indices);

                    // Compute ARI between overlapping samples (placeholder implementation)
                    let ari =
                        self.compute_adjusted_rand_index(&labels_i_common, &labels_j_common)?;
                    pairwise_stabilities.push(ari);
                }
            }
        }

        let mean_stability = if pairwise_stabilities.is_empty() {
            0.0
        } else {
            pairwise_stabilities.iter().sum::<f64>() / pairwise_stabilities.len() as f64
        };

        let std_stability = if pairwise_stabilities.len() > 1 {
            let variance = pairwise_stabilities
                .iter()
                .map(|x| (x - mean_stability).powi(2))
                .sum::<f64>()
                / (pairwise_stabilities.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        Ok(SubsampleStabilityResult {
            mean_stability,
            std_stability,
            min_stability: pairwise_stabilities
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            max_stability: pairwise_stabilities
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            n_successful_trials: all_labelings.len(),
            pairwise_stabilities,
            subsample_ratio,
        })
    }

    /// Consensus clustering stability analysis
    ///
    /// Evaluates clustering stability by running the same algorithm multiple times
    /// with different random seeds and measuring consistency of results.
    ///
    /// # Arguments
    /// * `clusterer` - Clustering function that takes data and seed, returns labels
    /// * `X` - Input data matrix
    /// * `n_runs` - Number of clustering runs with different seeds
    /// * `seeds` - Optional vector of seeds to use (if None, generates random seeds)
    ///
    /// # Returns
    /// ConsensusStabilityResult containing mean stability and detailed metrics
    ///
    /// # Mathematical Details
    /// - Run clustering algorithm n_runs times with different seeds
    /// - Compute pairwise ARI between all runs
    /// - Generate consensus matrix showing co-clustering probabilities
    /// - Aggregate metrics: mean ARI, cluster consistency, consensus strength
    pub fn consensus_stability<F>(
        &self,
        clusterer: F,
        X: &Array2<f64>,
        n_runs: usize,
        seeds: Option<Vec<u64>>,
    ) -> Result<ConsensusStabilityResult>
    where
        F: Fn(&Array2<f64>, u64) -> Result<Vec<i32>>,
    {
        if n_runs < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 runs for consensus stability".to_string(),
            ));
        }

        let n_samples = X.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        // Generate seeds if not provided
        let used_seeds = if let Some(s) = seeds {
            if s.len() != n_runs {
                return Err(SklearsError::InvalidInput(
                    "Seeds length must match n_runs".to_string(),
                ));
            }
            s
        } else {
            let mut rng = thread_rng();
            (0..n_runs).map(|_| rng.gen::<u64>()).collect()
        };

        // Run clustering with different seeds
        let mut all_labelings = Vec::new();
        for &seed in &used_seeds {
            match clusterer(X, seed) {
                Ok(labels) => {
                    if labels.len() != n_samples {
                        return Err(SklearsError::InvalidInput(
                            "Clusterer returned wrong number of labels".to_string(),
                        ));
                    }
                    all_labelings.push(labels);
                }
                Err(_) => {
                    // Skip failed runs but don't error out completely
                    continue;
                }
            }
        }

        if all_labelings.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Too few successful clustering runs".to_string(),
            ));
        }

        // Compute pairwise stability scores
        let mut pairwise_scores = Vec::new();
        let mut cluster_consistency_scores = Vec::new();

        for i in 0..all_labelings.len() {
            for j in (i + 1)..all_labelings.len() {
                let ari = self
                    .external_validator
                    .adjusted_rand_index(&all_labelings[i], &all_labelings[j])?;
                pairwise_scores.push(ari);

                // Compute cluster-level consistency
                let consistency =
                    self.compute_cluster_consistency(&all_labelings[i], &all_labelings[j]);
                cluster_consistency_scores.push(consistency);
            }
        }

        let mean_stability = pairwise_scores.iter().sum::<f64>() / pairwise_scores.len() as f64;
        let std_stability = if pairwise_scores.len() > 1 {
            let variance = pairwise_scores
                .iter()
                .map(|x| (x - mean_stability).powi(2))
                .sum::<f64>()
                / (pairwise_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let mean_cluster_consistency = cluster_consistency_scores.iter().sum::<f64>()
            / cluster_consistency_scores.len() as f64;

        // Compute consensus matrix
        let consensus_matrix = self.compute_consensus_matrix(&all_labelings);

        Ok(ConsensusStabilityResult {
            mean_stability,
            std_stability,
            min_stability: pairwise_scores
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            max_stability: pairwise_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            mean_cluster_consistency,
            pairwise_scores,
            cluster_consistency_scores,
            consensus_matrix,
            n_successful_runs: all_labelings.len(),
            seeds_used: used_seeds,
        })
    }

    /// Perturbation stability analysis
    ///
    /// Evaluates clustering stability under data perturbation by adding controlled noise.
    /// Tests how robust the clustering is to small changes in the data.
    ///
    /// # Arguments
    /// * `clusterer` - Clustering function that takes data, returns labels
    /// * `X` - Input data matrix
    /// * `noise_levels` - Vector of noise standard deviations to test
    /// * `n_trials_per_level` - Number of trials per noise level
    ///
    /// # Returns
    /// PerturbationStabilityResult with stability across different noise levels
    ///
    /// # Mathematical Details
    /// For each noise level σ:
    /// - Add Gaussian noise N(0, σ²) to each data point
    /// - Run clustering on perturbed data
    /// - Compare with baseline clustering using ARI
    /// - Compute degradation curve and robustness threshold
    pub fn perturbation_stability<F>(
        &self,
        clusterer: F,
        X: &Array2<f64>,
        noise_levels: &[f64],
        n_trials_per_level: usize,
    ) -> Result<PerturbationStabilityResult>
    where
        F: Fn(&Array2<f64>) -> Result<Vec<i32>>,
    {
        if noise_levels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Need at least one noise level".to_string(),
            ));
        }

        if n_trials_per_level < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 trials per noise level".to_string(),
            ));
        }

        // Get baseline clustering
        let baseline_labels = clusterer(X)?;

        let mut rng = thread_rng();
        let mut noise_level_results = HashMap::new();

        for &noise_std in noise_levels {
            let mut trial_stabilities = Vec::new();

            for _ in 0..n_trials_per_level {
                // Create perturbed data
                let mut perturbed_data = X.clone();
                let normal = RandNormal::new(0.0, noise_std).unwrap();
                for i in 0..perturbed_data.nrows() {
                    for j in 0..perturbed_data.ncols() {
                        let noise: f64 = normal.sample(&mut rng);
                        perturbed_data[[i, j]] += noise;
                    }
                }

                // Cluster perturbed data
                match clusterer(&perturbed_data) {
                    Ok(perturbed_labels) => {
                        let stability = self
                            .external_validator
                            .adjusted_rand_index(&baseline_labels, &perturbed_labels)?;
                        trial_stabilities.push(stability);
                    }
                    Err(_) => {
                        // Skip failed clustering
                        continue;
                    }
                }
            }

            if !trial_stabilities.is_empty() {
                let mean_stability =
                    trial_stabilities.iter().sum::<f64>() / trial_stabilities.len() as f64;
                let std_stability = if trial_stabilities.len() > 1 {
                    let variance = trial_stabilities
                        .iter()
                        .map(|x| (x - mean_stability).powi(2))
                        .sum::<f64>()
                        / (trial_stabilities.len() - 1) as f64;
                    variance.sqrt()
                } else {
                    0.0
                };

                noise_level_results.insert(
                    noise_std.into(),
                    NoiseStabilityResult {
                        noise_level: noise_std,
                        mean_stability,
                        std_stability,
                        min_stability: trial_stabilities
                            .iter()
                            .cloned()
                            .fold(f64::INFINITY, f64::min),
                        max_stability: trial_stabilities
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max),
                        n_successful_trials: trial_stabilities.len(),
                        trial_stabilities,
                    },
                );
            }
        }

        // Compute overall robustness metrics
        let stability_degradation = self.compute_stability_degradation(&noise_level_results);
        let robustness_threshold = self.estimate_robustness_threshold(&noise_level_results);

        Ok(PerturbationStabilityResult {
            baseline_clustering: baseline_labels,
            noise_level_results,
            stability_degradation,
            robustness_threshold,
            tested_noise_levels: noise_levels.to_vec(),
        })
    }

    /// Parameter sensitivity stability analysis
    ///
    /// Evaluates how sensitive clustering results are to parameter changes.
    /// Tests stability across a range of parameter values.
    ///
    /// # Arguments
    /// * `clusterer_factory` - Function that creates a clusterer given parameter value
    /// * `X` - Input data matrix
    /// * `parameter_values` - Vector of parameter values to test
    /// * `parameter_name` - Name of the parameter being varied
    ///
    /// # Returns
    /// ParameterSensitivityResult with stability across parameter values
    ///
    /// # Mathematical Details
    /// For parameter values p₁, p₂, ..., pₙ:
    /// - Run clustering with each parameter value
    /// - Compute ARI between adjacent parameter values: ARI(labels_pᵢ, labels_pᵢ₊₁)
    /// - Identify stability plateaus where ARI remains high
    /// - Compute overall sensitivity as variance of ARI scores
    pub fn parameter_sensitivity<F>(
        &self,
        clusterer_factory: F,
        X: &Array2<f64>,
        parameter_values: &[f64],
        parameter_name: String,
    ) -> Result<ParameterSensitivityResult>
    where
        F: Fn(f64) -> Box<dyn Fn(&Array2<f64>) -> Result<Vec<i32>>>,
    {
        if parameter_values.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 parameter values".to_string(),
            ));
        }

        let mut parameter_results = HashMap::new();
        let mut all_labelings = Vec::new();
        let mut successful_parameters = Vec::new();

        // Run clustering for each parameter value
        for &param_value in parameter_values {
            let clusterer = clusterer_factory(param_value);
            match clusterer(X) {
                Ok(labels) => {
                    parameter_results.insert(param_value.into(), labels.clone());
                    all_labelings.push(labels);
                    successful_parameters.push(param_value);
                }
                Err(_) => {
                    // Skip failed parameter values
                    continue;
                }
            }
        }

        if successful_parameters.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Too few successful parameter values".to_string(),
            ));
        }

        // Compute pairwise stability between adjacent parameter values
        let mut adjacent_stabilities = Vec::new();
        let mut parameter_pairs = Vec::new();

        for i in 0..(successful_parameters.len() - 1) {
            let param1 = successful_parameters[i];
            let param2 = successful_parameters[i + 1];
            let labels1 = &parameter_results[&param1.into()];
            let labels2 = &parameter_results[&param2.into()];

            let stability = self
                .external_validator
                .adjusted_rand_index(labels1, labels2)?;
            adjacent_stabilities.push(stability);
            parameter_pairs.push((param1, param2));
        }

        // Compute all pairwise stabilities
        let mut all_pairwise_stabilities = Vec::new();
        for i in 0..all_labelings.len() {
            for j in (i + 1)..all_labelings.len() {
                let stability = self
                    .external_validator
                    .adjusted_rand_index(&all_labelings[i], &all_labelings[j])?;
                all_pairwise_stabilities.push(stability);
            }
        }

        let mean_adjacent_stability =
            adjacent_stabilities.iter().sum::<f64>() / adjacent_stabilities.len() as f64;
        let mean_overall_stability =
            all_pairwise_stabilities.iter().sum::<f64>() / all_pairwise_stabilities.len() as f64;

        let std_adjacent_stability = if adjacent_stabilities.len() > 1 {
            let variance = adjacent_stabilities
                .iter()
                .map(|x| (x - mean_adjacent_stability).powi(2))
                .sum::<f64>()
                / (adjacent_stabilities.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Find stability plateau (range where stability is high)
        let stability_plateau =
            self.find_stability_plateau(&adjacent_stabilities, &successful_parameters);

        Ok(ParameterSensitivityResult {
            parameter_name,
            tested_parameters: parameter_values.to_vec(),
            successful_parameters,
            parameter_results,
            adjacent_stabilities,
            parameter_pairs,
            all_pairwise_stabilities,
            mean_adjacent_stability,
            mean_overall_stability,
            std_adjacent_stability,
            stability_plateau,
        })
    }

    /// Cross-validation stability analysis
    ///
    /// Evaluates clustering stability using k-fold cross-validation approach.
    /// Splits data into k folds and measures clustering consistency across folds.
    ///
    /// # Arguments
    /// * `clusterer` - Clustering function that takes data, returns labels
    /// * `X` - Input data matrix
    /// * `k_folds` - Number of folds for cross-validation
    /// * `n_repeats` - Number of times to repeat the cross-validation
    ///
    /// # Returns
    /// CrossValidationStabilityResult with stability metrics across folds
    ///
    /// # Mathematical Details
    /// For k-fold cross-validation:
    /// - Split data into k roughly equal folds
    /// - Run clustering on each fold independently
    /// - Compute pairwise ARI between all fold pairs
    /// - Repeat n_repeats times with different random splits
    /// - Aggregate stability metrics across all comparisons
    pub fn cross_validation_stability<F>(
        &self,
        clusterer: F,
        X: &Array2<f64>,
        k_folds: usize,
        n_repeats: usize,
    ) -> Result<CrossValidationStabilityResult>
    where
        F: Fn(&Array2<f64>) -> Result<Vec<i32>>,
    {
        if k_folds < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 folds for cross-validation".to_string(),
            ));
        }

        if n_repeats < 1 {
            return Err(SklearsError::InvalidInput(
                "Need at least 1 repeat".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples < k_folds {
            return Err(SklearsError::InvalidInput(
                "Not enough samples for the specified number of folds".to_string(),
            ));
        }

        let mut all_fold_stabilities = Vec::new();
        let mut all_repeat_stabilities = Vec::new();

        let mut rng = thread_rng();

        for repeat in 0..n_repeats {
            // Create random fold assignments
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let fold_size = n_samples / k_folds;
            let mut fold_labelings = Vec::new();
            let mut fold_indices_list = Vec::new();

            // Process each fold
            for fold in 0..k_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == k_folds - 1 {
                    n_samples // Last fold gets remaining samples
                } else {
                    (fold + 1) * fold_size
                };

                let fold_indices: Vec<usize> = indices[start_idx..end_idx].to_vec();
                fold_indices_list.push(fold_indices.clone());

                // Extract fold data
                let mut fold_data = Array2::zeros((fold_indices.len(), n_features));
                for (i, &original_idx) in fold_indices.iter().enumerate() {
                    for j in 0..n_features {
                        fold_data[[i, j]] = X[[original_idx, j]];
                    }
                }

                // Cluster fold data
                match clusterer(&fold_data) {
                    Ok(labels) => {
                        fold_labelings.push(labels);
                    }
                    Err(_) => {
                        // Skip failed fold clustering
                        continue;
                    }
                }
            }

            // Compute stability for this repeat
            if fold_labelings.len() >= 2 {
                let mut repeat_stabilities = Vec::new();

                for i in 0..fold_labelings.len() {
                    for j in (i + 1)..fold_labelings.len() {
                        // For clustering stability, we use Adjusted Rand Index on the labelings directly
                        let ari = self
                            .external_validator
                            .adjusted_rand_index(&fold_labelings[i], &fold_labelings[j])?;
                        repeat_stabilities.push(ari.clamp(0.0, 1.0));
                    }
                }

                if !repeat_stabilities.is_empty() {
                    let mean_repeat_stability =
                        repeat_stabilities.iter().sum::<f64>() / repeat_stabilities.len() as f64;
                    all_repeat_stabilities.push(mean_repeat_stability);
                    all_fold_stabilities.extend(repeat_stabilities);
                }
            }
        }

        if all_fold_stabilities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No successful fold comparisons".to_string(),
            ));
        }

        let mean_stability =
            all_fold_stabilities.iter().sum::<f64>() / all_fold_stabilities.len() as f64;
        let std_stability = if all_fold_stabilities.len() > 1 {
            let variance = all_fold_stabilities
                .iter()
                .map(|x| (x - mean_stability).powi(2))
                .sum::<f64>()
                / (all_fold_stabilities.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let mean_repeat_stability = if all_repeat_stabilities.is_empty() {
            0.0
        } else {
            all_repeat_stabilities.iter().sum::<f64>() / all_repeat_stabilities.len() as f64
        };

        let n_successful_comparisons = all_fold_stabilities.len();
        Ok(CrossValidationStabilityResult {
            mean_stability,
            std_stability,
            min_stability: all_fold_stabilities
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            max_stability: all_fold_stabilities
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            mean_repeat_stability,
            fold_stabilities: all_fold_stabilities,
            repeat_stabilities: all_repeat_stabilities,
            k_folds,
            n_repeats,
            n_successful_comparisons,
        })
    }

    // Helper methods

    /// Find common indices between two index vectors
    fn find_common_indices(&self, indices_a: &[usize], indices_b: &[usize]) -> Vec<usize> {
        let mut common = Vec::new();
        let set_b: HashSet<usize> = indices_b.iter().cloned().collect();

        for &idx in indices_a {
            if set_b.contains(&idx) {
                common.push(idx);
            }
        }

        common.sort();
        common
    }

    /// Map labels from subsample indices to common indices
    fn map_labels_to_common_indices(
        &self,
        subsample_indices: &[usize],
        subsample_labels: &[i32],
        common_indices: &[usize],
    ) -> Vec<i32> {
        let mut common_labels = Vec::new();
        let index_to_label: HashMap<usize, i32> = subsample_indices
            .iter()
            .zip(subsample_labels.iter())
            .map(|(&idx, &label)| (idx, label))
            .collect();

        for &common_idx in common_indices {
            if let Some(&label) = index_to_label.get(&common_idx) {
                common_labels.push(label);
            }
        }

        common_labels
    }

    /// Compute cluster consistency between two labelings
    fn compute_cluster_consistency(&self, labels1: &[i32], labels2: &[i32]) -> f64 {
        if labels1.len() != labels2.len() {
            return 0.0;
        }

        let n = labels1.len();
        if n <= 1 {
            return 1.0;
        }

        let mut agreement = 0usize;
        let mut total_pairs = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                total_pairs += 1;
                let same_in_first = labels1[i] == labels1[j];
                let same_in_second = labels2[i] == labels2[j];

                if same_in_first == same_in_second {
                    agreement += 1;
                }
            }
        }

        if total_pairs == 0 {
            1.0
        } else {
            agreement as f64 / total_pairs as f64
        }
    }

    /// Compute consensus matrix from multiple labelings
    fn compute_consensus_matrix(&self, all_labelings: &[Vec<i32>]) -> Array2<f64> {
        if all_labelings.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_samples = all_labelings[0].len();
        let mut consensus = Array2::<f64>::zeros((n_samples, n_samples));
        let mut pair_counts = Array2::<f64>::zeros((n_samples, n_samples));

        for labels in all_labelings {
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    pair_counts[[i, j]] += 1.0;
                    pair_counts[[j, i]] += 1.0;

                    if labels[i] == labels[j] && labels[i] != -1 {
                        consensus[[i, j]] += 1.0;
                        consensus[[j, i]] += 1.0;
                    }
                }
            }
        }

        // Normalize by number of labelings
        for i in 0..n_samples {
            for j in 0..n_samples {
                if pair_counts[[i, j]] > 0.0 {
                    consensus[[i, j]] /= pair_counts[[i, j]];
                }
            }
        }

        consensus
    }

    /// Compute stability degradation curve
    fn compute_stability_degradation(
        &self,
        noise_results: &HashMap<HashableFloat, NoiseStabilityResult>,
    ) -> Vec<(f64, f64)> {
        let mut degradation_curve = Vec::new();
        let mut sorted_noise_levels: Vec<f64> = noise_results.keys().map(|&k| k.into()).collect();
        sorted_noise_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for noise_level in sorted_noise_levels {
            if let Some(result) = noise_results.get(&noise_level.into()) {
                degradation_curve.push((noise_level, result.mean_stability));
            }
        }

        degradation_curve
    }

    /// Estimate robustness threshold (noise level where stability drops significantly)
    fn estimate_robustness_threshold(
        &self,
        noise_results: &HashMap<HashableFloat, NoiseStabilityResult>,
    ) -> Option<f64> {
        let degradation_curve = self.compute_stability_degradation(noise_results);

        if degradation_curve.len() < 3 {
            return None;
        }

        // Find the noise level where stability drops below 0.7 (or largest drop)
        let threshold_stability = 0.7;
        for (noise_level, stability) in degradation_curve {
            if stability < threshold_stability {
                return Some(noise_level);
            }
        }

        None
    }

    /// Find stability plateau in parameter sensitivity analysis
    fn find_stability_plateau(
        &self,
        stabilities: &[f64],
        parameters: &[f64],
    ) -> Option<(f64, f64)> {
        if stabilities.len() < 3 {
            return None;
        }

        let high_stability_threshold = 0.8;
        let mut plateau_start = None;
        let mut plateau_end = None;

        for (i, &stability) in stabilities.iter().enumerate() {
            if stability >= high_stability_threshold {
                if plateau_start.is_none() {
                    plateau_start = Some(parameters[i]);
                }
                plateau_end = Some(parameters[i + 1]);
            } else if plateau_start.is_some() {
                break;
            }
        }

        if let (Some(start), Some(end)) = (plateau_start, plateau_end) {
            Some((start, end))
        } else {
            None
        }
    }

    /// Compute adjusted rand index between two label arrays (placeholder implementation)
    fn compute_adjusted_rand_index(&self, labels1: &[i32], labels2: &[i32]) -> Result<f64> {
        // Simple placeholder - in practice this would be a proper ARI implementation
        if labels1.len() != labels2.len() {
            return Err(SklearsError::InvalidInput(
                "Label arrays must have same length".to_string(),
            ));
        }

        // Return a dummy similarity score between 0 and 1
        let matches = labels1
            .iter()
            .zip(labels2.iter())
            .filter(|(a, b)| a == b)
            .count();
        Ok(matches as f64 / labels1.len() as f64)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::ndarray::Axis;

    fn generate_test_data() -> (Array2<f64>, Vec<i32>) {
        let data = array![
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0]
        ];
        let labels = vec![0, 0, 1, 1, 0, 1];
        (data, labels)
    }

    fn simple_clustering_fn(data: &Array2<f64>) -> Result<Vec<i32>> {
        // Simple mock clustering that assigns clusters based on first coordinate
        let labels: Vec<i32> = data
            .axis_iter(Axis(0))
            .map(|row| if row[0] < 4.0 { 0 } else { 1 })
            .collect();
        Ok(labels)
    }

    fn seeded_clustering_fn(data: &Array2<f64>, _seed: u64) -> Result<Vec<i32>> {
        simple_clustering_fn(data)
    }

    #[test]
    fn test_subsample_stability() {
        let (data, _) = generate_test_data();
        let analyzer = StabilityAnalyzer::euclidean();

        let result = analyzer
            .subsample_stability(&data, simple_clustering_fn, 0.7, 10)
            .unwrap();

        assert!(result.mean_stability >= 0.0 && result.mean_stability <= 1.0);
        assert!(result.std_stability >= 0.0);
        assert!(result.n_successful_trials >= 2);
        assert_eq!(result.subsample_ratio, 0.7);
        assert!(!result.pairwise_stabilities.is_empty());
    }

    #[test]
    fn test_consensus_stability() {
        let (data, _) = generate_test_data();
        let analyzer = StabilityAnalyzer::euclidean();

        let result = analyzer
            .consensus_stability(seeded_clustering_fn, &data, 5, None)
            .unwrap();

        assert!(result.mean_stability >= 0.0 && result.mean_stability <= 1.0);
        assert!(result.std_stability >= 0.0);
        assert!(result.n_successful_runs >= 2);
        assert!(!result.pairwise_scores.is_empty());
        assert_eq!(result.consensus_matrix.nrows(), data.nrows());
        assert_eq!(result.consensus_matrix.ncols(), data.nrows());
    }

    #[test]
    fn test_perturbation_stability() {
        let (data, _) = generate_test_data();
        let analyzer = StabilityAnalyzer::euclidean();

        let noise_levels = vec![0.1, 0.2, 0.5];
        let result = analyzer
            .perturbation_stability(simple_clustering_fn, &data, &noise_levels, 5)
            .unwrap();

        assert_eq!(result.baseline_clustering.len(), data.nrows());
        assert_eq!(result.tested_noise_levels, noise_levels);
        assert_eq!(result.noise_level_results.len(), noise_levels.len());
        assert!(!result.stability_degradation.is_empty());

        // Check that noise level results are reasonable
        for (&noise_level, result) in &result.noise_level_results {
            assert!((result.noise_level - noise_level.0).abs() < 1e-10);
            assert!(result.mean_stability >= 0.0 && result.mean_stability <= 1.0);
            assert!(result.n_successful_trials > 0);
        }
    }

    #[test]
    fn test_parameter_sensitivity() {
        let (data, _) = generate_test_data();
        let analyzer = StabilityAnalyzer::euclidean();

        let parameters = vec![1.0, 2.0, 3.0, 4.0];
        let clusterer_factory = |_param: f64| -> Box<dyn Fn(&Array2<f64>) -> Result<Vec<i32>>> {
            Box::new(simple_clustering_fn)
        };

        let result = analyzer
            .parameter_sensitivity(
                clusterer_factory,
                &data,
                &parameters,
                "test_param".to_string(),
            )
            .unwrap();

        assert_eq!(result.parameter_name, "test_param");
        assert_eq!(result.tested_parameters, parameters);
        assert!(!result.successful_parameters.is_empty());
        assert!(!result.adjacent_stabilities.is_empty());
        assert!(result.mean_adjacent_stability >= 0.0 && result.mean_adjacent_stability <= 1.0);
        assert!(result.mean_overall_stability >= 0.0 && result.mean_overall_stability <= 1.0);
    }

    #[test]
    fn test_cross_validation_stability() {
        let (data, _) = generate_test_data();
        let analyzer = StabilityAnalyzer::euclidean();

        let result = analyzer
            .cross_validation_stability(simple_clustering_fn, &data, 3, 2)
            .unwrap();

        assert!(result.mean_stability >= 0.0 && result.mean_stability <= 1.0);
        assert!(result.std_stability >= 0.0);
        assert_eq!(result.k_folds, 3);
        assert_eq!(result.n_repeats, 2);
        assert!(result.n_successful_comparisons > 0);
        assert!(!result.fold_stabilities.is_empty());
    }

    #[test]
    fn test_find_common_indices() {
        let analyzer = StabilityAnalyzer::euclidean();
        let indices_a = vec![0, 2, 4, 6];
        let indices_b = vec![1, 2, 3, 4, 5];

        let common = analyzer.find_common_indices(&indices_a, &indices_b);
        assert_eq!(common, vec![2, 4]);
    }

    #[test]
    fn test_map_labels_to_common_indices() {
        let analyzer = StabilityAnalyzer::euclidean();
        let subsample_indices = vec![0, 2, 4, 6];
        let subsample_labels = vec![1, 2, 1, 3];
        let common_indices = vec![2, 4];

        let common_labels = analyzer.map_labels_to_common_indices(
            &subsample_indices,
            &subsample_labels,
            &common_indices,
        );
        assert_eq!(common_labels, vec![2, 1]);
    }

    #[test]
    fn test_compute_cluster_consistency() {
        let analyzer = StabilityAnalyzer::euclidean();
        let labels1 = vec![0, 0, 1, 1];
        let labels2 = vec![0, 0, 1, 1]; // Same clustering

        let consistency = analyzer.compute_cluster_consistency(&labels1, &labels2);
        assert_eq!(consistency, 1.0);

        let labels3 = vec![0, 1, 0, 1]; // Different clustering
        let consistency2 = analyzer.compute_cluster_consistency(&labels1, &labels3);
        assert!(consistency2 < 1.0);
    }

    #[test]
    fn test_compute_consensus_matrix() {
        let analyzer = StabilityAnalyzer::euclidean();
        let labelings = vec![
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![1, 1, 0, 0], // Flipped labels but same clusters
        ];

        let consensus = analyzer.compute_consensus_matrix(&labelings);
        assert_eq!(consensus.nrows(), 4);
        assert_eq!(consensus.ncols(), 4);

        // Points 0 and 1 should always be clustered together
        assert!((consensus[[0, 1]] - 1.0).abs() < 1e-10);
        // Points 2 and 3 should always be clustered together
        assert!((consensus[[2, 3]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_inputs() {
        let (data, _) = generate_test_data();
        let analyzer = StabilityAnalyzer::euclidean();

        // Test invalid subsample ratio
        assert!(analyzer
            .subsample_stability(&data, simple_clustering_fn, 1.5, 10)
            .is_err());

        // Test too few trials
        assert!(analyzer
            .subsample_stability(&data, simple_clustering_fn, 0.5, 1)
            .is_err());

        // Test too few consensus runs
        assert!(analyzer
            .consensus_stability(seeded_clustering_fn, &data, 1, None)
            .is_err());

        // Test empty noise levels
        assert!(analyzer
            .perturbation_stability(simple_clustering_fn, &data, &[], 5)
            .is_err());

        // Test too few CV folds
        assert!(analyzer
            .cross_validation_stability(simple_clustering_fn, &data, 1, 2)
            .is_err());
    }
}
