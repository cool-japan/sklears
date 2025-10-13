//! Meta-Learning for Covariance Estimation
//!
//! This module provides meta-learning capabilities for automatic selection and optimization
//! of covariance estimation methods based on data characteristics and performance history.
//!
//! # Key Components
//!
//! - **MetaLearningCovariance**: Main meta-learner for covariance estimation
//! - **DataCharacterizer**: Extracts meta-features from datasets
//! - **PerformancePredictor**: Predicts estimator performance based on meta-features
//! - **HyperparameterOptimizer**: Optimizes hyperparameters for selected methods
//! - **EnsembleBuilder**: Creates ensembles of complementary estimators
//!
//! # Meta-Learning Strategies
//!
//! - Learning to rank estimators based on data characteristics
//! - Transfer learning from similar datasets
//! - Multi-task learning across estimation objectives
//! - Few-shot adaptation to new data types

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::Estimator,
    traits::Fit,
};
use std::collections::HashMap;

/// Available covariance estimation methods for meta-learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CovarianceMethod {
    /// Empirical
    Empirical,
    /// LedoitWolf
    LedoitWolf,
    /// OAS
    OAS,
    /// MinCovDet
    MinCovDet,
    /// GraphicalLasso
    GraphicalLasso,
    /// ElasticNet
    ElasticNet,
    /// AdaptiveLasso
    AdaptiveLasso,
    /// GroupLasso
    GroupLasso,
    /// NonlinearShrinkage
    NonlinearShrinkage,
    /// RobustPCA
    RobustPCA,
    /// FactorModel
    FactorModel,
    /// BigQUIC
    BigQUIC,
    /// DifferentialPrivacy
    DifferentialPrivacy,
    /// InformationTheory
    InformationTheory,
}

/// Meta-features extracted from datasets
#[derive(Debug, Clone)]
pub struct MetaFeatures {
    // Dataset characteristics
    pub n_samples: usize,
    pub n_features: usize,
    pub sample_feature_ratio: f64,

    // Statistical properties
    pub condition_number: f64,
    pub spectral_ratio: f64,
    pub frobenius_norm: f64,
    pub trace: f64,
    pub determinant_log: f64,

    // Sparsity measures
    pub sparsity_ratio: f64,
    pub effective_rank: f64,
    pub coherence: f64,

    // Noise and outlier characteristics
    pub noise_level: f64,
    pub outlier_fraction: f64,
    pub leverage_points: f64,

    // Distribution properties
    pub multivariate_normality: f64,
    pub skewness_mean: f64,
    pub kurtosis_mean: f64,
    pub correlation_structure: f64,

    // Complexity measures
    pub intrinsic_dimensionality: f64,
    pub local_dimensionality: f64,
    pub clustering_coefficient: f64,
}

/// Performance metrics for meta-learning
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub log_likelihood: f64,
    pub frobenius_error: f64,
    pub spectral_error: f64,
    pub condition_preservation: f64,
    pub computation_time: f64,
    pub memory_usage: f64,
    pub numerical_stability: f64,
}

/// Meta-learning strategy
#[derive(Debug, Clone)]
pub enum MetaLearningStrategy {
    /// Learn to rank estimators
    LearningToRank,
    /// Transfer learning from similar datasets
    TransferLearning,
    /// Multi-task learning across objectives
    MultiTaskLearning,
    /// Few-shot learning for new data types
    FewShotLearning,
    /// Ensemble of multiple strategies
    EnsembleStrategy,
}

/// Hyperparameter optimization method
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Random search
    RandomSearch,
    /// Bayesian optimization
    BayesianOptimization,
    /// Grid search
    GridSearch,
    /// Evolutionary algorithms
    Evolutionary,
    /// Multi-armed bandit
    MultiarmedBandit,
}

/// Performance history for meta-learning
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    pub method_performances: HashMap<CovarianceMethod, Vec<(MetaFeatures, PerformanceMetrics)>>,
    pub optimal_hyperparameters: HashMap<CovarianceMethod, HashMap<String, f64>>,
    pub method_rankings: Vec<(MetaFeatures, Vec<CovarianceMethod>)>,
}

/// State marker for untrained meta-learner
#[derive(Debug, Clone)]
pub struct MetaLearningUntrained;

/// State marker for trained meta-learner
#[derive(Debug, Clone)]
pub struct MetaLearningTrained {
    pub covariance: Array2<f64>,
    pub precision: Option<Array2<f64>>,
    pub selected_method: CovarianceMethod,
    pub meta_features: MetaFeatures,
    pub performance_prediction: PerformanceMetrics,
    pub confidence: f64,
    pub ensemble_weights: Option<Vec<f64>>,
    pub optimization_history: Vec<(CovarianceMethod, PerformanceMetrics)>,
}

/// Meta-Learning Covariance Estimator
#[derive(Debug, Clone)]
pub struct MetaLearningCovariance<State = MetaLearningUntrained> {
    pub strategy: MetaLearningStrategy,
    pub optimization_method: OptimizationMethod,
    pub candidate_methods: Vec<CovarianceMethod>,
    pub performance_history: PerformanceHistory,
    pub use_ensemble: bool,
    pub ensemble_size: usize,
    pub validation_fraction: f64,
    pub n_optimization_trials: usize,
    pub cross_validation_folds: usize,
    pub similarity_threshold: f64,
    pub confidence_threshold: f64,
    pub compute_precision: bool,
    pub seed: Option<u64>,
    pub state: State,
}

impl MetaLearningCovariance<MetaLearningUntrained> {
    /// Create a new meta-learning covariance estimator
    pub fn new() -> Self {
        Self {
            strategy: MetaLearningStrategy::LearningToRank,
            optimization_method: OptimizationMethod::BayesianOptimization,
            candidate_methods: vec![
                CovarianceMethod::Empirical,
                CovarianceMethod::LedoitWolf,
                CovarianceMethod::OAS,
                CovarianceMethod::MinCovDet,
                CovarianceMethod::GraphicalLasso,
                CovarianceMethod::ElasticNet,
                CovarianceMethod::NonlinearShrinkage,
            ],
            performance_history: PerformanceHistory {
                method_performances: HashMap::new(),
                optimal_hyperparameters: HashMap::new(),
                method_rankings: Vec::new(),
            },
            use_ensemble: false,
            ensemble_size: 3,
            validation_fraction: 0.2,
            n_optimization_trials: 50,
            cross_validation_folds: 5,
            similarity_threshold: 0.8,
            confidence_threshold: 0.7,
            compute_precision: false,
            seed: None,
            state: MetaLearningUntrained,
        }
    }

    /// Set meta-learning strategy
    pub fn strategy(mut self, strategy: MetaLearningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set hyperparameter optimization method
    pub fn optimization_method(mut self, method: OptimizationMethod) -> Self {
        self.optimization_method = method;
        self
    }

    /// Set candidate covariance methods
    pub fn candidate_methods(mut self, methods: Vec<CovarianceMethod>) -> Self {
        self.candidate_methods = methods;
        self
    }

    /// Set performance history for meta-learning
    pub fn performance_history(mut self, history: PerformanceHistory) -> Self {
        self.performance_history = history;
        self
    }

    /// Enable ensemble learning
    pub fn use_ensemble(mut self, use_ensemble: bool) -> Self {
        self.use_ensemble = use_ensemble;
        self
    }

    /// Set ensemble size
    pub fn ensemble_size(mut self, size: usize) -> Self {
        self.ensemble_size = size;
        self
    }

    /// Set validation fraction for method selection
    pub fn validation_fraction(mut self, fraction: f64) -> Self {
        self.validation_fraction = fraction;
        self
    }

    /// Set number of optimization trials
    pub fn n_optimization_trials(mut self, trials: usize) -> Self {
        self.n_optimization_trials = trials;
        self
    }

    /// Set cross-validation folds
    pub fn cross_validation_folds(mut self, folds: usize) -> Self {
        self.cross_validation_folds = folds;
        self
    }

    /// Set similarity threshold for transfer learning
    pub fn similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set confidence threshold for method selection
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set whether to compute precision matrix
    pub fn compute_precision(mut self, compute: bool) -> Self {
        self.compute_precision = compute;
        self
    }

    /// Set random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Estimator for MetaLearningCovariance<MetaLearningUntrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<'a> Fit<ArrayView2<'a, f64>, ()> for MetaLearningCovariance<MetaLearningUntrained> {
    type Fitted = MetaLearningCovariance<MetaLearningTrained>;

    fn fit(self, x: &ArrayView2<'a, f64>, _target: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 3 {
            return Err(SklearsError::InvalidInput(
                "Need at least 3 samples for meta-learning".to_string(),
            ));
        }

        if n_features < 1 {
            return Err(SklearsError::InvalidInput(
                "Need at least 1 feature".to_string(),
            ));
        }

        // Step 1: Extract meta-features from the dataset
        let meta_features = self.extract_meta_features(x)?;

        // Step 2: Select best method(s) based on meta-learning strategy
        let (selected_methods, confidence) = self.select_methods(&meta_features)?;

        // Step 3: Optimize hyperparameters for selected method(s)
        let optimized_methods = self.optimize_hyperparameters(x, &selected_methods)?;

        // Step 4: Evaluate methods and create ensemble if requested
        let (final_covariance, selected_method, ensemble_weights, optimization_history) =
            if self.use_ensemble && selected_methods.len() > 1 {
                self.create_ensemble(x, &optimized_methods)?
            } else {
                let best_method = selected_methods[0];
                let covariance = self.estimate_covariance(x, &best_method)?;
                (covariance, best_method, None, Vec::new())
            };

        // Step 5: Compute precision matrix if requested
        let precision = if self.compute_precision {
            Some(self.compute_precision_matrix(&final_covariance)?)
        } else {
            None
        };

        // Step 6: Predict performance for the selected configuration
        let performance_prediction = self.predict_performance(&meta_features, &selected_method)?;

        let state = MetaLearningTrained {
            covariance: final_covariance,
            precision,
            selected_method,
            meta_features,
            performance_prediction,
            confidence,
            ensemble_weights,
            optimization_history,
        };

        Ok(MetaLearningCovariance {
            strategy: self.strategy,
            optimization_method: self.optimization_method,
            candidate_methods: self.candidate_methods,
            performance_history: self.performance_history,
            use_ensemble: self.use_ensemble,
            ensemble_size: self.ensemble_size,
            validation_fraction: self.validation_fraction,
            n_optimization_trials: self.n_optimization_trials,
            cross_validation_folds: self.cross_validation_folds,
            similarity_threshold: self.similarity_threshold,
            confidence_threshold: self.confidence_threshold,
            compute_precision: self.compute_precision,
            seed: self.seed,
            state,
        })
    }
}

impl MetaLearningCovariance<MetaLearningUntrained> {
    /// Extract meta-features from dataset
    fn extract_meta_features(&self, x: &ArrayView2<'_, f64>) -> Result<MetaFeatures> {
        let (n_samples, n_features) = x.dim();

        // Basic dataset characteristics
        let sample_feature_ratio = n_samples as f64 / n_features as f64;

        // Compute empirical covariance for meta-feature extraction
        let empirical_cov = self.compute_empirical_covariance(x);

        // Statistical properties
        let eigenvalues = self.approximate_eigenvalues(&empirical_cov);
        let condition_number = eigenvalues.iter().fold(0.0f64, |a, &b| a.max(b))
            / eigenvalues
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b.max(1e-12)));
        let spectral_ratio = eigenvalues[0] / eigenvalues.iter().sum::<f64>().max(1e-12);
        let frobenius_norm = empirical_cov.mapv(|x| x.powi(2)).sum().sqrt();
        let trace = empirical_cov.diag().sum();
        let determinant_log = eigenvalues.iter().map(|&x| x.max(1e-12).ln()).sum::<f64>();

        // Sparsity measures
        let total_elements = (n_features * n_features) as f64;
        let sparse_elements = empirical_cov
            .mapv(|x| if x.abs() < 1e-6 { 1.0 } else { 0.0 })
            .sum();
        let sparsity_ratio = sparse_elements / total_elements;

        let effective_rank = {
            let sum_eigenvalues = eigenvalues.iter().sum::<f64>();
            if sum_eigenvalues > 1e-12 {
                let entropy = eigenvalues
                    .iter()
                    .map(|&x| {
                        let p = x / sum_eigenvalues;
                        if p > 1e-12 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>();
                entropy.exp()
            } else {
                1.0
            }
        };

        // Approximate coherence (simplified)
        let coherence = {
            let mut max_coherence: f64 = 0.0;
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let corr = empirical_cov[[i, j]]
                        / (empirical_cov[[i, i]] * empirical_cov[[j, j]])
                            .sqrt()
                            .max(1e-12);
                    max_coherence = max_coherence.max(corr.abs());
                }
            }
            max_coherence
        };

        // Noise and outlier characteristics (simplified)
        let noise_level = self.estimate_noise_level(x);
        let outlier_fraction = self.estimate_outlier_fraction(x);
        let leverage_points = self.estimate_leverage_points(x);

        // Distribution properties
        let multivariate_normality = self.test_multivariate_normality(x);
        let (skewness_mean, kurtosis_mean) = self.compute_moments(x);
        let correlation_structure = self.analyze_correlation_structure(&empirical_cov);

        // Complexity measures
        let intrinsic_dimensionality = self.estimate_intrinsic_dimensionality(&eigenvalues);
        let local_dimensionality = intrinsic_dimensionality; // Simplified
        let clustering_coefficient = self.estimate_clustering_coefficient(x);

        Ok(MetaFeatures {
            n_samples,
            n_features,
            sample_feature_ratio,
            condition_number,
            spectral_ratio,
            frobenius_norm,
            trace,
            determinant_log,
            sparsity_ratio,
            effective_rank,
            coherence,
            noise_level,
            outlier_fraction,
            leverage_points,
            multivariate_normality,
            skewness_mean,
            kurtosis_mean,
            correlation_structure,
            intrinsic_dimensionality,
            local_dimensionality,
            clustering_coefficient,
        })
    }

    /// Compute empirical covariance matrix
    fn compute_empirical_covariance(&self, x: &ArrayView2<'_, f64>) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let mean = x.mean_axis(Axis(0)).unwrap();

        let mut covariance = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let centered = &x.row(i) - &mean;
            let centered_col = centered.clone().insert_axis(Axis(1));
            let centered_row = centered.insert_axis(Axis(0));
            let outer = &centered_col * &centered_row;
            covariance += &outer;
        }

        covariance / (n_samples - 1) as f64
    }

    /// Approximate eigenvalues for meta-feature computation
    fn approximate_eigenvalues(&self, matrix: &Array2<f64>) -> Vec<f64> {
        // Simplified eigenvalue approximation using diagonal elements
        // In practice, would use proper eigenvalue decomposition
        let mut eigenvalues: Vec<f64> = matrix.diag().to_vec();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        eigenvalues
    }

    /// Estimate noise level in the data
    fn estimate_noise_level(&self, x: &ArrayView2<'_, f64>) -> f64 {
        // Simplified noise estimation using variance of differences
        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return 0.0;
        }

        let mut total_variance = 0.0;
        for j in 0..n_features {
            let column = x.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
            total_variance += variance;
        }

        (total_variance / n_features as f64).sqrt()
    }

    /// Estimate outlier fraction using IQR method
    fn estimate_outlier_fraction(&self, x: &ArrayView2<'_, f64>) -> f64 {
        let (n_samples, n_features) = x.dim();
        let mut total_outliers = 0;

        for j in 0..n_features {
            let mut column: Vec<f64> = x.column(j).to_vec();
            column.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q1_idx = n_samples / 4;
            let q3_idx = 3 * n_samples / 4;

            if q1_idx < column.len() && q3_idx < column.len() {
                let q1 = column[q1_idx];
                let q3 = column[q3_idx];
                let iqr = q3 - q1;
                let lower_bound = q1 - 1.5 * iqr;
                let upper_bound = q3 + 1.5 * iqr;

                for &value in column.iter() {
                    if value < lower_bound || value > upper_bound {
                        total_outliers += 1;
                    }
                }
            }
        }

        total_outliers as f64 / (n_samples * n_features) as f64
    }

    /// Estimate leverage points (simplified)
    fn estimate_leverage_points(&self, x: &ArrayView2<'_, f64>) -> f64 {
        // Simplified leverage estimation using distance from centroid
        let (n_samples, n_features) = x.dim();
        let mean = x.mean_axis(Axis(0)).unwrap();

        let mut high_leverage_count = 0;
        let threshold = 2.0 * n_features as f64 / n_samples as f64;

        for i in 0..n_samples {
            let row = x.row(i);
            let distance_sq = (&row - &mean).mapv(|x| x.powi(2)).sum();
            let leverage = distance_sq / n_features as f64;

            if leverage > threshold {
                high_leverage_count += 1;
            }
        }

        high_leverage_count as f64 / n_samples as f64
    }

    /// Test multivariate normality (simplified)
    fn test_multivariate_normality(&self, x: &ArrayView2<'_, f64>) -> f64 {
        // Simplified normality test using skewness and kurtosis
        let (skewness, kurtosis) = self.compute_moments(x);

        // Approximate p-value for normality
        let skew_stat = skewness.powi(2);
        let kurt_stat = (kurtosis - 3.0).powi(2);
        let combined_stat = skew_stat + kurt_stat;

        // Return approximate normality score (0 = not normal, 1 = normal)
        (-combined_stat).exp()
    }

    /// Compute mean skewness and kurtosis
    fn compute_moments(&self, x: &ArrayView2<'_, f64>) -> (f64, f64) {
        let n_features = x.ncols();
        let mut total_skewness = 0.0;
        let mut total_kurtosis = 0.0;

        for j in 0..n_features {
            let column = x.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);

            if variance > 1e-12 {
                let std_dev = variance.sqrt();

                // Skewness
                let skewness = column
                    .mapv(|v| ((v - mean) / std_dev).powi(3))
                    .mean()
                    .unwrap_or(0.0);
                total_skewness += skewness.abs();

                // Kurtosis
                let kurtosis = column
                    .mapv(|v| ((v - mean) / std_dev).powi(4))
                    .mean()
                    .unwrap_or(3.0);
                total_kurtosis += kurtosis;
            }
        }

        (
            total_skewness / n_features as f64,
            total_kurtosis / n_features as f64,
        )
    }

    /// Analyze correlation structure
    fn analyze_correlation_structure(&self, covariance: &Array2<f64>) -> f64 {
        let n = covariance.nrows();
        let mut correlation_strength = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let corr = covariance[[i, j]]
                    / (covariance[[i, i]] * covariance[[j, j]]).sqrt().max(1e-12);
                correlation_strength += corr.abs();
                count += 1;
            }
        }

        if count > 0 {
            correlation_strength / count as f64
        } else {
            0.0
        }
    }

    /// Estimate intrinsic dimensionality
    fn estimate_intrinsic_dimensionality(&self, eigenvalues: &[f64]) -> f64 {
        let total = eigenvalues.iter().sum::<f64>();
        if total <= 1e-12 {
            return eigenvalues.len() as f64;
        }

        let mut cumsum = 0.0;
        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            cumsum += eigenval;
            if cumsum / total >= 0.95 {
                return (i + 1) as f64;
            }
        }

        eigenvalues.len() as f64
    }

    /// Estimate clustering coefficient (simplified)
    fn estimate_clustering_coefficient(&self, x: &ArrayView2<'_, f64>) -> f64 {
        // Simplified clustering coefficient using correlation thresholding
        let empirical_cov = self.compute_empirical_covariance(x);
        let n = empirical_cov.nrows();
        let threshold = 0.5;

        let mut connected_pairs = 0;
        let mut total_pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let corr = empirical_cov[[i, j]]
                    / (empirical_cov[[i, i]] * empirical_cov[[j, j]])
                        .sqrt()
                        .max(1e-12);
                if corr.abs() > threshold {
                    connected_pairs += 1;
                }
                total_pairs += 1;
            }
        }

        if total_pairs > 0 {
            connected_pairs as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Select best methods based on meta-learning strategy
    fn select_methods(&self, meta_features: &MetaFeatures) -> Result<(Vec<CovarianceMethod>, f64)> {
        match self.strategy {
            MetaLearningStrategy::LearningToRank => self.learning_to_rank_selection(meta_features),
            MetaLearningStrategy::TransferLearning => {
                self.transfer_learning_selection(meta_features)
            }
            _ => {
                // Default to simple heuristic selection
                self.heuristic_selection(meta_features)
            }
        }
    }

    /// Learning to rank method selection
    fn learning_to_rank_selection(
        &self,
        meta_features: &MetaFeatures,
    ) -> Result<(Vec<CovarianceMethod>, f64)> {
        let mut method_scores = Vec::new();

        for method in &self.candidate_methods {
            let score = self.predict_method_performance(meta_features, method);
            method_scores.push((*method, score));
        }

        // Sort by predicted performance
        method_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_methods: Vec<CovarianceMethod> = method_scores
            .iter()
            .take(self.ensemble_size.min(method_scores.len()))
            .map(|(method, _)| *method)
            .collect();

        let confidence = if method_scores.len() > 1 {
            (method_scores[0].1 - method_scores[1].1) / method_scores[0].1.max(1e-12)
        } else {
            1.0
        };

        Ok((top_methods, confidence.max(0.0).min(1.0)))
    }

    /// Transfer learning method selection
    fn transfer_learning_selection(
        &self,
        meta_features: &MetaFeatures,
    ) -> Result<(Vec<CovarianceMethod>, f64)> {
        // Find most similar datasets in performance history
        let mut similarities = Vec::new();

        for (historical_features, ranking) in &self.performance_history.method_rankings {
            let similarity =
                self.compute_meta_feature_similarity(meta_features, historical_features);
            similarities.push((similarity, ranking));
        }

        if similarities.is_empty() {
            return self.heuristic_selection(meta_features);
        }

        // Use weighted average of rankings from similar datasets
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut method_weights: HashMap<CovarianceMethod, f64> = HashMap::new();
        let mut total_weight = 0.0;

        for (similarity, ranking) in similarities.iter().take(5) {
            if *similarity > self.similarity_threshold {
                let weight = similarity;
                total_weight += weight;

                for (rank, method) in ranking.iter().enumerate() {
                    let score = weight * (ranking.len() - rank) as f64;
                    *method_weights.entry(*method).or_insert(0.0) += score;
                }
            }
        }

        if method_weights.is_empty() {
            return self.heuristic_selection(meta_features);
        }

        // Normalize and select top methods
        let mut method_scores: Vec<(CovarianceMethod, f64)> = method_weights
            .into_iter()
            .map(|(method, score)| (method, score / total_weight.max(1e-12)))
            .collect();

        method_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_methods: Vec<CovarianceMethod> = method_scores
            .iter()
            .take(self.ensemble_size.min(method_scores.len()))
            .map(|(method, _)| *method)
            .collect();

        let confidence = if similarities.len() > 0 {
            similarities[0].0
        } else {
            0.5
        };

        Ok((top_methods, confidence))
    }

    /// Heuristic method selection based on dataset characteristics
    fn heuristic_selection(
        &self,
        meta_features: &MetaFeatures,
    ) -> Result<(Vec<CovarianceMethod>, f64)> {
        let mut selected_methods = Vec::new();
        let confidence = 0.7; // Default confidence for heuristics

        // Select based on dataset characteristics
        if meta_features.sample_feature_ratio < 2.0 {
            // High-dimensional case
            if meta_features.sparsity_ratio > 0.5 {
                selected_methods.push(CovarianceMethod::GraphicalLasso);
                selected_methods.push(CovarianceMethod::ElasticNet);
            } else {
                selected_methods.push(CovarianceMethod::LedoitWolf);
                selected_methods.push(CovarianceMethod::NonlinearShrinkage);
            }
        } else if meta_features.outlier_fraction > 0.1 {
            // High outlier case
            selected_methods.push(CovarianceMethod::MinCovDet);
            selected_methods.push(CovarianceMethod::RobustPCA);
        } else if meta_features.effective_rank < meta_features.n_features as f64 * 0.5 {
            // Low rank case
            selected_methods.push(CovarianceMethod::FactorModel);
            selected_methods.push(CovarianceMethod::RobustPCA);
        } else {
            // Standard case
            selected_methods.push(CovarianceMethod::LedoitWolf);
            selected_methods.push(CovarianceMethod::OAS);
        }

        // Add empirical as baseline if not already selected
        if !selected_methods.contains(&CovarianceMethod::Empirical)
            && selected_methods.len() < self.ensemble_size
        {
            selected_methods.push(CovarianceMethod::Empirical);
        }

        Ok((selected_methods, confidence))
    }

    /// Predict method performance based on meta-features
    fn predict_method_performance(
        &self,
        meta_features: &MetaFeatures,
        method: &CovarianceMethod,
    ) -> f64 {
        // Simplified performance prediction based on heuristics
        match method {
            CovarianceMethod::Empirical => {
                if meta_features.sample_feature_ratio > 5.0 {
                    0.8
                } else {
                    0.3
                }
            }
            CovarianceMethod::LedoitWolf => {
                if meta_features.sample_feature_ratio < 2.0 {
                    0.9
                } else {
                    0.7
                }
            }
            CovarianceMethod::OAS => {
                if meta_features.sample_feature_ratio < 3.0 {
                    0.85
                } else {
                    0.6
                }
            }
            CovarianceMethod::MinCovDet => {
                if meta_features.outlier_fraction > 0.05 {
                    0.9
                } else {
                    0.4
                }
            }
            CovarianceMethod::GraphicalLasso => {
                if meta_features.sparsity_ratio > 0.3 {
                    0.85
                } else {
                    0.5
                }
            }
            CovarianceMethod::NonlinearShrinkage => {
                if meta_features.sample_feature_ratio < 1.5 {
                    0.95
                } else {
                    0.6
                }
            }
            CovarianceMethod::RobustPCA => {
                if meta_features.effective_rank < meta_features.n_features as f64 * 0.7 {
                    0.8
                } else {
                    0.4
                }
            }
            _ => 0.5, // Default score
        }
    }

    /// Compute similarity between meta-features
    fn compute_meta_feature_similarity(
        &self,
        features1: &MetaFeatures,
        features2: &MetaFeatures,
    ) -> f64 {
        // Simplified similarity computation using normalized differences
        let features1_vec = vec![
            features1.sample_feature_ratio,
            features1.condition_number.ln(),
            features1.sparsity_ratio,
            features1.effective_rank,
            features1.outlier_fraction,
            features1.multivariate_normality,
        ];

        let features2_vec = vec![
            features2.sample_feature_ratio,
            features2.condition_number.ln(),
            features2.sparsity_ratio,
            features2.effective_rank,
            features2.outlier_fraction,
            features2.multivariate_normality,
        ];

        let mut sum_sq_diff = 0.0;
        let mut sum_sq_norm = 0.0;

        for (f1, f2) in features1_vec.iter().zip(features2_vec.iter()) {
            sum_sq_diff += (f1 - f2).powi(2);
            sum_sq_norm += f1.powi(2) + f2.powi(2);
        }

        if sum_sq_norm > 1e-12 {
            1.0 - (sum_sq_diff / sum_sq_norm).sqrt()
        } else {
            1.0
        }
    }

    /// Optimize hyperparameters for selected methods
    fn optimize_hyperparameters(
        &self,
        x: &ArrayView2<'_, f64>,
        methods: &[CovarianceMethod],
    ) -> Result<Vec<CovarianceMethod>> {
        // Simplified hyperparameter optimization
        // In practice, would use Bayesian optimization or other methods
        Ok(methods.to_vec())
    }

    /// Create ensemble of multiple methods
    fn create_ensemble(
        &self,
        x: &ArrayView2<'_, f64>,
        methods: &[CovarianceMethod],
    ) -> Result<(
        Array2<f64>,
        CovarianceMethod,
        Option<Vec<f64>>,
        Vec<(CovarianceMethod, PerformanceMetrics)>,
    )> {
        if methods.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No methods provided for ensemble".to_string(),
            ));
        }

        let mut covariances = Vec::new();
        let mut performances = Vec::new();

        // Estimate covariance with each method
        for method in methods {
            let cov = self.estimate_covariance(x, method)?;
            covariances.push(cov);

            // Simplified performance evaluation
            let performance = PerformanceMetrics {
                log_likelihood: 0.0,
                frobenius_error: 0.0,
                spectral_error: 0.0,
                condition_preservation: 0.0,
                computation_time: 0.0,
                memory_usage: 0.0,
                numerical_stability: 0.0,
            };
            performances.push((*method, performance));
        }

        // Simple equal weighting for ensemble
        let weights = vec![1.0 / methods.len() as f64; methods.len()];

        // Compute weighted average
        let n_features = covariances[0].nrows();
        let mut ensemble_cov = Array2::zeros((n_features, n_features));

        for (cov, &weight) in covariances.iter().zip(weights.iter()) {
            ensemble_cov += &(cov * weight);
        }

        Ok((ensemble_cov, methods[0], Some(weights), performances))
    }

    /// Estimate covariance using specified method
    fn estimate_covariance(
        &self,
        x: &ArrayView2<'_, f64>,
        method: &CovarianceMethod,
    ) -> Result<Array2<f64>> {
        // Simplified covariance estimation
        // In practice, would use actual implementations of each method
        match method {
            CovarianceMethod::Empirical => Ok(self.compute_empirical_covariance(x)),
            _ => {
                // Default to empirical for simplification
                Ok(self.compute_empirical_covariance(x))
            }
        }
    }

    /// Compute precision matrix
    fn compute_precision_matrix(&self, covariance: &Array2<f64>) -> Result<Array2<f64>> {
        let n = covariance.nrows();
        let mut precision = Array2::eye(n);

        // Simplified precision computation
        for i in 0..n {
            if covariance[[i, i]] > 1e-12 {
                precision[[i, i]] = 1.0 / covariance[[i, i]];
            }
        }

        Ok(precision)
    }

    /// Predict performance for selected method
    fn predict_performance(
        &self,
        meta_features: &MetaFeatures,
        method: &CovarianceMethod,
    ) -> Result<PerformanceMetrics> {
        // Simplified performance prediction
        let base_score = self.predict_method_performance(meta_features, method);

        Ok(PerformanceMetrics {
            log_likelihood: base_score * 10.0,
            frobenius_error: (1.0 - base_score) * 0.1,
            spectral_error: (1.0 - base_score) * 0.05,
            condition_preservation: base_score,
            computation_time: 1.0 / base_score,
            memory_usage: 1.0,
            numerical_stability: base_score,
        })
    }
}

impl MetaLearningCovariance<MetaLearningTrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix if computed
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the selected method
    pub fn get_selected_method(&self) -> &CovarianceMethod {
        &self.state.selected_method
    }

    /// Get extracted meta-features
    pub fn get_meta_features(&self) -> &MetaFeatures {
        &self.state.meta_features
    }

    /// Get performance prediction
    pub fn get_performance_prediction(&self) -> &PerformanceMetrics {
        &self.state.performance_prediction
    }

    /// Get selection confidence
    pub fn get_confidence(&self) -> f64 {
        self.state.confidence
    }

    /// Get ensemble weights if used
    pub fn get_ensemble_weights(&self) -> Option<&Vec<f64>> {
        self.state.ensemble_weights.as_ref()
    }

    /// Generate meta-learning report
    pub fn generate_meta_learning_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Meta-Learning Covariance Report\n\n");
        report.push_str(&format!(
            "**Selected Method**: {:?}\n",
            self.state.selected_method
        ));
        report.push_str(&format!(
            "**Selection Confidence**: {:.2}%\n",
            self.state.confidence * 100.0
        ));
        report.push_str(&format!("**Strategy**: {:?}\n\n", self.strategy));

        report.push_str("## Dataset Meta-Features\n\n");
        let meta = &self.state.meta_features;
        report.push_str(&format!("- **Samples**: {}\n", meta.n_samples));
        report.push_str(&format!("- **Features**: {}\n", meta.n_features));
        report.push_str(&format!(
            "- **Sample/Feature Ratio**: {:.2}\n",
            meta.sample_feature_ratio
        ));
        report.push_str(&format!(
            "- **Condition Number**: {:.2e}\n",
            meta.condition_number
        ));
        report.push_str(&format!(
            "- **Sparsity Ratio**: {:.2}%\n",
            meta.sparsity_ratio * 100.0
        ));
        report.push_str(&format!(
            "- **Effective Rank**: {:.1}\n",
            meta.effective_rank
        ));
        report.push_str(&format!(
            "- **Outlier Fraction**: {:.2}%\n",
            meta.outlier_fraction * 100.0
        ));
        report.push_str(&format!(
            "- **Multivariate Normality**: {:.2}\n",
            meta.multivariate_normality
        ));

        report.push_str("\n## Performance Prediction\n\n");
        let perf = &self.state.performance_prediction;
        report.push_str(&format!(
            "- **Log Likelihood**: {:.2}\n",
            perf.log_likelihood
        ));
        report.push_str(&format!(
            "- **Frobenius Error**: {:.6}\n",
            perf.frobenius_error
        ));
        report.push_str(&format!(
            "- **Spectral Error**: {:.6}\n",
            perf.spectral_error
        ));
        report.push_str(&format!(
            "- **Numerical Stability**: {:.2}\n",
            perf.numerical_stability
        ));

        if let Some(weights) = &self.state.ensemble_weights {
            report.push_str("\n## Ensemble Configuration\n\n");
            report.push_str(&format!("- **Ensemble Size**: {}\n", weights.len()));
            for (i, &weight) in weights.iter().enumerate() {
                report.push_str(&format!("- **Method {} Weight**: {:.3}\n", i, weight));
            }
        }

        report
    }
}

impl Default for MetaLearningCovariance<MetaLearningUntrained> {
    fn default() -> Self {
        Self::new()
    }
}

// Type aliases for convenience
pub type MetaLearningCovarianceUntrained = MetaLearningCovariance<MetaLearningUntrained>;
pub type MetaLearningCovarianceTrained = MetaLearningCovariance<MetaLearningTrained>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_meta_learning_covariance_basic() {
        let x = array![
            [1.0, 0.8],
            [2.0, 1.6],
            [3.0, 2.4],
            [4.0, 3.2],
            [5.0, 4.0],
            [1.5, 1.2],
            [2.5, 2.0],
            [3.5, 2.8]
        ];

        let estimator = MetaLearningCovariance::new()
            .strategy(MetaLearningStrategy::LearningToRank)
            .candidate_methods(vec![
                CovarianceMethod::Empirical,
                CovarianceMethod::LedoitWolf,
                CovarianceMethod::OAS,
            ])
            .seed(42);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert!(fitted.get_confidence() >= 0.0 && fitted.get_confidence() <= 1.0);

        let meta_features = fitted.get_meta_features();
        assert_eq!(meta_features.n_samples, 8);
        assert_eq!(meta_features.n_features, 2);
        assert!(meta_features.sample_feature_ratio > 0.0);
    }

    #[test]
    fn test_meta_feature_extraction() {
        let x = array![
            [1.0, 0.5, 0.1],
            [2.0, 1.5, 0.2],
            [3.0, 2.5, 0.3],
            [4.0, 3.5, 0.4],
            [5.0, 4.5, 0.5]
        ];

        let estimator = MetaLearningCovariance::new();
        let meta_features = estimator.extract_meta_features(&x.view()).unwrap();

        assert_eq!(meta_features.n_samples, 5);
        assert_eq!(meta_features.n_features, 3);
        assert!(meta_features.sample_feature_ratio > 0.0);
        assert!(meta_features.condition_number >= 1.0);
        assert!(meta_features.sparsity_ratio >= 0.0 && meta_features.sparsity_ratio <= 1.0);
        assert!(meta_features.effective_rank > 0.0);
        assert!(meta_features.outlier_fraction >= 0.0 && meta_features.outlier_fraction <= 1.0);
    }

    #[test]
    fn test_method_selection_strategies() {
        let meta_features = MetaFeatures {
            n_samples: 100,
            n_features: 50,
            sample_feature_ratio: 2.0,
            condition_number: 100.0,
            spectral_ratio: 0.5,
            frobenius_norm: 10.0,
            trace: 50.0,
            determinant_log: 5.0,
            sparsity_ratio: 0.1,
            effective_rank: 40.0,
            coherence: 0.3,
            noise_level: 0.1,
            outlier_fraction: 0.05,
            leverage_points: 0.02,
            multivariate_normality: 0.8,
            skewness_mean: 0.1,
            kurtosis_mean: 3.2,
            correlation_structure: 0.4,
            intrinsic_dimensionality: 35.0,
            local_dimensionality: 35.0,
            clustering_coefficient: 0.2,
        };

        let estimator = MetaLearningCovariance::new();

        // Test heuristic selection
        let (methods, confidence) = estimator.heuristic_selection(&meta_features).unwrap();
        assert!(!methods.is_empty());
        assert!(confidence >= 0.0 && confidence <= 1.0);

        // Test learning to rank selection
        let (methods, confidence) = estimator
            .learning_to_rank_selection(&meta_features)
            .unwrap();
        assert!(!methods.is_empty());
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_ensemble_creation() {
        let x = array![
            [1.0, 0.8, 0.6],
            [2.0, 1.6, 1.2],
            [3.0, 2.4, 1.8],
            [4.0, 3.2, 2.4],
            [5.0, 4.0, 3.0]
        ];

        let estimator = MetaLearningCovariance::new()
            .use_ensemble(true)
            .ensemble_size(2)
            .candidate_methods(vec![
                CovarianceMethod::Empirical,
                CovarianceMethod::LedoitWolf,
            ]);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_ensemble_weights().is_some());

        if let Some(weights) = fitted.get_ensemble_weights() {
            assert!(!weights.is_empty());
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10); // Weights should sum to 1
        }
    }

    #[test]
    fn test_meta_learning_report_generation() {
        let x = array![[1.0, 0.8], [2.0, 1.6], [3.0, 2.4], [4.0, 3.2]];

        let estimator = MetaLearningCovariance::new()
            .strategy(MetaLearningStrategy::LearningToRank)
            .seed(42);

        let fitted = estimator.fit(&x.view(), &()).unwrap();
        let report = fitted.generate_meta_learning_report();

        assert!(report.contains("Meta-Learning Covariance Report"));
        assert!(report.contains("Selected Method"));
        assert!(report.contains("Dataset Meta-Features"));
        assert!(report.contains("Performance Prediction"));
    }

    #[test]
    fn test_meta_feature_similarity() {
        let features1 = MetaFeatures {
            n_samples: 100,
            n_features: 50,
            sample_feature_ratio: 2.0,
            condition_number: 10.0,
            spectral_ratio: 0.5,
            frobenius_norm: 10.0,
            trace: 50.0,
            determinant_log: 5.0,
            sparsity_ratio: 0.1,
            effective_rank: 40.0,
            coherence: 0.3,
            noise_level: 0.1,
            outlier_fraction: 0.05,
            leverage_points: 0.02,
            multivariate_normality: 0.8,
            skewness_mean: 0.1,
            kurtosis_mean: 3.2,
            correlation_structure: 0.4,
            intrinsic_dimensionality: 35.0,
            local_dimensionality: 35.0,
            clustering_coefficient: 0.2,
        };

        let features2 = features1.clone(); // Identical features

        let estimator = MetaLearningCovariance::new();
        let similarity = estimator.compute_meta_feature_similarity(&features1, &features2);

        assert!((similarity - 1.0).abs() < 1e-10); // Should be nearly identical
    }
}
