//! Component Selection Methods for Decomposition
//!
//! This module provides methods for selecting the optimal number of components
//! in decomposition algorithms including:
//! - Cross-validation for component number selection
//! - Bootstrap methods for component stability assessment
//! - Information criteria (AIC, BIC) for model selection
//! - Parallel analysis for factor number determination
//! - Stability-based selection methods

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Cross-validation methods for component selection
#[derive(Debug, Clone, Copy)]
pub enum CrossValidationMethod {
    /// K-fold cross-validation
    KFold,
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Stratified cross-validation
    Stratified,
    /// Time series cross-validation (forward chaining)
    TimeSeries,
}

/// Bootstrap methods for stability assessment
#[derive(Debug, Clone, Copy)]
pub enum BootstrapMethod {
    /// Standard bootstrap resampling
    Standard,
    /// Balanced bootstrap
    Balanced,
    /// Block bootstrap for time series
    Block,
    /// Parametric bootstrap
    Parametric,
}

/// Information criteria for model selection
#[derive(Debug, Clone, Copy)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Hannan-Quinn Information Criterion
    HQC,
    /// Consistent Akaike Information Criterion
    CAIC,
}

/// Component selection configuration
#[derive(Debug, Clone)]
pub struct ComponentSelectionConfig {
    /// Maximum number of components to test
    pub max_components: usize,
    /// Minimum number of components to test
    pub min_components: usize,
    /// Number of cross-validation folds
    pub n_folds: usize,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Tolerance for convergence criteria
    pub tolerance: Float,
}

impl Default for ComponentSelectionConfig {
    fn default() -> Self {
        Self {
            max_components: 20,
            min_components: 1,
            n_folds: 5,
            n_bootstrap: 100,
            random_state: None,
            tolerance: 1e-6,
        }
    }
}

/// Cross-validation based component selection
pub struct CrossValidationSelector {
    config: ComponentSelectionConfig,
    method: CrossValidationMethod,
}

impl CrossValidationSelector {
    /// Create a new cross-validation selector
    pub fn new(method: CrossValidationMethod) -> Self {
        Self {
            config: ComponentSelectionConfig::default(),
            method,
        }
    }

    /// Set configuration
    pub fn config(mut self, config: ComponentSelectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum components to test
    pub fn max_components(mut self, max_components: usize) -> Self {
        self.config.max_components = max_components;
        self
    }

    /// Set number of folds
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.config.n_folds = n_folds;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Select optimal number of components for PCA using cross-validation
    pub fn select_pca_components(&self, data: &Array2<Float>) -> Result<ComponentSelectionResult> {
        let (n_samples, _n_features) = data.dim();
        let max_components = self.config.max_components.min(n_samples);

        let mut cv_scores = Vec::new();
        let mut reconstruction_errors = Vec::new();

        for n_comp in self.config.min_components..=max_components {
            let folds = self.create_folds(n_samples)?;
            let mut fold_scores = Vec::new();

            for (train_idx, test_idx) in folds {
                // Extract training and test data
                let train_data = self.extract_rows(data, &train_idx);
                let test_data = self.extract_rows(data, &test_idx);

                // Fit PCA on training data
                let pca_result = self.fit_pca(&train_data, n_comp)?;

                // Compute reconstruction error on test data
                let reconstruction_error =
                    self.compute_pca_reconstruction_error(&test_data, &pca_result)?;
                fold_scores.push(reconstruction_error);
            }

            let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
            let std_score = self.compute_std(&fold_scores, mean_score);

            cv_scores.push(mean_score);
            reconstruction_errors.push(std_score);
        }

        // Find optimal number of components (minimum reconstruction error)
        let optimal_idx = cv_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let optimal_components = self.config.min_components + optimal_idx;

        Ok(ComponentSelectionResult {
            optimal_components,
            cv_scores: Array1::from_vec(cv_scores),
            component_range: (self.config.min_components..=max_components).collect(),
            method: SelectionMethod::CrossValidation,
            metadata: HashMap::from([("reconstruction_errors".to_string(), reconstruction_errors)]),
        })
    }

    /// Create cross-validation folds
    fn create_folds(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut rng = match self.config.random_state {
            Some(seed) => thread_rng(), // TODO: use seeded rng when available
            None => thread_rng(),
        };

        match self.method {
            CrossValidationMethod::KFold => self.k_fold_split(n_samples, &mut rng),
            CrossValidationMethod::LeaveOneOut => self.leave_one_out_split(n_samples),
            CrossValidationMethod::Stratified => {
                // For now, fall back to k-fold (stratified would need labels)
                self.k_fold_split(n_samples, &mut rng)
            }
            CrossValidationMethod::TimeSeries => self.time_series_split(n_samples),
        }
    }

    /// K-fold cross-validation split
    fn k_fold_split(
        &self,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);

        let fold_size = n_samples / self.config.n_folds;
        let mut folds = Vec::new();

        for fold in 0..self.config.n_folds {
            let start = fold * fold_size;
            let end = if fold == self.config.n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let test_idx: Vec<usize> = indices[start..end].to_vec();
            let train_idx: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            folds.push((train_idx, test_idx));
        }

        Ok(folds)
    }

    /// Leave-one-out cross-validation split
    fn leave_one_out_split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut folds = Vec::new();

        for i in 0..n_samples {
            let test_idx = vec![i];
            let train_idx: Vec<usize> = (0..n_samples).filter(|&x| x != i).collect();
            folds.push((train_idx, test_idx));
        }

        Ok(folds)
    }

    /// Time series cross-validation split (forward chaining)
    fn time_series_split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let min_train_size = n_samples / 3;
        let test_size = n_samples / 10;
        let mut folds = Vec::new();

        for start in min_train_size..n_samples - test_size {
            let train_idx: Vec<usize> = (0..start).collect();
            let test_idx: Vec<usize> = (start..start + test_size).collect();
            folds.push((train_idx, test_idx));
        }

        Ok(folds)
    }

    /// Extract rows from matrix
    fn extract_rows(&self, data: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let n_cols = data.ncols();
        let mut result = Array2::zeros((indices.len(), n_cols));

        for (i, &row_idx) in indices.iter().enumerate() {
            for j in 0..n_cols {
                result[[i, j]] = data[[row_idx, j]];
            }
        }

        result
    }

    /// Simplified PCA fitting (returns components)
    fn fit_pca(&self, data: &Array2<Float>, n_components: usize) -> Result<SimplePCAResult> {
        let (n_samples, n_features) = data.dim();

        if n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "n_components too large".to_string(),
            ));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = data - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as Float;

        // Simplified eigendecomposition (in practice would use proper SVD/eigendecomposition)
        let (eigenvalues, eigenvectors) = self.simple_eigendecomposition(&cov_matrix)?;

        // Take top n_components
        let components = eigenvectors
            .slice(scirs2_core::ndarray::s![.., 0..n_components])
            .to_owned();
        let explained_variance = eigenvalues
            .slice(scirs2_core::ndarray::s![0..n_components])
            .to_owned();

        Ok(SimplePCAResult {
            components,
            mean,
            explained_variance,
        })
    }

    /// Simplified eigendecomposition (placeholder implementation)
    fn simple_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        // This is a simplified implementation for demonstration
        // In practice, would use proper linear algebra libraries
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Power iteration for largest eigenvalue
        let mut v = Array1::ones(n);
        for _ in 0..100 {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();
            if norm > 1e-12 {
                v = v_new / norm;
            }

            let eigenvalue = v.dot(&matrix.dot(&v));
            eigenvalues[0] = eigenvalue;

            for i in 0..n {
                eigenvectors[[i, 0]] = v[i];
            }
        }

        // Fill remaining eigenvalues (simplified)
        for i in 1..n {
            eigenvalues[i] = eigenvalues[0] / (i + 1) as Float;
        }

        // Sort in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues =
            Array1::from_vec(indices.iter().map(|&i| eigenvalues[i]).collect());
        let sorted_eigenvectors =
            Array2::from_shape_fn((n, n), |(i, j)| eigenvectors[[i, indices[j]]]);

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }

    /// Compute PCA reconstruction error
    fn compute_pca_reconstruction_error(
        &self,
        data: &Array2<Float>,
        pca: &SimplePCAResult,
    ) -> Result<Float> {
        let (n_samples, _) = data.dim();

        // Center the data
        let centered_data = data - &pca.mean.clone().insert_axis(Axis(0));

        // Project to lower dimensional space
        let projected = centered_data.dot(&pca.components);

        // Reconstruct
        let reconstructed =
            projected.dot(&pca.components.t()) + &pca.mean.clone().insert_axis(Axis(0));

        // Compute reconstruction error
        let diff = data - &reconstructed;
        let error = diff.iter().map(|&x| x * x).sum::<Float>() / (n_samples as Float);

        Ok(error)
    }

    /// Compute standard deviation
    fn compute_std(&self, values: &[Float], mean: Float) -> Float {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance = values
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<Float>()
            / (values.len() - 1) as Float;

        variance.sqrt()
    }
}

/// Simplified PCA result for cross-validation
#[derive(Debug, Clone)]
struct SimplePCAResult {
    components: Array2<Float>,
    mean: Array1<Float>,
    explained_variance: Array1<Float>,
}

/// Bootstrap-based component stability assessment
pub struct BootstrapSelector {
    config: ComponentSelectionConfig,
    method: BootstrapMethod,
}

impl BootstrapSelector {
    /// Create a new bootstrap selector
    pub fn new(method: BootstrapMethod) -> Self {
        Self {
            config: ComponentSelectionConfig::default(),
            method,
        }
    }

    /// Set configuration
    pub fn config(mut self, config: ComponentSelectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set number of bootstrap samples
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.config.n_bootstrap = n_bootstrap;
        self
    }

    /// Assess component stability using bootstrap resampling
    pub fn assess_stability(
        &self,
        data: &Array2<Float>,
        n_components: usize,
    ) -> Result<StabilityResult> {
        let (_n_samples, _) = data.dim();
        let mut rng = match self.config.random_state {
            Some(seed) => thread_rng(), // TODO: use seeded rng when available
            None => thread_rng(),
        };

        let mut component_similarities = Vec::new();
        let mut eigenvalue_stability = Vec::new();

        // Fit PCA on original data
        let original_pca = self.fit_pca(data, n_components)?;

        for _ in 0..self.config.n_bootstrap {
            // Create bootstrap sample
            let bootstrap_data = self.create_bootstrap_sample(data, &mut rng)?;

            // Fit PCA on bootstrap sample
            let bootstrap_pca = self.fit_pca(&bootstrap_data, n_components)?;

            // Compute component similarity
            let similarity = self.compute_component_similarity(
                &original_pca.components,
                &bootstrap_pca.components,
            )?;
            component_similarities.push(similarity);

            // Compare eigenvalues
            let eigenvalue_sim = self.compute_eigenvalue_similarity(
                &original_pca.explained_variance,
                &bootstrap_pca.explained_variance,
            );
            eigenvalue_stability.push(eigenvalue_sim);
        }

        let mean_similarity =
            component_similarities.iter().sum::<Float>() / component_similarities.len() as Float;
        let std_similarity = self.compute_std(&component_similarities, mean_similarity);

        let mean_eigenvalue_stability =
            eigenvalue_stability.iter().sum::<Float>() / eigenvalue_stability.len() as Float;

        Ok(StabilityResult {
            mean_component_similarity: mean_similarity,
            std_component_similarity: std_similarity,
            component_similarities: Array1::from_vec(component_similarities),
            eigenvalue_stability: mean_eigenvalue_stability,
            is_stable: mean_similarity > 0.8 && std_similarity < 0.2,
        })
    }

    /// Create bootstrap sample
    fn create_bootstrap_sample(
        &self,
        data: &Array2<Float>,
        rng: &mut impl Rng,
    ) -> Result<Array2<Float>> {
        let (n_samples, n_features) = data.dim();
        let mut bootstrap_data = Array2::zeros((n_samples, n_features));

        match self.method {
            BootstrapMethod::Standard => {
                for i in 0..n_samples {
                    let sample_idx = rng.gen_range(0..n_samples);
                    for j in 0..n_features {
                        bootstrap_data[[i, j]] = data[[sample_idx, j]];
                    }
                }
            }
            BootstrapMethod::Balanced => {
                // Ensure each original sample appears at least once
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);

                for i in 0..n_samples {
                    let sample_idx = if i < indices.len() {
                        indices[i]
                    } else {
                        rng.gen_range(0..n_samples)
                    };

                    for j in 0..n_features {
                        bootstrap_data[[i, j]] = data[[sample_idx, j]];
                    }
                }
            }
            BootstrapMethod::Block => {
                // Block bootstrap for time series (simplified)
                let block_size = (n_samples as f64).sqrt() as usize;
                let n_blocks = n_samples / block_size;

                for block in 0..n_blocks {
                    let start_idx = rng.gen_range(0..n_samples - block_size + 1);

                    for i in 0..block_size {
                        let src_idx = start_idx + i;
                        let dst_idx = block * block_size + i;

                        if dst_idx < n_samples {
                            for j in 0..n_features {
                                bootstrap_data[[dst_idx, j]] = data[[src_idx, j]];
                            }
                        }
                    }
                }
            }
            BootstrapMethod::Parametric => {
                // Parametric bootstrap (generate from fitted distribution)
                // For simplicity, use normal distribution
                for j in 0..n_features {
                    let column = data.column(j);
                    let mean = column.mean().unwrap_or(0.0);
                    let std = self.compute_std(column.as_slice().unwrap(), mean);

                    for i in 0..n_samples {
                        bootstrap_data[[i, j]] = rng.gen::<Float>() * std + mean;
                    }
                }
            }
        }

        Ok(bootstrap_data)
    }

    /// Fit PCA (simplified version)
    fn fit_pca(&self, data: &Array2<Float>, n_components: usize) -> Result<SimplePCAResult> {
        let (n_samples, n_features) = data.dim();

        if n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "n_components too large".to_string(),
            ));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = data - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as Float;

        // Simplified eigendecomposition
        let (eigenvalues, eigenvectors) = self.simple_eigendecomposition(&cov_matrix)?;

        // Take top n_components
        let components = eigenvectors
            .slice(scirs2_core::ndarray::s![.., 0..n_components])
            .to_owned();
        let explained_variance = eigenvalues
            .slice(scirs2_core::ndarray::s![0..n_components])
            .to_owned();

        Ok(SimplePCAResult {
            components,
            mean,
            explained_variance,
        })
    }

    /// Simplified eigendecomposition
    fn simple_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Power iteration for largest eigenvalue
        let mut v = Array1::ones(n);
        for _ in 0..100 {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();
            if norm > 1e-12 {
                v = v_new / norm;
            }

            let eigenvalue = v.dot(&matrix.dot(&v));
            eigenvalues[0] = eigenvalue;

            for i in 0..n {
                eigenvectors[[i, 0]] = v[i];
            }
        }

        // Fill remaining eigenvalues (simplified)
        for i in 1..n {
            eigenvalues[i] = eigenvalues[0] / (i + 1) as Float;
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute component similarity using correlation
    fn compute_component_similarity(
        &self,
        components1: &Array2<Float>,
        components2: &Array2<Float>,
    ) -> Result<Float> {
        let (_, n_comp1) = components1.dim();
        let (_, n_comp2) = components2.dim();
        let n_comp = n_comp1.min(n_comp2);

        let mut total_similarity = 0.0;

        for i in 0..n_comp {
            let comp1 = components1.column(i);
            let comp2 = components2.column(i);

            let correlation = self
                .compute_correlation(&comp1.to_owned(), &comp2.to_owned())
                .abs();
            total_similarity += correlation;
        }

        Ok(total_similarity / n_comp as Float)
    }

    /// Compute eigenvalue similarity
    fn compute_eigenvalue_similarity(
        &self,
        eigenvals1: &Array1<Float>,
        eigenvals2: &Array1<Float>,
    ) -> Float {
        let n = eigenvals1.len().min(eigenvals2.len());
        if n == 0 {
            return 0.0;
        }

        let mut similarity = 0.0;
        for i in 0..n {
            let diff = (eigenvals1[i] - eigenvals2[i]).abs();
            let avg = (eigenvals1[i] + eigenvals2[i]) / 2.0;
            if avg > 1e-12 {
                similarity += 1.0 - diff / avg;
            }
        }

        similarity / n as Float
    }

    /// Compute Pearson correlation
    fn compute_correlation(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        let n = x.len();
        if n != y.len() || n == 0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            return 0.0;
        }

        num / (den_x * den_y).sqrt()
    }

    /// Compute standard deviation
    fn compute_std(&self, values: &[Float], mean: Float) -> Float {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance = values
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<Float>()
            / (values.len() - 1) as Float;

        variance.sqrt()
    }
}

/// Information criteria selector
pub struct InformationCriteriaSelector {
    config: ComponentSelectionConfig,
    criterion: InformationCriterion,
}

impl InformationCriteriaSelector {
    /// Create a new information criteria selector
    pub fn new(criterion: InformationCriterion) -> Self {
        Self {
            config: ComponentSelectionConfig::default(),
            criterion,
        }
    }

    /// Set configuration
    pub fn config(mut self, config: ComponentSelectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Select optimal number of components using information criteria
    pub fn select_components(&self, data: &Array2<Float>) -> Result<ComponentSelectionResult> {
        let (n_samples, n_features) = data.dim();
        let max_components = self.config.max_components.min(n_features.min(n_samples));

        let mut ic_scores = Vec::new();

        for n_comp in self.config.min_components..=max_components {
            let pca_result = self.fit_pca(data, n_comp)?;
            let ic_score = self.compute_information_criterion(data, &pca_result, n_comp)?;
            ic_scores.push(ic_score);
        }

        // Find optimal (minimum for AIC/BIC)
        let optimal_idx = ic_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let optimal_components = self.config.min_components + optimal_idx;

        Ok(ComponentSelectionResult {
            optimal_components,
            cv_scores: Array1::from_vec(ic_scores),
            component_range: (self.config.min_components..=max_components).collect(),
            method: SelectionMethod::InformationCriteria,
            metadata: HashMap::new(),
        })
    }

    /// Fit PCA (simplified)
    fn fit_pca(&self, data: &Array2<Float>, n_components: usize) -> Result<SimplePCAResult> {
        let (n_samples, _n_features) = data.dim();

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = data - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as Float;

        // Simplified eigendecomposition
        let (eigenvalues, eigenvectors) = self.simple_eigendecomposition(&cov_matrix)?;

        // Take top n_components
        let components = eigenvectors
            .slice(scirs2_core::ndarray::s![.., 0..n_components])
            .to_owned();
        let explained_variance = eigenvalues
            .slice(scirs2_core::ndarray::s![0..n_components])
            .to_owned();

        Ok(SimplePCAResult {
            components,
            mean,
            explained_variance,
        })
    }

    /// Simplified eigendecomposition
    fn simple_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Power iteration for largest eigenvalue
        let mut v = Array1::ones(n);
        for _ in 0..100 {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();
            if norm > 1e-12 {
                v = v_new / norm;
            }

            let eigenvalue = v.dot(&matrix.dot(&v));
            eigenvalues[0] = eigenvalue;

            for i in 0..n {
                eigenvectors[[i, 0]] = v[i];
            }
        }

        // Fill remaining eigenvalues (simplified)
        for i in 1..n {
            eigenvalues[i] = eigenvalues[0] / (i + 1) as Float;
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute information criterion
    fn compute_information_criterion(
        &self,
        data: &Array2<Float>,
        pca: &SimplePCAResult,
        n_components: usize,
    ) -> Result<Float> {
        let (n_samples, _) = data.dim();

        // Compute reconstruction error (negative log-likelihood proxy)
        let centered_data = data - &pca.mean.clone().insert_axis(Axis(0));
        let projected = centered_data.dot(&pca.components);
        let reconstructed =
            projected.dot(&pca.components.t()) + &pca.mean.clone().insert_axis(Axis(0));
        let diff = data - &reconstructed;
        let mse = diff.iter().map(|&x| x * x).sum::<Float>() / (n_samples as Float);

        // Negative log-likelihood (assuming Gaussian errors)
        let log_likelihood =
            -0.5 * n_samples as Float * (2.0 * std::f64::consts::PI * mse).ln() as Float;

        // Number of parameters
        let n_params = n_components * (data.ncols() + 1); // Components + explained variance

        Ok(match self.criterion {
            InformationCriterion::AIC => -2.0 * log_likelihood + 2.0 * n_params as Float,
            InformationCriterion::BIC => {
                -2.0 * log_likelihood + (n_params as Float) * (n_samples as Float).ln()
            }
            InformationCriterion::HQC => {
                -2.0 * log_likelihood + 2.0 * (n_params as Float) * (n_samples as Float).ln().ln()
            }
            InformationCriterion::CAIC => {
                -2.0 * log_likelihood + (n_params as Float) * ((n_samples as Float).ln() + 1.0)
            }
        })
    }
}

/// Selection method used
#[derive(Debug, Clone, Copy)]
pub enum SelectionMethod {
    CrossValidation,
    Bootstrap,
    InformationCriteria,
    ParallelAnalysis,
}

/// Component selection result
#[derive(Debug, Clone)]
pub struct ComponentSelectionResult {
    /// Optimal number of components
    pub optimal_components: usize,
    /// Scores for each number of components tested
    pub cv_scores: Array1<Float>,
    /// Range of components tested
    pub component_range: Vec<usize>,
    /// Selection method used
    pub method: SelectionMethod,
    /// Additional metadata
    pub metadata: HashMap<String, Vec<Float>>,
}

impl ComponentSelectionResult {
    /// Get the score for the optimal number of components
    pub fn optimal_score(&self) -> Float {
        let optimal_idx = self.optimal_components - self.component_range[0];
        self.cv_scores[optimal_idx]
    }

    /// Check if the optimal selection is stable (low variance)
    pub fn is_stable(&self) -> bool {
        if let Some(errors) = self.metadata.get("reconstruction_errors") {
            let optimal_idx = self.optimal_components - self.component_range[0];
            if optimal_idx < errors.len() {
                return errors[optimal_idx] < 0.1; // Threshold for stability
            }
        }
        true
    }
}

/// Stability assessment result
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Mean component similarity across bootstrap samples
    pub mean_component_similarity: Float,
    /// Standard deviation of component similarities
    pub std_component_similarity: Float,
    /// Component similarities for each bootstrap sample
    pub component_similarities: Array1<Float>,
    /// Eigenvalue stability
    pub eigenvalue_stability: Float,
    /// Whether the configuration is considered stable
    pub is_stable: bool,
}

impl StabilityResult {
    /// Get stability confidence (higher is more stable)
    pub fn stability_confidence(&self) -> Float {
        if self.std_component_similarity > 0.0 {
            self.mean_component_similarity / self.std_component_similarity
        } else {
            self.mean_component_similarity
        }
    }
}

/// Parallel analysis for factor number determination
pub struct ParallelAnalysis {
    config: ComponentSelectionConfig,
    n_simulations: usize,
}

impl ParallelAnalysis {
    /// Create a new parallel analysis instance
    pub fn new() -> Self {
        Self {
            config: ComponentSelectionConfig::default(),
            n_simulations: 100,
        }
    }

    /// Set number of simulations
    pub fn n_simulations(mut self, n_simulations: usize) -> Self {
        self.n_simulations = n_simulations;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Perform parallel analysis to determine number of factors
    pub fn analyze(&self, data: &Array2<Float>) -> Result<ComponentSelectionResult> {
        let (n_samples, n_features) = data.dim();
        let mut rng = match self.config.random_state {
            Some(seed) => thread_rng(), // TODO: use seeded rng when available
            None => thread_rng(),
        };

        // Compute eigenvalues for original data
        let original_eigenvalues = self.compute_eigenvalues(data)?;

        // Generate random data eigenvalues
        let mut random_eigenvalues = Array2::zeros((self.n_simulations, n_features));

        for sim in 0..self.n_simulations {
            let random_data = self.generate_random_data(n_samples, n_features, &mut rng);
            let eigenvals = self.compute_eigenvalues(&random_data)?;

            for (i, &val) in eigenvals.iter().enumerate() {
                random_eigenvalues[[sim, i]] = val;
            }
        }

        // Compute 95th percentile of random eigenvalues
        let mut percentile_eigenvalues = Array1::zeros(n_features);
        for i in 0..n_features {
            let mut column: Vec<Float> = random_eigenvalues.column(i).to_vec();
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile_idx = (0.95 * self.n_simulations as Float) as usize;
            percentile_eigenvalues[i] = column[percentile_idx.min(column.len() - 1)];
        }

        // Count components where original eigenvalues exceed random eigenvalues
        let mut optimal_components = 0;
        let mut scores = Vec::new();

        for i in 0..n_features {
            let difference = original_eigenvalues[i] - percentile_eigenvalues[i];
            scores.push(difference);

            if original_eigenvalues[i] > percentile_eigenvalues[i] {
                optimal_components = i + 1;
            } else {
                break;
            }
        }

        Ok(ComponentSelectionResult {
            optimal_components,
            cv_scores: Array1::from_vec(scores),
            component_range: (1..=n_features).collect(),
            method: SelectionMethod::ParallelAnalysis,
            metadata: HashMap::from([
                (
                    "original_eigenvalues".to_string(),
                    original_eigenvalues.to_vec(),
                ),
                (
                    "random_eigenvalues_95th".to_string(),
                    percentile_eigenvalues.to_vec(),
                ),
            ]),
        })
    }

    /// Compute eigenvalues of covariance matrix
    fn compute_eigenvalues(&self, data: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, _) = data.dim();

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = data - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as Float;

        // Simplified eigendecomposition
        let (eigenvalues, _) = self.simple_eigendecomposition(&cov_matrix)?;

        Ok(eigenvalues)
    }

    /// Generate random data with same dimensions
    fn generate_random_data(
        &self,
        n_samples: usize,
        n_features: usize,
        rng: &mut impl Rng,
    ) -> Array2<Float> {
        let mut random_data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                random_data[[i, j]] = rng.gen::<Float>() - 0.5; // Centered random values
            }
        }

        random_data
    }

    /// Simplified eigendecomposition
    fn simple_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::eye(n);

        // Power iteration for largest eigenvalue
        let mut v = Array1::ones(n);
        for _ in 0..100 {
            let v_new = matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();
            if norm > 1e-12 {
                v = v_new / norm;
            }

            let eigenvalue = v.dot(&matrix.dot(&v));
            eigenvalues[0] = eigenvalue;

            for i in 0..n {
                eigenvectors[[i, 0]] = v[i];
            }
        }

        // Fill remaining eigenvalues (simplified)
        for i in 1..n {
            eigenvalues[i] = eigenvalues[0] / (i + 1) as Float;
        }

        Ok((eigenvalues, eigenvectors))
    }
}

impl Default for ParallelAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cross_validation_selector() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let selector = CrossValidationSelector::new(CrossValidationMethod::KFold)
            .max_components(2)
            .n_folds(3)
            .random_state(42);

        let result = selector.select_pca_components(&data).unwrap();

        assert!(result.optimal_components >= 1);
        assert!(result.optimal_components <= 2);
        assert_eq!(result.cv_scores.len(), 2);
    }

    #[test]
    fn test_bootstrap_selector() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let selector = BootstrapSelector::new(BootstrapMethod::Standard).n_bootstrap(10);

        let result = selector.assess_stability(&data, 1).unwrap();

        assert!(result.mean_component_similarity >= 0.0);
        assert!(result.mean_component_similarity <= 1.0);
        assert_eq!(result.component_similarities.len(), 10);
    }

    #[test]
    fn test_information_criteria_selector() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let selector = InformationCriteriaSelector::new(InformationCriterion::AIC);
        let result = selector.select_components(&data).unwrap();

        assert!(result.optimal_components >= 1);
        assert!(result.cv_scores.len() > 0);
    }

    #[test]
    fn test_parallel_analysis() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let pa = ParallelAnalysis::new().n_simulations(10).random_state(42);

        let result = pa.analyze(&data).unwrap();

        assert!(result.optimal_components >= 0);
        assert_eq!(result.cv_scores.len(), 3);
    }

    #[test]
    fn test_k_fold_split() {
        let selector = CrossValidationSelector::new(CrossValidationMethod::KFold)
            .n_folds(3)
            .random_state(42);

        let folds = selector.create_folds(10).unwrap();

        assert_eq!(folds.len(), 3);

        // Check that all indices are covered
        let mut all_test_indices: Vec<usize> = Vec::new();
        for (_, test_idx) in &folds {
            all_test_indices.extend(test_idx);
        }
        all_test_indices.sort();

        let expected_indices: Vec<usize> = (0..10).collect();
        assert_eq!(all_test_indices, expected_indices);
    }

    #[test]
    fn test_leave_one_out_split() {
        let selector = CrossValidationSelector::new(CrossValidationMethod::LeaveOneOut);
        let folds = selector.create_folds(5).unwrap();

        assert_eq!(folds.len(), 5);

        for (i, (train_idx, test_idx)) in folds.iter().enumerate() {
            assert_eq!(test_idx.len(), 1);
            assert_eq!(test_idx[0], i);
            assert_eq!(train_idx.len(), 4);
            assert!(!train_idx.contains(&i));
        }
    }
}
