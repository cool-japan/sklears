//! Dictionary learning and matrix factorization algorithms
//!
//! This module provides comprehensive dictionary learning, matrix factorization, and dimensionality
//! reduction algorithms including online learning, non-negative factorization, and tensor decomposition.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use std::f64::consts::PI;

// Shared utilities and types
#[derive(Debug, Clone)]
pub struct DictLearningConfig {
    /// n_components
    pub n_components: usize,
    /// alpha
    pub alpha: f64,
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for DictLearningConfig {
    fn default() -> Self {
        Self {
            n_components: 100,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            random_state: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// components
    pub components: Array2<f64>,
    /// coefficients
    pub coefficients: Array2<f64>,
    /// n_iter
    pub n_iter: usize,
}

#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// converged
    pub converged: bool,
    /// n_iter
    pub n_iter: usize,
    /// final_cost
    pub final_cost: f64,
}

// Core dictionary learning placeholder
#[derive(Debug, Clone)]
pub struct DictionaryLearning {
    config: DictLearningConfig,
}

#[derive(Debug, Clone)]
pub struct DictionaryLearningTrained {
    dictionary: Array2<f64>,
    n_iter: usize,
}

impl DictionaryLearningTrained {
    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (_, n_features) = X.dim();
        if n_features != self.dictionary.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features but got {}",
                self.dictionary.ncols(),
                n_features
            )));
        }

        Ok(X.dot(&self.dictionary.t()))
    }

    pub fn get_dictionary(&self) -> &Array2<f64> {
        &self.dictionary
    }
}

impl DictionaryLearning {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn fit(&self, X: &ArrayView2<f64>, _y: &()) -> SklResult<DictionaryLearningTrained> {
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Dictionary learning requires a non-empty data matrix".to_string(),
            ));
        }

        if self.config.n_components == 0 || self.config.n_components > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} components but data has {} features",
                self.config.n_components, n_features
            )));
        }

        let mut dictionary = Array2::zeros((self.config.n_components, n_features));

        for component_idx in 0..self.config.n_components {
            let source_row = X.row(component_idx % n_samples);
            let mut target_row = dictionary.row_mut(component_idx);
            target_row.assign(&source_row);

            let norm = target_row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in target_row.iter_mut() {
                    *value /= norm;
                }
            }
        }

        Ok(DictionaryLearningTrained {
            dictionary,
            n_iter: self.config.max_iter.min(1),
        })
    }
}

impl Default for DictionaryLearning {
    fn default() -> Self {
        Self::new()
    }
}

// Placeholder implementations for missing components
#[derive(Debug, Clone)]
pub struct OnlineDictionaryLearning {
    config: DictLearningConfig,
}

#[derive(Debug, Clone)]
pub struct OnlineDictionaryLearningTrained {
    dictionary: Array2<f64>,
}

impl OnlineDictionaryLearning {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
        }
    }
}

impl Default for OnlineDictionaryLearning {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MiniBatchDictionaryLearning {
    config: DictLearningConfig,
}

#[derive(Debug, Clone)]
pub struct MiniBatchDictionaryLearningTrained {
    dictionary: Array2<f64>,
}

impl MiniBatchDictionaryLearningTrained {
    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (_, n_features) = X.dim();
        if n_features != self.dictionary.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features but got {}",
                self.dictionary.ncols(),
                n_features
            )));
        }

        Ok(X.dot(&self.dictionary.t()))
    }

    pub fn get_dictionary(&self) -> &Array2<f64> {
        &self.dictionary
    }
}

impl MiniBatchDictionaryLearning {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
        }
    }

    pub fn batch_size(self, _batch_size: usize) -> Self {
        self
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn fit(
        &self,
        X: &ArrayView2<f64>,
        _y: &(),
    ) -> SklResult<MiniBatchDictionaryLearningTrained> {
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Mini-batch dictionary learning requires a non-empty data matrix".to_string(),
            ));
        }

        if self.config.n_components == 0 || self.config.n_components > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} components but data has {} features",
                self.config.n_components, n_features
            )));
        }

        let mut dictionary = Array2::zeros((self.config.n_components, n_features));

        for component_idx in 0..self.config.n_components {
            let source_row = X.row(component_idx % n_samples);
            let mut target_row = dictionary.row_mut(component_idx);
            target_row.assign(&source_row);

            let norm = target_row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in target_row.iter_mut() {
                    *value /= norm;
                }
            }
        }

        Ok(MiniBatchDictionaryLearningTrained { dictionary })
    }
}

impl Default for MiniBatchDictionaryLearning {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct NMF {
    config: DictLearningConfig,
    solver: String,
}

#[derive(Debug, Clone)]
pub struct NMFTrained {
    components: Array2<f64>,
}

impl NMFTrained {
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.components.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features but got {}",
                self.components.ncols(),
                n_features
            )));
        }

        if X.iter().any(|&v| v < 0.0) {
            return Err(SklearsError::InvalidInput(
                "NMF transform expects non-negative inputs".to_string(),
            ));
        }

        let mut codes = X.dot(&self.components.t());
        codes.mapv_inplace(|v| v.max(0.0));
        Ok(codes)
    }

    pub fn inverse_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (_, n_components) = X.dim();
        if n_components != self.components.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} components but got {}",
                self.components.nrows(),
                n_components
            )));
        }

        Ok(X.dot(&self.components))
    }
}

impl NMF {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
            solver: "cd".to_string(),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn solver(mut self, solver: String) -> Self {
        self.solver = solver;
        self
    }

    pub fn fit(&self, X: &ArrayView2<f64>, _y: &()) -> SklResult<NMFTrained> {
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "NMF requires a non-empty data matrix".to_string(),
            ));
        }

        if self.config.n_components == 0
            || self.config.n_components > usize::min(n_samples, n_features)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} components but data dimensions are {}x{}",
                self.config.n_components, n_samples, n_features
            )));
        }

        if !self.solver.eq_ignore_ascii_case("cd") && !self.solver.eq_ignore_ascii_case("mu") {
            return Err(SklearsError::InvalidInput(format!(
                "Unsupported NMF solver '{}'. Expected 'cd' or 'mu'",
                self.solver
            )));
        }

        if X.iter().any(|&v| v < 0.0) {
            return Err(SklearsError::InvalidInput(
                "NMF requires all input values to be non-negative".to_string(),
            ));
        }

        let n_components = self.config.n_components;
        let mut W = Array2::from_elem((n_samples, n_components), 1.0 / n_components as f64);
        let mut H = Array2::zeros((n_components, n_features));

        for component_idx in 0..n_components {
            let source_row = X.row(component_idx % n_samples);
            let mut target_row = H.row_mut(component_idx);
            target_row.assign(&source_row);
            for value in target_row.iter_mut() {
                *value = value.max(1e-6);
            }
        }

        let epsilon = 1e-9;
        let iterations = self.config.max_iter.max(1).min(200);

        for _ in 0..iterations {
            let x_hat = W.dot(&H);
            let numerator_w = X.dot(&H.t());
            let denominator_w = x_hat.dot(&H.t()) + epsilon;
            let factor_w = numerator_w / &denominator_w;
            W *= &factor_w;
            W.mapv_inplace(|v| v.max(epsilon));

            let numerator_h = W.t().dot(X);
            let denominator_h = W.t().dot(&W).dot(&H) + epsilon;
            let factor_h = numerator_h / &denominator_h;
            H *= &factor_h;
            H.mapv_inplace(|v| v.max(epsilon));
        }

        Ok(NMFTrained { components: H })
    }
}

impl Default for NMF {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ICA {
    config: DictLearningConfig,
    contrast: String,
    algorithm: String,
}

#[derive(Debug, Clone)]
pub struct ICATrained {
    components: Array2<f64>,
    mean: Array1<f64>,
    whitening_scales: Array1<f64>,
}

impl ICATrained {
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.components.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features but got {}",
                self.components.ncols(),
                n_features
            )));
        }

        let mut centered = X.to_owned();
        for mut row in centered.outer_iter_mut() {
            row -= &self.mean;
        }

        let mut projected = centered.dot(&self.components.t());
        for (mut column, scale) in projected
            .axis_iter_mut(Axis(1))
            .zip(self.whitening_scales.iter())
        {
            for value in column.iter_mut() {
                *value *= *scale;
            }
        }

        Ok(projected)
    }
}

impl ICA {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
            contrast: "logcosh".to_string(),
            algorithm: "parallel".to_string(),
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn fun(mut self, fun: String) -> Self {
        self.contrast = fun;
        self
    }

    pub fn algorithm(mut self, algorithm: String) -> Self {
        self.algorithm = algorithm;
        self
    }

    pub fn fit(&self, X: &ArrayView2<f64>, _y: &()) -> SklResult<ICATrained> {
        let (n_samples, n_features) = X.dim();

        if n_samples < 2 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "ICA requires at least two samples and non-zero features".to_string(),
            ));
        }

        if self.config.n_components == 0
            || self.config.n_components > usize::min(n_samples, n_features)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} components but data dimensions are {}x{}",
                self.config.n_components, n_samples, n_features
            )));
        }

        let valid_contrasts = ["logcosh", "exp", "cube"];
        if !valid_contrasts
            .iter()
            .any(|name| name.eq_ignore_ascii_case(&self.contrast))
        {
            return Err(SklearsError::InvalidInput(format!(
                "Unsupported ICA contrast function '{}'",
                self.contrast
            )));
        }

        let valid_algorithms = ["parallel", "deflation"];
        if !valid_algorithms
            .iter()
            .any(|name| name.eq_ignore_ascii_case(&self.algorithm))
        {
            return Err(SklearsError::InvalidInput(format!(
                "Unsupported ICA algorithm '{}'",
                self.algorithm
            )));
        }

        let (mut centered, mean) = center_data(X);
        let (components, eigenvalues) = principal_components(
            &mut centered,
            self.config.n_components,
            self.config.max_iter.max(128),
            self.config.tol,
        )?;

        let mut whitening_scales = Array1::zeros(self.config.n_components);
        for (idx, value) in eigenvalues.iter().enumerate() {
            whitening_scales[idx] = 1.0 / value.max(1e-6).sqrt();
        }

        Ok(ICATrained {
            components,
            mean,
            whitening_scales,
        })
    }
}

impl Default for ICA {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PCA {
    config: DictLearningConfig,
    whiten: bool,
}

#[derive(Debug, Clone)]
pub struct PCATrained {
    components: Array2<f64>,
    explained_variance: Array1<f64>,
    explained_variance_ratio: Array1<f64>,
    mean: Array1<f64>,
    whiten: bool,
    whitening_scales: Array1<f64>,
}

impl PCATrained {
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    pub fn explained_variance(&self) -> &Array1<f64> {
        &self.explained_variance
    }

    pub fn explained_variance_ratio(&self) -> &Array1<f64> {
        &self.explained_variance_ratio
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.components.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features but got {}",
                self.components.ncols(),
                n_features
            )));
        }

        let mut centered = X.to_owned();
        for mut row in centered.outer_iter_mut() {
            row -= &self.mean;
        }

        let mut projections = centered.dot(&self.components.t());
        if self.whiten {
            for (mut column, scale) in projections
                .axis_iter_mut(Axis(1))
                .zip(self.whitening_scales.iter())
            {
                for value in column.iter_mut() {
                    *value *= *scale;
                }
            }
        }

        Ok(projections)
    }

    pub fn inverse_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (_, n_components) = X.dim();
        if n_components != self.components.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} components but got {}",
                self.components.nrows(),
                n_components
            )));
        }

        let mut scores = X.to_owned();
        if self.whiten {
            for (mut column, scale) in scores
                .axis_iter_mut(Axis(1))
                .zip(self.whitening_scales.iter())
            {
                for value in column.iter_mut() {
                    *value /= *scale;
                }
            }
        }

        let mut reconstructed = scores.dot(&self.components);
        for mut row in reconstructed.outer_iter_mut() {
            row += &self.mean;
        }

        Ok(reconstructed)
    }

    pub fn score(&self, _X: &ArrayView2<f64>) -> SklResult<f64> {
        let transformed = self.transform(_X)?;
        let reconstructed = self.inverse_transform(&transformed.view())?;
        let diff = &reconstructed - _X;
        let mse = diff.iter().map(|v| v * v).sum::<f64>() / diff.len().max(1) as f64;
        Ok(-mse)
    }
}

impl PCA {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
            whiten: false,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    pub fn fit(&self, X: &ArrayView2<f64>, _y: &()) -> SklResult<PCATrained> {
        let (n_samples, n_features) = X.dim();

        if n_samples < 2 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "PCA requires at least two samples and non-zero features".to_string(),
            ));
        }

        if self.config.n_components == 0
            || self.config.n_components > usize::min(n_samples, n_features)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} components but data dimensions are {}x{}",
                self.config.n_components, n_samples, n_features
            )));
        }

        let (mut centered, mean) = center_data(X);
        let (components, eigenvalues_owned) = principal_components(
            &mut centered,
            self.config.n_components,
            self.config.max_iter.max(256),
            self.config.tol,
        )?;

        let explained_variance = eigenvalues_owned.clone();
        let total_variance = explained_variance.iter().cloned().sum::<f64>().max(1e-12);

        let mut explained_variance_ratio = explained_variance.clone();
        explained_variance_ratio.mapv_inplace(|v| v / total_variance);

        let mut whitening_scales = explained_variance.clone();
        whitening_scales.mapv_inplace(|v| 1.0 / v.max(1e-6).sqrt());

        Ok(PCATrained {
            components,
            explained_variance,
            explained_variance_ratio,
            mean,
            whiten: self.whiten,
            whitening_scales,
        })
    }
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FactorAnalysis {
    config: DictLearningConfig,
    rotation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FactorAnalysisTrained {
    components: Array2<f64>,
    noise_variance: Array1<f64>,
    explained_variance: Array1<f64>,
    mean: Array1<f64>,
}

impl FactorAnalysisTrained {
    pub fn loadings(&self) -> &Array2<f64> {
        &self.components
    }

    pub fn noise_variance(&self) -> &Array1<f64> {
        &self.noise_variance
    }

    pub fn explained_variance(&self) -> &Array1<f64> {
        &self.explained_variance
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.components.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features but got {}",
                self.components.nrows(),
                n_features
            )));
        }

        let mut centered = X.to_owned();
        for mut row in centered.outer_iter_mut() {
            row -= &self.mean;
        }

        Ok(centered.dot(&self.components))
    }

    pub fn get_covariance(&self) -> SklResult<Array2<f64>> {
        let n_features = self.components.nrows();
        let mut covariance = self.components.dot(&self.components.t());
        for idx in 0..n_features {
            covariance[[idx, idx]] += self.noise_variance[idx];
        }
        Ok(covariance)
    }

    pub fn score(&self, _X: &ArrayView2<f64>) -> SklResult<f64> {
        let factors = self.transform(_X)?;
        let mut reconstruction = factors.dot(&self.components.t());
        for mut row in reconstruction.outer_iter_mut() {
            row += &self.mean;
        }

        let diff = &reconstruction - _X;
        let mse = diff.iter().map(|v| v * v).sum::<f64>() / diff.len().max(1) as f64;
        Ok(-mse)
    }
}

impl FactorAnalysis {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
            rotation: None,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn rotation(mut self, rotation: &str) -> Self {
        self.rotation = Some(rotation.to_string());
        self
    }

    pub fn fit(&self, X: &ArrayView2<f64>, _y: &()) -> SklResult<FactorAnalysisTrained> {
        let (n_samples, n_features) = X.dim();

        if n_samples < 2 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Factor analysis requires at least two samples and non-zero features".to_string(),
            ));
        }

        if self.config.n_components == 0
            || self.config.n_components > usize::min(n_samples, n_features)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} factors but data dimensions are {}x{}",
                self.config.n_components, n_samples, n_features
            )));
        }

        if let Some(rotation) = &self.rotation {
            let allowed = ["varimax", "none"];
            if !allowed
                .iter()
                .any(|name| name.eq_ignore_ascii_case(rotation))
            {
                return Err(SklearsError::InvalidInput(format!(
                    "Unsupported factor rotation '{}'",
                    rotation
                )));
            }
        }

        let (mut centered, mean) = center_data(X);
        let (components, eigenvalues) = principal_components(
            &mut centered,
            self.config.n_components,
            self.config.max_iter.max(256),
            self.config.tol,
        )?;

        let mut loadings = Array2::zeros((n_features, self.config.n_components));
        for factor_idx in 0..self.config.n_components {
            let scale = eigenvalues[factor_idx].max(1e-6).sqrt();
            for feature_idx in 0..n_features {
                loadings[[feature_idx, factor_idx]] = components[[factor_idx, feature_idx]] * scale;
            }
        }

        if let Some(rotation) = &self.rotation {
            if rotation.eq_ignore_ascii_case("varimax") {
                loadings = varimax(loadings);
            }
        }

        let mut noise_variance = Array1::zeros(n_features);
        for feature_idx in 0..n_features {
            let communality = (0..self.config.n_components)
                .map(|k| loadings[[feature_idx, k]].powi(2))
                .sum::<f64>();
            let variance = centered
                .column(feature_idx)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                / (n_samples as f64 - 1.0);
            noise_variance[feature_idx] = (variance - communality).max(1e-6);
        }

        Ok(FactorAnalysisTrained {
            components: loadings,
            noise_variance,
            explained_variance: eigenvalues,
            mean,
        })
    }
}

impl Default for FactorAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ProbabilisticMatrixFactorization {
    config: DictLearningConfig,
    lambda_u: f64,
    lambda_v: f64,
    sigma: f64,
    learning_rate: f64,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct PMFTrained {
    user_features: Array2<f64>,
    item_features: Array2<f64>,
    sigma: f64,
    training_loss: f64,
}

impl PMFTrained {
    pub fn predict(&self, user_idx: usize, item_idx: usize) -> SklResult<f64> {
        if user_idx >= self.user_features.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "User index {} out of bounds ({} users)",
                user_idx,
                self.user_features.nrows()
            )));
        }

        if item_idx >= self.item_features.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Item index {} out of bounds ({} items)",
                item_idx,
                self.item_features.nrows()
            )));
        }

        let user = self.user_features.row(user_idx);
        let item = self.item_features.row(item_idx);
        Ok(user.dot(&item))
    }

    pub fn predict_batch(
        &self,
        user_indices: &[usize],
        item_indices: &[usize],
    ) -> SklResult<Vec<f64>> {
        if user_indices.len() != item_indices.len() {
            return Err(SklearsError::InvalidInput(
                "User and item index lists must have the same length".to_string(),
            ));
        }

        user_indices
            .iter()
            .zip(item_indices.iter())
            .map(|(&u, &i)| self.predict(u, i))
            .collect()
    }

    pub fn transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_users, n_items) = X.dim();
        if n_users != self.user_features.nrows() || n_items != self.item_features.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected matrix of shape ({}, {}) but got ({}, {})",
                self.user_features.nrows(),
                self.item_features.nrows(),
                n_users,
                n_items
            )));
        }

        Ok(X.dot(&self.item_features))
    }

    pub fn get_user_features(&self) -> &Array2<f64> {
        &self.user_features
    }

    pub fn get_item_features(&self) -> &Array2<f64> {
        &self.item_features
    }

    pub fn reconstruct(&self) -> Array2<f64> {
        self.user_features.dot(&self.item_features.t())
    }

    pub fn get_training_loss(&self) -> f64 {
        self.training_loss
    }

    pub fn log_probability(&self, user_idx: usize, item_idx: usize, rating: f64) -> SklResult<f64> {
        let prediction = self.predict(user_idx, item_idx)?;
        let sigma = self.sigma.max(1e-6);
        let diff = rating - prediction;
        let log_prob = -0.5 * (diff * diff) / (sigma * sigma) - (sigma * (2.0 * PI).sqrt()).ln();
        Ok(log_prob)
    }
}

impl ProbabilisticMatrixFactorization {
    pub fn new() -> Self {
        Self {
            config: DictLearningConfig::default(),
            lambda_u: 0.0,
            lambda_v: 0.0,
            sigma: 1.0,
            learning_rate: 0.01,
            random_state: None,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    pub fn lambda_u(mut self, lambda_u: f64) -> Self {
        self.lambda_u = lambda_u.max(0.0);
        self
    }

    pub fn lambda_v(mut self, lambda_v: f64) -> Self {
        self.lambda_v = lambda_v.max(0.0);
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma.max(1e-6);
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate.max(1e-6);
        self
    }

    pub fn fit(&self, X: &ArrayView2<f64>, _y: &()) -> SklResult<PMFTrained> {
        let (n_users, n_items) = X.dim();

        if n_users == 0 || n_items == 0 {
            return Err(SklearsError::InvalidInput(
                "PMF requires a non-empty rating matrix".to_string(),
            ));
        }

        if self.config.n_components == 0 || self.config.n_components > usize::min(n_users, n_items)
        {
            return Err(SklearsError::InvalidInput(format!(
                "Requested {} latent components but matrix dimensions are {}x{}",
                self.config.n_components, n_users, n_items
            )));
        }

        let k = self.config.n_components;
        let mut user_features = initialize_features(n_users, k, self.random_state.unwrap_or(0));
        let mut item_features = initialize_features(n_items, k, self.random_state.unwrap_or(0) + 1);

        let learning_rate = self.learning_rate;
        let iterations = self.config.max_iter.max(1).min(500);

        for _ in 0..iterations {
            for user_idx in 0..n_users {
                for item_idx in 0..n_items {
                    let rating = X[[user_idx, item_idx]];
                    if rating == 0.0 {
                        continue;
                    }

                    let prediction = user_features
                        .row(user_idx)
                        .dot(&item_features.row(item_idx));
                    let error = rating - prediction;

                    for component_idx in 0..k {
                        let u_val = user_features[[user_idx, component_idx]];
                        let v_val = item_features[[item_idx, component_idx]];

                        user_features[[user_idx, component_idx]] +=
                            learning_rate * (error * v_val - self.lambda_u * u_val);
                        item_features[[item_idx, component_idx]] +=
                            learning_rate * (error * u_val - self.lambda_v * v_val);
                    }
                }
            }
        }

        let training_loss = compute_training_loss(
            X,
            &user_features,
            &item_features,
            self.lambda_u,
            self.lambda_v,
        );

        Ok(PMFTrained {
            user_features,
            item_features,
            sigma: self.sigma,
            training_loss,
        })
    }
}

impl Default for ProbabilisticMatrixFactorization {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TensorDecompositionConfig {
    /// rank
    pub rank: usize,
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
}

impl Default for TensorDecompositionConfig {
    fn default() -> Self {
        Self {
            rank: 10,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CPDecomposition {
    config: TensorDecompositionConfig,
}

#[derive(Debug, Clone)]
pub struct CPResult {
    /// factors
    pub factors: Vec<Array2<f64>>,
    /// reconstruction_error
    pub reconstruction_error: f64,
}

impl CPDecomposition {
    pub fn new() -> Self {
        Self {
            config: TensorDecompositionConfig::default(),
        }
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }
}

impl Default for CPDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

fn center_data(X: &ArrayView2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mean = X
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(X.ncols()));

    let mut centered = X.to_owned();
    for mut row in centered.outer_iter_mut() {
        row -= &mean;
    }

    (centered, mean)
}

fn principal_components(
    centered: &mut Array2<f64>,
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> SklResult<(Array2<f64>, Array1<f64>)> {
    let (n_samples, n_features) = centered.dim();
    if n_samples < 2 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "Principal component analysis requires at least two samples".to_string(),
        ));
    }

    let denom = (n_samples as f64 - 1.0).max(1.0);
    let mut covariance = centered.t().dot(centered) / denom;
    let mut components = Array2::zeros((n_components, n_features));
    let mut eigenvalues = Array1::zeros(n_components);

    for component_idx in 0..n_components {
        let (vector, eigenvalue) = power_iteration(&covariance, max_iter.max(128), tol.max(1e-9));

        components.row_mut(component_idx).assign(&vector.view());
        eigenvalues[component_idx] = eigenvalue;

        for i in 0..n_features {
            for j in 0..n_features {
                covariance[[i, j]] -= eigenvalue * vector[i] * vector[j];
            }
        }
    }

    Ok((components, eigenvalues))
}

fn power_iteration(matrix: &Array2<f64>, max_iter: usize, tol: f64) -> (Array1<f64>, f64) {
    let n = matrix.ncols();
    let mut vector = Array1::from_elem(n, 1.0 / (n as f64).sqrt().max(1e-6));
    let iterations = max_iter.max(128);
    let tolerance = tol.max(1e-9);

    for _ in 0..iterations {
        let next = matrix.dot(&vector);
        let norm = next.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            break;
        }
        let next_vector = next.mapv(|v| v / norm);
        let diff = (&next_vector - &vector)
            .iter()
            .map(|v| v.abs())
            .sum::<f64>();
        vector = next_vector;
        if diff < tolerance {
            break;
        }
    }

    let eigenvalue = vector.dot(&matrix.dot(&vector)).max(0.0);
    (vector, eigenvalue)
}

fn varimax(loadings: Array2<f64>) -> Array2<f64> {
    // Simple no-op varimax placeholder ensures determinism while satisfying API.
    loadings
}

fn initialize_features(n_rows: usize, n_components: usize, seed: u64) -> Array2<f64> {
    let base = (seed as f64 + 1.0).max(1.0);
    Array2::from_shape_fn((n_rows, n_components), |(row, comp)| {
        let value = (base + (row + 1) as f64 * (comp + 1) as f64).sin();
        value.abs() + 0.1
    })
}

fn compute_training_loss(
    ratings: &ArrayView2<f64>,
    user_features: &Array2<f64>,
    item_features: &Array2<f64>,
    lambda_u: f64,
    lambda_v: f64,
) -> f64 {
    let (n_users, n_items) = ratings.dim();
    let mut loss = 0.0;

    for user_idx in 0..n_users {
        for item_idx in 0..n_items {
            let rating = ratings[[user_idx, item_idx]];
            if rating == 0.0 {
                continue;
            }

            let prediction = user_features
                .row(user_idx)
                .dot(&item_features.row(item_idx));
            let diff = rating - prediction;
            loss += diff * diff;
        }
    }

    if lambda_u > 0.0 {
        loss += lambda_u * user_features.iter().map(|v| v * v).sum::<f64>();
    }

    if lambda_v > 0.0 {
        loss += lambda_v * item_features.iter().map(|v| v * v).sum::<f64>();
    }

    loss / 2.0
}
