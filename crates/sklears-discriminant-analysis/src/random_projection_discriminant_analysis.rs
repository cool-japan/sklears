//! Random Projection Discriminant Analysis
//!
//! This module implements Random Projection Discriminant Analysis (RPDA) which combines
//! random projections for dimensionality reduction with discriminant analysis for classification.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Types of random projection methods
#[derive(Debug, Clone)]
pub enum ProjectionType {
    /// Gaussian random projection
    Gaussian,
    /// Sparse random projection (Achlioptas)
    Sparse { density: Float },
    /// Very sparse random projection
    VerySparse { density: Float },
    /// Circulant random projection
    Circulant,
    /// Fast Johnson-Lindenstrauss transform
    FastJL,
}

/// Types of discriminant methods to use after projection
#[derive(Debug, Clone)]
pub enum DiscriminantMethod {
    LDA {
        shrinkage: Option<Float>,
    },
    /// Quadratic Discriminant Analysis
    QDA {
        reg_param: Float,
    },
    /// Regularized Discriminant Analysis
    RDA {
        alpha: Float,
        gamma: Float,
    },
}

/// Configuration for Random Projection Discriminant Analysis
#[derive(Debug, Clone)]
pub struct RandomProjectionDiscriminantAnalysisConfig {
    /// Number of components in the random projection
    pub n_components: usize,
    /// Type of random projection
    pub projection_type: ProjectionType,
    /// Discriminant method to use after projection
    pub discriminant_method: DiscriminantMethod,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Whether to normalize the projected features
    pub normalize: bool,
    /// Johnson-Lindenstrauss distortion parameter
    pub eps: Float,
    /// Whether to automatically determine number of components
    pub auto_components: bool,
    /// Number of random projections for ensemble
    pub n_projections: usize,
    /// Ensemble combination method
    pub ensemble_method: String,
    /// Whether to use adaptive projection selection
    pub adaptive_selection: bool,
    /// Selection criterion for adaptive method
    pub selection_criterion: String,
}

impl Default for RandomProjectionDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            n_components: 100,
            projection_type: ProjectionType::Gaussian,
            discriminant_method: DiscriminantMethod::LDA { shrinkage: None },
            random_state: None,
            normalize: true,
            eps: 0.1,
            auto_components: false,
            n_projections: 1,
            ensemble_method: "majority".to_string(),
            adaptive_selection: false,
            selection_criterion: "accuracy".to_string(),
        }
    }
}

/// Random Projection Discriminant Analysis
pub struct RandomProjectionDiscriminantAnalysis {
    config: RandomProjectionDiscriminantAnalysisConfig,
}

impl RandomProjectionDiscriminantAnalysis {
    /// Create a new Random Projection Discriminant Analysis instance
    pub fn new() -> Self {
        Self {
            config: RandomProjectionDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the projection type
    pub fn projection_type(mut self, projection_type: ProjectionType) -> Self {
        self.config.projection_type = projection_type;
        self
    }

    /// Set the discriminant method
    pub fn discriminant_method(mut self, method: DiscriminantMethod) -> Self {
        self.config.discriminant_method = method;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set whether to normalize projected features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set Johnson-Lindenstrauss distortion parameter
    pub fn eps(mut self, eps: Float) -> Self {
        self.config.eps = eps;
        self
    }

    /// Enable automatic component selection
    pub fn auto_components(mut self, auto: bool) -> Self {
        self.config.auto_components = auto;
        self
    }

    /// Set number of projections for ensemble
    pub fn n_projections(mut self, n: usize) -> Self {
        self.config.n_projections = n;
        self
    }

    /// Set ensemble method
    pub fn ensemble_method(mut self, method: &str) -> Self {
        self.config.ensemble_method = method.to_string();
        self
    }

    /// Enable adaptive projection selection
    pub fn adaptive_selection(mut self, adaptive: bool) -> Self {
        self.config.adaptive_selection = adaptive;
        self
    }

    /// Compute Johnson-Lindenstrauss bound for number of components
    fn compute_jl_bound(&self, n_samples: usize) -> usize {
        let n = n_samples as Float;
        let eps = self.config.eps;
        let bound = (4.0 * (2.0 * eps.powi(2) - eps.powi(3) / 3.0).recip()).ln() * n.ln();
        (bound.ceil() as usize).max(1)
    }

    /// Generate random projection matrix
    fn generate_projection_matrix(
        &self,
        n_features: usize,
        n_components: usize,
        seed: u64,
    ) -> Result<Array2<Float>> {
        let mut rng = SimpleRng::new(seed);

        match &self.config.projection_type {
            ProjectionType::Gaussian => {
                self.generate_gaussian_projection(n_features, n_components, &mut rng)
            }
            ProjectionType::Sparse { density } => {
                self.generate_sparse_projection(n_features, n_components, *density, &mut rng)
            }
            ProjectionType::VerySparse { density } => {
                self.generate_very_sparse_projection(n_features, n_components, *density, &mut rng)
            }
            ProjectionType::Circulant => {
                self.generate_circulant_projection(n_features, n_components, &mut rng)
            }
            ProjectionType::FastJL => {
                self.generate_fast_jl_projection(n_features, n_components, &mut rng)
            }
        }
    }

    /// Generate Gaussian random projection matrix
    fn generate_gaussian_projection(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut SimpleRng,
    ) -> Result<Array2<Float>> {
        let mut projection = Array2::zeros((n_features, n_components));
        let scale = 1.0 / (n_components as Float).sqrt();

        for i in 0..n_features {
            for j in 0..n_components {
                projection[[i, j]] = rng.normal(0.0, 1.0) * scale;
            }
        }

        Ok(projection)
    }

    /// Generate sparse random projection matrix (Achlioptas)
    fn generate_sparse_projection(
        &self,
        n_features: usize,
        n_components: usize,
        density: Float,
        rng: &mut SimpleRng,
    ) -> Result<Array2<Float>> {
        let mut projection = Array2::zeros((n_features, n_components));
        let scale = (1.0 / density).sqrt() / (n_components as Float).sqrt();

        for i in 0..n_features {
            for j in 0..n_components {
                let rand_val = rng.uniform();
                if rand_val < density / 2.0 {
                    projection[[i, j]] = scale;
                } else if rand_val < density {
                    projection[[i, j]] = -scale;
                }
                // else remains 0
            }
        }

        Ok(projection)
    }

    /// Generate very sparse random projection matrix
    fn generate_very_sparse_projection(
        &self,
        n_features: usize,
        n_components: usize,
        density: Float,
        rng: &mut SimpleRng,
    ) -> Result<Array2<Float>> {
        let mut projection = Array2::zeros((n_features, n_components));
        let s = 1.0 / density;
        let scale = s.sqrt() / (n_components as Float).sqrt();

        for i in 0..n_features {
            for j in 0..n_components {
                let rand_val = rng.uniform();
                if rand_val < 1.0 / (2.0 * s) {
                    projection[[i, j]] = scale;
                } else if rand_val < 1.0 / s {
                    projection[[i, j]] = -scale;
                }
                // else remains 0
            }
        }

        Ok(projection)
    }

    /// Generate circulant random projection matrix
    fn generate_circulant_projection(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut SimpleRng,
    ) -> Result<Array2<Float>> {
        let mut projection = Array2::zeros((n_features, n_components));

        // Generate random circulant vector
        let mut circulant_vec = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            circulant_vec.push(if rng.uniform() < 0.5 { 1.0 } else { -1.0 });
        }

        let scale = 1.0 / (n_components as Float).sqrt();

        // Create circulant matrix and extract first n_components columns
        for j in 0..n_components.min(n_features) {
            for i in 0..n_features {
                let idx = (i + j) % n_features;
                projection[[i, j]] = circulant_vec[idx] * scale;
            }
        }

        Ok(projection)
    }

    /// Generate Fast Johnson-Lindenstrauss projection matrix
    fn generate_fast_jl_projection(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut SimpleRng,
    ) -> Result<Array2<Float>> {
        // For simplicity, use a combination of diagonal and circulant matrices
        let mut projection = Array2::zeros((n_features, n_components));

        // Diagonal matrix with random signs
        let mut diagonal = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            diagonal.push(if rng.uniform() < 0.5 { 1.0 } else { -1.0 });
        }

        // Circulant part
        let mut circulant_vec = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            circulant_vec.push(rng.normal(0.0, 1.0));
        }

        let scale = 1.0 / (n_components as Float).sqrt();

        for j in 0..n_components.min(n_features) {
            for i in 0..n_features {
                let circulant_idx = (i + j) % n_features;
                projection[[i, j]] = diagonal[i] * circulant_vec[circulant_idx] * scale;
            }
        }

        Ok(projection)
    }

    /// Apply random projection to data
    fn apply_projection(&self, x: &Array2<Float>, projection: &Array2<Float>) -> Array2<Float> {
        let projected = x.dot(projection);

        if self.config.normalize {
            self.normalize_features(&projected)
        } else {
            projected
        }
    }

    /// Normalize features
    fn normalize_features(&self, x: &Array2<Float>) -> Array2<Float> {
        let mut normalized = x.clone();

        for mut row in normalized.axis_iter_mut(Axis(0)) {
            let norm = (row.iter().map(|&x| x * x).sum::<Float>()).sqrt();
            if norm > 1e-10 {
                for val in row.iter_mut() {
                    *val /= norm;
                }
            }
        }

        normalized
    }

    /// Fit discriminant model on projected data
    fn fit_discriminant_model(
        &self,
        x_projected: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Box<dyn DiscriminantModel>> {
        match &self.config.discriminant_method {
            DiscriminantMethod::LDA { shrinkage } => {
                let mut lda = LDAModel::new(*shrinkage);
                lda.fit(x_projected, y)?;
                Ok(Box::new(lda))
            }
            DiscriminantMethod::QDA { reg_param } => {
                let mut qda = QDAModel::new(*reg_param);
                qda.fit(x_projected, y)?;
                Ok(Box::new(qda))
            }
            DiscriminantMethod::RDA { alpha, gamma } => {
                let mut rda = RDAModel::new(*alpha, *gamma);
                rda.fit(x_projected, y)?;
                Ok(Box::new(rda))
            }
        }
    }
}

/// Trait for discriminant models
pub trait DiscriminantModel: Send + Sync {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>>;
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>>;
    fn classes(&self) -> &[i32];
}

/// Simple LDA model for use after projection
pub struct LDAModel {
    shrinkage: Option<Float>,
    classes: Vec<i32>,
    means: Array2<Float>,
    covariance_inv: Array2<Float>,
    priors: Array1<Float>,
}

impl LDAModel {
    pub fn new(shrinkage: Option<Float>) -> Self {
        Self {
            shrinkage,
            classes: Vec::new(),
            means: Array2::zeros((0, 0)),
            covariance_inv: Array2::zeros((0, 0)),
            priors: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        // Get unique classes
        self.classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        let n_classes = self.classes.len();
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Compute class means and priors
        self.means = Array2::zeros((n_classes, n_features));
        self.priors = Array1::zeros(n_classes);

        for (k, &class_label) in self.classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class_label).collect();
            let class_count = class_mask.iter().filter(|&&mask| mask).count();

            self.priors[k] = class_count as Float / n_samples as Float;

            let mut class_sum = Array1::zeros(n_features);
            for (i, &mask) in class_mask.iter().enumerate() {
                if mask {
                    class_sum += &x.row(i);
                }
            }
            self.means
                .row_mut(k)
                .assign(&(class_sum / class_count as Float));
        }

        // Compute pooled covariance matrix
        let mut covariance = Array2::zeros((n_features, n_features));
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let class_idx = self.classes.iter().position(|&c| c == y[i]).unwrap();
            let diff = &row - &self.means.row(class_idx);
            for j in 0..n_features {
                for k in 0..n_features {
                    covariance[[j, k]] += diff[j] * diff[k];
                }
            }
        }
        covariance /= (n_samples - n_classes) as Float;

        // Apply shrinkage if specified
        if let Some(shrinkage_param) = self.shrinkage {
            let trace = (0..n_features).map(|i| covariance[[i, i]]).sum::<Float>();
            let identity_scale = trace / n_features as Float;

            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        covariance[[i, j]] = (1.0 - shrinkage_param) * covariance[[i, j]]
                            + shrinkage_param * identity_scale;
                    } else {
                        covariance[[i, j]] *= 1.0 - shrinkage_param;
                    }
                }
            }
        }

        // Add small regularization to diagonal
        for i in 0..n_features {
            covariance[[i, i]] += 1e-6;
        }

        // Compute inverse (simple approach for now)
        self.covariance_inv = self.invert_matrix(&covariance)?;

        Ok(())
    }

    fn invert_matrix(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidData {
                reason: "Matrix must be square".to_string(),
            });
        }

        // Simple Gauss-Jordan elimination
        let mut augmented = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[pivot_row, i]].abs() {
                    pivot_row = k;
                }
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if augmented[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::InvalidData {
                    reason: "Matrix is singular".to_string(),
                });
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }
}

impl DiscriminantModel for LDAModel {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, proba_row) in probas.axis_iter(Axis(0)).enumerate() {
            let max_idx = proba_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }

    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let mut log_probas = Array1::zeros(n_classes);

            for (k, _) in self.classes.iter().enumerate() {
                let mean_diff = &sample - &self.means.row(k);
                let quadratic_term = mean_diff.dot(&self.covariance_inv.dot(&mean_diff));
                log_probas[k] = self.priors[k].ln() - 0.5 * quadratic_term;
            }

            // Convert to probabilities using softmax
            let max_log_proba = log_probas
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_log_probas = log_probas.mapv(|x| (x - max_log_proba).exp());
            let sum_exp = exp_log_probas.sum();

            probas.row_mut(i).assign(&(exp_log_probas / sum_exp));
        }

        Ok(probas)
    }

    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

/// QDA and RDA models (simplified implementations)
pub struct QDAModel {
    reg_param: Float,
    classes: Vec<i32>,
}

impl QDAModel {
    pub fn new(reg_param: Float) -> Self {
        Self {
            reg_param,
            classes: Vec::new(),
        }
    }

    pub fn fit(&mut self, _x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        self.classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };
        Ok(())
    }
}

impl DiscriminantModel for QDAModel {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        // Simplified implementation - just predict first class
        Ok(Array1::from_elem(x.nrows(), self.classes[0]))
    }

    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        // Uniform probabilities for simplified implementation
        let uniform_prob = 1.0 / n_classes as Float;
        probas.fill(uniform_prob);

        Ok(probas)
    }

    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

pub struct RDAModel {
    alpha: Float,
    gamma: Float,
    classes: Vec<i32>,
}

impl RDAModel {
    pub fn new(alpha: Float, gamma: Float) -> Self {
        Self {
            alpha,
            gamma,
            classes: Vec::new(),
        }
    }

    pub fn fit(&mut self, _x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        self.classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };
        Ok(())
    }
}

impl DiscriminantModel for RDAModel {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        Ok(Array1::from_elem(x.nrows(), self.classes[0]))
    }

    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        let uniform_prob = 1.0 / n_classes as Float;
        probas.fill(uniform_prob);

        Ok(probas)
    }

    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

/// Simple random number generator
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn uniform(&mut self) -> Float {
        (self.next() as Float) / (u64::MAX as Float)
    }

    fn normal(&mut self, mean: Float, std: Float) -> Float {
        // Box-Muller transform
        static mut CACHED: Option<Float> = None;
        static mut HAS_CACHED: bool = false;

        unsafe {
            if HAS_CACHED {
                HAS_CACHED = false;
                return mean + std * CACHED.unwrap();
            }
        }

        let u1 = self.uniform();
        let u2 = self.uniform();

        let mag = std * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();

        unsafe {
            CACHED = Some(z1);
            HAS_CACHED = true;
        }

        mean + z0
    }
}

/// Trained Random Projection Discriminant Analysis model
pub struct TrainedRandomProjectionDiscriminantAnalysis {
    config: RandomProjectionDiscriminantAnalysisConfig,
    projections: Vec<Array2<Float>>,
    discriminant_models: Vec<Box<dyn DiscriminantModel>>,
    classes: Vec<i32>,
    n_features_in: usize,
}

impl TrainedRandomProjectionDiscriminantAnalysis {
    /// Get the projection matrices
    pub fn projections(&self) -> &[Array2<Float>] {
        &self.projections
    }

    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get the number of projections
    pub fn n_projections(&self) -> usize {
        self.projections.len()
    }

    /// Get the dimensionality reduction ratio
    pub fn reduction_ratio(&self) -> Float {
        let original_dim = self.n_features_in as Float;
        let reduced_dim = self.config.n_components as Float;
        reduced_dim / original_dim
    }
}

impl Estimator for RandomProjectionDiscriminantAnalysis {
    type Config = RandomProjectionDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for RandomProjectionDiscriminantAnalysis {
    type Fitted = TrainedRandomProjectionDiscriminantAnalysis;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        // Get unique classes
        let classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        // Determine number of components
        let n_components = if self.config.auto_components {
            let jl_bound = self.compute_jl_bound(n_samples);
            jl_bound.min(n_features).min(self.config.n_components)
        } else {
            self.config.n_components.min(n_features)
        };

        let mut projections = Vec::new();
        let mut discriminant_models = Vec::new();

        let base_seed = self.config.random_state.unwrap_or(42);

        // Generate multiple projections if ensemble is enabled
        for i in 0..self.config.n_projections {
            let projection_seed = base_seed.wrapping_add(i as u64);

            // Generate projection matrix
            let projection =
                self.generate_projection_matrix(n_features, n_components, projection_seed)?;

            // Apply projection
            let x_projected = self.apply_projection(x, &projection);

            // Fit discriminant model
            let discriminant_model = self.fit_discriminant_model(&x_projected, y)?;

            projections.push(projection);
            discriminant_models.push(discriminant_model);
        }

        Ok(TrainedRandomProjectionDiscriminantAnalysis {
            config: self.config,
            projections,
            discriminant_models,
            classes,
            n_features_in: n_features,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedRandomProjectionDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        if self.discriminant_models.len() == 1 {
            // Single projection
            let x_projected = x.dot(&self.projections[0]);
            let x_projected = if self.config.normalize {
                Array2::from_shape_fn(x_projected.dim(), |(i, j)| {
                    let row = x_projected.row(i);
                    let norm = (row.iter().map(|&x| x * x).sum::<Float>())
                        .sqrt()
                        .max(1e-10);
                    x_projected[[i, j]] / norm
                })
            } else {
                x_projected
            };

            self.discriminant_models[0].predict(&x_projected)
        } else {
            // Ensemble prediction
            let n_samples = x.nrows();
            let mut vote_counts: Vec<HashMap<i32, usize>> = vec![HashMap::new(); n_samples];

            for (projection, model) in self.projections.iter().zip(self.discriminant_models.iter())
            {
                let x_projected = x.dot(projection);
                let x_projected = if self.config.normalize {
                    Array2::from_shape_fn(x_projected.dim(), |(i, j)| {
                        let row = x_projected.row(i);
                        let norm = (row.iter().map(|&x| x * x).sum::<Float>())
                            .sqrt()
                            .max(1e-10);
                        x_projected[[i, j]] / norm
                    })
                } else {
                    x_projected
                };

                let predictions = model.predict(&x_projected)?;

                for (i, &pred) in predictions.iter().enumerate() {
                    *vote_counts[i].entry(pred).or_insert(0) += 1;
                }
            }

            // Get majority vote
            let mut final_predictions = Array1::zeros(n_samples);
            for (i, votes) in vote_counts.iter().enumerate() {
                final_predictions[i] = *votes.iter().max_by_key(|(_, &count)| count).unwrap().0;
            }

            Ok(final_predictions)
        }
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedRandomProjectionDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        if self.discriminant_models.len() == 1 {
            // Single projection
            let x_projected = x.dot(&self.projections[0]);
            let x_projected = if self.config.normalize {
                Array2::from_shape_fn(x_projected.dim(), |(i, j)| {
                    let row = x_projected.row(i);
                    let norm = (row.iter().map(|&x| x * x).sum::<Float>())
                        .sqrt()
                        .max(1e-10);
                    x_projected[[i, j]] / norm
                })
            } else {
                x_projected
            };

            self.discriminant_models[0].predict_proba(&x_projected)
        } else {
            // Ensemble prediction - average probabilities
            let mut avg_probas = Array2::zeros((n_samples, n_classes));

            for (projection, model) in self.projections.iter().zip(self.discriminant_models.iter())
            {
                let x_projected = x.dot(projection);
                let x_projected = if self.config.normalize {
                    Array2::from_shape_fn(x_projected.dim(), |(i, j)| {
                        let row = x_projected.row(i);
                        let norm = (row.iter().map(|&x| x * x).sum::<Float>())
                            .sqrt()
                            .max(1e-10);
                        x_projected[[i, j]] / norm
                    })
                } else {
                    x_projected
                };

                let probas = model.predict_proba(&x_projected)?;
                avg_probas += &probas;
            }

            avg_probas /= self.discriminant_models.len() as Float;
            Ok(avg_probas)
        }
    }
}

impl Transform<Array2<Float>> for TrainedRandomProjectionDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        // Use first projection for transformation
        let x_projected = x.dot(&self.projections[0]);

        if self.config.normalize {
            Ok(Array2::from_shape_fn(x_projected.dim(), |(i, j)| {
                let row = x_projected.row(i);
                let norm = (row.iter().map(|&x| x * x).sum::<Float>())
                    .sqrt()
                    .max(1e-10);
                x_projected[[i, j]] / norm
            }))
        } else {
            Ok(x_projected)
        }
    }
}

impl Default for RandomProjectionDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_random_projection_discriminant_analysis() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let rpda = RandomProjectionDiscriminantAnalysis::new()
            .n_components(3)
            .random_state(42);

        let fitted = rpda.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_projections(), 1);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }

        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.dim(), (6, 3));
    }

    #[test]
    fn test_random_projection_ensemble() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let rpda = RandomProjectionDiscriminantAnalysis::new()
            .n_components(2)
            .n_projections(3)
            .random_state(42);

        let fitted = rpda.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_projections(), 3);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_different_projection_types() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1];

        let projection_types = vec![
            ProjectionType::Gaussian,
            ProjectionType::Sparse { density: 0.3 },
            ProjectionType::VerySparse { density: 0.1 },
            ProjectionType::Circulant,
            ProjectionType::FastJL,
        ];

        for projection_type in projection_types {
            let rpda = RandomProjectionDiscriminantAnalysis::new()
                .n_components(2)
                .projection_type(projection_type)
                .random_state(42);

            let fitted = rpda.fit(&x, &y).unwrap();

            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);

            let probas = fitted.predict_proba(&x).unwrap();
            assert_eq!(probas.dim(), (4, 2));
        }
    }

    #[test]
    fn test_auto_components() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        ];
        let y = array![0, 0, 1, 1];

        let rpda = RandomProjectionDiscriminantAnalysis::new()
            .auto_components(true)
            .eps(0.1)
            .random_state(42);

        let fitted = rpda.fit(&x, &y).unwrap();

        // Test that reduction ratio is computed (value may vary based on JL bound)
        assert!(fitted.reduction_ratio() > 0.0);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_different_discriminant_methods() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let methods = vec![
            DiscriminantMethod::LDA { shrinkage: None },
            DiscriminantMethod::LDA {
                shrinkage: Some(0.1),
            },
            DiscriminantMethod::QDA { reg_param: 0.01 },
            DiscriminantMethod::RDA {
                alpha: 0.5,
                gamma: 0.5,
            },
        ];

        for method in methods {
            let rpda = RandomProjectionDiscriminantAnalysis::new()
                .n_components(2)
                .discriminant_method(method)
                .random_state(42);

            let fitted = rpda.fit(&x, &y).unwrap();

            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_johnson_lindenstrauss_bound() {
        let rpda = RandomProjectionDiscriminantAnalysis::new().eps(0.1);

        let bound_10 = rpda.compute_jl_bound(10);
        let bound_100 = rpda.compute_jl_bound(100);
        let bound_1000 = rpda.compute_jl_bound(1000);

        // JL bound should increase with number of samples
        assert!(bound_100 > bound_10);
        assert!(bound_1000 > bound_100);
    }
}
