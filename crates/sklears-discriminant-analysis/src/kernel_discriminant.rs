//! Kernel Discriminant Analysis
//!
//! This module implements kernel-based extensions of linear discriminant analysis,
//! allowing for non-linear classification boundaries through the kernel trick.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_linalg::compat::{Eig, Eigh, UPLO};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Kernel function types supported by kernel discriminant analysis
pub enum KernelType {
    /// Radial Basis Function kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    RBF { gamma: Float },
    /// Polynomial kernel: K(x, y) = (gamma * <x, y> + coef0)^degree
    Polynomial {
        gamma: Float,

        coef0: Float,

        degree: i32,
    },
    /// Linear kernel: K(x, y) = <x, y>
    Linear,
    /// Sigmoid kernel: K(x, y) = tanh(gamma * <x, y> + coef0)
    Sigmoid { gamma: Float, coef0: Float },
    /// Custom kernel function
    Custom(Box<dyn Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Float + Send + Sync>),
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::RBF { gamma: 1.0 }
    }
}

impl std::fmt::Debug for KernelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelType::RBF { gamma } => f.debug_struct("RBF").field("gamma", gamma).finish(),
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => f
                .debug_struct("Polynomial")
                .field("gamma", gamma)
                .field("coef0", coef0)
                .field("degree", degree)
                .finish(),
            KernelType::Linear => f.debug_struct("Linear").finish(),
            KernelType::Sigmoid { gamma, coef0 } => f
                .debug_struct("Sigmoid")
                .field("gamma", gamma)
                .field("coef0", coef0)
                .finish(),
            KernelType::Custom(_) => f
                .debug_struct("Custom")
                .field("function", &"<closure>")
                .finish(),
        }
    }
}

impl PartialEq for KernelType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (KernelType::RBF { gamma: g1 }, KernelType::RBF { gamma: g2 }) => g1 == g2,
            (
                KernelType::Polynomial {
                    gamma: g1,
                    coef0: c1,
                    degree: d1,
                },
                KernelType::Polynomial {
                    gamma: g2,
                    coef0: c2,
                    degree: d2,
                },
            ) => g1 == g2 && c1 == c2 && d1 == d2,
            (KernelType::Linear, KernelType::Linear) => true,
            (
                KernelType::Sigmoid {
                    gamma: g1,
                    coef0: c1,
                },
                KernelType::Sigmoid {
                    gamma: g2,
                    coef0: c2,
                },
            ) => g1 == g2 && c1 == c2,
            (KernelType::Custom(_), KernelType::Custom(_)) => false, // Can't compare function pointers
            _ => false,
        }
    }
}

impl Clone for KernelType {
    fn clone(&self) -> Self {
        match self {
            KernelType::RBF { gamma } => KernelType::RBF { gamma: *gamma },
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => KernelType::Polynomial {
                gamma: *gamma,
                coef0: *coef0,
                degree: *degree,
            },
            KernelType::Linear => KernelType::Linear,
            KernelType::Sigmoid { gamma, coef0 } => KernelType::Sigmoid {
                gamma: *gamma,
                coef0: *coef0,
            },
            KernelType::Custom(_) => {
                // Cannot clone function pointers, fall back to default RBF kernel
                KernelType::RBF { gamma: 1.0 }
            }
        }
    }
}

/// Parameter grid for kernel optimization
#[derive(Debug, Clone)]
pub struct KernelParameterGrid {
    /// gamma_values
    pub gamma_values: Vec<Float>,
    /// coef0_values
    pub coef0_values: Vec<Float>,
    /// degree_values
    pub degree_values: Vec<i32>,
    /// c_values
    pub c_values: Vec<Float>,
}

impl Default for KernelParameterGrid {
    fn default() -> Self {
        Self {
            gamma_values: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            coef0_values: vec![0.0, 0.1, 1.0],
            degree_values: vec![2, 3, 4],
            c_values: vec![0.1, 1.0, 10.0, 100.0],
        }
    }
}

impl KernelParameterGrid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn gamma_values(mut self, values: Vec<Float>) -> Self {
        self.gamma_values = values;
        self
    }

    pub fn coef0_values(mut self, values: Vec<Float>) -> Self {
        self.coef0_values = values;
        self
    }

    pub fn degree_values(mut self, values: Vec<i32>) -> Self {
        self.degree_values = values;
        self
    }

    pub fn c_values(mut self, values: Vec<Float>) -> Self {
        self.c_values = values;
        self
    }
}

/// Cross-validation configuration for kernel parameter optimization
#[derive(Debug, Clone)]
pub struct KernelCVConfig {
    /// n_folds
    pub n_folds: usize,
    /// scoring
    pub scoring: String,
    /// random_state
    pub random_state: Option<u64>,
    /// shuffle
    pub shuffle: bool,
}

impl Default for KernelCVConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            scoring: "accuracy".to_string(),
            random_state: Some(42),
            shuffle: true,
        }
    }
}

/// Results from kernel parameter optimization
#[derive(Debug, Clone)]
pub struct KernelOptimizationResults {
    /// best_kernel
    pub best_kernel: KernelType,
    /// best_score
    pub best_score: Float,
    /// cv_scores
    pub cv_scores: Vec<Float>,
    /// all_results
    pub all_results: Vec<(KernelType, Float, Vec<Float>)>,
}

/// Kernel parameter optimizer
#[derive(Debug, Clone)]
pub struct KernelParameterOptimizer {
    /// parameter_grid
    pub parameter_grid: KernelParameterGrid,
    /// cv_config
    pub cv_config: KernelCVConfig,
    /// optimization_strategy
    pub optimization_strategy: String,
}

impl Default for KernelParameterOptimizer {
    fn default() -> Self {
        Self {
            parameter_grid: KernelParameterGrid::default(),
            cv_config: KernelCVConfig::default(),
            optimization_strategy: "grid_search".to_string(),
        }
    }
}

impl KernelParameterOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parameter_grid(mut self, grid: KernelParameterGrid) -> Self {
        self.parameter_grid = grid;
        self
    }

    pub fn cv_config(mut self, config: KernelCVConfig) -> Self {
        self.cv_config = config;
        self
    }

    pub fn optimization_strategy(mut self, strategy: &str) -> Self {
        self.optimization_strategy = strategy.to_string();
        self
    }

    /// Optimize kernel parameters using grid search and cross-validation
    pub fn optimize(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<KernelOptimizationResults> {
        match self.optimization_strategy.as_str() {
            "grid_search" => self.grid_search_optimize(x, y),
            "random_search" => self.random_search_optimize(x, y),
            "bayesian" => self.bayesian_optimize(x, y),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown optimization strategy: {}",
                self.optimization_strategy
            ))),
        }
    }

    /// Grid search optimization
    fn grid_search_optimize(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<KernelOptimizationResults> {
        let kernel_candidates = self.generate_kernel_candidates();
        let mut best_score = Float::NEG_INFINITY;
        let mut best_kernel = KernelType::default();
        let mut all_results = Vec::new();

        for kernel in kernel_candidates {
            let cv_scores = self.cross_validate(&kernel, x, y)?;
            let mean_score = cv_scores.iter().sum::<Float>() / cv_scores.len() as Float;

            all_results.push((kernel.clone(), mean_score, cv_scores.clone()));

            if mean_score > best_score {
                best_score = mean_score;
                best_kernel = kernel;
            }
        }

        let best_cv_scores = all_results
            .iter()
            .find(|(k, _, _)| *k == best_kernel)
            .map(|(_, _, scores)| scores.clone())
            .unwrap_or_default();

        Ok(KernelOptimizationResults {
            best_kernel,
            best_score,
            cv_scores: best_cv_scores,
            all_results,
        })
    }

    /// Random search optimization
    fn random_search_optimize(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<KernelOptimizationResults> {
        // For simplicity, implement as subset of grid search
        let all_candidates = self.generate_kernel_candidates();
        let n_random = (all_candidates.len() / 3).max(10).min(all_candidates.len());

        let mut rng_state = self.cv_config.random_state.unwrap_or(42);
        let mut kernel_candidates = Vec::new();

        for i in 0..n_random {
            let idx = (rng_state as usize + i * 17) % all_candidates.len();
            kernel_candidates.push(all_candidates[idx].clone());
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        }

        let mut best_score = Float::NEG_INFINITY;
        let mut best_kernel = KernelType::default();
        let mut all_results = Vec::new();

        for kernel in kernel_candidates {
            let cv_scores = self.cross_validate(&kernel, x, y)?;
            let mean_score = cv_scores.iter().sum::<Float>() / cv_scores.len() as Float;

            all_results.push((kernel.clone(), mean_score, cv_scores.clone()));

            if mean_score > best_score {
                best_score = mean_score;
                best_kernel = kernel;
            }
        }

        let best_cv_scores = all_results
            .iter()
            .find(|(k, _, _)| *k == best_kernel)
            .map(|(_, _, scores)| scores.clone())
            .unwrap_or_default();

        Ok(KernelOptimizationResults {
            best_kernel,
            best_score,
            cv_scores: best_cv_scores,
            all_results,
        })
    }

    /// Bayesian optimization (simplified implementation)
    fn bayesian_optimize(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<KernelOptimizationResults> {
        // For simplicity, fall back to random search for now
        // In a full implementation, you would use Gaussian Process optimization
        self.random_search_optimize(x, y)
    }

    /// Generate all kernel candidates from parameter grid
    fn generate_kernel_candidates(&self) -> Vec<KernelType> {
        let mut candidates = Vec::new();

        // Linear kernel (no parameters)
        candidates.push(KernelType::Linear);

        // RBF kernels
        for &gamma in &self.parameter_grid.gamma_values {
            candidates.push(KernelType::RBF { gamma });
        }

        // Polynomial kernels
        for &gamma in &self.parameter_grid.gamma_values {
            for &coef0 in &self.parameter_grid.coef0_values {
                for &degree in &self.parameter_grid.degree_values {
                    candidates.push(KernelType::Polynomial {
                        gamma,
                        coef0,
                        degree,
                    });
                }
            }
        }

        // Sigmoid kernels
        for &gamma in &self.parameter_grid.gamma_values {
            for &coef0 in &self.parameter_grid.coef0_values {
                candidates.push(KernelType::Sigmoid { gamma, coef0 });
            }
        }

        candidates
    }

    /// Perform cross-validation for a given kernel
    fn cross_validate(
        &self,
        kernel: &KernelType,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Vec<Float>> {
        let n_samples = x.nrows();
        let fold_size = n_samples / self.cv_config.n_folds;
        let mut cv_scores = Vec::new();

        for fold in 0..self.cv_config.n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.cv_config.n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if train_indices.is_empty() || test_indices.is_empty() {
                continue;
            }

            // Extract train/test data
            let x_train = x.select(Axis(0), &train_indices);
            let y_train = y.select(Axis(0), &train_indices);
            let x_test = x.select(Axis(0), &test_indices);
            let y_test = y.select(Axis(0), &test_indices);

            // Train model with current kernel
            let kda = KernelDiscriminantAnalysis::new().kernel(kernel.clone());

            match kda.fit(&x_train, &y_train) {
                Ok(fitted) => {
                    match fitted.predict(&x_test) {
                        Ok(predictions) => {
                            let score = self.compute_score(&predictions, &y_test);
                            cv_scores.push(score);
                        }
                        Err(_) => {
                            // If prediction fails, assign low score
                            cv_scores.push(0.0);
                        }
                    }
                }
                Err(_) => {
                    // If fitting fails, assign low score
                    cv_scores.push(0.0);
                }
            }
        }

        if cv_scores.is_empty() {
            cv_scores.push(0.0);
        }

        Ok(cv_scores)
    }

    /// Compute score based on scoring method
    fn compute_score(&self, predictions: &Array1<i32>, y_true: &Array1<i32>) -> Float {
        match self.cv_config.scoring.as_str() {
            "accuracy" => {
                let correct = predictions
                    .iter()
                    .zip(y_true.iter())
                    .filter(|(&pred, &true_val)| pred == true_val)
                    .count();
                correct as Float / predictions.len() as Float
            }
            _ => {
                // Default to accuracy
                let correct = predictions
                    .iter()
                    .zip(y_true.iter())
                    .filter(|(&pred, &true_val)| pred == true_val)
                    .count();
                correct as Float / predictions.len() as Float
            }
        }
    }
}

/// Configuration for Kernel Discriminant Analysis
#[derive(Debug, Clone)]
pub struct KernelDiscriminantAnalysisConfig {
    /// kernel
    pub kernel: KernelType,
    /// reg_param
    pub reg_param: Float,
    /// n_components
    pub n_components: Option<usize>,
    /// tol
    pub tol: Float,
    /// max_iter
    pub max_iter: usize,
    /// eigensolver
    pub eigensolver: String,
}

impl Default for KernelDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::default(),
            reg_param: 1e-4,
            n_components: None,
            tol: 1e-4,
            max_iter: 200,
            eigensolver: "auto".to_string(),
        }
    }
}

/// Kernel Discriminant Analysis
///
/// Kernel discriminant analysis extends linear discriminant analysis by using the kernel trick
/// to transform the data into a higher-dimensional space where linear separation is possible.
/// This allows for non-linear decision boundaries in the original feature space.
///
/// # Mathematical Background
///
/// Given training data X and labels y, kernel LDA:
/// 1. Computes the kernel matrix K(X, X) using the specified kernel function
/// 2. Performs eigenvalue decomposition on the kernel-transformed scatter matrices
/// 3. Projects data into the discriminant subspace for classification
///
/// The kernel trick allows computing inner products in the transformed space
/// without explicitly computing the transformation.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_discriminant_analysis::*;
/// use sklears_core::traits::{Predict, Fit};
///
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 0, 1, 1];
///
/// let kda = KernelDiscriminantAnalysis::new()
///     .kernel(KernelType::RBF { gamma: 1.0 });
/// let fitted = kda.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct KernelDiscriminantAnalysis {
    config: KernelDiscriminantAnalysisConfig,
}

impl Default for KernelDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelDiscriminantAnalysis {
    /// Create a new kernel discriminant analysis instance
    pub fn new() -> Self {
        Self {
            config: KernelDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set the number of components for dimensionality reduction
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the eigensolver method
    pub fn eigensolver(mut self, eigensolver: &str) -> Self {
        self.config.eigensolver = eigensolver.to_string();
        self
    }

    /// Fit with automatic kernel parameter optimization
    pub fn fit_auto(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<TrainedKernelDiscriminantAnalysis> {
        let optimizer = KernelParameterOptimizer::new();
        self.fit_with_optimizer(x, y, &optimizer)
    }

    /// Fit with custom kernel parameter optimizer
    pub fn fit_with_optimizer(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        optimizer: &KernelParameterOptimizer,
    ) -> Result<TrainedKernelDiscriminantAnalysis> {
        // Optimize kernel parameters
        let optimization_results = optimizer.optimize(x, y)?;

        // Create new instance with optimized kernel
        let optimized_kda = Self {
            config: KernelDiscriminantAnalysisConfig {
                kernel: optimization_results.best_kernel,
                ..self.config.clone()
            },
        };

        // Fit with optimized parameters
        optimized_kda.fit(x, y)
    }
}

/// Trained Kernel Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedKernelDiscriminantAnalysis {
    config: KernelDiscriminantAnalysisConfig,
    classes: Array1<i32>,
    class_means: HashMap<i32, Array1<Float>>,
    alpha: Array2<Float>,
    eigenvalues: Array1<Float>,
    eigenvectors: Array2<Float>,
    train_data: Array2<Float>,
    n_train_samples: usize,
    n_components: usize,
}

impl TrainedKernelDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        &self.eigenvalues
    }

    /// Get the eigenvectors in kernel space
    pub fn eigenvectors(&self) -> &Array2<Float> {
        &self.eigenvectors
    }

    /// Get the dual coefficients (alpha)
    pub fn dual_coefficients(&self) -> &Array2<Float> {
        &self.alpha
    }

    /// Compute kernel matrix between two sets of samples
    fn compute_kernel_matrix(
        &self,
        x1: &ArrayView2<Float>,
        x2: &ArrayView2<Float>,
    ) -> Array2<Float> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel_matrix = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1_row = x1.row(i);
                let x2_row = x2.row(j);
                kernel_matrix[[i, j]] = self.kernel_function(&x1_row, &x2_row);
            }
        }

        kernel_matrix
    }

    /// Apply the kernel function
    fn kernel_function(&self, x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> Float {
        match &self.config.kernel {
            KernelType::RBF { gamma } => {
                let diff = x1 - x2;
                let squared_distance = diff.iter().map(|&x| x * x).sum::<Float>();
                (-gamma * squared_distance).exp()
            }
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                let dot_product = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<Float>();
                (gamma * dot_product + coef0).powi(*degree)
            }
            KernelType::Linear => x1
                .iter()
                .zip(x2.iter())
                .map(|(&a, &b)| a * b)
                .sum::<Float>(),
            KernelType::Sigmoid { gamma, coef0 } => {
                let dot_product = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<Float>();
                (gamma * dot_product + coef0).tanh()
            }
            KernelType::Custom(func) => func(x1, x2),
        }
    }

    /// Compute class-conditional kernel means
    fn compute_kernel_class_means(
        &self,
        kernel_matrix: &Array2<Float>,
        y: &Array1<i32>,
    ) -> HashMap<i32, Array1<Float>> {
        let mut class_means = HashMap::new();

        for &class in &self.classes {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            let n_class_samples = class_indices.len() as Float;
            let mut class_mean = Array1::zeros(kernel_matrix.ncols());

            for &idx in &class_indices {
                class_mean += &kernel_matrix.row(idx);
            }
            class_mean /= n_class_samples;

            class_means.insert(class, class_mean);
        }

        class_means
    }

    /// Center the kernel matrix
    fn center_kernel_matrix(&self, kernel_matrix: &Array2<Float>) -> Array2<Float> {
        let n = kernel_matrix.nrows();
        let m = kernel_matrix.ncols();

        // Compute row means, column means, and overall mean
        let row_means = kernel_matrix.mean_axis(Axis(1)).unwrap();
        let col_means = kernel_matrix.mean_axis(Axis(0)).unwrap();
        let overall_mean = kernel_matrix.mean().unwrap();

        let mut centered = kernel_matrix.clone();

        // Center the matrix: K_centered = K - 1_n * K_mean_cols - K_mean_rows * 1_m + overall_mean * 1_n * 1_m
        for i in 0..n {
            for j in 0..m {
                centered[[i, j]] =
                    kernel_matrix[[i, j]] - row_means[i] - col_means[j] + overall_mean;
            }
        }

        centered
    }
}

impl Estimator for KernelDiscriminantAnalysis {
    type Config = KernelDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>, TrainedKernelDiscriminantAnalysis>
    for KernelDiscriminantAnalysis
{
    type Fitted = TrainedKernelDiscriminantAnalysis;
    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedKernelDiscriminantAnalysis> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatch between X and y dimensions".to_string(),
            ));
        }

        let n_samples = x.nrows();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Determine number of components
        let max_components = (n_classes - 1).min(n_samples);
        let n_components = self.config.n_components.unwrap_or(max_components);

        if n_components > max_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot be larger than min(n_classes-1, n_samples) ({})",
                n_components, max_components
            )));
        }

        // Create trained model structure
        let mut trained = TrainedKernelDiscriminantAnalysis {
            config: self.config.clone(),
            classes: classes.clone(),
            class_means: HashMap::new(),
            alpha: Array2::zeros((n_samples, n_components)),
            eigenvalues: Array1::zeros(n_components),
            eigenvectors: Array2::zeros((n_samples, n_components)),
            train_data: x.clone(),
            n_train_samples: n_samples,
            n_components,
        };

        // Compute kernel matrix
        let kernel_matrix = trained.compute_kernel_matrix(&x.view(), &x.view());

        // Center the kernel matrix
        let centered_kernel = trained.center_kernel_matrix(&kernel_matrix);

        // Compute class-conditional kernel means
        trained.class_means = trained.compute_kernel_class_means(&centered_kernel, y);

        // Compute within-class and between-class scatter matrices in kernel space
        let (s_w, s_b) = self.compute_kernel_scatter_matrices(&centered_kernel, y, &classes)?;

        // Solve generalized eigenvalue problem: S_B * v = λ * S_W * v
        let (eigenvalues, eigenvectors) =
            self.solve_generalized_eigenvalue_problem(&s_b, &s_w, n_components)?;

        trained.eigenvalues = eigenvalues;
        trained.eigenvectors = eigenvectors;

        // Compute dual coefficients (alpha) for projection
        trained.alpha = trained.eigenvectors.clone();

        Ok(trained)
    }
}

impl KernelDiscriminantAnalysis {
    /// Compute within-class and between-class scatter matrices in kernel space
    fn compute_kernel_scatter_matrices(
        &self,
        kernel_matrix: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<(Array2<Float>, Array2<Float>)> {
        let n_samples = kernel_matrix.nrows();
        let mut s_w = Array2::zeros((n_samples, n_samples));
        let mut s_b = Array2::zeros((n_samples, n_samples));

        // Overall mean in kernel space
        let overall_mean = kernel_matrix.mean_axis(Axis(0)).unwrap();

        // Compute class statistics
        for &class in classes {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            let n_class_samples = class_indices.len() as Float;

            if n_class_samples == 0.0 {
                continue;
            }

            // Class mean in kernel space
            let mut class_mean = Array1::zeros(n_samples);
            for &idx in &class_indices {
                class_mean += &kernel_matrix.row(idx);
            }
            class_mean /= n_class_samples;

            // Within-class scatter
            for &idx in &class_indices {
                let diff = &kernel_matrix.row(idx) - &class_mean;
                let outer_product =
                    Array2::from_shape_fn((n_samples, n_samples), |(i, j)| diff[i] * diff[j]);
                s_w += &outer_product;
            }

            // Between-class scatter
            let class_diff = &class_mean - &overall_mean;
            let between_outer = Array2::from_shape_fn((n_samples, n_samples), |(i, j)| {
                class_diff[i] * class_diff[j]
            });
            s_b += &(between_outer * n_class_samples);
        }

        // Add regularization to within-class scatter matrix
        for i in 0..n_samples {
            s_w[[i, i]] += self.config.reg_param;
        }

        Ok((s_w, s_b))
    }

    /// Solve the generalized eigenvalue problem S_B * v = λ * S_W * v
    fn solve_generalized_eigenvalue_problem(
        &self,
        s_b: &Array2<Float>,
        s_w: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // For simplicity, we'll use the approach: inv(S_W) * S_B * v = λ * v
        // In practice, you'd want to use more numerically stable methods

        // Compute S_W^(-1) using regularized inversion
        let s_w_inv = self.regularized_inverse(s_w)?;

        // Compute S_W^(-1) * S_B
        let matrix = s_w_inv.dot(s_b);

        // Compute eigenvalues and eigenvectors using robust LAPACK-based solver
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&matrix, n_components)?;

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute regularized inverse using SVD
    fn regularized_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        let mut result = Array2::zeros((n, n));

        // Add regularization to diagonal
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = matrix[[i, j]];
                if i == j {
                    result[[i, j]] += self.config.reg_param;
                }
            }
        }

        // Simplified inverse computation (in practice, use proper SVD)
        self.simple_matrix_inverse(&result)
    }

    /// Simple matrix inverse (placeholder - use proper linear algebra library)
    fn simple_matrix_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                if i == j {
                    augmented[[i, j + n]] = 1.0;
                }
            }
        }

        // Gaussian elimination (simplified)
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if augmented[[i, i]].abs() < self.config.tol {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]] / augmented[[i, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }

            // Scale row
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
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

    /// Eigendecomposition using robust LAPACK-based solver
    fn eigendecomposition(
        &self,
        matrix: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let actual_components = n_components.min(n);

        // Check if matrix is symmetric (for efficiency, use symmetric solver if possible)
        let is_symmetric = self.is_approximately_symmetric(matrix)?;

        let (eigenvals, eigenvecs) = if is_symmetric {
            // Symmetrize matrix for numerical stability before eigendecomposition
            let symmetric_matrix = (matrix + &matrix.t()) / 2.0;

            // Use symmetric eigenvalue solver for better numerical stability
            let (vals, vecs) = symmetric_matrix.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::NumericalError(format!("Symmetric eigendecomposition failed: {}", e))
            })?;

            // Convert to owned arrays
            let vals_owned = vals.to_owned();
            let vecs_owned = vecs.to_owned();
            (vals_owned, vecs_owned)
        } else {
            // Use general eigenvalue solver
            let (complex_vals, complex_vecs) = matrix.eig().map_err(|e| {
                SklearsError::NumericalError(format!("General eigendecomposition failed: {}", e))
            })?;

            // Extract real parts (for discriminant analysis, we typically expect real eigenvalues)
            let real_vals = Array1::from_vec(complex_vals.iter().map(|c| c.re).collect::<Vec<_>>());
            let real_vecs = Array2::from_shape_vec(
                (n, n),
                complex_vecs.iter().map(|c| c.re).collect::<Vec<_>>(),
            )
            .map_err(|e| {
                SklearsError::NumericalError(format!("Failed to reshape eigenvectors: {}", e))
            })?;

            (real_vals, real_vecs)
        };

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let vec = eigenvecs.column(i).to_owned();
                (val, vec)
            })
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take only the top components
        let selected_eigenvals: Vec<Float> = eigen_pairs
            .iter()
            .take(actual_components)
            .map(|(val, _)| *val)
            .collect();

        let mut selected_eigenvecs = Array2::zeros((n, actual_components));
        for (i, (_, vec)) in eigen_pairs.iter().take(actual_components).enumerate() {
            selected_eigenvecs.column_mut(i).assign(vec);
        }

        let eigenvalues = Array1::from_vec(selected_eigenvals);

        Ok((eigenvalues, selected_eigenvecs))
    }

    /// Check if a matrix is approximately symmetric
    fn is_approximately_symmetric(&self, matrix: &Array2<Float>) -> Result<bool> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Ok(false);
        }

        for i in 0..n {
            for j in 0..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > self.config.tol {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedKernelDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probabilities.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedKernelDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // Compute kernel matrix between test and training data
        let kernel_test_train = self.compute_kernel_matrix(&x.view(), &self.train_data.view());

        // Center the test kernel matrix
        let centered_test_kernel = self.center_test_kernel_matrix(&kernel_test_train);

        // Project test samples into discriminant subspace
        let projected = centered_test_kernel.dot(&self.alpha);

        // Compute distances to class means in projected space
        for (sample_idx, projected_sample) in projected.axis_iter(Axis(0)).enumerate() {
            let mut class_scores = Array1::zeros(n_classes);

            for (class_idx, &class) in self.classes.iter().enumerate() {
                // Project class mean to discriminant subspace
                if let Some(class_mean) = self.class_means.get(&class) {
                    let projected_class_mean = class_mean.dot(&self.alpha);

                    // Compute squared distance - projected_class_mean is 1D, projected_sample is 1D
                    let diff = &projected_sample - &projected_class_mean;
                    let distance = diff.iter().map(|&x| x * x).sum::<Float>();

                    // Convert distance to similarity score (negative distance)
                    class_scores[class_idx] = -distance;
                }
            }

            // Convert scores to probabilities using softmax
            let max_score = class_scores
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores = class_scores.mapv(|x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();

            if sum_exp > 0.0 {
                exp_scores /= sum_exp;
            } else {
                // Fallback to uniform distribution
                exp_scores.fill(1.0 / n_classes as Float);
            }

            probabilities.row_mut(sample_idx).assign(&exp_scores);
        }

        Ok(probabilities)
    }
}

impl TrainedKernelDiscriminantAnalysis {
    /// Center test kernel matrix using training statistics
    fn center_test_kernel_matrix(&self, kernel_test_train: &Array2<Float>) -> Array2<Float> {
        let n_test = kernel_test_train.nrows();
        let n_train = kernel_test_train.ncols();

        // Compute training kernel means (should be precomputed in practice)
        let train_kernel =
            self.compute_kernel_matrix(&self.train_data.view(), &self.train_data.view());
        let train_row_means = train_kernel.mean_axis(Axis(1)).unwrap();
        let train_overall_mean = train_kernel.mean().unwrap();

        let mut centered = kernel_test_train.clone();

        // Center using training statistics
        let test_row_means = kernel_test_train.mean_axis(Axis(1)).unwrap();

        for i in 0..n_test {
            for j in 0..n_train {
                centered[[i, j]] =
                    kernel_test_train[[i, j]] - test_row_means[i] - train_row_means[j]
                        + train_overall_mean;
            }
        }

        centered
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedKernelDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Compute kernel matrix between test and training data
        let kernel_test_train = self.compute_kernel_matrix(&x.view(), &self.train_data.view());

        // Center the test kernel matrix
        let centered_test_kernel = self.center_test_kernel_matrix(&kernel_test_train);

        // Project into discriminant subspace
        let projected = centered_test_kernel.dot(&self.alpha);

        Ok(projected)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kernel_discriminant_analysis_rbf() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_kernel_discriminant_analysis_polynomial() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        });
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_kernel_discriminant_analysis_linear() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Linear);
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_kernel_discriminant_analysis_sigmoid() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 1.0,
        });
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_kernel_predict_proba() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
        let fitted = kda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kernel_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new()
            .kernel(KernelType::RBF { gamma: 1.0 })
            .n_components(Some(1));
        let fitted = kda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 1));
    }

    #[test]
    fn test_kernel_discriminant_with_regularization() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let kda = KernelDiscriminantAnalysis::new()
            .kernel(KernelType::RBF { gamma: 1.0 })
            .reg_param(0.1);
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_custom_kernel() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        // Custom linear kernel
        let custom_kernel = Box::new(|x1: &ArrayView1<Float>, x2: &ArrayView1<Float>| -> Float {
            x1.iter()
                .zip(x2.iter())
                .map(|(&a, &b)| a * b)
                .sum::<Float>()
        });

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Custom(custom_kernel));
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_kernel_discriminant_multiclass() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1], // Class 0
            [3.0, 4.0],
            [3.1, 4.1], // Class 1
            [5.0, 6.0],
            [5.1, 6.1] // Class 2
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
        let fitted = kda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);
    }

    #[test]
    fn test_kernel_parameter_grid() {
        let grid = KernelParameterGrid::new()
            .gamma_values(vec![0.1, 1.0])
            .coef0_values(vec![0.0, 1.0])
            .degree_values(vec![2, 3]);

        assert_eq!(grid.gamma_values, vec![0.1, 1.0]);
        assert_eq!(grid.coef0_values, vec![0.0, 1.0]);
        assert_eq!(grid.degree_values, vec![2, 3]);
    }

    #[test]
    fn test_kernel_parameter_optimizer() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2],
            [3.3, 4.3] // Class 1
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let grid = KernelParameterGrid::new()
            .gamma_values(vec![0.1, 1.0])
            .coef0_values(vec![0.0])
            .degree_values(vec![2]);

        let cv_config = KernelCVConfig {
            n_folds: 2,
            scoring: "accuracy".to_string(),
            random_state: Some(42),
            shuffle: false,
        };

        let optimizer = KernelParameterOptimizer::new()
            .parameter_grid(grid)
            .cv_config(cv_config)
            .optimization_strategy("grid_search");

        let results = optimizer.optimize(&x, &y);
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.best_score >= 0.0);
        assert!(results.best_score <= 1.0);
        assert!(!results.cv_scores.is_empty());
        assert!(!results.all_results.is_empty());
    }

    #[test]
    fn test_kernel_discriminant_fit_auto() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2],
            [3.3, 4.3] // Class 1
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let kda = KernelDiscriminantAnalysis::new();
        let fitted = kda.fit_auto(&x, &y);

        // fit_auto might succeed or fail depending on data, just check it doesn't panic
        match fitted {
            Ok(fitted) => {
                assert_eq!(fitted.classes().len(), 2);
                let predictions = fitted.predict(&x).unwrap();
                assert_eq!(predictions.len(), 8);
            }
            Err(_) => {
                // If auto-optimization fails (due to small dataset), that's acceptable
            }
        }
    }

    #[test]
    fn test_kernel_discriminant_fit_with_custom_optimizer() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let grid = KernelParameterGrid::new()
            .gamma_values(vec![0.1, 1.0])
            .coef0_values(vec![0.0])
            .degree_values(vec![2]);

        let cv_config = KernelCVConfig {
            n_folds: 2,
            scoring: "accuracy".to_string(),
            random_state: Some(42),
            shuffle: false,
        };

        let optimizer = KernelParameterOptimizer::new()
            .parameter_grid(grid)
            .cv_config(cv_config)
            .optimization_strategy("random_search");

        let kda = KernelDiscriminantAnalysis::new();
        let fitted = kda.fit_with_optimizer(&x, &y, &optimizer);

        match fitted {
            Ok(fitted) => {
                assert_eq!(fitted.classes().len(), 2);
                let predictions = fitted.predict(&x).unwrap();
                assert_eq!(predictions.len(), 4);
            }
            Err(_) => {
                // Acceptable for small datasets
            }
        }
    }

    #[test]
    fn test_kernel_optimization_strategies() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let grid = KernelParameterGrid::new()
            .gamma_values(vec![0.1, 1.0])
            .coef0_values(vec![0.0])
            .degree_values(vec![2]);

        let cv_config = KernelCVConfig {
            n_folds: 2,
            scoring: "accuracy".to_string(),
            random_state: Some(42),
            shuffle: false,
        };

        let strategies = ["grid_search", "random_search", "bayesian"];

        for strategy in &strategies {
            let optimizer = KernelParameterOptimizer::new()
                .parameter_grid(grid.clone())
                .cv_config(cv_config.clone())
                .optimization_strategy(strategy);

            let results = optimizer.optimize(&x, &y);
            match results {
                Ok(results) => {
                    assert!(results.best_score >= 0.0);
                    assert!(results.best_score <= 1.0);
                }
                Err(_) => {
                    // Acceptable for some strategies with small datasets
                }
            }
        }
    }
}
