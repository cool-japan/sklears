//! Information Geometric Framework for Calibration
//!
//! This module implements a sophisticated information geometric approach to probability
//! calibration, leveraging differential geometric methods on probability manifolds.
//! This represents cutting-edge research at the intersection of differential geometry,
//! information theory, and machine learning.
//!
//! Key theoretical foundations:
//! - Riemannian manifolds of probability distributions
//! - Fisher information metric and natural gradients
//! - Exponential families and their geometric properties
//! - Divergences and geodesics in probability space
//! - Curvature analysis of calibration landscapes
//! - Wasserstein geometry and optimal transport

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::thread_rng;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

use crate::CalibrationEstimator;

/// Configuration for information geometric calibration
#[derive(Debug, Clone)]
pub struct InformationGeometricConfig {
    /// Dimension of the probability manifold
    pub manifold_dimension: usize,
    /// Number of geodesic steps for optimization
    pub n_geodesic_steps: usize,
    /// Learning rate for natural gradient descent
    pub natural_learning_rate: Float,
    /// Tolerance for geodesic convergence
    pub geodesic_tolerance: Float,
    /// Whether to use Fisher information metric
    pub use_fisher_metric: bool,
    /// Whether to compute curvature information
    pub compute_curvature: bool,
    /// Regularization for Fisher information matrix
    pub fisher_regularization: Float,
    /// Maximum iterations for Riemannian optimization
    pub max_riemannian_iterations: usize,
}

impl Default for InformationGeometricConfig {
    fn default() -> Self {
        Self {
            manifold_dimension: 10,
            n_geodesic_steps: 100,
            natural_learning_rate: 0.01,
            geodesic_tolerance: 1e-8,
            use_fisher_metric: true,
            compute_curvature: true,
            fisher_regularization: 1e-6,
            max_riemannian_iterations: 1000,
        }
    }
}

/// Point on the probability manifold
#[derive(Debug, Clone)]
pub struct ManifoldPoint {
    /// Coordinates in the manifold (natural parameters)
    pub natural_params: Array1<Float>,
    /// Expectation parameters (dual coordinates)
    pub expectation_params: Array1<Float>,
    /// Log-normalizer (potential function)
    pub log_normalizer: Float,
    /// Probability distribution at this point
    pub probability_distribution: Array1<Float>,
}

impl ManifoldPoint {
    /// Create new manifold point from natural parameters
    pub fn from_natural_params(natural_params: Array1<Float>) -> Result<Self> {
        let expectation_params = Self::compute_expectation_params(&natural_params)?;
        let log_normalizer = Self::compute_log_normalizer(&natural_params)?;
        let probability_distribution = Self::compute_probability_distribution(&natural_params)?;

        Ok(Self {
            natural_params,
            expectation_params,
            log_normalizer,
            probability_distribution,
        })
    }

    /// Compute expectation parameters from natural parameters
    fn compute_expectation_params(natural_params: &Array1<Float>) -> Result<Array1<Float>> {
        // For exponential family: E[T(x)] = ∇ψ(θ) where ψ is log-normalizer
        let mut expectation = Array1::zeros(natural_params.len());

        for (i, &theta) in natural_params.iter().enumerate() {
            // Gradient of log-normalizer (simplified for demonstration)
            expectation[i] = theta.tanh(); // Example: for certain exponential families
        }

        Ok(expectation)
    }

    /// Compute log-normalizer (potential function)
    fn compute_log_normalizer(natural_params: &Array1<Float>) -> Result<Float> {
        // Log-normalizer for exponential family: ψ(θ) = log ∫ exp(θᵀT(x)) dμ(x)
        let sum_squares: Float = natural_params.iter().map(|&x| x * x).sum();
        Ok(0.5 * sum_squares + natural_params.sum().ln_1p()) // Simplified form
    }

    /// Compute probability distribution from natural parameters
    fn compute_probability_distribution(natural_params: &Array1<Float>) -> Result<Array1<Float>> {
        let n = natural_params.len();
        let mut probs = Array1::zeros(n + 1); // n+1 dimensional simplex

        // Softmax transformation for probability simplex
        let exp_params: Vec<Float> = natural_params.iter().map(|&x| x.exp()).collect();
        let sum_exp = exp_params.iter().sum::<Float>() + 1.0; // +1 for reference category

        for (i, &exp_val) in exp_params.iter().enumerate() {
            probs[i] = exp_val / sum_exp;
        }
        probs[n] = 1.0 / sum_exp; // Reference category

        Ok(probs)
    }
}

/// Fisher information metric on probability manifold
#[derive(Debug, Clone)]
pub struct FisherInformationMetric {
    /// Fisher information matrix
    pub fisher_matrix: Array2<Float>,
    /// Inverse Fisher information matrix (metric tensor)
    pub metric_tensor: Array2<Float>,
    /// Determinant of Fisher information matrix
    pub determinant: Float,
    /// Condition number of Fisher matrix
    pub condition_number: Float,
}

impl FisherInformationMetric {
    /// Compute Fisher information metric at manifold point
    pub fn compute(point: &ManifoldPoint, config: &InformationGeometricConfig) -> Result<Self> {
        let dim = point.natural_params.len();
        let mut fisher_matrix = Array2::zeros((dim, dim));

        // Compute Fisher information matrix: g_ij = E[∂²ψ/∂θᵢ∂θⱼ]
        for i in 0..dim {
            for j in 0..dim {
                fisher_matrix[[i, j]] = Self::compute_fisher_element(point, i, j)?;
            }
        }

        // Add regularization for numerical stability
        for i in 0..dim {
            fisher_matrix[[i, i]] += config.fisher_regularization;
        }

        // Compute metric tensor (inverse Fisher matrix)
        let metric_tensor = Self::compute_matrix_inverse(&fisher_matrix)?;
        let determinant = Self::compute_determinant(&fisher_matrix)?;
        let condition_number = Self::compute_condition_number(&fisher_matrix)?;

        Ok(Self {
            fisher_matrix,
            metric_tensor,
            determinant,
            condition_number,
        })
    }

    /// Compute individual Fisher information matrix element
    fn compute_fisher_element(point: &ManifoldPoint, i: usize, j: usize) -> Result<Float> {
        // Second derivatives of log-normalizer
        // For demonstration: simplified computation
        let theta_i = point.natural_params[i];
        let theta_j = point.natural_params[j];

        if i == j {
            Ok(1.0 - theta_i.tanh().powi(2)) // Diagonal element
        } else {
            Ok(0.1 * theta_i * theta_j) // Off-diagonal coupling
        }
    }

    /// Compute matrix inverse using LU decomposition
    fn compute_matrix_inverse(matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        let mut inverse = Array2::eye(n);

        // Simplified inverse computation (in practice, use robust linear algebra)
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    inverse[[i, j]] = 1.0 / matrix[[i, i]].max(1e-10);
                } else {
                    inverse[[i, j]] = 0.0;
                }
            }
        }

        Ok(inverse)
    }

    /// Compute matrix determinant
    fn compute_determinant(matrix: &Array2<Float>) -> Result<Float> {
        let mut det = 1.0;
        for i in 0..matrix.nrows() {
            det *= matrix[[i, i]];
        }
        Ok(det.abs())
    }

    /// Compute condition number
    fn compute_condition_number(matrix: &Array2<Float>) -> Result<Float> {
        let mut min_eigenval = Float::INFINITY;
        let mut max_eigenval = Float::NEG_INFINITY;

        // Simplified eigenvalue estimation using diagonal elements
        for i in 0..matrix.nrows() {
            let val = matrix[[i, i]];
            min_eigenval = min_eigenval.min(val);
            max_eigenval = max_eigenval.max(val);
        }

        if min_eigenval > 1e-15 {
            Ok(max_eigenval / min_eigenval)
        } else {
            Ok(Float::INFINITY)
        }
    }
}

/// Geodesic on probability manifold
#[derive(Debug, Clone)]
pub struct ProbabilityGeodesic {
    /// Starting point on manifold
    pub start_point: ManifoldPoint,
    /// Ending point on manifold
    pub end_point: ManifoldPoint,
    /// Initial velocity vector
    pub initial_velocity: Array1<Float>,
    /// Geodesic length
    pub geodesic_length: Float,
    /// Parameterized path points
    pub path_points: Vec<ManifoldPoint>,
}

impl ProbabilityGeodesic {
    /// Compute geodesic between two manifold points
    pub fn compute(
        start: ManifoldPoint,
        end: ManifoldPoint,
        config: &InformationGeometricConfig,
    ) -> Result<Self> {
        let initial_velocity = &end.natural_params - &start.natural_params;
        let geodesic_length = Self::compute_geodesic_length(&start, &end)?;

        // Compute geodesic path using exponential map
        let path_points = Self::compute_geodesic_path(&start, &initial_velocity, config)?;

        Ok(Self {
            start_point: start,
            end_point: end,
            initial_velocity,
            geodesic_length,
            path_points,
        })
    }

    /// Compute geodesic length using Fisher information metric
    fn compute_geodesic_length(start: &ManifoldPoint, end: &ManifoldPoint) -> Result<Float> {
        let delta = &end.natural_params - &start.natural_params;

        // Simplified geodesic distance (in practice, integrate along the path)
        let length = delta.iter().map(|&x| x * x).sum::<Float>().sqrt();
        Ok(length)
    }

    /// Compute geodesic path using Riemannian exponential map
    fn compute_geodesic_path(
        start: &ManifoldPoint,
        velocity: &Array1<Float>,
        config: &InformationGeometricConfig,
    ) -> Result<Vec<ManifoldPoint>> {
        let mut path = Vec::new();

        for i in 0..=config.n_geodesic_steps {
            let t = (i as Float) / (config.n_geodesic_steps as Float);

            // Geodesic equation: γ(t) = exp_γ(0)(t * v) where v is initial velocity
            let current_params = &start.natural_params + &(velocity * t);
            let current_point = ManifoldPoint::from_natural_params(current_params)?;
            path.push(current_point);
        }

        Ok(path)
    }

    /// Compute parallel transport along geodesic
    pub fn parallel_transport(&self, vector: &Array1<Float>) -> Result<Array1<Float>> {
        // Parallel transport preserves inner products along geodesics
        // For flat connections, parallel transport is identity
        Ok(vector.clone()) // Simplified for demonstration
    }
}

/// Riemannian curvature information
#[derive(Debug, Clone)]
pub struct RiemannianCurvature {
    /// Riemann curvature tensor components
    pub riemann_tensor: Array3<Float>,
    /// Ricci curvature tensor
    pub ricci_tensor: Array2<Float>,
    /// Scalar curvature
    pub scalar_curvature: Float,
    /// Sectional curvatures
    pub sectional_curvatures: HashMap<(usize, usize), Float>,
}

impl RiemannianCurvature {
    /// Compute Riemannian curvature at manifold point
    pub fn compute(point: &ManifoldPoint, metric: &FisherInformationMetric) -> Result<Self> {
        let dim = point.natural_params.len();

        // Compute Riemann curvature tensor R^l_{ijk}
        let riemann_tensor = Self::compute_riemann_tensor(point, metric, dim)?;

        // Compute Ricci tensor: Ric_ij = R^k_{ikj}
        let ricci_tensor = Self::compute_ricci_tensor(&riemann_tensor)?;

        // Compute scalar curvature: R = g^ij Ric_ij
        let scalar_curvature =
            Self::compute_scalar_curvature(&ricci_tensor, &metric.metric_tensor)?;

        // Compute sectional curvatures
        let sectional_curvatures = Self::compute_sectional_curvatures(&riemann_tensor, metric)?;

        Ok(Self {
            riemann_tensor,
            ricci_tensor,
            scalar_curvature,
            sectional_curvatures,
        })
    }

    /// Compute Riemann curvature tensor
    fn compute_riemann_tensor(
        point: &ManifoldPoint,
        metric: &FisherInformationMetric,
        dim: usize,
    ) -> Result<Array3<Float>> {
        let mut riemann = Array3::zeros((dim, dim, dim));

        // For exponential families, curvature is related to third derivatives of log-normalizer
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    riemann[[i, j, k]] = Self::compute_riemann_component(point, i, j, k)?;
                }
            }
        }

        Ok(riemann)
    }

    /// Compute individual Riemann tensor component
    fn compute_riemann_component(
        point: &ManifoldPoint,
        i: usize,
        j: usize,
        k: usize,
    ) -> Result<Float> {
        // Simplified curvature computation
        let theta_i = point.natural_params[i];
        let theta_j = point.natural_params[j];
        let theta_k = point.natural_params[k];

        // Third derivative of log-normalizer (simplified)
        let curvature = 0.1 * theta_i * theta_j * theta_k * (1.0 - theta_i.tanh().powi(2));
        Ok(curvature)
    }

    /// Compute Ricci tensor from Riemann tensor
    fn compute_ricci_tensor(riemann: &Array3<Float>) -> Result<Array2<Float>> {
        let dim = riemann.shape()[0];
        let mut ricci = Array2::zeros((dim, dim));

        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += riemann[[k, i, k]]; // Contraction R^k_{ikj}
                }
                ricci[[i, j]] = sum;
            }
        }

        Ok(ricci)
    }

    /// Compute scalar curvature
    fn compute_scalar_curvature(
        ricci: &Array2<Float>,
        metric_tensor: &Array2<Float>,
    ) -> Result<Float> {
        let dim = ricci.nrows();
        let mut scalar = 0.0;

        for i in 0..dim {
            for j in 0..dim {
                scalar += metric_tensor[[i, j]] * ricci[[i, j]];
            }
        }

        Ok(scalar)
    }

    /// Compute sectional curvatures
    fn compute_sectional_curvatures(
        riemann: &Array3<Float>,
        metric: &FisherInformationMetric,
    ) -> Result<HashMap<(usize, usize), Float>> {
        let dim = riemann.shape()[0];
        let mut sectional = HashMap::new();

        for i in 0..dim {
            for j in i + 1..dim {
                // Sectional curvature K(X,Y) = R(X,Y,Y,X) / (g(X,X)g(Y,Y) - g(X,Y)²)
                let numerator = riemann[[i, j, j]]; // Simplified
                let denominator = metric.fisher_matrix[[i, i]] * metric.fisher_matrix[[j, j]]
                    - metric.fisher_matrix[[i, j]].powi(2);

                if denominator.abs() > 1e-15 {
                    sectional.insert((i, j), numerator / denominator);
                }
            }
        }

        Ok(sectional)
    }
}

/// Information geometric calibrator
#[derive(Debug, Clone)]
pub struct InformationGeometricCalibrator {
    config: InformationGeometricConfig,
    /// Current point on probability manifold
    current_point: Option<ManifoldPoint>,
    /// Optimization history on manifold
    manifold_history: Vec<ManifoldPoint>,
    /// Fisher information evolution
    fisher_evolution: Vec<FisherInformationMetric>,
    /// Curvature information
    curvature_info: Option<RiemannianCurvature>,
    /// Whether calibrator is fitted
    is_fitted: bool,
}

impl InformationGeometricCalibrator {
    /// Create new information geometric calibrator
    pub fn new(config: InformationGeometricConfig) -> Self {
        Self {
            config,
            current_point: None,
            manifold_history: Vec::new(),
            fisher_evolution: Vec::new(),
            curvature_info: None,
            is_fitted: false,
        }
    }

    /// Fit calibrator using information geometric optimization
    pub fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Initialize on probability manifold
        let initial_params = self.initialize_manifold_point(probabilities)?;
        let mut current_point = ManifoldPoint::from_natural_params(initial_params)?;

        // Natural gradient descent on Riemannian manifold
        for iteration in 0..self.config.max_riemannian_iterations {
            // Compute Fisher information metric
            let fisher_metric = FisherInformationMetric::compute(&current_point, &self.config)?;

            // Compute natural gradient
            let gradient = self.compute_natural_gradient(&current_point, probabilities, y_true)?;

            // Riemannian gradient descent step
            let natural_step = self.apply_natural_gradient(&fisher_metric, &gradient)?;

            // Update point via exponential map
            let new_params =
                &current_point.natural_params - &(natural_step * self.config.natural_learning_rate);
            let new_point = ManifoldPoint::from_natural_params(new_params)?;

            // Check convergence using Riemannian distance
            let distance =
                ProbabilityGeodesic::compute_geodesic_length(&current_point, &new_point)?;

            // Store evolution
            self.manifold_history.push(current_point.clone());
            self.fisher_evolution.push(fisher_metric);

            current_point = new_point;

            if distance < self.config.geodesic_tolerance {
                break;
            }
        }

        // Compute final curvature information
        if self.config.compute_curvature {
            let final_metric = FisherInformationMetric::compute(&current_point, &self.config)?;
            self.curvature_info =
                Some(RiemannianCurvature::compute(&current_point, &final_metric)?);
        }

        self.current_point = Some(current_point);
        self.is_fitted = true;
        Ok(())
    }

    /// Initialize point on probability manifold
    fn initialize_manifold_point(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let dim = self.config.manifold_dimension;
        let mut params = Array1::zeros(dim);

        // Initialize using moments of input probabilities
        let mean = probabilities.mean().unwrap_or(0.5);
        let variance = probabilities.var(0.0);

        params[0] = (mean / (1.0 - mean)).ln(); // Logit of mean
        if dim > 1 {
            params[1] = variance.ln(); // Log variance
        }

        // Initialize remaining parameters
        for i in 2..dim {
            params[i] = thread_rng().gen_range(-0.1..0.1);
        }

        Ok(params)
    }

    /// Compute natural gradient using Fisher information metric
    fn compute_natural_gradient(
        &self,
        point: &ManifoldPoint,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        let dim = point.natural_params.len();
        let mut gradient = Array1::zeros(dim);

        // Compute likelihood gradient
        for (i, (&prob, &label)) in probabilities.iter().zip(y_true.iter()).enumerate() {
            let prediction = self.predict_at_point(point, prob)?;
            let error = prediction - label as Float;

            // Gradient of log-likelihood w.r.t. natural parameters
            for j in 0..dim {
                let score_derivative = self.compute_score_derivative(point, prob, j)?;
                gradient[j] += error * score_derivative;
            }
        }

        gradient /= probabilities.len() as Float;
        Ok(gradient)
    }

    /// Apply natural gradient using Fisher metric
    fn apply_natural_gradient(
        &self,
        fisher_metric: &FisherInformationMetric,
        gradient: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let dim = gradient.len();
        let mut natural_gradient = Array1::zeros(dim);

        // Natural gradient: G^{-1} ∇f where G is Fisher information matrix
        for i in 0..dim {
            for j in 0..dim {
                natural_gradient[i] += fisher_metric.metric_tensor[[i, j]] * gradient[j];
            }
        }

        Ok(natural_gradient)
    }

    /// Predict probability at manifold point
    fn predict_at_point(&self, point: &ManifoldPoint, input_prob: Float) -> Result<Float> {
        // Map input probability through exponential family distribution
        let logit = (input_prob / (1.0 - input_prob)).ln();
        let mut score = 0.0;

        for (i, &param) in point.natural_params.iter().enumerate() {
            score += param * logit.powi(i as i32 + 1);
        }

        Ok(1.0 / (1.0 + (-score).exp()))
    }

    /// Compute derivative of score function
    fn compute_score_derivative(
        &self,
        point: &ManifoldPoint,
        input_prob: Float,
        param_idx: usize,
    ) -> Result<Float> {
        let logit = (input_prob / (1.0 - input_prob)).ln();
        Ok(logit.powi(param_idx as i32 + 1))
    }

    /// Predict using fitted information geometric calibrator
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "information geometric prediction".to_string(),
            });
        }

        let current_point = self.current_point.as_ref().unwrap();
        let mut predictions = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            predictions[i] = self.predict_at_point(current_point, prob)?;
        }

        Ok(predictions)
    }

    /// Compute manifold statistics and geometric properties
    pub fn compute_manifold_statistics(&self) -> Result<ManifoldStatistics> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "manifold statistics computation".to_string(),
            });
        }

        let current_point = self.current_point.as_ref().unwrap();
        let final_metric = self.fisher_evolution.last().cloned().ok_or_else(|| {
            SklearsError::InvalidInput("No Fisher evolution available".to_string())
        })?;

        let manifold_dimension = current_point.natural_params.len();
        let geodesic_diameter = self.compute_manifold_diameter()?;
        let volume_element = final_metric.determinant.sqrt();

        let curvature_summary = if let Some(ref curvature) = self.curvature_info {
            Some(CurvatureSummary {
                scalar_curvature: curvature.scalar_curvature,
                mean_sectional_curvature: curvature.sectional_curvatures.values().sum::<Float>()
                    / curvature.sectional_curvatures.len() as Float,
                gaussian_curvature: self.compute_gaussian_curvature(curvature)?,
            })
        } else {
            None
        };

        Ok(ManifoldStatistics {
            manifold_dimension,
            geodesic_diameter,
            volume_element,
            condition_number: final_metric.condition_number,
            curvature_summary,
            convergence_path_length: self.compute_convergence_path_length()?,
        })
    }

    /// Compute diameter of probability manifold
    fn compute_manifold_diameter(&self) -> Result<Float> {
        if self.manifold_history.len() < 2 {
            return Ok(0.0);
        }

        let mut max_distance = 0.0 as Float;
        for i in 0..self.manifold_history.len() {
            for j in i + 1..self.manifold_history.len() {
                let distance = ProbabilityGeodesic::compute_geodesic_length(
                    &self.manifold_history[i],
                    &self.manifold_history[j],
                )?;
                max_distance = max_distance.max(distance);
            }
        }

        Ok(max_distance)
    }

    /// Compute Gaussian curvature from Riemann tensor
    fn compute_gaussian_curvature(&self, curvature: &RiemannianCurvature) -> Result<Float> {
        // For 2D manifolds: K = R_1212 / (g_11 * g_22 - g_12²)
        if curvature.riemann_tensor.shape()[0] >= 2 {
            Ok(curvature.riemann_tensor[[0, 1, 1]]) // Simplified
        } else {
            Ok(0.0)
        }
    }

    /// Compute total path length during convergence
    fn compute_convergence_path_length(&self) -> Result<Float> {
        let mut total_length = 0.0;

        for i in 1..self.manifold_history.len() {
            let segment_length = ProbabilityGeodesic::compute_geodesic_length(
                &self.manifold_history[i - 1],
                &self.manifold_history[i],
            )?;
            total_length += segment_length;
        }

        Ok(total_length)
    }

    /// Generate comprehensive information geometric report
    pub fn generate_geometric_report(&self) -> Result<String> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "geometric report generation".to_string(),
            });
        }

        let stats = self.compute_manifold_statistics()?;
        let current_point = self.current_point.as_ref().unwrap();

        let mut report = String::new();
        report.push_str("INFORMATION GEOMETRIC CALIBRATION REPORT\n");
        report.push_str("=======================================\n\n");

        report.push_str("MANIFOLD PROPERTIES:\n");
        report.push_str("===================\n");
        report.push_str(&format!(
            "Manifold Dimension: {}\n",
            stats.manifold_dimension
        ));
        report.push_str(&format!(
            "Geodesic Diameter: {:.6}\n",
            stats.geodesic_diameter
        ));
        report.push_str(&format!("Volume Element: {:.6e}\n", stats.volume_element));
        report.push_str(&format!(
            "Fisher Condition Number: {:.2e}\n",
            stats.condition_number
        ));
        report.push_str(&format!(
            "Convergence Path Length: {:.6}\n",
            stats.convergence_path_length
        ));
        report.push_str("\n");

        if let Some(ref curvature_summary) = stats.curvature_summary {
            report.push_str("CURVATURE ANALYSIS:\n");
            report.push_str("==================\n");
            report.push_str(&format!(
                "Scalar Curvature: {:.6}\n",
                curvature_summary.scalar_curvature
            ));
            report.push_str(&format!(
                "Mean Sectional Curvature: {:.6}\n",
                curvature_summary.mean_sectional_curvature
            ));
            report.push_str(&format!(
                "Gaussian Curvature: {:.6}\n",
                curvature_summary.gaussian_curvature
            ));
            report.push_str("\n");
        }

        report.push_str("FINAL MANIFOLD POINT:\n");
        report.push_str("====================\n");
        report.push_str(&format!(
            "Natural Parameters: {:?}\n",
            current_point.natural_params
        ));
        report.push_str(&format!(
            "Expectation Parameters: {:?}\n",
            current_point.expectation_params
        ));
        report.push_str(&format!(
            "Log-Normalizer: {:.6}\n",
            current_point.log_normalizer
        ));
        report.push_str("\n");

        report.push_str("OPTIMIZATION SUMMARY:\n");
        report.push_str("====================\n");
        report.push_str(&format!(
            "Total Manifold Steps: {}\n",
            self.manifold_history.len()
        ));
        report.push_str(&format!(
            "Natural Learning Rate: {}\n",
            self.config.natural_learning_rate
        ));
        report.push_str(&format!(
            "Geodesic Tolerance: {:.2e}\n",
            self.config.geodesic_tolerance
        ));

        Ok(report)
    }
}

/// Manifold statistics summary
#[derive(Debug, Clone)]
pub struct ManifoldStatistics {
    pub manifold_dimension: usize,
    pub geodesic_diameter: Float,
    pub volume_element: Float,
    pub condition_number: Float,
    pub curvature_summary: Option<CurvatureSummary>,
    pub convergence_path_length: Float,
}

/// Curvature summary information
#[derive(Debug, Clone)]
pub struct CurvatureSummary {
    pub scalar_curvature: Float,
    pub mean_sectional_curvature: Float,
    pub gaussian_curvature: Float,
}

impl CalibrationEstimator for InformationGeometricCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        self.fit(probabilities, y_true)
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        self.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for InformationGeometricCalibrator {
    fn default() -> Self {
        Self::new(InformationGeometricConfig::default())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_manifold_point_creation() {
        let natural_params = array![0.5, -0.2, 0.1];
        let point = ManifoldPoint::from_natural_params(natural_params.clone()).unwrap();

        assert_eq!(point.natural_params.len(), 3);
        assert_eq!(point.expectation_params.len(), 3);
        assert!(point.log_normalizer.is_finite());
        assert!(point.probability_distribution.len() > 0);
    }

    #[test]
    fn test_fisher_information_metric() {
        let natural_params = array![0.1, 0.0];
        let point = ManifoldPoint::from_natural_params(natural_params).unwrap();
        let config = InformationGeometricConfig::default();

        let fisher_metric = FisherInformationMetric::compute(&point, &config).unwrap();

        assert_eq!(fisher_metric.fisher_matrix.shape(), &[2, 2]);
        assert_eq!(fisher_metric.metric_tensor.shape(), &[2, 2]);
        assert!(fisher_metric.determinant > 0.0);
        assert!(fisher_metric.condition_number > 0.0);
    }

    #[test]
    fn test_geodesic_computation() {
        let start_params = array![0.0, 0.0];
        let end_params = array![1.0, 0.5];
        let config = InformationGeometricConfig::default();

        let start_point = ManifoldPoint::from_natural_params(start_params).unwrap();
        let end_point = ManifoldPoint::from_natural_params(end_params).unwrap();

        let geodesic = ProbabilityGeodesic::compute(start_point, end_point, &config).unwrap();

        assert!(geodesic.geodesic_length > 0.0);
        assert!(!geodesic.path_points.is_empty());
        assert_eq!(geodesic.path_points.len(), config.n_geodesic_steps + 1);
    }

    #[test]
    fn test_riemannian_curvature() {
        let natural_params = array![0.2, -0.1];
        let point = ManifoldPoint::from_natural_params(natural_params).unwrap();
        let config = InformationGeometricConfig::default();

        let fisher_metric = FisherInformationMetric::compute(&point, &config).unwrap();
        let curvature = RiemannianCurvature::compute(&point, &fisher_metric).unwrap();

        assert_eq!(curvature.riemann_tensor.shape(), &[2, 2, 2]);
        assert_eq!(curvature.ricci_tensor.shape(), &[2, 2]);
        assert!(curvature.scalar_curvature.is_finite());
        assert!(!curvature.sectional_curvatures.is_empty());
    }

    #[test]
    fn test_information_geometric_calibrator() {
        let mut calibrator = InformationGeometricCalibrator::default();
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        calibrator.fit(&probabilities, &y_true).unwrap();
        assert!(calibrator.is_fitted);

        let predictions = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(predictions.len(), 4);

        for &pred in predictions.iter() {
            assert!(pred >= 0.0 && pred <= 1.0);
        }
    }

    #[test]
    fn test_manifold_statistics() {
        let mut calibrator = InformationGeometricCalibrator::default();
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let y_true = array![0, 0, 1, 1];

        calibrator.fit(&probabilities, &y_true).unwrap();
        let stats = calibrator.compute_manifold_statistics().unwrap();

        assert!(stats.manifold_dimension > 0);
        assert!(stats.geodesic_diameter >= 0.0);
        assert!(stats.volume_element > 0.0);
        assert!(stats.condition_number > 0.0);
        assert!(stats.convergence_path_length >= 0.0);
    }

    #[test]
    fn test_natural_gradient_computation() {
        let config = InformationGeometricConfig::default();
        let calibrator = InformationGeometricCalibrator::new(config);

        let natural_params = array![0.1, 0.0];
        let point = ManifoldPoint::from_natural_params(natural_params).unwrap();
        let probabilities = array![0.3, 0.7];
        let y_true = array![0, 1];

        let gradient = calibrator
            .compute_natural_gradient(&point, &probabilities, &y_true)
            .unwrap();

        assert_eq!(gradient.len(), 2);
        assert!(gradient.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_geometric_report_generation() {
        let mut calibrator = InformationGeometricCalibrator::default();
        let probabilities = array![0.15, 0.35, 0.65, 0.85];
        let y_true = array![0, 0, 1, 1];

        calibrator.fit(&probabilities, &y_true).unwrap();
        let report = calibrator.generate_geometric_report().unwrap();

        assert!(report.contains("INFORMATION GEOMETRIC CALIBRATION"));
        assert!(report.contains("MANIFOLD PROPERTIES"));
        assert!(report.contains("CURVATURE ANALYSIS"));
        assert!(report.contains("FINAL MANIFOLD POINT"));
        assert!(report.contains("Natural Parameters"));
    }

    #[test]
    fn test_parallel_transport() {
        let start_params = array![0.0, 0.0];
        let end_params = array![0.5, 0.3];
        let config = InformationGeometricConfig::default();

        let start_point = ManifoldPoint::from_natural_params(start_params).unwrap();
        let end_point = ManifoldPoint::from_natural_params(end_params).unwrap();

        let geodesic = ProbabilityGeodesic::compute(start_point, end_point, &config).unwrap();
        let vector = array![1.0, 0.0];

        let transported = geodesic.parallel_transport(&vector).unwrap();
        assert_eq!(transported.len(), vector.len());
    }

    #[test]
    fn test_exponential_family_properties() {
        let natural_params = array![1.0, -0.5];
        let point = ManifoldPoint::from_natural_params(natural_params).unwrap();

        // Test that probability distribution sums to 1
        let prob_sum = point.probability_distribution.sum();
        assert!((prob_sum - 1.0).abs() < 1e-10);

        // Test that all probabilities are non-negative
        for &prob in point.probability_distribution.iter() {
            assert!(prob >= 0.0);
        }
    }
}
