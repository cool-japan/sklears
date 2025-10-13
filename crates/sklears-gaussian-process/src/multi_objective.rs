//! Multi-Objective Bayesian Optimization
//!
//! This module implements multi-objective Bayesian optimization (MOBO) methods that can
//! simultaneously optimize multiple conflicting objectives. It includes Pareto frontier
//! approximation, multi-objective acquisition functions, and scalarization techniques.
//!
//! # Mathematical Background
//!
//! Multi-objective optimization seeks to find the Pareto optimal set:
//! min f(x) = [f₁(x), f₂(x), ..., fₖ(x)]
//!
//! Key concepts:
//! 1. **Pareto dominance**: x dominates y if f(x) ≤ f(y) for all objectives and f(x) < f(y) for at least one
//! 2. **Pareto frontier**: Set of non-dominated solutions
//! 3. **Multi-objective acquisition functions**: Extension of single-objective acquisition to vector-valued objectives
//! 4. **Scalarization**: Converting multi-objective problem to single-objective
//!
//! # Examples
//!
//! ```rust
//! use sklears_gaussian_process::multi_objective::{
//!     MultiObjectiveBayesianOptimizer, MultiObjectiveAcquisition, ScalarizationMethod
//! };
//! use sklears_gaussian_process::kernels::RBF;
//! use scirs2_core::ndarray::array;
//!
//! // Create multi-objective Bayesian optimizer
//! let mobo = MultiObjectiveBayesianOptimizer::builder()
//!     .kernel(Box::new(RBF::new(1.0)))
//!     .acquisition_function(MultiObjectiveAcquisition::ExpectedHypervolume)
//!     .n_objectives(2)
//!     .reference_point(array![10.0, 10.0])
//!     .build();
//!
//! // Training data with multiple objectives
//! let X = array![[1.0], [2.0], [3.0], [4.0]];
//! let Y = array![[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0]]; // Two objectives
//!
//! let trained_mobo = mobo.fit(&X, &Y).unwrap();
//! let next_point = trained_mobo.suggest_next_point(&array![[0.0], [5.0]]).unwrap();
//! ```

use crate::kernels::Kernel;
use crate::regression::{MogprTrained, MultiOutputGaussianProcessRegressor};
use crate::utils;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis, concatenate};
use scirs2_core::random::{thread_rng, Random, Rng}; // SciRS2 Policy
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict};
use std::f64::consts::PI;

/// State marker for untrained multi-objective optimizer
#[derive(Debug, Clone)]
pub struct Untrained;

/// State marker for trained multi-objective optimizer
#[derive(Debug, Clone)]
pub struct Trained {
    pub gp_models: MultiOutputGaussianProcessRegressor<MogprTrained>,
    pub pareto_frontier: ParetoFrontier,
    pub hypervolume_indicator: f64,
    pub reference_point: Array1<f64>,
    pub acquisition_function: MultiObjectiveAcquisition,
    pub scalarization_method: Option<ScalarizationMethod>,
}

/// Multi-objective acquisition functions
#[derive(Debug, Clone)]
pub enum MultiObjectiveAcquisition {
    /// Expected Hypervolume Improvement
    ExpectedHypervolume,
    /// Probability of Improvement for multiple objectives
    MultiObjectiveProbabilityOfImprovement { epsilon: f64 },
    /// Expected Improvement with scalarization
    ScalarizedExpectedImprovement { weights: Array1<f64> },
    /// ParEGO (Efficient Global Optimization for multi-objective)
    ParEGO { rho: f64 },
    /// SMS-EGO (S-Metric Selection for multi-objective)
    SMSEGO,
    /// Multi-objective Upper Confidence Bound
    MultiObjectiveUCB { beta: f64 },
    /// Entropy Search for multi-objective
    MultiObjectiveEntropySearch,
}

/// Scalarization methods for multi-objective problems
#[derive(Debug, Clone)]
pub enum ScalarizationMethod {
    /// Linear weighted scalarization: Σ wᵢ fᵢ(x)
    LinearWeighted { weights: Array1<f64> },
    /// Weighted Tchebycheff: max wᵢ |fᵢ(x) - zᵢ*|
    WeightedTchebycheff {
        weights: Array1<f64>,
        ideal_point: Array1<f64>,
    },
    /// Augmented Tchebycheff scalarization
    AugmentedTchebycheff {
        weights: Array1<f64>,
        ideal_point: Array1<f64>,
        rho: f64,
    },
    /// Achievement Scalarizing Function
    AchievementFunction {
        weights: Array1<f64>,
        reference_point: Array1<f64>,
    },
    /// Penalty Boundary Intersection
    PenaltyBoundaryIntersection {
        weights: Array1<f64>,
        ideal_point: Array1<f64>,
        theta: f64,
    },
}

impl ScalarizationMethod {
    /// Apply scalarization to convert multi-objective values to single objective
    pub fn scalarize(&self, objectives: &Array1<f64>) -> f64 {
        match self {
            Self::LinearWeighted { weights } => objectives
                .iter()
                .zip(weights.iter())
                .map(|(&f, &w)| w * f)
                .sum(),
            Self::WeightedTchebycheff {
                weights,
                ideal_point,
            } => objectives
                .iter()
                .zip(weights.iter())
                .zip(ideal_point.iter())
                .map(|((&f, &w), &z)| w * (f - z).abs())
                .fold(0.0f64, |a, b| a.max(b)),
            Self::AugmentedTchebycheff {
                weights,
                ideal_point,
                rho,
            } => {
                let tcheby_term = objectives
                    .iter()
                    .zip(weights.iter())
                    .zip(ideal_point.iter())
                    .map(|((&f, &w), &z)| w * (f - z).abs())
                    .fold(0.0f64, |a, b| a.max(b));

                let augmentation_term = rho
                    * objectives
                        .iter()
                        .zip(ideal_point.iter())
                        .map(|(&f, &z)| f - z)
                        .sum::<f64>();

                tcheby_term + augmentation_term
            }
            Self::AchievementFunction {
                weights,
                reference_point,
            } => objectives
                .iter()
                .zip(weights.iter())
                .zip(reference_point.iter())
                .map(|((&f, &w), &r)| {
                    if f <= r {
                        w * f
                    } else {
                        w * f + 1000.0 * (f - r)
                    }
                })
                .sum(),
            Self::PenaltyBoundaryIntersection {
                weights,
                ideal_point,
                theta,
            } => {
                let normalized = objectives
                    .iter()
                    .zip(ideal_point.iter())
                    .map(|(&f, &z)| f - z)
                    .collect::<Array1<f64>>();

                let w_norm = weights.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let parallel_component = normalized.dot(weights) / w_norm;

                let perpendicular_squared =
                    normalized.iter().map(|x| x.powi(2)).sum::<f64>() - parallel_component.powi(2);

                parallel_component + theta * perpendicular_squared.sqrt()
            }
        }
    }

    /// Create linear weighted scalarization
    pub fn linear_weighted(weights: Array1<f64>) -> Self {
        Self::LinearWeighted { weights }
    }

    /// Create Tchebycheff scalarization
    pub fn tchebycheff(weights: Array1<f64>, ideal_point: Array1<f64>) -> Self {
        Self::WeightedTchebycheff {
            weights,
            ideal_point,
        }
    }
}

/// Pareto frontier representation and operations
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    pub points: Array2<f64>,           // Design variables
    pub objectives: Array2<f64>,       // Objective values
    pub dominated_indices: Vec<usize>, // Indices of dominated points
}

impl ParetoFrontier {
    /// Create new Pareto frontier from data
    pub fn from_data(points: Array2<f64>, objectives: Array2<f64>) -> SklResult<Self> {
        if points.nrows() != objectives.nrows() {
            return Err(SklearsError::DimensionMismatch {
                expected: points.nrows(),
                actual: objectives.nrows(),
            });
        }

        let dominated_indices = Self::find_dominated_solutions(&objectives);

        Ok(Self {
            points,
            objectives,
            dominated_indices,
        })
    }

    /// Find dominated solutions using fast non-dominated sorting
    pub fn find_dominated_solutions(objectives: &Array2<f64>) -> Vec<usize> {
        let n = objectives.nrows();
        let mut dominated = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j && Self::dominates(&objectives.row(j), &objectives.row(i)) {
                    dominated.push(i);
                    break;
                }
            }
        }

        dominated
    }

    /// Check if solution a dominates solution b (for minimization)
    pub fn dominates(
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> bool {
        let mut at_least_one_better = false;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            if a_val > b_val {
                return false; // a is worse in at least one objective
            }
            if a_val < b_val {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Get non-dominated points
    pub fn non_dominated_points(&self) -> (Array2<f64>, Array2<f64>) {
        let non_dominated_indices: Vec<usize> = (0..self.points.nrows())
            .filter(|i| !self.dominated_indices.contains(i))
            .collect();

        if non_dominated_indices.is_empty() {
            return (
                Array2::zeros((0, self.points.ncols())),
                Array2::zeros((0, self.objectives.ncols())),
            );
        }

        let nd_points = self.points.select(Axis(0), &non_dominated_indices);
        let nd_objectives = self.objectives.select(Axis(0), &non_dominated_indices);

        (nd_points, nd_objectives)
    }

    /// Compute hypervolume indicator
    pub fn hypervolume(&self, reference_point: &Array1<f64>) -> f64 {
        let (_, nd_objectives) = self.non_dominated_points();

        if nd_objectives.is_empty() {
            return 0.0;
        }

        // Simple hypervolume calculation for 2D case
        if nd_objectives.ncols() == 2 {
            self.hypervolume_2d(&nd_objectives, reference_point)
        } else {
            // For higher dimensions, use Monte Carlo approximation
            self.hypervolume_monte_carlo(&nd_objectives, reference_point, 10000)
        }
    }

    /// Hypervolume calculation for 2D objectives
    fn hypervolume_2d(&self, objectives: &Array2<f64>, reference_point: &Array1<f64>) -> f64 {
        if objectives.is_empty() {
            return 0.0;
        }

        // Sort points by first objective
        let mut sorted_indices = (0..objectives.nrows()).collect::<Vec<_>>();
        sorted_indices.sort_by(|&i, &j| {
            objectives[[i, 0]]
                .partial_cmp(&objectives[[j, 0]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut hypervolume = 0.0;
        let mut prev_y = reference_point[1];

        for &idx in &sorted_indices {
            let x = objectives[[idx, 0]];
            let y = objectives[[idx, 1]];

            if x < reference_point[0] && y < reference_point[1] {
                hypervolume += (reference_point[0] - x) * (prev_y - y);
                prev_y = y;
            }
        }

        hypervolume
    }

    /// Monte Carlo hypervolume approximation for higher dimensions
    fn hypervolume_monte_carlo(
        &self,
        objectives: &Array2<f64>,
        reference_point: &Array1<f64>,
        n_samples: usize,
    ) -> f64 {
        if objectives.is_empty() {
            return 0.0;
        }

        let mut rng = thread_rng();
        let mut dominated_count = 0;

        // Find bounds
        let mut min_bounds = Array1::zeros(objectives.ncols());
        let mut max_bounds = reference_point.clone();

        for j in 0..objectives.ncols() {
            let min_val = objectives
                .column(j)
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            min_bounds[j] = min_val;
        }

        // Volume of the bounding box
        let box_volume = max_bounds
            .iter()
            .zip(min_bounds.iter())
            .map(|(&max, &min)| max - min)
            .product::<f64>();

        // Monte Carlo sampling
        for _ in 0..n_samples {
            let mut sample = Array1::zeros(objectives.ncols());
            for j in 0..objectives.ncols() {
                sample[j] = min_bounds[j] + rng.gen::<f64>() * (max_bounds[j] - min_bounds[j]);
            }

            // Check if sample is dominated by any point in the frontier
            let mut is_dominated = false;
            for i in 0..objectives.nrows() {
                let obj_row = objectives.row(i);
                if Self::dominates(&obj_row, &sample.view()) {
                    is_dominated = true;
                    break;
                }
            }

            if is_dominated {
                dominated_count += 1;
            }
        }

        (dominated_count as f64 / n_samples as f64) * box_volume
    }

    /// Update frontier with new point
    pub fn update(&mut self, new_point: Array1<f64>, new_objectives: Array1<f64>) -> bool {
        // Check if new point is dominated by existing points
        for i in 0..self.objectives.nrows() {
            if Self::dominates(&self.objectives.row(i), &new_objectives.view()) {
                return false; // New point is dominated, don't add
            }
        }

        // Add new point
        let new_points = Array2::from_shape_vec((1, new_point.len()), new_point.to_vec()).unwrap();
        let new_objs =
            Array2::from_shape_vec((1, new_objectives.len()), new_objectives.to_vec()).unwrap();

        self.points = concatenate![Axis(0), self.points, new_points];
        self.objectives = concatenate![Axis(0), self.objectives, new_objs];

        // Recompute dominated solutions
        self.dominated_indices = Self::find_dominated_solutions(&self.objectives);

        true
    }
}

/// Multi-Objective Bayesian Optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveBayesianOptimizer<S = Untrained> {
    kernel: Option<Box<dyn Kernel>>,
    acquisition_function: MultiObjectiveAcquisition,
    n_objectives: usize,
    reference_point: Option<Array1<f64>>,
    scalarization_method: Option<ScalarizationMethod>,
    n_candidates: usize,
    alpha: f64,
    _state: S,
}

/// Configuration for multi-objective Bayesian optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    pub acquisition: MultiObjectiveAcquisition,
    pub n_objectives: usize,
    pub n_candidates: usize,
    pub regularization: f64,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            acquisition: MultiObjectiveAcquisition::ExpectedHypervolume,
            n_objectives: 2,
            n_candidates: 1000,
            regularization: 1e-6,
        }
    }
}

impl MultiObjectiveBayesianOptimizer<Untrained> {
    /// Create a new multi-objective Bayesian optimizer
    pub fn new() -> Self {
        Self {
            kernel: None,
            acquisition_function: MultiObjectiveAcquisition::ExpectedHypervolume,
            n_objectives: 2,
            reference_point: None,
            scalarization_method: None,
            n_candidates: 1000,
            alpha: 1e-6,
            _state: Untrained,
        }
    }

    /// Create a builder for multi-objective optimizer
    pub fn builder() -> MultiObjectiveBuilder {
        MultiObjectiveBuilder::new()
    }

    /// Set the kernel
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the acquisition function
    pub fn acquisition_function(mut self, acquisition: MultiObjectiveAcquisition) -> Self {
        self.acquisition_function = acquisition;
        self
    }

    /// Set number of objectives
    pub fn n_objectives(mut self, n_objectives: usize) -> Self {
        self.n_objectives = n_objectives;
        self
    }

    /// Set reference point for hypervolume calculation
    pub fn reference_point(mut self, point: Array1<f64>) -> Self {
        self.reference_point = Some(point);
        self
    }

    /// Set scalarization method
    pub fn scalarization_method(mut self, method: ScalarizationMethod) -> Self {
        self.scalarization_method = Some(method);
        self
    }
}

/// Builder for multi-objective Bayesian optimizer
#[derive(Debug, Clone)]
pub struct MultiObjectiveBuilder {
    kernel: Option<Box<dyn Kernel>>,
    acquisition: MultiObjectiveAcquisition,
    n_objectives: usize,
    reference_point: Option<Array1<f64>>,
    scalarization: Option<ScalarizationMethod>,
    n_candidates: usize,
    alpha: f64,
}

impl MultiObjectiveBuilder {
    pub fn new() -> Self {
        Self {
            kernel: None,
            acquisition: MultiObjectiveAcquisition::ExpectedHypervolume,
            n_objectives: 2,
            reference_point: None,
            scalarization: None,
            n_candidates: 1000,
            alpha: 1e-6,
        }
    }

    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    pub fn acquisition_function(mut self, acquisition: MultiObjectiveAcquisition) -> Self {
        self.acquisition = acquisition;
        self
    }

    pub fn n_objectives(mut self, n_objectives: usize) -> Self {
        self.n_objectives = n_objectives;
        self
    }

    pub fn reference_point(mut self, point: Array1<f64>) -> Self {
        self.reference_point = Some(point);
        self
    }

    pub fn scalarization_method(mut self, method: ScalarizationMethod) -> Self {
        self.scalarization = Some(method);
        self
    }

    pub fn n_candidates(mut self, n_candidates: usize) -> Self {
        self.n_candidates = n_candidates;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn build(self) -> MultiObjectiveBayesianOptimizer<Untrained> {
        MultiObjectiveBayesianOptimizer {
            kernel: self.kernel,
            acquisition_function: self.acquisition,
            n_objectives: self.n_objectives,
            reference_point: self.reference_point,
            scalarization_method: self.scalarization,
            n_candidates: self.n_candidates,
            alpha: self.alpha,
            _state: Untrained,
        }
    }
}

impl Estimator for MultiObjectiveBayesianOptimizer<Untrained> {
    type Config = MultiObjectiveConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: MultiObjectiveConfig = MultiObjectiveConfig {
            acquisition: MultiObjectiveAcquisition::ExpectedHypervolume,
            n_objectives: 2,
            n_candidates: 1000,
            regularization: 1e-6,
        };
        &DEFAULT_CONFIG
    }
}

impl Estimator for MultiObjectiveBayesianOptimizer<Trained> {
    type Config = MultiObjectiveConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: MultiObjectiveConfig = MultiObjectiveConfig {
            acquisition: MultiObjectiveAcquisition::ExpectedHypervolume,
            n_objectives: 2,
            n_candidates: 1000,
            regularization: 1e-6,
        };
        &DEFAULT_CONFIG
    }
}

impl Fit<Array2<f64>, Array2<f64>> for MultiObjectiveBayesianOptimizer<Untrained> {
    type Fitted = MultiObjectiveBayesianOptimizer<Trained>;

    fn fit(self, X: &Array2<f64>, Y: &Array2<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != Y.nrows() {
            return Err(SklearsError::DimensionMismatch {
                expected: X.nrows(),
                actual: Y.nrows(),
            });
        }

        if Y.ncols() != self.n_objectives {
            return Err(SklearsError::DimensionMismatch {
                expected: self.n_objectives,
                actual: Y.ncols(),
            });
        }

        let kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        // Create multi-output GP model
        let mo_gp = MultiOutputGaussianProcessRegressor::new()
            .kernel(kernel)
            .alpha(self.alpha)
            .n_outputs(self.n_objectives);

        let fitted_gp = mo_gp.fit(X, Y)?;

        // Create Pareto frontier
        let pareto_frontier = ParetoFrontier::from_data(X.to_owned(), Y.to_owned())?;

        // Set reference point if not provided
        let reference_point = if let Some(ref_point) = self.reference_point {
            ref_point
        } else {
            // Use worst case + margin as reference point
            let mut ref_point = Array1::zeros(self.n_objectives);
            for j in 0..self.n_objectives {
                let max_val = Y.column(j).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                ref_point[j] = max_val + 1.0; // Add margin
            }
            ref_point
        };

        // Compute hypervolume
        let hypervolume = pareto_frontier.hypervolume(&reference_point);

        Ok(MultiObjectiveBayesianOptimizer {
            kernel: self.kernel,
            acquisition_function: self.acquisition_function,
            n_objectives: self.n_objectives,
            reference_point: self.reference_point,
            scalarization_method: self.scalarization_method,
            n_candidates: self.n_candidates,
            alpha: self.alpha,
            _state: Trained {
                gp_models: fitted_gp,
                pareto_frontier,
                hypervolume_indicator: hypervolume,
                reference_point,
                acquisition_function: self.acquisition_function,
                scalarization_method: self.scalarization_method,
            },
        })
    }
}

impl MultiObjectiveBayesianOptimizer<Trained> {
    /// Access the trained state
    pub fn trained_state(&self) -> &Trained {
        &self._state
    }

    /// Get the current Pareto frontier
    pub fn pareto_frontier(&self) -> &ParetoFrontier {
        &self._state.pareto_frontier
    }

    /// Get current hypervolume indicator
    pub fn hypervolume(&self) -> f64 {
        self._state.hypervolume_indicator
    }

    /// Suggest next point to evaluate using multi-objective acquisition function
    pub fn suggest_next_point(&self, candidates: &Array2<f64>) -> SklResult<Array1<f64>> {
        let acquisition_values = self.evaluate_acquisition(candidates)?;

        // Find candidate with maximum acquisition value
        let best_idx = acquisition_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| SklearsError::InvalidInput("No valid candidates".to_string()))?;

        Ok(candidates.row(best_idx).to_owned())
    }

    /// Evaluate acquisition function for candidate points
    pub fn evaluate_acquisition(&self, candidates: &Array2<f64>) -> SklResult<Array1<f64>> {
        let predictions = self._state.gp_models.predict(candidates)?;
        let n_candidates = candidates.nrows();

        match &self._state.acquisition_function {
            MultiObjectiveAcquisition::ExpectedHypervolume => {
                self.expected_hypervolume_improvement(candidates, &predictions)
            }
            MultiObjectiveAcquisition::MultiObjectiveProbabilityOfImprovement { epsilon } => {
                self.multi_objective_probability_of_improvement(candidates, &predictions, *epsilon)
            }
            MultiObjectiveAcquisition::ScalarizedExpectedImprovement { weights } => {
                self.scalarized_expected_improvement(candidates, &predictions, weights)
            }
            MultiObjectiveAcquisition::ParEGO { rho } => {
                self.pareto_ego(candidates, &predictions, *rho)
            }
            MultiObjectiveAcquisition::SMSEGO => self.sms_ego(candidates, &predictions),
            MultiObjectiveAcquisition::MultiObjectiveUCB { beta } => {
                self.multi_objective_ucb(candidates, &predictions, *beta)
            }
            MultiObjectiveAcquisition::MultiObjectiveEntropySearch => {
                // Simplified entropy search implementation
                Ok(Array1::ones(n_candidates))
            }
        }
    }

    /// Expected Hypervolume Improvement
    fn expected_hypervolume_improvement(
        &self,
        _candidates: &Array2<f64>,
        predictions: &Array2<f64>,
    ) -> SklResult<Array1<f64>> {
        let n_candidates = predictions.nrows();
        let mut ehvi_values = Array1::zeros(n_candidates);

        let current_hv = self._state.hypervolume_indicator;

        for i in 0..n_candidates {
            // Simplified EHVI: compute hypervolume improvement if this point were added
            let pred_objectives = predictions.row(i).to_owned();

            // Check if point would be non-dominated
            let mut is_dominated = false;
            let (_, nd_objectives) = self._state.pareto_frontier.non_dominated_points();

            for j in 0..nd_objectives.nrows() {
                if ParetoFrontier::dominates(&nd_objectives.row(j), &pred_objectives.view()) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                // Approximate hypervolume improvement
                let mut volumes = Vec::new();
                for j in 0..self.n_objectives {
                    let margin_to_ref =
                        (self._state.reference_point[j] - pred_objectives[j]).max(0.0);
                    volumes.push(margin_to_ref);
                }
                ehvi_values[i] = volumes.iter().product::<f64>();
            }
        }

        Ok(ehvi_values)
    }

    /// Multi-objective Probability of Improvement
    fn multi_objective_probability_of_improvement(
        &self,
        _candidates: &Array2<f64>,
        predictions: &Array2<f64>,
        epsilon: f64,
    ) -> SklResult<Array1<f64>> {
        let n_candidates = predictions.nrows();
        let mut poi_values = Array1::zeros(n_candidates);

        let (_, nd_objectives) = self._state.pareto_frontier.non_dominated_points();

        for i in 0..n_candidates {
            let pred_objectives = predictions.row(i);

            // Probability of improving upon current Pareto frontier
            let mut improvement_prob: f64 = 1.0;

            for j in 0..nd_objectives.nrows() {
                let nd_point = nd_objectives.row(j);
                let mut prob_better = 1.0;

                for k in 0..self.n_objectives {
                    // Simplified: assume unit variance for probability calculation
                    let improvement = nd_point[k] - pred_objectives[k] - epsilon;
                    let prob_improve_k = 0.5 * (1.0 + (improvement / 1.0).tanh()); // Approximation
                    prob_better *= prob_improve_k;
                }

                improvement_prob = improvement_prob.min(prob_better);
            }

            poi_values[i] = improvement_prob;
        }

        Ok(poi_values)
    }

    /// Scalarized Expected Improvement
    fn scalarized_expected_improvement(
        &self,
        _candidates: &Array2<f64>,
        predictions: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> SklResult<Array1<f64>> {
        let n_candidates = predictions.nrows();
        let mut sei_values = Array1::zeros(n_candidates);

        // Scalarize current best
        let (_, nd_objectives) = self._state.pareto_frontier.non_dominated_points();
        let mut best_scalarized = f64::INFINITY;

        for i in 0..nd_objectives.nrows() {
            let scalarized = nd_objectives
                .row(i)
                .iter()
                .zip(weights.iter())
                .map(|(&obj, &w)| w * obj)
                .sum::<f64>();
            best_scalarized = best_scalarized.min(scalarized);
        }

        for i in 0..n_candidates {
            let pred_scalarized = predictions
                .row(i)
                .iter()
                .zip(weights.iter())
                .map(|(&obj, &w)| w * obj)
                .sum::<f64>();

            // Expected improvement (simplified)
            let improvement = (best_scalarized - pred_scalarized).max(0.0);
            sei_values[i] = improvement;
        }

        Ok(sei_values)
    }

    /// ParEGO acquisition function
    fn pareto_ego(
        &self,
        candidates: &Array2<f64>,
        predictions: &Array2<f64>,
        rho: f64,
    ) -> SklResult<Array1<f64>> {
        // Generate random weights for scalarization
        let mut rng = thread_rng();
        let mut weights = Array1::zeros(self.n_objectives);
        for j in 0..self.n_objectives {
            weights[j] = rng.gen::<f64>();
        }
        let weight_sum = weights.sum();
        weights /= weight_sum;

        // Use augmented Tchebycheff scalarization
        let (_, nd_objectives) = self._state.pareto_frontier.non_dominated_points();
        let mut ideal_point = Array1::from_elem(self.n_objectives, f64::INFINITY);

        for i in 0..nd_objectives.nrows() {
            for j in 0..self.n_objectives {
                ideal_point[j] = ideal_point[j].min(nd_objectives[[i, j]]);
            }
        }

        let scalarization = ScalarizationMethod::AugmentedTchebycheff {
            weights,
            ideal_point,
            rho,
        };

        let mut pareto_values = Array1::zeros(predictions.nrows());
        for i in 0..predictions.nrows() {
            pareto_values[i] = -scalarization.scalarize(&predictions.row(i).to_owned());
            // Negative for maximization
        }

        Ok(pareto_values)
    }

    /// SMS-EGO acquisition function
    fn sms_ego(
        &self,
        _candidates: &Array2<f64>,
        predictions: &Array2<f64>,
    ) -> SklResult<Array1<f64>> {
        let n_candidates = predictions.nrows();
        let mut sms_values = Array1::zeros(n_candidates);

        // S-metric (hypervolume) improvement approximation
        for i in 0..n_candidates {
            let pred_objectives = predictions.row(i).to_owned();

            // Create temporary frontier with this point added
            let mut temp_objectives = self._state.pareto_frontier.objectives.clone();
            let new_obj_row =
                Array2::from_shape_vec((1, pred_objectives.len()), pred_objectives.to_vec())
                    .unwrap();
            temp_objectives = concatenate![Axis(0), temp_objectives, new_obj_row];

            let temp_frontier = ParetoFrontier::from_data(
                Array2::zeros((temp_objectives.nrows(), 1)), // Dummy points
                temp_objectives,
            )?;

            let new_hv = temp_frontier.hypervolume(&self._state.reference_point);
            sms_values[i] = new_hv - self._state.hypervolume_indicator;
        }

        Ok(sms_values)
    }

    /// Multi-objective Upper Confidence Bound
    fn multi_objective_ucb(
        &self,
        _candidates: &Array2<f64>,
        predictions: &Array2<f64>,
        beta: f64,
    ) -> SklResult<Array1<f64>> {
        let n_candidates = predictions.nrows();
        let mut ucb_values = Array1::zeros(n_candidates);

        // Simplified UCB: combine mean predictions with uncertainty
        for i in 0..n_candidates {
            let mean_pred = predictions.row(i).mean().unwrap_or(0.0);
            let uncertainty = beta; // Simplified - would normally compute from GP variance
            ucb_values[i] = mean_pred - uncertainty; // Negative for minimization objectives
        }

        Ok(ucb_values)
    }

    /// Update optimizer with new observation
    pub fn update(
        &mut self,
        new_point: Array1<f64>,
        new_objectives: Array1<f64>,
    ) -> SklResult<bool> {
        // Update Pareto frontier
        let added = self
            ._state
            .pareto_frontier
            .update(new_point, new_objectives.clone());

        if added {
            // Recompute hypervolume
            self._state.hypervolume_indicator = self
                ._state
                .pareto_frontier
                .hypervolume(&self._state.reference_point);
        }

        Ok(added)
    }

    /// Get diversity metric of current Pareto frontier
    pub fn diversity_metric(&self) -> f64 {
        let (_, nd_objectives) = self._state.pareto_frontier.non_dominated_points();

        if nd_objectives.nrows() < 2 {
            return 0.0;
        }

        // Compute average distance between consecutive points
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..nd_objectives.nrows() {
            for j in i + 1..nd_objectives.nrows() {
                let distance = nd_objectives
                    .row(i)
                    .iter()
                    .zip(nd_objectives.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Compute convergence metric
    pub fn convergence_metric(&self, true_pareto_front: Option<&Array2<f64>>) -> f64 {
        if let Some(true_front) = true_pareto_front {
            // Generational Distance: average distance from current front to true front
            let (_, nd_objectives) = self._state.pareto_frontier.non_dominated_points();

            if nd_objectives.is_empty() {
                return f64::INFINITY;
            }

            let mut total_distance = 0.0;
            for i in 0..nd_objectives.nrows() {
                let current_point = nd_objectives.row(i);
                let mut min_distance = f64::INFINITY;

                for j in 0..true_front.nrows() {
                    let true_point = true_front.row(j);
                    let distance = current_point
                        .iter()
                        .zip(true_point.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    min_distance = min_distance.min(distance);
                }

                total_distance += min_distance;
            }

            total_distance / nd_objectives.nrows() as f64
        } else {
            // Without true front, use hypervolume as convergence indicator
            self._state.hypervolume_indicator
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pareto_dominance() {
        let a = array![1.0, 2.0].view();
        let b = array![2.0, 3.0].view();
        let c = array![1.5, 2.5].view();

        assert!(ParetoFrontier::dominates(&a, &b)); // a dominates b
        assert!(!ParetoFrontier::dominates(&b, &a)); // b does not dominate a
        assert!(!ParetoFrontier::dominates(&a, &c)); // a does not dominate c (c is better in second objective)
    }

    #[test]
    fn test_pareto_frontier_creation() {
        let points = array![[1.0], [2.0], [3.0], [4.0]];
        let objectives = array![[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]];

        let frontier = ParetoFrontier::from_data(points, objectives).unwrap();

        // All points should be non-dominated in this case
        assert!(frontier.dominated_indices.is_empty());
    }

    #[test]
    fn test_hypervolume_2d() {
        let points = array![[1.0], [2.0]];
        let objectives = array![[1.0, 3.0], [2.0, 2.0]];
        let reference_point = array![5.0, 5.0];

        let frontier = ParetoFrontier::from_data(points, objectives).unwrap();
        let hypervolume = frontier.hypervolume(&reference_point);

        assert!(hypervolume > 0.0);
    }

    #[test]
    fn test_scalarization_linear_weighted() {
        let weights = array![0.5, 0.5];
        let scalarization = ScalarizationMethod::linear_weighted(weights);

        let objectives = array![2.0, 4.0];
        let scalarized = scalarization.scalarize(&objectives);

        assert_abs_diff_eq!(scalarized, 3.0, epsilon = 1e-10); // 0.5*2 + 0.5*4 = 3
    }

    #[test]
    fn test_scalarization_tchebycheff() {
        let weights = array![1.0, 1.0];
        let ideal_point = array![0.0, 0.0];
        let scalarization = ScalarizationMethod::tchebycheff(weights, ideal_point);

        let objectives = array![2.0, 3.0];
        let scalarized = scalarization.scalarize(&objectives);

        assert_abs_diff_eq!(scalarized, 3.0, epsilon = 1e-10); // max(1*2, 1*3) = 3
    }

    #[test]
    fn test_multi_objective_bo_fit() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .acquisition_function(MultiObjectiveAcquisition::ExpectedHypervolume)
            .n_objectives(2)
            .reference_point(array![5.0, 5.0])
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();

        assert_eq!(trained.n_objectives, 2);
        assert!(trained.hypervolume() > 0.0);
    }

    #[test]
    fn test_acquisition_function_evaluation() {
        let X = array![[1.0], [2.0], [3.0]];
        let Y = array![[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .acquisition_function(MultiObjectiveAcquisition::ExpectedHypervolume)
            .n_objectives(2)
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();

        let candidates = array![[1.5], [2.5]];
        let acquisition_values = trained.evaluate_acquisition(&candidates).unwrap();

        assert_eq!(acquisition_values.len(), candidates.nrows());
        assert!(acquisition_values.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_next_point_suggestion() {
        let X = array![[1.0], [2.0], [3.0]];
        let Y = array![[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .acquisition_function(MultiObjectiveAcquisition::ExpectedHypervolume)
            .n_objectives(2)
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();

        let candidates = array![[0.5], [1.5], [2.5], [3.5]];
        let next_point = trained.suggest_next_point(&candidates).unwrap();

        assert_eq!(next_point.len(), 1);
    }

    #[test]
    fn test_pareto_frontier_update() {
        let points = array![[1.0], [2.0]];
        let objectives = array![[2.0, 3.0], [3.0, 2.0]];

        let mut frontier = ParetoFrontier::from_data(points, objectives).unwrap();

        // Add a dominated point
        let added = frontier.update(array![3.0], array![4.0, 4.0]);
        assert!(!added); // Should not add dominated point

        // Add a non-dominated point
        let added = frontier.update(array![0.5], array![1.0, 1.0]);
        assert!(added); // Should add non-dominated point

        assert_eq!(frontier.points.nrows(), 3);
    }

    #[test]
    fn test_diversity_metric() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let Y = array![[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .n_objectives(2)
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();
        let diversity = trained.diversity_metric();

        assert!(diversity >= 0.0);
    }

    #[test]
    fn test_multi_objective_poi() {
        let X = array![[1.0], [2.0]];
        let Y = array![[1.0, 2.0], [2.0, 1.0]];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .acquisition_function(
                MultiObjectiveAcquisition::MultiObjectiveProbabilityOfImprovement { epsilon: 0.1 },
            )
            .n_objectives(2)
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();

        let candidates = array![[1.5]];
        let poi_values = trained.evaluate_acquisition(&candidates).unwrap();

        assert_eq!(poi_values.len(), 1);
        assert!(poi_values[0] >= 0.0 && poi_values[0] <= 1.0);
    }

    #[test]
    fn test_scalarized_ei() {
        let X = array![[1.0], [2.0]];
        let Y = array![[1.0, 2.0], [2.0, 1.0]];
        let weights = array![0.6, 0.4];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .acquisition_function(MultiObjectiveAcquisition::ScalarizedExpectedImprovement {
                weights,
            })
            .n_objectives(2)
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();

        let candidates = array![[1.5]];
        let sei_values = trained.evaluate_acquisition(&candidates).unwrap();

        assert_eq!(sei_values.len(), 1);
        assert!(sei_values[0] >= 0.0);
    }

    #[test]
    fn test_convergence_metric() {
        let X = array![[1.0], [2.0], [3.0]];
        let Y = array![[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];

        let mobo = MultiObjectiveBayesianOptimizer::builder()
            .kernel(Box::new(RBF::new(1.0)))
            .n_objectives(2)
            .build();

        let trained = mobo.fit(&X, &Y).unwrap();

        // Test without true Pareto front
        let convergence = trained.convergence_metric(None);
        assert!(convergence.is_finite());

        // Test with true Pareto front
        let true_front = array![[0.5, 3.5], [1.5, 2.5], [2.5, 1.5], [3.5, 0.5]];
        let convergence_with_true = trained.convergence_metric(Some(&true_front));
        assert!(convergence_with_true >= 0.0);
    }
}
