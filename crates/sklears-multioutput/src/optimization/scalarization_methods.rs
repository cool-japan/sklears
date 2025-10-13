//! Scalarization Methods for Multi-Objective Optimization
//!
//! This module provides various scalarization techniques for converting multi-objective
//! optimization problems into single-objective problems. These methods enable the systematic
//! exploration of trade-offs between conflicting objectives.
//!
//! ## Key Features
//!
//! - **Weighted Sum Method**: Simple linear combination of objectives with user-defined weights
//! - **Epsilon-Constraint Method**: Optimize one objective while constraining others
//! - **Achievement Scalarizing Function**: Reference point-based optimization
//! - **Augmented Weighted Tchebycheff**: Improved Tchebycheff scalarization
//! - **Normalized Normal Constraint**: Advanced constraint handling for Pareto front generation
//! - **Problem Generation**: Systematic generation of scalarized subproblems

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use scirs2_core::random::RandNormal;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Scalarization method types for multi-objective optimization
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarizationMethod {
    /// Weighted sum method with objective weights
    WeightedSum(Vec<Float>),
    /// Epsilon-constraint method with constraints and objective index
    EpsilonConstraint {
        objective_index: usize,
        epsilon_values: Vec<Float>,
    },
    /// Achievement scalarizing function with reference point
    AchievementScalarizingFunction {
        reference_point: Vec<Float>,
        augmentation_coefficient: Float,
    },
    /// Augmented weighted Tchebycheff method
    AugmentedWeightedTchebycheff {
        reference_point: Vec<Float>,
        weights: Vec<Float>,
        augmentation_coefficient: Float,
    },
    /// Normalized normal constraint method
    NormalizedNormalConstraint {
        anchor_points: Array2<Float>,
        utopia_point: Vec<Float>,
    },
}

/// Configuration for scalarization optimizer
#[derive(Debug, Clone)]
pub struct ScalarizationConfig {
    /// Scalarization method to use
    pub method: ScalarizationMethod,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for ScalarizationConfig {
    fn default() -> Self {
        Self {
            method: ScalarizationMethod::WeightedSum(vec![0.5, 0.5]),
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            random_state: None,
        }
    }
}

/// Scalarization optimizer for converting multi-objective problems to single-objective
#[derive(Debug, Clone)]
pub struct ScalarizationOptimizer<S = Untrained> {
    state: S,
    config: ScalarizationConfig,
}

/// Trained state for scalarization optimizer
#[derive(Debug, Clone)]
pub struct ScalarizationOptimizerTrained {
    /// Model parameters
    pub parameters: Array1<Float>,
    /// Scalarized objective value
    pub scalarized_value: Float,
    /// Original objective values
    pub objective_values: Vec<Float>,
    /// Convergence history
    pub convergence_history: Vec<Float>,
    /// Number of features
    pub n_features: usize,
    /// Number of objectives
    pub n_objectives: usize,
    /// Configuration used
    pub config: ScalarizationConfig,
}

impl ScalarizationOptimizer<Untrained> {
    /// Create a new scalarization optimizer
    pub fn new(method: ScalarizationMethod) -> Self {
        Self {
            state: Untrained,
            config: ScalarizationConfig {
                method,
                ..Default::default()
            },
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: ScalarizationConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Compute scalarized objective value
    pub fn scalarize_objectives(&self, objectives: &[Float]) -> SklResult<Float> {
        match &self.config.method {
            ScalarizationMethod::WeightedSum(weights) => {
                if weights.len() != objectives.len() {
                    return Err(SklearsError::InvalidInput(
                        "Weight vector length must match number of objectives".to_string(),
                    ));
                }
                Ok(objectives
                    .iter()
                    .zip(weights.iter())
                    .map(|(obj, w)| obj * w)
                    .sum())
            }

            ScalarizationMethod::EpsilonConstraint {
                objective_index,
                epsilon_values,
            } => {
                if *objective_index >= objectives.len() {
                    return Err(SklearsError::InvalidInput(
                        "Objective index out of bounds".to_string(),
                    ));
                }
                if epsilon_values.len() != objectives.len() - 1 {
                    return Err(SklearsError::InvalidInput(
                        "Epsilon values length must be objectives - 1".to_string(),
                    ));
                }

                // Check epsilon constraints
                let mut eps_idx = 0;
                for (i, &obj_val) in objectives.iter().enumerate() {
                    if i != *objective_index {
                        if obj_val > epsilon_values[eps_idx] {
                            return Ok(Float::INFINITY); // Constraint violated
                        }
                        eps_idx += 1;
                    }
                }

                Ok(objectives[*objective_index])
            }

            ScalarizationMethod::AchievementScalarizingFunction {
                reference_point,
                augmentation_coefficient,
            } => {
                if reference_point.len() != objectives.len() {
                    return Err(SklearsError::InvalidInput(
                        "Reference point length must match number of objectives".to_string(),
                    ));
                }

                let max_normalized_diff = objectives
                    .iter()
                    .zip(reference_point.iter())
                    .map(|(obj, ref_pt)| (obj - ref_pt).max(0.0))
                    .fold(0.0, Float::max);

                let augmentation_term = augmentation_coefficient
                    * objectives
                        .iter()
                        .zip(reference_point.iter())
                        .map(|(obj, ref_pt)| obj - ref_pt)
                        .sum::<Float>();

                Ok(max_normalized_diff + augmentation_term)
            }

            ScalarizationMethod::AugmentedWeightedTchebycheff {
                reference_point,
                weights,
                augmentation_coefficient,
            } => {
                if reference_point.len() != objectives.len() || weights.len() != objectives.len() {
                    return Err(SklearsError::InvalidInput(
                        "Reference point and weights length must match number of objectives"
                            .to_string(),
                    ));
                }

                let max_weighted_diff = objectives
                    .iter()
                    .zip(reference_point.iter())
                    .zip(weights.iter())
                    .map(|((obj, ref_pt), w)| w * (obj - ref_pt).abs())
                    .fold(0.0, Float::max);

                let augmentation_term = augmentation_coefficient
                    * objectives
                        .iter()
                        .zip(reference_point.iter())
                        .map(|(obj, ref_pt)| obj - ref_pt)
                        .sum::<Float>();

                Ok(max_weighted_diff + augmentation_term)
            }

            ScalarizationMethod::NormalizedNormalConstraint {
                anchor_points,
                utopia_point,
            } => {
                if utopia_point.len() != objectives.len() {
                    return Err(SklearsError::InvalidInput(
                        "Utopia point length must match number of objectives".to_string(),
                    ));
                }

                // Normalize objectives
                let normalized_objectives: Vec<Float> = objectives
                    .iter()
                    .zip(utopia_point.iter())
                    .enumerate()
                    .map(|(i, (obj, utopia))| {
                        let anchor = anchor_points[[i, i]];
                        if (anchor - utopia).abs() > 1e-10 {
                            (obj - utopia) / (anchor - utopia)
                        } else {
                            0.0
                        }
                    })
                    .collect();

                // Compute Euclidean distance from origin
                Ok(normalized_objectives
                    .iter()
                    .map(|x| x * x)
                    .sum::<Float>()
                    .sqrt())
            }
        }
    }

    /// Generate multiple scalarized problems for comprehensive optimization
    pub fn generate_scalarized_problems(
        &self,
        n_problems: usize,
        n_objectives: usize,
    ) -> SklResult<Vec<ScalarizationMethod>> {
        let mut problems = Vec::new();

        match &self.config.method {
            ScalarizationMethod::WeightedSum(_) => {
                // Generate uniformly distributed weight vectors
                for i in 0..n_problems {
                    let mut weights = vec![0.0; n_objectives];
                    let step = 1.0 / (n_problems - 1) as Float;
                    weights[0] = i as Float * step;
                    weights[1] = 1.0 - weights[0];

                    // Extend to higher dimensions using Dirichlet-like distribution
                    if n_objectives > 2 {
                        let remaining = weights[1];
                        weights[1] = remaining * (i as Float / n_problems as Float);
                        for j in 2..n_objectives {
                            weights[j] = remaining / (n_objectives - 1) as Float;
                        }
                    }

                    problems.push(ScalarizationMethod::WeightedSum(weights));
                }
            }

            ScalarizationMethod::EpsilonConstraint {
                objective_index, ..
            } => {
                // Generate different epsilon values
                for i in 0..n_problems {
                    let step = 1.0 / n_problems as Float;
                    let epsilon_values = (0..n_objectives - 1)
                        .map(|_| (i as Float + 1.0) * step)
                        .collect();

                    problems.push(ScalarizationMethod::EpsilonConstraint {
                        objective_index: *objective_index,
                        epsilon_values,
                    });
                }
            }

            _ => {
                // For other methods, generate variations of the current method
                for _ in 0..n_problems {
                    problems.push(self.config.method.clone());
                }
            }
        }

        Ok(problems)
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for ScalarizationOptimizer<Untrained> {
    type Fitted = ScalarizationOptimizer<ScalarizationOptimizerTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let (y_samples, n_objectives) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let mut rng = thread_rng();

        // Initialize parameters
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        let mut parameters = Array1::<Float>::zeros(n_features * n_objectives);
        for i in 0..(n_features * n_objectives) {
            parameters[i] = rng.sample(normal_dist);
        }

        let mut convergence_history = Vec::new();
        let mut prev_scalarized_value = Float::INFINITY;

        for iteration in 0..self.config.max_iter {
            // Compute current objectives (simplified for demonstration)
            let objectives: Vec<Float> = (0..n_objectives)
                .map(|i| {
                    let param_slice = parameters.slice(s![i * n_features..(i + 1) * n_features]);
                    // Simplified objective computation - in practice this would be problem-specific
                    param_slice.iter().map(|x| x * x).sum::<Float>() / n_features as Float
                })
                .collect();

            // Scalarize objectives
            let scalarized_value = self.scalarize_objectives(&objectives)?;
            convergence_history.push(scalarized_value);

            // Check convergence
            if (prev_scalarized_value - scalarized_value).abs() < self.config.tol {
                break;
            }
            prev_scalarized_value = scalarized_value;

            // Simplified gradient descent update
            for i in 0..parameters.len() {
                let gradient = 2.0 * parameters[i] / n_features as Float; // Simplified gradient
                parameters[i] -= self.config.learning_rate * gradient;
            }
        }

        // Final objective computation
        let final_objectives: Vec<Float> = (0..n_objectives)
            .map(|i| {
                let param_slice = parameters.slice(s![i * n_features..(i + 1) * n_features]);
                param_slice.iter().map(|x| x * x).sum::<Float>() / n_features as Float
            })
            .collect();

        let final_scalarized_value = self.scalarize_objectives(&final_objectives)?;

        Ok(ScalarizationOptimizer {
            state: ScalarizationOptimizerTrained {
                parameters,
                scalarized_value: final_scalarized_value,
                objective_values: final_objectives,
                convergence_history,
                n_features,
                n_objectives,
                config: self.config.clone(),
            },
            config: self.config,
        })
    }
}

impl ScalarizationOptimizer<ScalarizationOptimizerTrained> {
    /// Get the optimized parameters
    pub fn parameters(&self) -> &Array1<Float> {
        &self.state.parameters
    }

    /// Get the scalarized objective value
    pub fn scalarized_value(&self) -> Float {
        self.state.scalarized_value
    }

    /// Get the original objective values
    pub fn objective_values(&self) -> &[Float] {
        &self.state.objective_values
    }

    /// Get the convergence history
    pub fn convergence_history(&self) -> &[Float] {
        &self.state.convergence_history
    }

    /// Get the scalarization method used
    pub fn method(&self) -> &ScalarizationMethod {
        &self.state.config.method
    }

    /// Check if the solution is feasible (for constraint-based methods)
    pub fn is_feasible(&self) -> bool {
        match &self.state.config.method {
            ScalarizationMethod::EpsilonConstraint {
                epsilon_values,
                objective_index,
            } => {
                let mut eps_idx = 0;
                for (i, &obj_val) in self.state.objective_values.iter().enumerate() {
                    if i != *objective_index {
                        if obj_val > epsilon_values[eps_idx] {
                            return false;
                        }
                        eps_idx += 1;
                    }
                }
                true
            }
            _ => true, // Other methods don't have hard constraints
        }
    }
}

impl Estimator for ScalarizationOptimizer<Untrained> {
    type Config = ScalarizationConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for ScalarizationOptimizer<ScalarizationOptimizerTrained> {
    type Config = ScalarizationConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.state.config
    }
}
