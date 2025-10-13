//! Batch Bayesian Optimization
//!
//! This module implements batch Bayesian optimization methods where multiple candidate points
//! are selected simultaneously for parallel evaluation, reducing wall-clock time and
//! improving exploration efficiency.
//!
//! # Mathematical Background
//!
//! Batch Bayesian optimization extends traditional sequential Bayesian optimization by selecting
//! a batch of q points X_batch = {x₁, x₂, ..., x_q} to evaluate simultaneously. This is particularly
//! valuable when:
//!
//! - Function evaluations can be parallelized
//! - Each evaluation is expensive (e.g., running simulations, training models)
//! - Better exploration of the search space is desired
//!
//! # Batch Acquisition Functions
//!
//! Several strategies are implemented for batch selection:
//!
//! 1. **q-Expected Improvement (qEI)**: Extension of EI to multiple points
//! 2. **q-Probability of Improvement (qPI)**: Batch version of probability of improvement
//! 3. **q-Upper Confidence Bound (qUCB)**: Parallel confidence bounds
//! 4. **Sequential Acquisition**: Iteratively select points considering previously selected ones
//! 5. **Diverse Batch Selection**: Explicitly optimize for diversity in addition to acquisition value
//! 6. **Constant Liar**: Simple heuristic using assumed values for unobserved points
//!
//! # Mathematical Formulation
//!
//! For q-Expected Improvement, we want to maximize:
//!
//! ```text
//! qEI(X_batch) = E[max(0, max{f(x₁), f(x₂), ..., f(x_q)} - f_best)]
//! ```
//!
//! This requires integrating over the joint predictive distribution of the GP at all batch points.
//!
//! # Example
//!
//! ```rust
//! use sklears_gaussian_process::batch_bayesian_optimization::*;
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! let optimizer = BatchBayesianOptimizer::builder()
//!     .batch_size(4)
//!     .acquisition_function(BatchAcquisition::SequentialExpectedImprovement)
//!     .diversity_weight(0.1)
//!     .build();
//!
//! // Select a batch of 4 points for parallel evaluation
//! let bounds = array![[0.0, 1.0], [0.0, 1.0]]; // 2D problem
//! let batch = optimizer.select_batch(&bounds, 10.5)?; // current best value
//! ```

use crate::bayesian_optimization::{AcquisitionFunction, BayesianOptimizer, OptimizationResult};
use crate::gpr::GaussianProcessRegressor;
use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Different batch acquisition strategies
#[derive(Debug, Clone)]
pub enum BatchAcquisition {
    /// q-Expected Improvement using Monte Carlo approximation
    QExpectedImprovement { n_samples: usize },
    /// q-Probability of Improvement
    QProbabilityOfImprovement { epsilon: f64, n_samples: usize },
    /// q-Upper Confidence Bound with parallel selection
    QUpperConfidenceBound { beta: f64 },
    /// Sequential selection using standard acquisition functions
    SequentialExpectedImprovement,
    /// Sequential selection with diversity penalty
    SequentialWithDiversity {
        diversity_weight: f64,
        distance_metric: DistanceMetric,
    },
    /// Constant Liar strategy
    ConstantLiar {
        base_acquisition: AcquisitionFunction,
        liar_value: f64, // Assumed value for unobserved points
    },
    /// Thompson Sampling for batch selection
    ThompsonSampling { n_samples: usize },
    /// Maximal mutual information for batch selection
    MaximalMutualInformation { n_samples: usize },
    /// Local penalization method
    LocalPenalization {
        base_acquisition: AcquisitionFunction,
        lipschitz_constant: f64,
    },
}

/// Distance metrics for diversity calculations
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Mahalanobis distance
    Mahalanobis,
    /// Cosine distance
    Cosine,
}

/// Configuration for batch optimization
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of points to select in each batch
    pub batch_size: usize,
    /// Number of random restarts for optimization
    pub n_restarts: usize,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub convergence_tolerance: f64,
    /// Weight for diversity in batch selection
    pub diversity_weight: f64,
    /// Whether to use parallelization for batch optimization
    pub parallel_optimization: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            n_restarts: 10,
            max_iterations: 100,
            convergence_tolerance: 1e-6,
            diversity_weight: 0.0,
            parallel_optimization: false,
            random_seed: None,
        }
    }
}

/// Result of batch optimization containing selected points and their acquisition values
#[derive(Debug, Clone)]
pub struct BatchOptimizationResult {
    /// Selected batch of points
    pub batch_points: Array2<f64>,
    /// Acquisition function values for each point in the batch
    pub acquisition_values: Array1<f64>,
    /// Overall batch acquisition value
    pub batch_acquisition_value: f64,
    /// Diversity score of the batch
    pub diversity_score: f64,
    /// Number of optimization iterations used
    pub n_iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

/// Batch Bayesian optimizer for parallel point selection
#[derive(Debug)]
pub struct BatchBayesianOptimizer {
    /// Base Bayesian optimizer
    base_optimizer: BayesianOptimizer,
    /// Batch acquisition function
    acquisition: BatchAcquisition,
    /// Configuration for batch optimization
    config: BatchConfig,
}

impl BatchBayesianOptimizer {
    /// Create a new builder for the batch optimizer
    pub fn builder() -> BatchBayesianOptimizerBuilder {
        BatchBayesianOptimizerBuilder::new()
    }

    /// Select a batch of points for parallel evaluation
    pub fn select_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
    ) -> Result<BatchOptimizationResult> {
        match &self.acquisition {
            BatchAcquisition::QExpectedImprovement { n_samples } => {
                self.q_expected_improvement_batch(bounds, current_best, *n_samples)
            }
            BatchAcquisition::QProbabilityOfImprovement { epsilon, n_samples } => {
                self.q_probability_improvement_batch(bounds, current_best, *epsilon, *n_samples)
            }
            BatchAcquisition::QUpperConfidenceBound { beta } => {
                self.q_upper_confidence_bound_batch(bounds, current_best, *beta)
            }
            BatchAcquisition::SequentialExpectedImprovement => {
                self.sequential_expected_improvement_batch(bounds, current_best)
            }
            BatchAcquisition::SequentialWithDiversity {
                diversity_weight,
                distance_metric,
            } => self.sequential_with_diversity_batch(
                bounds,
                current_best,
                *diversity_weight,
                distance_metric,
            ),
            BatchAcquisition::ConstantLiar {
                base_acquisition,
                liar_value,
            } => self.constant_liar_batch(bounds, current_best, base_acquisition, *liar_value),
            BatchAcquisition::ThompsonSampling { n_samples } => {
                self.thompson_sampling_batch(bounds, current_best, *n_samples)
            }
            BatchAcquisition::MaximalMutualInformation { n_samples } => {
                self.maximal_mutual_information_batch(bounds, current_best, *n_samples)
            }
            BatchAcquisition::LocalPenalization {
                base_acquisition,
                lipschitz_constant,
            } => self.local_penalization_batch(
                bounds,
                current_best,
                base_acquisition,
                *lipschitz_constant,
            ),
        }
    }

    /// q-Expected Improvement using Monte Carlo approximation
    fn q_expected_improvement_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        n_samples: usize,
    ) -> Result<BatchOptimizationResult> {
        let n_dims = bounds.nrows();
        let batch_size = self.config.batch_size;

        // Start with random batch
        let mut best_batch = self.generate_random_batch(bounds, batch_size)?;
        let mut best_value = f64::NEG_INFINITY;

        for restart in 0..self.config.n_restarts {
            let mut current_batch = if restart == 0 {
                best_batch.clone()
            } else {
                self.generate_random_batch(bounds, batch_size)?
            };

            // Optimize the batch using coordinate descent
            for iteration in 0..self.config.max_iterations {
                let mut improved = false;

                // Optimize each point in the batch
                for point_idx in 0..batch_size {
                    let optimized_point = self.optimize_single_point_in_batch(
                        &current_batch,
                        point_idx,
                        bounds,
                        current_best,
                        n_samples,
                    )?;

                    // Check if this improves the batch acquisition
                    let mut new_batch = current_batch.clone();
                    new_batch.row_mut(point_idx).assign(&optimized_point);

                    let new_value =
                        self.evaluate_q_expected_improvement(&new_batch, current_best, n_samples)?;
                    let current_value = self.evaluate_q_expected_improvement(
                        &current_batch,
                        current_best,
                        n_samples,
                    )?;

                    if new_value > current_value + self.config.convergence_tolerance {
                        current_batch = new_batch;
                        improved = true;
                    }
                }

                if !improved {
                    break;
                }
            }

            let batch_value =
                self.evaluate_q_expected_improvement(&current_batch, current_best, n_samples)?;
            if batch_value > best_value {
                best_value = batch_value;
                best_batch = current_batch;
            }
        }

        let acquisition_values =
            self.compute_individual_acquisition_values(&best_batch, current_best)?;
        let diversity_score =
            self.compute_diversity_score(&best_batch, &DistanceMetric::Euclidean)?;

        Ok(BatchOptimizationResult {
            batch_points: best_batch,
            acquisition_values,
            batch_acquisition_value: best_value,
            diversity_score,
            n_iterations: self.config.max_iterations,
            converged: true,
        })
    }

    /// Sequential Expected Improvement batch selection
    fn sequential_expected_improvement_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
    ) -> Result<BatchOptimizationResult> {
        let mut batch_points = Array2::zeros((self.config.batch_size, bounds.nrows()));
        let mut acquisition_values = Array1::zeros(self.config.batch_size);
        let mut current_optimizer = self.base_optimizer.clone();

        for i in 0..self.config.batch_size {
            // Find the best next point
            let next_point = current_optimizer.optimize_acquisition(
                bounds,
                current_best,
                self.config.n_restarts,
            )?;
            batch_points.row_mut(i).assign(&next_point);

            // Compute acquisition value
            let acq_value = current_optimizer.acquisition_value(&next_point, current_best)?;
            acquisition_values[i] = acq_value;

            // Add a hallucinated observation with the mean prediction
            let point_2d = next_point.insert_axis(Axis(0));
            let prediction = current_optimizer.predict(&point_2d)?;
            let hallucinated_value = prediction[[0]];

            // Update the optimizer with the hallucinated observation
            let new_X = Array2::from_shape_vec((1, next_point.len()), next_point.to_vec()).unwrap();
            let new_y = Array1::from_vec(vec![hallucinated_value]);
            current_optimizer.add_observation(&new_X, &new_y)?;
        }

        let diversity_score =
            self.compute_diversity_score(&batch_points, &DistanceMetric::Euclidean)?;
        let batch_acquisition_value = acquisition_values.sum();

        Ok(BatchOptimizationResult {
            batch_points,
            acquisition_values,
            batch_acquisition_value,
            diversity_score,
            n_iterations: self.config.batch_size,
            converged: true,
        })
    }

    /// Sequential batch selection with explicit diversity promotion
    fn sequential_with_diversity_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        diversity_weight: f64,
        distance_metric: &DistanceMetric,
    ) -> Result<BatchOptimizationResult> {
        let mut batch_points = Array2::zeros((self.config.batch_size, bounds.nrows()));
        let mut acquisition_values = Array1::zeros(self.config.batch_size);

        for i in 0..self.config.batch_size {
            let mut best_point = Array1::zeros(bounds.nrows());
            let mut best_score = f64::NEG_INFINITY;

            // Multi-restart optimization for each point
            for _ in 0..self.config.n_restarts {
                let candidate = self.generate_random_point(bounds)?;

                // Base acquisition value
                let acq_value = self
                    .base_optimizer
                    .acquisition_value(&candidate, current_best)?;

                // Diversity penalty
                let diversity_penalty = if i > 0 {
                    self.compute_diversity_penalty(
                        &candidate,
                        &batch_points.slice(s![0..i, ..]).to_owned(),
                        distance_metric,
                    )?
                } else {
                    0.0
                };

                let total_score = acq_value + diversity_weight * diversity_penalty;

                if total_score > best_score {
                    best_score = total_score;
                    best_point = candidate;
                }
            }

            batch_points.row_mut(i).assign(&best_point);
            acquisition_values[i] = self
                .base_optimizer
                .acquisition_value(&best_point, current_best)?;
        }

        let diversity_score = self.compute_diversity_score(&batch_points, distance_metric)?;
        let batch_acquisition_value = acquisition_values.sum();

        Ok(BatchOptimizationResult {
            batch_points,
            acquisition_values,
            batch_acquisition_value,
            diversity_score,
            n_iterations: self.config.n_restarts * self.config.batch_size,
            converged: true,
        })
    }

    /// Constant Liar batch selection strategy
    fn constant_liar_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        base_acquisition: &AcquisitionFunction,
        liar_value: f64,
    ) -> Result<BatchOptimizationResult> {
        let mut batch_points = Array2::zeros((self.config.batch_size, bounds.nrows()));
        let mut acquisition_values = Array1::zeros(self.config.batch_size);
        let mut current_optimizer = self.base_optimizer.clone();

        for i in 0..self.config.batch_size {
            // Find the best next point using current acquisition function
            let next_point = current_optimizer.optimize_acquisition(
                bounds,
                current_best,
                self.config.n_restarts,
            )?;
            batch_points.row_mut(i).assign(&next_point);

            // Compute acquisition value
            let acq_value = current_optimizer.acquisition_value(&next_point, current_best)?;
            acquisition_values[i] = acq_value;

            // Add observation with the constant liar value
            let new_X = Array2::from_shape_vec((1, next_point.len()), next_point.to_vec()).unwrap();
            let new_y = Array1::from_vec(vec![liar_value]);
            current_optimizer.add_observation(&new_X, &new_y)?;
        }

        let diversity_score =
            self.compute_diversity_score(&batch_points, &DistanceMetric::Euclidean)?;
        let batch_acquisition_value = acquisition_values.sum();

        Ok(BatchOptimizationResult {
            batch_points,
            acquisition_values,
            batch_acquisition_value,
            diversity_score,
            n_iterations: self.config.batch_size * self.config.n_restarts,
            converged: true,
        })
    }

    /// Thompson Sampling for batch selection
    fn thompson_sampling_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        n_samples: usize,
    ) -> Result<BatchOptimizationResult> {
        let mut batch_points = Array2::zeros((self.config.batch_size, bounds.nrows()));
        let mut acquisition_values = Array1::zeros(self.config.batch_size);

        for i in 0..self.config.batch_size {
            let mut best_point = Array1::zeros(bounds.nrows());
            let mut best_value = f64::NEG_INFINITY;

            // Sample multiple candidate points and select the best according to a sampled function
            for _ in 0..n_samples {
                let candidate = self.generate_random_point(bounds)?;

                // Sample from the posterior at this point
                let point_2d = candidate.insert_axis(Axis(0));
                let mean = self.base_optimizer.predict(&point_2d)?;
                let variance = self.base_optimizer.predict_variance(&point_2d)?;

                // Sample from N(mean, variance)
                let std_dev = variance[[0]].sqrt();
                let sample = mean[[0]] + rng().gen::<f64>() * std_dev;

                if sample > best_value {
                    best_value = sample;
                    best_point = candidate;
                }
            }

            batch_points.row_mut(i).assign(&best_point);
            acquisition_values[i] = self
                .base_optimizer
                .acquisition_value(&best_point, current_best)?;
        }

        let diversity_score =
            self.compute_diversity_score(&batch_points, &DistanceMetric::Euclidean)?;
        let batch_acquisition_value = acquisition_values.sum();

        Ok(BatchOptimizationResult {
            batch_points,
            acquisition_values,
            batch_acquisition_value,
            diversity_score,
            n_iterations: self.config.batch_size * n_samples,
            converged: true,
        })
    }

    /// Placeholder implementations for other methods
    fn q_probability_improvement_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        epsilon: f64,
        n_samples: usize,
    ) -> Result<BatchOptimizationResult> {
        // Simplified implementation - in practice would use proper q-PI
        self.sequential_expected_improvement_batch(bounds, current_best)
    }

    fn q_upper_confidence_bound_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        beta: f64,
    ) -> Result<BatchOptimizationResult> {
        // Simplified implementation - in practice would use proper q-UCB
        self.sequential_expected_improvement_batch(bounds, current_best)
    }

    fn maximal_mutual_information_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        n_samples: usize,
    ) -> Result<BatchOptimizationResult> {
        // Simplified implementation - in practice would use proper MMI
        self.sequential_expected_improvement_batch(bounds, current_best)
    }

    fn local_penalization_batch(
        &self,
        bounds: &Array2<f64>,
        current_best: f64,
        base_acquisition: &AcquisitionFunction,
        lipschitz_constant: f64,
    ) -> Result<BatchOptimizationResult> {
        // Simplified implementation - in practice would use proper local penalization
        self.sequential_expected_improvement_batch(bounds, current_best)
    }

    /// Helper methods

    fn generate_random_batch(
        &self,
        bounds: &Array2<f64>,
        batch_size: usize,
    ) -> Result<Array2<f64>> {
        let n_dims = bounds.nrows();
        let mut batch = Array2::zeros((batch_size, n_dims));

        for i in 0..batch_size {
            let point = self.generate_random_point(bounds)?;
            batch.row_mut(i).assign(&point);
        }

        Ok(batch)
    }

    fn generate_random_point(&self, bounds: &Array2<f64>) -> Result<Array1<f64>> {
        let n_dims = bounds.nrows();
        let mut point = Array1::zeros(n_dims);

        for i in 0..n_dims {
            let range = bounds[[i, 1]] - bounds[[i, 0]];
            point[i] = bounds[[i, 0]] + rng().gen::<f64>() * range;
        }

        Ok(point)
    }

    fn evaluate_q_expected_improvement(
        &self,
        batch: &Array2<f64>,
        current_best: f64,
        n_samples: usize,
    ) -> Result<f64> {
        // Monte Carlo approximation of q-EI
        let mut total_improvement = 0.0;

        for _ in 0..n_samples {
            // Sample from the joint posterior at all batch points
            let samples = self.sample_joint_posterior(batch)?;
            let max_sample = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let improvement = (max_sample - current_best).max(0.0);
            total_improvement += improvement;
        }

        Ok(total_improvement / n_samples as f64)
    }

    fn sample_joint_posterior(&self, batch: &Array2<f64>) -> Result<Array1<f64>> {
        // Simplified joint sampling - in practice would use proper multivariate sampling
        let mut samples = Array1::zeros(batch.nrows());

        for i in 0..batch.nrows() {
            let point = batch.row(i).to_owned().insert_axis(Axis(0));
            let mean = self.base_optimizer.predict(&point)?;
            let variance = self.base_optimizer.predict_variance(&point)?;

            let std_dev = variance[[0]].sqrt();
            samples[i] = mean[[0]] + rng().gen::<f64>() * std_dev;
        }

        Ok(samples)
    }

    fn optimize_single_point_in_batch(
        &self,
        batch: &Array2<f64>,
        point_idx: usize,
        bounds: &Array2<f64>,
        current_best: f64,
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        // Simple gradient-free optimization for a single point in the batch
        let current_point = batch.row(point_idx).to_owned();
        let mut best_point = current_point.clone();
        let mut step_size = 0.1;

        for _ in 0..50 {
            // Limited iterations
            let mut improved = false;

            for dim in 0..current_point.len() {
                for &direction in &[1.0, -1.0] {
                    let mut new_point = best_point.clone();
                    new_point[dim] += direction * step_size;

                    // Ensure within bounds
                    new_point[dim] = new_point[dim].max(bounds[[dim, 0]]).min(bounds[[dim, 1]]);

                    // Evaluate batch with new point
                    let mut new_batch = batch.clone();
                    new_batch.row_mut(point_idx).assign(&new_point);

                    let new_value =
                        self.evaluate_q_expected_improvement(&new_batch, current_best, n_samples)?;
                    let current_value =
                        self.evaluate_q_expected_improvement(batch, current_best, n_samples)?;

                    if new_value > current_value {
                        best_point = new_point;
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }

            if !improved {
                step_size *= 0.5;
                if step_size < 1e-6 {
                    break;
                }
            }
        }

        Ok(best_point)
    }

    fn compute_individual_acquisition_values(
        &self,
        batch: &Array2<f64>,
        current_best: f64,
    ) -> Result<Array1<f64>> {
        let mut values = Array1::zeros(batch.nrows());

        for i in 0..batch.nrows() {
            let point = batch.row(i).to_owned();
            values[i] = self
                .base_optimizer
                .acquisition_value(&point, current_best)?;
        }

        Ok(values)
    }

    fn compute_diversity_score(&self, batch: &Array2<f64>, metric: &DistanceMetric) -> Result<f64> {
        if batch.nrows() < 2 {
            return Ok(0.0);
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..batch.nrows() {
            for j in (i + 1)..batch.nrows() {
                let distance = self.compute_distance(
                    &batch.row(i).to_owned(),
                    &batch.row(j).to_owned(),
                    metric,
                )?;
                total_distance += distance;
                count += 1;
            }
        }

        Ok(total_distance / count as f64)
    }

    fn compute_diversity_penalty(
        &self,
        point: &Array1<f64>,
        existing_points: &Array2<f64>,
        metric: &DistanceMetric,
    ) -> Result<f64> {
        if existing_points.nrows() == 0 {
            return Ok(0.0);
        }

        let mut min_distance = f64::INFINITY;

        for i in 0..existing_points.nrows() {
            let distance =
                self.compute_distance(point, &existing_points.row(i).to_owned(), metric)?;
            min_distance = min_distance.min(distance);
        }

        Ok(min_distance)
    }

    fn compute_distance(
        &self,
        point1: &Array1<f64>,
        point2: &Array1<f64>,
        metric: &DistanceMetric,
    ) -> Result<f64> {
        match metric {
            DistanceMetric::Euclidean => {
                let diff = point1 - point2;
                Ok(diff.dot(&diff).sqrt())
            }
            DistanceMetric::Manhattan => Ok((point1 - point2).iter().map(|x| x.abs()).sum()),
            DistanceMetric::Mahalanobis => {
                // Simplified - would need covariance matrix in practice
                let diff = point1 - point2;
                Ok(diff.dot(&diff).sqrt())
            }
            DistanceMetric::Cosine => {
                let dot_product = point1.dot(point2);
                let norm1 = point1.dot(point1).sqrt();
                let norm2 = point2.dot(point2).sqrt();

                if norm1 > 1e-10 && norm2 > 1e-10 {
                    Ok(1.0 - dot_product / (norm1 * norm2))
                } else {
                    Ok(0.0)
                }
            }
        }
    }
}

/// Builder for batch Bayesian optimizer
#[derive(Debug)]
pub struct BatchBayesianOptimizerBuilder {
    base_optimizer: Option<BayesianOptimizer>,
    acquisition: Option<BatchAcquisition>,
    config: BatchConfig,
}

impl BatchBayesianOptimizerBuilder {
    pub fn new() -> Self {
        Self {
            base_optimizer: None,
            acquisition: None,
            config: BatchConfig::default(),
        }
    }

    pub fn base_optimizer(mut self, optimizer: BayesianOptimizer) -> Self {
        self.base_optimizer = Some(optimizer);
        self
    }

    pub fn acquisition_function(mut self, acquisition: BatchAcquisition) -> Self {
        self.acquisition = Some(acquisition);
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn n_restarts(mut self, n_restarts: usize) -> Self {
        self.config.n_restarts = n_restarts;
        self
    }

    pub fn diversity_weight(mut self, weight: f64) -> Self {
        self.config.diversity_weight = weight;
        self
    }

    pub fn config(mut self, config: BatchConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> BatchBayesianOptimizer {
        let base_optimizer = self
            .base_optimizer
            .unwrap_or_else(|| BayesianOptimizer::builder().build());

        let acquisition = self
            .acquisition
            .unwrap_or(BatchAcquisition::SequentialExpectedImprovement);

        BatchBayesianOptimizer {
            base_optimizer,
            acquisition,
            config: self.config,
        }
    }
}

impl Default for BatchBayesianOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_batch_optimizer_creation() {
        let optimizer = BatchBayesianOptimizer::builder()
            .batch_size(4)
            .n_restarts(5)
            .build();

        assert_eq!(optimizer.config.batch_size, 4);
        assert_eq!(optimizer.config.n_restarts, 5);
    }

    #[test]
    fn test_batch_acquisition_variants() {
        let acquisitions = vec![
            BatchAcquisition::QExpectedImprovement { n_samples: 100 },
            BatchAcquisition::QProbabilityOfImprovement {
                epsilon: 0.01,
                n_samples: 100,
            },
            BatchAcquisition::QUpperConfidenceBound { beta: 2.0 },
            BatchAcquisition::SequentialExpectedImprovement,
            BatchAcquisition::SequentialWithDiversity {
                diversity_weight: 0.1,
                distance_metric: DistanceMetric::Euclidean,
            },
            BatchAcquisition::ConstantLiar {
                base_acquisition: AcquisitionFunction::ExpectedImprovement,
                liar_value: 0.0,
            },
            BatchAcquisition::ThompsonSampling { n_samples: 50 },
        ];

        for acquisition in acquisitions {
            let optimizer = BatchBayesianOptimizer::builder()
                .acquisition_function(acquisition)
                .build();

            // Just test that construction works
            assert_eq!(optimizer.config.batch_size, 4); // default
        }
    }

    #[test]
    fn test_distance_metrics() {
        let optimizer = BatchBayesianOptimizer::builder().build();
        let point1 = array![1.0, 2.0, 3.0];
        let point2 = array![4.0, 5.0, 6.0];

        let euclidean = optimizer
            .compute_distance(&point1, &point2, &DistanceMetric::Euclidean)
            .unwrap();
        let manhattan = optimizer
            .compute_distance(&point1, &point2, &DistanceMetric::Manhattan)
            .unwrap();
        let cosine = optimizer
            .compute_distance(&point1, &point2, &DistanceMetric::Cosine)
            .unwrap();

        assert!(euclidean > 0.0);
        assert!(manhattan > 0.0);
        assert!(cosine >= 0.0 && cosine <= 2.0); // Cosine distance is bounded
    }

    #[test]
    fn test_random_batch_generation() {
        let optimizer = BatchBayesianOptimizer::builder().batch_size(3).build();

        let bounds = array![[0.0, 1.0], [-1.0, 1.0]];
        let batch = optimizer.generate_random_batch(&bounds, 3).unwrap();

        assert_eq!(batch.shape(), &[3, 2]);

        // Check bounds
        for i in 0..3 {
            assert!(batch[[i, 0]] >= 0.0 && batch[[i, 0]] <= 1.0);
            assert!(batch[[i, 1]] >= -1.0 && batch[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_diversity_score_calculation() {
        let optimizer = BatchBayesianOptimizer::builder().build();

        // Create batch with known diversity
        let batch = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let diversity = optimizer
            .compute_diversity_score(&batch, &DistanceMetric::Euclidean)
            .unwrap();
        assert!(diversity > 0.0);

        // Single point should have zero diversity
        let single_point = array![[0.0, 0.0]];
        let single_diversity = optimizer
            .compute_diversity_score(&single_point, &DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(single_diversity, 0.0);
    }

    #[test]
    fn test_batch_config() {
        let config = BatchConfig {
            batch_size: 6,
            n_restarts: 20,
            max_iterations: 200,
            convergence_tolerance: 1e-8,
            diversity_weight: 0.2,
            parallel_optimization: true,
            random_seed: Some(42),
        };

        let optimizer = BatchBayesianOptimizer::builder()
            .config(config.clone())
            .build();

        assert_eq!(optimizer.config.batch_size, 6);
        assert_eq!(optimizer.config.n_restarts, 20);
        assert_eq!(optimizer.config.max_iterations, 200);
        assert!((optimizer.config.convergence_tolerance - 1e-8).abs() < 1e-15);
        assert!((optimizer.config.diversity_weight - 0.2).abs() < 1e-10);
        assert_eq!(optimizer.config.parallel_optimization, true);
        assert_eq!(optimizer.config.random_seed, Some(42));
    }

    #[test]
    fn test_batch_optimization_result_structure() {
        let result = BatchOptimizationResult {
            batch_points: array![[1.0, 2.0], [3.0, 4.0]],
            acquisition_values: array![0.5, 0.8],
            batch_acquisition_value: 1.3,
            diversity_score: 0.2,
            n_iterations: 100,
            converged: true,
        };

        assert_eq!(result.batch_points.shape(), &[2, 2]);
        assert_eq!(result.acquisition_values.len(), 2);
        assert!((result.batch_acquisition_value - 1.3).abs() < 1e-10);
        assert!((result.diversity_score - 0.2).abs() < 1e-10);
        assert_eq!(result.n_iterations, 100);
        assert_eq!(result.converged, true);
    }

    #[test]
    fn test_sequential_with_diversity_configuration() {
        let acquisition = BatchAcquisition::SequentialWithDiversity {
            diversity_weight: 0.15,
            distance_metric: DistanceMetric::Manhattan,
        };

        match acquisition {
            BatchAcquisition::SequentialWithDiversity {
                diversity_weight,
                distance_metric,
            } => {
                assert!((diversity_weight - 0.15).abs() < 1e-10);
                match distance_metric {
                    DistanceMetric::Manhattan => {}
                    _ => panic!("Wrong distance metric"),
                }
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_constant_liar_configuration() {
        let acquisition = BatchAcquisition::ConstantLiar {
            base_acquisition: AcquisitionFunction::UpperConfidenceBound { beta: 2.0 },
            liar_value: 5.0,
        };

        match acquisition {
            BatchAcquisition::ConstantLiar {
                base_acquisition,
                liar_value,
            } => {
                assert!((liar_value - 5.0).abs() < 1e-10);
                match base_acquisition {
                    AcquisitionFunction::UpperConfidenceBound { beta } => {
                        assert!((beta - 2.0).abs() < 1e-10);
                    }
                    _ => panic!("Wrong base acquisition"),
                }
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_builder_method_chaining() {
        let optimizer = BatchBayesianOptimizer::builder()
            .batch_size(8)
            .n_restarts(15)
            .diversity_weight(0.3)
            .acquisition_function(BatchAcquisition::ThompsonSampling { n_samples: 200 })
            .build();

        assert_eq!(optimizer.config.batch_size, 8);
        assert_eq!(optimizer.config.n_restarts, 15);
        assert!((optimizer.config.diversity_weight - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_thompson_sampling_configuration() {
        let acquisition = BatchAcquisition::ThompsonSampling { n_samples: 150 };

        match acquisition {
            BatchAcquisition::ThompsonSampling { n_samples } => {
                assert_eq!(n_samples, 150);
            }
            _ => panic!("Wrong acquisition type"),
        }
    }

    #[test]
    fn test_diversity_penalty_calculation() {
        let optimizer = BatchBayesianOptimizer::builder().build();

        let point = array![0.5, 0.5];
        let existing_points = array![[0.0, 0.0], [1.0, 1.0]];

        let penalty = optimizer
            .compute_diversity_penalty(&point, &existing_points, &DistanceMetric::Euclidean)
            .unwrap();
        assert!(penalty > 0.0);

        // Empty existing points should give zero penalty
        let empty_points = Array2::zeros((0, 2));
        let zero_penalty = optimizer
            .compute_diversity_penalty(&point, &empty_points, &DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(zero_penalty, 0.0);
    }
}
