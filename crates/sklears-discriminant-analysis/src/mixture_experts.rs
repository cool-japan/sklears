//! Mixture of Experts Discriminant Analysis
//!
//! This module implements mixture of experts discriminant models, where multiple expert classifiers
//! are combined using a gating network to make predictions. Each expert specializes in a different
//! region of the input space.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::hash::Hash;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub enum ExpertType {
    /// Linear Discriminant Analysis expert
    LDA,
    /// Quadratic Discriminant Analysis expert
    QDA,
    /// Neural network expert
    Neural,
}

#[derive(Debug, Clone)]
pub enum GatingNetworkType {
    /// Softmax gating network
    Softmax,
    /// Linear gating network
    Linear,
    /// Neural gating network
    Neural,
}

#[derive(Debug, Clone)]
pub struct MixtureOfExpertsConfig {
    /// Number of expert models
    pub n_experts: usize,
    /// Type of expert models to use
    pub expert_type: ExpertType,
    /// Type of gating network
    pub gating_type: GatingNetworkType,
    /// Maximum number of iterations for EM algorithm
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Regularization parameter
    pub reg_param: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for MixtureOfExpertsConfig {
    fn default() -> Self {
        Self {
            n_experts: 3,
            expert_type: ExpertType::LDA,
            gating_type: GatingNetworkType::Softmax,
            max_iter: 100,
            tol: 1e-6,
            learning_rate: 0.01,
            reg_param: 1e-4,
            random_state: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MixtureOfExpertsDiscriminantAnalysis<State = Untrained> {
    config: MixtureOfExpertsConfig,
    data: Option<TrainedData>,
    _state: PhantomData<State>,
}

#[derive(Debug, Clone)]
struct TrainedData {
    experts: Vec<ExpertModel>,
    gating_network: GatingNetwork,
    classes: Array1<i32>,
    n_features: usize,
    log_likelihood_history: Vec<Float>,
}

impl MixtureOfExpertsDiscriminantAnalysis<Untrained> {
    pub fn new() -> Self {
        Self {
            config: MixtureOfExpertsConfig::default(),
            data: None,
            _state: PhantomData,
        }
    }

    pub fn with_config(config: MixtureOfExpertsConfig) -> Self {
        Self {
            config,
            data: None,
            _state: PhantomData,
        }
    }

    pub fn n_experts(mut self, n_experts: usize) -> Self {
        self.config.n_experts = n_experts;
        self
    }

    pub fn expert_type(mut self, expert_type: ExpertType) -> Self {
        self.config.expert_type = expert_type;
        self
    }

    pub fn gating_type(mut self, gating_type: GatingNetworkType) -> Self {
        self.config.gating_type = gating_type;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Default for MixtureOfExpertsDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

pub type TrainedMixtureOfExpertsDiscriminantAnalysis =
    MixtureOfExpertsDiscriminantAnalysis<Trained>;

impl MixtureOfExpertsDiscriminantAnalysis<Trained> {
    pub fn experts(&self) -> &Vec<ExpertModel> {
        &self.data.as_ref().unwrap().experts
    }

    pub fn gating_network(&self) -> &GatingNetwork {
        &self.data.as_ref().unwrap().gating_network
    }

    pub fn classes(&self) -> &Array1<i32> {
        &self.data.as_ref().unwrap().classes
    }

    pub fn n_features(&self) -> usize {
        self.data.as_ref().unwrap().n_features
    }

    pub fn log_likelihood_history(&self) -> &Vec<Float> {
        &self.data.as_ref().unwrap().log_likelihood_history
    }

    pub fn expert_weights(&self, expert_idx: usize) -> Option<&Array2<Float>> {
        self.data
            .as_ref()
            .unwrap()
            .experts
            .get(expert_idx)
            .map(|expert| &expert.weights)
    }

    pub fn gating_weights(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().gating_network.weights
    }
}

#[derive(Debug, Clone)]
struct ExpertModel {
    /// Expert weights for LDA/QDA
    weights: Array2<Float>,
    /// Expert bias
    bias: Array1<Float>,
    /// Class means for the expert
    means: Array2<Float>,
    /// Covariance matrix for the expert
    covariance: Array2<Float>,
    /// Expert type
    expert_type: ExpertType,
}

impl ExpertModel {
    fn new(n_features: usize, n_classes: usize, expert_type: ExpertType) -> Self {
        Self {
            weights: Array2::zeros((n_classes, n_features)),
            bias: Array1::zeros(n_classes),
            means: Array2::zeros((n_classes, n_features)),
            covariance: Array2::eye(n_features),
            expert_type,
        }
    }

    fn predict_log_proba(&self, x: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let n_classes = self.means.nrows();
        let mut log_proba = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            for (j, class_mean) in self.means.axis_iter(Axis(0)).enumerate() {
                // Compute log probability for this expert
                let diff = &sample - &class_mean;
                let log_prob = match self.expert_type {
                    ExpertType::LDA => {
                        // Linear discriminant: log p(x|class) ~ x^T W + b
                        let linear_score = sample.dot(&self.weights.row(j)) + self.bias[j];
                        linear_score
                    }
                    ExpertType::QDA => {
                        // Quadratic discriminant: log p(x|class) ~ -0.5 * (x-μ)^T Σ^-1 (x-μ)
                        let inv_cov = &self.covariance;
                        let mahalanobis = diff.dot(&inv_cov.dot(&diff));
                        -0.5 * mahalanobis
                    }
                    ExpertType::Neural => {
                        // Neural network activation
                        let linear_score = sample.dot(&self.weights.row(j)) + self.bias[j];
                        1.0 / (1.0 + (-linear_score).exp()) // sigmoid
                    }
                };
                log_proba[[i, j]] = log_prob;
            }
        }

        log_proba
    }
}

#[derive(Debug, Clone)]
struct GatingNetwork {
    /// Gating network weights
    weights: Array2<Float>,
    /// Gating network bias
    bias: Array1<Float>,
    /// Type of gating network
    gating_type: GatingNetworkType,
}

impl GatingNetwork {
    fn new(n_features: usize, n_experts: usize, gating_type: GatingNetworkType) -> Self {
        Self {
            weights: Array2::zeros((n_experts, n_features)),
            bias: Array1::zeros(n_experts),
            gating_type,
        }
    }

    fn compute_gates(&self, x: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let n_experts = self.weights.nrows();
        let mut gates = Array2::zeros((n_samples, n_experts));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let logits = sample.dot(&self.weights.t()) + &self.bias;

            // Apply activation based on gating type
            let gate_values = match self.gating_type {
                GatingNetworkType::Softmax => {
                    // Softmax activation
                    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_logits: Array1<Float> = logits.mapv(|x| (x - max_logit).exp());
                    let sum_exp: Float = exp_logits.sum();
                    exp_logits.mapv(|x| x / sum_exp)
                }
                GatingNetworkType::Linear => {
                    // Linear activation with normalization
                    let sum: Float = logits.sum();
                    if sum.abs() > 1e-10 {
                        logits.mapv(|x| x / sum)
                    } else {
                        Array1::from_elem(n_experts, 1.0 / n_experts as Float)
                    }
                }
                GatingNetworkType::Neural => {
                    // Sigmoid + normalization
                    let sigmoid_logits: Array1<Float> = logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                    let sum: Float = sigmoid_logits.sum();
                    if sum > 1e-10 {
                        sigmoid_logits.mapv(|x| x / sum)
                    } else {
                        Array1::from_elem(n_experts, 1.0 / n_experts as Float)
                    }
                }
            };

            gates.row_mut(i).assign(&gate_values);
        }

        gates
    }
}

impl Estimator<Untrained> for MixtureOfExpertsDiscriminantAnalysis<Untrained> {
    type Config = MixtureOfExpertsConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for MixtureOfExpertsDiscriminantAnalysis<Untrained> {
    type Fitted = MixtureOfExpertsDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Input data and targets have different lengths".to_string(),
            ));
        }

        // Extract unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Initialize experts
        let mut experts = Vec::new();
        for _ in 0..self.config.n_experts {
            experts.push(ExpertModel::new(
                n_features,
                n_classes,
                self.config.expert_type.clone(),
            ));
        }

        // Initialize gating network
        let mut gating_network = GatingNetwork::new(
            n_features,
            self.config.n_experts,
            self.config.gating_type.clone(),
        );

        // Initialize parameters randomly
        self.initialize_parameters(&mut experts, &mut gating_network, &classes, x, y)?;

        // EM algorithm
        let mut log_likelihood_history = Vec::new();
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..self.config.max_iter {
            // E-step: Compute responsibilities
            let (responsibilities, log_likelihood) =
                self.e_step(x, y, &experts, &gating_network, &classes)?;
            log_likelihood_history.push(log_likelihood);

            // Check convergence
            if iteration > 0 && (log_likelihood - prev_log_likelihood).abs() < self.config.tol {
                break;
            }
            prev_log_likelihood = log_likelihood;

            // M-step: Update parameters
            self.m_step(
                x,
                y,
                &mut experts,
                &mut gating_network,
                &responsibilities,
                &classes,
            )?;
        }

        let trained_data = TrainedData {
            experts,
            gating_network,
            classes,
            n_features,
            log_likelihood_history,
        };

        Ok(MixtureOfExpertsDiscriminantAnalysis {
            config: self.config,
            data: Some(trained_data),
            _state: PhantomData,
        })
    }
}

impl MixtureOfExpertsDiscriminantAnalysis<Untrained> {
    fn initialize_parameters(
        &self,
        experts: &mut Vec<ExpertModel>,
        gating_network: &mut GatingNetwork,
        classes: &Array1<i32>,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<()> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let n_classes = classes.len();
        let n_features = x.ncols();

        // Simple pseudorandom number generator using hash
        let mut hasher = DefaultHasher::new();
        self.config.random_state.unwrap_or(42).hash(&mut hasher);
        let mut seed = hasher.finish();

        let next_random = |seed: &mut u64| -> Float {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((*seed / 65536) % 32768) as Float / 32768.0
        };

        // Initialize class means for each expert
        for expert in experts.iter_mut() {
            for (class_idx, &class_label) in classes.iter().enumerate() {
                let class_mask: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == class_label)
                    .map(|(idx, _)| idx)
                    .collect();

                if !class_mask.is_empty() {
                    let class_data = x.select(Axis(0), &class_mask);
                    let mean = class_data.mean_axis(Axis(0)).unwrap();
                    expert.means.row_mut(class_idx).assign(&mean);
                } else {
                    // If no samples for this class, use random initialization
                    for j in 0..n_features {
                        expert.means[[class_idx, j]] = next_random(&mut seed) * 0.1;
                    }
                }
            }

            // Initialize weights randomly
            for i in 0..n_classes {
                for j in 0..n_features {
                    expert.weights[[i, j]] = (next_random(&mut seed) - 0.5) * 0.1;
                }
                expert.bias[i] = (next_random(&mut seed) - 0.5) * 0.1;
            }
        }

        // Initialize gating network weights randomly
        for i in 0..self.config.n_experts {
            for j in 0..n_features {
                gating_network.weights[[i, j]] = (next_random(&mut seed) - 0.5) * 0.1;
            }
            gating_network.bias[i] = (next_random(&mut seed) - 0.5) * 0.1;
        }

        Ok(())
    }

    fn e_step(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        experts: &Vec<ExpertModel>,
        gating_network: &GatingNetwork,
        classes: &Array1<i32>,
    ) -> Result<(Array2<Float>, Float)> {
        let n_samples = x.nrows();
        let n_experts = experts.len();
        let mut responsibilities = Array2::zeros((n_samples, n_experts));
        let mut total_log_likelihood = 0.0;

        // Compute gating network outputs
        let gates = gating_network.compute_gates(x);

        for i in 0..n_samples {
            let sample = x.row(i).to_owned().insert_axis(Axis(0));
            let true_class = y[i];
            let class_idx = classes.iter().position(|&c| c == true_class).unwrap();

            let mut expert_likelihoods = Array1::zeros(n_experts);

            for (expert_idx, expert) in experts.iter().enumerate() {
                // Compute likelihood of this sample under this expert
                let log_proba = expert.predict_log_proba(&sample);
                expert_likelihoods[expert_idx] = log_proba[[0, class_idx]].exp();
            }

            // Compute responsibilities using Bayes' rule
            let weighted_likelihoods = &expert_likelihoods * &gates.row(i);
            let evidence = weighted_likelihoods.sum();

            if evidence > 1e-10 {
                responsibilities
                    .row_mut(i)
                    .assign(&(weighted_likelihoods / evidence));
                total_log_likelihood += evidence.ln();
            } else {
                // If evidence is too small, assign equal responsibilities
                responsibilities.row_mut(i).fill(1.0 / n_experts as Float);
            }
        }

        Ok((responsibilities, total_log_likelihood))
    }

    fn m_step(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        experts: &mut Vec<ExpertModel>,
        gating_network: &mut GatingNetwork,
        responsibilities: &Array2<Float>,
        classes: &Array1<i32>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_experts = experts.len();
        let n_classes = classes.len();

        // Update expert parameters
        for (expert_idx, expert) in experts.iter_mut().enumerate() {
            let expert_responsibilities = responsibilities.column(expert_idx);
            let total_responsibility: Float = expert_responsibilities.sum();

            if total_responsibility > 1e-10 {
                // Update class means and weights for this expert
                for (class_idx, &class_label) in classes.iter().enumerate() {
                    let class_mask: Vec<usize> = y
                        .iter()
                        .enumerate()
                        .filter(|(_, &label)| label == class_label)
                        .map(|(idx, _)| idx)
                        .collect();

                    if !class_mask.is_empty() {
                        let class_responsibilities: Array1<Float> =
                            expert_responsibilities.select(Axis(0), &class_mask);
                        let class_data = x.select(Axis(0), &class_mask);

                        let weighted_sum_resp: Float = class_responsibilities.sum();

                        if weighted_sum_resp > 1e-10 {
                            // Update mean
                            let mut weighted_mean = Array1::zeros(n_features);
                            for (i, &sample_idx) in class_mask.iter().enumerate() {
                                let sample = x.row(sample_idx);
                                weighted_mean = weighted_mean + &sample * class_responsibilities[i];
                            }
                            weighted_mean /= weighted_sum_resp;
                            expert.means.row_mut(class_idx).assign(&weighted_mean);

                            // Update weights using gradient ascent (simplified)
                            for j in 0..n_features {
                                let gradient =
                                    class_data.column(j).sum() / class_data.nrows() as Float;
                                expert.weights[[class_idx, j]] +=
                                    self.config.learning_rate * gradient;
                            }
                        }
                    }
                }
            }
        }

        // Update gating network parameters
        for expert_idx in 0..n_experts {
            let expert_responsibilities = responsibilities.column(expert_idx);

            // Compute gradient for gating network weights
            for j in 0..n_features {
                let mut gradient = 0.0;
                for i in 0..n_samples {
                    gradient += expert_responsibilities[i] * x[[i, j]];
                }
                gradient /= n_samples as Float;
                gating_network.weights[[expert_idx, j]] += self.config.learning_rate * gradient;
            }

            // Update bias
            let bias_gradient = expert_responsibilities.mean().unwrap();
            gating_network.bias[expert_idx] += self.config.learning_rate * bias_gradient;
        }

        Ok(())
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MixtureOfExpertsDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probas.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes()[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for MixtureOfExpertsDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_features
            )));
        }

        let n_classes = self.classes().len();
        let n_experts = self.experts().len();
        let mut predictions = Array2::zeros((n_samples, n_classes));

        // Compute gating network outputs
        let gates = self.gating_network().compute_gates(x);

        for i in 0..n_samples {
            let sample = x.row(i).to_owned().insert_axis(Axis(0));
            let mut weighted_probas = Array1::zeros(n_classes);

            for (expert_idx, expert) in self.experts().iter().enumerate() {
                // Get expert predictions
                let expert_log_probas = expert.predict_log_proba(&sample);
                let expert_probas = expert_log_probas.mapv(|x| x.exp());

                // Weight by gating network output
                let gate_weight = gates[[i, expert_idx]];
                weighted_probas = weighted_probas + &expert_probas.row(0) * gate_weight;
            }

            // Normalize probabilities
            let sum = weighted_probas.sum();
            if sum > 1e-10 {
                weighted_probas /= sum;
            } else {
                weighted_probas.fill(1.0 / n_classes as Float);
            }

            predictions.row_mut(i).assign(&weighted_probas);
        }

        Ok(predictions)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MixtureOfExpertsDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Transform returns the gating network outputs (expert weights for each sample)
        Ok(self.gating_network().compute_gates(x))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mixture_of_experts_basic() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2] // Class 1
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let moe = MixtureOfExpertsDiscriminantAnalysis::new()
            .n_experts(2)
            .expert_type(ExpertType::LDA)
            .max_iter(10);

        let fitted = moe.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.experts().len(), 2);
    }

    #[test]
    fn test_mixture_of_experts_predict_proba() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let moe = MixtureOfExpertsDiscriminantAnalysis::new()
            .n_experts(2)
            .expert_type(ExpertType::QDA)
            .max_iter(5);

        let fitted = moe.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mixture_of_experts_transform() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let moe = MixtureOfExpertsDiscriminantAnalysis::new()
            .n_experts(3)
            .max_iter(5);

        let fitted = moe.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 3)); // 4 samples, 3 experts

        // Check that expert weights sum to 1 for each sample
        for row in transformed.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_different_expert_types() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let expert_types = vec![ExpertType::LDA, ExpertType::QDA, ExpertType::Neural];

        for expert_type in expert_types {
            let moe = MixtureOfExpertsDiscriminantAnalysis::new()
                .n_experts(2)
                .expert_type(expert_type)
                .max_iter(5);

            let fitted = moe.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 4);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_different_gating_types() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let gating_types = vec![
            GatingNetworkType::Softmax,
            GatingNetworkType::Linear,
            GatingNetworkType::Neural,
        ];

        for gating_type in gating_types {
            let moe = MixtureOfExpertsDiscriminantAnalysis::new()
                .n_experts(2)
                .gating_type(gating_type)
                .max_iter(5);

            let fitted = moe.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 4);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_mixture_of_experts_convergence() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let moe = MixtureOfExpertsDiscriminantAnalysis::new()
            .n_experts(2)
            .max_iter(50)
            .tol(1e-6);

        let fitted = moe.fit(&x, &y).unwrap();
        let history = fitted.log_likelihood_history();

        // Check that log likelihood generally increases
        assert!(history.len() > 0);
        if history.len() > 1 {
            let initial_ll = history[0];
            let final_ll = *history.last().unwrap();
            // Final likelihood should be at least as good as initial
            assert!(final_ll >= initial_ll - 1e-6);
        }
    }

    #[test]
    fn test_expert_weights_access() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let moe = MixtureOfExpertsDiscriminantAnalysis::new()
            .n_experts(3)
            .max_iter(5);

        let fitted = moe.fit(&x, &y).unwrap();

        // Test expert weights access
        for expert_idx in 0..3 {
            let weights = fitted.expert_weights(expert_idx);
            assert!(weights.is_some());
            let weights = weights.unwrap();
            assert_eq!(weights.dim(), (2, 2)); // 2 classes, 2 features
        }

        // Test invalid expert index
        assert!(fitted.expert_weights(5).is_none());

        // Test gating weights access
        let gating_weights = fitted.gating_weights();
        assert_eq!(gating_weights.dim(), (3, 2)); // 3 experts, 2 features
    }
}
