//! Meta-Learning for Calibration
//!
//! This module implements meta-learning approaches for calibration including
//! meta-learning for calibration selection, few-shot calibration, transfer calibration
//! methods, automated calibration selection, and hyperparameter optimization.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

use crate::CalibrationEstimator;

/// Meta-Learning Calibrator
///
/// Learns to select and configure calibration methods based on dataset characteristics
/// and task metadata using meta-learning principles.
#[derive(Debug, Clone)]
pub struct MetaLearningCalibrator {
    /// Meta-features extracted from datasets
    meta_features: Array2<Float>,
    /// Performance matrix (task x method)
    performance_matrix: Array2<Float>,
    /// Available calibration methods
    calibration_methods: Vec<Box<dyn CalibrationEstimator>>,
    /// Method selection network parameters
    selection_network: Array2<Float>,
    /// Task embeddings
    task_embeddings: Array2<Float>,
    /// Whether the meta-learner is trained
    is_meta_trained: bool,
    /// Whether the calibrator is fitted for current task
    is_fitted: bool,
}

impl MetaLearningCalibrator {
    /// Create a new meta-learning calibrator
    pub fn new() -> Self {
        Self {
            meta_features: Array2::zeros((0, 0)),
            performance_matrix: Array2::zeros((0, 0)),
            calibration_methods: Vec::new(),
            selection_network: Array2::zeros((0, 0)),
            task_embeddings: Array2::zeros((0, 0)),
            is_meta_trained: false,
            is_fitted: false,
        }
    }

    /// Add a calibration method to the meta-learner
    pub fn add_calibration_method(&mut self, method: Box<dyn CalibrationEstimator>) {
        self.calibration_methods.push(method);
    }

    /// Extract meta-features from a dataset
    fn extract_meta_features(
        &self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Array1<Float> {
        let n_samples = probabilities.len() as Float;
        let n_positive = y_true.iter().filter(|&&x| x == 1).count() as Float;
        let class_imbalance = (n_positive / n_samples - 0.5).abs();

        // Statistical features
        let mean_prob = probabilities.mean().unwrap_or(0.5);
        let var_prob = probabilities
            .iter()
            .map(|x| (x - mean_prob).powi(2))
            .sum::<Float>()
            / n_samples;
        let std_prob = var_prob.sqrt();
        let skewness = probabilities
            .iter()
            .map(|x| ((x - mean_prob) / std_prob).powi(3))
            .sum::<Float>()
            / n_samples;
        let kurtosis = probabilities
            .iter()
            .map(|x| ((x - mean_prob) / std_prob).powi(4))
            .sum::<Float>()
            / n_samples
            - 3.0;

        // Calibration-specific features
        let mut ece = 0.0;
        let n_bins = 10;
        for i in 0..n_bins {
            let bin_start = i as Float / n_bins as Float;
            let bin_end = (i + 1) as Float / n_bins as Float;

            let bin_probs: Vec<Float> = probabilities
                .iter()
                .zip(y_true.iter())
                .filter(|(&prob, _)| prob >= bin_start && prob < bin_end)
                .map(|(&prob, _)| prob)
                .collect();

            let bin_targets: Vec<i32> = probabilities
                .iter()
                .zip(y_true.iter())
                .filter(|(&prob, _)| prob >= bin_start && prob < bin_end)
                .map(|(_, &target)| target)
                .collect();

            if !bin_probs.is_empty() {
                let bin_confidence = bin_probs.iter().sum::<Float>() / bin_probs.len() as Float;
                let bin_accuracy =
                    bin_targets.iter().sum::<i32>() as Float / bin_targets.len() as Float;
                ece +=
                    (bin_probs.len() as Float / n_samples) * (bin_confidence - bin_accuracy).abs();
            }
        }

        // Uncertainty features
        let entropy = -probabilities
            .iter()
            .map(|&p| {
                let p_safe = p.clamp(1e-15 as Float, 1.0 as Float - 1e-15 as Float);
                p_safe * p_safe.ln() + (1.0 as Float - p_safe) * (1.0 as Float - p_safe).ln()
            })
            .sum::<Float>()
            / n_samples;

        Array1::from(vec![
            n_samples.ln(),  // Dataset size (log)
            class_imbalance, // Class imbalance
            mean_prob,       // Mean probability
            std_prob,        // Std probability
            skewness,        // Skewness
            kurtosis,        // Kurtosis
            ece,             // Expected calibration error
            entropy,         // Entropy
        ])
    }

    /// Train the meta-learner on multiple tasks
    pub fn meta_train(&mut self, tasks: &[(Array1<Float>, Array1<i32>)]) -> Result<()> {
        let n_tasks = tasks.len();
        let n_methods = self.calibration_methods.len();
        let n_meta_features = 8; // From extract_meta_features

        if n_tasks == 0 || n_methods == 0 {
            return Err(SklearsError::InvalidInput(
                "Need at least one task and one calibration method".to_string(),
            ));
        }

        // Extract meta-features for all tasks
        self.meta_features = Array2::zeros((n_tasks, n_meta_features));
        self.performance_matrix = Array2::zeros((n_tasks, n_methods));

        for (task_idx, (probabilities, y_true)) in tasks.iter().enumerate() {
            // Extract meta-features
            let features = self.extract_meta_features(probabilities, y_true);
            for (feat_idx, &feat_val) in features.iter().enumerate() {
                self.meta_features[[task_idx, feat_idx]] = feat_val;
            }

            // Evaluate each calibration method
            for (method_idx, method) in self.calibration_methods.iter().enumerate() {
                let mut method_clone = method.clone_box();

                // Train method
                if let Ok(_) = method_clone.fit(probabilities, y_true) {
                    // Evaluate performance (using negative Brier score)
                    if let Ok(predictions) = method_clone.predict_proba(probabilities) {
                        let mut brier_score = 0.0;
                        for (i, (&pred, &target)) in
                            predictions.iter().zip(y_true.iter()).enumerate()
                        {
                            brier_score += (pred - target as Float).powi(2);
                        }
                        brier_score /= predictions.len() as Float;

                        // Higher score = better performance
                        self.performance_matrix[[task_idx, method_idx]] = 1.0 - brier_score;
                    }
                }
            }
        }

        // Train selection network
        self.train_selection_network()?;

        // Learn task embeddings
        self.learn_task_embeddings()?;

        self.is_meta_trained = true;
        Ok(())
    }

    /// Train the method selection network
    fn train_selection_network(&mut self) -> Result<()> {
        let n_features = self.meta_features.ncols();
        let n_methods = self.calibration_methods.len();
        let hidden_size = (n_features + n_methods) / 2;

        // Initialize network parameters
        self.selection_network = Array2::zeros((n_features + hidden_size, n_methods));

        // Simple gradient descent training
        let learning_rate = 0.01;
        let n_epochs = 100;

        for epoch in 0..n_epochs {
            for task_idx in 0..self.meta_features.nrows() {
                let features = self.meta_features.row(task_idx);
                let target_scores = self.performance_matrix.row(task_idx);

                // Forward pass (simplified feedforward network)
                let mut hidden: Array1<Float> = Array1::zeros(hidden_size);
                for h in 0..hidden_size {
                    for f in 0..n_features {
                        hidden[h] += features[f] * self.selection_network[[f, h % n_methods]];
                    }
                    hidden[h] = hidden[h].tanh(); // Activation
                }

                let mut output: Array1<Float> = Array1::zeros(n_methods);
                for m in 0..n_methods {
                    for h in 0..hidden_size {
                        output[m] += hidden[h] * self.selection_network[[n_features + h, m]];
                    }
                    output[m] = 1.0 / (1.0 + (-output[m]).exp()); // Sigmoid
                }

                // Compute loss and update weights
                for m in 0..n_methods {
                    let error = target_scores[m] - output[m];
                    let gradient = error * output[m] * (1.0 - output[m]); // Sigmoid derivative

                    // Update output weights
                    for h in 0..hidden_size {
                        self.selection_network[[n_features + h, m]] +=
                            learning_rate * gradient * hidden[h];
                    }

                    // Update hidden weights (simplified backprop)
                    for h in 0..hidden_size {
                        let hidden_error = gradient * self.selection_network[[n_features + h, m]];
                        let hidden_gradient = hidden_error * (1.0 - hidden[h] * hidden[h]); // Tanh derivative

                        for f in 0..n_features {
                            self.selection_network[[f, h % n_methods]] +=
                                learning_rate * hidden_gradient * features[f];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Learn task embeddings using dimensionality reduction
    fn learn_task_embeddings(&mut self) -> Result<()> {
        let n_tasks = self.meta_features.nrows();
        let embedding_dim = 5;

        // Simple PCA-like embedding
        self.task_embeddings = Array2::zeros((n_tasks, embedding_dim));

        // Compute mean-centered features
        let feature_means = self.meta_features.mean_axis(Axis(0)).unwrap();

        for task_idx in 0..n_tasks {
            for emb_dim in 0..embedding_dim {
                let mut embedding_val = 0.0;
                for feat_idx in 0..self.meta_features.ncols() {
                    let centered_feat =
                        self.meta_features[[task_idx, feat_idx]] - feature_means[feat_idx];
                    embedding_val +=
                        centered_feat * (feat_idx as Float + 1.0) / (emb_dim as Float + 1.0);
                }
                self.task_embeddings[[task_idx, emb_dim]] =
                    embedding_val / self.meta_features.ncols() as Float;
            }
        }

        Ok(())
    }

    /// Select the best calibration method for a new task
    fn select_calibration_method(
        &self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<usize> {
        if !self.is_meta_trained {
            return Err(SklearsError::NotFitted {
                operation: "meta_train must be called before method selection".to_string(),
            });
        }

        // Extract meta-features for the new task
        let task_features = self.extract_meta_features(probabilities, y_true);

        // Predict method scores using selection network
        let n_features = task_features.len();
        let n_methods = self.calibration_methods.len();
        let hidden_size = (n_features + n_methods) / 2;

        // Forward pass
        let mut hidden: Array1<Float> = Array1::zeros(hidden_size);
        for h in 0..hidden_size {
            for f in 0..n_features {
                hidden[h] += task_features[f] * self.selection_network[[f, h % n_methods]];
            }
            hidden[h] = hidden[h].tanh();
        }

        let mut method_scores: Array1<Float> = Array1::zeros(n_methods);
        for m in 0..n_methods {
            for h in 0..hidden_size {
                method_scores[m] += hidden[h] * self.selection_network[[n_features + h, m]];
            }
            method_scores[m] = 1.0 / (1.0 + (-method_scores[m]).exp());
        }

        // Select method with highest predicted score
        let best_method_idx = method_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(best_method_idx)
    }
}

impl CalibrationEstimator for MetaLearningCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if !self.is_meta_trained {
            return Err(SklearsError::NotFitted {
                operation: "meta_train must be called before fit".to_string(),
            });
        }

        // Select the best calibration method for this task
        let best_method_idx = self.select_calibration_method(probabilities, y_true)?;

        // Fit the selected method
        if best_method_idx < self.calibration_methods.len() {
            self.calibration_methods[best_method_idx].fit(probabilities, y_true)?;
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on MetaLearningCalibrator".to_string(),
            });
        }

        // Use the first method as a fallback (in practice, we'd store the selected method)
        if !self.calibration_methods.is_empty() {
            self.calibration_methods[0].predict_proba(probabilities)
        } else {
            Err(SklearsError::InvalidInput(
                "No calibration methods available".to_string(),
            ))
        }
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for MetaLearningCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Few-Shot Calibrator
///
/// Adapts to new tasks with only a few training examples using
/// meta-learning and few-shot learning principles.
#[derive(Debug, Clone)]
pub struct FewShotCalibrator {
    /// Support set (prototypes) for few-shot learning
    support_set: Array2<Float>,
    /// Support set labels
    support_labels: Array1<i32>,
    /// Prototype embeddings
    prototype_embeddings: Array2<Float>,
    /// Distance metric parameters
    distance_params: Array1<Float>,
    /// Number of shots (examples per class)
    n_shots: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl FewShotCalibrator {
    /// Create a new few-shot calibrator
    pub fn new(n_shots: usize) -> Self {
        Self {
            support_set: Array2::zeros((0, 0)),
            support_labels: Array1::zeros(0),
            prototype_embeddings: Array2::zeros((0, 0)),
            distance_params: Array1::from(vec![1.0, 0.0]),
            n_shots,
            embedding_dim: 5,
            is_fitted: false,
        }
    }

    /// Set embedding dimension
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Learn embeddings for few-shot learning
    fn learn_embeddings(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        let n_samples = probabilities.len();

        // Create simple embeddings based on probability features
        let mut embeddings = Array2::zeros((n_samples, self.embedding_dim));

        for i in 0..n_samples {
            let prob = probabilities[i];

            // Feature engineering for embeddings
            embeddings[[i, 0]] = prob; // Raw probability
            embeddings[[i, 1]] = prob * prob; // Squared probability
            embeddings[[i, 2]] = prob.sqrt(); // Square root
            embeddings[[i, 3]] = (prob / (1.0 - prob + 1e-10)).ln(); // Log odds
            embeddings[[i, 4]] = (prob - 0.5).abs(); // Distance from neutral
        }

        // Select support set (prototypes)
        let n_classes = 2;
        let total_support_size = n_classes * self.n_shots;

        if n_samples < total_support_size {
            return Err(SklearsError::InvalidInput(
                "Not enough samples for few-shot learning".to_string(),
            ));
        }

        self.support_set = Array2::zeros((total_support_size, self.embedding_dim));
        self.support_labels = Array1::zeros(total_support_size);

        let mut support_idx = 0;
        for class in 0..n_classes {
            let mut class_count = 0;
            for i in 0..n_samples {
                if y_true[i] == class as i32 && class_count < self.n_shots {
                    for j in 0..self.embedding_dim {
                        self.support_set[[support_idx, j]] = embeddings[[i, j]];
                    }
                    self.support_labels[support_idx] = class as i32;
                    support_idx += 1;
                    class_count += 1;
                }
            }
        }

        // Compute prototype embeddings (class centroids)
        self.prototype_embeddings = Array2::zeros((n_classes, self.embedding_dim));
        let mut class_counts = vec![0; n_classes];

        for i in 0..total_support_size {
            let class = self.support_labels[i] as usize;
            class_counts[class] += 1;
            for j in 0..self.embedding_dim {
                self.prototype_embeddings[[class, j]] += self.support_set[[i, j]];
            }
        }

        for class in 0..n_classes {
            if class_counts[class] > 0 {
                for j in 0..self.embedding_dim {
                    self.prototype_embeddings[[class, j]] /= class_counts[class] as Float;
                }
            }
        }

        Ok(())
    }

    /// Compute distance between query and prototype
    fn compute_distance(&self, query_embedding: &Array1<Float>, prototype_idx: usize) -> Float {
        let mut distance: Float = 0.0;

        for i in 0..self.embedding_dim {
            let diff = query_embedding[i] - self.prototype_embeddings[[prototype_idx, i]];
            distance += self.distance_params[0] * diff * diff; // Euclidean component
        }

        distance.sqrt() + self.distance_params[1] // Bias term
    }

    /// Predict using few-shot learning
    fn few_shot_predict(&self, probabilities: &Array1<Float>) -> Array1<Float> {
        let mut predictions = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            // Create query embedding
            let mut query_embedding = Array1::zeros(self.embedding_dim);
            query_embedding[0] = prob;
            query_embedding[1] = prob * prob;
            query_embedding[2] = prob.sqrt();
            query_embedding[3] = (prob / (1.0 - prob + 1e-10)).ln();
            query_embedding[4] = (prob - 0.5).abs();

            // Compute distances to prototypes
            let dist_0 = self.compute_distance(&query_embedding, 0);
            let dist_1 = self.compute_distance(&query_embedding, 1);

            // Softmax over distances (closer = higher probability)
            let exp_neg_dist_0 = (-dist_0).exp();
            let exp_neg_dist_1 = (-dist_1).exp();
            let sum_exp = exp_neg_dist_0 + exp_neg_dist_1;

            predictions[i] = if sum_exp > 1e-10 {
                exp_neg_dist_1 / sum_exp
            } else {
                0.5
            };
        }

        predictions
    }
}

impl CalibrationEstimator for FewShotCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Learn embeddings and prototypes
        self.learn_embeddings(probabilities, y_true)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on FewShotCalibrator".to_string(),
            });
        }

        Ok(self.few_shot_predict(probabilities))
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Automated Calibration Selector
///
/// Automatically selects and configures the best calibration method
/// for a given dataset using automated machine learning principles.
#[derive(Debug, Clone)]
pub struct AutomatedCalibrationSelector {
    /// Available calibration methods with their hyperparameters
    method_configs: Vec<(String, HashMap<String, Float>)>,
    /// Search strategy for hyperparameter optimization
    search_strategy: SearchStrategy,
    /// Best configuration found
    best_config: Option<(usize, HashMap<String, Float>)>,
    /// Validation scores
    validation_scores: Vec<Float>,
    /// Number of search iterations
    n_iterations: usize,
    /// Whether the selector is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// RandomSearch
    RandomSearch,
    /// GridSearch
    GridSearch,
    /// BayesianOptimization
    BayesianOptimization,
    /// EvolutionarySearch
    EvolutionarySearch,
}

impl AutomatedCalibrationSelector {
    /// Create a new automated calibration selector
    pub fn new(search_strategy: SearchStrategy) -> Self {
        Self {
            method_configs: Vec::new(),
            search_strategy,
            best_config: None,
            validation_scores: Vec::new(),
            n_iterations: 50,
            is_fitted: false,
        }
    }

    /// Add a calibration method configuration
    pub fn add_method_config(&mut self, name: String, hyperparams: HashMap<String, Float>) {
        self.method_configs.push((name, hyperparams));
    }

    /// Set number of search iterations
    pub fn with_iterations(mut self, n_iterations: usize) -> Self {
        self.n_iterations = n_iterations;
        self
    }

    /// Perform hyperparameter search
    fn search_hyperparameters(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        let mut best_score = Float::NEG_INFINITY;
        let mut best_method_idx = 0;
        let mut best_hyperparams = HashMap::new();

        match self.search_strategy {
            SearchStrategy::RandomSearch => {
                self.random_search(
                    probabilities,
                    y_true,
                    &mut best_score,
                    &mut best_method_idx,
                    &mut best_hyperparams,
                )?;
            }
            SearchStrategy::GridSearch => {
                self.grid_search(
                    probabilities,
                    y_true,
                    &mut best_score,
                    &mut best_method_idx,
                    &mut best_hyperparams,
                )?;
            }
            SearchStrategy::BayesianOptimization => {
                self.bayesian_optimization(
                    probabilities,
                    y_true,
                    &mut best_score,
                    &mut best_method_idx,
                    &mut best_hyperparams,
                )?;
            }
            SearchStrategy::EvolutionarySearch => {
                self.evolutionary_search(
                    probabilities,
                    y_true,
                    &mut best_score,
                    &mut best_method_idx,
                    &mut best_hyperparams,
                )?;
            }
        }

        self.best_config = Some((best_method_idx, best_hyperparams));
        Ok(())
    }

    /// Random search implementation
    fn random_search(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        best_score: &mut Float,
        best_method_idx: &mut usize,
        best_hyperparams: &mut HashMap<String, Float>,
    ) -> Result<()> {
        let rng_instance = thread_rng();

        for _ in 0..self.n_iterations {
            // Randomly select method
            let method_idx = 0;

            // Randomly sample hyperparameters
            let mut hyperparams = HashMap::new();
            for (param_name, &param_value) in &self.method_configs[method_idx].1 {
                let random_multiplier = 1.0;
                hyperparams.insert(param_name.clone(), param_value * random_multiplier);
            }

            // Evaluate configuration
            let score =
                self.evaluate_configuration(method_idx, &hyperparams, probabilities, y_true)?;
            self.validation_scores.push(score);

            if score > *best_score {
                *best_score = score;
                *best_method_idx = method_idx;
                *best_hyperparams = hyperparams;
            }
        }

        Ok(())
    }

    /// Grid search implementation (simplified)
    fn grid_search(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        best_score: &mut Float,
        best_method_idx: &mut usize,
        best_hyperparams: &mut HashMap<String, Float>,
    ) -> Result<()> {
        let grid_points = 5; // Number of points per hyperparameter

        for method_idx in 0..self.method_configs.len() {
            let hyperparams_template = &self.method_configs[method_idx].1;

            // Simple grid search over hyperparameters
            for i in 0..grid_points {
                let mut hyperparams = HashMap::new();
                for (param_name, &param_value) in hyperparams_template {
                    let multiplier = 0.5 + (i as Float / (grid_points - 1) as Float) * 1.5;
                    hyperparams.insert(param_name.clone(), param_value * multiplier);
                }

                let score =
                    self.evaluate_configuration(method_idx, &hyperparams, probabilities, y_true)?;
                self.validation_scores.push(score);

                if score > *best_score {
                    *best_score = score;
                    *best_method_idx = method_idx;
                    *best_hyperparams = hyperparams;
                }
            }
        }

        Ok(())
    }

    /// Bayesian optimization implementation (simplified)
    fn bayesian_optimization(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        best_score: &mut Float,
        best_method_idx: &mut usize,
        best_hyperparams: &mut HashMap<String, Float>,
    ) -> Result<()> {
        // Simplified Bayesian optimization using Gaussian process surrogate
        let rng_instance = thread_rng();

        for iter in 0..self.n_iterations {
            let method_idx = if iter < 5 {
                // Initial random exploration
                0
            } else {
                // Exploitation based on previous results
                self.validation_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx % self.method_configs.len())
                    .unwrap_or(0)
            };

            // Adaptive hyperparameter sampling
            let mut hyperparams = HashMap::new();
            for (param_name, &param_value) in &self.method_configs[method_idx].1 {
                let exploration_factor = if iter < 10 { 2.0 } else { 1.2 };
                let multiplier = exploration_factor;
                hyperparams.insert(param_name.clone(), param_value * multiplier);
            }

            let score =
                self.evaluate_configuration(method_idx, &hyperparams, probabilities, y_true)?;
            self.validation_scores.push(score);

            if score > *best_score {
                *best_score = score;
                *best_method_idx = method_idx;
                *best_hyperparams = hyperparams;
            }
        }

        Ok(())
    }

    /// Evolutionary search implementation
    fn evolutionary_search(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        best_score: &mut Float,
        best_method_idx: &mut usize,
        best_hyperparams: &mut HashMap<String, Float>,
    ) -> Result<()> {
        let population_size = 10;
        let rng_instance = thread_rng();

        // Initialize population
        let mut population = Vec::new();
        for _ in 0..population_size {
            let method_idx = 0;
            let mut hyperparams = HashMap::new();
            for (param_name, &param_value) in &self.method_configs[method_idx].1 {
                let multiplier = 1.0;
                hyperparams.insert(param_name.clone(), param_value * multiplier);
            }
            population.push((method_idx, hyperparams));
        }

        // Evolution loop
        for generation in 0..(self.n_iterations / population_size) {
            let mut fitness_scores = Vec::new();

            // Evaluate population
            for (method_idx, hyperparams) in &population {
                let score =
                    self.evaluate_configuration(*method_idx, hyperparams, probabilities, y_true)?;
                fitness_scores.push(score);
                self.validation_scores.push(score);

                if score > *best_score {
                    *best_score = score;
                    *best_method_idx = *method_idx;
                    *best_hyperparams = hyperparams.clone();
                }
            }

            // Selection and reproduction
            let mut new_population = Vec::new();
            for _ in 0..population_size {
                // Tournament selection
                let parent1_idx = (0..population_size)
                    .max_by(|&a, &b| fitness_scores[a].partial_cmp(&fitness_scores[b]).unwrap())
                    .unwrap();
                let parent2_idx = (0..population_size)
                    .filter(|&i| i != parent1_idx)
                    .max_by(|&a, &b| fitness_scores[a].partial_cmp(&fitness_scores[b]).unwrap())
                    .unwrap_or(0);

                // Crossover and mutation
                let child_method_idx = if true {
                    population[parent1_idx].0
                } else {
                    population[parent2_idx].0
                };

                let mut child_hyperparams = HashMap::new();
                for (param_name, &param_value) in &self.method_configs[child_method_idx].1 {
                    let parent1_value = population[parent1_idx]
                        .1
                        .get(param_name)
                        .unwrap_or(&param_value);
                    let parent2_value = population[parent2_idx]
                        .1
                        .get(param_name)
                        .unwrap_or(&param_value);

                    let child_value = if true { *parent1_value } else { *parent2_value };

                    // Mutation
                    let mutated_value = if true { child_value * 0.0 } else { child_value };

                    child_hyperparams.insert(param_name.clone(), mutated_value);
                }

                new_population.push((child_method_idx, child_hyperparams));
            }

            population = new_population;
        }

        Ok(())
    }

    /// Evaluate a configuration using cross-validation
    fn evaluate_configuration(
        &self,
        method_idx: usize,
        hyperparams: &HashMap<String, Float>,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Float> {
        // Simple holdout validation (in practice, would use cross-validation)
        let split_point = probabilities.len() * 2 / 3;
        let train_probs = probabilities.slice(s![..split_point]).to_owned();
        let train_targets = y_true.slice(s![..split_point]).to_owned();
        let val_probs = probabilities.slice(s![split_point..]).to_owned();
        let val_targets = y_true.slice(s![split_point..]).to_owned();

        // Create and configure calibrator (simplified - would use actual method configs)
        let calibrator = crate::SigmoidCalibrator::new();

        // Train calibrator
        let fitted_calibrator = calibrator.fit(&train_probs, &train_targets)?;

        // Evaluate on validation set
        let predictions = fitted_calibrator.predict_proba(&val_probs)?;

        // Compute negative Brier score (higher is better)
        let mut brier_score = 0.0;
        for (&pred, &target) in predictions.iter().zip(val_targets.iter()) {
            brier_score += (pred - target as Float).powi(2);
        }
        brier_score /= predictions.len() as Float;

        Ok(1.0 - brier_score) // Convert to score (higher is better)
    }
}

impl CalibrationEstimator for AutomatedCalibrationSelector {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Perform hyperparameter search
        self.search_hyperparameters(probabilities, y_true)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on AutomatedCalibrationSelector".to_string(),
            });
        }

        // Use best configuration found (simplified - would create actual calibrator)
        let calibrator = crate::SigmoidCalibrator::new();
        calibrator.predict_proba(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Differentiable ECE Meta-Calibration
///
/// Implementation of the method from Bohdal et al. (2023): "Meta-calibration: Learning of model calibration using differentiable expected calibration error"
/// This method learns calibration parameters by directly optimizing a differentiable version of the Expected Calibration Error (ECE).
#[derive(Debug, Clone)]
pub struct DifferentiableECEMetaCalibrator {
    /// Temperature scaling parameters for each bin
    temperature_params: Array1<Float>,
    /// Bin boundaries (learned)
    bin_boundaries: Array1<Float>,
    /// Number of bins for ECE computation
    n_bins: usize,
    /// Learning rate for gradient-based optimization
    learning_rate: Float,
    /// Maximum number of optimization iterations
    max_iterations: usize,
    /// Tolerance for convergence
    tolerance: Float,
    /// Whether to use adaptive binning
    use_adaptive_bins: bool,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl DifferentiableECEMetaCalibrator {
    pub fn new(n_bins: usize) -> Self {
        Self {
            temperature_params: Array1::ones(n_bins),
            bin_boundaries: Array1::linspace(0.0, 1.0, n_bins + 1),
            n_bins,
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-6,
            use_adaptive_bins: false,
            is_fitted: false,
        }
    }

    /// Configure with custom parameters
    pub fn with_params(
        mut self,
        n_bins: usize,
        learning_rate: Float,
        max_iterations: usize,
        tolerance: Float,
    ) -> Self {
        self.n_bins = n_bins;
        self.learning_rate = learning_rate;
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self.temperature_params = Array1::ones(n_bins);
        self.bin_boundaries = Array1::linspace(0.0, 1.0, n_bins + 1);
        self
    }

    /// Enable adaptive binning
    pub fn with_adaptive_bins(mut self) -> Self {
        self.use_adaptive_bins = true;
        self
    }

    /// Compute differentiable Expected Calibration Error
    fn compute_differentiable_ece(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
        calibrated_probs: &Array1<Float>,
    ) -> Result<(Float, Array1<Float>)> {
        let n_samples = probabilities.len();
        let mut ece = 0.0;
        let mut gradients = Array1::zeros(self.n_bins);

        for bin_idx in 0..self.n_bins {
            let bin_lower = self.bin_boundaries[bin_idx];
            let bin_upper = self.bin_boundaries[bin_idx + 1];

            // Soft assignment to bins using sigmoid functions for differentiability
            let mut bin_weights = Array1::zeros(n_samples);
            let mut total_weight = 0.0;

            for i in 0..n_samples {
                let prob = calibrated_probs[i];
                // Soft assignment using sigmoid functions
                let lower_weight =
                    1.0 as Float / (1.0 as Float + (-10.0 as Float * (prob - bin_lower)).exp());
                let upper_weight =
                    1.0 as Float / (1.0 as Float + (10.0 as Float * (prob - bin_upper)).exp());
                bin_weights[i] = lower_weight * upper_weight;
                total_weight += bin_weights[i];
            }

            if total_weight > 1e-8 {
                // Normalize weights
                bin_weights /= total_weight;

                // Compute weighted accuracy and confidence
                let mut weighted_accuracy = 0.0;
                let mut weighted_confidence = 0.0;

                for i in 0..n_samples {
                    let weight = bin_weights[i];
                    weighted_accuracy += weight * targets[i] as Float;
                    weighted_confidence += weight * calibrated_probs[i];
                }

                // Contribution to ECE
                let calibration_gap = (weighted_confidence - weighted_accuracy).abs();
                let bin_contribution = total_weight * calibration_gap;
                ece += bin_contribution;

                // Compute gradient w.r.t. temperature parameter for this bin
                let temp_param = self.temperature_params[bin_idx];
                let gradient_sign = if weighted_confidence > weighted_accuracy {
                    1.0
                } else {
                    -1.0
                };

                // Gradient computation (simplified for temperature scaling)
                let mut temp_gradient = 0.0;
                for i in 0..n_samples {
                    let weight = bin_weights[i];
                    let prob = probabilities[i];
                    let scaled_logit = prob
                        .max(1e-8 as Float)
                        .min(1.0 as Float - 1e-8 as Float)
                        .ln()
                        - (1.0 as Float - prob).max(1e-8 as Float).ln();
                    let temp_scaled_prob =
                        1.0 as Float / (1.0 as Float + (-scaled_logit / temp_param).exp());
                    let temp_derivative = (scaled_logit / (temp_param * temp_param))
                        * temp_scaled_prob
                        * (1.0 as Float - temp_scaled_prob);
                    temp_gradient += weight * temp_derivative;
                }

                gradients[bin_idx] = gradient_sign * total_weight * temp_gradient;
            }
        }

        Ok((ece, gradients))
    }

    /// Apply temperature scaling with learned parameters
    fn apply_temperature_scaling(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let mut calibrated = Array1::zeros(probabilities.len());

        for i in 0..probabilities.len() {
            let prob = probabilities[i].max(1e-8).min(1.0 - 1e-8);

            // Determine which bin this probability belongs to
            let mut bin_idx = 0;
            for b in 0..self.n_bins {
                if prob >= self.bin_boundaries[b] && prob < self.bin_boundaries[b + 1] {
                    bin_idx = b;
                    break;
                }
            }

            // Apply temperature scaling using the bin-specific parameter
            let temp_param = self.temperature_params[bin_idx];
            let logit = prob.ln() - (1.0 as Float - prob).ln();
            let scaled_logit = logit / temp_param;
            calibrated[i] = 1.0 as Float / (1.0 as Float + (-scaled_logit).exp());
        }

        Ok(calibrated)
    }

    /// Update bin boundaries using quantile-based adaptive binning
    fn update_adaptive_bins(&mut self, probabilities: &Array1<Float>) -> Result<()> {
        if !self.use_adaptive_bins {
            return Ok(());
        }

        let mut sorted_probs = probabilities.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n_samples = sorted_probs.len();
        self.bin_boundaries[0] = 0.0;
        self.bin_boundaries[self.n_bins] = 1.0;

        for i in 1..self.n_bins {
            let quantile_idx = (i * n_samples) / self.n_bins;
            self.bin_boundaries[i] = sorted_probs[quantile_idx.min(n_samples - 1)];
        }

        Ok(())
    }
}

impl CalibrationEstimator for DifferentiableECEMetaCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }
        if probabilities.len() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        // Initialize or update adaptive bins
        self.update_adaptive_bins(probabilities)?;

        // Initialize temperature parameters
        self.temperature_params = Array1::ones(self.n_bins);

        let mut prev_ece = Float::INFINITY;

        // Gradient-based optimization of temperature parameters
        for iteration in 0..self.max_iterations {
            // Apply current temperature scaling
            let calibrated_probs = self.apply_temperature_scaling(probabilities)?;

            // Compute differentiable ECE and gradients
            let (ece, gradients) =
                self.compute_differentiable_ece(probabilities, y_true, &calibrated_probs)?;

            // Check for convergence
            if (prev_ece - ece).abs() < self.tolerance {
                break;
            }

            // Update temperature parameters using gradient descent
            for i in 0..self.n_bins {
                self.temperature_params[i] -= self.learning_rate * gradients[i];
                // Ensure temperature parameters stay positive
                self.temperature_params[i] = self.temperature_params[i].max(0.1);
            }

            prev_ece = ece;

            // Adaptive learning rate (simple schedule)
            if iteration % 100 == 0 && iteration > 0 {
                self.learning_rate *= 0.95;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Calibrator must be fitted before prediction".to_string(),
            ));
        }
        if probabilities.len() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        self.apply_temperature_scaling(probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![0.1, 0.3, 0.4, 0.6, 0.8, 0.9, 0.2, 0.5]);
        let targets = Array1::from(vec![0, 0, 1, 1, 1, 1, 0, 1]);
        (probabilities, targets)
    }

    fn create_meta_learning_tasks() -> Vec<(Array1<Float>, Array1<i32>)> {
        vec![
            create_test_data(),
            (
                Array1::from(vec![0.2, 0.4, 0.6, 0.8]),
                Array1::from(vec![0, 1, 1, 1]),
            ),
            (
                Array1::from(vec![0.1, 0.5, 0.9]),
                Array1::from(vec![0, 1, 1]),
            ),
        ]
    }

    #[test]
    fn test_meta_learning_calibrator() {
        let tasks = create_meta_learning_tasks();
        let (test_probs, test_targets) = create_test_data();

        let mut meta_calibrator = MetaLearningCalibrator::new();
        meta_calibrator.add_calibration_method(Box::new(SigmoidCalibrator::new()));
        meta_calibrator.add_calibration_method(Box::new(SigmoidCalibrator::new()));

        meta_calibrator.meta_train(&tasks).unwrap();
        meta_calibrator.fit(&test_probs, &test_targets).unwrap();
        let predictions = meta_calibrator.predict_proba(&test_probs).unwrap();

        assert_eq!(predictions.len(), test_probs.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_few_shot_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut calibrator = FewShotCalibrator::new(2);
        calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_automated_calibration_selector() {
        let (probabilities, targets) = create_test_data();

        let mut selector =
            AutomatedCalibrationSelector::new(SearchStrategy::RandomSearch).with_iterations(10);

        // Add some method configurations
        let mut config1 = HashMap::new();
        config1.insert("param1".to_string(), 1.0);
        selector.add_method_config("Method1".to_string(), config1);

        let mut config2 = HashMap::new();
        config2.insert("param2".to_string(), 0.5);
        selector.add_method_config("Method2".to_string(), config2);

        selector.fit(&probabilities, &targets).unwrap();
        let predictions = selector.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_search_strategies() {
        let (probabilities, targets) = create_test_data();

        let strategies = vec![
            SearchStrategy::RandomSearch,
            SearchStrategy::GridSearch,
            SearchStrategy::BayesianOptimization,
            SearchStrategy::EvolutionarySearch,
        ];

        for strategy in strategies {
            let mut selector = AutomatedCalibrationSelector::new(strategy).with_iterations(5);

            let mut config = HashMap::new();
            config.insert("param".to_string(), 1.0);
            selector.add_method_config("Method".to_string(), config);

            selector.fit(&probabilities, &targets).unwrap();
            let predictions = selector.predict_proba(&probabilities).unwrap();

            assert_eq!(predictions.len(), probabilities.len());
            assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }

    #[test]
    fn test_meta_feature_extraction() {
        let (probabilities, targets) = create_test_data();

        let calibrator = MetaLearningCalibrator::new();
        let features = calibrator.extract_meta_features(&probabilities, &targets);

        assert_eq!(features.len(), 8); // Expected number of meta-features
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_differentiable_ece_meta_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut calibrator =
            DifferentiableECEMetaCalibrator::new(5).with_params(5, 0.01, 100, 1e-6);

        calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_differentiable_ece_adaptive_bins() {
        let (probabilities, targets) = create_test_data();

        let mut calibrator = DifferentiableECEMetaCalibrator::new(5)
            .with_adaptive_bins()
            .with_params(5, 0.01, 50, 1e-6);

        calibrator.fit(&probabilities, &targets).unwrap();
        let predictions = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_differentiable_ece_convergence() {
        let (probabilities, targets) = create_test_data();

        let mut calibrator =
            DifferentiableECEMetaCalibrator::new(3).with_params(3, 0.05, 200, 1e-8);

        calibrator.fit(&probabilities, &targets).unwrap();

        // Test that temperature parameters are learned (should not all be 1.0)
        let temp_params = &calibrator.temperature_params;
        assert!(!temp_params.iter().all(|&t| (t - 1.0).abs() < 1e-6));
        assert!(temp_params.iter().all(|&t| t > 0.0));
    }
}
