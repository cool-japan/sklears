//! Dirichlet Process Naive Bayes implementation
//!
//! Implements non-parametric Bayesian classification using Dirichlet Process
//! mixtures, allowing for automatic discovery of the number of components.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::SeedableRng;
// SciRS2 Policy Compliance - Use scirs2-core for random distributions
use scirs2_core::random::essentials::Uniform as RandUniform;
use scirs2_core::random::Distribution;
use scirs2_core::Beta;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DirichletProcessError {
    #[error("Features and targets have different number of samples")]
    DimensionMismatch,
    #[error("Empty dataset provided")]
    EmptyDataset,
    #[error("Invalid concentration parameter: {0}")]
    InvalidConcentration(f64),
    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
    #[error("Numerical computation error: {0}")]
    NumericalError(String),
    #[error("Stick-breaking construction failed: {0}")]
    StickBreakingError(String),
}

/// Configuration for Dirichlet Process Naive Bayes
#[derive(Debug, Clone)]
pub struct DirichletProcessConfig {
    pub concentration_alpha: f64,
    pub base_distribution_params: BaseDistributionParams,
    pub max_components: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub random_seed: Option<u64>,
    pub inference_method: InferenceMethod,
    pub stick_breaking_truncation: usize,
}

impl Default for DirichletProcessConfig {
    fn default() -> Self {
        Self {
            concentration_alpha: 1.0,
            base_distribution_params: BaseDistributionParams::default(),
            max_components: 50,
            max_iterations: 100,
            tolerance: 1e-6,
            random_seed: None,
            inference_method: InferenceMethod::Gibbs,
            stick_breaking_truncation: 20,
        }
    }
}

#[derive(Debug, Clone)]
pub enum InferenceMethod {
    /// Gibbs
    Gibbs,
    /// VariationalBayes
    VariationalBayes,
    /// StickBreaking
    StickBreaking,
}

/// Base distribution parameters for Dirichlet Process
#[derive(Debug, Clone)]
pub struct BaseDistributionParams {
    pub prior_mean: f64,
    pub prior_precision: f64,
    pub wishart_degrees: f64,
    pub wishart_scale: f64,
}

impl Default for BaseDistributionParams {
    fn default() -> Self {
        Self {
            prior_mean: 0.0,
            prior_precision: 1.0,
            wishart_degrees: 1.0,
            wishart_scale: 1.0,
        }
    }
}

/// Component in the Dirichlet Process mixture
#[derive(Debug, Clone)]
pub struct DirichletComponent {
    pub weight: f64,
    pub mean: Array1<f64>,
    pub precision: Array2<f64>, // Precision matrix (inverse covariance)
    pub sample_count: usize,
    pub class_counts: HashMap<i32, usize>,
}

impl DirichletComponent {
    pub fn new(n_features: usize) -> Self {
        Self {
            weight: 0.0,
            mean: Array1::zeros(n_features),
            precision: Array2::eye(n_features),
            sample_count: 0,
            class_counts: HashMap::new(),
        }
    }

    pub fn add_sample(&mut self, sample: &Array1<f64>, class: i32) {
        self.sample_count += 1;
        *self.class_counts.entry(class).or_insert(0) += 1;

        // Update sufficient statistics (simplified online update)
        let learning_rate = 1.0 / self.sample_count as f64;
        let diff = sample - &self.mean;
        self.mean = &self.mean + &(&diff * learning_rate);
    }

    pub fn remove_sample(&mut self, sample: &Array1<f64>, class: i32) {
        if self.sample_count > 0 {
            self.sample_count -= 1;
            if let Some(count) = self.class_counts.get_mut(&class) {
                if *count > 0 {
                    *count -= 1;
                    if *count == 0 {
                        self.class_counts.remove(&class);
                    }
                }
            }

            // Update mean (simplified)
            if self.sample_count > 0 {
                let learning_rate = 1.0 / (self.sample_count + 1) as f64;
                let diff = sample - &self.mean;
                self.mean = &self.mean - &(&diff * learning_rate);
            }
        }
    }

    pub fn log_likelihood(&self, sample: &Array1<f64>) -> f64 {
        // Simplified Gaussian log-likelihood
        let diff = sample - &self.mean;
        let mahalanobis = diff.dot(&self.precision.dot(&diff));

        let log_det = self.precision.diag().map(|x| x.ln()).sum();
        let n_features = sample.len() as f64;

        -0.5 * (mahalanobis + n_features * (2.0 * std::f64::consts::PI).ln() - log_det)
    }

    pub fn most_likely_class(&self) -> Option<i32> {
        self.class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&class, _)| class)
    }
}

/// Stick-breaking representation for Dirichlet Process
#[derive(Debug, Clone)]
pub struct StickBreaking {
    pub stick_lengths: Array1<f64>,
    pub weights: Array1<f64>,
    pub concentration: f64,
    pub truncation: usize,
}

impl StickBreaking {
    pub fn new(
        concentration: f64,
        truncation: usize,
        rng: &mut scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
    ) -> Result<Self, DirichletProcessError> {
        let beta = Beta::new(1.0, concentration).map_err(|e| {
            DirichletProcessError::StickBreakingError(format!(
                "Failed to create Beta distribution: {}",
                e
            ))
        })?;

        let stick_lengths: Vec<f64> = (0..truncation).map(|_| beta.sample(rng)).collect();

        let stick_array = Array1::from_vec(stick_lengths);
        let weights = Self::compute_weights(&stick_array);

        Ok(Self {
            stick_lengths: stick_array,
            weights,
            concentration,
            truncation,
        })
    }

    fn compute_weights(stick_lengths: &Array1<f64>) -> Array1<f64> {
        let mut weights = Array1::zeros(stick_lengths.len());
        let mut remaining = 1.0;

        for (i, &stick) in stick_lengths.iter().enumerate() {
            weights[i] = stick * remaining;
            remaining *= 1.0 - stick;

            // Prevent numerical issues
            if remaining < 1e-10 {
                break;
            }
        }

        // Normalize to ensure exact sum of 1.0
        let total = weights.sum();
        if total > 0.0 {
            weights /= total;
        }

        weights
    }

    pub fn sample_component(
        &self,
        rng: &mut scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
    ) -> usize {
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let u = uniform.sample(rng);

        let mut cumulative = 0.0;
        for (i, &weight) in self.weights.iter().enumerate() {
            cumulative += weight;
            if u <= cumulative {
                return i;
            }
        }

        self.truncation - 1 // Fallback to last component
    }

    pub fn update_sticks(
        &mut self,
        assignments: &[usize],
        rng: &mut scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
    ) -> Result<(), DirichletProcessError> {
        let n_samples = assignments.len();

        for k in 0..self.truncation {
            let n_k = assignments.iter().filter(|&&z| z == k).count();
            let n_greater = assignments.iter().filter(|&&z| z > k).count();

            let alpha = 1.0 + n_k as f64;
            let beta = self.concentration + n_greater as f64;

            let beta_dist = Beta::new(alpha, beta).map_err(|e| {
                DirichletProcessError::StickBreakingError(format!(
                    "Failed to update stick {}: {}",
                    k, e
                ))
            })?;

            self.stick_lengths[k] = beta_dist.sample(rng);
        }

        self.weights = Self::compute_weights(&self.stick_lengths);
        Ok(())
    }
}

/// Dirichlet Process Naive Bayes classifier
pub struct DirichletProcessNB {
    config: DirichletProcessConfig,
    components: Vec<DirichletComponent>,
    component_assignments: Vec<usize>,
    stick_breaking: Option<StickBreaking>,
    rng: scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
    n_features: usize,
    classes: HashSet<i32>,
    is_fitted: bool,
}

impl DirichletProcessNB {
    pub fn new(config: DirichletProcessConfig) -> Self {
        let rng = match config.random_seed {
            Some(seed) => scirs2_core::random::CoreRandom::seed_from_u64(seed),
            None => {
                scirs2_core::random::CoreRandom::from_rng(&mut scirs2_core::random::thread_rng())
            }
        };

        Self {
            config,
            components: Vec::new(),
            component_assignments: Vec::new(),
            stick_breaking: None,
            rng,
            n_features: 0,
            classes: HashSet::new(),
            is_fitted: false,
        }
    }

    /// Fit the Dirichlet Process Naive Bayes to training data
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<i32>) -> Result<(), DirichletProcessError> {
        if X.nrows() != y.len() {
            return Err(DirichletProcessError::DimensionMismatch);
        }
        if X.nrows() == 0 {
            return Err(DirichletProcessError::EmptyDataset);
        }

        self.n_features = X.ncols();
        self.classes = y.iter().cloned().collect();

        // Initialize components and assignments
        self.initialize_components(X, y)?;

        // Run inference
        match self.config.inference_method {
            InferenceMethod::Gibbs => self.gibbs_sampling(X, y)?,
            InferenceMethod::VariationalBayes => self.variational_inference(X, y)?,
            InferenceMethod::StickBreaking => self.stick_breaking_inference(X, y)?,
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Predict class labels for test samples
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, DirichletProcessError> {
        if !self.is_fitted {
            return Err(DirichletProcessError::NumericalError(
                "Model not fitted".to_string(),
            ));
        }
        if X.ncols() != self.n_features {
            return Err(DirichletProcessError::DimensionMismatch);
        }

        let mut predictions = Vec::new();

        for sample in X.axis_iter(scirs2_core::ndarray::Axis(0)) {
            let sample_array = sample.to_owned();
            let predicted_class = self.predict_single_sample(&sample_array)?;
            predictions.push(predicted_class);
        }

        Ok(Array1::from_vec(predictions))
    }

    /// Predict class probabilities for test samples
    pub fn predict_proba(
        &self,
        X: &Array2<f64>,
    ) -> Result<HashMap<i32, Array1<f64>>, DirichletProcessError> {
        if !self.is_fitted {
            return Err(DirichletProcessError::NumericalError(
                "Model not fitted".to_string(),
            ));
        }
        if X.ncols() != self.n_features {
            return Err(DirichletProcessError::DimensionMismatch);
        }

        let mut class_probabilities: HashMap<i32, Vec<f64>> = HashMap::new();
        for &class in &self.classes {
            class_probabilities.insert(class, Vec::new());
        }

        for sample in X.axis_iter(scirs2_core::ndarray::Axis(0)) {
            let sample_array = sample.to_owned();
            let proba_map = self.predict_proba_single(&sample_array)?;

            for &class in &self.classes {
                let prob = proba_map.get(&class).copied().unwrap_or(0.0);
                class_probabilities.get_mut(&class).unwrap().push(prob);
            }
        }

        let mut result = HashMap::new();
        for (class, probs) in class_probabilities {
            result.insert(class, Array1::from_vec(probs));
        }

        Ok(result)
    }

    fn initialize_components(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), DirichletProcessError> {
        let n_samples = X.nrows();

        // Initialize with K-means-like clustering or random assignment
        self.components.clear();
        self.component_assignments = vec![0; n_samples];

        // Start with one component per class
        let unique_classes: Vec<i32> = self.classes.iter().cloned().collect();

        for (i, &class) in unique_classes.iter().enumerate() {
            let mut component = DirichletComponent::new(self.n_features);
            component.weight = 1.0 / unique_classes.len() as f64;

            // Initialize with samples from this class
            for (sample_idx, &sample_class) in y.iter().enumerate() {
                if sample_class == class {
                    let sample = X.row(sample_idx).to_owned();
                    component.add_sample(&sample, class);
                    self.component_assignments[sample_idx] = i;
                }
            }

            self.components.push(component);
        }

        // Initialize stick-breaking if needed
        if matches!(self.config.inference_method, InferenceMethod::StickBreaking) {
            self.stick_breaking = Some(StickBreaking::new(
                self.config.concentration_alpha,
                self.config.stick_breaking_truncation,
                &mut self.rng,
            )?);
        }

        Ok(())
    }

    fn gibbs_sampling(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), DirichletProcessError> {
        let n_samples = X.nrows();

        for iteration in 0..self.config.max_iterations {
            let mut component_changed = false;

            // Gibbs sweep: reassign each sample
            for sample_idx in 0..n_samples {
                let sample = X.row(sample_idx).to_owned();
                let class = y[sample_idx];
                let old_assignment = self.component_assignments[sample_idx];

                // Remove sample from current component
                if old_assignment < self.components.len() {
                    self.components[old_assignment].remove_sample(&sample, class);
                }

                // Compute assignment probabilities
                let new_assignment = self.sample_component_assignment(&sample, class)?;

                if new_assignment != old_assignment {
                    component_changed = true;
                }

                // Add sample to new component
                if new_assignment >= self.components.len() {
                    // Create new component
                    let mut new_component = DirichletComponent::new(self.n_features);
                    new_component.add_sample(&sample, class);
                    self.components.push(new_component);
                }

                self.components[new_assignment].add_sample(&sample, class);
                self.component_assignments[sample_idx] = new_assignment;
            }

            // Remove empty components
            self.cleanup_empty_components();

            // Update component weights
            self.update_component_weights();

            // Check convergence
            if !component_changed && iteration > 10 {
                break;
            }
        }

        Ok(())
    }

    fn sample_component_assignment(
        &mut self,
        sample: &Array1<f64>,
        class: i32,
    ) -> Result<usize, DirichletProcessError> {
        let mut log_probs = Vec::new();
        let n_samples = self.component_assignments.len() as f64;

        // Existing components
        for (k, component) in self.components.iter().enumerate() {
            let log_likelihood = component.log_likelihood(sample);
            let log_prior = if component.sample_count > 0 {
                (component.sample_count as f64
                    / (n_samples - 1.0 + self.config.concentration_alpha))
                    .ln()
            } else {
                f64::NEG_INFINITY
            };
            log_probs.push(log_likelihood + log_prior);
        }

        // New component (Chinese Restaurant Process)
        let new_component_prob =
            self.config.concentration_alpha / (n_samples - 1.0 + self.config.concentration_alpha);
        let base_likelihood = self.base_distribution_likelihood(sample);
        log_probs.push(new_component_prob.ln() + base_likelihood);

        // Sample from categorical distribution
        let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = log_probs
            .iter()
            .map(|&log_p| (log_p - max_log_prob).exp())
            .collect();
        let prob_sum: f64 = probs.iter().sum();

        if prob_sum <= 0.0 {
            return Ok(0); // Fallback
        }

        let normalized_probs: Vec<f64> = probs.iter().map(|&p| p / prob_sum).collect();

        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let u = uniform.sample(&mut self.rng);

        let mut cumulative = 0.0;
        for (i, &prob) in normalized_probs.iter().enumerate() {
            cumulative += prob;
            if u <= cumulative {
                return Ok(i);
            }
        }

        Ok(self.components.len()) // New component
    }

    fn base_distribution_likelihood(&self, sample: &Array1<f64>) -> f64 {
        // Simplified base distribution (standard Gaussian)
        let log_likelihood = sample
            .iter()
            .map(|&x| -0.5 * (x.powi(2) + (2.0 * std::f64::consts::PI).ln()))
            .sum();
        log_likelihood
    }

    fn cleanup_empty_components(&mut self) {
        let mut keep_indices = Vec::new();
        let mut component_mapping = HashMap::new();

        for (i, component) in self.components.iter().enumerate() {
            if component.sample_count > 0 {
                component_mapping.insert(i, keep_indices.len());
                keep_indices.push(i);
            }
        }

        // Update assignments
        for assignment in &mut self.component_assignments {
            if let Some(&new_idx) = component_mapping.get(assignment) {
                *assignment = new_idx;
            } else {
                *assignment = 0; // Fallback
            }
        }

        // Keep only non-empty components
        let mut new_components = Vec::new();
        for &i in &keep_indices {
            new_components.push(self.components[i].clone());
        }
        self.components = new_components;
    }

    fn update_component_weights(&mut self) {
        let total_samples = self.component_assignments.len() as f64;

        for component in &mut self.components {
            component.weight = component.sample_count as f64 / total_samples;
        }
    }

    fn variational_inference(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), DirichletProcessError> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let max_components = self.config.max_components.min(n_samples);

        // Initialize variational parameters
        let mut responsibilities = Array2::<f64>::zeros((n_samples, max_components));
        let mut alpha =
            vec![self.config.concentration_alpha / max_components as f64; max_components];
        let mut beta = vec![1.0; max_components];

        // Initialize with K-means-like initialization
        self.initialize_variational_components(X, y, max_components)?;

        let mut prev_elbo = f64::NEG_INFINITY;

        for iteration in 0..self.config.max_iterations {
            // E-step: Update responsibilities using current component parameters
            for i in 0..n_samples {
                let sample = X.row(i);
                let sample_array = sample.to_owned();

                // Compute log-likelihood for each component
                let mut log_probs = Vec::with_capacity(max_components);
                for k in 0..max_components.min(self.components.len()) {
                    let log_like = self.components[k].log_likelihood(&sample_array);
                    let log_weight = (alpha[k] / alpha.iter().sum::<f64>()).ln();
                    log_probs.push(log_weight + log_like);
                }

                // Normalize to get responsibilities using log-sum-exp trick
                if !log_probs.is_empty() {
                    let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log_prob).exp()).sum();
                    let log_sum = max_log_prob + sum_exp.ln();

                    for k in 0..log_probs.len() {
                        responsibilities[[i, k]] = (log_probs[k] - log_sum).exp();
                    }
                }
            }

            // M-step: Update component parameters based on responsibilities
            for k in 0..max_components.min(self.components.len()) {
                let resp_k = responsibilities.column(k);
                let n_k = resp_k.sum();

                if n_k > 1e-10 {
                    // Update component weight (Dirichlet variational parameter)
                    alpha[k] = self.config.concentration_alpha / max_components as f64 + n_k;

                    // Update mean with weighted samples
                    let mut new_mean = Array1::zeros(n_features);
                    for i in 0..n_samples {
                        let sample = X.row(i);
                        new_mean = new_mean + sample.to_owned() * responsibilities[[i, k]];
                    }
                    new_mean /= n_k;
                    self.components[k].mean = new_mean;

                    // Update precision matrix (simplified - using diagonal)
                    let mut new_precision = Array2::zeros((n_features, n_features));
                    for i in 0..n_samples {
                        let sample = X.row(i).to_owned();
                        let diff = &sample - &self.components[k].mean;
                        let outer = diff
                            .clone()
                            .insert_axis(scirs2_core::ndarray::Axis(1))
                            .dot(&diff.clone().insert_axis(scirs2_core::ndarray::Axis(0)));
                        new_precision = new_precision + outer * responsibilities[[i, k]];
                    }

                    // Regularize and invert (simplified using diagonal + small ridge)
                    for d in 0..n_features {
                        let var_d: f64 = new_precision[[d, d]] / n_k + 1e-6;
                        new_precision[[d, d]] = var_d.recip();
                    }
                    self.components[k].precision = new_precision;

                    // Update sample counts and class counts
                    self.components[k].sample_count = n_k as usize;
                    self.components[k].class_counts.clear();
                    for i in 0..n_samples {
                        if responsibilities[[i, k]] > 0.01 {
                            let class = y[i];
                            *self.components[k].class_counts.entry(class).or_insert(0) +=
                                (responsibilities[[i, k]] * 100.0) as usize;
                        }
                    }
                }
            }

            // Compute ELBO (Evidence Lower Bound) for convergence checking
            let elbo = self.compute_elbo(X, &responsibilities, &alpha);

            // Check convergence
            if (elbo - prev_elbo).abs() < self.config.tolerance {
                break;
            }
            prev_elbo = elbo;

            // Prune components with very small weight
            self.prune_small_components(0.01);
        }

        // Update final component weights and assignments
        for i in 0..n_samples {
            let mut max_resp = 0.0;
            let mut max_k = 0;
            for k in 0..max_components.min(self.components.len()) {
                if responsibilities[[i, k]] > max_resp {
                    max_resp = responsibilities[[i, k]];
                    max_k = k;
                }
            }
            self.component_assignments[i] = max_k;
        }

        self.update_component_weights();

        Ok(())
    }

    /// Initialize variational components for variational inference
    fn initialize_variational_components(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        max_components: usize,
    ) -> Result<(), DirichletProcessError> {
        self.components.clear();
        let n_samples = X.nrows();

        // Initialize with stratified sampling from classes
        let unique_classes: HashSet<i32> = y.iter().cloned().collect();
        let n_init = max_components.min(unique_classes.len() * 2);

        for i in 0..n_init {
            let idx = (i * n_samples / n_init) % n_samples;
            let mut component = DirichletComponent::new(self.n_features);
            component.mean = X.row(idx).to_owned();
            component.weight = 1.0 / n_init as f64;
            self.components.push(component);
        }

        Ok(())
    }

    /// Compute Evidence Lower Bound (ELBO) for convergence monitoring
    fn compute_elbo(&self, X: &Array2<f64>, responsibilities: &Array2<f64>, alpha: &[f64]) -> f64 {
        let n_samples = X.nrows();
        let n_components = self.components.len();

        let mut elbo = 0.0;

        // Expected log-likelihood term
        for i in 0..n_samples {
            let sample = X.row(i).to_owned();
            for k in 0..n_components {
                let resp = responsibilities[[i, k]];
                if resp > 1e-10 {
                    let log_like = self.components[k].log_likelihood(&sample);
                    elbo += resp * log_like;
                }
            }
        }

        // KL divergence term for Dirichlet (simplified)
        let alpha_sum: f64 = alpha.iter().sum();
        let alpha_0 = self.config.concentration_alpha;

        for &a in alpha {
            if a > 0.0 {
                elbo += (a - alpha_0 / n_components as f64) * (a.ln() - alpha_sum.ln());
            }
        }

        // Entropy term for responsibilities
        for i in 0..n_samples {
            for k in 0..n_components {
                let resp = responsibilities[[i, k]];
                if resp > 1e-10 {
                    elbo -= resp * resp.ln();
                }
            }
        }

        elbo
    }

    /// Prune components with weight below threshold
    fn prune_small_components(&mut self, threshold: f64) {
        self.components
            .retain(|c| c.weight >= threshold || c.sample_count > 0);
    }

    fn stick_breaking_inference(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), DirichletProcessError> {
        let n_samples = X.nrows();

        for _iteration in 0..self.config.max_iterations {
            // Sample component assignments using stick-breaking weights
            if let Some(ref stick_breaking) = self.stick_breaking {
                for sample_idx in 0..n_samples {
                    let new_assignment = stick_breaking.sample_component(&mut self.rng);
                    self.component_assignments[sample_idx] = new_assignment;
                }
            }

            // Update stick-breaking parameters
            if let Some(ref mut stick_breaking) = self.stick_breaking {
                stick_breaking.update_sticks(&self.component_assignments, &mut self.rng)?;
            }

            // Update component parameters
            self.update_components_from_assignments(X, y)?;
        }

        Ok(())
    }

    fn update_components_from_assignments(
        &mut self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<(), DirichletProcessError> {
        // Clear current components
        self.components.clear();

        let max_component = self
            .component_assignments
            .iter()
            .max()
            .copied()
            .unwrap_or(0);

        // Initialize components
        for _ in 0..=max_component {
            self.components
                .push(DirichletComponent::new(self.n_features));
        }

        // Add samples to components
        for (sample_idx, &assignment) in self.component_assignments.iter().enumerate() {
            if assignment < self.components.len() {
                let sample = X.row(sample_idx).to_owned();
                let class = y[sample_idx];
                self.components[assignment].add_sample(&sample, class);
            }
        }

        self.update_component_weights();
        Ok(())
    }

    fn predict_single_sample(&self, sample: &Array1<f64>) -> Result<i32, DirichletProcessError> {
        let mut best_class = 0;
        let mut best_score = f64::NEG_INFINITY;

        for component in &self.components {
            if let Some(class) = component.most_likely_class() {
                let log_likelihood = component.log_likelihood(sample);
                let score = log_likelihood + component.weight.ln();

                if score > best_score {
                    best_score = score;
                    best_class = class;
                }
            }
        }

        if self.classes.contains(&best_class) {
            Ok(best_class)
        } else {
            Ok(*self.classes.iter().next().unwrap_or(&0))
        }
    }

    fn predict_proba_single(
        &self,
        sample: &Array1<f64>,
    ) -> Result<HashMap<i32, f64>, DirichletProcessError> {
        let mut class_scores: HashMap<i32, f64> = HashMap::new();

        // Initialize scores for all classes
        for &class in &self.classes {
            class_scores.insert(class, f64::NEG_INFINITY);
        }

        // Compute scores for each component
        for component in &self.components {
            if let Some(class) = component.most_likely_class() {
                let log_likelihood = component.log_likelihood(sample);
                let score = log_likelihood + component.weight.ln();

                let current_score = class_scores
                    .get(&class)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                if score > current_score {
                    class_scores.insert(class, score);
                }
            }
        }

        // Convert log scores to probabilities
        let max_score = class_scores
            .values()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: HashMap<i32, f64> = class_scores
            .iter()
            .map(|(&class, &score)| (class, (score - max_score).exp()))
            .collect();

        let total: f64 = exp_scores.values().sum();
        let probabilities: HashMap<i32, f64> = exp_scores
            .iter()
            .map(|(&class, &score)| (class, score / total))
            .collect();

        Ok(probabilities)
    }

    /// Get the number of active components
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Get component weights
    pub fn component_weights(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.weight).collect()
    }

    /// Get component assignments for training data
    pub fn get_component_assignments(&self) -> &[usize] {
        &self.component_assignments
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_component() {
        let mut component = DirichletComponent::new(2);
        let sample = Array1::from_vec(vec![1.0, 2.0]);

        component.add_sample(&sample, 1);
        assert_eq!(component.sample_count, 1);
        assert_eq!(component.class_counts[&1], 1);

        let likelihood = component.log_likelihood(&sample);
        assert!(likelihood.is_finite());
    }

    #[test]
    fn test_stick_breaking() {
        let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(42);
        let stick_breaking = StickBreaking::new(1.0, 5, &mut rng).unwrap();

        assert_eq!(stick_breaking.weights.len(), 5);
        assert!((stick_breaking.weights.sum() - 1.0).abs() < 1e-6);

        let component = stick_breaking.sample_component(&mut rng);
        assert!(component < 5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dirichlet_process_nb_basic() {
        let config = DirichletProcessConfig::default();
        let mut dp_nb = DirichletProcessNB::new(config);

        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1, 3.0, 3.0, 3.1, 3.1],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        assert!(dp_nb.fit(&X, &y).is_ok());
        assert!(dp_nb.is_fitted);
        assert!(dp_nb.n_components() > 0);

        let predictions = dp_nb.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        let probabilities = dp_nb.predict_proba(&X).unwrap();
        assert_eq!(probabilities.len(), 3); // 3 classes
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gibbs_sampling() {
        let mut config = DirichletProcessConfig::default();
        config.inference_method = InferenceMethod::Gibbs;
        config.max_iterations = 10;

        let mut dp_nb = DirichletProcessNB::new(config);

        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        assert!(dp_nb.fit(&X, &y).is_ok());
        assert!(dp_nb.n_components() > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stick_breaking_inference() {
        let mut config = DirichletProcessConfig::default();
        config.inference_method = InferenceMethod::StickBreaking;
        config.max_iterations = 5;

        let mut dp_nb = DirichletProcessNB::new(config);

        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 2.0, 2.0, 2.1, 2.1]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        assert!(dp_nb.fit(&X, &y).is_ok());
        assert!(dp_nb.stick_breaking.is_some());
    }
}
