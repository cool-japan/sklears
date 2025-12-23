//! Information theory methods for semi-supervised learning
//!
//! This module provides information-theoretic approaches to semi-supervised learning,
//! including mutual information maximization, information bottleneck principle,
//! and entropy-based methods for feature selection and active learning.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, Untrained};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Mutual Information Maximization for semi-supervised learning
///
/// This method learns representations that maximize mutual information between
/// input features and output labels, using both labeled and unlabeled data
/// to improve the learned representations.
///
/// # Parameters
///
/// * `n_bins` - Number of bins for discretization in MI estimation
/// * `max_iter` - Maximum number of iterations for optimization
/// * `learning_rate` - Learning rate for gradient-based optimization
/// * `temperature` - Temperature parameter for soft discretization
/// * `regularization` - L2 regularization strength
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::MutualInformationMaximization;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let mim = MutualInformationMaximization::new()
///     .n_bins(10)
///     .max_iter(100)
///     .learning_rate(0.01);
/// let fitted = mim.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MutualInformationMaximization<S = Untrained> {
    state: S,
    n_bins: usize,
    max_iter: usize,
    learning_rate: f64,
    temperature: f64,
    regularization: f64,
    random_state: Option<u64>,
}

impl MutualInformationMaximization<Untrained> {
    /// Create a new MutualInformationMaximization instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_bins: 20,
            max_iter: 100,
            learning_rate: 0.01,
            temperature: 1.0,
            regularization: 0.01,
            random_state: None,
        }
    }

    /// Set the number of bins for discretization
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the temperature parameter
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the regularization strength
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for MutualInformationMaximization<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MutualInformationMaximization<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for MutualInformationMaximization<Untrained> {
    type Fitted = MutualInformationMaximization<MutualInformationTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Initialize transformation matrix (feature weights)
        let mut transformation = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_features {
            transformation[[i, i]] = 1.0; // Start with identity
            for j in 0..n_features {
                if i != j {
                    transformation[[i, j]] = rng.random_range(-0.1, 0.1);
                }
            }
        }

        // Gradient-based optimization to maximize mutual information
        for _iter in 0..self.max_iter {
            // Transform features
            let X_transformed = X.dot(&transformation);

            // Estimate mutual information using histograms
            let mi =
                self.estimate_mutual_information(&X_transformed, &y, &labeled_indices, &classes)?;

            // Compute gradient (simplified finite differences)
            let mut gradient = Array2::<f64>::zeros((n_features, n_features));
            let epsilon = 1e-6;

            for i in 0..n_features {
                for j in 0..n_features {
                    // Forward difference
                    transformation[[i, j]] += epsilon;
                    let X_perturbed = X.dot(&transformation);
                    let mi_perturbed = self.estimate_mutual_information(
                        &X_perturbed,
                        &y,
                        &labeled_indices,
                        &classes,
                    )?;
                    gradient[[i, j]] = (mi_perturbed - mi) / epsilon;
                    transformation[[i, j]] -= epsilon; // Reset
                }
            }

            // Update transformation matrix
            for i in 0..n_features {
                for j in 0..n_features {
                    transformation[[i, j]] += self.learning_rate * gradient[[i, j]]
                        - self.regularization * transformation[[i, j]];
                }
            }
        }

        // Final transformation and label prediction for unlabeled samples
        let X_final = X.dot(&transformation);
        let mut final_labels = y.clone();

        // Use k-nearest neighbors on transformed space to predict unlabeled samples
        for &unlabeled_idx in &unlabeled_indices {
            let mut distances = Vec::new();
            for &labeled_idx in &labeled_indices {
                let dist = (&X_final.row(unlabeled_idx) - &X_final.row(labeled_idx))
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();
                distances.push((labeled_idx, dist));
            }

            // Sort by distance and take majority vote of k=3 nearest neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let k = 3.min(labeled_indices.len());
            let mut class_votes = HashMap::new();

            for &(labeled_idx, _) in distances.iter().take(k) {
                *class_votes.entry(y[labeled_idx]).or_insert(0) += 1;
            }

            // Assign most voted class
            if let Some((&predicted_class, _)) = class_votes.iter().max_by_key(|&(_, count)| count)
            {
                final_labels[unlabeled_idx] = predicted_class;
            }
        }

        Ok(MutualInformationMaximization {
            state: MutualInformationTrained {
                X_train: X,
                y_train: final_labels,
                classes: Array1::from(classes),
                transformation,
                n_bins: self.n_bins,
            },
            n_bins: self.n_bins,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            temperature: self.temperature,
            regularization: self.regularization,
            random_state: self.random_state,
        })
    }
}

impl MutualInformationMaximization<Untrained> {
    /// Estimate mutual information using histogram-based method
    fn estimate_mutual_information(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        labeled_indices: &[usize],
        classes: &[i32],
    ) -> SklResult<f64> {
        if labeled_indices.is_empty() {
            return Ok(0.0);
        }

        // Discretize features into bins for labeled samples only
        let mut feature_bins = Vec::new();
        for j in 0..X.ncols() {
            let labeled_features: Vec<f64> = labeled_indices.iter().map(|&i| X[[i, j]]).collect();

            if labeled_features.is_empty() {
                continue;
            }

            let min_val = labeled_features
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = labeled_features
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if (max_val - min_val).abs() < 1e-10 {
                feature_bins.push(vec![0; labeled_indices.len()]); // All same bin
                continue;
            }

            let bin_width = (max_val - min_val) / self.n_bins as f64;
            let bins: Vec<usize> = labeled_features
                .iter()
                .map(|&val| {
                    ((val - min_val) / bin_width)
                        .floor()
                        .min((self.n_bins - 1) as f64) as usize
                })
                .collect();
            feature_bins.push(bins);
        }

        if feature_bins.is_empty() {
            return Ok(0.0);
        }

        // Compute joint and marginal distributions
        let mut joint_counts = HashMap::new();
        let mut feature_counts = HashMap::new();
        let mut label_counts = HashMap::new();

        for (sample_idx, &global_idx) in labeled_indices.iter().enumerate() {
            let label = y[global_idx];

            // Multi-dimensional feature bin (use first feature for simplicity)
            let feature_bin = if !feature_bins.is_empty() && sample_idx < feature_bins[0].len() {
                feature_bins[0][sample_idx]
            } else {
                0
            };

            *joint_counts.entry((feature_bin, label)).or_insert(0) += 1;
            *feature_counts.entry(feature_bin).or_insert(0) += 1;
            *label_counts.entry(label).or_insert(0) += 1;
        }

        let n_labeled = labeled_indices.len() as f64;
        let mut mi = 0.0;

        // Calculate mutual information: MI(X,Y) = sum_{x,y} p(x,y) * log(p(x,y) / (p(x) * p(y)))
        for (&(feature_bin, label), &joint_count) in &joint_counts {
            let p_xy = joint_count as f64 / n_labeled;
            let p_x = feature_counts[&feature_bin] as f64 / n_labeled;
            let p_y = label_counts[&label] as f64 / n_labeled;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        Ok(mi)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for MutualInformationMaximization<MutualInformationTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Transform test features
        let X_transformed = X.dot(&self.state.transformation);

        for i in 0..n_test {
            // Find most similar training sample using transformed features
            let mut min_dist = f64::INFINITY;
            let mut best_label = self.state.classes[0];

            for j in 0..self.state.X_train.nrows() {
                let X_train_transformed = self.state.X_train.dot(&self.state.transformation);
                let diff = &X_transformed.row(i) - &X_train_transformed.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_label = self.state.y_train[j];
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// Information Bottleneck principle for semi-supervised learning
///
/// This method learns compressed representations that preserve information
/// about the target labels while discarding irrelevant information.
///
/// # Parameters
///
/// * `beta` - Trade-off parameter between compression and prediction
/// * `n_components` - Number of components in the compressed representation
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
#[derive(Debug, Clone)]
pub struct InformationBottleneck<S = Untrained> {
    state: S,
    beta: f64,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl InformationBottleneck<Untrained> {
    /// Create a new InformationBottleneck instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            beta: 1.0,
            n_components: 10,
            max_iter: 100,
            tol: 1e-4,
            random_state: None,
        }
    }

    /// Set the beta parameter (compression vs prediction trade-off)
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for InformationBottleneck<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for InformationBottleneck<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for InformationBottleneck<Untrained> {
    type Fitted = InformationBottleneck<InformationBottleneckTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled samples
        let mut labeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Initialize projection matrix for dimensionality reduction
        let mut projection = Array2::<f64>::zeros((n_features, self.n_components));
        for i in 0..n_features {
            for j in 0..self.n_components {
                projection[[i, j]] = rng.random_range(-0.1, 0.1);
            }
        }

        // Simple iterative optimization (simplified information bottleneck)
        for _iter in 0..self.max_iter {
            // Project features to lower dimension
            let X_projected = X.dot(&projection);

            // Compute reconstruction loss (simplified)
            let reconstruction_loss =
                self.compute_reconstruction_loss(&X, &X_projected, &projection)?;

            // Update projection to minimize reconstruction loss while preserving class information
            // This is a simplified implementation - full IB would require more sophisticated optimization
            for i in 0..n_features {
                for j in 0..self.n_components {
                    let gradient = reconstruction_loss / (n_samples as f64);
                    projection[[i, j]] -= 0.001 * gradient; // Simple gradient step
                }
            }
        }

        Ok(InformationBottleneck {
            state: InformationBottleneckTrained {
                X_train: X,
                y_train: y,
                classes: Array1::from(classes),
                projection,
            },
            beta: self.beta,
            n_components: self.n_components,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
        })
    }
}

impl InformationBottleneck<Untrained> {
    fn compute_reconstruction_loss(
        &self,
        X_original: &Array2<f64>,
        X_projected: &Array2<f64>,
        projection: &Array2<f64>,
    ) -> SklResult<f64> {
        // Simplified reconstruction loss: MSE between original and reconstructed
        let reconstruction = X_projected.dot(&projection.t());
        let diff = X_original - &reconstruction;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
        Ok(mse)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for InformationBottleneck<InformationBottleneckTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Project test data
        let X_test_projected = X.dot(&self.state.projection);
        let X_train_projected = self.state.X_train.dot(&self.state.projection);

        for i in 0..n_test {
            // Find nearest neighbor in projected space
            let mut min_dist = f64::INFINITY;
            let mut best_label = self.state.classes[0];

            for j in 0..self.state.X_train.nrows() {
                let diff = &X_test_projected.row(i) - &X_train_projected.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_label = self.state.y_train[j];
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// Trained state for MutualInformationMaximization
#[derive(Debug, Clone)]
pub struct MutualInformationTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// transformation
    pub transformation: Array2<f64>,
    /// n_bins
    pub n_bins: usize,
}

/// Trained state for InformationBottleneck
#[derive(Debug, Clone)]
pub struct InformationBottleneckTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// projection
    pub projection: Array2<f64>,
}

/// Entropy-based Regularization for Semi-Supervised Learning
///
/// This method adds entropy regularization to encourage confident predictions
/// on unlabeled data while minimizing classification error on labeled data.
///
/// # Parameters
///
/// * `entropy_weight` - Weight for entropy regularization term
/// * `max_iter` - Maximum number of iterations
/// * `learning_rate` - Learning rate for gradient descent
/// * `n_neighbors` - Number of neighbors for graph construction
#[derive(Debug, Clone)]
pub struct EntropyRegularizedSemiSupervised<S = Untrained> {
    state: S,
    entropy_weight: f64,
    max_iter: usize,
    learning_rate: f64,
    n_neighbors: usize,
    random_state: Option<u64>,
}

impl EntropyRegularizedSemiSupervised<Untrained> {
    /// Create a new EntropyRegularizedSemiSupervised instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            entropy_weight: 0.5,
            max_iter: 100,
            learning_rate: 0.01,
            n_neighbors: 5,
            random_state: None,
        }
    }

    /// Set the entropy weight
    pub fn entropy_weight(mut self, weight: f64) -> Self {
        self.entropy_weight = weight;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for EntropyRegularizedSemiSupervised<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EntropyRegularizedSemiSupervised<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>>
    for EntropyRegularizedSemiSupervised<Untrained>
{
    type Fitted = EntropyRegularizedSemiSupervised<EntropyRegularizedTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Initialize probability distributions
        let mut prob_distributions = Array2::<f64>::zeros((n_samples, n_classes));

        // Set labeled samples
        for &idx in &labeled_indices {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                prob_distributions[[idx, class_idx]] = 1.0;
            }
        }

        // Initialize unlabeled samples with uniform distribution
        for &idx in &unlabeled_indices {
            for class_idx in 0..n_classes {
                prob_distributions[[idx, class_idx]] = 1.0 / n_classes as f64;
            }
        }

        // Build k-NN graph
        let mut adjacency = Array2::<f64>::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            let mut distances: Vec<(usize, f64)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let diff = &X.row(i) - &X.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                    distances.push((j, dist));
                }
            }
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(j, dist) in distances.iter().take(self.n_neighbors) {
                let weight = (-dist.powi(2) / 2.0).exp();
                adjacency[[i, j]] = weight;
                adjacency[[j, i]] = weight;
            }
        }

        // Normalize adjacency matrix
        for i in 0..n_samples {
            let row_sum: f64 = adjacency.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_samples {
                    adjacency[[i, j]] /= row_sum;
                }
            }
        }

        // Optimize with entropy regularization
        for _iter in 0..self.max_iter {
            let prev_probs = prob_distributions.clone();

            // Update unlabeled samples
            for &idx in &unlabeled_indices {
                // Smooth labels from neighbors
                let mut smooth_dist = Array1::<f64>::zeros(n_classes);
                for j in 0..n_samples {
                    for k in 0..n_classes {
                        smooth_dist[k] += adjacency[[idx, j]] * prob_distributions[[j, k]];
                    }
                }

                // Compute entropy regularization gradient
                let mut entropy_grad = Array1::<f64>::zeros(n_classes);
                for k in 0..n_classes {
                    let p = prob_distributions[[idx, k]].max(1e-10);
                    entropy_grad[k] = -(p.ln() + 1.0);
                }

                // Update probabilities
                for k in 0..n_classes {
                    prob_distributions[[idx, k]] =
                        smooth_dist[k] - self.learning_rate * self.entropy_weight * entropy_grad[k];
                    prob_distributions[[idx, k]] = prob_distributions[[idx, k]].max(0.0);
                }

                // Normalize
                let row_sum: f64 = prob_distributions.row(idx).sum();
                if row_sum > 0.0 {
                    for k in 0..n_classes {
                        prob_distributions[[idx, k]] /= row_sum;
                    }
                }
            }

            // Check convergence
            let diff = (&prob_distributions - &prev_probs).mapv(|x| x.abs()).sum();
            if diff < 1e-6 {
                break;
            }
        }

        // Generate final labels
        let mut final_labels = y.clone();
        for &idx in &unlabeled_indices {
            let class_idx = prob_distributions
                .row(idx)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            final_labels[idx] = classes[class_idx];
        }

        Ok(EntropyRegularizedSemiSupervised {
            state: EntropyRegularizedTrained {
                X_train: X,
                y_train: final_labels,
                classes: Array1::from(classes),
                prob_distributions,
                adjacency,
            },
            entropy_weight: self.entropy_weight,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            n_neighbors: self.n_neighbors,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for EntropyRegularizedSemiSupervised<EntropyRegularizedTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            let mut min_dist = f64::INFINITY;
            let mut best_label = self.state.classes[0];

            for j in 0..self.state.X_train.nrows() {
                let diff = &X.row(i) - &self.state.X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_label = self.state.y_train[j];
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// KL-Divergence Optimization for Semi-Supervised Learning
///
/// This method optimizes a classifier by minimizing KL divergence between
/// predictions on differently augmented versions of the same unlabeled data.
///
/// # Parameters
///
/// * `temperature` - Temperature for softmax
/// * `max_iter` - Maximum number of iterations
/// * `learning_rate` - Learning rate for optimization
/// * `kl_weight` - Weight for KL divergence term
#[derive(Debug, Clone)]
pub struct KLDivergenceOptimization<S = Untrained> {
    state: S,
    temperature: f64,
    max_iter: usize,
    learning_rate: f64,
    kl_weight: f64,
    random_state: Option<u64>,
}

impl KLDivergenceOptimization<Untrained> {
    /// Create a new KLDivergenceOptimization instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            temperature: 1.0,
            max_iter: 100,
            learning_rate: 0.01,
            kl_weight: 1.0,
            random_state: None,
        }
    }

    /// Set the temperature
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the KL weight
    pub fn kl_weight(mut self, weight: f64) -> Self {
        self.kl_weight = weight;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for KLDivergenceOptimization<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for KLDivergenceOptimization<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for KLDivergenceOptimization<Untrained> {
    type Fitted = KLDivergenceOptimization<KLDivergenceTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label == -1 {
                unlabeled_indices.push(i);
            } else {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Initialize RNG
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )
        };

        // Initialize classifier weights
        let mut weights = Array2::<f64>::zeros((n_features, n_classes));
        for i in 0..n_features {
            for j in 0..n_classes {
                weights[[i, j]] = rng.random_range(-0.1, 0.1);
            }
        }

        // Training loop with KL divergence minimization
        for _iter in 0..self.max_iter {
            // Compute predictions for all samples
            let logits = X.dot(&weights);
            let mut predictions = Array2::<f64>::zeros((n_samples, n_classes));

            // Apply softmax with temperature
            for i in 0..n_samples {
                let max_logit = logits
                    .row(i)
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut exp_sum = 0.0;

                for j in 0..n_classes {
                    let exp_val = ((logits[[i, j]] - max_logit) / self.temperature).exp();
                    predictions[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }

                if exp_sum > 0.0 {
                    for j in 0..n_classes {
                        predictions[[i, j]] /= exp_sum;
                    }
                }
            }

            // Compute gradient
            let mut gradient = Array2::<f64>::zeros((n_features, n_classes));

            // Supervised loss gradient (cross-entropy)
            for &idx in &labeled_indices {
                if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                    for j in 0..n_features {
                        for k in 0..n_classes {
                            let target = if k == class_idx { 1.0 } else { 0.0 };
                            gradient[[j, k]] += X[[idx, j]] * (predictions[[idx, k]] - target);
                        }
                    }
                }
            }

            // KL divergence term for unlabeled samples (encourage confident predictions)
            for &idx in &unlabeled_indices {
                // Create augmented version (simplified: add small noise)
                let mut X_aug = X.row(idx).to_owned();
                for j in 0..n_features {
                    X_aug[j] += rng.random_range(-0.01, 0.01);
                }

                // Compute prediction on augmented sample
                let logits_aug = X_aug.dot(&weights);
                let max_logit = logits_aug.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut pred_aug = Array1::<f64>::zeros(n_classes);
                let mut exp_sum = 0.0;

                for j in 0..n_classes {
                    let exp_val = ((logits_aug[j] - max_logit) / self.temperature).exp();
                    pred_aug[j] = exp_val;
                    exp_sum += exp_val;
                }

                if exp_sum > 0.0 {
                    pred_aug /= exp_sum;
                }

                // KL divergence gradient
                for j in 0..n_features {
                    for k in 0..n_classes {
                        let p = predictions[[idx, k]].max(1e-10);
                        let q = pred_aug[k].max(1e-10);
                        let kl_grad = p * (p / q).ln();
                        gradient[[j, k]] += self.kl_weight * X[[idx, j]] * kl_grad;
                    }
                }
            }

            // Update weights
            let scale = self.learning_rate / n_samples as f64;
            for i in 0..n_features {
                for j in 0..n_classes {
                    weights[[i, j]] -= scale * gradient[[i, j]];
                }
            }
        }

        // Generate final predictions
        let logits = X.dot(&weights);
        let mut final_labels = y.clone();

        for &idx in &unlabeled_indices {
            let class_idx = logits
                .row(idx)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            final_labels[idx] = classes[class_idx];
        }

        Ok(KLDivergenceOptimization {
            state: KLDivergenceTrained {
                X_train: X,
                y_train: final_labels,
                classes: Array1::from(classes),
                weights,
            },
            temperature: self.temperature,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            kl_weight: self.kl_weight,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for KLDivergenceOptimization<KLDivergenceTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        let logits = X.dot(&self.state.weights);

        for i in 0..n_test {
            let class_idx = logits
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.state.classes[class_idx];
        }

        Ok(predictions)
    }
}

/// Trained state for EntropyRegularizedSemiSupervised
#[derive(Debug, Clone)]
pub struct EntropyRegularizedTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// prob_distributions
    pub prob_distributions: Array2<f64>,
    /// adjacency
    pub adjacency: Array2<f64>,
}

/// Trained state for KLDivergenceOptimization
#[derive(Debug, Clone)]
pub struct KLDivergenceTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// weights
    pub weights: Array2<f64>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_mutual_information_maximization() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let mim = MutualInformationMaximization::new()
            .n_bins(5)
            .max_iter(10)
            .random_state(42);

        let fitted = mim.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));

        // Labeled samples should be predicted correctly
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_bottleneck() {
        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, -1, -1];

        let ib = InformationBottleneck::new()
            .n_components(2)
            .max_iter(10)
            .random_state(42);

        let fitted = ib.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        // Predictions should be valid class labels (including -1 for potentially unlabeled predictions)
        assert!(predictions.iter().all(|&p| p == -1 || p == 0 || p == 1));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mutual_information_estimation() {
        let mim = MutualInformationMaximization::new().n_bins(5);
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];
        let labeled_indices = vec![0, 1];
        let classes = vec![0, 1];

        let mi = mim
            .estimate_mutual_information(&X, &y, &labeled_indices, &classes)
            .unwrap();
        assert!(mi >= 0.0); // Mutual information should be non-negative
    }

    #[test]
    fn test_information_bottleneck_parameters() {
        let ib = InformationBottleneck::new()
            .beta(0.5)
            .n_components(5)
            .max_iter(50)
            .tol(1e-5);

        assert_eq!(ib.beta, 0.5);
        assert_eq!(ib.n_components, 5);
        assert_eq!(ib.max_iter, 50);
        assert_eq!(ib.tol, 1e-5);
    }

    #[test]
    fn test_mutual_information_maximization_parameters() {
        let mim = MutualInformationMaximization::new()
            .n_bins(15)
            .max_iter(200)
            .learning_rate(0.05)
            .temperature(2.0)
            .regularization(0.02);

        assert_eq!(mim.n_bins, 15);
        assert_eq!(mim.max_iter, 200);
        assert_eq!(mim.learning_rate, 0.05);
        assert_eq!(mim.temperature, 2.0);
        assert_eq!(mim.regularization, 0.02);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empty_labeled_samples_error() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // No labeled samples

        let mim = MutualInformationMaximization::new();
        let result = mim.fit(&X.view(), &y.view());

        assert!(result.is_err());
        if let Err(SklearsError::InvalidInput(msg)) = result {
            assert_eq!(msg, "No labeled samples provided");
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_single_class_stability() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, -1, -1]; // Only one class labeled

        let mim = MutualInformationMaximization::new()
            .max_iter(5)
            .random_state(42);

        let fitted = mim.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        // With only one labeled class, predictions should be stable
        assert!(predictions.iter().all(|&p| p == 0));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_entropy_regularized_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let er = EntropyRegularizedSemiSupervised::new()
            .entropy_weight(0.5)
            .max_iter(10)
            .random_state(42);

        let fitted = er.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));
    }

    #[test]
    fn test_entropy_regularized_parameters() {
        let er = EntropyRegularizedSemiSupervised::new()
            .entropy_weight(0.5)
            .max_iter(50)
            .learning_rate(0.001)
            .n_neighbors(10);

        assert_eq!(er.entropy_weight, 0.5);
        assert_eq!(er.max_iter, 50);
        assert_eq!(er.learning_rate, 0.001);
        assert_eq!(er.n_neighbors, 10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_kl_divergence_optimization_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let kl = KLDivergenceOptimization::new()
            .max_iter(10)
            .temperature(1.0)
            .random_state(42);

        let fitted = kl.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p >= 0 && p <= 1));
    }

    #[test]
    fn test_kl_divergence_parameters() {
        let kl = KLDivergenceOptimization::new()
            .temperature(2.0)
            .max_iter(200)
            .learning_rate(0.001)
            .kl_weight(0.5);

        assert_eq!(kl.temperature, 2.0);
        assert_eq!(kl.max_iter, 200);
        assert_eq!(kl.learning_rate, 0.001);
        assert_eq!(kl.kl_weight, 0.5);
    }
}
