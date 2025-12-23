//! Similarity learning methods for manifold learning
//! This module implements various similarity learning approaches for manifold embeddings,
//! including metric learning, contrastive learning, and related techniques.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, Norm, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::{seq::SliceRandom, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};
use std::collections::HashMap;

/// Metric learning for manifold embeddings
///
/// This implements a general framework for learning distance metrics
/// that preserve manifold structure during embedding.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `metric_type` - Type of metric to learn ("mahalanobis", "euclidean_learned", "cosine_learned")
/// * `learning_rate` - Learning rate for metric optimization
/// * `n_iter` - Maximum number of iterations
/// * `regularization` - L2 regularization parameter
/// * `triplet_margin` - Margin for triplet constraints
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::similarity::MetricLearning;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let labels = array![0, 0, 1, 1];
///
/// let metric_learner = MetricLearning::new()
///     .n_components(2)
///     .metric_type("mahalanobis".to_string())
///     .learning_rate(0.01);
///
/// let fitted = metric_learner.fit(&x.view(), &labels.view()).unwrap();
/// let transformed = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MetricLearning<S = Untrained> {
    state: S,
    n_components: usize,
    metric_type: String,
    learning_rate: f64,
    n_iter: usize,
    regularization: f64,
    triplet_margin: f64,
    random_state: Option<u64>,
}

/// Trained state for MetricLearning
#[derive(Debug, Clone)]
pub struct TrainedMetricLearning {
    metric_matrix: Array2<f64>,
    n_features: usize,
    n_components: usize,
    transformation_matrix: Array2<f64>,
}

impl Default for MetricLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricLearning<Untrained> {
    /// Create a new MetricLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            metric_type: "mahalanobis".to_string(),
            learning_rate: 0.01,
            n_iter: 1000,
            regularization: 0.001,
            triplet_margin: 1.0,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the metric type
    pub fn metric_type(mut self, metric_type: String) -> Self {
        self.metric_type = metric_type;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the triplet margin
    pub fn triplet_margin(mut self, triplet_margin: f64) -> Self {
        self.triplet_margin = triplet_margin;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for MetricLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for MetricLearning<Untrained> {
    type Fitted = MetricLearning<TrainedMetricLearning>;
    fn fit(self, x: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must be the same".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize metric matrix and transformation matrix
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        // Initialize metric matrix based on type
        let mut metric_matrix = match self.metric_type.as_str() {
            "mahalanobis" => {
                // Initialize as identity matrix with small random perturbations
                let mut m = Array2::eye(n_features);
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            m[[i, j]] += rng.gen_range(-0.01..0.01);
                        }
                    }
                }
                m
            }
            _ => Array2::eye(n_features),
        };

        // Generate triplets for training
        let triplets = self.generate_triplets(x, y, &mut rng)?;

        // Optimize metric using gradient descent on triplet loss
        for iter in 0..self.n_iter {
            let mut gradient = Array2::zeros((n_features, n_features));
            let mut total_loss = 0.0;

            for (anchor_idx, positive_idx, negative_idx) in &triplets {
                let anchor = x.row(*anchor_idx);
                let positive = x.row(*positive_idx);
                let negative = x.row(*negative_idx);

                // Compute distances using current metric
                let dist_pos = self.mahalanobis_distance(&anchor, &positive, &metric_matrix)?;
                let dist_neg = self.mahalanobis_distance(&anchor, &negative, &metric_matrix)?;

                // Triplet loss
                let loss = (dist_pos - dist_neg + self.triplet_margin).max(0.0);
                total_loss += loss;

                // Compute gradients if loss > 0
                if loss > 0.0 {
                    let diff_pos = &anchor - &positive;
                    let diff_neg = &anchor - &negative;

                    // Gradient with respect to metric matrix
                    let grad_pos = diff_pos
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&diff_pos.to_owned().insert_axis(Axis(0)));
                    let grad_neg = diff_neg
                        .to_owned()
                        .insert_axis(Axis(1))
                        .dot(&diff_neg.to_owned().insert_axis(Axis(0)));

                    gradient = gradient + grad_pos - grad_neg;
                }
            }

            // Add regularization gradient
            gradient = gradient + self.regularization * &metric_matrix;

            // Update metric matrix
            metric_matrix = metric_matrix - self.learning_rate * gradient;

            // Ensure metric matrix remains positive semidefinite
            metric_matrix = self.project_to_psd(&metric_matrix)?;

            // Early stopping check
            if iter > 100 && total_loss < 1e-6 {
                break;
            }
        }

        // Compute transformation matrix for dimensionality reduction
        let transformation_matrix = if self.n_components < n_features {
            // Use eigendecomposition to find the top eigenvectors
            let (eigenvalues, eigenvectors) = metric_matrix.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e))
            })?;

            // Sort eigenvectors by eigenvalues (descending)
            let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
            indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

            let mut transform = Array2::zeros((n_features, self.n_components));
            for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
                transform.column_mut(i).assign(&eigenvectors.column(idx));
            }
            transform
        } else {
            Array2::eye(n_features)
        };

        Ok(MetricLearning {
            state: TrainedMetricLearning {
                metric_matrix,
                n_features,
                n_components: self.n_components,
                transformation_matrix,
            },
            n_components: self.n_components,
            metric_type: self.metric_type.clone(),
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            regularization: self.regularization,
            triplet_margin: self.triplet_margin,
            random_state: self.random_state,
        })
    }
}

impl MetricLearning<Untrained> {
    /// Generate triplets for metric learning
    fn generate_triplets(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(usize, usize, usize)>> {
        let n_samples = x.nrows();
        let mut triplets = Vec::new();

        // Group samples by label
        let mut label_to_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            label_to_indices.entry(label).or_default().push(i);
        }

        // Generate triplets: (anchor, positive, negative)
        let target_triplets = (n_samples * 2).min(10000); // Limit number of triplets

        for _ in 0..target_triplets {
            // Select anchor
            let anchor_idx = rng.gen_range(0..n_samples);
            let anchor_label = y[anchor_idx];

            // Select positive (same label as anchor)
            let positive_candidates = &label_to_indices[&anchor_label];
            if positive_candidates.len() < 2 {
                continue; // Skip if not enough positive samples
            }
            let positive_idx = loop {
                let idx = positive_candidates[rng.gen_range(0..positive_candidates.len())];
                if idx != anchor_idx {
                    break idx;
                }
            };

            // Select negative (different label from anchor)
            let negative_candidates: Vec<usize> = label_to_indices
                .iter()
                .filter(|(&label, _)| label != anchor_label)
                .flat_map(|(_, indices)| indices.iter().cloned())
                .collect();

            if negative_candidates.is_empty() {
                continue; // Skip if no negative samples available
            }

            let negative_idx = negative_candidates[rng.gen_range(0..negative_candidates.len())];

            triplets.push((anchor_idx, positive_idx, negative_idx));
        }

        Ok(triplets)
    }

    /// Compute Mahalanobis distance
    fn mahalanobis_distance(
        &self,
        a: &ArrayView1<f64>,
        b: &ArrayView1<f64>,
        metric: &Array2<f64>,
    ) -> SklResult<f64> {
        let diff = a - b;
        let dist_sq = diff.dot(&metric.dot(&diff));
        Ok(dist_sq.sqrt())
    }

    /// Project matrix to positive semidefinite
    fn project_to_psd(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (eigenvalues, eigenvectors) = matrix
            .eigh(UPLO::Upper)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {}", e)))?;

        // Clip negative eigenvalues to small positive value
        let mut clipped_eigenvalues = eigenvalues.clone();
        for val in clipped_eigenvalues.iter_mut() {
            if *val < 1e-8 {
                *val = 1e-8;
            }
        }

        // Reconstruct matrix
        let diag = Array2::from_diag(&clipped_eigenvalues);
        let result = eigenvectors.dot(&diag).dot(&eigenvectors.t());
        Ok(result)
    }
}

impl Estimator for MetricLearning<TrainedMetricLearning> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for MetricLearning<TrainedMetricLearning> {
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        // Apply learned transformation
        let transformed = x.dot(&self.state.transformation_matrix);
        Ok(transformed)
    }
}

/// Contrastive learning for manifold embeddings
///
/// Implements contrastive learning to learn embeddings that bring similar
/// samples closer and push dissimilar samples apart.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `temperature` - Temperature parameter for contrastive loss
/// * `learning_rate` - Learning rate for optimization
/// * `n_iter` - Maximum number of iterations
/// * `batch_size` - Batch size for mini-batch training
/// * `margin` - Margin for contrastive loss
/// * `random_state` - Random seed for reproducibility
#[derive(Debug, Clone)]
pub struct ContrastiveLearning<S = Untrained> {
    state: S,
    n_components: usize,
    temperature: f64,
    learning_rate: f64,
    n_iter: usize,
    batch_size: usize,
    margin: f64,
    random_state: Option<u64>,
}

/// Trained state for ContrastiveLearning
#[derive(Debug, Clone)]
pub struct TrainedContrastiveLearning {
    embedding_matrix: Array2<f64>,
    n_features: usize,
    n_components: usize,
}

impl Default for ContrastiveLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ContrastiveLearning<Untrained> {
    /// Create a new ContrastiveLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            temperature: 0.1,
            learning_rate: 0.01,
            n_iter: 1000,
            batch_size: 32,
            margin: 1.0,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the temperature parameter
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the margin
    pub fn margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for ContrastiveLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for ContrastiveLearning<Untrained> {
    type Fitted = ContrastiveLearning<TrainedContrastiveLearning>;
    fn fit(self, x: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must be the same".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize embedding matrix
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        let mut embedding_matrix = Array2::zeros((n_features, self.n_components));
        for i in 0..n_features {
            for j in 0..self.n_components {
                embedding_matrix[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        // Training loop
        for iter in 0..self.n_iter {
            let mut total_loss = 0.0;
            let mut gradient = Array2::zeros((n_features, self.n_components));

            // Generate pairs for contrastive learning
            let pairs = self.generate_pairs(y, &mut rng)?;

            for (i, j, is_similar) in pairs.iter().take(self.batch_size.min(pairs.len())) {
                let xi = x.row(*i);
                let xj = x.row(*j);

                // Compute embeddings
                let ei = xi.dot(&embedding_matrix);
                let ej = xj.dot(&embedding_matrix);

                // Compute distance in embedding space
                let dist = (&ei - &ej).norm_l2();

                // Compute contrastive loss and gradients
                let (loss, grad_factor) = if *is_similar {
                    // Similar pairs: minimize distance
                    let loss = dist.powi(2);
                    let grad_factor = 2.0 * dist;
                    (loss, grad_factor)
                } else {
                    // Dissimilar pairs: maximize distance up to margin
                    let loss = (self.margin - dist).max(0.0).powi(2);
                    let grad_factor = if dist < self.margin {
                        -2.0 * (self.margin - dist)
                    } else {
                        0.0
                    };
                    (loss, grad_factor)
                };

                total_loss += loss;

                // Update gradients
                if grad_factor.abs() > 1e-8 {
                    let diff_embedding = &ei - &ej;
                    let diff_input = &xi - &xj;

                    for k in 0..n_features {
                        for l in 0..self.n_components {
                            gradient[[k, l]] += grad_factor * diff_embedding[l] * diff_input[k];
                        }
                    }
                }
            }

            // Apply gradient update
            embedding_matrix = embedding_matrix - self.learning_rate * gradient;

            // Early stopping
            if iter > 100 && total_loss < 1e-6 {
                break;
            }
        }

        Ok(ContrastiveLearning {
            state: TrainedContrastiveLearning {
                embedding_matrix,
                n_features,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            temperature: self.temperature,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            batch_size: self.batch_size,
            margin: self.margin,
            random_state: self.random_state,
        })
    }
}

impl ContrastiveLearning<Untrained> {
    /// Generate pairs for contrastive learning
    fn generate_pairs(
        &self,
        y: &ArrayView1<i32>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(usize, usize, bool)>> {
        let n_samples = y.len();
        let mut pairs = Vec::new();

        // Generate positive pairs (same label)
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                if y[i] == y[j] {
                    pairs.push((i, j, true));
                }
            }
        }

        // Generate negative pairs (different labels) - sample to balance
        let n_positive = pairs.len();
        let mut negative_count = 0;
        let max_negative = n_positive * 2; // Limit negative pairs

        for _ in 0..max_negative {
            let i = rng.gen_range(0..n_samples);
            let j = rng.gen_range(0..n_samples);

            if i != j && y[i] != y[j] {
                pairs.push((i, j, false));
                negative_count += 1;

                if negative_count >= n_positive {
                    break;
                }
            }
        }

        // Shuffle pairs
        pairs.shuffle(rng);
        Ok(pairs)
    }
}

impl Estimator for ContrastiveLearning<TrainedContrastiveLearning> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>>
    for ContrastiveLearning<TrainedContrastiveLearning>
{
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        let embedded = x.dot(&self.state.embedding_matrix);
        Ok(embedded)
    }
}

/// Triplet loss for manifold learning
///
/// Implements triplet loss to learn embeddings where the distance between
/// an anchor and positive example is smaller than the distance between
/// the anchor and negative example by at least a margin.
#[derive(Debug, Clone)]
pub struct TripletLoss<S = Untrained> {
    state: S,
    n_components: usize,
    margin: f64,
    learning_rate: f64,
    n_iter: usize,
    batch_size: usize,
    mining_strategy: String, // "random", "hard", "semi_hard"
    random_state: Option<u64>,
}

/// Trained state for TripletLoss
#[derive(Debug, Clone)]
pub struct TrainedTripletLoss {
    embedding_matrix: Array2<f64>,
    n_features: usize,
    n_components: usize,
}

impl Default for TripletLoss<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl TripletLoss<Untrained> {
    /// Create new TripletLoss instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            margin: 1.0,
            learning_rate: 0.01,
            n_iter: 1000,
            batch_size: 32,
            mining_strategy: "random".to_string(),
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the margin
    pub fn margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the mining strategy
    pub fn mining_strategy(mut self, strategy: String) -> Self {
        self.mining_strategy = strategy;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for TripletLoss<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for TripletLoss<Untrained> {
    type Fitted = TripletLoss<TrainedTripletLoss>;
    fn fit(self, x: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must be the same".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize embedding matrix
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        let mut embedding_matrix = Array2::zeros((n_features, self.n_components));
        for i in 0..n_features {
            for j in 0..self.n_components {
                embedding_matrix[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        // Training loop with triplet loss
        for iter in 0..self.n_iter {
            let mut total_loss = 0.0;
            let mut gradient = Array2::zeros((n_features, self.n_components));

            // Generate triplets
            let triplets = self.mine_triplets(x, y, &embedding_matrix, &mut rng)?;

            for (anchor_idx, positive_idx, negative_idx) in
                triplets.iter().take(self.batch_size.min(triplets.len()))
            {
                let anchor = x.row(*anchor_idx);
                let positive = x.row(*positive_idx);
                let negative = x.row(*negative_idx);

                // Compute embeddings
                let ea = anchor.dot(&embedding_matrix);
                let ep = positive.dot(&embedding_matrix);
                let en = negative.dot(&embedding_matrix);

                // Compute distances
                let dist_pos = (&ea - &ep).norm_l2();
                let dist_neg = (&ea - &en).norm_l2();

                // Triplet loss
                let loss = (dist_pos - dist_neg + self.margin).max(0.0);
                total_loss += loss;

                // Compute gradients if loss > 0
                if loss > 0.0 {
                    // Gradients w.r.t embeddings
                    let grad_anchor =
                        2.0 * ((&ep - &ea) / dist_pos.max(1e-8) - (&en - &ea) / dist_neg.max(1e-8));
                    let grad_positive = 2.0 * (&ea - &ep) / dist_pos.max(1e-8);
                    let grad_negative = 2.0 * (&en - &ea) / dist_neg.max(1e-8);

                    // Update embedding matrix gradients
                    for i in 0..n_features {
                        for j in 0..self.n_components {
                            gradient[[i, j]] += grad_anchor[j] * anchor[i];
                            gradient[[i, j]] += grad_positive[j] * positive[i];
                            gradient[[i, j]] += grad_negative[j] * negative[i];
                        }
                    }
                }
            }

            // Apply gradient update
            embedding_matrix = embedding_matrix - self.learning_rate * gradient;

            // Early stopping
            if iter > 100 && total_loss < 1e-6 {
                break;
            }
        }

        Ok(TripletLoss {
            state: TrainedTripletLoss {
                embedding_matrix,
                n_features,
                n_components: self.n_components,
            },
            n_components: self.n_components,
            margin: self.margin,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            batch_size: self.batch_size,
            mining_strategy: self.mining_strategy.clone(),
            random_state: self.random_state,
        })
    }
}

impl TripletLoss<Untrained> {
    /// Mine triplets based on the current embedding
    fn mine_triplets(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
        embedding_matrix: &Array2<f64>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(usize, usize, usize)>> {
        let n_samples = x.nrows();
        let mut triplets = Vec::new();

        // Group samples by label
        let mut label_to_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            label_to_indices.entry(label).or_default().push(i);
        }

        match self.mining_strategy.as_str() {
            "random" => {
                // Random triplet mining
                let target_triplets = (n_samples * 2).min(1000);

                for _ in 0..target_triplets {
                    let anchor_idx = rng.gen_range(0..n_samples);
                    let anchor_label = y[anchor_idx];

                    // Select positive
                    let positive_candidates = &label_to_indices[&anchor_label];
                    if positive_candidates.len() < 2 {
                        continue;
                    }
                    let positive_idx = loop {
                        let idx = positive_candidates[rng.gen_range(0..positive_candidates.len())];
                        if idx != anchor_idx {
                            break idx;
                        }
                    };

                    // Select negative
                    let negative_candidates: Vec<usize> = label_to_indices
                        .iter()
                        .filter(|(&label, _)| label != anchor_label)
                        .flat_map(|(_, indices)| indices.iter().cloned())
                        .collect();

                    if negative_candidates.is_empty() {
                        continue;
                    }

                    let negative_idx =
                        negative_candidates[rng.gen_range(0..negative_candidates.len())];
                    triplets.push((anchor_idx, positive_idx, negative_idx));
                }
            }
            "hard" => {
                // Hard negative mining - select hardest negatives
                for anchor_idx in 0..n_samples {
                    let anchor_label = y[anchor_idx];
                    let anchor_embedding = x.row(anchor_idx).dot(embedding_matrix);

                    // Find positive samples
                    let positive_candidates = &label_to_indices[&anchor_label];
                    if positive_candidates.len() < 2 {
                        continue;
                    }

                    // Find hardest negative (closest negative sample)
                    let mut hardest_neg_dist = f64::INFINITY;
                    let mut hardest_negative = None;

                    for (&neg_label, neg_indices) in label_to_indices.iter() {
                        if neg_label != anchor_label {
                            for &neg_idx in neg_indices {
                                let neg_embedding = x.row(neg_idx).dot(embedding_matrix);
                                let dist = (&anchor_embedding - &neg_embedding).norm_l2();
                                if dist < hardest_neg_dist {
                                    hardest_neg_dist = dist;
                                    hardest_negative = Some(neg_idx);
                                }
                            }
                        }
                    }

                    if let Some(negative_idx) = hardest_negative {
                        // Select random positive
                        let positive_idx = loop {
                            let idx =
                                positive_candidates[rng.gen_range(0..positive_candidates.len())];
                            if idx != anchor_idx {
                                break idx;
                            }
                        };
                        triplets.push((anchor_idx, positive_idx, negative_idx));
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown mining strategy: {}",
                    self.mining_strategy
                )));
            }
        }

        Ok(triplets)
    }
}

impl Estimator for TripletLoss<TrainedTripletLoss> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for TripletLoss<TrainedTripletLoss> {
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        let embedded = x.dot(&self.state.embedding_matrix);
        Ok(embedded)
    }
}

/// Siamese networks for learning embeddings
///
/// Implements Siamese neural networks that learn to map inputs to an embedding space
/// where similar inputs are close and dissimilar inputs are far apart.
///
/// # Parameters
///
/// * `n_components` - Target dimensionality for embeddings
/// * `hidden_layers` - Sizes of hidden layers in the network
/// * `learning_rate` - Learning rate for optimization
/// * `n_iter` - Maximum number of iterations
/// * `batch_size` - Batch size for mini-batch training
/// * `margin` - Margin for contrastive loss
/// * `distance_metric` - Distance metric to use ("euclidean", "cosine")
/// * `random_state` - Random seed for reproducibility
#[derive(Debug, Clone)]
pub struct SiameseNetworks<S = Untrained> {
    state: S,
    n_components: usize,
    hidden_layers: Vec<usize>,
    learning_rate: f64,
    n_iter: usize,
    batch_size: usize,
    margin: f64,
    distance_metric: String,
    random_state: Option<u64>,
}

/// Trained state for SiameseNetworks
#[derive(Debug, Clone)]
pub struct TrainedSiameseNetworks {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    n_features: usize,
    n_components: usize,
    hidden_layers: Vec<usize>,
}

impl Default for SiameseNetworks<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SiameseNetworks<Untrained> {
    /// Create a new SiameseNetworks instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            hidden_layers: vec![64, 32],
            learning_rate: 0.001,
            n_iter: 1000,
            batch_size: 32,
            margin: 1.0,
            distance_metric: "euclidean".to_string(),
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the hidden layer sizes
    pub fn hidden_layers(mut self, hidden_layers: Vec<usize>) -> Self {
        self.hidden_layers = hidden_layers;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the margin
    pub fn margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, distance_metric: String) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for SiameseNetworks<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>> for SiameseNetworks<Untrained> {
    type Fitted = SiameseNetworks<TrainedSiameseNetworks>;
    fn fit(self, x: &ArrayView2<'_, f64>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must be the same".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize network architecture
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        // Build layer sizes: input -> hidden layers -> output
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend_from_slice(&self.hidden_layers);
        layer_sizes.push(self.n_components);

        // Initialize weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier initialization for weights
            let limit = (6.0 / (input_size + output_size) as f64).sqrt();
            let mut weight = Array2::zeros((input_size, output_size));
            for elem in weight.iter_mut() {
                *elem = rng.gen_range(-limit..limit);
            }
            weights.push(weight);

            // Initialize biases to small random values
            let mut bias = Array1::zeros(output_size);
            for elem in bias.iter_mut() {
                *elem = rng.gen_range(-0.01..0.01);
            }
            biases.push(bias);
        }

        // Training loop
        for iter in 0..self.n_iter {
            let mut total_loss = 0.0;

            // Generate pairs for training
            let pairs = self.generate_siamese_pairs(y, &mut rng)?;

            // Mini-batch training
            let n_batches = (pairs.len() + self.batch_size - 1) / self.batch_size;

            for batch_idx in 0..n_batches {
                let batch_start = batch_idx * self.batch_size;
                let batch_end = (batch_start + self.batch_size).min(pairs.len());
                let batch_pairs = &pairs[batch_start..batch_end];

                let mut weight_gradients: Vec<Array2<f64>> =
                    weights.iter().map(|w| Array2::zeros(w.raw_dim())).collect();
                let mut bias_gradients: Vec<Array1<f64>> =
                    biases.iter().map(|b| Array1::zeros(b.raw_dim())).collect();

                let mut batch_loss = 0.0;

                for &(i, j, is_similar) in batch_pairs {
                    let x1 = x.row(i);
                    let x2 = x.row(j);

                    // Forward pass for both inputs
                    let (embed1, activations1) = self.forward_pass(&x1, &weights, &biases)?;
                    let (embed2, activations2) = self.forward_pass(&x2, &weights, &biases)?;

                    // Compute distance and loss
                    let distance = self.compute_distance(&embed1, &embed2)?;
                    let (loss, distance_grad) =
                        self.compute_contrastive_loss(distance, is_similar)?;
                    batch_loss += loss;

                    // Backward pass - compute proper gradients based on distance metric
                    let (dist_grad1, dist_grad2) =
                        self.compute_distance_gradient(&embed1, &embed2, distance)?;
                    let embed_grad1 = distance_grad * dist_grad1;
                    let embed_grad2 = distance_grad * dist_grad2;

                    // Backpropagate gradients
                    self.backpropagate(
                        &weights,
                        &activations1,
                        &embed_grad1,
                        &mut weight_gradients,
                        &mut bias_gradients,
                    )?;
                    self.backpropagate(
                        &weights,
                        &activations2,
                        &embed_grad2,
                        &mut weight_gradients,
                        &mut bias_gradients,
                    )?;
                }

                // Apply gradients
                for (weight, grad) in weights.iter_mut().zip(weight_gradients.iter()) {
                    *weight = weight.clone() - self.learning_rate * grad / batch_pairs.len() as f64;
                }
                for (bias, grad) in biases.iter_mut().zip(bias_gradients.iter()) {
                    *bias = bias.clone() - self.learning_rate * grad / batch_pairs.len() as f64;
                }

                total_loss += batch_loss;
            }

            // Early stopping
            if iter > 100 && total_loss < 1e-6 {
                break;
            }
        }

        Ok(SiameseNetworks {
            state: TrainedSiameseNetworks {
                weights,
                biases,
                n_features,
                n_components: self.n_components,
                hidden_layers: self.hidden_layers.clone(),
            },
            n_components: self.n_components,
            hidden_layers: self.hidden_layers.clone(),
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            batch_size: self.batch_size,
            margin: self.margin,
            distance_metric: self.distance_metric.clone(),
            random_state: self.random_state,
        })
    }
}

impl SiameseNetworks<Untrained> {
    /// Generate pairs for Siamese network training
    fn generate_siamese_pairs(
        &self,
        y: &ArrayView1<i32>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(usize, usize, bool)>> {
        let n_samples = y.len();
        let mut pairs = Vec::new();

        // Generate positive pairs (same label)
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                if y[i] == y[j] {
                    pairs.push((i, j, true));
                }
            }
        }

        // Generate negative pairs (different labels) - balance with positives
        let n_positive = pairs.len();
        let mut negative_count = 0;
        let max_attempts = n_positive * 10;

        for _ in 0..max_attempts {
            let i = rng.gen_range(0..n_samples);
            let j = rng.gen_range(0..n_samples);

            if i != j && y[i] != y[j] {
                pairs.push((i, j, false));
                negative_count += 1;

                if negative_count >= n_positive {
                    break;
                }
            }
        }

        // Shuffle pairs
        pairs.shuffle(rng);
        Ok(pairs)
    }

    /// Forward pass through the network
    fn forward_pass(
        &self,
        input: &ArrayView1<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> SklResult<(Array1<f64>, Vec<Array1<f64>>)> {
        let mut activations = Vec::new();
        let mut current = input.to_owned();
        activations.push(current.clone());

        for (weight, bias) in weights.iter().zip(biases.iter()) {
            // Linear transformation: z = W^T * x + b
            let z = weight.t().dot(&current) + bias;

            // Apply ReLU activation (except for the last layer)
            if weight == weights.last().unwrap() {
                // No activation for output layer
                current = z;
            } else {
                // ReLU activation for hidden layers
                current = z.mapv(|x| x.max(0.0));
            }
            activations.push(current.clone());
        }

        Ok((current, activations))
    }

    /// Compute distance between embeddings
    fn compute_distance(&self, embed1: &Array1<f64>, embed2: &Array1<f64>) -> SklResult<f64> {
        match self.distance_metric.as_str() {
            "euclidean" => Ok((embed1 - embed2).norm_l2()),
            "cosine" => {
                let norm1 = embed1.norm_l2();
                let norm2 = embed2.norm_l2();
                if norm1 < 1e-8 || norm2 < 1e-8 {
                    Ok(1.0) // Maximum cosine distance
                } else {
                    let cosine_sim = embed1.dot(embed2) / (norm1 * norm2);
                    // Clamp cosine similarity to avoid numerical issues
                    let cosine_sim = cosine_sim.max(-1.0).min(1.0);
                    Ok(1.0 - cosine_sim)
                }
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown distance metric: {}",
                self.distance_metric
            ))),
        }
    }

    /// Compute gradient of distance w.r.t. embeddings
    fn compute_distance_gradient(
        &self,
        embed1: &Array1<f64>,
        embed2: &Array1<f64>,
        distance: f64,
    ) -> SklResult<(Array1<f64>, Array1<f64>)> {
        match self.distance_metric.as_str() {
            "euclidean" => {
                let diff = embed1 - embed2;
                let norm = diff.norm_l2().max(1e-8);
                let grad1 = &diff / norm;
                let grad2 = -&grad1;
                Ok((grad1, grad2))
            }
            "cosine" => {
                let norm1 = embed1.norm_l2();
                let norm2 = embed2.norm_l2();

                if norm1 < 1e-8 || norm2 < 1e-8 {
                    // Handle degenerate case - return zero gradients
                    Ok((Array1::zeros(embed1.len()), Array1::zeros(embed2.len())))
                } else {
                    let dot_product = embed1.dot(embed2);
                    let cosine_sim = dot_product / (norm1 * norm2);

                    // Gradient of cosine similarity w.r.t. embed1
                    let grad1 =
                        (embed2 / (norm1 * norm2)) - (embed1 * cosine_sim / (norm1 * norm1));
                    // Gradient of cosine similarity w.r.t. embed2
                    let grad2 =
                        (embed1 / (norm1 * norm2)) - (embed2 * cosine_sim / (norm2 * norm2));

                    // Since cosine distance = 1 - cosine similarity, negate the gradients
                    Ok((-grad1, -grad2))
                }
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown distance metric: {}",
                self.distance_metric
            ))),
        }
    }

    /// Compute contrastive loss and gradient
    fn compute_contrastive_loss(&self, distance: f64, is_similar: bool) -> SklResult<(f64, f64)> {
        if is_similar {
            // Similar pairs: minimize distance
            let loss = distance.powi(2);
            let grad = 2.0 * distance;
            Ok((loss, grad))
        } else {
            // Dissimilar pairs: maximize distance up to margin
            let loss = (self.margin - distance).max(0.0).powi(2);
            let grad = if distance < self.margin {
                -2.0 * (self.margin - distance)
            } else {
                0.0
            };
            Ok((loss, grad))
        }
    }

    /// Backpropagate gradients through the network
    fn backpropagate(
        &self,
        weights: &[Array2<f64>],
        activations: &[Array1<f64>],
        output_grad: &Array1<f64>,
        weight_gradients: &mut [Array2<f64>],
        bias_gradients: &mut [Array1<f64>],
    ) -> SklResult<()> {
        let n_layers = weight_gradients.len();
        let mut current_grad = output_grad.clone();

        for i in (0..n_layers).rev() {
            // Gradient w.r.t. bias
            bias_gradients[i] = &bias_gradients[i] + &current_grad;

            // Gradient w.r.t. weights
            let input_activation = &activations[i];
            for j in 0..weight_gradients[i].nrows() {
                for k in 0..weight_gradients[i].ncols() {
                    weight_gradients[i][[j, k]] += input_activation[j] * current_grad[k];
                }
            }

            // Gradient w.r.t. input (for next layer)
            if i > 0 {
                let next_grad = weights[i].dot(&current_grad);

                // Apply ReLU derivative (hidden layers only)
                let mut relu_grad = next_grad;
                for (j, &activation) in activations[i].iter().enumerate() {
                    if activation <= 0.0 {
                        relu_grad[j] = 0.0;
                    }
                }
                current_grad = relu_grad;
            }
        }

        Ok(())
    }
}

impl Estimator for SiameseNetworks<TrainedSiameseNetworks> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for SiameseNetworks<TrainedSiameseNetworks> {
    fn transform(&self, x: &ArrayView2<'_, f64>) -> SklResult<Array2<f64>> {
        if x.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.state.n_features,
                x.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut embeddings = Array2::zeros((n_samples, self.state.n_components));

        for (i, row) in x.rows().into_iter().enumerate() {
            let (embedding, _) =
                self.forward_pass_trained(&row, &self.state.weights, &self.state.biases)?;
            embeddings.row_mut(i).assign(&embedding);
        }

        Ok(embeddings)
    }
}

impl SiameseNetworks<TrainedSiameseNetworks> {
    /// Forward pass for trained network
    fn forward_pass_trained(
        &self,
        input: &ArrayView1<f64>,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> SklResult<(Array1<f64>, Vec<Array1<f64>>)> {
        let mut activations = Vec::new();
        let mut current = input.to_owned();
        activations.push(current.clone());

        for (i, (weight, bias)) in weights.iter().zip(biases.iter()).enumerate() {
            // Linear transformation
            let z = weight.t().dot(&current) + bias;

            // Apply activation (ReLU for hidden layers, none for output)
            if i == weights.len() - 1 {
                current = z; // No activation for output layer
            } else {
                current = z.mapv(|x| x.max(0.0)); // ReLU for hidden layers
            }
            activations.push(current.clone());
        }

        Ok((current, activations))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, s, ArrayView1, ArrayView2};

    #[test]
    fn test_metric_learning_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        let metric_learner = MetricLearning::new()
            .n_components(2)
            .n_iter(10)
            .random_state(42);

        let fitted = metric_learner.fit(&x.view(), &y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_contrastive_learning_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        let contrastive = ContrastiveLearning::new()
            .n_components(2)
            .n_iter(10)
            .random_state(42);

        let fitted = contrastive.fit(&x.view(), &y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_triplet_loss_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        let triplet = TripletLoss::new()
            .n_components(2)
            .n_iter(10)
            .random_state(42);

        let fitted = triplet.fit(&x.view(), &y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_metric_learning_dimensionality_reduction() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [8.0, 9.0, 10.0, 11.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let y = array![0, 0, 1, 1];

        let metric_learner = MetricLearning::new()
            .n_components(2)
            .n_iter(10)
            .random_state(42);

        let fitted = metric_learner.fit(&x.view(), &y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_siamese_networks_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        let siamese = SiameseNetworks::new()
            .n_components(2)
            .hidden_layers(vec![4])
            .n_iter(5)
            .random_state(42);

        let fitted = siamese.fit(&x.view(), &y.view()).unwrap();
        let transformed = fitted.transform(&x.view()).unwrap();

        assert_eq!(transformed.shape(), [4, 2]);
        assert!(transformed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_siamese_networks_distance_metrics() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        // Test Euclidean distance metric
        let siamese_euclidean = SiameseNetworks::new()
            .n_components(2)
            .hidden_layers(vec![4])
            .distance_metric("euclidean".to_string())
            .n_iter(5)
            .random_state(42);

        let fitted_euclidean = siamese_euclidean.fit(&x.view(), &y.view()).unwrap();
        let transformed_euclidean = fitted_euclidean.transform(&x.view()).unwrap();

        assert_eq!(transformed_euclidean.shape(), [4, 2]);
        assert!(transformed_euclidean.iter().all(|&x| x.is_finite()));

        // Test Cosine distance metric
        let siamese_cosine = SiameseNetworks::new()
            .n_components(2)
            .hidden_layers(vec![4])
            .distance_metric("cosine".to_string())
            .n_iter(5)
            .random_state(42);

        let fitted_cosine = siamese_cosine.fit(&x.view(), &y.view()).unwrap();
        let transformed_cosine = fitted_cosine.transform(&x.view()).unwrap();

        assert_eq!(transformed_cosine.shape(), [4, 2]);
        assert!(transformed_cosine.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_triplet_loss_mining_strategies() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        // Test random mining strategy
        let triplet_random = TripletLoss::new()
            .n_components(2)
            .mining_strategy("random".to_string())
            .n_iter(5)
            .random_state(42);

        let fitted_random = triplet_random.fit(&x.view(), &y.view()).unwrap();
        let transformed_random = fitted_random.transform(&x.view()).unwrap();

        assert_eq!(transformed_random.shape(), [4, 2]);
        assert!(transformed_random.iter().all(|&x| x.is_finite()));

        // Test hard mining strategy
        let triplet_hard = TripletLoss::new()
            .n_components(2)
            .mining_strategy("hard".to_string())
            .n_iter(5)
            .random_state(42);

        let fitted_hard = triplet_hard.fit(&x.view(), &y.view()).unwrap();
        let transformed_hard = fitted_hard.transform(&x.view()).unwrap();

        assert_eq!(transformed_hard.shape(), [4, 2]);
        assert!(transformed_hard.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_consistency_across_similarity_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0]];
        let y = array![0, 0, 1, 1];

        // All methods should produce embeddings of correct shape
        let metric_learner = MetricLearning::new()
            .n_components(2)
            .n_iter(5)
            .random_state(42);

        let contrastive = ContrastiveLearning::new()
            .n_components(2)
            .n_iter(5)
            .random_state(42);

        let triplet = TripletLoss::new()
            .n_components(2)
            .n_iter(5)
            .random_state(42);

        let siamese = SiameseNetworks::new()
            .n_components(2)
            .hidden_layers(vec![4])
            .n_iter(5)
            .random_state(42);

        let fitted_metric = metric_learner.fit(&x.view(), &y.view()).unwrap();
        let fitted_contrastive = contrastive.fit(&x.view(), &y.view()).unwrap();
        let fitted_triplet = triplet.fit(&x.view(), &y.view()).unwrap();
        let fitted_siamese = siamese.fit(&x.view(), &y.view()).unwrap();

        let transformed_metric = fitted_metric.transform(&x.view()).unwrap();
        let transformed_contrastive = fitted_contrastive.transform(&x.view()).unwrap();
        let transformed_triplet = fitted_triplet.transform(&x.view()).unwrap();
        let transformed_siamese = fitted_siamese.transform(&x.view()).unwrap();

        // All should have correct shape
        assert_eq!(transformed_metric.shape(), [4, 2]);
        assert_eq!(transformed_contrastive.shape(), [4, 2]);
        assert_eq!(transformed_triplet.shape(), [4, 2]);
        assert_eq!(transformed_siamese.shape(), [4, 2]);

        // All should have finite values
        assert!(transformed_metric.iter().all(|&x| x.is_finite()));
        assert!(transformed_contrastive.iter().all(|&x| x.is_finite()));
        assert!(transformed_triplet.iter().all(|&x| x.is_finite()));
        assert!(transformed_siamese.iter().all(|&x| x.is_finite()));
    }
}
