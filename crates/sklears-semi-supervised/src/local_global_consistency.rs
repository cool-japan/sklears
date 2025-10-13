//! Local and Global Consistency algorithm implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Local and Global Consistency classifier
///
/// Local and Global Consistency is a graph-based semi-supervised learning method that
/// finds a function which is smooth with respect to the intrinsic structure (local consistency)
/// and close to the given labels (global consistency). It minimizes:
/// min f^T * L * f + α * ||f - y||²
///
/// # Parameters
///
/// * `kernel` - Kernel function ('knn' or 'rbf')
/// * `gamma` - Parameter for RBF kernel
/// * `n_neighbors` - Number of neighbors for KNN kernel
/// * `alpha` - Regularization parameter balancing smoothness and label fitting
/// * `max_iter` - Maximum number of iterations for iterative solution
/// * `tol` - Convergence tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::LocalGlobalConsistency;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let lgc = LocalGlobalConsistency::new()
///     .kernel("rbf".to_string())
///     .gamma(20.0)
///     .alpha(0.99);
/// let fitted = lgc.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LocalGlobalConsistency<S = Untrained> {
    state: S,
    kernel: String,
    gamma: f64,
    n_neighbors: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
}

impl LocalGlobalConsistency<Untrained> {
    /// Create a new LocalGlobalConsistency instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: "rbf".to_string(),
            gamma: 20.0,
            n_neighbors: 7,
            alpha: 0.99,
            max_iter: 1000,
            tol: 1e-6,
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the number of neighbors for KNN kernel
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
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

    fn build_affinity_matrix(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let mut W = Array2::zeros((n_samples, n_samples));

        match self.kernel.as_str() {
            "rbf" => {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let diff = &X.row(i) - &X.row(j);
                            let dist_sq = diff.mapv(|x| x * x).sum();
                            W[[i, j]] = (-self.gamma * dist_sq).exp();
                        }
                    }
                }
            }
            "knn" => {
                for i in 0..n_samples {
                    let mut distances: Vec<(usize, f64)> = Vec::new();
                    for j in 0..n_samples {
                        if i != j {
                            let diff = &X.row(i) - &X.row(j);
                            let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
                            distances.push((j, dist));
                        }
                    }

                    distances
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    for &(j, _) in distances.iter().take(self.n_neighbors) {
                        W[[i, j]] = 1.0;
                        W[[j, i]] = 1.0; // Make symmetric
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown kernel: {}",
                    self.kernel
                )));
            }
        }

        Ok(W)
    }

    #[allow(non_snake_case)]
    fn solve_lgc_function(
        &self,
        W: &Array2<f64>,
        labeled_indices: &[usize],
        y: &Array1<i32>,
        classes: &[i32],
    ) -> SklResult<Array2<f64>> {
        let n_samples = W.nrows();
        let n_classes = classes.len();

        // Compute degree matrix and normalized Laplacian
        let D = W.sum_axis(Axis(1));
        let mut D_sqrt_inv = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            if D[i] > 0.0 {
                D_sqrt_inv[[i, i]] = 1.0 / D[i].sqrt();
            }
        }

        // Symmetric normalized Laplacian: L_sym = I - D^(-1/2) * W * D^(-1/2)
        let S = D_sqrt_inv.dot(W).dot(&D_sqrt_inv);

        // Initialize label matrix Y
        let mut Y = Array2::zeros((n_samples, n_classes));
        for &idx in labeled_indices {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                Y[[idx, class_idx]] = 1.0;
            }
        }

        let Y_static = Y.clone();

        // Iteratively solve: F = α * S * F + (1 - α) * Y
        let mut prev_F = Y.clone();
        for _iter in 0..self.max_iter {
            Y = self.alpha * S.dot(&Y) + (1.0 - self.alpha) * &Y_static;

            // Check convergence
            let diff = (&Y - &prev_F).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
            prev_F = Y.clone();
        }

        Ok(Y)
    }
}

impl Default for LocalGlobalConsistency<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LocalGlobalConsistency<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for LocalGlobalConsistency<Untrained> {
    type Fitted = LocalGlobalConsistency<LocalGlobalConsistencyTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        let (n_samples, _n_features) = X.dim();

        // Identify labeled and unlabeled samples
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

        // Build affinity matrix
        let W = self.build_affinity_matrix(&X)?;

        // Solve local global consistency function
        let F = self.solve_lgc_function(&W, &labeled_indices, &y, &classes)?;

        Ok(LocalGlobalConsistency {
            state: LocalGlobalConsistencyTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                label_distributions: F,
                affinity_matrix: W,
            },
            kernel: self.kernel,
            gamma: self.gamma,
            n_neighbors: self.n_neighbors,
            alpha: self.alpha,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for LocalGlobalConsistency<LocalGlobalConsistencyTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            // Find most similar training sample
            let mut min_dist = f64::INFINITY;
            let mut best_idx = 0;

            for j in 0..self.state.X_train.nrows() {
                let diff = &X.row(i) - &self.state.X_train.row(j);
                let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = j;
                }
            }

            // Use the label distribution of the most similar sample
            let distributions = self.state.label_distributions.row(best_idx);
            let max_idx = distributions
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;

            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for LocalGlobalConsistency<LocalGlobalConsistencyTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut probas = Array2::zeros((n_test, n_classes));

        for i in 0..n_test {
            // Find most similar training sample
            let mut min_dist = f64::INFINITY;
            let mut best_idx = 0;

            for j in 0..self.state.X_train.nrows() {
                let diff = &X.row(i) - &self.state.X_train.row(j);
                let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = j;
                }
            }

            // Copy the label distribution
            for k in 0..n_classes {
                probas[[i, k]] = self.state.label_distributions[[best_idx, k]];
            }
        }

        Ok(probas)
    }
}

/// Trained state for LocalGlobalConsistency
#[derive(Debug, Clone)]
pub struct LocalGlobalConsistencyTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// label_distributions
    pub label_distributions: Array2<f64>,
    /// affinity_matrix
    pub affinity_matrix: Array2<f64>,
}
