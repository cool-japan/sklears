//! Label Propagation algorithm implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};
use std::collections::BTreeSet;

/// Label Propagation classifier
///
/// Label propagation is a graph-based semi-supervised learning algorithm that
/// propagates labels through a neighborhood graph. It uses the graph Laplacian
/// to propagate labels from labeled to unlabeled points.
///
/// # Parameters
///
/// * `kernel` - Kernel function ('knn' or 'rbf')
/// * `gamma` - Parameter for RBF kernel
/// * `n_neighbors` - Number of neighbors for KNN kernel
/// * `alpha` - Clamping factor (0.0 = hard clamping, 1.0 = soft clamping)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::LabelPropagation;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let lp = LabelPropagation::new()
///     .kernel("rbf".to_string())
///     .gamma(20.0);
/// let fitted = lp.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LabelPropagation<S = Untrained> {
    state: S,
    kernel: String,
    gamma: f64,
    n_neighbors: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
}

impl LabelPropagation<Untrained> {
    /// Create a new LabelPropagation instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: "rbf".to_string(),
            gamma: 20.0,
            n_neighbors: 7,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-3,
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

    /// Set the clamping factor
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
}

impl Default for LabelPropagation<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LabelPropagation<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for LabelPropagation<Untrained> {
    type Fitted = LabelPropagation<LabelPropagationTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        let (n_samples, _n_features) = X.dim();

        // Identify labeled and unlabeled samples
        let mut labeled_indices = Vec::new();
        let mut unlabeled_indices = Vec::new();
        let mut classes = BTreeSet::new();

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

        // Build affinity matrix
        let W = self.build_affinity_matrix(&X)?;

        // Normalize the matrix (row-wise normalization)
        let D = W.sum_axis(Axis(1));
        let mut P = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            if D[i] > 0.0 {
                for j in 0..n_samples {
                    P[[i, j]] = W[[i, j]] / D[i];
                }
            }
        }

        // Initialize label distributions
        let mut Y = Array2::zeros((n_samples, n_classes));
        for &idx in &labeled_indices {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                Y[[idx, class_idx]] = 1.0;
            }
        }

        // Label propagation iterations
        let mut prev_Y = Y.clone();
        for _iter in 0..self.max_iter {
            // Propagate: Y = P * Y
            Y = P.dot(&Y);

            // Clamp labeled nodes
            for &idx in &labeled_indices {
                if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                    for k in 0..n_classes {
                        Y[[idx, k]] = if k == class_idx { 1.0 } else { 0.0 };
                    }
                }
            }

            // Check convergence
            let diff = (&Y - &prev_Y).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
            prev_Y = Y.clone();
        }

        Ok(LabelPropagation {
            state: LabelPropagationTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                label_distributions: Y,
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

impl LabelPropagation<Untrained> {
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

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for LabelPropagation<LabelPropagationTrained> {
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for LabelPropagation<LabelPropagationTrained>
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

/// Trained state for LabelPropagation
#[derive(Debug, Clone)]
pub struct LabelPropagationTrained {
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
