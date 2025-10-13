//! Manifold Regularization for semi-supervised learning
//!
//! Manifold regularization combines supervised learning with unsupervised manifold learning
//! by adding a regularization term that penalizes functions that are not smooth with respect
//! to the intrinsic manifold structure.

use crate::graph::{graph_laplacian, knn_graph};
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashSet;

/// Manifold Regularization classifier
///
/// Manifold regularization extends supervised learning by incorporating the geometry
/// of the marginal distribution P(X). It optimizes both the empirical risk on labeled
/// data and the smoothness of the function with respect to the data manifold.
///
/// The objective function is:
/// min Σ V(yi, f(xi)) + λA ||f||²_K + λI ∫ ||∇f||² dP(x)
///
/// where:
/// - V is the loss function
/// - λA controls Tikhonov regularization in RKHS
/// - λI controls manifold regularization
/// - The integral term enforces smoothness on the manifold
///
/// # Parameters
///
/// * `lambda_a` - Ambient regularization parameter (RKHS regularization)
/// * `lambda_i` - Intrinsic regularization parameter (manifold regularization)
/// * `kernel` - Kernel function ('rbf', 'linear', 'polynomial')
/// * `gamma` - Parameter for RBF kernel
/// * `degree` - Degree for polynomial kernel
/// * `graph_kernel` - Kernel for graph construction ('knn' or 'rbf')
/// * `n_neighbors` - Number of neighbors for graph construction
/// * `max_iter` - Maximum iterations for optimization
/// * `tol` - Convergence tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::ManifoldRegularization;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let mr = ManifoldRegularization::new()
///     .lambda_a(0.01)
///     .lambda_i(0.1)
///     .kernel("rbf".to_string())
///     .gamma(1.0);
/// let fitted = mr.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ManifoldRegularization<S = Untrained> {
    state: S,
    lambda_a: f64,
    lambda_i: f64,
    kernel: String,
    gamma: f64,
    degree: usize,
    graph_kernel: String,
    n_neighbors: usize,
    max_iter: usize,
    tol: f64,
}

impl ManifoldRegularization<Untrained> {
    /// Create a new ManifoldRegularization instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            lambda_a: 0.01,
            lambda_i: 0.1,
            kernel: "rbf".to_string(),
            gamma: 1.0,
            degree: 3,
            graph_kernel: "knn".to_string(),
            n_neighbors: 7,
            max_iter: 1000,
            tol: 1e-6,
        }
    }

    /// Set ambient regularization parameter
    pub fn lambda_a(mut self, lambda_a: f64) -> Self {
        self.lambda_a = lambda_a;
        self
    }

    /// Set intrinsic regularization parameter
    pub fn lambda_i(mut self, lambda_i: f64) -> Self {
        self.lambda_i = lambda_i;
        self
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set degree for polynomial kernel
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set graph kernel for manifold construction
    pub fn graph_kernel(mut self, graph_kernel: String) -> Self {
        self.graph_kernel = graph_kernel;
        self
    }

    /// Set number of neighbors for graph construction
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn compute_kernel_matrix(&self, X1: &Array2<f64>, X2: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::zeros((n1, n2));

        match self.kernel.as_str() {
            "rbf" => {
                for i in 0..n1 {
                    for j in 0..n2 {
                        let diff = &X1.row(i) - &X2.row(j);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        K[[i, j]] = (-self.gamma * dist_sq).exp();
                    }
                }
            }
            "linear" => {
                K = X1.dot(&X2.t());
            }
            "polynomial" => {
                let linear_kernel = X1.dot(&X2.t());
                for i in 0..n1 {
                    for j in 0..n2 {
                        K[[i, j]] =
                            (self.gamma * linear_kernel[[i, j]] + 1.0).powi(self.degree as i32);
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

        Ok(K)
    }

    fn build_manifold_graph(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        match self.graph_kernel.as_str() {
            "knn" => knn_graph(X, self.n_neighbors, "connectivity"),
            "rbf" => {
                let n_samples = X.nrows();
                let mut W = Array2::zeros((n_samples, n_samples));
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let diff = &X.row(i) - &X.row(j);
                            let dist_sq = diff.mapv(|x| x * x).sum();
                            W[[i, j]] = (-self.gamma * dist_sq).exp();
                        }
                    }
                }
                Ok(W)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown graph kernel: {}",
                self.graph_kernel
            ))),
        }
    }

    fn solve_manifold_regularized_problem(
        &self,
        K: &Array2<f64>,
        L: &Array2<f64>,
        labeled_indices: &[usize],
        y_labeled: &Array1<i32>,
        classes: &[i32],
    ) -> SklResult<Array2<f64>> {
        let n_samples = K.nrows();
        let n_classes = classes.len();
        let n_labeled = labeled_indices.len();

        // Create labeled data matrix Y_l
        let mut Y_l = Array2::zeros((n_labeled, n_classes));
        for (i, &idx) in labeled_indices.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == y_labeled[i]) {
                Y_l[[i, class_idx]] = 1.0;
            }
        }

        // Extract kernel matrix for labeled samples
        let mut K_ll = Array2::zeros((n_labeled, n_labeled));
        for (i, &idx_i) in labeled_indices.iter().enumerate() {
            for (j, &idx_j) in labeled_indices.iter().enumerate() {
                K_ll[[i, j]] = K[[idx_i, idx_j]];
            }
        }

        // Solve the system: (K_ll + λ_A * I + λ_I * K_l * L * K_l) * α = Y_l
        // For simplicity, we use the labeled part of the Laplacian
        let mut L_ll = Array2::zeros((n_labeled, n_labeled));
        for (i, &idx_i) in labeled_indices.iter().enumerate() {
            for (j, &idx_j) in labeled_indices.iter().enumerate() {
                L_ll[[i, j]] = L[[idx_i, idx_j]];
            }
        }

        // Construct the regularized matrix
        let mut A = K_ll.clone();

        // Add ambient regularization
        for i in 0..n_labeled {
            A[[i, i]] += self.lambda_a;
        }

        // Add manifold regularization (simplified)
        A = A + self.lambda_i * &L_ll;

        // Solve the linear system A * α = Y_l using iterative method
        let mut alpha = Array2::zeros((n_labeled, n_classes));

        // Simple iterative solver (Jacobi method)
        for _iter in 0..self.max_iter {
            let mut new_alpha = Array2::zeros((n_labeled, n_classes));

            for i in 0..n_labeled {
                for k in 0..n_classes {
                    let mut sum = Y_l[[i, k]];
                    for j in 0..n_labeled {
                        if i != j {
                            sum -= A[[i, j]] * alpha[[j, k]];
                        }
                    }
                    new_alpha[[i, k]] = sum / A[[i, i]];
                }
            }

            // Check convergence
            let diff = (&new_alpha - &alpha).mapv(|x| x.abs()).sum();
            alpha = new_alpha;

            if diff < self.tol {
                break;
            }
        }

        // Compute function values for all samples
        let mut F = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for k in 0..n_classes {
                let mut sum = 0.0;
                for (j, &labeled_idx) in labeled_indices.iter().enumerate() {
                    sum += K[[i, labeled_idx]] * alpha[[j, k]];
                }
                F[[i, k]] = sum;
            }
        }

        Ok(F)
    }
}

impl Default for ManifoldRegularization<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ManifoldRegularization<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for ManifoldRegularization<Untrained> {
    type Fitted = ManifoldRegularization<ManifoldRegularizationTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        // Identify labeled samples
        let mut labeled_indices = Vec::new();
        let mut y_labeled = Vec::new();
        let mut classes = HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                labeled_indices.push(i);
                y_labeled.push(label);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let y_labeled = Array1::from(y_labeled);

        // Build kernel matrix
        let K = self.compute_kernel_matrix(&X, &X)?;

        // Build manifold graph and Laplacian
        let W = self.build_manifold_graph(&X)?;
        let L = graph_laplacian(&W, false)?;

        // Solve manifold regularized problem
        let F = self.solve_manifold_regularized_problem(
            &K,
            &L,
            &labeled_indices,
            &y_labeled,
            &classes,
        )?;

        Ok(ManifoldRegularization {
            state: ManifoldRegularizationTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                alpha: F,
                kernel_matrix: K,
                manifold_laplacian: L,
                labeled_indices,
            },
            lambda_a: self.lambda_a,
            lambda_i: self.lambda_i,
            kernel: self.kernel,
            gamma: self.gamma,
            degree: self.degree,
            graph_kernel: self.graph_kernel,
            n_neighbors: self.n_neighbors,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for ManifoldRegularization<ManifoldRegularizationTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Compute kernel matrix between test and training data
        let K_test = self.compute_kernel_matrix(&X, &self.state.X_train)?;

        for i in 0..n_test {
            // Compute function values for test sample
            let mut max_value = f64::NEG_INFINITY;
            let mut best_class_idx = 0;

            for k in 0..self.state.classes.len() {
                let mut value = 0.0;
                for j in 0..self.state.X_train.nrows() {
                    value += K_test[[i, j]] * self.state.alpha[[j, k]];
                }

                if value > max_value {
                    max_value = value;
                    best_class_idx = k;
                }
            }

            predictions[i] = self.state.classes[best_class_idx];
        }

        Ok(predictions)
    }
}

impl ManifoldRegularization<ManifoldRegularizationTrained> {
    fn compute_kernel_matrix(&self, X1: &Array2<f64>, X2: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n1 = X1.nrows();
        let n2 = X2.nrows();
        let mut K = Array2::zeros((n1, n2));

        match self.kernel.as_str() {
            "rbf" => {
                for i in 0..n1 {
                    for j in 0..n2 {
                        let diff = &X1.row(i) - &X2.row(j);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        K[[i, j]] = (-self.gamma * dist_sq).exp();
                    }
                }
            }
            "linear" => {
                K = X1.dot(&X2.t());
            }
            "polynomial" => {
                let linear_kernel = X1.dot(&X2.t());
                for i in 0..n1 {
                    for j in 0..n2 {
                        K[[i, j]] =
                            (self.gamma * linear_kernel[[i, j]] + 1.0).powi(self.degree as i32);
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

        Ok(K)
    }
}

/// Trained state for ManifoldRegularization
#[derive(Debug, Clone)]
pub struct ManifoldRegularizationTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// alpha
    pub alpha: Array2<f64>,
    /// kernel_matrix
    pub kernel_matrix: Array2<f64>,
    /// manifold_laplacian
    pub manifold_laplacian: Array2<f64>,
    /// labeled_indices
    pub labeled_indices: Vec<usize>,
}
