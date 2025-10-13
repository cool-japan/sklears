//! Harmonic Functions algorithm implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Harmonic Functions classifier
///
/// Harmonic functions is a graph-based semi-supervised learning method that
/// finds a function that is harmonic on the unlabeled points and satisfies
/// label constraints on the labeled points. The harmonic property means that
/// the value at each unlabeled point is the average of its neighbors.
///
/// # Parameters
///
/// * `kernel` - Kernel function ('knn' or 'rbf')
/// * `gamma` - Parameter for RBF kernel
/// * `n_neighbors` - Number of neighbors for KNN kernel
/// * `max_iter` - Maximum number of iterations for iterative solution
/// * `tol` - Convergence tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::HarmonicFunctions;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let hf = HarmonicFunctions::new()
///     .kernel("rbf".to_string())
///     .gamma(20.0);
/// let fitted = hf.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HarmonicFunctions<S = Untrained> {
    state: S,
    kernel: String,
    gamma: f64,
    n_neighbors: usize,
    max_iter: usize,
    tol: f64,
}

impl HarmonicFunctions<Untrained> {
    /// Create a new HarmonicFunctions instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: "rbf".to_string(),
            gamma: 20.0,
            n_neighbors: 7,
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

    #[allow(non_snake_case)]
    fn solve_harmonic_function(
        &self,
        W: &Array2<f64>,
        labeled_indices: &[usize],
        unlabeled_indices: &[usize],
        y: &Array1<i32>,
        classes: &[i32],
    ) -> SklResult<Array2<f64>> {
        let n_samples = W.nrows();
        let n_classes = classes.len();
        let n_labeled = labeled_indices.len();
        let n_unlabeled = unlabeled_indices.len();

        // Initialize label matrix Y
        let mut Y = Array2::zeros((n_samples, n_classes));

        // Set initial values for labeled points
        for &idx in labeled_indices {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                Y[[idx, class_idx]] = 1.0;
            }
        }

        // Compute normalized transition matrix
        let D = W.sum_axis(Axis(1));
        let mut P = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            if D[i] > 0.0 {
                for j in 0..n_samples {
                    P[[i, j]] = W[[i, j]] / D[i];
                }
            }
        }

        // Separate transition matrix into blocks
        // P = [P_ll  P_lu]
        //     [P_ul  P_uu]
        // where l=labeled, u=unlabeled

        let mut P_uu = Array2::zeros((n_unlabeled, n_unlabeled));
        let mut P_ul = Array2::zeros((n_unlabeled, n_labeled));

        for (i, &u_idx) in unlabeled_indices.iter().enumerate() {
            for (j, &u_idx2) in unlabeled_indices.iter().enumerate() {
                P_uu[[i, j]] = P[[u_idx, u_idx2]];
            }
            for (j, &l_idx) in labeled_indices.iter().enumerate() {
                P_ul[[i, j]] = P[[u_idx, l_idx]];
            }
        }

        // Extract labeled values Y_l
        let mut Y_l = Array2::zeros((n_labeled, n_classes));
        for (i, &l_idx) in labeled_indices.iter().enumerate() {
            for k in 0..n_classes {
                Y_l[[i, k]] = Y[[l_idx, k]];
            }
        }

        // Solve harmonic function: Y_u = (I - P_uu)^(-1) * P_ul * Y_l
        // We use iterative method: Y_u^(t+1) = P_uu * Y_u^(t) + P_ul * Y_l

        let mut Y_u = Array2::zeros((n_unlabeled, n_classes));
        let mut prev_Y_u = Y_u.clone();

        for _iter in 0..self.max_iter {
            // Y_u = P_uu * Y_u + P_ul * Y_l
            Y_u = P_uu.dot(&Y_u) + P_ul.dot(&Y_l);

            // Check convergence
            let diff = (&Y_u - &prev_Y_u).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
            prev_Y_u = Y_u.clone();
        }

        // Put unlabeled predictions back into full matrix
        for (i, &u_idx) in unlabeled_indices.iter().enumerate() {
            for k in 0..n_classes {
                Y[[u_idx, k]] = Y_u[[i, k]];
            }
        }

        Ok(Y)
    }
}

impl Default for HarmonicFunctions<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HarmonicFunctions<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for HarmonicFunctions<Untrained> {
    type Fitted = HarmonicFunctions<HarmonicFunctionsTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        let (n_samples, _n_features) = X.dim();

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

        if unlabeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No unlabeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();

        // Build affinity matrix
        let W = self.build_affinity_matrix(&X)?;

        // Solve harmonic function
        let Y =
            self.solve_harmonic_function(&W, &labeled_indices, &unlabeled_indices, &y, &classes)?;

        Ok(HarmonicFunctions {
            state: HarmonicFunctionsTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                label_distributions: Y,
                affinity_matrix: W,
            },
            kernel: self.kernel,
            gamma: self.gamma,
            n_neighbors: self.n_neighbors,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for HarmonicFunctions<HarmonicFunctionsTrained> {
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
    for HarmonicFunctions<HarmonicFunctionsTrained>
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

/// Trained state for HarmonicFunctions
#[derive(Debug, Clone)]
pub struct HarmonicFunctionsTrained {
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
