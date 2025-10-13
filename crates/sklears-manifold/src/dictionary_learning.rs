//! Dictionary Learning implementation
//! This module provides Dictionary Learning for manifold learning through sparse representation.

use scirs2_core::ndarray::ndarray_linalg::SVD;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Dictionary Learning for manifold learning
///
/// Dictionary learning learns an over-complete dictionary of basis vectors
/// and sparse codes simultaneously. This is useful for manifold learning
/// when the data has sparse representation in some basis.
///
/// # Parameters
///
/// * `n_components` - Number of dictionary atoms
/// * `alpha` - Sparsity regularization parameter
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Tolerance for convergence
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::DictionaryLearning;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let dl = DictionaryLearning::new()
///     .n_components(2)
///     .alpha(0.1);
///
/// let fitted = dl.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DictionaryLearning<S = Untrained> {
    state: S,
    n_components: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl DictionaryLearning<Untrained> {
    /// Create a new DictionaryLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 100,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            random_state: None,
        }
    }

    /// Set the number of dictionary atoms
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the sparsity regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for DictionaryLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trained state for Dictionary Learning
#[derive(Debug, Clone)]
pub struct DLTrained {
    dictionary: Array2<f64>,
    mean: Array1<f64>,
}

impl Estimator for DictionaryLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for DictionaryLearning<Untrained> {
    type Fitted = DictionaryLearning<DLTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Dictionary Learning requires at least 2 samples".to_string(),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Center the data
        let mean = x_f64.mean_axis(Axis(0)).unwrap();
        let x_centered = &x_f64 - &mean.view().broadcast(x_f64.dim()).unwrap();

        // Initialize dictionary randomly
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut dictionary = Array2::zeros((n_features, self.n_components));
        for i in 0..n_features {
            for j in 0..self.n_components {
                dictionary[(i, j)] = rng.sample::<f64, _>(scirs2_core::StandardNormal);
            }
        }

        // Normalize dictionary atoms
        for mut col in dictionary.columns_mut() {
            let norm = col.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                col /= norm;
            }
        }

        // Dictionary learning loop
        for iter in 0..self.max_iter {
            let prev_dictionary = dictionary.clone();

            // Sparse coding step: find sparse codes for each sample
            let mut codes = Array2::zeros((n_samples, self.n_components));

            for i in 0..n_samples {
                let sample = x_centered.row(i);
                let code = self.sparse_coding(&sample, &dictionary)?;
                codes.row_mut(i).assign(&code);
            }

            // Dictionary update step
            for k in 0..self.n_components {
                // Find samples that use atom k
                let using_k: Vec<usize> = codes
                    .column(k)
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &coef)| if coef.abs() > 1e-10 { Some(i) } else { None })
                    .collect();

                if using_k.is_empty() {
                    continue;
                }

                // Compute residual matrix
                let mut residual = Array2::zeros((n_features, using_k.len()));
                for (j, &sample_idx) in using_k.iter().enumerate() {
                    let mut r = x_centered.row(sample_idx).to_owned();
                    for l in 0..self.n_components {
                        if l != k {
                            let coef = codes[(sample_idx, l)];
                            let atom = dictionary.column(l);
                            r.scaled_add(-coef, &atom);
                        }
                    }
                    residual.column_mut(j).assign(&r);
                }

                // Update dictionary atom using SVD
                if let Ok((u, _s, _vt)) = residual.svd(true, true) {
                    if let Some(u) = u {
                        dictionary.column_mut(k).assign(&u.column(0));
                    }
                }
            }

            // Check convergence
            let diff = &dictionary - &prev_dictionary;
            let change = diff.mapv(|x| x * x).sum().sqrt();
            if change < self.tol {
                eprintln!(
                    "Dictionary learning converged after {} iterations",
                    iter + 1
                );
                break;
            }
        }

        Ok(DictionaryLearning {
            state: DLTrained { dictionary, mean },
            n_components: self.n_components,
            alpha: self.alpha,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for DictionaryLearning<DLTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let x_f64 = x.mapv(|v| v);
        let x_centered = &x_f64 - &self.state.mean.view().broadcast(x_f64.dim()).unwrap();

        // Compute sparse codes using coordinate descent
        let mut codes = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = x_centered.row(i);
            let mut code = Array1::<f64>::zeros(self.n_components);

            // Coordinate descent for sparse coding
            for _ in 0..100 {
                let mut max_change = 0.0f64;

                for k in 0..self.n_components {
                    // Compute residual without component k
                    let mut residual = sample.to_owned();
                    for j in 0..self.n_components {
                        if j != k {
                            let atom_j = self.state.dictionary.column(j);
                            residual.scaled_add(-code[j], &atom_j);
                        }
                    }

                    // Update component k
                    let atom_k = self.state.dictionary.column(k);
                    let new_code_k =
                        SparseCoding::soft_threshold(residual.dot(&atom_k), self.alpha);
                    let change = (new_code_k - code[k]).abs();
                    max_change = max_change.max(change);
                    code[k] = new_code_k;
                }

                if max_change < 1e-6 {
                    break;
                }
            }

            codes.row_mut(i).assign(&code);
        }

        Ok(codes)
    }
}

impl DictionaryLearning<Untrained> {
    fn sparse_coding(
        &self,
        sample: &ArrayView1<f64>,
        dictionary: &Array2<f64>,
    ) -> SklResult<Array1<f64>> {
        let mut code = Array1::<f64>::zeros(self.n_components);

        // Coordinate descent algorithm for sparse coding
        for _ in 0..100 {
            let mut max_change = 0.0f64;

            for k in 0..self.n_components {
                // Compute residual excluding component k
                let mut residual = sample.to_owned();
                for j in 0..self.n_components {
                    if j != k {
                        let atom_j = dictionary.column(j);
                        residual.scaled_add(-code[j], &atom_j);
                    }
                }

                // Update coefficient k using soft thresholding
                let atom_k = dictionary.column(k);
                let dot_product = residual.dot(&atom_k);
                let new_code_k = SparseCoding::soft_threshold(dot_product, self.alpha);

                let change = (new_code_k - code[k]).abs();
                max_change = max_change.max(change);
                code[k] = new_code_k;
            }

            if max_change < 1e-6 {
                break;
            }
        }

        Ok(code)
    }
}

/// Sparse coding utilities
struct SparseCoding;

impl SparseCoding {
    /// Soft thresholding operator
    fn soft_threshold(x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }
}

impl DictionaryLearning<DLTrained> {
    /// Get the learned dictionary
    pub fn dictionary(&self) -> &Array2<f64> {
        &self.state.dictionary
    }

    /// Get the mean of the training data
    pub fn mean(&self) -> &Array1<f64> {
        &self.state.mean
    }
}
