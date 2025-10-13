//! Information Bottleneck Embedding
//!
//! The Information Bottleneck principle finds a compressed representation that
//! preserves relevant information about a target variable while minimizing
//! irrelevant information.

use super::utils::{
    compute_ib_gradient, compute_mutual_information, compute_mutual_information_2d,
};
use scirs2_core::ndarray::ndarray_linalg::SVD;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Information Bottleneck Embedding
///
/// The Information Bottleneck principle finds a compressed representation that
/// preserves relevant information about a target variable while minimizing
/// irrelevant information.
#[derive(Debug, Clone)]
pub struct InformationBottleneck<S = Untrained> {
    state: S,
    n_components: usize,
    beta: f64, // Trade-off parameter between compression and prediction
    n_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct IBTrained {
    encoder_weights: Array2<f64>,
    mean: Array1<f64>,
    explained_variance_ratio: Array1<f64>,
    mutual_information: f64,
}

impl Default for InformationBottleneck<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl InformationBottleneck<Untrained> {
    /// Create a new Information Bottleneck instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            beta: 1.0,
            n_iter: 100,
            tol: 1e-6,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the trade-off parameter beta
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the tolerance for convergence
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

impl Estimator for InformationBottleneck<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for InformationBottleneck<Untrained> {
    type Fitted = InformationBottleneck<IBTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);
        let y_f64 = y.mapv(|v| v);

        // Center the data
        let mean = x_f64.mean_axis(Axis(0)).unwrap();
        let x_centered = &x_f64 - &mean.clone().insert_axis(Axis(0));

        // Initialize encoder weights randomly
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut encoder_weights = Array2::from_shape_fn((n_features, self.n_components), |_| {
            scirs2_core::StandardNormal.sample(&mut rng)
        });

        // Iterative optimization of information bottleneck objective
        let mut prev_objective = f64::NEG_INFINITY;

        for iter in 0..self.n_iter {
            // Encode data
            let z = x_centered.dot(&encoder_weights);

            // Compute mutual information terms
            let i_z_y = compute_mutual_information(&z, &y_f64)?;
            let i_x_z = compute_mutual_information_2d(&x_centered, &z)?;

            // Information bottleneck objective: I(Z, Y) - β * I(X, Z)
            let objective = i_z_y - self.beta * i_x_z;

            // Check for convergence
            if (objective - prev_objective).abs() < self.tol {
                break;
            }
            prev_objective = objective;

            // Update encoder weights using gradient ascent
            let grad = compute_ib_gradient(&x_centered, &y_f64, &encoder_weights, self.beta)?;
            let learning_rate = 0.01 / (1.0 + 0.1 * iter as f64);
            encoder_weights = encoder_weights + learning_rate * grad;

            // Orthogonalize weights (optional regularization)
            let (u, _, vt) = encoder_weights.svd(true, true).unwrap();
            if let (Some(u_mat), Some(vt_mat)) = (u, vt) {
                use scirs2_core::ndarray::s;
                encoder_weights = u_mat.slice(s![.., ..self.n_components]).dot(&vt_mat);
            }
        }

        // Compute final embedding and statistics
        let final_z = x_centered.dot(&encoder_weights);
        let mutual_information = compute_mutual_information(&final_z, &y_f64)?;

        // Compute explained variance ratio
        let z_var = final_z.var_axis(Axis(0), 0.0);
        let total_var = z_var.sum();
        let explained_variance_ratio = if total_var > 0.0 {
            &z_var / total_var
        } else {
            Array1::zeros(self.n_components)
        };

        let state = IBTrained {
            encoder_weights,
            mean,
            explained_variance_ratio,
            mutual_information,
        };

        Ok(InformationBottleneck {
            state,
            n_components: self.n_components,
            beta: self.beta,
            n_iter: self.n_iter,
            tol: self.tol,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for InformationBottleneck<IBTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let x_f64 = x.mapv(|v| v);
        let x_centered = &x_f64 - &self.state.mean.clone().insert_axis(Axis(0));
        Ok(x_centered.dot(&self.state.encoder_weights))
    }
}
