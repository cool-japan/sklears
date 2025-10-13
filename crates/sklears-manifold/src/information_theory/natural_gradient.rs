//! Natural Gradient Manifold Learning
//!
//! Uses natural gradients based on the Fisher information metric for more
//! efficient optimization on manifolds, particularly for exponential families.

use super::utils::{
    compute_embedding_stress, compute_fisher_information_matrix, compute_natural_gradient,
    compute_stress_gradient,
};
use scirs2_core::ndarray::{Array2, ArrayView2};
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

/// Natural Gradient Manifold Learning
///
/// Uses natural gradients based on the Fisher information metric for more
/// efficient optimization on manifolds, particularly for exponential families.
#[derive(Debug, Clone)]
pub struct NaturalGradientEmbedding<S = Untrained> {
    state: S,
    n_components: usize,
    learning_rate: f64,
    n_iter: usize,
    tol: f64,
    momentum: f64,
    fisher_regularization: f64,
    batch_size: Option<usize>,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct NaturalGradientTrained {
    embedding: Array2<f64>,
    fisher_matrix: Array2<f64>,
    natural_gradient_history: Vec<Array2<f64>>,
    convergence_history: Vec<f64>,
}

impl Default for NaturalGradientEmbedding<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalGradientEmbedding<Untrained> {
    /// Create a new Natural Gradient Embedding instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            learning_rate: 0.01,
            n_iter: 100,
            tol: 1e-6,
            momentum: 0.9,
            fisher_regularization: 1e-6,
            batch_size: None,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
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

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the momentum parameter
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the Fisher information regularization
    pub fn fisher_regularization(mut self, fisher_regularization: f64) -> Self {
        self.fisher_regularization = fisher_regularization;
        self
    }

    /// Set the batch size for stochastic natural gradients
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for NaturalGradientEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for NaturalGradientEmbedding<Untrained> {
    type Fitted = NaturalGradientEmbedding<NaturalGradientTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let x_f64 = x.mapv(|v| v);

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than n_features".to_string(),
            ));
        }

        // Initialize embedding randomly
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut embedding = Array2::from_shape_fn((n_samples, self.n_components), |_| {
            scirs2_core::StandardNormal.sample(&mut rng)
        });

        // Initialize momentum
        let mut momentum_term = Array2::zeros((n_samples, self.n_components));

        let mut natural_gradient_history = Vec::new();
        let mut convergence_history = Vec::new();
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.n_iter {
            // Compute Fisher information matrix for current embedding
            let fisher_matrix = compute_fisher_information_matrix(&embedding)?;

            // Compute standard gradient (for simplicity, using stress-based objective)
            let gradient = compute_stress_gradient(&x_f64, &embedding)?;

            // Compute natural gradient using Fisher information metric
            let natural_gradient =
                compute_natural_gradient(&gradient, &fisher_matrix, self.fisher_regularization)?;

            // Update embedding using natural gradient with momentum
            momentum_term = self.momentum * &momentum_term + self.learning_rate * &natural_gradient;
            embedding -= &momentum_term;

            // Compute loss for convergence checking
            let loss = compute_embedding_stress(&x_f64, &embedding)?;
            convergence_history.push(loss);

            // Check for convergence
            if (prev_loss - loss).abs() < self.tol {
                break;
            }
            prev_loss = loss;

            // Store natural gradient for analysis
            if iter % 10 == 0 || iter == self.n_iter - 1 {
                natural_gradient_history.push(natural_gradient);
            }
        }

        // Compute final Fisher matrix
        let final_fisher_matrix = compute_fisher_information_matrix(&embedding)?;

        let state = NaturalGradientTrained {
            embedding,
            fisher_matrix: final_fisher_matrix,
            natural_gradient_history,
            convergence_history,
        };

        Ok(NaturalGradientEmbedding {
            state,
            n_components: self.n_components,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            tol: self.tol,
            momentum: self.momentum,
            fisher_regularization: self.fisher_regularization,
            batch_size: self.batch_size,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>>
    for NaturalGradientEmbedding<NaturalGradientTrained>
{
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // For fitted data, return the stored embedding
        // For new data, would need out-of-sample extension
        Ok(self.state.embedding.clone())
    }
}
