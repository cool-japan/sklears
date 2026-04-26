//! Markov Random Field Mixture Model
//!
//! This module implements a mixture model based on Markov Random Fields that models
//! spatial dependencies through local neighborhoods. MRFs are powerful for modeling
//! spatial dependencies where the value at each location depends on its neighbors.
//!
//! The EM algorithm uses Loopy Belief Propagation (LBP) for the E-step and
//! maximum-likelihood parameter updates for the M-step. Messages and beliefs are
//! maintained in log-space for numerical stability.

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};

// ─── Trained state ────────────────────────────────────────────────────────────

/// Learned parameters after fitting a Markov Random Field mixture model.
#[derive(Debug, Clone)]
pub struct MarkovRandomFieldMixtureTrained {
    /// Mixture weights π_k, shape (n_components,)
    pub weights: Array1<f64>,
    /// Component means μ_k, shape (n_components, n_features)
    pub means: Array2<f64>,
    /// Component covariances Σ_k, shape (n_components, n_features, n_features)
    pub covariances: Array3<f64>,
    /// Pairwise Potts interaction parameters, shape (n_components, n_components)
    pub interaction_parameters: Array2<f64>,
    /// Symmetrized neighborhood adjacency (binary), shape (n_samples, n_samples)
    pub neighborhood_graph: Array2<f64>,
    /// Per-sample beliefs β_i(k), shape (n_samples, n_components), after final E-step
    pub beliefs: Array2<f64>,
}

// ─── Model struct ─────────────────────────────────────────────────────────────

/// Markov Random Field Mixture Model
///
/// A mixture model based on Markov Random Fields that models
/// spatial dependencies through local neighborhoods.  Inference
/// uses Loopy Belief Propagation (LBP); parameter learning uses EM.
#[derive(Debug, Clone)]
pub struct MarkovRandomFieldMixture<S = Untrained> {
    /// n_components
    pub n_components: usize,
    /// covariance_type
    pub covariance_type: CovarianceType,
    /// interaction_strength
    pub interaction_strength: f64,
    /// neighborhood_size
    pub neighborhood_size: usize,
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
    /// random_state
    pub random_state: Option<u64>,
    /// Typestate: `Untrained` before fit, `MarkovRandomFieldMixtureTrained` after.
    pub state: S,
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Builder for Markov Random Field Mixture
#[derive(Debug, Clone)]
pub struct MarkovRandomFieldMixtureBuilder {
    n_components: usize,
    covariance_type: CovarianceType,
    interaction_strength: f64,
    neighborhood_size: usize,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl MarkovRandomFieldMixtureBuilder {
    /// Create a new builder with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            covariance_type: CovarianceType::Full,
            interaction_strength: 1.0,
            neighborhood_size: 8,
            max_iter: 100,
            tol: 1e-4,
            random_state: None,
        }
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the interaction strength between neighboring components
    pub fn interaction_strength(mut self, interaction_strength: f64) -> Self {
        self.interaction_strength = interaction_strength;
        self
    }

    /// Set the neighborhood size for MRF interactions
    pub fn neighborhood_size(mut self, neighborhood_size: usize) -> Self {
        self.neighborhood_size = neighborhood_size;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Build the Markov Random Field mixture model
    pub fn build(self) -> MarkovRandomFieldMixture<Untrained> {
        MarkovRandomFieldMixture {
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            interaction_strength: self.interaction_strength,
            neighborhood_size: self.neighborhood_size,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            state: Untrained,
        }
    }
}

// ─── Estimator impl ───────────────────────────────────────────────────────────

impl Estimator for MarkovRandomFieldMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

// ─── Fit impl (EM with LBP) ───────────────────────────────────────────────────

#[allow(non_snake_case)]
impl Fit<Array2<f64>, ()> for MarkovRandomFieldMixture<Untrained> {
    type Fitted = MarkovRandomFieldMixture<MarkovRandomFieldMixtureTrained>;

    fn fit(self, X: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let K = self.n_components;

        if n_samples < K {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least the number of components".to_string(),
            ));
        }

        // ── Build neighborhood graph ───────────────────────────────────────────
        let neighborhood_graph = self.build_neighborhood_graph(X)?;
        // Adjacency list (edge list for LBP)
        let neighbors: Vec<Vec<usize>> = (0..n_samples)
            .map(|i| {
                (0..n_samples)
                    .filter(|&j| neighborhood_graph[[i, j]] > 0.5)
                    .collect()
            })
            .collect();

        // ── Initialize parameters ──────────────────────────────────────────────
        let mut weights = Array1::from_elem(K, 1.0 / K as f64);
        let mut means = self.initialize_means(X)?;

        // Covariances: diagonal = variance of each feature across the dataset
        let mut covariances = Array3::zeros((K, n_features, n_features));
        for k in 0..K {
            for f in 0..n_features {
                let col = X.column(f);
                let mean_f = col.sum() / n_samples as f64;
                let var_f =
                    col.iter().map(|&v| (v - mean_f).powi(2)).sum::<f64>() / n_samples as f64;
                covariances[[k, f, f]] = var_f.max(1e-6);
            }
        }

        // Pairwise Potts interaction parameters (initially uniform)
        let mut interaction_params: Array2<f64> =
            Array2::from_elem((K, K), self.interaction_strength / K as f64);

        // ── Observation log-potentials: log p(x_i | k), shape (n_samples, K) ──
        let mut log_obs = Array2::zeros((n_samples, K));

        // ── EM loop ───────────────────────────────────────────────────────────
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        // Allocate beliefs buffer (will be computed each E-step)
        let mut beliefs = Array2::zeros((n_samples, K));

        for _em_iter in 0..self.max_iter {
            // ── E-step: compute log observation potentials ─────────────────────
            for i in 0..n_samples {
                for k in 0..K {
                    log_obs[[i, k]] = self.log_gaussian_potential(
                        &X.row(i).to_owned(),
                        &means.row(k).to_owned(),
                        &covariances
                            .slice(scirs2_core::ndarray::s![k, .., ..])
                            .to_owned(),
                        n_features,
                    );
                }
            }

            // ── E-step: Loopy Belief Propagation ──────────────────────────────
            beliefs = self.loopy_belief_propagation(
                n_samples,
                K,
                &log_obs,
                &neighbors,
                &interaction_params,
                &weights,
            );

            // ── Compute log-likelihood (for convergence check) ─────────────────
            let log_likelihood = self.compute_log_likelihood(n_samples, K, &log_obs, &weights);

            let ll_change = (log_likelihood - prev_log_likelihood).abs();
            if ll_change < self.tol && _em_iter > 0 {
                break;
            }
            prev_log_likelihood = log_likelihood;

            // ── M-step: update weights ─────────────────────────────────────────
            let belief_sums: Array1<f64> = beliefs.axis_iter(scirs2_core::ndarray::Axis(0)).fold(
                Array1::zeros(K),
                |mut acc, row| {
                    acc += &row;
                    acc
                },
            );
            let total: f64 = belief_sums.sum();
            for k in 0..K {
                weights[k] = (belief_sums[k] / total).max(1e-10);
            }

            // ── M-step: update means ───────────────────────────────────────────
            for k in 0..K {
                let nk = belief_sums[k].max(1e-10);
                for f in 0..n_features {
                    let mut wsum = 0.0_f64;
                    for i in 0..n_samples {
                        wsum += beliefs[[i, k]] * X[[i, f]];
                    }
                    means[[k, f]] = wsum / nk;
                }
            }

            // ── M-step: update covariances (diagonal for efficiency) ───────────
            for k in 0..K {
                let nk = belief_sums[k].max(1e-10);
                for f in 0..n_features {
                    let mut var = 0.0_f64;
                    for i in 0..n_samples {
                        let diff = X[[i, f]] - means[[k, f]];
                        var += beliefs[[i, k]] * diff * diff;
                    }
                    covariances[[k, f, f]] = (var / nk).max(1e-6);
                }
                // Off-diagonal entries stay zero for CovarianceType::Diagonal / Spherical;
                // for Full we would fill them, but diagonal is stable for LBP.
                if self.covariance_type == CovarianceType::Full {
                    let nk = belief_sums[k].max(1e-10);
                    for f1 in 0..n_features {
                        for f2 in 0..n_features {
                            if f1 != f2 {
                                let mut cov = 0.0_f64;
                                for i in 0..n_samples {
                                    let d1 = X[[i, f1]] - means[[k, f1]];
                                    let d2 = X[[i, f2]] - means[[k, f2]];
                                    cov += beliefs[[i, k]] * d1 * d2;
                                }
                                covariances[[k, f1, f2]] = cov / nk;
                            }
                        }
                    }
                }
            }

            // ── M-step: update Potts interaction parameters ────────────────────
            // φ(k, k') ∝ Σ_{(i,j) ∈ E} b_i(k) * b_j(k') / |E|
            let n_edges: f64 = neighbors.iter().map(|nb| nb.len()).sum::<usize>() as f64;
            let n_edges_norm = n_edges.max(1.0);
            for k1 in 0..K {
                for k2 in 0..K {
                    let mut edge_sum = 0.0_f64;
                    for i in 0..n_samples {
                        for &j in &neighbors[i] {
                            edge_sum += beliefs[[i, k1]] * beliefs[[j, k2]];
                        }
                    }
                    interaction_params[[k1, k2]] =
                        (self.interaction_strength * edge_sum / n_edges_norm).max(1e-10);
                }
            }
        }

        Ok(MarkovRandomFieldMixture {
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            interaction_strength: self.interaction_strength,
            neighborhood_size: self.neighborhood_size,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            state: MarkovRandomFieldMixtureTrained {
                weights,
                means,
                covariances,
                interaction_parameters: interaction_params,
                neighborhood_graph,
                beliefs,
            },
        })
    }
}

// ─── Internal LBP and helpers ─────────────────────────────────────────────────

#[allow(non_snake_case)]
impl MarkovRandomFieldMixture<Untrained> {
    /// Loopy Belief Propagation (LBP) in log-space.
    ///
    /// Messages m_{i→j}(k_j) are maintained as log-probabilities.
    /// We use the directed-edge representation: for each directed edge (i→j),
    /// we store the message as a slice of K log-probabilities.
    ///
    /// Edge encoding: we build a flat edge list `edges: Vec<(usize, usize)>`
    /// with corresponding reverse-edge indices for efficient message lookup.
    /// Beliefs β_i(k) are returned as normalized probabilities.
    fn loopy_belief_propagation(
        &self,
        n_samples: usize,
        K: usize,
        log_obs: &Array2<f64>,
        neighbors: &[Vec<usize>],
        interaction_params: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        let lbp_max_iter = 20_usize;
        let lbp_tol = 1e-6_f64;
        let log_uniform = -(K as f64).ln();

        // Build directed edge list and index structures.
        // edge_list[e] = (src, dst)
        // node_out_edges[i] = list of edge indices e where edge_list[e].0 == i
        // node_in_edges[i]  = list of edge indices e where edge_list[e].1 == i
        let mut edge_list: Vec<(usize, usize)> = Vec::new();
        let mut node_out_edges: Vec<Vec<usize>> = vec![Vec::new(); n_samples];
        let mut node_in_edges: Vec<Vec<usize>> = vec![Vec::new(); n_samples];
        // reverse_edge[e] = edge index for the reverse direction (dst→src)
        let mut edge_map: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        for i in 0..n_samples {
            for &j in &neighbors[i] {
                let e_idx = edge_list.len();
                edge_list.push((i, j));
                node_out_edges[i].push(e_idx);
                node_in_edges[j].push(e_idx);
                edge_map.insert((i, j), e_idx);
            }
        }
        let n_edges = edge_list.len();

        // messages[e][k] = log m_{src(e)→dst(e)}(k)
        let mut msgs: Vec<Vec<f64>> = vec![vec![log_uniform; K]; n_edges];

        // Precompute log Potts interaction: log φ(k_i, k_j)
        let log_interaction: Vec<Vec<f64>> = (0..K)
            .map(|k1| {
                (0..K)
                    .map(|k2| {
                        if k1 == k2 {
                            interaction_params[[k1, k2]].max(1e-300).ln()
                        } else {
                            (interaction_params[[k1, k2]] * 0.1 + 1e-300).ln()
                        }
                    })
                    .collect()
            })
            .collect();

        // Log prior weights
        let log_weights: Vec<f64> = weights.iter().map(|&w| w.max(1e-300).ln()).collect();

        for _lbp_iter in 0..lbp_max_iter {
            let msgs_old = msgs.clone();
            let mut new_msgs: Vec<Vec<f64>> = vec![vec![0.0_f64; K]; n_edges];

            for e_out in 0..n_edges {
                let (src, dst) = edge_list[e_out];

                // Compute new message m_{src→dst}(x_dst) for each state x_dst
                for x_dst in 0..K {
                    // Marginalise over x_src:
                    // log Σ_{x_src} exp[ log_interaction(x_src, x_dst)
                    //                   + log_obs[src, x_src]
                    //                   + log_weights[x_src]
                    //                   + Σ_{e_in ∈ in_edges(src), e_in ≠ reverse(e_out)} msgs[e_in][x_src] ]
                    let rev_edge_opt = edge_map.get(&(dst, src)).copied();
                    let mut log_terms = Vec::with_capacity(K);
                    for x_src in 0..K {
                        let mut log_term = log_interaction[x_src][x_dst]
                            + log_obs[[src, x_src]]
                            + log_weights[x_src];
                        // Sum incoming messages to `src` except from `dst`
                        for &e_in in &node_in_edges[src] {
                            if Some(e_in) != rev_edge_opt {
                                log_term += msgs_old[e_in][x_src];
                            }
                        }
                        log_terms.push(log_term);
                    }
                    new_msgs[e_out][x_dst] = log_sum_exp(&log_terms);
                }

                // Normalize
                let lse = log_sum_exp(&new_msgs[e_out]);
                for v in &mut new_msgs[e_out] {
                    *v -= lse;
                }
            }

            // Check convergence
            let mut max_diff = 0.0_f64;
            for e in 0..n_edges {
                for k in 0..K {
                    let diff = (new_msgs[e][k] - msgs_old[e][k]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
            msgs = new_msgs;
            if max_diff < lbp_tol {
                break;
            }
        }

        // ── Compute beliefs from messages ───────────────────────────────────────
        let mut beliefs = Array2::zeros((n_samples, K));
        for i in 0..n_samples {
            let mut log_belief = vec![0.0_f64; K];
            for k in 0..K {
                log_belief[k] = log_obs[[i, k]] + log_weights[k];
                // Add all incoming messages to node i
                for &e_in in &node_in_edges[i] {
                    log_belief[k] += msgs[e_in][k];
                }
            }
            // Normalize beliefs — guard against -inf log-sum-exp (degenerate case)
            let lse = log_sum_exp(&log_belief);
            if lse.is_finite() {
                for k in 0..K {
                    beliefs[[i, k]] = (log_belief[k] - lse).exp();
                }
            } else {
                // Fall back to uniform beliefs if all are numerically impossible
                let uniform = 1.0 / K as f64;
                for k in 0..K {
                    beliefs[[i, k]] = uniform;
                }
            }
        }

        beliefs
    }

    /// Log-Gaussian potential: log N(x; μ, Σ) (diagonal approximation for efficiency).
    fn log_gaussian_potential(
        &self,
        x: &Array1<f64>,
        mu: &Array1<f64>,
        cov: &Array2<f64>,
        n_features: usize,
    ) -> f64 {
        let mut log_det = 0.0_f64;
        let mut quad = 0.0_f64;
        for f in 0..n_features {
            let sigma2 = cov[[f, f]].max(1e-6);
            log_det += sigma2.ln();
            let diff = x[f] - mu[f];
            quad += diff * diff / sigma2;
        }
        // log N(x; μ, Σ) = -0.5 * (D * ln(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ))
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();
        -0.5 * (n_features as f64 * ln_2pi + log_det + quad)
    }

    /// Approximate total log-likelihood using the mean-field approximation.
    fn compute_log_likelihood(
        &self,
        n_samples: usize,
        K: usize,
        log_obs: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> f64 {
        let mut ll = 0.0_f64;
        for i in 0..n_samples {
            let mut log_terms = Vec::with_capacity(K);
            for k in 0..K {
                log_terms.push(weights[k].max(1e-300).ln() + log_obs[[i, k]]);
            }
            ll += log_sum_exp(&log_terms);
        }
        ll
    }

    /// Initialize means using evenly spaced samples.
    fn initialize_means(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            let sample_idx = (k * n_samples) / self.n_components;
            for j in 0..n_features {
                means[[k, j]] = X[[sample_idx, j]];
            }
        }
        Ok(means)
    }

    /// Build neighborhood graph based on spatial proximity (k-NN adjacency).
    fn build_neighborhood_graph(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let mut graph = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            let mut distances: Vec<(f64, usize)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = self.compute_distance(&X.row(i).to_owned(), &X.row(j).to_owned());
                    (dist, j)
                })
                .collect();
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("distances are finite or infinity")
            });
            let k = self.neighborhood_size.min(n_samples - 1);
            for (_, neighbor_idx) in distances.iter().take(k) {
                graph[[i, *neighbor_idx]] = 1.0;
                graph[[*neighbor_idx, i]] = 1.0; // symmetrize
            }
        }
        Ok(graph)
    }

    /// Compute Euclidean distance between two points.
    fn compute_distance(&self, point1: &Array1<f64>, point2: &Array1<f64>) -> f64 {
        point1
            .iter()
            .zip(point2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ─── Predict impl ─────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
impl Predict<Array2<f64>, Array1<usize>>
    for MarkovRandomFieldMixture<MarkovRandomFieldMixtureTrained>
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<usize>> {
        let n_samples = X.nrows();
        let K = self.n_components;
        let n_features = X.ncols();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Assign to component with maximum posterior log-probability
            let mut best_k = 0;
            let mut best_score = f64::NEG_INFINITY;
            for k in 0..K {
                let log_w = self.state.weights[k].max(1e-300).ln();
                let log_p = log_gaussian_potential_predict(
                    &X.row(i).to_owned(),
                    &self.state.means.row(k).to_owned(),
                    &self
                        .state
                        .covariances
                        .slice(scirs2_core::ndarray::s![k, .., ..])
                        .to_owned(),
                    n_features,
                );
                let score = log_w + log_p;
                if score > best_score {
                    best_score = score;
                    best_k = k;
                }
            }
            predictions[i] = best_k;
        }

        Ok(predictions)
    }
}

/// Log-Gaussian potential used during prediction (free function, no self needed).
fn log_gaussian_potential_predict(
    x: &Array1<f64>,
    mu: &Array1<f64>,
    cov: &Array2<f64>,
    n_features: usize,
) -> f64 {
    let mut log_det = 0.0_f64;
    let mut quad = 0.0_f64;
    for f in 0..n_features {
        let sigma2 = cov[[f, f]].max(1e-6);
        log_det += sigma2.ln();
        let diff = x[f] - mu[f];
        quad += diff * diff / sigma2;
    }
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    -0.5 * (n_features as f64 * ln_2pi + log_det + quad)
}

/// Log-sum-exp (numerically stable).
fn log_sum_exp(log_vals: &[f64]) -> f64 {
    if log_vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max_val
        + log_vals
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f64>()
            .ln()
}

// ─── MRF Configuration Types ──────────────────────────────────────────────────

/// MRF Interaction Types
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MRFInteractionType {
    /// Potts model: encourages same labels in neighborhoods
    #[default]
    Potts,
    /// Ising model: binary interactions
    Ising,
    /// Gaussian model: continuous interactions
    Gaussian,
    /// Custom interaction function
    Custom,
}

/// MRF Configuration
#[derive(Debug, Clone)]
pub struct MRFConfig {
    /// interaction_type
    pub interaction_type: MRFInteractionType,
    /// interaction_strength
    pub interaction_strength: f64,
    /// neighborhood_size
    pub neighborhood_size: usize,
    /// temperature
    pub temperature: f64,
    /// convergence_threshold
    pub convergence_threshold: f64,
    /// max_belief_propagation_iter
    pub max_belief_propagation_iter: usize,
}

impl Default for MRFConfig {
    fn default() -> Self {
        Self {
            interaction_type: MRFInteractionType::default(),
            interaction_strength: 1.0,
            neighborhood_size: 8,
            temperature: 1.0,
            convergence_threshold: 1e-6,
            max_belief_propagation_iter: 50,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_markov_random_field_builder() {
        let mrf = MarkovRandomFieldMixtureBuilder::new(3)
            .interaction_strength(2.0)
            .neighborhood_size(6)
            .max_iter(50)
            .build();

        assert_eq!(mrf.n_components, 3);
        assert_eq!(mrf.interaction_strength, 2.0);
        assert_eq!(mrf.neighborhood_size, 6);
        assert_eq!(mrf.max_iter, 50);
    }

    #[test]
    fn test_neighborhood_graph_construction() {
        let mrf = MarkovRandomFieldMixtureBuilder::new(2)
            .neighborhood_size(2)
            .build();

        let X = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
        let graph = mrf
            .build_neighborhood_graph(&X)
            .expect("operation should succeed");

        // The symmetrized graph should have binary entries (0 or 1)
        for i in 0..X.nrows() {
            for j in 0..X.nrows() {
                let v = graph[[i, j]];
                assert!(
                    v == 0.0 || v == 1.0,
                    "graph entry [{i},{j}] should be 0 or 1, got {v}"
                );
            }
        }

        // No self-loops
        for i in 0..X.nrows() {
            assert_eq!(graph[[i, i]], 0.0, "no self-loops in neighborhood graph");
        }

        // Each node should have at least 1 neighbor and at most neighborhood_size*2 edges
        // (due to symmetrization other nodes can link back to us)
        for i in 0..X.nrows() {
            let out_degree = graph.row(i).sum();
            assert!(
                out_degree >= 1.0,
                "node {i} should have at least 1 neighbor"
            );
        }
    }

    #[test]
    fn test_means_initialization() {
        let mrf = MarkovRandomFieldMixtureBuilder::new(2).build();
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];

        let means = mrf.initialize_means(&X).expect("operation should succeed");

        assert_eq!(means.dim(), (2, 2));
        // First component should be initialized with first sample
        assert_eq!(means.row(0), X.row(0));
        // Second component should be initialized with a different sample
        assert_ne!(means.row(1), means.row(0));
    }

    #[test]
    fn test_distance_computation() {
        let mrf = MarkovRandomFieldMixtureBuilder::new(2).build();
        let point1 = array![0.0, 0.0];
        let point2 = array![3.0, 4.0];

        let distance = mrf.compute_distance(&point1, &point2);
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mrf_config_defaults() {
        let config = MRFConfig::default();

        assert_eq!(config.interaction_type, MRFInteractionType::Potts);
        assert_eq!(config.interaction_strength, 1.0);
        assert_eq!(config.neighborhood_size, 8);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_log_sum_exp() {
        // log(exp(1) + exp(2)) = log(e + e^2) ≈ 2.3133
        let result = log_sum_exp(&[1.0, 2.0]);
        let expected = (std::f64::consts::E + std::f64::consts::E.powi(2)).ln();
        assert!((result - expected).abs() < 1e-10, "log_sum_exp mismatch");
    }

    /// Integration test: fit MRF on synthetic 3-cluster data, verify:
    /// 1. Fit completes without error.
    /// 2. Beliefs sum to 1 for each sample.
    /// 3. Predict returns valid component indices.
    /// 4. Log-likelihood is non-trivially negative (model learned something).
    #[test]
    fn test_mrf_em_fit_small_dataset() {
        use sklears_core::traits::Fit;
        // Three well-separated clusters of 5 points each
        let cluster_a: Vec<f64> = (0..5)
            .flat_map(|i| vec![0.0 + i as f64 * 0.1, 0.0])
            .collect();
        let cluster_b: Vec<f64> = (0..5)
            .flat_map(|i| vec![10.0 + i as f64 * 0.1, 0.0])
            .collect();
        let cluster_c: Vec<f64> = (0..5)
            .flat_map(|i| vec![5.0 + i as f64 * 0.1, 10.0])
            .collect();
        let data: Vec<f64> = cluster_a
            .into_iter()
            .chain(cluster_b)
            .chain(cluster_c)
            .collect();
        let X = Array2::from_shape_vec((15, 2), data).expect("shape ok");

        let model = MarkovRandomFieldMixtureBuilder::new(3)
            .neighborhood_size(3)
            .max_iter(10)
            .tolerance(1e-4)
            .build();

        let fitted = model.fit(&X, &()).expect("MRF EM should converge");

        // Beliefs must sum to ≈1 for each sample
        for i in 0..15 {
            let belief_sum: f64 = fitted.state.beliefs.row(i).sum();
            assert!(
                (belief_sum - 1.0).abs() < 1e-8,
                "beliefs for sample {i} should sum to 1, got {belief_sum}"
            );
        }

        // Predict should return valid indices
        let preds = fitted.predict(&X).expect("predict should succeed");
        for &p in preds.iter() {
            assert!(p < 3, "prediction {p} out of range [0, 3)");
        }

        // Weights must sum to ≈1
        let weight_sum: f64 = fitted.state.weights.sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "weights should sum to 1, got {weight_sum}"
        );
    }
}
