//! Kernel functions for Support Vector Machines
//!
//! This module provides comprehensive kernel implementations for SVM including basic kernels,
//! composite kernels, graph kernels, and advanced kernel methods.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use std::sync::Arc;

/// Kernel trait for all kernel functions
pub trait Kernel: Send + Sync + std::fmt::Debug {
    /// Compute kernel value between two vectors
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64;

    /// Compute kernel matrix for two datasets
    fn compute_matrix(&self, x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        let (n_x, _) = x.dim();
        let (n_y, _) = y.dim();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                kernel_matrix[[i, j]] = self.compute(x.row(i), y.row(j));
            }
        }

        kernel_matrix
    }

    /// Get kernel parameters for serialization
    fn parameters(&self) -> HashMap<String, f64>;
}

/// Main kernel type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Linear kernel: K(x,y) = x^T y
    Linear,
    /// RBF/Gaussian kernel: K(x,y) = exp(-γ||x-y||²)
    Rbf { gamma: f64 },
    /// Polynomial kernel: K(x,y) = (γ x^T y + r)^d
    Polynomial { gamma: f64, coef0: f64, degree: f64 },
    /// Sigmoid kernel: K(x,y) = tanh(γ x^T y + r)
    Sigmoid { gamma: f64, coef0: f64 },
    /// Precomputed kernel matrix
    Precomputed,
    /// Custom user-defined kernel
    Custom(String),
    /// Cosine similarity kernel
    Cosine,
    /// Chi-squared kernel
    ChiSquared { gamma: f64 },
    /// Histogram intersection kernel
    Intersection,
    /// Hellinger kernel
    Hellinger,
    /// Jensen-Shannon kernel
    JensenShannon,
    /// Periodic kernel
    Periodic { length_scale: f64, period: f64 },
}

/// Create a kernel instance from a KernelType
pub fn create_kernel(kernel_type: KernelType) -> Result<Box<dyn Kernel>> {
    match kernel_type {
        KernelType::Linear => Ok(Box::new(LinearKernel)),
        KernelType::Rbf { gamma } => Ok(Box::new(RbfKernel { gamma })),
        KernelType::Polynomial {
            gamma,
            coef0,
            degree,
        } => Ok(Box::new(PolynomialKernel {
            gamma,
            coef0,
            degree,
        })),
        KernelType::Sigmoid { gamma, coef0 } => Ok(Box::new(SigmoidKernel { gamma, coef0 })),
        KernelType::Cosine => Ok(Box::new(CosineKernel)),
        KernelType::ChiSquared { gamma } => Ok(Box::new(ChiSquaredKernel { gamma })),
        KernelType::Intersection => Ok(Box::new(IntersectionKernel)),
        KernelType::Hellinger => Ok(Box::new(HellingerKernel)),
        KernelType::JensenShannon => Ok(Box::new(JensenShannonKernel)),
        KernelType::Periodic {
            length_scale,
            period,
        } => Ok(Box::new(PeriodicKernel {
            length_scale,
            period,
        })),
        KernelType::Precomputed => Err(SklearsError::InvalidParameter {
            name: "kernel_type".to_string(),
            reason: "precomputed kernels must be created with data".to_string(),
        }),
        KernelType::Custom(name) => Err(SklearsError::InvalidParameter {
            name: "kernel_type".to_string(),
            reason: format!("custom kernel '{}' not implemented", name),
        }),
    }
}

impl<K: Kernel> Kernel for Arc<K> {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        (**self).compute(x, y)
    }

    fn compute_matrix(&self, x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        (**self).compute_matrix(x, y)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        (**self).parameters()
    }
}

/// Linear kernel implementation
#[derive(Debug, Clone)]
pub struct LinearKernel;

impl Default for LinearKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Kernel for LinearKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        x.dot(&y)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// RBF (Gaussian) kernel implementation
#[derive(Debug, Clone)]
pub struct RbfKernel {
    pub gamma: f64,
}

impl RbfKernel {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl Kernel for RbfKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Compute squared distance inline without allocating temporary arrays
        let squared_distance: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let diff = xi - yi;
                diff * diff
            })
            .sum();

        (-self.gamma * squared_distance).exp()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), self.gamma);
        params
    }
}

/// Polynomial kernel implementation
#[derive(Debug, Clone)]
pub struct PolynomialKernel {
    pub gamma: f64,
    pub coef0: f64,
    pub degree: f64,
}

impl PolynomialKernel {
    pub fn new(gamma: f64, coef0: f64, degree: f64) -> Self {
        Self {
            gamma,
            coef0,
            degree,
        }
    }
}

impl Kernel for PolynomialKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let dot_product = x.dot(&y);
        (self.gamma * dot_product + self.coef0).powf(self.degree)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), self.gamma);
        params.insert("coef0".to_string(), self.coef0);
        params.insert("degree".to_string(), self.degree);
        params
    }
}

/// Sigmoid kernel implementation
#[derive(Debug, Clone)]
pub struct SigmoidKernel {
    pub gamma: f64,
    pub coef0: f64,
}

impl SigmoidKernel {
    pub fn new(gamma: f64, coef0: f64) -> Self {
        Self { gamma, coef0 }
    }
}

impl Kernel for SigmoidKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let dot_product = x.dot(&y);
        (self.gamma * dot_product + self.coef0).tanh()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), self.gamma);
        params.insert("coef0".to_string(), self.coef0);
        params
    }
}

/// Cosine similarity kernel implementation
#[derive(Debug, Clone)]
pub struct CosineKernel;

impl Kernel for CosineKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let dot_product = x.dot(&y);
        let x_norm = x.dot(&x).sqrt();
        let y_norm = y.dot(&y).sqrt();

        if x_norm == 0.0 || y_norm == 0.0 {
            0.0
        } else {
            dot_product / (x_norm * y_norm)
        }
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Chi-squared kernel implementation
#[derive(Debug, Clone)]
pub struct ChiSquaredKernel {
    pub gamma: f64,
}

impl ChiSquaredKernel {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl Kernel for ChiSquaredKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let chi_squared_distance = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| {
                if a + b > 0.0 {
                    (a - b).powi(2) / (a + b)
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        (-self.gamma * chi_squared_distance).exp()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), self.gamma);
        params
    }
}

/// Histogram intersection kernel implementation
#[derive(Debug, Clone)]
pub struct IntersectionKernel;

impl Kernel for IntersectionKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a.min(*b)).sum()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Periodic kernel implementation
#[derive(Debug, Clone)]
pub struct PeriodicKernel {
    pub length_scale: f64,
    pub period: f64,
}

impl PeriodicKernel {
    pub fn new(length_scale: f64, period: f64) -> Self {
        Self {
            length_scale,
            period,
        }
    }
}

impl Kernel for PeriodicKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Compute inline without allocating temporary arrays
        let sin_squared: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let diff = xi - yi;
                let sin_val = (std::f64::consts::PI * diff / self.period).sin();
                sin_val * sin_val
            })
            .sum();

        (-2.0 * sin_squared / (self.length_scale * self.length_scale)).exp()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("length_scale".to_string(), self.length_scale);
        params.insert("period".to_string(), self.period);
        params
    }
}

/// Custom kernel implementation
#[derive(Debug, Clone)]
pub struct CustomKernel {
    pub name: String,
    pub function: fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
}

impl CustomKernel {
    pub fn new(name: String, function: fn(ArrayView1<f64>, ArrayView1<f64>) -> f64) -> Self {
        Self { name, function }
    }
}

impl Kernel for CustomKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        (self.function)(x, y)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Main kernel function wrapper
#[derive(Debug, Clone)]
pub struct KernelFunction {
    kernel_type: KernelType,
}

impl KernelFunction {
    pub fn new(kernel_type: KernelType) -> Self {
        Self { kernel_type }
    }

    pub fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        match &self.kernel_type {
            KernelType::Linear => LinearKernel.compute(x, y),
            KernelType::Rbf { gamma } => RbfKernel::new(*gamma).compute(x, y),
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => PolynomialKernel::new(*gamma, *coef0, *degree).compute(x, y),
            KernelType::Sigmoid { gamma, coef0 } => {
                SigmoidKernel::new(*gamma, *coef0).compute(x, y)
            }
            KernelType::Cosine => CosineKernel.compute(x, y),
            KernelType::ChiSquared { gamma } => ChiSquaredKernel::new(*gamma).compute(x, y),
            KernelType::Intersection => IntersectionKernel.compute(x, y),
            KernelType::Periodic {
                length_scale,
                period,
            } => PeriodicKernel::new(*length_scale, *period).compute(x, y),
            KernelType::Precomputed => {
                // A precomputed kernel has no closed-form value for two arbitrary
                // feature vectors: the kernel matrix must be supplied directly and
                // indexed by sample position. The pointwise `compute` API cannot
                // express this, so refuse rather than fabricate a 0.0 entry.
                // `create_kernel` rejects `Precomputed` up front, so reaching here
                // indicates misuse of the pointwise API.
                panic!(
                    "precomputed kernel cannot be evaluated pointwise; supply the \
                     kernel matrix and index it by sample position instead"
                )
            }
            KernelType::Custom(_name) => {
                // Default custom implementation
                x.dot(&y)
            }
            KernelType::Hellinger => {
                // Hellinger kernel (Bhattacharyya coefficient)
                let x_normalized = normalize_vector(&x.to_owned());
                let y_normalized = normalize_vector(&y.to_owned());
                x_normalized
                    .iter()
                    .zip(y_normalized.iter())
                    .map(|(a, b)| (a * b).sqrt())
                    .sum::<f64>()
                    .sqrt()
            }
            KernelType::JensenShannon => {
                // Jensen-Shannon kernel
                let x_normalized = normalize_vector(&x.to_owned());
                let y_normalized = normalize_vector(&y.to_owned());

                let mut js_divergence = 0.0;
                for i in 0..x_normalized.len() {
                    let p = x_normalized[i];
                    let q = y_normalized[i];
                    let m = (p + q) / 2.0;

                    if p > 0.0 && m > 0.0 {
                        js_divergence += p * (p / m).ln();
                    }
                    if q > 0.0 && m > 0.0 {
                        js_divergence += q * (q / m).ln();
                    }
                }
                js_divergence /= 2.0;

                (-js_divergence).exp()
            }
        }
    }

    pub fn compute_matrix(&self, x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        let (n_x, _) = x.dim();
        let (n_y, _) = y.dim();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                kernel_matrix[[i, j]] = self.compute(x.row(i), y.row(j));
            }
        }

        kernel_matrix
    }

    pub fn kernel_type(&self) -> &KernelType {
        &self.kernel_type
    }
}

/// Utility function to normalize a vector
fn normalize_vector(vec: &Array1<f64>) -> Array1<f64> {
    let sum: f64 = vec.iter().sum();
    if sum == 0.0 {
        vec.clone()
    } else {
        vec / sum
    }
}

/// Graph structure for graph kernels
#[derive(Debug, Clone)]
pub struct Graph {
    pub adjacency_matrix: Array2<f64>,
    pub node_labels: Option<Array1<usize>>,
    pub edge_labels: Option<Array2<usize>>,
}

impl Graph {
    pub fn new(adjacency_matrix: Array2<f64>) -> Self {
        Self {
            adjacency_matrix,
            node_labels: None,
            edge_labels: None,
        }
    }

    pub fn with_node_labels(mut self, labels: Array1<usize>) -> Self {
        self.node_labels = Some(labels);
        self
    }

    pub fn with_edge_labels(mut self, labels: Array2<usize>) -> Self {
        self.edge_labels = Some(labels);
        self
    }
}

/// Random Walk kernel for graphs
#[derive(Debug, Clone)]
pub struct RandomWalkKernel {
    pub lambda: f64, // decay parameter
    pub max_steps: usize,
}

impl RandomWalkKernel {
    pub fn new(lambda: f64, max_steps: usize) -> Self {
        Self { lambda, max_steps }
    }

    /// Compute the random-walk graph kernel (Gärtner / Kashima et al.).
    ///
    /// The geometric random-walk kernel between graphs `G1` and `G2` is
    /// ```text
    /// K(G1, G2) = Σ_{i,j} [ (I - λ A_×)^{-1} ]_{ij}
    /// ```
    /// where `A_×` is the adjacency matrix of the direct (tensor) product graph
    /// `G1 × G2`, whose nodes are pairs `(u, v)` with `u ∈ G1`, `v ∈ G2`, and
    /// `A_×[(u,v),(u',v')] = A1[u,u'] · A2[v,v']`. The closed form sums the
    /// geometric series `Σ_t λ^t A_×^t = (I - λ A_×)^{-1}` over all start/end
    /// pairs (uniform start/stop probabilities).
    ///
    /// We form `A_×` of size `(n1·n2) × (n1·n2)`, then solve the linear system
    /// `(I - λ A_×) s = 1` for `s` and return `Σ_i s_i`. The kernel is
    /// well-defined when `λ < 1 / ρ(A_×)`; if the system is singular or the
    /// product graph is empty we fall back to `0.0`.
    pub fn compute_graph_kernel(&self, g1: &Graph, g2: &Graph) -> f64 {
        let n1 = g1.adjacency_matrix.nrows();
        let n2 = g2.adjacency_matrix.nrows();
        let n = n1 * n2;

        if n == 0 {
            return 0.0;
        }

        // Build M = I - λ A_×, where A_×[(a*n2+b), (c*n2+d)] = A1[a,c] * A2[b,d].
        let a1 = &g1.adjacency_matrix;
        let a2 = &g2.adjacency_matrix;
        let mut m = Array2::<f64>::zeros((n, n));
        for a in 0..n1 {
            for c in 0..n1 {
                let a1_ac = a1[[a, c]];
                if a1_ac == 0.0 {
                    continue;
                }
                for b in 0..n2 {
                    for d in 0..n2 {
                        let a2_bd = a2[[b, d]];
                        if a2_bd == 0.0 {
                            continue;
                        }
                        let row = a * n2 + b;
                        let col = c * n2 + d;
                        m[[row, col]] = -self.lambda * a1_ac * a2_bd;
                    }
                }
            }
        }
        for i in 0..n {
            m[[i, i]] += 1.0;
        }

        // Solve (I - λ A_×) s = 1 via Gaussian elimination with partial pivoting.
        let mut rhs = Array1::<f64>::from_elem(n, 1.0);
        match gaussian_solve(&mut m, &mut rhs) {
            Some(s) => s.sum(),
            None => 0.0,
        }
    }
}

/// Solve a dense linear system `A x = b` via Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is (numerically) singular. `a` and
/// `b` are consumed/overwritten as scratch space.
fn gaussian_solve(a: &mut Array2<f64>, b: &mut Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n || b.len() != n {
        return None;
    }

    for col in 0..n {
        // Partial pivoting: find the row with the largest magnitude in this column.
        let mut pivot_row = col;
        let mut pivot_val = a[[col, col]].abs();
        for row in (col + 1)..n {
            let val = a[[row, col]].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }

        if pivot_val < 1e-12 {
            return None; // Singular.
        }

        if pivot_row != col {
            for k in 0..n {
                let tmp = a[[col, k]];
                a[[col, k]] = a[[pivot_row, k]];
                a[[pivot_row, k]] = tmp;
            }
            b.swap(col, pivot_row);
        }

        // Eliminate below the pivot.
        let pivot = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                let sub = factor * a[[col, k]];
                a[[row, k]] -= sub;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution.
    let mut x = Array1::<f64>::zeros(n);
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[[row, k]] * x[k];
        }
        let diag = a[[row, row]];
        if diag.abs() < 1e-12 {
            return None;
        }
        x[row] = sum / diag;
    }

    Some(x)
}

/// Hellinger kernel implementation
#[derive(Debug)]
pub struct HellingerKernel;

impl Kernel for HellingerKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Hellinger kernel: K(x,y) = sum(sqrt(x_i * y_i))
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi * yi).sqrt())
            .sum()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Jensen-Shannon kernel implementation
#[derive(Debug)]
pub struct JensenShannonKernel;

impl JensenShannonKernel {
    fn jensen_shannon_divergence(&self, p: ArrayView1<f64>, q: ArrayView1<f64>) -> f64 {
        // Jensen-Shannon divergence
        let m: Vec<f64> = p
            .iter()
            .zip(q.iter())
            .map(|(pi, qi)| 0.5 * (pi + qi))
            .collect();
        let m = Array1::from_vec(m);

        let kl_pm = self.kl_divergence(p, m.view());
        let kl_qm = self.kl_divergence(q, m.view());

        0.5 * kl_pm + 0.5 * kl_qm
    }

    fn kl_divergence(&self, p: ArrayView1<f64>, q: ArrayView1<f64>) -> f64 {
        // Kullback-Leibler divergence
        p.iter()
            .zip(q.iter())
            .map(|(pi, qi)| {
                if *pi > 0.0 && *qi > 0.0 {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

impl Kernel for JensenShannonKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Jensen-Shannon kernel: K(x,y) = exp(-JS(x,y))
        let js_div = self.jensen_shannon_divergence(x, y);
        (-js_div).exp()
    }

    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Implement Kernel trait for KernelType to enable direct usage
impl Kernel for KernelType {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        match self {
            KernelType::Linear => LinearKernel.compute(x, y),
            KernelType::Rbf { gamma } => RbfKernel::new(*gamma).compute(x, y),
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => PolynomialKernel::new(*gamma, *coef0, *degree).compute(x, y),
            KernelType::Sigmoid { gamma, coef0 } => {
                SigmoidKernel::new(*gamma, *coef0).compute(x, y)
            }
            KernelType::Cosine => CosineKernel.compute(x, y),
            KernelType::ChiSquared { gamma } => ChiSquaredKernel::new(*gamma).compute(x, y),
            KernelType::Intersection => IntersectionKernel.compute(x, y),
            KernelType::Hellinger => HellingerKernel.compute(x, y),
            KernelType::JensenShannon => JensenShannonKernel.compute(x, y),
            KernelType::Periodic {
                length_scale,
                period,
            } => PeriodicKernel::new(*length_scale, *period).compute(x, y),
            KernelType::Precomputed => {
                // See the note in `KernelFunction::compute`: a precomputed kernel
                // must be indexed directly from a supplied matrix and has no
                // pointwise value. Refuse rather than fabricate a 0.0 entry.
                panic!(
                    "precomputed kernel cannot be evaluated pointwise; supply the \
                     kernel matrix and index it by sample position instead"
                )
            }
            KernelType::Custom(_name) => {
                // Default custom implementation
                x.dot(&y)
            }
        }
    }

    fn parameters(&self) -> HashMap<String, f64> {
        match self {
            KernelType::Linear => HashMap::new(),
            KernelType::Rbf { gamma } => {
                let mut params = HashMap::new();
                params.insert("gamma".to_string(), *gamma);
                params
            }
            KernelType::Polynomial {
                gamma,
                coef0,
                degree,
            } => {
                let mut params = HashMap::new();
                params.insert("gamma".to_string(), *gamma);
                params.insert("coef0".to_string(), *coef0);
                params.insert("degree".to_string(), *degree);
                params
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let mut params = HashMap::new();
                params.insert("gamma".to_string(), *gamma);
                params.insert("coef0".to_string(), *coef0);
                params
            }
            KernelType::Cosine => HashMap::new(),
            KernelType::ChiSquared { gamma } => {
                let mut params = HashMap::new();
                params.insert("gamma".to_string(), *gamma);
                params
            }
            KernelType::Intersection => HashMap::new(),
            KernelType::Hellinger => HashMap::new(),
            KernelType::JensenShannon => HashMap::new(),
            KernelType::Periodic {
                length_scale,
                period,
            } => {
                let mut params = HashMap::new();
                params.insert("length_scale".to_string(), *length_scale);
                params.insert("period".to_string(), *period);
                params
            }
            KernelType::Precomputed => HashMap::new(),
            KernelType::Custom(_name) => HashMap::new(),
        }
    }
}

/// Implement Kernel trait for `Box<dyn Kernel>` to enable polymorphic usage
impl Kernel for Box<dyn Kernel> {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        self.as_ref().compute(x, y)
    }

    fn compute_matrix(&self, x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        self.as_ref().compute_matrix(x, y)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        self.as_ref().parameters()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_kernel() {
        let kernel = LinearKernel;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let result = kernel.compute(x.view(), y.view());
        assert_abs_diff_eq!(result, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let kernel = RbfKernel::new(1.0);
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let result = kernel.compute(x.view(), y.view());
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_kernel() {
        let kernel = PolynomialKernel::new(1.0, 1.0, 2.0);
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![3.0, 4.0]);

        let result = kernel.compute(x.view(), y.view());
        let expected = (1.0_f64 * (1.0 * 3.0 + 2.0 * 4.0) + 1.0).powf(2.0);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_kernel() {
        let kernel = CosineKernel;
        let x = Array1::from_vec(vec![1.0, 0.0]);
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let result = kernel.compute(x.view(), y.view());
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_function() {
        let kernel_fn = KernelFunction::new(KernelType::Rbf { gamma: 0.5 });
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let result = kernel_fn.compute(x.view(), y.view());
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_matrix() {
        let kernel_fn = KernelFunction::new(KernelType::Linear);
        let x =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("array shape mismatch");
        let y =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("array shape mismatch");

        let kernel_matrix = kernel_fn.compute_matrix(&x, &y);

        assert_eq!(kernel_matrix.dim(), (2, 2));
        assert_abs_diff_eq!(kernel_matrix[[0, 0]], 5.0, epsilon = 1e-10); // [1,2] · [1,2] = 5
        assert_abs_diff_eq!(kernel_matrix[[1, 1]], 25.0, epsilon = 1e-10); // [3,4] · [3,4] = 25
    }
}
