//! Non-negative Matrix Factorization (NMF) implementation.
//!
//! NMF factorizes a non-negative matrix X into two non-negative matrices W and H
//! such that X ≈ WH. This is useful for dimensionality reduction and feature
//! extraction when the data is inherently non-negative.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{thread_rng, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform, Untrained},
};

/// NMF algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NMFSolver {
    /// Multiplicative Update algorithm
    MultiplicativeUpdate,
    /// Coordinate Descent algorithm
    #[default]
    CoordinateDescent,
    /// Alternating Least Squares algorithm
    ALS,
    /// Semi-NMF for mixed-sign data
    SemiNMF,
    /// Online NMF for streaming data
    OnlineNMF,
}

/// Initialization methods for NMF
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NMFInit {
    /// Random initialization
    #[default]
    Random,
    /// Non-negative Double Singular Value Decomposition
    Nndsvd,
    /// NNDSVD with zeros filled with average
    NndsvdA,
    /// NNDSVD with zeros filled with small random values
    NndsvdAr,
}

/// Regularization type for NMF
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NMFRegularization {
    /// No regularization
    #[default]
    None,
    /// L1 regularization
    L1,
    /// L2 regularization
    L2,
    /// Both L1 and L2 regularization
    Both,
}

/// Non-negative Matrix Factorization transformer
#[derive(Debug, Clone)]
pub struct NMF<State = Untrained> {
    /// Number of components/topics
    pub n_components: usize,
    /// Initialization method
    pub init: NMFInit,
    /// Solver algorithm
    pub solver: NMFSolver,
    /// Regularization type
    pub regularization: NMFRegularization,
    /// L1 regularization strength
    pub alpha_w: f64,
    /// L2 regularization strength
    pub alpha_h: f64,
    /// L1 ratio for elastic net regularization
    pub l1_ratio: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,

    /// Trained state
    state: State,
}

/// Trained NMF state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedNMF {
    /// Components matrix (H) - shape (n_components, n_features)
    pub components: Array2<f64>,
    /// Basis matrix (W) - shape (n_samples, n_components)
    pub w_matrix: Array2<f64>,
    /// Number of features in training data
    pub n_features_in: usize,
    /// Number of components
    pub n_components: usize,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Final reconstruction error
    pub reconstruction_err: f64,
}

impl NMF<Untrained> {
    /// Create a new NMF transformer
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            init: NMFInit::Random,
            solver: NMFSolver::CoordinateDescent,
            regularization: NMFRegularization::None,
            alpha_w: 0.0,
            alpha_h: 0.0,
            l1_ratio: 0.0,
            max_iter: 200,
            tol: 1e-4,
            random_state: None,
            state: Untrained,
        }
    }

    /// Set the initialization method
    pub fn init(mut self, init: NMFInit) -> Self {
        self.init = init;
        self
    }

    /// Set the solver algorithm
    pub fn solver(mut self, solver: NMFSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set regularization parameters
    pub fn regularization(mut self, regularization: NMFRegularization, alpha: f64) -> Self {
        self.regularization = regularization;
        match regularization {
            NMFRegularization::L1 => {
                self.alpha_w = alpha;
                self.alpha_h = alpha;
            }
            NMFRegularization::L2 => {
                self.alpha_w = alpha;
                self.alpha_h = alpha;
            }
            NMFRegularization::Both => {
                self.alpha_w = alpha;
                self.alpha_h = alpha;
            }
            NMFRegularization::None => {}
        }
        self
    }

    /// Set L1 ratio for elastic net
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Fit<Array2<f64>, ()> for NMF<Untrained> {
    type Fitted = NMF<TrainedNMF>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        // Check for non-negative data (except for Semi-NMF which allows mixed-sign data)
        if self.solver != NMFSolver::SemiNMF {
            for &val in x.iter() {
                if val < 0.0 {
                    return Err(SklearsError::InvalidInput(
                        "NMF requires non-negative data (use Semi-NMF for mixed-sign data)"
                            .to_string(),
                    ));
                }
            }
        }

        if self.n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot be larger than min(n_samples, n_features)".to_string(),
            ));
        }

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Initialize W and H matrices
        let (mut w, mut h) = self.initialize_matrices(x, &mut rng)?;

        // Run NMF algorithm
        let (n_iter, reconstruction_err) = match self.solver {
            NMFSolver::MultiplicativeUpdate => self.multiplicative_update(x, &mut w, &mut h)?,
            NMFSolver::CoordinateDescent => self.coordinate_descent(x, &mut w, &mut h)?,
            NMFSolver::ALS => self.alternating_least_squares(x, &mut w, &mut h)?,
            NMFSolver::SemiNMF => self.semi_nmf_update(x, &mut w, &mut h)?,
            NMFSolver::OnlineNMF => self.online_nmf_update(x, &mut w, &mut h)?,
        };

        Ok(NMF {
            n_components: self.n_components,
            init: self.init,
            solver: self.solver,
            regularization: self.regularization,
            alpha_w: self.alpha_w,
            alpha_h: self.alpha_h,
            l1_ratio: self.l1_ratio,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            state: TrainedNMF {
                components: h,
                w_matrix: w,
                n_features_in: n_features,
                n_components: self.n_components,
                n_iter,
                reconstruction_err,
            },
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for NMF<TrainedNMF> {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.state.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: self.state.n_features_in,
                actual: n_features,
            });
        }

        // Check for non-negative data (except for Semi-NMF which allows mixed-sign data)
        if self.solver != NMFSolver::SemiNMF {
            for &val in x.iter() {
                if val < 0.0 {
                    return Err(SklearsError::InvalidInput(
                        "NMF requires non-negative data (use Semi-NMF for mixed-sign data)"
                            .to_string(),
                    ));
                }
            }
        }

        // Solve for W given fixed H: X ≈ WH, so W = argmin_W ||X - WH||_F^2 s.t. W >= 0
        // where H is the learned components matrix
        let mut w = Array2::from_elem((n_samples, self.state.n_components), 0.1);

        // Use coordinate descent to solve for non-negative W
        for _ in 0..100 {
            // Fixed number of iterations for transform
            for k in 0..self.state.n_components {
                // Compute residual for component k
                let mut residual = x.clone();
                for j in 0..self.state.n_components {
                    if j != k {
                        let wj = w.column(j);
                        let hj = self.state.components.row(j);
                        for i in 0..residual.nrows() {
                            for l in 0..residual.ncols() {
                                residual[[i, l]] -= wj[i] * hj[l];
                            }
                        }
                    }
                }

                // Update W[:,k]
                let hk = self.state.components.row(k);
                let hk_norm_sq = hk.mapv(|x| x * x).sum();

                if hk_norm_sq > 1e-12 {
                    for i in 0..w.nrows() {
                        let numerator = residual.row(i).dot(&hk);
                        w[[i, k]] = (numerator / hk_norm_sq).max(0.0);
                    }
                }
            }
        }

        Ok(w)
    }
}

impl NMF<Untrained> {
    /// Initialize W and H matrices
    fn initialize_matrices(
        &self,
        x: &Array2<f64>,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x.dim();

        match self.init {
            NMFInit::Random => {
                let w = self.random_init((n_samples, self.n_components), rng);
                let h = self.random_init((self.n_components, n_features), rng);
                Ok((w, h))
            }
            NMFInit::Nndsvd | NMFInit::NndsvdA | NMFInit::NndsvdAr => self.nndsvd_init(x, rng),
        }
    }

    /// Random initialization
    fn random_init(&self, shape: (usize, usize), rng: &mut impl Rng) -> Array2<f64> {
        let (rows, cols) = shape;
        let mut matrix = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                matrix[[i, j]] = rng.gen::<f64>();
            }
        }

        matrix
    }

    /// NNDSVD initialization (simplified version)
    fn nndsvd_init(
        &self,
        x: &Array2<f64>,
        rng: &mut impl Rng,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x.dim();

        // For simplicity, we'll use a basic SVD approximation
        // In a full implementation, we would use proper SVD decomposition

        let avg = x.mean().unwrap_or(0.1);
        let mut w = Array2::from_elem((n_samples, self.n_components), avg.sqrt());
        let mut h = Array2::from_elem((self.n_components, n_features), avg.sqrt());

        // Add some randomness based on the initialization variant
        match self.init {
            NMFInit::NndsvdA => {
                // Fill zeros with average
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        if w[[i, j]] == 0.0 {
                            w[[i, j]] = avg;
                        }
                    }
                }
                for i in 0..self.n_components {
                    for j in 0..n_features {
                        if h[[i, j]] == 0.0 {
                            h[[i, j]] = avg;
                        }
                    }
                }
            }
            NMFInit::NndsvdAr => {
                // Fill zeros with small random values
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        if w[[i, j]] == 0.0 {
                            w[[i, j]] = avg * rng.gen::<f64>() * 0.01;
                        }
                    }
                }
                for i in 0..self.n_components {
                    for j in 0..n_features {
                        if h[[i, j]] == 0.0 {
                            h[[i, j]] = avg * rng.gen::<f64>() * 0.01;
                        }
                    }
                }
            }
            _ => {}
        }

        Ok((w, h))
    }

    /// Multiplicative update algorithm with regularization support
    fn multiplicative_update(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<(usize, f64)> {
        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update H with regularization: H := H ⊙ (W^T X) / (W^T W H + α_h * R_h)
            self.update_h_multiplicative_regularized(x, w, h)?;

            // Update W with regularization: W := W ⊙ (X H^T) / (W H H^T + α_w * R_w)
            self.update_w_multiplicative_regularized(x, w, h)?;

            // Calculate reconstruction error including regularization
            let reconstruction = w.dot(h);
            let mut error = self.frobenius_norm_squared(&(x - &reconstruction));

            // Add regularization terms to error
            error += self.compute_regularization_penalty(w, h);

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }

            prev_error = error;
        }

        Ok((n_iter, prev_error))
    }

    /// Update H matrix with multiplicative update and regularization
    fn update_h_multiplicative_regularized(
        &self,
        x: &Array2<f64>,
        w: &Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<()> {
        let numerator_h = w.t().dot(x);
        let base_denominator_h = w.t().dot(w).dot(h);

        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                let mut denominator = base_denominator_h[[i, j]];

                // Add regularization terms to denominator
                match self.regularization {
                    NMFRegularization::L1 => {
                        denominator += self.alpha_h;
                    }
                    NMFRegularization::L2 => {
                        denominator += 2.0 * self.alpha_h * h[[i, j]];
                    }
                    NMFRegularization::Both => {
                        // Elastic net: L1 + L2
                        denominator += self.l1_ratio * self.alpha_h
                            + (1.0 - self.l1_ratio) * 2.0 * self.alpha_h * h[[i, j]];
                    }
                    NMFRegularization::None => {}
                }

                if denominator > 1e-12 {
                    h[[i, j]] *= numerator_h[[i, j]] / denominator;
                    h[[i, j]] = h[[i, j]].max(1e-12); // Ensure non-negativity
                }
            }
        }
        Ok(())
    }

    /// Update W matrix with multiplicative update and regularization
    fn update_w_multiplicative_regularized(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &Array2<f64>,
    ) -> Result<()> {
        let numerator_w = x.dot(&h.t());
        let base_denominator_w = w.dot(h).dot(&h.t());

        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                let mut denominator = base_denominator_w[[i, j]];

                // Add regularization terms to denominator
                match self.regularization {
                    NMFRegularization::L1 => {
                        denominator += self.alpha_w;
                    }
                    NMFRegularization::L2 => {
                        denominator += 2.0 * self.alpha_w * w[[i, j]];
                    }
                    NMFRegularization::Both => {
                        // Elastic net: L1 + L2
                        denominator += self.l1_ratio * self.alpha_w
                            + (1.0 - self.l1_ratio) * 2.0 * self.alpha_w * w[[i, j]];
                    }
                    NMFRegularization::None => {}
                }

                if denominator > 1e-12 {
                    w[[i, j]] *= numerator_w[[i, j]] / denominator;
                    w[[i, j]] = w[[i, j]].max(1e-12); // Ensure non-negativity
                }
            }
        }
        Ok(())
    }

    /// Compute regularization penalty for the objective function
    fn compute_regularization_penalty(&self, w: &Array2<f64>, h: &Array2<f64>) -> f64 {
        let mut penalty = 0.0;

        match self.regularization {
            NMFRegularization::L1 => {
                // L1 penalty: α_w * ||W||_1 + α_h * ||H||_1
                penalty += self.alpha_w * w.mapv(|x| x.abs()).sum();
                penalty += self.alpha_h * h.mapv(|x| x.abs()).sum();
            }
            NMFRegularization::L2 => {
                // L2 penalty: α_w * ||W||_F^2 + α_h * ||H||_F^2
                penalty += self.alpha_w * w.mapv(|x| x * x).sum();
                penalty += self.alpha_h * h.mapv(|x| x * x).sum();
            }
            NMFRegularization::Both => {
                // Elastic net penalty
                let l1_penalty = w.mapv(|x| x.abs()).sum() + h.mapv(|x| x.abs()).sum();
                let l2_penalty = w.mapv(|x| x * x).sum() + h.mapv(|x| x * x).sum();
                penalty += self.l1_ratio * (self.alpha_w + self.alpha_h) * l1_penalty
                    + (1.0 - self.l1_ratio) * (self.alpha_w + self.alpha_h) * l2_penalty;
            }
            NMFRegularization::None => {}
        }

        penalty
    }

    /// Alternating Least Squares (ALS) algorithm for NMF
    ///
    /// This algorithm alternates between solving least squares problems for W and H
    /// while enforcing non-negativity constraints using projected gradient methods.
    fn alternating_least_squares(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<(usize, f64)> {
        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update H using least squares: min_H ||X - WH||_F^2 s.t. H >= 0
            self.update_h_als(x, w, h)?;

            // Update W using least squares: min_W ||X - WH||_F^2 s.t. W >= 0
            self.update_w_als(x, w, h)?;

            // Calculate reconstruction error
            let reconstruction = w.dot(h);
            let error = self.frobenius_norm_squared(&(x - &reconstruction));

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }

            prev_error = error;
        }

        Ok((n_iter, prev_error))
    }

    /// Update H matrix using projected least squares
    ///
    /// Solves: min_H ||X - WH||_F^2 s.t. H >= 0
    /// This is equivalent to solving: min_H ||W^T(X - WH)||_F^2 s.t. H >= 0
    /// Which gives us the normal equations: (W^T W) H = W^T X with non-negativity constraints
    fn update_h_als(&self, x: &Array2<f64>, w: &Array2<f64>, h: &mut Array2<f64>) -> Result<()> {
        // Compute W^T W (Gram matrix)
        let wtw = w.t().dot(w);

        // Compute W^T X (right-hand side)
        let wtx = w.t().dot(x);

        // Solve (W^T W) H = W^T X for each column of H using projected gradient descent
        for j in 0..h.ncols() {
            let mut h_col = h.column(j).to_owned();
            let rhs = wtx.column(j);

            // Projected gradient descent for this column
            let step_size = 0.01; // Fixed step size - could be adaptive
            for _gd_iter in 0..50 {
                // Inner iterations for gradient descent
                // Compute gradient: (W^T W) h_col - rhs
                let gradient = wtw.dot(&h_col) - rhs;

                // Gradient descent step
                h_col = &h_col - step_size * &gradient;

                // Project onto non-negative orthant
                h_col.mapv_inplace(|x| x.max(0.0));
            }

            // Update the column in H
            h.column_mut(j).assign(&h_col);
        }

        Ok(())
    }

    /// Update W matrix using projected least squares
    ///
    /// Solves: min_W ||X - WH||_F^2 s.t. W >= 0
    /// This is equivalent to solving: min_W ||(X - WH)H^T||_F^2 s.t. W >= 0
    /// Which gives us the normal equations: W (H H^T) = X H^T with non-negativity constraints
    fn update_w_als(&self, x: &Array2<f64>, w: &mut Array2<f64>, h: &Array2<f64>) -> Result<()> {
        // Compute H H^T (Gram matrix)
        let hht = h.dot(&h.t());

        // Compute X H^T (right-hand side)
        let xht = x.dot(&h.t());

        // Solve W (H H^T) = X H^T for each row of W using projected gradient descent
        for i in 0..w.nrows() {
            let mut w_row = w.row(i).to_owned();
            let rhs = xht.row(i);

            // Projected gradient descent for this row
            let step_size = 0.01; // Fixed step size - could be adaptive
            for _gd_iter in 0..50 {
                // Inner iterations for gradient descent
                // Compute gradient: w_row (H H^T) - rhs
                let gradient = w_row.dot(&hht) - rhs;

                // Gradient descent step
                w_row = &w_row - step_size * &gradient;

                // Project onto non-negative orthant
                w_row.mapv_inplace(|x| x.max(0.0));
            }

            // Update the row in W
            w.row_mut(i).assign(&w_row);
        }

        Ok(())
    }

    /// Coordinate descent algorithm with L1 regularization support
    fn coordinate_descent(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<(usize, f64)> {
        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update W column by column with regularization
            for k in 0..self.n_components {
                // Compute residual
                let mut residual = x.clone();
                for j in 0..self.n_components {
                    if j != k {
                        let wj = w.column(j);
                        let hj = h.row(j);
                        for i in 0..residual.nrows() {
                            for l in 0..residual.ncols() {
                                residual[[i, l]] -= wj[i] * hj[l];
                            }
                        }
                    }
                }

                // Update W[:,k] with regularization
                let hk = h.row(k);
                let hk_norm_sq = hk.mapv(|x| x * x).sum();

                if hk_norm_sq > 1e-12 {
                    for i in 0..w.nrows() {
                        let numerator = residual.row(i).dot(&hk);
                        let raw_update = numerator / hk_norm_sq;

                        // Apply regularization (soft thresholding for L1)
                        w[[i, k]] = self.apply_soft_thresholding_w(raw_update);
                    }
                }

                // Update H[k,:] with regularization
                let wk = w.column(k);
                let wk_norm_sq = wk.mapv(|x| x * x).sum();

                if wk_norm_sq > 1e-12 {
                    for j in 0..h.ncols() {
                        let numerator = wk.dot(&residual.column(j));
                        let raw_update = numerator / wk_norm_sq;

                        // Apply regularization (soft thresholding for L1)
                        h[[k, j]] = self.apply_soft_thresholding_h(raw_update);
                    }
                }
            }

            // Calculate reconstruction error including regularization
            let reconstruction = w.dot(h);
            let mut error = self.frobenius_norm_squared(&(x - &reconstruction));
            error += self.compute_regularization_penalty(w, h);

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }

            prev_error = error;
        }

        Ok((n_iter, prev_error))
    }

    /// Apply soft thresholding for L1 regularization on W matrix elements
    fn apply_soft_thresholding_w(&self, value: f64) -> f64 {
        match self.regularization {
            NMFRegularization::L1 => {
                // Soft thresholding: sign(x) * max(0, |x| - α)
                let threshold = self.alpha_w;
                if value > threshold {
                    value - threshold
                } else if value < -threshold {
                    value + threshold
                } else {
                    0.0
                }
            }
            NMFRegularization::L2 => {
                // L2 regularization: x / (1 + α)
                value / (1.0 + self.alpha_w)
            }
            NMFRegularization::Both => {
                // Elastic net: combine L1 and L2
                let l1_threshold = self.l1_ratio * self.alpha_w;
                let l2_factor = 1.0 + (1.0 - self.l1_ratio) * self.alpha_w;

                let soft_thresholded = if value > l1_threshold {
                    value - l1_threshold
                } else if value < -l1_threshold {
                    value + l1_threshold
                } else {
                    0.0
                };

                soft_thresholded / l2_factor
            }
            NMFRegularization::None => value,
        }
        .max(0.0) // Ensure non-negativity
    }

    /// Apply soft thresholding for L1 regularization on H matrix elements
    fn apply_soft_thresholding_h(&self, value: f64) -> f64 {
        match self.regularization {
            NMFRegularization::L1 => {
                // Soft thresholding: sign(x) * max(0, |x| - α)
                let threshold = self.alpha_h;
                if value > threshold {
                    value - threshold
                } else if value < -threshold {
                    value + threshold
                } else {
                    0.0
                }
            }
            NMFRegularization::L2 => {
                // L2 regularization: x / (1 + α)
                value / (1.0 + self.alpha_h)
            }
            NMFRegularization::Both => {
                // Elastic net: combine L1 and L2
                let l1_threshold = self.l1_ratio * self.alpha_h;
                let l2_factor = 1.0 + (1.0 - self.l1_ratio) * self.alpha_h;

                let soft_thresholded = if value > l1_threshold {
                    value - l1_threshold
                } else if value < -l1_threshold {
                    value + l1_threshold
                } else {
                    0.0
                };

                soft_thresholded / l2_factor
            }
            NMFRegularization::None => value,
        }
        .max(0.0) // Ensure non-negativity
    }

    /// Compute Frobenius norm squared
    fn frobenius_norm_squared(&self, matrix: &Array2<f64>) -> f64 {
        matrix.mapv(|x| x * x).sum()
    }

    /// Semi-NMF algorithm for mixed-sign data
    ///
    /// Semi-NMF allows the data matrix X to have negative values but enforces
    /// non-negativity constraints on the factor matrices W and H.
    /// The algorithm factorizes X ≈ WH where W >= 0 and H >= 0, but X can contain negative values.
    fn semi_nmf_update(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<(usize, f64)> {
        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update H: min_H ||X - WH||_F^2 s.t. H >= 0
            self.update_h_semi_nmf(x, w, h)?;

            // Update W: min_W ||X - WH||_F^2 s.t. W >= 0
            self.update_w_semi_nmf(x, w, h)?;

            // Compute reconstruction error
            let reconstruction = w.dot(h);
            let error = (x - &reconstruction).mapv(|x| x * x).sum().sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        Ok((n_iter, prev_error))
    }

    /// Update H matrix for Semi-NMF using multiplicative update rule
    fn update_h_semi_nmf(
        &self,
        x: &Array2<f64>,
        w: &Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<()> {
        let wt = w.t();
        let wt_x = wt.dot(x);
        let wt_w = wt.dot(w);
        let wt_w_h = wt_w.dot(h);

        // Separate positive and negative parts of W^T X
        let mut wt_x_pos = Array2::zeros(wt_x.dim());
        let mut wt_x_neg = Array2::zeros(wt_x.dim());

        for i in 0..wt_x.nrows() {
            for j in 0..wt_x.ncols() {
                if wt_x[[i, j]] >= 0.0 {
                    wt_x_pos[[i, j]] = wt_x[[i, j]];
                } else {
                    wt_x_neg[[i, j]] = -wt_x[[i, j]];
                }
            }
        }

        // Multiplicative update rule for Semi-NMF: H_ij = H_ij * (W^T X)^+_ij / ((W^T W H + (W^T X)^-)_ij + eps)
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                let numerator = wt_x_pos[[i, j]];
                let denominator = wt_w_h[[i, j]] + wt_x_neg[[i, j]] + 1e-10;
                h[[i, j]] *= numerator / denominator;
                h[[i, j]] = h[[i, j]].max(0.0); // Ensure non-negativity
            }
        }

        Ok(())
    }

    /// Update W matrix for Semi-NMF using multiplicative update rule
    fn update_w_semi_nmf(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &Array2<f64>,
    ) -> Result<()> {
        let ht = h.t();
        let x_ht = x.dot(&ht);
        let h_ht = h.dot(&ht);
        let w_h_ht = w.dot(&h_ht);

        // Separate positive and negative parts of X H^T
        let mut x_ht_pos = Array2::zeros(x_ht.dim());
        let mut x_ht_neg = Array2::zeros(x_ht.dim());

        for i in 0..x_ht.nrows() {
            for j in 0..x_ht.ncols() {
                if x_ht[[i, j]] >= 0.0 {
                    x_ht_pos[[i, j]] = x_ht[[i, j]];
                } else {
                    x_ht_neg[[i, j]] = -x_ht[[i, j]];
                }
            }
        }

        // Multiplicative update rule for Semi-NMF: W_ij = W_ij * (X H^T)^+_ij / ((W H H^T + (X H^T)^-)_ij + eps)
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                let numerator = x_ht_pos[[i, j]];
                let denominator = w_h_ht[[i, j]] + x_ht_neg[[i, j]] + 1e-10;
                w[[i, j]] *= numerator / denominator;
                w[[i, j]] = w[[i, j]].max(0.0); // Ensure non-negativity
            }
        }

        Ok(())
    }

    /// Online NMF algorithm for streaming data
    ///
    /// Online NMF processes data incrementally, updating the factorization as new data arrives.
    /// This is useful for streaming applications where data cannot be stored in memory.
    /// The algorithm uses stochastic gradient descent with momentum.
    fn online_nmf_update(
        &self,
        x: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &mut Array2<f64>,
    ) -> Result<(usize, f64)> {
        let (n_samples, _n_features) = x.dim();
        let batch_size = (n_samples / 10).max(1); // Process in mini-batches
        let learning_rate = 0.01;
        let momentum = 0.9;

        // Initialize momentum terms
        let mut w_momentum = Array2::zeros(w.dim());
        let mut h_momentum = Array2::zeros(h.dim());

        let mut total_error = 0.0;
        let mut n_iter = 0;

        // Process data in mini-batches
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch_indices: Vec<usize> = (batch_start..batch_end).collect();
            let x_batch = x.select(scirs2_core::ndarray::Axis(0), &batch_indices);

            n_iter += 1;

            // Update H for this batch using online learning
            self.update_h_online(&x_batch, w, h, &mut h_momentum, learning_rate, momentum)?;

            // Update W for this batch using online learning
            self.update_w_online(&x_batch, w, h, &mut w_momentum, learning_rate, momentum)?;

            // Compute batch reconstruction error
            let w_batch = w.select(scirs2_core::ndarray::Axis(0), &batch_indices);
            let reconstruction = w_batch.dot(h);
            let batch_error = (&x_batch - &reconstruction).mapv(|x| x * x).sum();
            total_error += batch_error;
        }

        let avg_error = (total_error / n_samples as f64).sqrt();
        Ok((n_iter, avg_error))
    }

    /// Update H matrix for Online NMF using stochastic gradient descent
    fn update_h_online(
        &self,
        x_batch: &Array2<f64>,
        w: &Array2<f64>,
        h: &mut Array2<f64>,
        h_momentum: &mut Array2<f64>,
        learning_rate: f64,
        momentum_coeff: f64,
    ) -> Result<()> {
        let (batch_size, _) = x_batch.dim();
        let _wt = w.t();

        // Compute gradients for H: ∇H = W^T(WH - X)
        let wh = w.dot(h);
        let wh_batch = wh.slice(scirs2_core::ndarray::s![0..batch_size, ..]);
        let residual = &wh_batch.to_owned() - x_batch;
        let w_batch = w.slice(scirs2_core::ndarray::s![0..batch_size, ..]);
        let grad_h = w_batch.t().dot(&residual);

        // Update H with momentum: H = H - lr * grad + momentum * prev_momentum
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                // Apply momentum
                h_momentum[[i, j]] =
                    momentum_coeff * h_momentum[[i, j]] - learning_rate * grad_h[[i, j]];

                // Update H
                h[[i, j]] += h_momentum[[i, j]];

                // Ensure non-negativity
                h[[i, j]] = h[[i, j]].max(0.0);
            }
        }

        Ok(())
    }

    /// Update W matrix for Online NMF using stochastic gradient descent
    fn update_w_online(
        &self,
        x_batch: &Array2<f64>,
        w: &mut Array2<f64>,
        h: &Array2<f64>,
        w_momentum: &mut Array2<f64>,
        learning_rate: f64,
        momentum_coeff: f64,
    ) -> Result<()> {
        let (batch_size, _) = x_batch.dim();

        // Compute gradients for W: ∇W = (WH - X)H^T
        let wh = w.dot(h);
        let wh_batch = wh.slice(scirs2_core::ndarray::s![0..batch_size, ..]);
        let residual = &wh_batch.to_owned() - x_batch;
        let grad_w_batch = residual.dot(&h.t());

        // Update W with momentum for the current batch
        for i in 0..batch_size {
            let global_i = i; // In real streaming, this would be mapped to global indices
            if global_i < w.nrows() {
                for j in 0..w.ncols() {
                    // Apply momentum
                    w_momentum[[global_i, j]] = momentum_coeff * w_momentum[[global_i, j]]
                        - learning_rate * grad_w_batch[[i, j]];

                    // Update W
                    w[[global_i, j]] += w_momentum[[global_i, j]];

                    // Ensure non-negativity
                    w[[global_i, j]] = w[[global_i, j]].max(0.0);
                }
            }
        }

        Ok(())
    }
}

impl NMF<TrainedNMF> {
    /// Get the components matrix (W)
    pub fn components(&self) -> Array2<f64> {
        self.state.components.t().to_owned()
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the reconstruction error
    pub fn reconstruction_err(&self) -> f64 {
        self.state.reconstruction_err
    }

    /// Inverse transform (reconstruct data from components)
    pub fn inverse_transform(&self, h: &Array2<f64>) -> Result<Array2<f64>> {
        let (_n_samples, n_components) = h.dim();

        if n_components != self.state.n_components {
            return Err(SklearsError::FeatureMismatch {
                expected: self.state.n_components,
                actual: n_components,
            });
        }

        // Reconstruct: X ≈ W * H (where h is n_samples x n_components, H is n_components x n_features)
        let reconstructed = h.dot(&self.state.components);
        Ok(reconstructed)
    }
}

impl Default for NMF<Untrained> {
    fn default() -> Self {
        Self::new(2)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nmf_creation() {
        let nmf = NMF::new(3)
            .init(NMFInit::Random)
            .solver(NMFSolver::MultiplicativeUpdate)
            .max_iter(100)
            .tol(1e-6)
            .random_state(42);

        assert_eq!(nmf.n_components, 3);
        assert_eq!(nmf.init, NMFInit::Random);
        assert_eq!(nmf.solver, NMFSolver::MultiplicativeUpdate);
        assert_eq!(nmf.max_iter, 100);
        assert_abs_diff_eq!(nmf.tol, 1e-6, epsilon = 1e-10);
    }

    #[test]
    fn test_nmf_fit_transform() {
        // Create non-negative data
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 3.0, 5.0],
            [2.0, 5.0, 8.0],
        ];

        let nmf = NMF::new(2).random_state(42);
        let trained_nmf = nmf.fit(&x, &()).unwrap();
        let h = trained_nmf.transform(&x).unwrap();

        assert_eq!(h.dim(), (4, 2));
        assert_eq!(trained_nmf.state.n_features_in, 3);
        assert_eq!(trained_nmf.state.n_components, 2);

        // Check that all values in H are non-negative
        for &val in h.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_nmf_inverse_transform() {
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0],];

        let nmf = NMF::new(2).random_state(123);
        let trained_nmf = nmf.fit(&x, &()).unwrap();
        let h = trained_nmf.transform(&x).unwrap();
        let x_reconstructed = trained_nmf.inverse_transform(&h).unwrap();

        assert_eq!(x_reconstructed.dim(), x.dim());

        // Check that all reconstructed values are non-negative
        for &val in x_reconstructed.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_nmf_different_solvers() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0],];

        let solvers = vec![
            NMFSolver::MultiplicativeUpdate,
            NMFSolver::CoordinateDescent,
            NMFSolver::ALS,
        ];

        for solver in solvers {
            let nmf = NMF::new(2).solver(solver).random_state(42);

            let trained_nmf = nmf.fit(&x, &()).unwrap();
            let h = trained_nmf.transform(&x).unwrap();

            assert_eq!(h.dim(), (3, 2));

            // Check non-negativity
            for &val in h.iter() {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_nmf_different_inits() {
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0],];

        let inits = vec![
            NMFInit::Random,
            NMFInit::Nndsvd,
            NMFInit::NndsvdA,
            NMFInit::NndsvdAr,
        ];

        for init in inits {
            let nmf = NMF::new(2).init(init).random_state(42);

            let trained_nmf = nmf.fit(&x, &()).unwrap();
            let h = trained_nmf.transform(&x).unwrap();

            assert_eq!(h.dim(), (3, 2));
        }
    }

    #[test]
    fn test_nmf_negative_data_error() {
        let x = array![
            [1.0, -2.0], // Contains negative value
            [2.0, 4.0],
        ];

        let nmf = NMF::new(2);
        let result = nmf.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_nmf_too_many_components() {
        let x = array![[1.0, 2.0], [2.0, 4.0],];

        let nmf = NMF::new(5); // More components than min(n_samples, n_features)
        let result = nmf.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_nmf_components_access() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 3.0, 5.0],];

        let nmf = NMF::new(2).random_state(42);
        let trained_nmf = nmf.fit(&x, &()).unwrap();

        let components = trained_nmf.components();
        assert_eq!(components.dim(), (3, 2));

        // Check that all components are non-negative
        for &val in components.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_sparse_nmf_l1_regularization() {
        let x = array![
            [1.0, 2.0, 0.0, 3.0],
            [0.0, 4.0, 5.0, 1.0],
            [2.0, 0.0, 6.0, 2.0],
            [3.0, 1.0, 0.0, 4.0],
        ];

        // Compare NMF without regularization vs with L1 regularization
        let nmf_no_reg = NMF::new(2)
            .regularization(NMFRegularization::None, 0.0)
            .solver(NMFSolver::CoordinateDescent)
            .max_iter(100)
            .random_state(42);

        let nmf_l1 = NMF::new(2)
            .regularization(NMFRegularization::L1, 0.5) // Higher regularization
            .solver(NMFSolver::CoordinateDescent)
            .max_iter(100)
            .random_state(42);

        let trained_no_reg = nmf_no_reg.fit(&x, &()).unwrap();
        let trained_l1 = nmf_l1.fit(&x, &()).unwrap();

        let h_no_reg = trained_no_reg.transform(&x).unwrap();
        let h_l1 = trained_l1.transform(&x).unwrap();

        // Check that results are non-negative
        for &val in h_no_reg.iter() {
            assert!(val >= 0.0);
        }
        for &val in h_l1.iter() {
            assert!(val >= 0.0);
        }

        // Count number of small values (closer to zero)
        let threshold = 0.1; // More reasonable threshold
        let sparse_count_no_reg = h_no_reg.iter().filter(|&&val| val < threshold).count();
        let sparse_count_l1 = h_l1.iter().filter(|&&val| val < threshold).count();

        // L1 regularization should tend to produce more small values (sparsity)
        // If not more sparse elements, then at least smaller values on average
        let mean_no_reg = h_no_reg.mean().unwrap();
        let mean_l1 = h_l1.mean().unwrap();

        // L1 regularization should generally produce smaller coefficients (shrinkage)
        assert!(
            sparse_count_l1 >= sparse_count_no_reg || mean_l1 <= mean_no_reg,
            "L1 regularization should induce sparsity or smaller coefficients. No reg: sparse={}, mean={:.3}, L1: sparse={}, mean={:.3}",
            sparse_count_no_reg, mean_no_reg, sparse_count_l1, mean_l1
        );
    }

    #[test]
    fn test_nmf_l2_regularization() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 3.0, 5.0],];

        let nmf_l2 = NMF::new(2)
            .regularization(NMFRegularization::L2, 0.01)
            .solver(NMFSolver::MultiplicativeUpdate)
            .random_state(42);

        let trained_l2 = nmf_l2.fit(&x, &()).unwrap();
        let h_l2 = trained_l2.transform(&x).unwrap();

        // Check that results are non-negative
        for &val in h_l2.iter() {
            assert!(val >= 0.0);
        }

        assert_eq!(h_l2.dim(), (3, 2));
    }

    #[test]
    fn test_nmf_elastic_net_regularization() {
        let x = array![
            [1.0, 2.0, 3.0, 0.0],
            [0.0, 4.0, 1.0, 2.0],
            [2.0, 0.0, 5.0, 1.0],
            [1.0, 3.0, 0.0, 4.0],
        ];

        let nmf_elastic = NMF::new(2)
            .regularization(NMFRegularization::Both, 0.05)
            .l1_ratio(0.5) // Equal mix of L1 and L2
            .solver(NMFSolver::CoordinateDescent)
            .max_iter(50)
            .random_state(42);

        let trained_elastic = nmf_elastic.fit(&x, &()).unwrap();
        let h_elastic = trained_elastic.transform(&x).unwrap();

        // Check that results are non-negative
        for &val in h_elastic.iter() {
            assert!(val >= 0.0);
        }

        assert_eq!(h_elastic.dim(), (4, 2));
    }

    #[test]
    fn test_nmf_regularization_comparison() {
        let x = array![[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 2.0, 2.0],];

        // Compare unregularized vs L1 regularized
        let nmf_none = NMF::new(2)
            .regularization(NMFRegularization::None, 0.0)
            .random_state(42);

        let nmf_l1 = NMF::new(2)
            .regularization(NMFRegularization::L1, 0.1)
            .random_state(42);

        let trained_none = nmf_none.fit(&x, &()).unwrap();
        let trained_l1 = nmf_l1.fit(&x, &()).unwrap();

        let h_none = trained_none.transform(&x).unwrap();
        let h_l1 = trained_l1.transform(&x).unwrap();

        // Both should be non-negative
        for &val in h_none.iter() {
            assert!(val >= 0.0);
        }
        for &val in h_l1.iter() {
            assert!(val >= 0.0);
        }

        // L1 regularized should generally have smaller values (shrinkage effect)
        let l1_sum = h_l1.sum();
        let none_sum = h_none.sum();

        // L1 regularization typically reduces the magnitude of coefficients
        assert!(l1_sum <= none_sum || (l1_sum - none_sum).abs() < 1e-2);
    }

    #[test]
    fn test_nmf_regularization_parameters() {
        let nmf = NMF::new(2)
            .regularization(NMFRegularization::L1, 0.05)
            .l1_ratio(0.7);

        assert_eq!(nmf.regularization, NMFRegularization::L1);
        assert_eq!(nmf.alpha_w, 0.05);
        assert_eq!(nmf.alpha_h, 0.05);
        assert_eq!(nmf.l1_ratio, 0.7);
    }
}
