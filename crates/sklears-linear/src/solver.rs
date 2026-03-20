//! Solver types for linear models

/// Solver algorithms for linear models
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Solver {
    /// Automatically choose the best solver
    #[default]
    Auto,
    /// Normal equations (closed form)
    Normal,
    /// L-BFGS optimizer from scirs2
    Lbfgs,
    /// Stochastic Average Gradient
    Sag,
    /// SAGA (improved SAG with L1 support)
    Saga,
    /// Coordinate Descent (for L1/ElasticNet)
    CoordinateDescent,
    /// Conjugate Gradient
    ConjugateGradient,
    /// Newton method
    Newton,
    /// ADMM (Alternating Direction Method of Multipliers)
    Admm,
    /// Proximal Gradient Method
    ProximalGradient,
    /// Accelerated Proximal Gradient Method (FISTA)
    Fista,
    /// Nesterov Accelerated Gradient Descent
    Nesterov,
}
