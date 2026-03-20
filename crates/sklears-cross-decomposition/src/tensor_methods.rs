//! Tensor-based cross-decomposition methods
//!
//! This module provides tensor-based extensions of cross-decomposition algorithms,
//! including Tensor CCA, Tucker decomposition, PARAFAC/CANDECOMP decomposition,
//! and advanced probabilistic tensor methods for multi-way data analysis.

mod common;
mod parafac;
mod probabilistic_tensor;
mod sparse_tensor;
mod tensor_cca;
mod tensor_completion;
mod tucker;

pub use common::{TensorInitMethod, Trained, Untrained};
pub use parafac::ParafacDecomposition;
pub use probabilistic_tensor::{
    BayesianParafac, ProbabilisticConfig, ProbabilisticTensorResults, ProbabilisticTucker,
    RobustProbabilisticTensor,
};
pub use sparse_tensor::SparseTensorDecomposition;
pub use tensor_cca::TensorCCA;
pub use tensor_completion::TensorCompletion;
pub use tucker::TuckerDecomposition;
