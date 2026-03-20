//! Modular synthetic data generators
//!
//! This module organizes dataset generators into focused sub-modules:
//! - `basic`: Fundamental generators (blobs, classification, regression, circles, moons)
//! - `privacy`: Privacy-preserving and federated learning
//! - `multimodal`: Multi-modal and multi-agent environments
//! - `spatial`: Spatial statistics and geostatistical data
//! - `experimental`: A/B testing and experimental design
//! - `manifold`: Manifold learning and geometric patterns (TODO)
//! - `time_series`: Time series and temporal data (TODO)
//! - `adversarial`: Adversarial examples and robust datasets (TODO)
//! - `causal`: Causal inference and structural models (TODO)
//! - `domain_specific`: Domain-specific datasets (bioinformatics, NLP, computer vision) (TODO)
//! - `statistical`: Advanced statistical distributions and methods (TODO)

pub mod basic;
pub mod experimental;
pub mod multimodal;
pub mod performance;
pub mod privacy;
pub mod simd;
pub mod spatial;
pub mod type_safe;

// Re-export all functions from sub-modules for backward compatibility
pub use basic::*;
pub use experimental::*;
pub use multimodal::*;
pub use performance::*;
pub use privacy::*;
pub use simd::*;
pub use spatial::*;
pub use type_safe::*;

// TODO: Add other modules as they are created
// pub mod manifold;
// pub mod time_series;
// pub mod adversarial;
// pub mod causal;
// pub mod domain_specific;
// pub mod statistical;
