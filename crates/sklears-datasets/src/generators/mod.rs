//! Modular synthetic data generators
//!
//! This module organizes dataset generators into focused sub-modules:
//! - `basic`: Fundamental generators (blobs, classification, regression, circles, moons)
//! - `privacy`: Privacy-preserving and federated learning
//! - `multimodal`: Multi-modal and multi-agent environments
//! - `spatial`: Spatial statistics and geostatistical data
//! - `experimental`: A/B testing and experimental design
//! - `manifold`: Manifold learning and geometric patterns
//! - `time_series`: Time series and temporal data
//! - `adversarial`: Adversarial examples and robust datasets
//! - `causal`: Causal inference and structural models
//! - `domain_specific`: Domain-specific datasets (finance, sensors, survival analysis)
//! - `statistical`: Advanced statistical distributions and methods

pub mod adversarial;
pub mod basic;
pub mod causal;
pub mod domain_specific;
pub mod experimental;
pub mod manifold;
pub mod multimodal;
pub mod performance;
pub mod privacy;
pub mod simd;
pub mod spatial;
pub mod statistical;
pub mod time_series;
pub mod type_safe;

// Re-export all functions from sub-modules for backward compatibility
pub use adversarial::*;
pub use basic::*;
pub use causal::*;
pub use domain_specific::*;
pub use experimental::*;
pub use manifold::*;
pub use multimodal::*;
pub use performance::*;
pub use privacy::*;
pub use simd::*;
pub use spatial::*;
pub use statistical::*;
pub use time_series::*;
pub use type_safe::*;
