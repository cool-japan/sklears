//! Cross-Modal Discriminant Learning
//!
//! This module implements discriminant analysis for multi-modal data,
//! where features come from different modalities (e.g., text, images, audio).

pub mod types;
// pub mod modalities;
// pub mod untrained;
// pub mod trained;

// Re-export main types for convenience
pub use types::{
    CrossModalDiscriminantLearningConfig, DomainAdaptationStrategy, DomainInfo, FusionStrategy,
    ModalityAlignment, ModalityInfo,
};
// pub use modalities::{ModalityInfo, DomainInfo};
// pub use untrained::CrossModalDiscriminantLearningUntrained;
// pub use trained::CrossModalDiscriminantLearningTrained;

// Re-export with generic state parameter for backward compatibility
// pub use untrained::CrossModalDiscriminantLearningUntrained as CrossModalDiscriminantLearning;
