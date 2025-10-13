//! Few-shot learning methods for semi-supervised learning
//!
//! This module provides few-shot learning algorithms that can learn from very
//! limited labeled examples per class, often combined with unlabeled data.

pub mod maml;
pub mod matching_networks;
pub mod prototypical_networks;
pub mod relation_networks;

pub use maml::*;
pub use matching_networks::*;
pub use prototypical_networks::*;
pub use relation_networks::*;
