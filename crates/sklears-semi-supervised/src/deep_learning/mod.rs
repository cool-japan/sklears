//! Deep learning-based semi-supervised learning methods
//!
//! This module provides neural network-based semi-supervised learning algorithms
//! including Variational Autoencoders, consistency training methods, and modern
//! deep semi-supervised techniques.

pub mod autoregressive_models;
pub mod consistency_training;
pub mod deep_belief_networks;
pub mod deep_gaussian_processes;
pub mod energy_based_models;
pub mod flow_based_models;
pub mod ladder_networks;
pub mod mean_teacher;
pub mod neural_ode;
pub mod pi_model;
pub mod semi_supervised_gan;
pub mod semi_supervised_vae;
pub mod stacked_autoencoders;
pub mod temporal_ensembling;
pub mod virtual_adversarial_training;

pub use autoregressive_models::*;
pub use consistency_training::*;
pub use deep_belief_networks::*;
pub use deep_gaussian_processes::*;
pub use energy_based_models::*;
pub use flow_based_models::*;
pub use ladder_networks::*;
pub use mean_teacher::*;
pub use neural_ode::*;
pub use pi_model::*;
pub use semi_supervised_gan::*;
pub use semi_supervised_vae::*;
pub use stacked_autoencoders::*;
pub use temporal_ensembling::*;
pub use virtual_adversarial_training::*;
