//! AdaBoost ensemble methods
//!
//! This module implements AdaBoost and LogitBoost algorithms for classification.

mod ada_classifier;
mod config;
mod decision_tree;
mod helpers;
mod logit_classifier;
mod types;

#[cfg(test)]
mod tests;

pub use types::*;
