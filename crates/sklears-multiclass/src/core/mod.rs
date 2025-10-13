//! Core multiclass classification strategies
//!
//! This module provides the fundamental binary-to-multiclass decomposition strategies:
//! One-vs-Rest, One-vs-One, and Error-Correcting Output Codes (ECOC).

pub mod ecoc;
pub mod one_vs_one;
pub mod one_vs_rest;

pub use ecoc::*;
pub use one_vs_one::*;
pub use one_vs_rest::*;
