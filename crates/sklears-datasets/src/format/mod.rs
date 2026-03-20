//! Dataset format support for import/export
//!
//! This module provides functionality to export and import datasets in various formats.

pub mod core;
pub mod csv;
pub mod text_formats;

#[cfg(feature = "parquet")]
pub mod parquet;

#[cfg(feature = "hdf5")]
pub mod hdf5;

#[cfg(feature = "cloud-storage")]
pub mod cloud;

#[cfg(test)]
mod tests;

// Re-export main types
pub use core::*;
pub use csv::*;
pub use text_formats::*;

#[cfg(feature = "parquet")]
pub use self::parquet::*;

#[cfg(feature = "hdf5")]
pub use self::hdf5::*;

#[cfg(feature = "cloud-storage")]
pub use cloud::*;
