//! Modular test organization for sklears-feature-extraction
//!
//! This module organizes the comprehensive test suite into domain-specific modules
//! for better maintainability and focused testing of feature extraction functionality.
//!
//! ## Test Domains
//!
//! - **text_tests**: Text processing and vectorization (Count, TF-IDF, LSA)
//! - **dict_tests**: Dictionary learning and dimensionality reduction (PCA, ICA, NMF, FA, PMF)
//! - **audio_tests**: Audio feature extraction (MFCC, spectral, chroma, tonnetz)
//! - **graph_tests**: Graph analysis and manifold learning (spectral, centrality, clustering)
//! - **signal_tests**: Signal processing (STFT, wavelets, Hilbert, filters)
//! - **simd_tests**: SIMD-accelerated operations (vectorized computations)
//! - **info_theory_tests**: Information theory, TDA, biological, image, and engineering features

pub mod audio_tests;
pub mod dict_tests;
pub mod graph_tests;
pub mod info_theory_tests;
pub mod signal_tests;
pub mod simd_tests;
pub mod text_tests;
pub mod trait_tests;
