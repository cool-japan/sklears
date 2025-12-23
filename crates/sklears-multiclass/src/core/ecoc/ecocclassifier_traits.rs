//! # ECOCClassifier - Trait Implementations
//!
//! This module contains trait implementations for `ECOCClassifier`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//! - `Estimator`
//! - `Fit`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::types::ECOCClassifier;

impl<C: Clone> Clone for ECOCClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for ECOCClassifier<C, Untrained> {
    type Config = ECOCConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Implementation for classifiers that can fit binary problems
impl<C> Fit<Array2<Float>, Array1<i32>> for ECOCClassifier<C, Untrained>
where
    C: Clone + Send + Sync + Fit<Array2<Float>, Array1<Float>>,
    C::Fitted: Predict<Array2<Float>, Array1<Float>> + Send,
{
    type Fitted = TrainedECOC<C::Fitted>;
    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;
        let (_n_samples, n_features) = x.dim();
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();
        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for multiclass classification".to_string(),
            ));
        }
        let n_classes = classes.len();
        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };
        let code_matrix = if matches!(self.config.gpu_mode, GPUMode::MatrixOps | GPUMode::Full) {
            let dense_matrix = self.generate_code_matrix_gpu(n_classes, &mut rng)?;
            if self.config.use_sparse {
                let default_value = self.find_most_common_value(&dense_matrix);
                let sparse_matrix = SparseMatrix::from_dense(&dense_matrix, default_value);
                CodeMatrix::Sparse(sparse_matrix)
            } else {
                CodeMatrix::Dense(dense_matrix)
            }
        } else {
            self.generate_code_matrix(n_classes, &mut rng)?
        };
        let code_length = code_matrix.ncols();
        let binary_problems: Vec<_> = (0..code_length)
            .map(|bit_idx| {
                let binary_y: Array1<Float> = y.mapv(|label| {
                    let class_idx = classes.iter().position(|&c| c == label).unwrap();
                    if code_matrix.get(class_idx, bit_idx) == 1 {
                        1.0
                    } else {
                        0.0
                    }
                });
                binary_y
            })
            .collect();
        let estimators: SklResult<Vec<_>> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            binary_problems
                .into_par_iter()
                .map(|binary_y| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        } else {
            binary_problems
                .into_iter()
                .map(|binary_y| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        };
        let estimators = estimators?;
        Ok(ECOCClassifier {
            base_estimator: ECOCTrainedData {
                estimators,
                classes: Array1::from(classes),
                code_matrix,
                n_features,
            },
            config: self.config,
            state: PhantomData,
        })
    }
}
