//! Distributed Discriminant Analysis
//!
//! This module provides distributed implementations of discriminant analysis algorithms
//! for large-scale datasets that can be processed across multiple cores or machines.

use crate::lda::LinearDiscriminantAnalysis;
use crate::qda::QuadraticDiscriminantAnalysis;
// ✅ Using SciRS2 dependencies following SciRS2 policy
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for distributed discriminant analysis
#[derive(Debug, Clone)]
pub struct DistributedDiscriminantConfig {
    /// chunk_size
    pub chunk_size: usize,
    /// min_chunk_size
    pub min_chunk_size: usize,
    /// max_parallel_jobs
    pub max_parallel_jobs: Option<usize>,
    /// enable_load_balancing
    pub enable_load_balancing: bool,
    /// merge_strategy
    pub merge_strategy: MergeStrategy,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// WeightedAverage
    WeightedAverage,
    /// SimpleAverage
    SimpleAverage,
}

impl Default for DistributedDiscriminantConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10000,

            min_chunk_size: 1000,

            max_parallel_jobs: None,
            enable_load_balancing: true,
            merge_strategy: MergeStrategy::WeightedAverage,
        }
    }
}

/// Distributed Linear Discriminant Analysis
#[derive(Debug, Clone)]
pub struct DistributedLinearDiscriminantAnalysis<State = Untrained> {
    config: DistributedDiscriminantConfig,
    base_lda: LinearDiscriminantAnalysis<State>,
    state: PhantomData<State>,
}

impl DistributedLinearDiscriminantAnalysis<Untrained> {
    pub fn new(config: DistributedDiscriminantConfig) -> Self {
        Self {
            config,
            base_lda: LinearDiscriminantAnalysis::default(),
            state: PhantomData,
        }
    }

    pub fn with_base_lda(
        config: DistributedDiscriminantConfig,
        base_lda: LinearDiscriminantAnalysis<Untrained>,
    ) -> Self {
        Self {
            config,
            base_lda,
            state: PhantomData,
        }
    }

    fn chunk_data<'a>(
        &self,
        x: ArrayView2<'a, Float>,
        y: &'a Array1<i32>,
    ) -> Result<Vec<(Array2<Float>, Array1<i32>)>> {
        let n_samples = x.shape()[0];
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No data chunks available".to_string(),
            ));
        }

        let chunk_size = if self.config.enable_load_balancing {
            let n_cpus = self.config.max_parallel_jobs.unwrap_or_else(num_cpus::get);
            std::cmp::max(self.config.min_chunk_size, n_samples / n_cpus)
        } else {
            self.config.chunk_size
        };

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < n_samples {
            let end = std::cmp::min(start + chunk_size, n_samples);

            let chunk_x = x.slice(s![start..end, ..]).to_owned();
            let chunk_y = y.slice(s![start..end]).to_owned();

            chunks.push((chunk_x, chunk_y));
            start = end;
        }

        Ok(chunks)
    }
}

impl Estimator for DistributedLinearDiscriminantAnalysis<Untrained> {
    type Config = DistributedDiscriminantConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for DistributedLinearDiscriminantAnalysis<Untrained> {
    type Fitted = DistributedLinearDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let chunks = self.chunk_data(x.view(), y)?;

        let models: std::result::Result<Vec<_>, _> = chunks
            .into_par_iter()
            .map(|(chunk_x, chunk_y)| self.base_lda.clone().fit(&chunk_x, &chunk_y))
            .collect();

        let models = models?;

        // For now, just return the first model
        // A proper implementation would aggregate the models
        let aggregated_model = models
            .into_iter()
            .next()
            .ok_or_else(|| SklearsError::InvalidInput("No models to aggregate".to_string()))?;

        Ok(DistributedLinearDiscriminantAnalysis {
            config: self.config,
            base_lda: aggregated_model,
            state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for DistributedLinearDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        self.base_lda.predict(x)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for DistributedLinearDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.base_lda.predict_proba(x)
    }
}

/// Distributed Quadratic Discriminant Analysis
#[derive(Debug, Clone)]
pub struct DistributedQuadraticDiscriminantAnalysis<State = Untrained> {
    config: DistributedDiscriminantConfig,
    base_qda: QuadraticDiscriminantAnalysis<State>,
    state: PhantomData<State>,
}

impl DistributedQuadraticDiscriminantAnalysis<Untrained> {
    pub fn new(config: DistributedDiscriminantConfig) -> Self {
        Self {
            config,
            base_qda: QuadraticDiscriminantAnalysis::default(),
            state: PhantomData,
        }
    }

    pub fn with_base_qda(
        config: DistributedDiscriminantConfig,
        base_qda: QuadraticDiscriminantAnalysis<Untrained>,
    ) -> Self {
        Self {
            config,
            base_qda,
            state: PhantomData,
        }
    }

    fn chunk_data<'a>(
        &self,
        x: ArrayView2<'a, Float>,
        y: &'a Array1<i32>,
    ) -> Result<Vec<(Array2<Float>, Array1<i32>)>> {
        let n_samples = x.shape()[0];
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No data chunks available".to_string(),
            ));
        }

        let chunk_size = if self.config.enable_load_balancing {
            let n_cpus = self.config.max_parallel_jobs.unwrap_or_else(num_cpus::get);
            std::cmp::max(self.config.min_chunk_size, n_samples / n_cpus)
        } else {
            self.config.chunk_size
        };

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < n_samples {
            let end = std::cmp::min(start + chunk_size, n_samples);

            let chunk_x = x.slice(s![start..end, ..]).to_owned();
            let chunk_y = y.slice(s![start..end]).to_owned();

            chunks.push((chunk_x, chunk_y));
            start = end;
        }

        Ok(chunks)
    }
}

impl Estimator for DistributedQuadraticDiscriminantAnalysis<Untrained> {
    type Config = DistributedDiscriminantConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for DistributedQuadraticDiscriminantAnalysis<Untrained> {
    type Fitted = DistributedQuadraticDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let chunks = self.chunk_data(x.view(), y)?;

        let models: std::result::Result<Vec<_>, _> = chunks
            .into_par_iter()
            .map(|(chunk_x, chunk_y)| self.base_qda.clone().fit(&chunk_x, &chunk_y))
            .collect();

        let models = models?;

        // For now, just return the first model
        // A proper implementation would aggregate the models
        let aggregated_model = models
            .into_iter()
            .next()
            .ok_or_else(|| SklearsError::InvalidInput("No models to aggregate".to_string()))?;

        Ok(DistributedQuadraticDiscriminantAnalysis {
            config: self.config,
            base_qda: aggregated_model,
            state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for DistributedQuadraticDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        self.base_qda.predict(x)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>>
    for DistributedQuadraticDiscriminantAnalysis<Trained>
{
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.base_qda.predict_proba(x)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_distributed_lda_basic() {
        let x = Array::from_shape_vec((100, 2), (0..200).map(|i| i as Float).collect()).unwrap();
        let y = Array::from_shape_vec(100, (0..100).map(|i| (i % 2) as i32).collect()).unwrap();

        let config = DistributedDiscriminantConfig {
            chunk_size: 25,
            ..Default::default()
        };

        let distributed_lda = DistributedLinearDiscriminantAnalysis::new(config);
        let trained = distributed_lda.fit(&x, &y).unwrap();

        let test_x = Array::from_shape_vec((10, 2), (0..20).map(|i| i as Float).collect()).unwrap();
        let predictions = trained.predict(&test_x).unwrap();

        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_distributed_qda_basic() {
        let x = Array::from_shape_vec((100, 2), (0..200).map(|i| i as Float).collect()).unwrap();
        let y = Array::from_shape_vec(100, (0..100).map(|i| (i % 2) as i32).collect()).unwrap();

        let config = DistributedDiscriminantConfig {
            chunk_size: 25,
            ..Default::default()
        };

        let distributed_qda = DistributedQuadraticDiscriminantAnalysis::new(config);
        let trained = distributed_qda.fit(&x, &y).unwrap();

        let test_x = Array::from_shape_vec((10, 2), (0..20).map(|i| i as Float).collect()).unwrap();
        let predictions = trained.predict(&test_x).unwrap();

        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_chunk_data() {
        let config = DistributedDiscriminantConfig {
            chunk_size: 25,
            min_chunk_size: 10,
            enable_load_balancing: false,
            ..Default::default()
        };

        let x = Array::from_shape_vec((100, 2), (0..200).map(|i| i as Float).collect()).unwrap();
        let y = Array::from_shape_vec(100, (0..100).map(|i| (i % 2) as i32).collect()).unwrap();

        let distributed_lda = DistributedLinearDiscriminantAnalysis::new(config);
        let chunks = distributed_lda.chunk_data(x.view(), &y).unwrap();

        assert_eq!(chunks.len(), 4); // 100 / 25 = 4 chunks
        assert_eq!(chunks[0].0.shape(), [25, 2]);
        assert_eq!(chunks[0].1.len(), 25);
    }
}
