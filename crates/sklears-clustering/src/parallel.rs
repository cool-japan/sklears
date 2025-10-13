//! Simplified parallel K-Means implementation
//!
//! This is a minimal working implementation to demonstrate parallel traits.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result,
    parallel::{ParallelConfig, ParallelFit, ParallelPredict},
};

use crate::kmeans::KMeansConfig;

/// Simple parallel K-Means for demonstration
#[derive(Debug, Clone)]
pub struct SimpleParallelKMeans {
    config: KMeansConfig,
    parallel_config: ParallelConfig,
}

/// Simple fitted parallel K-Means
#[derive(Debug, Clone)]
pub struct SimpleParallelKMeansFitted {
    cluster_centers: Array2<f64>,
    parallel_config: ParallelConfig,
}

impl SimpleParallelKMeans {
    /// Create new simple parallel K-Means
    pub fn new(config: KMeansConfig, parallel_config: ParallelConfig) -> Self {
        Self {
            config,
            parallel_config,
        }
    }
}

#[cfg(feature = "parallel")]
impl ParallelFit<Array2<f64>, ()> for SimpleParallelKMeans {
    type Fitted = SimpleParallelKMeansFitted;

    fn fit_parallel(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let config = self.parallel_config.clone();
        self.fit_parallel_with_config(x, _y, &config)
    }

    fn fit_parallel_with_config(
        self,
        x: &Array2<f64>,
        _y: &(),
        config: &ParallelConfig,
    ) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        // Simple initialization: use first K points as centers
        let mut centers = Array2::zeros((self.config.n_clusters, n_features));
        for i in 0..self.config.n_clusters.min(n_samples) {
            centers.row_mut(i).assign(&x.row(i));
        }

        // Simple implementation for demo
        for _iter in 0..5 {
            let labels = self.assign_points_parallel(x, &centers, config)?;
            centers = self.update_centers_parallel(x, &labels, config)?;
        }

        Ok(SimpleParallelKMeansFitted {
            cluster_centers: centers,
            parallel_config: config.clone(),
        })
    }
}

impl SimpleParallelKMeans {
    fn assign_points_parallel(
        &self,
        x: &Array2<f64>,
        centers: &Array2<f64>,
        config: &ParallelConfig,
    ) -> Result<Array1<usize>> {
        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        if !config.enabled || n_samples < config.min_parallel_batch_size {
            // Sequential assignment
            for i in 0..n_samples {
                labels[i] = self.find_closest_center(&x.row(i), centers);
            }
        } else {
            // Parallel assignment
            #[cfg(feature = "parallel")]
            {
                if let Some(labels_slice) = labels.as_slice_mut() {
                    labels_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, label)| {
                            *label = self.find_closest_center(&x.row(i), centers);
                        });
                }
            }
        }

        Ok(labels)
    }

    fn find_closest_center(
        &self,
        point: &scirs2_core::ndarray::ArrayView1<f64>,
        centers: &Array2<f64>,
    ) -> usize {
        let mut min_dist = f64::INFINITY;
        let mut closest = 0;

        for (i, center) in centers.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            let dist: f64 = point
                .iter()
                .zip(center.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();

            if dist < min_dist {
                min_dist = dist;
                closest = i;
            }
        }

        closest
    }

    fn update_centers_parallel(
        &self,
        x: &Array2<f64>,
        labels: &Array1<usize>,
        config: &ParallelConfig,
    ) -> Result<Array2<f64>> {
        let (_, n_features) = x.dim();
        let mut new_centers = Array2::zeros((self.config.n_clusters, n_features));

        for k in 0..self.config.n_clusters {
            let cluster_points: Vec<_> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == k)
                .map(|(i, _)| i)
                .collect();

            if !cluster_points.is_empty() {
                for j in 0..n_features {
                    let sum: f64 = cluster_points.iter().map(|&i| x[[i, j]]).sum();
                    new_centers[[k, j]] = sum / cluster_points.len() as f64;
                }
            }
        }

        Ok(new_centers)
    }
}

#[cfg(feature = "parallel")]
impl ParallelPredict<Array2<f64>, Array1<usize>> for SimpleParallelKMeansFitted {
    fn predict_parallel(&self, x: &Array2<f64>) -> Result<Array1<usize>> {
        self.predict_parallel_with_config(x, &self.parallel_config)
    }

    fn predict_parallel_with_config(
        &self,
        x: &Array2<f64>,
        config: &ParallelConfig,
    ) -> Result<Array1<usize>> {
        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        if !config.enabled || n_samples < config.min_parallel_batch_size {
            // Sequential prediction
            for i in 0..n_samples {
                labels[i] = self.find_closest_center(&x.row(i));
            }
        } else {
            // Parallel prediction
            #[cfg(feature = "parallel")]
            {
                if let Some(labels_slice) = labels.as_slice_mut() {
                    labels_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, label)| {
                            *label = self.find_closest_center(&x.row(i));
                        });
                }
            }
        }

        Ok(labels)
    }
}

impl SimpleParallelKMeansFitted {
    /// Get cluster centers
    pub fn cluster_centers(&self) -> &Array2<f64> {
        &self.cluster_centers
    }

    fn find_closest_center(&self, point: &scirs2_core::ndarray::ArrayView1<f64>) -> usize {
        let mut min_dist = f64::INFINITY;
        let mut closest = 0;

        for (i, center) in self
            .cluster_centers
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .enumerate()
        {
            let dist: f64 = point
                .iter()
                .zip(center.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();

            if dist < min_dist {
                min_dist = dist;
                closest = i;
            }
        }

        closest
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::KMeansInit;

    #[test]
    #[cfg(feature = "parallel")]
    fn test_simple_parallel_kmeans() {
        // Create simple test data
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let config = KMeansConfig {
            n_clusters: 2,
            init: KMeansInit::Random,
            max_iter: 10,
            tolerance: 1e-4,
            random_seed: Some(42),
        };

        let parallel_config = ParallelConfig {
            enabled: true,
            min_parallel_batch_size: 1,
            num_threads: None,
        };

        let kmeans = SimpleParallelKMeans::new(config, parallel_config);
        let fitted = kmeans.fit_parallel(&x, &()).unwrap();

        assert_eq!(fitted.cluster_centers().nrows(), 2);

        let predictions = fitted.predict_parallel(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
