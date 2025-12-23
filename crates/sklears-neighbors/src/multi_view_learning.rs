//! Multi-View Learning for Neighbor-Based Methods
//!
//! This module implements neighbor algorithms that work with multiple views or
//! representations of the same data. Multi-view learning is useful when data can be
//! represented in multiple ways (e.g., images with color, texture, and shape features).
//!
//! # Key Concepts
//!
//! - **View**: A specific representation or feature set of the data
//! - **Consensus**: Agreement across multiple views on neighbor relationships
//! - **View-Specific Learning**: Adapting distance metrics per view
//! - **Cross-View Analysis**: Analyzing consistency of neighbor patterns across views
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::multi_view_learning::{MultiViewKNeighborsClassifier, ViewConfig};
//! use sklears_neighbors::Distance;
//! use sklears_core::traits::Fit;
//! use scirs2_core::ndarray::Array2;
//!
//! // Create two views of data
//! let view1 = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
//! let view2 = Array2::from_shape_vec((4, 2), vec![5.0, 6.0, 6.0, 7.0, 7.0, 5.0, 8.0, 6.0]).unwrap();
//! let views = vec![view1, view2];
//! # let y = scirs2_core::ndarray::Array1::from_vec(vec![0, 0, 1, 1]);
//! # let configs = vec![ViewConfig::new(Distance::Euclidean, 1.0); 2];
//! # let classifier = MultiViewKNeighborsClassifier::new(3, configs);
//! # let _fitted = classifier.fit(&views, &y);
//! ```

use crate::distance::Distance;
use crate::NeighborsError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_core::types::Float;

/// Configuration for a single view
#[derive(Debug, Clone)]
pub struct ViewConfig {
    /// Distance metric for this view
    pub distance: Distance,
    /// Weight assigned to this view (higher = more important)
    pub weight: Float,
    /// Whether to learn an adaptive distance metric for this view
    pub learn_metric: bool,
}

impl ViewConfig {
    /// Create a new view configuration
    pub fn new(distance: Distance, weight: Float) -> Self {
        Self {
            distance,
            weight,
            learn_metric: false,
        }
    }

    /// Enable metric learning for this view
    pub fn with_metric_learning(mut self) -> Self {
        self.learn_metric = true;
        self
    }
}

/// Fusion strategy for combining predictions from multiple views
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// Weight predictions by view weights
    WeightedAverage,
    /// Multiply probabilities (product of experts)
    Product,
    /// Use majority voting
    MajorityVoting,
    /// Stack predictions and learn optimal combination
    Stacking,
}

/// Multi-View K-Nearest Neighbors Classifier
///
/// Learns from multiple views/representations of the data and combines
/// predictions using configurable fusion strategies.
#[derive(Debug, Clone)]
pub struct MultiViewKNeighborsClassifier<S> {
    k: usize,
    view_configs: Vec<ViewConfig>,
    fusion_strategy: FusionStrategy,
    consensus_threshold: Float,
    state: S,
}

/// Untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Trained state for multi-view classifier
#[derive(Debug, Clone)]
pub struct Trained {
    /// Training data for each view
    x_train_views: Vec<Array2<Float>>,
    /// Training labels
    y_train: Array1<i32>,
    /// Unique class labels
    classes: Array1<i32>,
    /// Learned view weights (may be updated during training)
    view_weights: Array1<Float>,
}

impl MultiViewKNeighborsClassifier<Untrained> {
    /// Create a new multi-view KNN classifier
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors to use
    /// * `view_configs` - Configuration for each view
    pub fn new(k: usize, view_configs: Vec<ViewConfig>) -> Self {
        Self {
            k,
            view_configs,
            fusion_strategy: FusionStrategy::WeightedAverage,
            consensus_threshold: 0.5,
            state: Untrained,
        }
    }

    /// Set the fusion strategy for combining view predictions
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set the consensus threshold for neighbor agreement across views
    pub fn with_consensus_threshold(mut self, threshold: Float) -> Self {
        self.consensus_threshold = threshold;
        self
    }
}

impl Fit<Vec<Array2<Float>>, Array1<i32>, MultiViewKNeighborsClassifier<Trained>>
    for MultiViewKNeighborsClassifier<Untrained>
{
    type Fitted = MultiViewKNeighborsClassifier<Trained>;

    fn fit(
        self,
        x_views: &Vec<Array2<Float>>,
        y: &Array1<i32>,
    ) -> Result<Self::Fitted, SklearsError> {
        // Validation
        if x_views.is_empty() {
            return Err(NeighborsError::InvalidInput("No views provided".to_string()).into());
        }

        if x_views.len() != self.view_configs.len() {
            return Err(NeighborsError::InvalidInput(format!(
                "Number of views ({}) doesn't match number of view configs ({})",
                x_views.len(),
                self.view_configs.len()
            ))
            .into());
        }

        let n_samples = x_views[0].nrows();

        // Check all views have same number of samples
        for view in x_views.iter() {
            if view.nrows() != n_samples {
                return Err(NeighborsError::ShapeMismatch {
                    expected: vec![n_samples],
                    actual: vec![view.nrows()],
                }
                .into());
            }
        }

        if y.len() != n_samples {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
            }
            .into());
        }

        if self.k >= n_samples {
            return Err(NeighborsError::InvalidInput(format!(
                "k={} should be less than n_samples={}",
                self.k, n_samples
            ))
            .into());
        }

        // Extract unique classes
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);

        // Initialize view weights from configs
        let view_weights = Array1::from_vec(
            self.view_configs
                .iter()
                .map(|config| config.weight)
                .collect(),
        );

        Ok(MultiViewKNeighborsClassifier {
            k: self.k,
            view_configs: self.view_configs,
            fusion_strategy: self.fusion_strategy,
            consensus_threshold: self.consensus_threshold,
            state: Trained {
                x_train_views: x_views.clone(),
                y_train: y.clone(),
                classes,
                view_weights,
            },
        })
    }
}

impl Predict<Vec<Array2<Float>>, Array1<i32>> for MultiViewKNeighborsClassifier<Trained> {
    fn predict(&self, x_views: &Vec<Array2<Float>>) -> Result<Array1<i32>, SklearsError> {
        if x_views.len() != self.state.x_train_views.len() {
            return Err(NeighborsError::InvalidInput(format!(
                "Expected {} views, got {}",
                self.state.x_train_views.len(),
                x_views.len()
            ))
            .into());
        }

        let n_samples = x_views[0].nrows();
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let pred = self.predict_single(x_views, i)?;
            predictions.push(pred);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl PredictProba<Vec<Array2<Float>>, Array2<Float>> for MultiViewKNeighborsClassifier<Trained> {
    fn predict_proba(&self, x_views: &Vec<Array2<Float>>) -> Result<Array2<Float>, SklearsError> {
        if x_views.len() != self.state.x_train_views.len() {
            return Err(NeighborsError::InvalidInput(format!(
                "Expected {} views, got {}",
                self.state.x_train_views.len(),
                x_views.len()
            ))
            .into());
        }

        let n_samples = x_views[0].nrows();
        let n_classes = self.state.classes.len();
        let mut proba_matrix = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let proba = self.predict_proba_single(x_views, i)?;
            proba_matrix.row_mut(i).assign(&proba);
        }

        Ok(proba_matrix)
    }
}

impl MultiViewKNeighborsClassifier<Trained> {
    /// Predict class for a single sample across all views
    fn predict_single(
        &self,
        x_views: &[Array2<Float>],
        sample_idx: usize,
    ) -> Result<i32, SklearsError> {
        let proba = self.predict_proba_single(x_views, sample_idx)?;
        let class_idx = proba
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        Ok(self.state.classes[class_idx])
    }

    /// Predict class probabilities for a single sample
    fn predict_proba_single(
        &self,
        x_views: &[Array2<Float>],
        sample_idx: usize,
    ) -> Result<Array1<Float>, SklearsError> {
        let n_classes = self.state.classes.len();

        match self.fusion_strategy {
            FusionStrategy::WeightedAverage => {
                let mut combined_proba = Array1::zeros(n_classes);
                let mut total_weight = 0.0;

                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let proba = self.predict_proba_for_view(view_idx, &query)?;
                    let weight = self.state.view_weights[view_idx];
                    combined_proba = combined_proba + &(proba * weight);
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    combined_proba = combined_proba / total_weight;
                }

                Ok(combined_proba)
            }
            FusionStrategy::Product => {
                let mut combined_proba = Array1::ones(n_classes);

                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let proba = self.predict_proba_for_view(view_idx, &query)?;
                    combined_proba = combined_proba * proba;
                }

                // Normalize
                let sum: Float = combined_proba.sum();
                if sum > 0.0 {
                    combined_proba = combined_proba / sum;
                }

                Ok(combined_proba)
            }
            FusionStrategy::MajorityVoting => {
                let mut votes = Array1::zeros(n_classes);

                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let proba = self.predict_proba_for_view(view_idx, &query)?;
                    let class_idx = proba
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    votes[class_idx] += 1.0;
                }

                // Normalize votes to probabilities
                let total: Float = votes.sum();
                if total > 0.0 {
                    votes = votes / total;
                }

                Ok(votes)
            }
            FusionStrategy::Stacking => {
                // Simple stacking: average probabilities from all views
                // In a full implementation, this would use a meta-learner
                let mut combined_proba = Array1::zeros(n_classes);

                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let proba = self.predict_proba_for_view(view_idx, &query)?;
                    combined_proba = combined_proba + &proba;
                }

                combined_proba = combined_proba / (x_views.len() as Float);
                Ok(combined_proba)
            }
        }
    }

    /// Predict probabilities for a single view
    fn predict_proba_for_view(
        &self,
        view_idx: usize,
        query: &ArrayView1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let distance = &self.view_configs[view_idx].distance;
        let train_view = &self.state.x_train_views[view_idx];

        // Find k nearest neighbors in this view
        let mut distances_indices: Vec<(Float, usize)> = train_view
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_indices: Vec<usize> = distances_indices
            .into_iter()
            .take(self.k)
            .map(|(_, idx)| idx)
            .collect();

        // Count classes among neighbors
        let mut class_counts = Array1::zeros(self.state.classes.len());
        for &idx in &neighbor_indices {
            let label = self.state.y_train[idx];
            if let Some(class_idx) = self.state.classes.iter().position(|&c| c == label) {
                class_counts[class_idx] += 1.0;
            }
        }

        // Normalize to probabilities
        let total: Float = class_counts.sum();
        if total > 0.0 {
            class_counts = class_counts / total;
        }

        Ok(class_counts)
    }

    /// Analyze consensus among views for neighbor relationships
    pub fn analyze_consensus(
        &self,
        x_views: &[Array2<Float>],
    ) -> Result<ConsensusAnalysis, SklearsError> {
        if x_views.len() != self.state.x_train_views.len() {
            return Err(NeighborsError::InvalidInput(format!(
                "Expected {} views, got {}",
                self.state.x_train_views.len(),
                x_views.len()
            ))
            .into());
        }

        let n_samples = x_views[0].nrows();
        let mut consensus_scores = Vec::with_capacity(n_samples);
        let mut view_agreements = Array2::zeros((x_views.len(), x_views.len()));

        for i in 0..n_samples {
            // Get neighbors from each view
            let mut view_neighbors = Vec::new();
            for (view_idx, view) in x_views.iter().enumerate() {
                let query = view.row(i);
                let neighbors = self.get_neighbors_for_view(view_idx, &query)?;
                view_neighbors.push(neighbors);
            }

            // Calculate consensus score (Jaccard similarity of neighbor sets)
            let mut consensus_sum = 0.0;
            let mut count = 0;
            for j in 0..view_neighbors.len() {
                for k in (j + 1)..view_neighbors.len() {
                    let intersection: usize = view_neighbors[j]
                        .iter()
                        .filter(|&n| view_neighbors[k].contains(n))
                        .count();
                    let union = view_neighbors[j].len() + view_neighbors[k].len() - intersection;
                    let jaccard = if union > 0 {
                        intersection as Float / union as Float
                    } else {
                        0.0
                    };
                    consensus_sum += jaccard;
                    view_agreements[[j, k]] += jaccard;
                    view_agreements[[k, j]] += jaccard;
                    count += 1;
                }
            }

            let consensus = if count > 0 {
                consensus_sum / count as Float
            } else {
                0.0
            };
            consensus_scores.push(consensus);
        }

        // Normalize view agreements
        if n_samples > 0 {
            view_agreements = view_agreements / (n_samples as Float);
        }

        let mean_consensus =
            consensus_scores.iter().sum::<Float>() / consensus_scores.len() as Float;

        Ok(ConsensusAnalysis {
            sample_consensus_scores: Array1::from_vec(consensus_scores),
            mean_consensus,
            view_agreement_matrix: view_agreements,
        })
    }

    /// Get neighbors for a single view
    fn get_neighbors_for_view(
        &self,
        view_idx: usize,
        query: &ArrayView1<Float>,
    ) -> Result<Vec<usize>, SklearsError> {
        let distance = &self.view_configs[view_idx].distance;
        let train_view = &self.state.x_train_views[view_idx];

        let mut distances_indices: Vec<(Float, usize)> = train_view
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(distances_indices
            .into_iter()
            .take(self.k)
            .map(|(_, idx)| idx)
            .collect())
    }
}

/// Result of consensus analysis across views
#[derive(Debug, Clone)]
pub struct ConsensusAnalysis {
    /// Consensus score for each sample (0-1, higher = more agreement)
    pub sample_consensus_scores: Array1<Float>,
    /// Mean consensus across all samples
    pub mean_consensus: Float,
    /// Agreement matrix between views (n_views x n_views)
    pub view_agreement_matrix: Array2<Float>,
}

/// Multi-View K-Nearest Neighbors Regressor
#[derive(Debug, Clone)]
pub struct MultiViewKNeighborsRegressor<S> {
    k: usize,
    view_configs: Vec<ViewConfig>,
    fusion_strategy: RegressionFusionStrategy,
    state: S,
}

/// Fusion strategy for regression
#[derive(Debug, Clone, Copy)]
pub enum RegressionFusionStrategy {
    /// Weight predictions by view weights
    WeightedAverage,
    /// Use median prediction
    Median,
    /// Stack predictions and learn optimal combination
    Stacking,
}

/// Trained state for regressor
#[derive(Debug, Clone)]
pub struct RegressorTrained {
    x_train_views: Vec<Array2<Float>>,
    y_train: Array1<Float>,
    view_weights: Array1<Float>,
}

impl MultiViewKNeighborsRegressor<Untrained> {
    /// Create a new multi-view KNN regressor
    pub fn new(k: usize, view_configs: Vec<ViewConfig>) -> Self {
        Self {
            k,
            view_configs,
            fusion_strategy: RegressionFusionStrategy::WeightedAverage,
            state: Untrained,
        }
    }

    /// Set the fusion strategy
    pub fn with_fusion_strategy(mut self, strategy: RegressionFusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }
}

impl Fit<Vec<Array2<Float>>, Array1<Float>, MultiViewKNeighborsRegressor<RegressorTrained>>
    for MultiViewKNeighborsRegressor<Untrained>
{
    type Fitted = MultiViewKNeighborsRegressor<RegressorTrained>;

    fn fit(
        self,
        x_views: &Vec<Array2<Float>>,
        y: &Array1<Float>,
    ) -> Result<Self::Fitted, SklearsError> {
        if x_views.is_empty() {
            return Err(NeighborsError::InvalidInput("No views provided".to_string()).into());
        }

        if x_views.len() != self.view_configs.len() {
            return Err(NeighborsError::InvalidInput(format!(
                "Number of views ({}) doesn't match number of view configs ({})",
                x_views.len(),
                self.view_configs.len()
            ))
            .into());
        }

        let n_samples = x_views[0].nrows();

        for view in x_views.iter() {
            if view.nrows() != n_samples {
                return Err(NeighborsError::ShapeMismatch {
                    expected: vec![n_samples],
                    actual: vec![view.nrows()],
                }
                .into());
            }
        }

        if y.len() != n_samples {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
            }
            .into());
        }

        let view_weights = Array1::from_vec(
            self.view_configs
                .iter()
                .map(|config| config.weight)
                .collect(),
        );

        Ok(MultiViewKNeighborsRegressor {
            k: self.k,
            view_configs: self.view_configs,
            fusion_strategy: self.fusion_strategy,
            state: RegressorTrained {
                x_train_views: x_views.clone(),
                y_train: y.clone(),
                view_weights,
            },
        })
    }
}

impl Predict<Vec<Array2<Float>>, Array1<Float>> for MultiViewKNeighborsRegressor<RegressorTrained> {
    fn predict(&self, x_views: &Vec<Array2<Float>>) -> Result<Array1<Float>, SklearsError> {
        if x_views.len() != self.state.x_train_views.len() {
            return Err(NeighborsError::InvalidInput(format!(
                "Expected {} views, got {}",
                self.state.x_train_views.len(),
                x_views.len()
            ))
            .into());
        }

        let n_samples = x_views[0].nrows();
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let pred = self.predict_single(x_views, i)?;
            predictions.push(pred);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl MultiViewKNeighborsRegressor<RegressorTrained> {
    fn predict_single(
        &self,
        x_views: &[Array2<Float>],
        sample_idx: usize,
    ) -> Result<Float, SklearsError> {
        match self.fusion_strategy {
            RegressionFusionStrategy::WeightedAverage => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let pred = self.predict_for_view(view_idx, &query)?;
                    let weight = self.state.view_weights[view_idx];
                    weighted_sum += pred * weight;
                    total_weight += weight;
                }

                Ok(if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                })
            }
            RegressionFusionStrategy::Median => {
                let mut predictions = Vec::new();
                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let pred = self.predict_for_view(view_idx, &query)?;
                    predictions.push(pred);
                }
                predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(predictions[predictions.len() / 2])
            }
            RegressionFusionStrategy::Stacking => {
                // Simple stacking: average predictions
                let mut sum = 0.0;
                for (view_idx, view) in x_views.iter().enumerate() {
                    let query = view.row(sample_idx);
                    let pred = self.predict_for_view(view_idx, &query)?;
                    sum += pred;
                }
                Ok(sum / (x_views.len() as Float))
            }
        }
    }

    fn predict_for_view(
        &self,
        view_idx: usize,
        query: &ArrayView1<Float>,
    ) -> Result<Float, SklearsError> {
        let distance = &self.view_configs[view_idx].distance;
        let train_view = &self.state.x_train_views[view_idx];

        let mut distances_indices: Vec<(Float, usize)> = train_view
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (distance.calculate(query, &row), i))
            .collect();

        distances_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_values: Vec<Float> = distances_indices
            .into_iter()
            .take(self.k)
            .map(|(_, idx)| self.state.y_train[idx])
            .collect();

        Ok(neighbor_values.iter().sum::<Float>() / neighbor_values.len() as Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_view_classifier_basic() {
        // Create two views of the same data
        let view1 = Array2::from_shape_vec(
            (6, 2),
            vec![
                -1.0, -1.0, -0.5, -1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 1.5, 1.5,
            ],
        )
        .unwrap();
        let view2 = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 0.0, 0.0, -1.0, -1.0, -0.5, -1.0, -1.5, -1.5,
            ],
        )
        .unwrap();
        let views = vec![view1, view2];
        let y = array![0, 0, 1, 1, 1, 1];

        let configs = vec![
            ViewConfig::new(Distance::Euclidean, 1.0),
            ViewConfig::new(Distance::Euclidean, 1.0),
        ];

        let classifier = MultiViewKNeighborsClassifier::new(3, configs);
        let fitted = classifier.fit(&views, &y).unwrap();

        let test_view1 = Array2::from_shape_vec((2, 2), vec![-0.8, -0.8, 0.8, 0.8]).unwrap();
        let test_view2 = Array2::from_shape_vec((2, 2), vec![0.8, 0.8, -0.8, -0.8]).unwrap();
        let test_views = vec![test_view1, test_view2];

        let predictions = fitted.predict(&test_views).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_multi_view_regressor() {
        let view1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        let view2 =
            Array2::from_shape_vec((4, 2), vec![5.0, 5.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0]).unwrap();
        let views = vec![view1, view2];
        let y = array![2.0, 3.0, 3.0, 4.0];

        let configs = vec![
            ViewConfig::new(Distance::Euclidean, 1.0),
            ViewConfig::new(Distance::Euclidean, 1.0),
        ];

        let regressor = MultiViewKNeighborsRegressor::new(2, configs);
        let fitted = regressor.fit(&views, &y).unwrap();

        let test_view1 = Array2::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();
        let test_view2 = Array2::from_shape_vec((1, 2), vec![5.5, 5.5]).unwrap();
        let test_views = vec![test_view1, test_view2];

        let predictions = fitted.predict(&test_views).unwrap();
        assert_eq!(predictions.len(), 1);
        assert!(predictions[0] > 2.0 && predictions[0] < 4.0);
    }

    #[test]
    fn test_consensus_analysis() {
        let view1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        let view2 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        let views_train = vec![view1, view2];
        let y = array![0, 0, 1, 1];

        let configs = vec![
            ViewConfig::new(Distance::Euclidean, 1.0),
            ViewConfig::new(Distance::Euclidean, 1.0),
        ];

        let classifier = MultiViewKNeighborsClassifier::new(2, configs);
        let fitted = classifier.fit(&views_train, &y).unwrap();

        let consensus = fitted.analyze_consensus(&views_train).unwrap();
        assert_eq!(consensus.sample_consensus_scores.len(), 4);
        assert!(consensus.mean_consensus >= 0.0 && consensus.mean_consensus <= 1.0);
    }

    #[test]
    fn test_fusion_strategies() {
        let view1 = Array2::from_shape_vec(
            (6, 2),
            vec![
                -1.0, -1.0, -0.5, -1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 1.5, 1.5,
            ],
        )
        .unwrap();
        let view2 = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 0.0, 0.0, -1.0, -1.0, -0.5, -1.0, -1.5, -1.5,
            ],
        )
        .unwrap();
        let views = vec![view1, view2];
        let y = array![0, 0, 1, 1, 1, 1];

        let strategies = vec![
            FusionStrategy::WeightedAverage,
            FusionStrategy::Product,
            FusionStrategy::MajorityVoting,
            FusionStrategy::Stacking,
        ];

        for strategy in strategies {
            let configs = vec![
                ViewConfig::new(Distance::Euclidean, 1.0),
                ViewConfig::new(Distance::Euclidean, 1.0),
            ];
            let classifier =
                MultiViewKNeighborsClassifier::new(3, configs).with_fusion_strategy(strategy);
            let fitted = classifier.fit(&views, &y).unwrap();

            let test_view1 = Array2::from_shape_vec((1, 2), vec![-0.8, -0.8]).unwrap();
            let test_view2 = Array2::from_shape_vec((1, 2), vec![0.8, 0.8]).unwrap();
            let test_views = vec![test_view1, test_view2];

            let predictions = fitted.predict(&test_views).unwrap();
            assert_eq!(predictions.len(), 1);
        }
    }

    #[test]
    fn test_view_config() {
        let config = ViewConfig::new(Distance::Euclidean, 1.5).with_metric_learning();
        assert_eq!(config.weight, 1.5);
        assert!(config.learn_metric);
    }

    #[test]
    fn test_error_handling() {
        // Mismatched number of views
        let view1 = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let views = vec![view1];
        let y = array![0, 0, 1, 1];
        let configs = vec![
            ViewConfig::new(Distance::Euclidean, 1.0),
            ViewConfig::new(Distance::Euclidean, 1.0),
        ];

        let classifier = MultiViewKNeighborsClassifier::new(2, configs);
        assert!(classifier.fit(&views, &y).is_err());

        // Empty views
        let empty_views: Vec<Array2<Float>> = vec![];
        let configs = vec![ViewConfig::new(Distance::Euclidean, 1.0)];
        let classifier = MultiViewKNeighborsClassifier::new(2, configs);
        assert!(classifier.fit(&empty_views, &y).is_err());
    }
}
