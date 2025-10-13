//! Visualization and interpretation utilities for matrix decomposition
//!
//! This module provides tools for visualizing and interpreting the results of
//! matrix decomposition algorithms including:
//! - Component visualization utilities
//! - Loading plot generation
//! - Biplot creation
//! - Component contribution analysis
//! - Feature importance ranking

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Component visualization data structure
#[derive(Debug, Clone)]
pub struct ComponentVisualization {
    /// Component loadings/weights
    pub loadings: Array2<Float>,
    /// Component scores (transformed data)
    pub scores: Array2<Float>,
    /// Explained variance for each component
    pub explained_variance: Array1<Float>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Array1<Float>,
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
    /// Component names (optional)
    pub component_names: Option<Vec<String>>,
}

impl ComponentVisualization {
    /// Create a new ComponentVisualization
    pub fn new(
        loadings: Array2<Float>,
        scores: Array2<Float>,
        explained_variance: Array1<Float>,
    ) -> Self {
        let total_variance = explained_variance.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|x| x / total_variance)
        } else {
            Array1::zeros(explained_variance.len())
        };

        Self {
            loadings,
            scores,
            explained_variance,
            explained_variance_ratio,
            feature_names: None,
            component_names: None,
        }
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set component names
    pub fn with_component_names(mut self, names: Vec<String>) -> Self {
        self.component_names = Some(names);
        self
    }

    /// Get loading plot data for specified components
    pub fn loading_plot_data(&self, comp1: usize, comp2: usize) -> Result<LoadingPlotData> {
        if comp1 >= self.loadings.ncols() || comp2 >= self.loadings.ncols() {
            return Err(SklearsError::InvalidInput(
                "Component indices out of bounds".to_string(),
            ));
        }

        let loadings1 = self.loadings.column(comp1).to_owned();
        let loadings2 = self.loadings.column(comp2).to_owned();

        Ok(LoadingPlotData {
            x_loadings: loadings1,
            y_loadings: loadings2,
            component1: comp1,
            component2: comp2,
            explained_variance1: self.explained_variance_ratio[comp1],
            explained_variance2: self.explained_variance_ratio[comp2],
            feature_names: self.feature_names.clone(),
        })
    }

    /// Generate biplot data combining scores and loadings
    pub fn biplot_data(&self, comp1: usize, comp2: usize) -> Result<BiplotData> {
        if comp1 >= self.scores.ncols() || comp2 >= self.scores.ncols() {
            return Err(SklearsError::InvalidInput(
                "Component indices out of bounds".to_string(),
            ));
        }

        let scores1 = self.scores.column(comp1).to_owned();
        let scores2 = self.scores.column(comp2).to_owned();

        let loading_plot = self.loading_plot_data(comp1, comp2)?;

        Ok(BiplotData {
            x_scores: scores1,
            y_scores: scores2,
            loading_plot,
        })
    }

    /// Calculate feature importance for each component
    pub fn feature_importance(&self) -> Array2<Float> {
        // Calculate importance as squared loadings normalized by explained variance
        let mut importance = Array2::zeros(self.loadings.dim());

        for i in 0..self.loadings.ncols() {
            let variance_weight = self.explained_variance_ratio[i];
            for j in 0..self.loadings.nrows() {
                importance[[j, i]] = self.loadings[[j, i]].powi(2) * variance_weight;
            }
        }

        importance
    }

    /// Get the most important features for each component
    pub fn top_features_per_component(&self, n_features: usize) -> Vec<Vec<FeatureImportance>> {
        let importance = self.feature_importance();
        let mut results = Vec::new();

        for comp in 0..importance.ncols() {
            let mut feature_scores: Vec<(usize, Float)> = importance
                .column(comp)
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();

            // Sort by importance (descending)
            feature_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_features: Vec<FeatureImportance> = feature_scores
                .into_iter()
                .take(n_features)
                .map(|(idx, score)| FeatureImportance {
                    feature_index: idx,
                    feature_name: self
                        .feature_names
                        .as_ref()
                        .and_then(|names| names.get(idx))
                        .cloned()
                        .unwrap_or_else(|| format!("feature_{idx}")),
                    importance_score: score,
                    loading_value: self.loadings[[idx, comp]],
                })
                .collect();

            results.push(top_features);
        }

        results
    }

    /// Calculate component contribution to total variance
    pub fn component_contributions(&self) -> ComponentContribution {
        let cumulative_variance: Array1<Float> = {
            let mut cum = Array1::zeros(self.explained_variance_ratio.len());
            let mut sum = 0.0;
            for i in 0..self.explained_variance_ratio.len() {
                sum += self.explained_variance_ratio[i];
                cum[i] = sum;
            }
            cum
        };

        let total_variance_explained = cumulative_variance[cumulative_variance.len() - 1];

        ComponentContribution {
            individual_variance: self.explained_variance_ratio.clone(),
            cumulative_variance,
            total_variance_explained,
        }
    }

    /// Generate scree plot data (eigenvalues vs component number)
    pub fn scree_plot_data(&self) -> ScreePlotData {
        let component_numbers: Array1<Float> =
            Array1::from_shape_fn(self.explained_variance.len(), |i| (i + 1) as Float);

        ScreePlotData {
            component_numbers,
            eigenvalues: self.explained_variance.clone(),
            explained_variance_ratios: self.explained_variance_ratio.clone(),
        }
    }

    /// Find optimal number of components using various criteria
    pub fn optimal_components(&self) -> OptimalComponentsAnalysis {
        let n_components = self.explained_variance.len();

        // Kaiser criterion: eigenvalues > mean eigenvalue
        let mean_eigenvalue = self.explained_variance.mean().unwrap_or(0.0);
        let kaiser_components = self
            .explained_variance
            .iter()
            .position(|&x| x <= mean_eigenvalue)
            .unwrap_or(n_components);

        // Elbow method: find the "elbow" in the scree plot
        let elbow_components = self.find_elbow_point();

        // 95% variance criterion
        let variance_95_components = self
            .explained_variance_ratio
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .position(|x| x >= 0.95)
            .map(|x| x + 1)
            .unwrap_or(n_components);

        OptimalComponentsAnalysis {
            kaiser_criterion: kaiser_components,
            elbow_method: elbow_components,
            variance_95_percent: variance_95_components,
            total_components: n_components,
        }
    }

    /// Find elbow point using second derivative method
    fn find_elbow_point(&self) -> usize {
        let n = self.explained_variance.len();
        if n < 3 {
            return n;
        }

        let mut second_derivatives = Vec::new();

        // Calculate second derivatives
        for i in 1..n - 1 {
            let second_deriv = self.explained_variance[i - 1] - 2.0 * self.explained_variance[i]
                + self.explained_variance[i + 1];
            second_derivatives.push(second_deriv.abs());
        }

        // Find the point with maximum second derivative
        let elbow_idx = second_derivatives
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx + 2) // Adjust for the offset
            .unwrap_or(n);

        elbow_idx.min(n)
    }
}

/// Loading plot data structure
#[derive(Debug, Clone)]
pub struct LoadingPlotData {
    /// Loadings for first component (x-axis)
    pub x_loadings: Array1<Float>,
    /// Loadings for second component (y-axis)
    pub y_loadings: Array1<Float>,
    /// Index of first component
    pub component1: usize,
    /// Index of second component
    pub component2: usize,
    /// Explained variance ratio for first component
    pub explained_variance1: Float,
    /// Explained variance ratio for second component
    pub explained_variance2: Float,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl LoadingPlotData {
    /// Get axis labels with explained variance information
    pub fn axis_labels(&self) -> (String, String) {
        let x_label = format!(
            "PC{} ({:.1}%)",
            self.component1 + 1,
            self.explained_variance1 * 100.0
        );
        let y_label = format!(
            "PC{} ({:.1}%)",
            self.component2 + 1,
            self.explained_variance2 * 100.0
        );
        (x_label, y_label)
    }

    /// Get feature coordinates for plotting
    pub fn feature_coordinates(&self) -> Vec<(Float, Float, String)> {
        let n_features = self.x_loadings.len();
        let mut coordinates = Vec::with_capacity(n_features);

        for i in 0..n_features {
            let name = self
                .feature_names
                .as_ref()
                .and_then(|names| names.get(i))
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}"));

            coordinates.push((self.x_loadings[i], self.y_loadings[i], name));
        }

        coordinates
    }
}

/// Biplot data combining scores and loadings
#[derive(Debug, Clone)]
pub struct BiplotData {
    /// Scores for first component (x-axis)
    pub x_scores: Array1<Float>,
    /// Scores for second component (y-axis)
    pub y_scores: Array1<Float>,
    /// Loading plot data
    pub loading_plot: LoadingPlotData,
}

impl BiplotData {
    /// Get sample coordinates for plotting
    pub fn sample_coordinates(&self) -> Vec<(Float, Float)> {
        self.x_scores
            .iter()
            .zip(self.y_scores.iter())
            .map(|(&x, &y)| (x, y))
            .collect()
    }

    /// Scale loadings for better visualization with scores
    pub fn scaled_loadings(&self, scale_factor: Float) -> Vec<(Float, Float, String)> {
        self.loading_plot
            .feature_coordinates()
            .into_iter()
            .map(|(x, y, name)| (x * scale_factor, y * scale_factor, name))
            .collect()
    }
}

/// Feature importance information
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Index of the feature
    pub feature_index: usize,
    /// Name of the feature
    pub feature_name: String,
    /// Importance score (squared loading weighted by explained variance)
    pub importance_score: Float,
    /// Raw loading value
    pub loading_value: Float,
}

/// Component contribution analysis
#[derive(Debug, Clone)]
pub struct ComponentContribution {
    /// Individual explained variance ratio for each component
    pub individual_variance: Array1<Float>,
    /// Cumulative explained variance ratio
    pub cumulative_variance: Array1<Float>,
    /// Total variance explained by all components
    pub total_variance_explained: Float,
}

impl ComponentContribution {
    /// Get components needed to explain a certain percentage of variance
    pub fn components_for_variance(&self, threshold: Float) -> usize {
        self.cumulative_variance
            .iter()
            .position(|&x| x >= threshold)
            .map(|x| x + 1)
            .unwrap_or(self.cumulative_variance.len())
    }

    /// Get variance explained by first n components
    pub fn variance_explained_by_n_components(&self, n: usize) -> Float {
        if n == 0 {
            0.0
        } else if n >= self.cumulative_variance.len() {
            self.total_variance_explained
        } else {
            self.cumulative_variance[n - 1]
        }
    }
}

/// Scree plot data
#[derive(Debug, Clone)]
pub struct ScreePlotData {
    /// Component numbers (1-indexed)
    pub component_numbers: Array1<Float>,
    /// Eigenvalues for each component
    pub eigenvalues: Array1<Float>,
    /// Explained variance ratios
    pub explained_variance_ratios: Array1<Float>,
}

/// Optimal number of components analysis
#[derive(Debug, Clone)]
pub struct OptimalComponentsAnalysis {
    /// Kaiser criterion (eigenvalues > mean)
    pub kaiser_criterion: usize,
    /// Elbow method result
    pub elbow_method: usize,
    /// Components for 95% variance
    pub variance_95_percent: usize,
    /// Total number of components
    pub total_components: usize,
}

impl OptimalComponentsAnalysis {
    /// Get recommendation based on multiple criteria
    pub fn recommendation(&self) -> ComponentRecommendation {
        let criteria = vec![
            self.kaiser_criterion,
            self.elbow_method,
            self.variance_95_percent,
        ];

        // Simple consensus: take the median
        let mut sorted_criteria = criteria.clone();
        sorted_criteria.sort();
        let median = sorted_criteria[sorted_criteria.len() / 2];

        ComponentRecommendation {
            recommended_components: median,
            confidence: self.calculate_confidence(&criteria),
            reasoning: self.generate_reasoning(&criteria),
        }
    }

    fn calculate_confidence(&self, criteria: &[usize]) -> Float {
        let max_val = *criteria.iter().max().unwrap_or(&1);
        let min_val = *criteria.iter().min().unwrap_or(&1);

        if max_val == min_val {
            1.0 // Perfect agreement
        } else {
            1.0 - (max_val - min_val) as Float / self.total_components as Float
        }
    }

    fn generate_reasoning(&self, _criteria: &[usize]) -> String {
        format!(
            "Kaiser criterion suggests {} components, elbow method suggests {}, \
             and 95% variance threshold suggests {} components.",
            self.kaiser_criterion, self.elbow_method, self.variance_95_percent
        )
    }
}

/// Component recommendation result
#[derive(Debug, Clone)]
pub struct ComponentRecommendation {
    /// Recommended number of components
    pub recommended_components: usize,
    /// Confidence in the recommendation (0.0 to 1.0)
    pub confidence: Float,
    /// Reasoning behind the recommendation
    pub reasoning: String,
}

/// Visualization utilities for specific decomposition methods
pub struct DecompositionVisualizer;

impl DecompositionVisualizer {
    /// Create visualization for PCA results
    pub fn pca_visualization(
        components: &Array2<Float>,
        transformed_data: &Array2<Float>,
        explained_variance: &Array1<Float>,
    ) -> ComponentVisualization {
        ComponentVisualization::new(
            components.t().to_owned(), // Transpose to get features x components
            transformed_data.clone(),
            explained_variance.clone(),
        )
    }

    /// Create visualization for ICA results
    pub fn ica_visualization(
        mixing_matrix: &Array2<Float>,
        sources: &Array2<Float>,
    ) -> ComponentVisualization {
        // For ICA, use unit variance for all components
        let n_components = mixing_matrix.ncols();
        let explained_variance = Array1::ones(n_components);

        ComponentVisualization::new(mixing_matrix.clone(), sources.clone(), explained_variance)
    }

    /// Create visualization for NMF results
    pub fn nmf_visualization(
        components: &Array2<Float>,
        transformed_data: &Array2<Float>,
    ) -> ComponentVisualization {
        // For NMF, calculate pseudo-variance from component norms
        let explained_variance = components
            .axis_iter(Axis(0))
            .map(|comp| comp.dot(&comp))
            .collect::<Array1<Float>>();

        ComponentVisualization::new(
            components.t().to_owned(),
            transformed_data.clone(),
            explained_variance,
        )
    }

    /// Create visualization for Factor Analysis results
    pub fn factor_analysis_visualization(
        loadings: &Array2<Float>,
        factors: &Array2<Float>,
        explained_variance: &Array1<Float>,
    ) -> ComponentVisualization {
        ComponentVisualization::new(
            loadings.clone(),
            factors.clone(),
            explained_variance.clone(),
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_component_visualization_creation() {
        let loadings = Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.6, 0.5, 0.1, 0.9]).unwrap();
        let scores =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);

        assert_eq!(viz.explained_variance_ratio.len(), 2);
        assert_abs_diff_eq!(viz.explained_variance_ratio[0], 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(viz.explained_variance_ratio[1], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_loading_plot_data() {
        let loadings = Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.6, 0.5, 0.1, 0.9]).unwrap();
        let scores =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);
        let loading_plot = viz.loading_plot_data(0, 1).unwrap();

        assert_eq!(loading_plot.x_loadings.len(), 3);
        assert_eq!(loading_plot.y_loadings.len(), 3);
        assert_eq!(loading_plot.component1, 0);
        assert_eq!(loading_plot.component2, 1);

        let (x_label, y_label) = loading_plot.axis_labels();
        assert!(x_label.contains("PC1"));
        assert!(y_label.contains("PC2"));
    }

    #[test]
    fn test_feature_importance() {
        let loadings = Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.6, 0.5, 0.1, 0.9]).unwrap();
        let scores =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);
        let importance = viz.feature_importance();

        assert_eq!(importance.dim(), (3, 2));
        // Check that importance is always non-negative
        for &val in importance.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_top_features_per_component() {
        let loadings = Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.6, 0.5, 0.1, 0.9]).unwrap();
        let scores =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);
        let top_features = viz.top_features_per_component(2);

        assert_eq!(top_features.len(), 2); // 2 components
        assert_eq!(top_features[0].len(), 2); // Top 2 features for component 0
        assert_eq!(top_features[1].len(), 2); // Top 2 features for component 1
    }

    #[test]
    fn test_component_contributions() {
        let loadings = Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.6, 0.5, 0.1, 0.9]).unwrap();
        let scores =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);
        let contributions = viz.component_contributions();

        assert_eq!(contributions.cumulative_variance.len(), 2);
        assert!(contributions.cumulative_variance[0] <= contributions.cumulative_variance[1]);
        assert_abs_diff_eq!(contributions.total_variance_explained, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_optimal_components_analysis() {
        let loadings = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.8, 0.2, 0.1, 0.05, 0.02, 0.6, 0.5, 0.3, 0.1, 0.01, 0.1, 0.9, 0.2, 0.05, 0.01,
                0.05, 0.1, 0.8, 0.4, 0.1, 0.02, 0.01, 0.1, 0.7, 0.6,
            ],
        )
        .unwrap();
        let scores = Array2::zeros((10, 5));
        let explained_variance = Array1::from_vec(vec![3.0, 2.0, 1.5, 0.8, 0.2]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);
        let analysis = viz.optimal_components();

        assert!(analysis.kaiser_criterion <= 5);
        assert!(analysis.elbow_method <= 5);
        assert!(analysis.variance_95_percent <= 5);
        assert_eq!(analysis.total_components, 5);

        let recommendation = analysis.recommendation();
        assert!(recommendation.recommended_components <= 5);
        assert!(recommendation.confidence >= 0.0 && recommendation.confidence <= 1.0);
        assert!(!recommendation.reasoning.is_empty());
    }

    #[test]
    fn test_decomposition_visualizer() {
        let components =
            Array2::from_shape_vec((2, 3), vec![0.8, 0.6, 0.1, 0.2, 0.5, 0.9]).unwrap();
        let transformed_data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = DecompositionVisualizer::pca_visualization(
            &components,
            &transformed_data,
            &explained_variance,
        );

        assert_eq!(viz.loadings.dim(), (3, 2)); // Transposed
        assert_eq!(viz.scores.dim(), (4, 2));
        assert_eq!(viz.explained_variance.len(), 2);
    }

    #[test]
    fn test_biplot_data() {
        let loadings = Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.6, 0.5, 0.1, 0.9]).unwrap();
        let scores =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]).unwrap();
        let explained_variance = Array1::from_vec(vec![2.0, 1.0]);

        let viz = ComponentVisualization::new(loadings, scores, explained_variance);
        let biplot = viz.biplot_data(0, 1).unwrap();

        assert_eq!(biplot.x_scores.len(), 4);
        assert_eq!(biplot.y_scores.len(), 4);

        let sample_coords = biplot.sample_coordinates();
        assert_eq!(sample_coords.len(), 4);

        let scaled_loadings = biplot.scaled_loadings(1.5);
        assert_eq!(scaled_loadings.len(), 3);
    }
}
