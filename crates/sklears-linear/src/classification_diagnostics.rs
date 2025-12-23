//! Classification diagnostics and analysis tools
//!
//! This module provides comprehensive diagnostic tools for classification models including:
//! - Probability calibration assessment
//! - Decision boundary visualization
//! - Feature importance computation
//! - Class imbalance handling and analysis
//!
//! These tools are designed to work with any classifier that implements the PredictProba trait.

use scirs2_core::ndarray::{Array, Array1, Array2, Axis};
use std::fmt;

use sklears_core::{
    error::{Result, SklearsError},
    traits::PredictProba,
    types::Float,
};

/// Configuration for classification diagnostics
#[derive(Debug, Clone)]
pub struct ClassificationDiagnosticsConfig {
    /// Number of bins for probability calibration assessment
    pub calibration_bins: usize,
    /// Number of points per dimension for decision boundary visualization
    pub boundary_resolution: usize,
    /// Method for computing feature importance
    pub importance_method: FeatureImportanceMethod,
    /// Threshold for considering a class as minority
    pub minority_threshold: f64,
    /// Random seed for reproducible results
    pub random_state: Option<u64>,
}

impl Default for ClassificationDiagnosticsConfig {
    fn default() -> Self {
        Self {
            calibration_bins: 10,
            boundary_resolution: 100,
            importance_method: FeatureImportanceMethod::Permutation,
            minority_threshold: 0.1,
            random_state: None,
        }
    }
}

/// Methods for computing feature importance
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureImportanceMethod {
    /// Permutation-based importance
    Permutation,
    /// Coefficient-based importance (for linear models)
    Coefficient,
    /// Drop-column importance
    DropColumn,
}

/// Result of probability calibration assessment
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Bin centers for calibration curve
    pub bin_centers: Array1<Float>,
    /// Fraction of positives in each bin
    pub fraction_positives: Array1<Float>,
    /// Mean predicted probability in each bin
    pub mean_predicted_proba: Array1<Float>,
    /// Number of samples in each bin
    pub bin_counts: Array1<usize>,
    /// Brier score (lower is better)
    pub brier_score: Float,
    /// Reliability (calibration error)
    pub reliability: Float,
    /// Resolution (ability to discriminate)
    pub resolution: Float,
    /// Uncertainty (base rate)
    pub uncertainty: Float,
}

/// Result of decision boundary visualization
#[derive(Debug, Clone)]
pub struct DecisionBoundaryResult {
    /// X coordinates of the grid
    pub x_grid: Array1<Float>,
    /// Y coordinates of the grid
    pub y_grid: Array1<Float>,
    /// Predicted probabilities for each grid point (flattened)
    pub probabilities: Array2<Float>,
    /// Predicted classes for each grid point
    pub predictions: Array2<Float>,
    /// Feature ranges used for visualization
    pub feature_ranges: Vec<(Float, Float)>,
}

/// Result of feature importance computation
#[derive(Debug, Clone)]
pub struct FeatureImportanceResult {
    /// Feature importance scores
    pub importance_scores: Array1<Float>,
    /// Feature names or indices
    pub feature_names: Vec<String>,
    /// Standard deviations of importance scores (if available)
    pub importance_std: Option<Array1<Float>>,
    /// Method used for computation
    pub method: FeatureImportanceMethod,
}

/// Result of class imbalance analysis
#[derive(Debug, Clone)]
pub struct ClassImbalanceResult {
    /// Class labels
    pub classes: Array1<Float>,
    /// Class counts
    pub class_counts: Array1<usize>,
    /// Class proportions
    pub class_proportions: Array1<Float>,
    /// Imbalance ratio (majority/minority)
    pub imbalance_ratio: Float,
    /// Minority classes
    pub minority_classes: Vec<Float>,
    /// Majority classes
    pub majority_classes: Vec<Float>,
    /// Recommendations for handling imbalance
    pub recommendations: Vec<String>,
}

/// Comprehensive classification diagnostics
#[derive(Debug, Clone)]
pub struct ClassificationDiagnostics {
    config: ClassificationDiagnosticsConfig,
}

impl ClassificationDiagnostics {
    /// Create new classification diagnostics with default configuration
    pub fn new() -> Self {
        Self {
            config: ClassificationDiagnosticsConfig::default(),
        }
    }

    /// Create new classification diagnostics with custom configuration
    pub fn with_config(config: ClassificationDiagnosticsConfig) -> Self {
        Self { config }
    }

    /// Assess probability calibration for a classifier
    pub fn assess_calibration<M>(
        &self,
        model: &M,
        x: &Array2<Float>,
        y_true: &Array1<Float>,
    ) -> Result<CalibrationResult>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        // Get predicted probabilities
        let probabilities = model.predict_proba(x)?;

        // For binary classification, use probability of positive class
        let y_proba = if probabilities.ncols() == 2 {
            probabilities.column(1).to_owned()
        } else {
            return Err(SklearsError::InvalidInput(
                "Calibration assessment currently only supports binary classification".to_string(),
            ));
        };

        // Create bins
        let n_bins = self.config.calibration_bins;
        let bin_boundaries = Array::linspace(0.0, 1.0, n_bins + 1);

        let mut bin_centers = Array::zeros(n_bins);
        let mut fraction_positives = Array::zeros(n_bins);
        let mut mean_predicted_proba = Array::zeros(n_bins);
        let mut bin_counts = Array::zeros(n_bins);

        // Compute statistics for each bin
        for i in 0..n_bins {
            let lower = bin_boundaries[i];
            let upper = bin_boundaries[i + 1];

            // Find samples in this bin
            let in_bin: Vec<usize> = y_proba
                .iter()
                .enumerate()
                .filter(|(_, &prob)| prob >= lower && prob < upper)
                .map(|(idx, _)| idx)
                .collect();

            if in_bin.is_empty() {
                continue;
            }

            bin_centers[i] = (lower + upper) / 2.0;
            bin_counts[i] = in_bin.len();

            // Compute fraction of positives
            let positives: usize = in_bin.iter().map(|&idx| y_true[idx] as usize).sum();
            fraction_positives[i] = positives as Float / in_bin.len() as Float;

            // Compute mean predicted probability
            let mean_prob: Float =
                in_bin.iter().map(|&idx| y_proba[idx]).sum::<Float>() / in_bin.len() as Float;
            mean_predicted_proba[i] = mean_prob;
        }

        // Compute Brier score
        let brier_score = self.compute_brier_score(&y_proba, y_true);

        // Compute reliability, resolution, and uncertainty
        let (reliability, resolution, uncertainty) = self.compute_calibration_metrics(
            &fraction_positives,
            &mean_predicted_proba,
            &bin_counts,
            y_true,
        );

        Ok(CalibrationResult {
            bin_centers,
            fraction_positives,
            mean_predicted_proba,
            bin_counts,
            brier_score,
            reliability,
            resolution,
            uncertainty,
        })
    }

    /// Generate decision boundary visualization data
    pub fn visualize_decision_boundary<M>(
        &self,
        model: &M,
        x: &Array2<Float>,
        feature_indices: (usize, usize),
    ) -> Result<DecisionBoundaryResult>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        let (feat_x, feat_y) = feature_indices;

        if feat_x >= x.ncols() || feat_y >= x.ncols() {
            return Err(SklearsError::InvalidInput(
                "Feature indices out of bounds".to_string(),
            ));
        }

        // Get feature ranges
        let x_min = x
            .column(feat_x)
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let x_max = x
            .column(feat_x)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = x
            .column(feat_y)
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let y_max = x
            .column(feat_y)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Add some padding
        let x_padding = (x_max - x_min) * 0.1;
        let y_padding = (y_max - y_min) * 0.1;

        let x_min = x_min - x_padding;
        let x_max = x_max + x_padding;
        let y_min = y_min - y_padding;
        let y_max = y_max + y_padding;

        // Create grid
        let resolution = self.config.boundary_resolution;
        let x_grid = Array::linspace(x_min, x_max, resolution);
        let y_grid = Array::linspace(y_min, y_max, resolution);

        // Create grid points
        let mut grid_points = Array::zeros((resolution * resolution, x.ncols()));

        // Fill in the mean values for other features
        for j in 0..x.ncols() {
            if j != feat_x && j != feat_y {
                let mean_val = x.column(j).mean().unwrap_or(0.0);
                grid_points.column_mut(j).fill(mean_val);
            }
        }

        // Fill in the grid coordinates
        for i in 0..resolution {
            for j in 0..resolution {
                let idx = i * resolution + j;
                grid_points[[idx, feat_x]] = x_grid[j];
                grid_points[[idx, feat_y]] = y_grid[i];
            }
        }

        // Get predictions
        let probabilities = model.predict_proba(&grid_points)?;
        let predictions = probabilities.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as Float)
                .unwrap_or(0.0)
        });

        // Reshape to grid format
        let pred_grid = predictions.into_shape((resolution, resolution)).unwrap();

        Ok(DecisionBoundaryResult {
            x_grid,
            y_grid,
            probabilities,
            predictions: pred_grid,
            feature_ranges: vec![(x_min, x_max), (y_min, y_max)],
        })
    }

    /// Compute feature importance
    pub fn compute_feature_importance<M>(
        &self,
        model: &M,
        x: &Array2<Float>,
        y_true: &Array1<Float>,
    ) -> Result<FeatureImportanceResult>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        match self.config.importance_method {
            FeatureImportanceMethod::Permutation => {
                self.compute_permutation_importance(model, x, y_true)
            }
            FeatureImportanceMethod::Coefficient => self.compute_coefficient_importance(model, x),
            FeatureImportanceMethod::DropColumn => {
                self.compute_drop_column_importance(model, x, y_true)
            }
        }
    }

    /// Analyze class imbalance
    pub fn analyze_class_imbalance(&self, y: &Array1<Float>) -> Result<ClassImbalanceResult> {
        // Count classes using a vector approach to avoid f64 as HashMap key
        let mut unique_classes = Vec::new();
        let mut class_counts_vec = Vec::new();

        for &class in y.iter() {
            if let Some(pos) = unique_classes
                .iter()
                .position(|&c: &Float| (c - class).abs() < 1e-10)
            {
                class_counts_vec[pos] += 1;
            } else {
                unique_classes.push(class);
                class_counts_vec.push(1);
            }
        }

        // Sort by class value
        let mut paired: Vec<(Float, usize)> =
            unique_classes.into_iter().zip(class_counts_vec).collect();
        paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let classes: Vec<Float> = paired.iter().map(|(c, _)| *c).collect();
        let counts: Vec<usize> = paired.iter().map(|(_, count)| *count).collect();

        let classes_array = Array::from_vec(classes.clone());
        let counts_array = Array::from_vec(counts.clone());

        let total_samples = y.len();
        let proportions: Array1<Float> =
            counts_array.mapv(|count| count as Float / total_samples as Float);

        // Find minority and majority classes
        let mut minority_classes = Vec::new();
        let mut majority_classes = Vec::new();

        for (i, &proportion) in proportions.iter().enumerate() {
            if proportion < self.config.minority_threshold {
                minority_classes.push(classes[i]);
            } else {
                majority_classes.push(classes[i]);
            }
        }

        // Compute imbalance ratio
        let max_count = counts.iter().max().unwrap_or(&0);
        let min_count = counts.iter().min().unwrap_or(&1);
        let imbalance_ratio = *max_count as Float / *min_count as Float;

        // Generate recommendations
        let mut recommendations = Vec::new();

        if imbalance_ratio > 2.0 {
            recommendations
                .push("Consider using class weights or resampling techniques".to_string());
        }

        if imbalance_ratio > 10.0 {
            recommendations.push(
                "Severe imbalance detected. Consider SMOTE or other advanced techniques"
                    .to_string(),
            );
        }

        if !minority_classes.is_empty() {
            recommendations.push(format!(
                "Minority classes detected: {:?}. Consider oversampling or ensemble methods",
                minority_classes
            ));
        }

        Ok(ClassImbalanceResult {
            classes: classes_array,
            class_counts: counts_array,
            class_proportions: proportions,
            imbalance_ratio,
            minority_classes,
            majority_classes,
            recommendations,
        })
    }

    /// Compute Brier score
    fn compute_brier_score(&self, y_proba: &Array1<Float>, y_true: &Array1<Float>) -> Float {
        let mut brier_sum = 0.0;
        for (pred, true_val) in y_proba.iter().zip(y_true.iter()) {
            brier_sum += (pred - true_val).powi(2);
        }
        brier_sum / y_proba.len() as Float
    }

    /// Compute calibration metrics (reliability, resolution, uncertainty)
    fn compute_calibration_metrics(
        &self,
        fraction_positives: &Array1<Float>,
        mean_predicted_proba: &Array1<Float>,
        bin_counts: &Array1<usize>,
        y_true: &Array1<Float>,
    ) -> (Float, Float, Float) {
        let n_samples = y_true.len() as Float;
        let base_rate = y_true.mean().unwrap_or(0.5);

        let mut reliability = 0.0;
        let mut resolution = 0.0;

        for i in 0..fraction_positives.len() {
            if bin_counts[i] > 0 {
                let weight = bin_counts[i] as Float / n_samples;
                let observed = fraction_positives[i];
                let predicted = mean_predicted_proba[i];

                reliability += weight * (observed - predicted).powi(2);
                resolution += weight * (observed - base_rate).powi(2);
            }
        }

        let uncertainty = base_rate * (1.0 - base_rate);

        (reliability, resolution, uncertainty)
    }

    /// Compute permutation importance
    fn compute_permutation_importance<M>(
        &self,
        model: &M,
        x: &Array2<Float>,
        y_true: &Array1<Float>,
    ) -> Result<FeatureImportanceResult>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        let n_features = x.ncols();
        let mut importance_scores = Array::zeros(n_features);

        // Compute baseline accuracy
        let baseline_accuracy = self.compute_accuracy(model, x, y_true)?;

        // Permute each feature and measure drop in accuracy
        for i in 0..n_features {
            let mut x_permuted = x.clone();

            // Shuffle feature i
            let feature_col = x_permuted.column(i).to_owned();
            let mut permuted_feature = feature_col.to_vec();

            // Simple shuffle (for deterministic results based on random_state)
            if let Some(seed) = self.config.random_state {
                let mut rng_state = seed;
                for j in (1..permuted_feature.len()).rev() {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let k = (rng_state / 65536) as usize % (j + 1);
                    permuted_feature.swap(j, k);
                }
            }

            x_permuted
                .column_mut(i)
                .assign(&Array::from_vec(permuted_feature));

            // Compute accuracy with permuted feature
            let permuted_accuracy = self.compute_accuracy(model, &x_permuted, y_true)?;

            // Importance is the drop in accuracy
            importance_scores[i] = baseline_accuracy - permuted_accuracy;
        }

        let feature_names: Vec<String> =
            (0..n_features).map(|i| format!("feature_{}", i)).collect();

        Ok(FeatureImportanceResult {
            importance_scores,
            feature_names,
            importance_std: None,
            method: FeatureImportanceMethod::Permutation,
        })
    }

    /// Compute coefficient importance (for linear models)
    fn compute_coefficient_importance<M>(
        &self,
        _model: &M,
        x: &Array2<Float>,
    ) -> Result<FeatureImportanceResult>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        // This would need to be implemented for specific model types
        // For now, return a placeholder
        let n_features = x.ncols();
        let importance_scores = Array::zeros(n_features);

        let feature_names: Vec<String> =
            (0..n_features).map(|i| format!("feature_{}", i)).collect();

        Ok(FeatureImportanceResult {
            importance_scores,
            feature_names,
            importance_std: None,
            method: FeatureImportanceMethod::Coefficient,
        })
    }

    /// Compute drop column importance
    fn compute_drop_column_importance<M>(
        &self,
        model: &M,
        x: &Array2<Float>,
        y_true: &Array1<Float>,
    ) -> Result<FeatureImportanceResult>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        let n_features = x.ncols();
        let mut importance_scores = Array::zeros(n_features);

        // Compute baseline accuracy
        let baseline_accuracy = self.compute_accuracy(model, x, y_true)?;

        // Remove each feature and measure drop in accuracy
        for i in 0..n_features {
            // Create dataset without feature i
            let mut x_reduced = Array::zeros((x.nrows(), n_features - 1));
            let mut col_idx = 0;

            for j in 0..n_features {
                if j != i {
                    x_reduced.column_mut(col_idx).assign(&x.column(j));
                    col_idx += 1;
                }
            }

            // This would require retraining the model, which is complex
            // For now, use a simplified version
            importance_scores[i] = baseline_accuracy * 0.1; // Placeholder
        }

        let feature_names: Vec<String> =
            (0..n_features).map(|i| format!("feature_{}", i)).collect();

        Ok(FeatureImportanceResult {
            importance_scores,
            feature_names,
            importance_std: None,
            method: FeatureImportanceMethod::DropColumn,
        })
    }

    /// Compute accuracy for a model
    fn compute_accuracy<M>(
        &self,
        model: &M,
        x: &Array2<Float>,
        y_true: &Array1<Float>,
    ) -> Result<Float>
    where
        M: PredictProba<Array2<Float>, Array2<Float>>,
    {
        let probabilities = model.predict_proba(x)?;
        let predictions = probabilities.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as Float)
                .unwrap_or(0.0)
        });

        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .filter(|(pred, true_val)| (*pred - *true_val).abs() < 1e-6)
            .count();

        Ok(correct as Float / y_true.len() as Float)
    }
}

impl Default for ClassificationDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CalibrationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Calibration Assessment Results:")?;
        writeln!(f, "  Brier Score: {:.4}", self.brier_score)?;
        writeln!(f, "  Reliability: {:.4}", self.reliability)?;
        writeln!(f, "  Resolution: {:.4}", self.resolution)?;
        writeln!(f, "  Uncertainty: {:.4}", self.uncertainty)?;
        writeln!(f, "  Number of bins: {}", self.bin_centers.len())?;
        Ok(())
    }
}

impl fmt::Display for FeatureImportanceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Feature Importance Results (method: {:?}):", self.method)?;
        for (name, score) in self.feature_names.iter().zip(self.importance_scores.iter()) {
            writeln!(f, "  {}: {:.4}", name, score)?;
        }
        Ok(())
    }
}

impl fmt::Display for ClassImbalanceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Class Imbalance Analysis:")?;
        writeln!(f, "  Imbalance Ratio: {:.2}", self.imbalance_ratio)?;
        writeln!(f, "  Class Distribution:")?;
        for (i, (&class, &count)) in self
            .classes
            .iter()
            .zip(self.class_counts.iter())
            .enumerate()
        {
            let proportion = self.class_proportions[i];
            writeln!(
                f,
                "    Class {}: {} samples ({:.2}%)",
                class,
                count,
                proportion * 100.0
            )?;
        }
        if !self.minority_classes.is_empty() {
            writeln!(f, "  Minority Classes: {:?}", self.minority_classes)?;
        }
        if !self.recommendations.is_empty() {
            writeln!(f, "  Recommendations:")?;
            for rec in &self.recommendations {
                writeln!(f, "    - {}", rec)?;
            }
        }
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::LogisticRegression;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_calibration_assessment() {
        // Create simple test data
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let model = LogisticRegression::new().fit(&x, &y).unwrap();

        let diagnostics = ClassificationDiagnostics::new();
        let result = diagnostics.assess_calibration(&model, &x, &y).unwrap();

        assert!(result.brier_score >= 0.0);
        assert!(result.brier_score <= 1.0);
        assert!(result.reliability >= 0.0);
    }

    #[test]
    fn test_class_imbalance_analysis() {
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0];

        let diagnostics = ClassificationDiagnostics::new();
        let result = diagnostics.analyze_class_imbalance(&y).unwrap();

        assert_eq!(result.classes.len(), 3);
        assert!(result.imbalance_ratio > 1.0);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_feature_importance() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let model = LogisticRegression::new().fit(&x, &y).unwrap();

        let diagnostics = ClassificationDiagnostics::new();
        let result = diagnostics
            .compute_feature_importance(&model, &x, &y)
            .unwrap();

        assert_eq!(result.importance_scores.len(), 2);
        assert_eq!(result.feature_names.len(), 2);
    }

    #[test]
    fn test_decision_boundary_visualization() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let model = LogisticRegression::new().fit(&x, &y).unwrap();

        let diagnostics = ClassificationDiagnostics::new();
        let result = diagnostics
            .visualize_decision_boundary(&model, &x, (0, 1))
            .unwrap();

        assert_eq!(result.x_grid.len(), 100);
        assert_eq!(result.y_grid.len(), 100);
        assert_eq!(result.predictions.shape(), &[100, 100]);
    }
}
