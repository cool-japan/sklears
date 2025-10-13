//! Feature set visualization utilities for feature selection analysis
//!
//! This module provides comprehensive text-based visualization capabilities for feature selection
//! analysis, including stability plots, importance visualizations, and selection frequency analysis.
//! All implementations follow the SciRS2 policy.

use scirs2_core::ndarray::ArrayView2;
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;

impl From<VisualizationError> for SklearsError {
    fn from(err: VisualizationError) -> Self {
        SklearsError::FitError(format!("Visualization error: {}", err))
    }
}
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VisualizationError {
    #[error("Empty data provided for visualization")]
    EmptyData,
    #[error("Invalid feature indices")]
    InvalidFeatureIndices,
    #[error("Mismatched data dimensions")]
    DimensionMismatch,
    #[error("Invalid visualization parameters")]
    InvalidParameters,
}

/// Text-based feature importance visualization
#[derive(Debug, Clone)]
pub struct FeatureImportancePlots;

impl FeatureImportancePlots {
    /// Create horizontal bar chart for feature importance
    pub fn horizontal_bar_chart(
        feature_indices: &[usize],
        importance_scores: &[f64],
        feature_names: Option<&[String]>,
        max_width: usize,
        title: &str,
    ) -> Result<String> {
        if feature_indices.len() != importance_scores.len() {
            return Err(VisualizationError::DimensionMismatch.into());
        }

        if feature_indices.is_empty() {
            return Err(VisualizationError::EmptyData.into());
        }

        let mut chart = String::new();

        // Title
        chart.push_str(&format!("=== {} ===\n\n", title));

        // Find max importance for scaling
        let max_importance = importance_scores
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x.abs()));

        if max_importance <= 0.0 {
            chart.push_str("All features have zero importance\n");
            return Ok(chart);
        }

        // Create sorted indices by importance
        let mut sorted_indices: Vec<usize> = (0..feature_indices.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            importance_scores[b]
                .abs()
                .partial_cmp(&importance_scores[a].abs())
                .unwrap()
        });

        // Draw bars
        for &idx in &sorted_indices {
            let feature_idx = feature_indices[idx];
            let importance = importance_scores[idx];
            let normalized_importance = importance / max_importance;

            let bar_width = ((normalized_importance.abs() * max_width as f64) as usize).max(1);
            let bar_char = if importance >= 0.0 { '█' } else { '▓' };

            let feature_name = if let Some(names) = feature_names {
                if idx < names.len() {
                    names[idx].clone()
                } else {
                    format!("Feature_{}", feature_idx)
                }
            } else {
                format!("Feature_{}", feature_idx)
            };

            let bar = bar_char.to_string().repeat(bar_width);
            chart.push_str(&format!(
                "{:>15} |{:<width$} {:>8.4}\n",
                feature_name,
                bar,
                importance,
                width = max_width + 2
            ));
        }

        // Legend
        chart.push_str(&format!("\n{:>15} |{}\n", "", "-".repeat(max_width + 2)));
        chart.push_str(&format!(
            "{:>15} |{:>width$}\n",
            "",
            max_importance,
            width = max_width + 10
        ));
        chart.push_str("Legend: █ = positive importance, ▓ = negative importance\n");

        Ok(chart)
    }

    /// Create vertical bar chart for feature importance
    pub fn vertical_bar_chart(
        feature_indices: &[usize],
        importance_scores: &[f64],
        feature_names: Option<&[String]>,
        max_height: usize,
        title: &str,
    ) -> Result<String> {
        if feature_indices.len() != importance_scores.len() {
            return Err(VisualizationError::DimensionMismatch.into());
        }

        if feature_indices.is_empty() {
            return Err(VisualizationError::EmptyData.into());
        }

        let mut chart = String::new();

        // Title
        chart.push_str(&format!("=== {} ===\n\n", title));

        // Find max importance for scaling
        let max_importance = importance_scores
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x.abs()));

        if max_importance <= 0.0 {
            chart.push_str("All features have zero importance\n");
            return Ok(chart);
        }

        // Take top features to display (limit for readability)
        let max_features = 15.min(feature_indices.len());
        let mut sorted_indices: Vec<usize> = (0..feature_indices.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            importance_scores[b]
                .abs()
                .partial_cmp(&importance_scores[a].abs())
                .unwrap()
        });
        sorted_indices.truncate(max_features);

        // Create bars from top to bottom
        for row in (0..max_height).rev() {
            let threshold = (row + 1) as f64 / max_height as f64;

            for &idx in &sorted_indices {
                let importance = importance_scores[idx];
                let normalized_importance = importance.abs() / max_importance;

                if normalized_importance >= threshold {
                    let bar_char = if importance >= 0.0 { '█' } else { '▓' };
                    chart.push_str(&format!("{} ", bar_char));
                } else {
                    chart.push_str("  ");
                }
            }
            chart.push('\n');
        }

        // Draw base line
        chart.push_str(&"-".repeat(sorted_indices.len() * 2));
        chart.push('\n');

        // Feature labels
        for &idx in &sorted_indices {
            let feature_idx = feature_indices[idx];
            let label = if let Some(names) = feature_names {
                if idx < names.len() {
                    format!("{}", names[idx].chars().next().unwrap_or('F'))
                } else {
                    format!("{}", feature_idx % 10)
                }
            } else {
                format!("{}", feature_idx % 10)
            };
            chart.push_str(&format!("{} ", label));
        }
        chart.push('\n');

        // Feature importance values
        for &idx in &sorted_indices {
            let importance = importance_scores[idx];
            chart.push_str(&format!("{:.1} ", importance));
        }
        chart.push('\n');

        chart.push_str("Legend: █ = positive importance, ▓ = negative importance\n");

        Ok(chart)
    }
}

/// Stability visualization for feature selection consistency
#[derive(Debug, Clone)]
pub struct StabilityPlots;

impl StabilityPlots {
    /// Create stability frequency plot
    pub fn stability_frequency_plot(
        feature_selections: &[Vec<usize>],
        total_features: usize,
        feature_names: Option<&[String]>,
        title: &str,
    ) -> Result<String> {
        if feature_selections.is_empty() {
            return Err(VisualizationError::EmptyData.into());
        }

        let mut plot = String::new();

        // Title
        plot.push_str(&format!("=== {} ===\n\n", title));

        // Count feature frequencies
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();
        let total_selections = feature_selections.len();

        for selection in feature_selections {
            for &feature_idx in selection {
                *feature_counts.entry(feature_idx).or_insert(0) += 1;
            }
        }

        if feature_counts.is_empty() {
            plot.push_str("No features were selected in any iteration\n");
            return Ok(plot);
        }

        // Sort features by frequency
        let mut sorted_features: Vec<(usize, usize)> = feature_counts.into_iter().collect();
        sorted_features.sort_by(|a, b| b.1.cmp(&a.1));

        // Display top features (limit for readability)
        let max_features = 20.min(sorted_features.len());
        let max_width = 50;

        plot.push_str(&format!(
            "Selection Frequency (out of {} iterations):\n\n",
            total_selections
        ));

        for (feature_idx, count) in sorted_features.iter().take(max_features) {
            let frequency = *count as f64 / total_selections as f64;
            let bar_width = (frequency * max_width as f64) as usize;
            let bar = "█".repeat(bar_width);

            let feature_name = if let Some(names) = feature_names {
                if *feature_idx < names.len() {
                    names[*feature_idx].clone()
                } else {
                    format!("Feature_{}", feature_idx)
                }
            } else {
                format!("Feature_{}", feature_idx)
            };

            plot.push_str(&format!(
                "{:>15} |{:<width$} {:>3}/{:<3} ({:>5.1}%)\n",
                feature_name,
                bar,
                count,
                total_selections,
                frequency * 100.0,
                width = max_width + 2
            ));
        }

        if sorted_features.len() > max_features {
            plot.push_str(&format!(
                "... and {} more features\n",
                sorted_features.len() - max_features
            ));
        }

        // Summary statistics
        let high_stability_count = sorted_features
            .iter()
            .filter(|(_, count)| *count as f64 / total_selections as f64 >= 0.8)
            .count();

        plot.push_str(&"\nStability Summary:\n".to_string());
        plot.push_str(&format!(
            "  High stability features (≥80%): {}\n",
            high_stability_count
        ));
        plot.push_str(&format!(
            "  Total unique features selected: {}\n",
            sorted_features.len()
        ));
        plot.push_str(&format!(
            "  Average features per selection: {:.1}\n",
            feature_selections.iter().map(|s| s.len()).sum::<usize>() as f64
                / total_selections as f64
        ));

        Ok(plot)
    }

    /// Create stability heatmap showing feature co-occurrence
    pub fn feature_cooccurrence_heatmap(
        feature_selections: &[Vec<usize>],
        top_n_features: usize,
        feature_names: Option<&[String]>,
        title: &str,
    ) -> Result<String> {
        if feature_selections.is_empty() {
            return Err(VisualizationError::EmptyData.into());
        }

        let mut heatmap = String::new();

        // Title
        heatmap.push_str(&format!("=== {} ===\n\n", title));

        // Get most frequently selected features
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();
        for selection in feature_selections {
            for &feature_idx in selection {
                *feature_counts.entry(feature_idx).or_insert(0) += 1;
            }
        }

        let mut sorted_features: Vec<(usize, usize)> = feature_counts.into_iter().collect();
        sorted_features.sort_by(|a, b| b.1.cmp(&a.1));

        let n_features = top_n_features.min(sorted_features.len()).min(10); // Limit for readability
        let top_features: Vec<usize> = sorted_features
            .iter()
            .take(n_features)
            .map(|(idx, _)| *idx)
            .collect();

        if top_features.is_empty() {
            heatmap.push_str("No features to display\n");
            return Ok(heatmap);
        }

        // Compute co-occurrence matrix
        let mut cooccurrence_matrix = vec![vec![0; n_features]; n_features];

        for selection in feature_selections {
            let selection_set: std::collections::HashSet<usize> =
                selection.iter().cloned().collect();

            for i in 0..n_features {
                for j in 0..n_features {
                    if selection_set.contains(&top_features[i])
                        && selection_set.contains(&top_features[j])
                    {
                        cooccurrence_matrix[i][j] += 1;
                    }
                }
            }
        }

        // Create header
        heatmap.push_str("       ");
        for j in 0..n_features {
            heatmap.push_str(&format!("{:>4}", j));
        }
        heatmap.push('\n');

        // Create heatmap rows
        for i in 0..n_features {
            let feature_name = if let Some(names) = feature_names {
                if top_features[i] < names.len() {
                    format!(
                        "{:>6}",
                        names[top_features[i]].chars().take(6).collect::<String>()
                    )
                } else {
                    format!("F_{:>3}", top_features[i])
                }
            } else {
                format!("F_{:>3}", top_features[i])
            };

            heatmap.push_str(&format!("{} ", feature_name));

            for j in 0..n_features {
                let cooccurrence = cooccurrence_matrix[i][j];
                let intensity = cooccurrence as f64 / feature_selections.len() as f64;
                let symbol = match intensity {
                    x if x >= 0.9 => "██",
                    x if x >= 0.7 => "▓▓",
                    x if x >= 0.5 => "▒▒",
                    x if x >= 0.3 => "░░",
                    x if x >= 0.1 => "··",
                    _ => "  ",
                };
                heatmap.push_str(&symbol.to_string());
            }
            heatmap.push('\n');
        }

        // Legend
        heatmap
            .push_str("\nIntensity: ██ ≥90%  ▓▓ ≥70%  ▒▒ ≥50%  ░░ ≥30%  ·· ≥10%  [space] <10%\n");

        // Feature mapping
        heatmap.push_str("\nFeature Mapping:\n");
        for (i, &feature_idx) in top_features.iter().enumerate() {
            let feature_name = if let Some(names) = feature_names {
                if feature_idx < names.len() {
                    &names[feature_idx]
                } else {
                    "Unknown"
                }
            } else {
                "Unknown"
            };
            heatmap.push_str(&format!(
                "  {}: Feature_{} ({})\n",
                i, feature_idx, feature_name
            ));
        }

        Ok(heatmap)
    }
}

/// Redundancy heatmaps for feature correlation analysis
#[derive(Debug, Clone)]
pub struct RedundancyHeatmaps;

impl RedundancyHeatmaps {
    /// Create correlation heatmap
    pub fn correlation_heatmap(
        correlation_matrix: ArrayView2<f64>,
        feature_indices: &[usize],
        feature_names: Option<&[String]>,
        title: &str,
    ) -> Result<String> {
        let n_features = correlation_matrix.nrows();
        if n_features != correlation_matrix.ncols() {
            return Err(VisualizationError::DimensionMismatch.into());
        }

        if n_features != feature_indices.len() {
            return Err(VisualizationError::DimensionMismatch.into());
        }

        let mut heatmap = String::new();

        // Title
        heatmap.push_str(&format!("=== {} ===\n\n", title));

        // Limit size for readability
        let display_size = n_features.min(12);

        if display_size == 0 {
            heatmap.push_str("No features to display\n");
            return Ok(heatmap);
        }

        // Create header
        heatmap.push_str("       ");
        for j in 0..display_size {
            heatmap.push_str(&format!("{:>4}", j));
        }
        heatmap.push('\n');

        // Create heatmap rows
        for i in 0..display_size {
            let feature_name = if let Some(names) = feature_names {
                if i < names.len() {
                    format!("{:>6}", names[i].chars().take(6).collect::<String>())
                } else {
                    format!("F_{:>3}", feature_indices[i])
                }
            } else {
                format!("F_{:>3}", feature_indices[i])
            };

            heatmap.push_str(&format!("{} ", feature_name));

            for j in 0..display_size {
                let correlation = correlation_matrix[[i, j]].abs();
                let symbol = match correlation {
                    x if x >= 0.9 => "██",
                    x if x >= 0.7 => "▓▓",
                    x if x >= 0.5 => "▒▒",
                    x if x >= 0.3 => "░░",
                    x if x >= 0.1 => "··",
                    _ => "  ",
                };
                heatmap.push_str(&symbol.to_string());
            }
            heatmap.push('\n');
        }

        if n_features > display_size {
            heatmap.push_str(&format!(
                "... and {} more features (truncated for display)\n",
                n_features - display_size
            ));
        }

        // Legend
        heatmap
            .push_str("\nCorrelation: ██ ≥0.9  ▓▓ ≥0.7  ▒▒ ≥0.5  ░░ ≥0.3  ·· ≥0.1  [space] <0.1\n");

        // Feature mapping
        heatmap.push_str("\nFeature Mapping:\n");
        for i in 0..display_size {
            let feature_name = if let Some(names) = feature_names {
                if i < names.len() {
                    &names[i]
                } else {
                    "Unknown"
                }
            } else {
                "Unknown"
            };
            heatmap.push_str(&format!(
                "  {}: Feature_{} ({})\n",
                i, feature_indices[i], feature_name
            ));
        }

        Ok(heatmap)
    }

    /// Create redundancy summary visualization
    pub fn redundancy_summary(
        highly_correlated_pairs: &[(usize, usize, f64)],
        feature_names: Option<&[String]>,
        title: &str,
    ) -> Result<String> {
        let mut summary = String::new();

        // Title
        summary.push_str(&format!("=== {} ===\n\n", title));

        if highly_correlated_pairs.is_empty() {
            summary.push_str("No highly correlated feature pairs found\n");
            return Ok(summary);
        }

        summary.push_str(&format!(
            "Highly Correlated Feature Pairs ({} pairs):\n\n",
            highly_correlated_pairs.len()
        ));

        let max_pairs = 15.min(highly_correlated_pairs.len());

        for (i, &(feat1, feat2, corr)) in highly_correlated_pairs.iter().take(max_pairs).enumerate()
        {
            let name1 = if let Some(names) = feature_names {
                if feat1 < names.len() {
                    &names[feat1]
                } else {
                    "Unknown"
                }
            } else {
                "Unknown"
            };

            let name2 = if let Some(names) = feature_names {
                if feat2 < names.len() {
                    &names[feat2]
                } else {
                    "Unknown"
                }
            } else {
                "Unknown"
            };

            // Create visual correlation strength bar
            let bar_length = (corr.abs() * 20.0) as usize;
            let bar = "█".repeat(bar_length);

            summary.push_str(&format!(
                "{:>2}. Feature_{:>3} ({:<10}) ↔ Feature_{:>3} ({:<10}) |{:<20} {:>6.3}\n",
                i + 1,
                feat1,
                name1.chars().take(10).collect::<String>(),
                feat2,
                name2.chars().take(10).collect::<String>(),
                bar,
                corr
            ));
        }

        if highly_correlated_pairs.len() > max_pairs {
            summary.push_str(&format!(
                "... and {} more pairs\n",
                highly_correlated_pairs.len() - max_pairs
            ));
        }

        // Statistics
        let avg_correlation = highly_correlated_pairs
            .iter()
            .map(|(_, _, corr)| corr.abs())
            .sum::<f64>()
            / highly_correlated_pairs.len() as f64;
        let max_correlation = highly_correlated_pairs
            .iter()
            .map(|(_, _, corr)| corr.abs())
            .fold(0.0, f64::max);

        summary.push_str(&"\nRedundancy Statistics:\n".to_string());
        summary.push_str(&format!("  Average correlation: {:.3}\n", avg_correlation));
        summary.push_str(&format!("  Maximum correlation: {:.3}\n", max_correlation));
        summary.push_str(&format!(
            "  Total redundant pairs: {}\n",
            highly_correlated_pairs.len()
        ));

        Ok(summary)
    }
}

/// Selection frequency charts
#[derive(Debug, Clone)]
pub struct SelectionFrequencyCharts;

impl SelectionFrequencyCharts {
    /// Create feature selection frequency histogram
    pub fn frequency_histogram(
        feature_frequencies: &[(usize, f64)],
        feature_names: Option<&[String]>,
        title: &str,
    ) -> Result<String> {
        if feature_frequencies.is_empty() {
            return Err(VisualizationError::EmptyData.into());
        }

        let mut histogram = String::new();

        // Title
        histogram.push_str(&format!("=== {} ===\n\n", title));

        // Sort by frequency (descending)
        let mut sorted_frequencies = feature_frequencies.to_vec();
        sorted_frequencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let max_features = 20.min(sorted_frequencies.len());
        let max_width = 40;

        histogram.push_str("Feature Selection Frequencies:\n\n");

        for (feature_idx, frequency) in sorted_frequencies.iter().take(max_features) {
            let bar_width = (*frequency * max_width as f64) as usize;
            let bar = "█".repeat(bar_width.max(1));

            let feature_name = if let Some(names) = feature_names {
                if *feature_idx < names.len() {
                    names[*feature_idx].clone()
                } else {
                    format!("Feature_{}", feature_idx)
                }
            } else {
                format!("Feature_{}", feature_idx)
            };

            histogram.push_str(&format!(
                "{:>15} |{:<width$} {:>6.1}%\n",
                feature_name,
                bar,
                frequency * 100.0,
                width = max_width + 2
            ));
        }

        if sorted_frequencies.len() > max_features {
            histogram.push_str(&format!(
                "... and {} more features\n",
                sorted_frequencies.len() - max_features
            ));
        }

        // Statistics
        let frequencies: Vec<f64> = sorted_frequencies.iter().map(|(_, f)| *f).collect();
        let avg_frequency = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        let std_frequency = {
            let variance = frequencies
                .iter()
                .map(|f| (f - avg_frequency).powi(2))
                .sum::<f64>()
                / frequencies.len() as f64;
            variance.sqrt()
        };

        histogram.push_str(&"\nFrequency Statistics:\n".to_string());
        histogram.push_str(&format!(
            "  Average frequency: {:.1}%\n",
            avg_frequency * 100.0
        ));
        histogram.push_str(&format!(
            "  Std deviation:     {:.1}%\n",
            std_frequency * 100.0
        ));
        histogram.push_str(&format!(
            "  Total features:    {}\n",
            sorted_frequencies.len()
        ));

        Ok(histogram)
    }
}

/// Comprehensive feature set visualization
#[derive(Debug, Clone)]
pub struct FeatureSetVisualization;

impl FeatureSetVisualization {
    /// Create comprehensive feature selection report with visualizations
    pub fn comprehensive_report(
        feature_indices: &[usize],
        importance_scores: &[f64],
        stability_data: Option<&[Vec<usize>]>,
        correlation_matrix: Option<ArrayView2<f64>>,
        feature_names: Option<&[String]>,
        title: &str,
    ) -> Result<String> {
        let mut report = String::new();

        // Main title
        report.push_str(
            &"╔═══════════════════════════════════════════════════════════════════════════╗\n"
                .to_string(),
        );
        report.push_str(&format!("║ {:<73} ║\n", title));
        report.push_str(
            &"╚═══════════════════════════════════════════════════════════════════════════╝\n\n"
                .to_string(),
        );

        // Feature importance visualization
        if !importance_scores.is_empty() {
            let importance_plot = FeatureImportancePlots::horizontal_bar_chart(
                feature_indices,
                importance_scores,
                feature_names,
                50,
                "Feature Importance Scores",
            )?;
            report.push_str(&importance_plot);
            report.push('\n');
        }

        // Stability analysis
        if let Some(stability_data) = stability_data {
            let stability_plot = StabilityPlots::stability_frequency_plot(
                stability_data,
                feature_indices.len(),
                feature_names,
                "Feature Selection Stability",
            )?;
            report.push_str(&stability_plot);
            report.push('\n');

            // Co-occurrence heatmap
            let cooccurrence_plot = StabilityPlots::feature_cooccurrence_heatmap(
                stability_data,
                10,
                feature_names,
                "Feature Co-occurrence Matrix",
            )?;
            report.push_str(&cooccurrence_plot);
            report.push('\n');
        }

        // Correlation analysis
        if let Some(corr_matrix) = correlation_matrix {
            let correlation_plot = RedundancyHeatmaps::correlation_heatmap(
                corr_matrix,
                feature_indices,
                feature_names,
                "Feature Correlation Matrix",
            )?;
            report.push_str(&correlation_plot);
            report.push('\n');
        }

        // Summary statistics
        report.push_str("=== Feature Selection Summary ===\n\n");
        report.push_str(&format!(
            "Total features selected: {}\n",
            feature_indices.len()
        ));

        if !importance_scores.is_empty() {
            let avg_importance =
                importance_scores.iter().sum::<f64>() / importance_scores.len() as f64;
            let max_importance = importance_scores
                .iter()
                .fold(0.0f64, |acc, &x| acc.max(x.abs()));
            report.push_str(&format!("Average importance: {:.4}\n", avg_importance));
            report.push_str(&format!("Maximum importance: {:.4}\n", max_importance));
        }

        if let Some(stability_data) = stability_data {
            let avg_selected = stability_data.iter().map(|s| s.len()).sum::<usize>() as f64
                / stability_data.len() as f64;
            report.push_str(&format!(
                "Average features per iteration: {:.1}\n",
                avg_selected
            ));
            report.push_str(&format!("Stability iterations: {}\n", stability_data.len()));
        }

        Ok(report)
    }

    /// Create quick feature summary visualization
    pub fn quick_summary(
        feature_indices: &[usize],
        importance_scores: &[f64],
        feature_names: Option<&[String]>,
    ) -> Result<String> {
        let mut summary = String::new();

        summary.push_str("=== Quick Feature Selection Summary ===\n\n");

        if feature_indices.is_empty() {
            summary.push_str("No features selected\n");
            return Ok(summary);
        }

        // Top 10 features
        let mut indexed_scores: Vec<(usize, f64)> = feature_indices
            .iter()
            .zip(importance_scores.iter())
            .map(|(&idx, &score)| (idx, score))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let top_n = 10.min(indexed_scores.len());
        summary.push_str(&format!("Top {} Selected Features:\n", top_n));
        summary.push_str("─────────────────────────────────────────────────\n");

        for (i, (feature_idx, importance)) in indexed_scores.iter().take(top_n).enumerate() {
            let feature_name = if let Some(names) = feature_names {
                if *feature_idx < names.len() {
                    names[*feature_idx].clone()
                } else {
                    format!("Feature_{}", feature_idx)
                }
            } else {
                format!("Feature_{}", feature_idx)
            };

            let bar_length = ((importance.abs() / indexed_scores[0].1.abs()) * 20.0) as usize;
            let bar = "█".repeat(bar_length.max(1));

            summary.push_str(&format!(
                "{:>2}. {:>15} |{:<20} {:>8.4}\n",
                i + 1,
                feature_name,
                bar,
                importance
            ));
        }

        // Basic statistics
        summary.push_str(&"\nSelection Statistics:\n".to_string());
        summary.push_str(&format!("  Total features: {}\n", feature_indices.len()));

        if !importance_scores.is_empty() {
            let positive_count = importance_scores.iter().filter(|&&x| x > 0.0).count();
            let negative_count = importance_scores.iter().filter(|&&x| x < 0.0).count();
            let zero_count = importance_scores.iter().filter(|&&x| x == 0.0).count();

            summary.push_str(&format!("  Positive importance: {}\n", positive_count));
            summary.push_str(&format!("  Negative importance: {}\n", negative_count));
            summary.push_str(&format!("  Zero importance: {}\n", zero_count));
        }

        Ok(summary)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_importance_plots() {
        let feature_indices = vec![0, 1, 2, 3, 4];
        let importance_scores = vec![0.8, 0.6, -0.4, 0.9, 0.2];
        let feature_names = vec![
            "Feature_A".to_string(),
            "Feature_B".to_string(),
            "Feature_C".to_string(),
            "Feature_D".to_string(),
            "Feature_E".to_string(),
        ];

        let chart = FeatureImportancePlots::horizontal_bar_chart(
            &feature_indices,
            &importance_scores,
            Some(&feature_names),
            30,
            "Test Importance",
        )
        .unwrap();

        assert!(chart.contains("Test Importance"));
        assert!(chart.contains("Feature_A"));
        assert!(chart.contains("0.8"));

        let vertical_chart = FeatureImportancePlots::vertical_bar_chart(
            &feature_indices,
            &importance_scores,
            Some(&feature_names),
            10,
            "Test Vertical",
        )
        .unwrap();

        assert!(chart.contains("Test Importance"));
    }

    #[test]
    fn test_stability_plots() {
        let feature_selections = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 2, 4], vec![1, 2, 5]];

        let plot = StabilityPlots::stability_frequency_plot(
            &feature_selections,
            6,
            None,
            "Test Stability",
        )
        .unwrap();

        assert!(plot.contains("Stability"));
        assert!(plot.contains("Feature_"));

        let heatmap = StabilityPlots::feature_cooccurrence_heatmap(
            &feature_selections,
            5,
            None,
            "Test Cooccurrence",
        )
        .unwrap();

        assert!(heatmap.contains("Cooccurrence"));
    }

    #[test]
    fn test_redundancy_heatmaps() {
        let correlation_matrix = array![[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0],];
        let feature_indices = vec![0, 1, 2];

        let heatmap = RedundancyHeatmaps::correlation_heatmap(
            correlation_matrix.view(),
            &feature_indices,
            None,
            "Test Correlation",
        )
        .unwrap();

        assert!(heatmap.contains("Correlation"));

        let pairs = vec![(0, 1, 0.8), (1, 2, 0.3)];
        let summary = RedundancyHeatmaps::redundancy_summary(&pairs, None, "Test Summary").unwrap();

        assert!(summary.contains("Correlated"));
        assert!(summary.contains("0.8"));
    }

    #[test]
    fn test_comprehensive_visualization() {
        let feature_indices = vec![0, 1, 2];
        let importance_scores = vec![0.8, 0.6, 0.4];
        let stability_data = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let correlation_matrix = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0],];

        let report = FeatureSetVisualization::comprehensive_report(
            &feature_indices,
            &importance_scores,
            Some(&stability_data),
            Some(correlation_matrix.view()),
            None,
            "Test Comprehensive Report",
        )
        .unwrap();

        assert!(report.contains("Comprehensive Report"));
        assert!(report.contains("Feature Importance"));
        assert!(report.contains("Stability"));
        assert!(report.contains("Summary"));

        let quick_summary =
            FeatureSetVisualization::quick_summary(&feature_indices, &importance_scores, None)
                .unwrap();

        assert!(quick_summary.contains("Quick"));
        assert!(quick_summary.contains("Top"));
    }

    #[test]
    fn test_selection_frequency_charts() {
        let frequencies = vec![(0, 0.9), (1, 0.7), (2, 0.5), (3, 0.3)];

        let histogram =
            SelectionFrequencyCharts::frequency_histogram(&frequencies, None, "Test Frequencies")
                .unwrap();

        assert!(histogram.contains("Frequencies"));
        assert!(histogram.contains("90.0%"));
        assert!(histogram.contains("Feature_0"));
    }
}
