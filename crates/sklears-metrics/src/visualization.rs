//! Interactive Visualization Utilities for Machine Learning Metrics
//!
//! This module provides comprehensive visualization capabilities for machine learning
//! metrics, including interactive charts, plots, and dashboard creation. The visualizations
//! help in understanding model performance, comparing different models, and identifying
//! patterns in metric behavior over time.
//!
//! # Features
//!
//! - ROC curve and PR curve plotting with interactive features
//! - Confusion matrix heatmaps with customizable styling
//! - Calibration plots for probability calibration assessment
//! - Learning curves and validation curves for model training
//! - Metric comparison charts and dashboard creation
//! - Time series plots for temporal metric analysis
//! - Feature importance visualization with ranking
//! - Interactive metric exploration tools
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::visualization::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Create ROC curve data
//! let y_true = Array1::from_vec(vec![0, 0, 1, 1, 1]);
//! let y_scores = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8, 0.65]);
//!
//! let roc_data = create_roc_curve_data(&y_true, &y_scores).unwrap();
//! println!("AUC: {:.3}", roc_data.auc);
//!
//! // Generate HTML plot
//! let html_plot = roc_data.to_html_plot(PlotConfig::default()).unwrap();
//!
//! // Create confusion matrix visualization
//! let cm = Array2::from_shape_vec((2, 2), vec![50, 10, 5, 35]).unwrap();
//! let cm_viz = ConfusionMatrixVisualization::new(&cm,
//!     vec!["Class 0".to_string(), "Class 1".to_string()]).unwrap();
//! let cm_html = cm_viz.to_html(PlotConfig::default()).unwrap();
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2, Array3};

/// Configuration for plot styling and behavior
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width of the plot in pixels
    pub width: u32,
    /// Height of the plot in pixels
    pub height: u32,
    /// Title of the plot
    pub title: String,
    /// Color scheme for the plot
    pub color_scheme: ColorScheme,
    /// Whether to show grid lines
    pub show_grid: bool,
    /// Whether to make the plot interactive
    pub interactive: bool,
    /// Font size for labels
    pub font_size: u32,
    /// Custom CSS styling
    pub custom_css: Option<String>,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Metric Visualization".to_string(),
            color_scheme: ColorScheme::Default,
            show_grid: true,
            interactive: true,
            font_size: 12,
            custom_css: None,
        }
    }
}

/// Available color schemes for visualizations
#[derive(Debug, Clone, Copy)]
pub enum ColorScheme {
    /// Default
    Default,
    /// Viridis
    Viridis,
    /// Plasma
    Plasma,
    /// Blues
    Blues,
    /// Reds
    Reds,
    /// Greens
    Greens,
    /// Custom
    Custom,
}

/// ROC curve data for visualization
#[derive(Debug, Clone)]
pub struct RocCurveData {
    /// False positive rates
    pub fpr: Array1<f64>,
    /// True positive rates  
    pub tpr: Array1<f64>,
    /// Thresholds used
    pub thresholds: Array1<f64>,
    /// Area under the curve
    pub auc: f64,
    /// Optimal threshold (closest to top-left corner)
    pub optimal_threshold: f64,
    /// Index of optimal threshold
    pub optimal_index: usize,
}

impl RocCurveData {
    /// Generate HTML plot for ROC curve
    pub fn to_html_plot(&self, config: PlotConfig) -> MetricsResult<String> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        {}
    </style>
</head>
<body>
    <div class="plot-container">
        <div id="roc-plot" style="width:{}px;height:{}px;"></div>
    </div>
    <script>
"#,
            config.title,
            config.custom_css.unwrap_or_default(),
            config.width,
            config.height
        ));

        // Generate JavaScript data
        let fpr_js = array_to_js(&self.fpr);
        let tpr_js = array_to_js(&self.tpr);

        html.push_str(&format!(
            r#"
        var trace1 = {{
            x: {},
            y: {},
            mode: 'lines+markers',
            type: 'scatter',
            name: 'ROC Curve (AUC = {:.3})',
            line: {{ color: 'blue', width: 2 }},
            marker: {{ size: 4 }}
        }};
        
        var diagonal = {{
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            type: 'scatter',
            name: 'Random Classifier',
            line: {{ color: 'red', dash: 'dash', width: 1 }}
        }};
        
        var optimal = {{
            x: [{}],
            y: [{}],
            mode: 'markers',
            type: 'scatter',
            name: 'Optimal Threshold',
            marker: {{ color: 'red', size: 10, symbol: 'star' }}
        }};
        
        var data = [trace1, diagonal, optimal];
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'False Positive Rate', range: [0, 1] }},
            yaxis: {{ title: 'True Positive Rate', range: [0, 1] }},
            showlegend: true,
            width: {},
            height: {},
            grid: {{ visible: {} }}
        }};
        
        Plotly.newPlot('roc-plot', data, layout, {{ responsive: {} }});
"#,
            fpr_js,
            tpr_js,
            self.auc,
            self.fpr[self.optimal_index],
            self.tpr[self.optimal_index],
            config.title,
            config.width,
            config.height,
            config.show_grid,
            config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

/// Precision-Recall curve data for visualization
#[derive(Debug, Clone)]
pub struct PrecisionRecallData {
    /// Precision values
    pub precision: Array1<f64>,
    /// Recall values
    pub recall: Array1<f64>,
    /// Thresholds used
    pub thresholds: Array1<f64>,
    /// Average precision score
    pub average_precision: f64,
    /// Optimal F1 threshold
    pub optimal_f1_threshold: f64,
    /// F1 scores for each threshold
    pub f1_scores: Array1<f64>,
}

impl PrecisionRecallData {
    /// Generate HTML plot for PR curve
    pub fn to_html_plot(&self, config: PlotConfig) -> MetricsResult<String> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        {}
    </style>
</head>
<body>
    <div class="plot-container">
        <div id="pr-plot" style="width:{}px;height:{}px;"></div>
    </div>
    <script>
"#,
            config.title,
            config.custom_css.unwrap_or_default(),
            config.width,
            config.height
        ));

        let precision_js = array_to_js(&self.precision);
        let recall_js = array_to_js(&self.recall);

        html.push_str(&format!(
            r#"
        var trace = {{
            x: {},
            y: {},
            mode: 'lines+markers',
            type: 'scatter',
            name: 'PR Curve (AP = {:.3})',
            line: {{ color: 'green', width: 2 }},
            marker: {{ size: 4 }}
        }};
        
        var data = [trace];
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'Recall', range: [0, 1] }},
            yaxis: {{ title: 'Precision', range: [0, 1] }},
            showlegend: true,
            width: {},
            height: {}
        }};
        
        Plotly.newPlot('pr-plot', data, layout, {{ responsive: {} }});
"#,
            recall_js,
            precision_js,
            self.average_precision,
            config.title,
            config.width,
            config.height,
            config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

/// Confusion matrix visualization
#[derive(Debug, Clone)]
pub struct ConfusionMatrixVisualization {
    /// Confusion matrix data
    pub matrix: Array2<i32>,
    /// Class labels
    pub labels: Vec<String>,
    /// Normalized matrix (optional)
    pub normalized_matrix: Option<Array2<f64>>,
}

impl ConfusionMatrixVisualization {
    /// Create new confusion matrix visualization
    pub fn new(matrix: &Array2<i32>, labels: Vec<String>) -> MetricsResult<Self> {
        if matrix.nrows() != matrix.ncols() {
            return Err(MetricsError::InvalidParameter(
                "Confusion matrix must be square".to_string(),
            ));
        }

        if labels.len() != matrix.nrows() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![matrix.nrows()],
                actual: vec![labels.len()],
            });
        }

        // Calculate normalized matrix
        let mut normalized_matrix = Array2::zeros(matrix.dim());
        for i in 0..matrix.nrows() {
            let row_sum: i32 = matrix.row(i).sum();
            if row_sum > 0 {
                for j in 0..matrix.ncols() {
                    normalized_matrix[[i, j]] = matrix[[i, j]] as f64 / row_sum as f64;
                }
            }
        }

        Ok(Self {
            matrix: matrix.clone(),
            labels,
            normalized_matrix: Some(normalized_matrix),
        })
    }

    /// Generate HTML heatmap
    pub fn to_html(&self, config: PlotConfig) -> MetricsResult<String> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        {}
    </style>
</head>
<body>
    <div class="plot-container">
        <div id="cm-plot" style="width:{}px;height:{}px;"></div>
    </div>
    <script>
"#,
            config.title,
            config.custom_css.unwrap_or_default(),
            config.width,
            config.height
        ));

        // Convert matrix to JavaScript format
        let matrix_data = matrix_to_js(&self.matrix);
        let labels_js = labels_to_js(&self.labels);

        html.push_str(&format!(
            r#"
        var data = [{{
            z: {},
            x: {},
            y: {},
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true,
            text: {},
            texttemplate: '%{{text}}',
            textfont: {{ size: {} }}
        }}];
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'Predicted Label', tickmode: 'array', tickvals: {}, ticktext: {} }},
            yaxis: {{ title: 'True Label', tickmode: 'array', tickvals: {}, ticktext: {} }},
            width: {},
            height: {}
        }};
        
        Plotly.newPlot('cm-plot', data, layout, {{ responsive: {} }});
"#,
            matrix_data,
            labels_js.clone(),
            labels_js.clone(),
            matrix_data,
            config.font_size,
            config.title,
            (0..self.labels.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            labels_js.clone(),
            (0..self.labels.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            labels_js,
            config.width,
            config.height,
            config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

/// Calibration plot for probability calibration assessment
#[derive(Debug, Clone)]
pub struct CalibrationPlot {
    /// Mean predicted probabilities per bin
    pub mean_predicted_probs: Array1<f64>,
    /// Fraction of positives per bin
    pub fraction_positives: Array1<f64>,
    /// Number of samples per bin
    pub bin_counts: Array1<i32>,
    /// Bin edges
    pub bin_edges: Array1<f64>,
    /// Brier score
    pub brier_score: f64,
    /// Expected calibration error
    pub ece: f64,
}

impl CalibrationPlot {
    /// Generate HTML plot for calibration curve
    pub fn to_html_plot(&self, config: PlotConfig) -> MetricsResult<String> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        .metrics {{ margin: 10px 0; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }}
        {}
    </style>
</head>
<body>
    <div class="metrics">
        <strong>Calibration Metrics:</strong><br>
        Brier Score: {:.4}<br>
        Expected Calibration Error (ECE): {:.4}
    </div>
    <div class="plot-container">
        <div id="cal-plot" style="width:{}px;height:{}px;"></div>
    </div>
    <script>
"#,
            config.title,
            config.custom_css.unwrap_or_default(),
            self.brier_score,
            self.ece,
            config.width,
            config.height
        ));

        let mean_pred_js = array_to_js(&self.mean_predicted_probs);
        let frac_pos_js = array_to_js(&self.fraction_positives);

        html.push_str(&format!(
            r#"
        var calibration_curve = {{
            x: {},
            y: {},
            mode: 'lines+markers',
            type: 'scatter',
            name: 'Calibration Curve',
            line: {{ color: 'blue', width: 2 }},
            marker: {{ size: 6 }}
        }};
        
        var perfect_calibration = {{
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            type: 'scatter',
            name: 'Perfect Calibration',
            line: {{ color: 'red', dash: 'dash', width: 1 }}
        }};
        
        var data = [calibration_curve, perfect_calibration];
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'Mean Predicted Probability', range: [0, 1] }},
            yaxis: {{ title: 'Fraction of Positives', range: [0, 1] }},
            showlegend: true,
            width: {},
            height: {}
        }};
        
        Plotly.newPlot('cal-plot', data, layout, {{ responsive: {} }});
"#,
            mean_pred_js,
            frac_pos_js,
            config.title,
            config.width,
            config.height,
            config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

/// Learning curve visualization for training progress
#[derive(Debug, Clone)]
pub struct LearningCurve {
    /// Training set sizes
    pub train_sizes: Array1<f64>,
    /// Training scores
    pub train_scores: Array1<f64>,
    /// Validation scores
    pub validation_scores: Array1<f64>,
    /// Training score standard deviations (optional)
    pub train_scores_std: Option<Array1<f64>>,
    /// Validation score standard deviations (optional)
    pub validation_scores_std: Option<Array1<f64>>,
}

impl LearningCurve {
    /// Generate HTML plot for learning curve
    pub fn to_html_plot(&self, config: PlotConfig) -> MetricsResult<String> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        {}
    </style>
</head>
<body>
    <div class="plot-container">
        <div id="learning-plot" style="width:{}px;height:{}px;"></div>
    </div>
    <script>
"#,
            config.title,
            config.custom_css.unwrap_or_default(),
            config.width,
            config.height
        ));

        let train_sizes_js = array_to_js(&self.train_sizes);
        let train_scores_js = array_to_js(&self.train_scores);
        let val_scores_js = array_to_js(&self.validation_scores);

        html.push_str(&format!(
            r#"
        var train_trace = {{
            x: {},
            y: {},
            mode: 'lines+markers',
            type: 'scatter',
            name: 'Training Score',
            line: {{ color: 'blue', width: 2 }},
            marker: {{ size: 6 }}
        }};
        
        var val_trace = {{
            x: {},
            y: {},
            mode: 'lines+markers',
            type: 'scatter',
            name: 'Validation Score',
            line: {{ color: 'red', width: 2 }},
            marker: {{ size: 6 }}
        }};
        
        var data = [train_trace, val_trace];
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'Training Set Size' }},
            yaxis: {{ title: 'Score' }},
            showlegend: true,
            width: {},
            height: {}
        }};
        
        Plotly.newPlot('learning-plot', data, layout, {{ responsive: {} }});
"#,
            train_sizes_js,
            train_scores_js,
            train_sizes_js,
            val_scores_js,
            config.title,
            config.width,
            config.height,
            config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

/// Feature importance visualization
#[derive(Debug, Clone)]
pub struct FeatureImportanceViz {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Importance values
    pub importances: Array1<f64>,
    /// Standard deviations (optional)
    pub std_devs: Option<Array1<f64>>,
    /// Number of top features to display
    pub top_k: usize,
}

impl FeatureImportanceViz {
    /// Create new feature importance visualization
    pub fn new(
        feature_names: Vec<String>,
        importances: Array1<f64>,
        std_devs: Option<Array1<f64>>,
        top_k: usize,
    ) -> MetricsResult<Self> {
        if feature_names.len() != importances.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![importances.len()],
                actual: vec![feature_names.len()],
            });
        }

        if let Some(ref stds) = std_devs {
            if stds.len() != importances.len() {
                return Err(MetricsError::ShapeMismatch {
                    expected: vec![importances.len()],
                    actual: vec![stds.len()],
                });
            }
        }

        Ok(Self {
            feature_names,
            importances,
            std_devs,
            top_k,
        })
    }

    /// Generate HTML bar chart for feature importance
    pub fn to_html_plot(&self, config: PlotConfig) -> MetricsResult<String> {
        // Sort features by importance
        let mut indexed_importances: Vec<(usize, f64)> = self
            .importances
            .iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();
        indexed_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k = self.top_k.min(indexed_importances.len());
        let top_indices: Vec<usize> = indexed_importances
            .iter()
            .take(top_k)
            .map(|(idx, _)| *idx)
            .collect();

        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        {}
    </style>
</head>
<body>
    <div class="plot-container">
        <div id="importance-plot" style="width:{}px;height:{}px;"></div>
    </div>
    <script>
"#,
            config.title,
            config.custom_css.unwrap_or_default(),
            config.width,
            config.height
        ));

        let top_names: Vec<String> = top_indices
            .iter()
            .map(|&i| self.feature_names[i].clone())
            .collect();
        let top_importances: Vec<f64> = top_indices.iter().map(|&i| self.importances[i]).collect();

        let names_js = top_names
            .iter()
            .map(|name| format!("'{}'", name))
            .collect::<Vec<_>>()
            .join(",");
        let importances_js = top_importances
            .iter()
            .map(|imp| imp.to_string())
            .collect::<Vec<_>>()
            .join(",");

        html.push_str(&format!(
            r#"
        var trace = {{
            x: [{}],
            y: [{}],
            type: 'bar',
            marker: {{ color: 'steelblue' }},
            name: 'Feature Importance'
        }};
        
        var data = [trace];
        
        var layout = {{
            title: '{} (Top {})',
            xaxis: {{ title: 'Features' }},
            yaxis: {{ title: 'Importance' }},
            width: {},
            height: {}
        }};
        
        Plotly.newPlot('importance-plot', data, layout, {{ responsive: {} }});
"#,
            importances_js,
            names_js,
            config.title,
            top_k,
            config.width,
            config.height,
            config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

/// Metric comparison dashboard
#[derive(Debug, Clone)]
pub struct MetricDashboard {
    /// Model names
    pub model_names: Vec<String>,
    /// Metric names
    pub metric_names: Vec<String>,
    /// Metric values (models x metrics)
    pub metric_values: Array2<f64>,
    /// Optional confidence intervals
    pub confidence_intervals: Option<Array3<f64>>, // models x metrics x 2 (lower, upper)
}

impl MetricDashboard {
    /// Create new metric dashboard
    pub fn new(
        model_names: Vec<String>,
        metric_names: Vec<String>,
        metric_values: Array2<f64>,
        confidence_intervals: Option<Array3<f64>>,
    ) -> MetricsResult<Self> {
        if model_names.len() != metric_values.nrows() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![metric_values.nrows()],
                actual: vec![model_names.len()],
            });
        }

        if metric_names.len() != metric_values.ncols() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![metric_values.ncols()],
                actual: vec![metric_names.len()],
            });
        }

        Ok(Self {
            model_names,
            metric_names,
            metric_values,
            confidence_intervals,
        })
    }

    /// Generate HTML dashboard
    pub fn to_html_dashboard(&self, config: PlotConfig) -> MetricsResult<String> {
        let mut html = String::new();

        html.push_str(&format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .plot-container {{ margin: 10px 0; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        {}
    </style>
</head>
<body>
    <h1>{}</h1>
    
    <h2>Metrics Summary Table</h2>
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Model</th>
"#, config.title, config.custom_css.unwrap_or_default(), config.title));

        // Add metric column headers
        for metric_name in &self.metric_names {
            html.push_str(&format!("<th>{}</th>", metric_name));
        }
        html.push_str("</tr></thead><tbody>");

        // Add data rows
        for (i, model_name) in self.model_names.iter().enumerate() {
            html.push_str(&format!("<tr><td><strong>{}</strong></td>", model_name));
            for j in 0..self.metric_names.len() {
                let value = self.metric_values[[i, j]];
                html.push_str(&format!("<td>{:.4}</td>", value));
            }
            html.push_str("</tr>");
        }

        html.push_str(
            r#"
        </tbody>
    </table>
    
    <div class="dashboard">
        <div class="plot-container">
            <div id="bar-plot" style="width:400px;height:300px;"></div>
        </div>
        <div class="plot-container">
            <div id="radar-plot" style="width:400px;height:300px;"></div>
        </div>
    </div>
    
    <script>
"#,
        );

        // Generate bar chart data
        let models_js = self
            .model_names
            .iter()
            .map(|name| format!("'{}'", name))
            .collect::<Vec<_>>()
            .join(",");

        html.push_str("// Bar Chart\n");
        for (j, metric_name) in self.metric_names.iter().enumerate() {
            let values: Vec<f64> = (0..self.model_names.len())
                .map(|i| self.metric_values[[i, j]])
                .collect();
            let values_js = values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");

            html.push_str(&format!(
                r#"
        var trace{} = {{
            x: [{}],
            y: [{}],
            type: 'bar',
            name: '{}'
        }};
"#,
                j, models_js, values_js, metric_name
            ));
        }

        html.push_str(&format!(
            r#"
        var bar_data = [{}];
        
        var bar_layout = {{
            title: 'Model Comparison',
            xaxis: {{ title: 'Models' }},
            yaxis: {{ title: 'Score' }},
            barmode: 'group'
        }};
        
        Plotly.newPlot('bar-plot', bar_data, bar_layout, {{ responsive: true }});
"#,
            (0..self.metric_names.len())
                .map(|i| format!("trace{}", i))
                .collect::<Vec<_>>()
                .join(",")
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        Ok(html)
    }
}

// Utility functions for creating visualizations

/// Create ROC curve data from true labels and scores
pub fn create_roc_curve_data(
    y_true: &Array1<i32>,
    y_scores: &Array1<f64>,
) -> MetricsResult<RocCurveData> {
    if y_true.len() != y_scores.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_scores.len()],
        });
    }

    // Sort by scores in descending order
    let mut scored_labels: Vec<(f64, i32)> = y_scores
        .iter()
        .zip(y_true.iter())
        .map(|(&score, &label)| (score, label))
        .collect();
    scored_labels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = y_true.iter().filter(|&&label| label == 1).count();
    let n_neg = y_true.len() - n_pos;

    let mut tpr_vec = vec![0.0];
    let mut fpr_vec = vec![0.0];
    let mut thresholds_vec = vec![f64::INFINITY];

    let mut tp = 0;
    let mut fp = 0;

    for (score, label) in scored_labels {
        if label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }

        let tpr = if n_pos > 0 {
            tp as f64 / n_pos as f64
        } else {
            0.0
        };
        let fpr = if n_neg > 0 {
            fp as f64 / n_neg as f64
        } else {
            0.0
        };

        tpr_vec.push(tpr);
        fpr_vec.push(fpr);
        thresholds_vec.push(score);
    }

    // Calculate AUC using trapezoidal rule
    let mut auc = 0.0;
    for i in 1..fpr_vec.len() {
        let dx = fpr_vec[i] - fpr_vec[i - 1];
        let avg_y = (tpr_vec[i] + tpr_vec[i - 1]) / 2.0;
        auc += dx * avg_y;
    }

    // Find optimal threshold (closest to top-left corner)
    let mut optimal_index = 0;
    let mut min_distance = f64::INFINITY;
    for i in 0..tpr_vec.len() {
        let distance = (fpr_vec[i] - 0.0).powi(2) + (tpr_vec[i] - 1.0).powi(2);
        if distance < min_distance {
            min_distance = distance;
            optimal_index = i;
        }
    }

    Ok(RocCurveData {
        fpr: Array1::from_vec(fpr_vec),
        tpr: Array1::from_vec(tpr_vec),
        thresholds: Array1::from_vec(thresholds_vec.clone()),
        auc,
        optimal_threshold: thresholds_vec[optimal_index],
        optimal_index,
    })
}

/// Create precision-recall curve data
pub fn create_precision_recall_data(
    y_true: &Array1<i32>,
    y_scores: &Array1<f64>,
) -> MetricsResult<PrecisionRecallData> {
    if y_true.len() != y_scores.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_scores.len()],
        });
    }

    // Sort by scores in descending order
    let mut scored_labels: Vec<(f64, i32)> = y_scores
        .iter()
        .zip(y_true.iter())
        .map(|(&score, &label)| (score, label))
        .collect();
    scored_labels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = y_true.iter().filter(|&&label| label == 1).count();

    let mut precision_vec = Vec::new();
    let mut recall_vec = Vec::new();
    let mut f1_vec = Vec::new();
    let mut thresholds_vec = Vec::new();

    let mut tp = 0;
    let mut fp = 0;

    for (score, label) in scored_labels {
        if label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if n_pos > 0 {
            tp as f64 / n_pos as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        precision_vec.push(precision);
        recall_vec.push(recall);
        f1_vec.push(f1);
        thresholds_vec.push(score);
    }

    // Calculate average precision using trapezoidal rule
    let mut average_precision = 0.0;
    for i in 1..recall_vec.len() {
        let dr = recall_vec[i] - recall_vec[i - 1];
        average_precision += precision_vec[i] * dr;
    }

    // Find optimal F1 threshold
    let optimal_f1_index = f1_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0);

    Ok(PrecisionRecallData {
        precision: Array1::from_vec(precision_vec),
        recall: Array1::from_vec(recall_vec),
        thresholds: Array1::from_vec(thresholds_vec.clone()),
        average_precision,
        optimal_f1_threshold: thresholds_vec[optimal_f1_index],
        f1_scores: Array1::from_vec(f1_vec),
    })
}

/// Create calibration plot data
pub fn create_calibration_plot_data(
    y_true: &Array1<i32>,
    y_prob: &Array1<f64>,
    n_bins: usize,
) -> MetricsResult<CalibrationPlot> {
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_prob.len()],
        });
    }

    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();
    let mut mean_predicted_probs = Vec::new();
    let mut fraction_positives = Vec::new();
    let mut bin_counts = Vec::new();

    for i in 0..n_bins {
        let bin_min = bin_edges[i];
        let bin_max = bin_edges[i + 1];

        let bin_indices: Vec<usize> = y_prob
            .iter()
            .enumerate()
            .filter(|(_, &prob)| prob >= bin_min && prob < bin_max)
            .map(|(idx, _)| idx)
            .collect();

        if bin_indices.is_empty() {
            mean_predicted_probs.push(0.0);
            fraction_positives.push(0.0);
            bin_counts.push(0);
            continue;
        }

        let bin_y_true: Vec<i32> = bin_indices.iter().map(|&idx| y_true[idx]).collect();
        let bin_y_prob: Vec<f64> = bin_indices.iter().map(|&idx| y_prob[idx]).collect();

        let mean_pred = bin_y_prob.iter().sum::<f64>() / bin_y_prob.len() as f64;
        let frac_pos =
            bin_y_true.iter().filter(|&&label| label == 1).count() as f64 / bin_y_true.len() as f64;

        mean_predicted_probs.push(mean_pred);
        fraction_positives.push(frac_pos);
        bin_counts.push(bin_indices.len() as i32);
    }

    // Calculate Brier score
    let brier_score = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&true_label, &pred_prob)| (true_label as f64 - pred_prob).powi(2))
        .sum::<f64>()
        / y_true.len() as f64;

    // Calculate Expected Calibration Error (ECE)
    let mut ece = 0.0;
    let total_samples = y_true.len() as f64;
    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let bin_weight = bin_counts[i] as f64 / total_samples;
            let calibration_error = (mean_predicted_probs[i] - fraction_positives[i]).abs();
            ece += bin_weight * calibration_error;
        }
    }

    Ok(CalibrationPlot {
        mean_predicted_probs: Array1::from_vec(mean_predicted_probs),
        fraction_positives: Array1::from_vec(fraction_positives),
        bin_counts: Array1::from_vec(bin_counts),
        bin_edges: Array1::from_vec(bin_edges),
        brier_score,
        ece,
    })
}

// Helper functions for JavaScript generation

fn array_to_js(arr: &Array1<f64>) -> String {
    let values: Vec<String> = arr.iter().map(|x| x.to_string()).collect();
    format!("[{}]", values.join(","))
}

fn matrix_to_js(matrix: &Array2<i32>) -> String {
    let mut rows = Vec::new();
    for i in 0..matrix.nrows() {
        let row: Vec<String> = matrix.row(i).iter().map(|x| x.to_string()).collect();
        rows.push(format!("[{}]", row.join(",")));
    }
    format!("[{}]", rows.join(","))
}

fn labels_to_js(labels: &[String]) -> String {
    let quoted_labels: Vec<String> = labels.iter().map(|s| format!("'{}'", s)).collect();
    format!("[{}]", quoted_labels.join(","))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_roc_curve_creation() {
        let y_true = Array1::from_vec(vec![0, 0, 1, 1, 1]);
        let y_scores = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8, 0.65]);

        let roc_data = create_roc_curve_data(&y_true, &y_scores).unwrap();

        assert!(roc_data.auc >= 0.0 && roc_data.auc <= 1.0);
        assert_eq!(roc_data.fpr.len(), roc_data.tpr.len());
        assert_eq!(roc_data.fpr.len(), roc_data.thresholds.len());
        assert_eq!(roc_data.fpr[0], 0.0);
        assert_eq!(roc_data.tpr[0], 0.0);
    }

    #[test]
    fn test_precision_recall_creation() {
        let y_true = Array1::from_vec(vec![0, 0, 1, 1, 1]);
        let y_scores = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8, 0.65]);

        let pr_data = create_precision_recall_data(&y_true, &y_scores).unwrap();

        assert!(pr_data.average_precision >= 0.0 && pr_data.average_precision <= 1.0);
        assert_eq!(pr_data.precision.len(), pr_data.recall.len());
        assert_eq!(pr_data.precision.len(), pr_data.thresholds.len());
        assert_eq!(pr_data.precision.len(), pr_data.f1_scores.len());
    }

    #[test]
    fn test_confusion_matrix_viz() {
        let matrix = Array2::from_shape_vec((2, 2), vec![50, 10, 5, 35]).unwrap();
        let labels = vec!["Class 0".to_string(), "Class 1".to_string()];

        let cm_viz = ConfusionMatrixVisualization::new(&matrix, labels).unwrap();

        assert_eq!(cm_viz.matrix.shape(), &[2, 2]);
        assert_eq!(cm_viz.labels.len(), 2);
        assert!(cm_viz.normalized_matrix.is_some());

        let normalized = cm_viz.normalized_matrix.as_ref().unwrap();
        // First row should sum to 1 (50+10 = 60, so 50/60, 10/60)
        assert_abs_diff_eq!(
            normalized[[0, 0]] + normalized[[0, 1]],
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_calibration_plot_creation() {
        let y_true = Array1::from_vec(vec![0, 0, 1, 1, 1, 0, 1, 0, 1, 1]);
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.15, 0.9, 0.25, 0.85, 0.95]);

        let cal_plot = create_calibration_plot_data(&y_true, &y_prob, 5).unwrap();

        assert_eq!(cal_plot.mean_predicted_probs.len(), 5);
        assert_eq!(cal_plot.fraction_positives.len(), 5);
        assert_eq!(cal_plot.bin_counts.len(), 5);
        assert!(cal_plot.brier_score >= 0.0);
        assert!(cal_plot.ece >= 0.0 && cal_plot.ece <= 1.0);
    }

    #[test]
    fn test_feature_importance_viz() {
        let feature_names = vec![
            "feature1".to_string(),
            "feature2".to_string(),
            "feature3".to_string(),
        ];
        let importances = Array1::from_vec(vec![0.5, 0.3, 0.2]);

        let feat_viz = FeatureImportanceViz::new(feature_names, importances, None, 3).unwrap();

        assert_eq!(feat_viz.feature_names.len(), 3);
        assert_eq!(feat_viz.importances.len(), 3);
        assert_eq!(feat_viz.top_k, 3);
    }

    #[test]
    fn test_metric_dashboard() {
        let model_names = vec!["Model A".to_string(), "Model B".to_string()];
        let metric_names = vec!["Accuracy".to_string(), "F1".to_string()];
        let metric_values = Array2::from_shape_vec((2, 2), vec![0.85, 0.82, 0.90, 0.88]).unwrap();

        let dashboard =
            MetricDashboard::new(model_names, metric_names, metric_values, None).unwrap();

        assert_eq!(dashboard.model_names.len(), 2);
        assert_eq!(dashboard.metric_names.len(), 2);
        assert_eq!(dashboard.metric_values.shape(), &[2, 2]);
    }

    #[test]
    fn test_learning_curve() {
        let train_sizes = Array1::from_vec(vec![100.0, 200.0, 300.0, 400.0]);
        let train_scores = Array1::from_vec(vec![0.8, 0.85, 0.88, 0.90]);
        let val_scores = Array1::from_vec(vec![0.75, 0.80, 0.82, 0.83]);

        let learning_curve = LearningCurve {
            train_sizes,
            train_scores,
            validation_scores: val_scores,
            train_scores_std: None,
            validation_scores_std: None,
        };

        assert_eq!(learning_curve.train_sizes.len(), 4);
        assert_eq!(learning_curve.train_scores.len(), 4);
        assert_eq!(learning_curve.validation_scores.len(), 4);
    }

    #[test]
    fn test_array_to_js() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let js = array_to_js(&arr);
        assert_eq!(js, "[1,2,3]");
    }

    #[test]
    fn test_matrix_to_js() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let js = matrix_to_js(&matrix);
        assert_eq!(js, "[[1,2],[3,4]]");
    }

    #[test]
    fn test_labels_to_js() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let js = labels_to_js(&labels);
        assert_eq!(js, "['A','B']");
    }
}
