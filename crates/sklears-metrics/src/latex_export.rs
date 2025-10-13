//! LaTeX/PDF Report Export functionality for sklears-metrics
//!
//! This module provides comprehensive LaTeX/PDF export capabilities for metric reports,
//! including customizable templates, charts, and visualizations.
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::latex_export::{LatexReportBuilder, ReportTemplate};
//! use scirs2_core::ndarray::array;
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 1];
//!
//! let report = LatexReportBuilder::new()
//!     .template(ReportTemplate::Classification)
//!     .title("Model Performance Report")
//!     .add_classification_metrics(&y_true, &y_pred)
//!     .build()
//!     .unwrap();
//!
//! // Export to LaTeX
//! let latex_code = report.to_latex().unwrap();
//!
//! // Export to PDF
//! report.to_pdf("report.pdf").unwrap();
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "serde"))]
pub trait Serialize {}

#[cfg(not(feature = "serde"))]
pub trait Deserialize<'de> {}
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// LaTeX report templates for different types of metrics
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ReportTemplate {
    /// Basic classification report with confusion matrix and common metrics
    Classification,
    /// Comprehensive regression report with error analysis
    Regression,
    /// Clustering evaluation report with internal and external validation
    Clustering,
    /// Multi-objective comparison report
    MultiObjective,
    /// Time series analysis report with trend and seasonality
    TimeSeries,
    /// Custom template loaded from file
    Custom,
}

/// Configuration for LaTeX report generation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatexConfig {
    /// Document class (article, report, book)
    pub document_class: String,
    /// Paper size (a4paper, letterpaper, etc.)
    pub paper_size: String,
    /// Font size (10pt, 11pt, 12pt)
    pub font_size: String,
    /// LaTeX packages to include
    pub packages: Vec<String>,
    /// Whether to include page numbers
    pub page_numbers: bool,
    /// Whether to include table of contents
    pub table_of_contents: bool,
    /// Whether to include bibliography
    pub bibliography: bool,
    /// Output directory for generated files
    pub output_dir: PathBuf,
    /// Whether to clean up auxiliary files
    pub cleanup_aux_files: bool,
}

impl Default for LatexConfig {
    fn default() -> Self {
        Self {
            document_class: "article".to_string(),
            paper_size: "a4paper".to_string(),
            font_size: "11pt".to_string(),
            packages: vec![
                "graphicx".to_string(),
                "booktabs".to_string(),
                "amsmath".to_string(),
                "amsfonts".to_string(),
                "array".to_string(),
                "float".to_string(),
                "geometry".to_string(),
                "hyperref".to_string(),
                "xcolor".to_string(),
            ],
            page_numbers: true,
            table_of_contents: true,
            bibliography: false,
            output_dir: PathBuf::from("./reports"),
            cleanup_aux_files: true,
        }
    }
}

/// Data structure for metric values to be included in reports
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricData {
    pub name: String,
    pub value: f64,
    pub description: String,
    pub interpretation: Option<String>,
    pub confidence_interval: Option<(f64, f64)>,
}

/// Data structure for tables in LaTeX reports
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatexTable {
    pub caption: String,
    pub label: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub alignment: String, // e.g., "c|c|c" for center-aligned columns with vertical lines
}

/// Data structure for figures and charts
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatexFigure {
    pub caption: String,
    pub label: String,
    pub file_path: String,
    pub width: String,    // e.g., "0.8\\textwidth"
    pub position: String, // e.g., "htbp"
}

/// Main LaTeX report structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatexReport {
    /// Report configuration
    pub config: LatexConfig,
    /// Report title
    pub title: String,
    /// Report author
    pub author: String,
    /// Report date
    pub date: String,
    /// Abstract/summary
    pub abstract_text: Option<String>,
    /// Sections of the report
    pub sections: Vec<ReportSection>,
    /// List of metrics
    pub metrics: Vec<MetricData>,
    /// Tables to include
    pub tables: Vec<LatexTable>,
    /// Figures to include
    pub figures: Vec<LatexFigure>,
    /// Bibliography entries
    pub bibliography: Vec<BibEntry>,
}

/// Report section structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReportSection {
    pub title: String,
    pub content: String,
    pub subsections: Vec<ReportSection>,
}

/// Bibliography entry
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BibEntry {
    pub key: String,
    pub entry_type: String, // article, book, inproceedings, etc.
    pub fields: HashMap<String, String>,
}

/// Builder for creating LaTeX reports
pub struct LatexReportBuilder {
    report: LatexReport,
}

impl LatexReportBuilder {
    /// Create a new LaTeX report builder
    pub fn new() -> Self {
        Self {
            report: LatexReport {
                config: LatexConfig::default(),
                title: "Machine Learning Metrics Report".to_string(),
                author: "sklears-metrics".to_string(),
                date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
                abstract_text: None,
                sections: Vec::new(),
                metrics: Vec::new(),
                tables: Vec::new(),
                figures: Vec::new(),
                bibliography: Vec::new(),
            },
        }
    }

    /// Set the report template
    pub fn template(mut self, template: ReportTemplate) -> Self {
        // Store current title and author before applying template
        let current_title = self.report.title.clone();
        let current_author = self.report.author.clone();

        self.apply_template(template);

        // Restore title and author if they were custom (not default)
        if current_title != "Machine Learning Metrics Report" {
            self.report.title = current_title;
        }
        if current_author != "sklears-metrics" {
            self.report.author = current_author;
        }

        self
    }

    /// Set the report title
    pub fn title<S: Into<String>>(mut self, title: S) -> Self {
        self.report.title = title.into();
        self
    }

    /// Set the report author
    pub fn author<S: Into<String>>(mut self, author: S) -> Self {
        self.report.author = author.into();
        self
    }

    /// Set the abstract
    pub fn abstract_text<S: Into<String>>(mut self, abstract_text: S) -> Self {
        self.report.abstract_text = Some(abstract_text.into());
        self
    }

    /// Add a section to the report
    pub fn add_section<S: Into<String>>(mut self, title: S, content: S) -> Self {
        self.report.sections.push(ReportSection {
            title: title.into(),
            content: content.into(),
            subsections: Vec::new(),
        });
        self
    }

    /// Add classification metrics to the report
    pub fn add_classification_metrics(
        mut self,
        y_true: &ArrayView1<i32>,
        y_pred: &ArrayView1<i32>,
    ) -> MetricsResult<Self> {
        use crate::classification::{
            accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
        };

        // Convert views to owned arrays for metric computation
        let y_true_owned = y_true.to_owned();
        let y_pred_owned = y_pred.to_owned();

        // Calculate metrics
        let accuracy = accuracy_score(&y_true_owned, &y_pred_owned)?;

        // Calculate macro-averaged metrics by calculating per-class metrics and averaging
        let unique_labels: std::collections::BTreeSet<_> = y_true_owned
            .iter()
            .chain(y_pred_owned.iter())
            .cloned()
            .collect();
        let mut precisions = Vec::new();
        let mut recalls = Vec::new();
        let mut f1s = Vec::new();

        for &label in &unique_labels {
            if let Ok(p) = precision_score(&y_true_owned, &y_pred_owned, Some(label)) {
                precisions.push(p);
            }
            if let Ok(r) = recall_score(&y_true_owned, &y_pred_owned, Some(label)) {
                recalls.push(r);
            }
            if let Ok(f) = f1_score(&y_true_owned, &y_pred_owned, Some(label)) {
                f1s.push(f);
            }
        }

        let precision = if precisions.is_empty() {
            0.0
        } else {
            precisions.iter().sum::<f64>() / precisions.len() as f64
        };
        let recall = if recalls.is_empty() {
            0.0
        } else {
            recalls.iter().sum::<f64>() / recalls.len() as f64
        };
        let f1 = if f1s.is_empty() {
            0.0
        } else {
            f1s.iter().sum::<f64>() / f1s.len() as f64
        };

        // Add metrics
        self.report.metrics.push(MetricData {
            name: "Accuracy".to_string(),
            value: accuracy,
            description: "Fraction of predictions that match the true labels".to_string(),
            interpretation: Some(self.interpret_accuracy(accuracy)),
            confidence_interval: None,
        });

        self.report.metrics.push(MetricData {
            name: "Precision (Macro)".to_string(),
            value: precision,
            description: "Average precision across all classes".to_string(),
            interpretation: Some(self.interpret_precision(precision)),
            confidence_interval: None,
        });

        self.report.metrics.push(MetricData {
            name: "Recall (Macro)".to_string(),
            value: recall,
            description: "Average recall across all classes".to_string(),
            interpretation: Some(self.interpret_recall(recall)),
            confidence_interval: None,
        });

        self.report.metrics.push(MetricData {
            name: "F1-Score (Macro)".to_string(),
            value: f1,
            description: "Harmonic mean of precision and recall".to_string(),
            interpretation: Some(self.interpret_f1(f1)),
            confidence_interval: None,
        });

        // Add confusion matrix table
        let cm = confusion_matrix(&y_true_owned, &y_pred_owned)?;
        let cm_table = self.create_confusion_matrix_table(&cm);
        self.report.tables.push(cm_table);

        Ok(self)
    }

    /// Add regression metrics to the report
    pub fn add_regression_metrics(
        mut self,
        y_true: &ArrayView1<f64>,
        y_pred: &ArrayView1<f64>,
    ) -> MetricsResult<Self> {
        use crate::regression::{
            mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
        };

        // Convert views to owned arrays for metric computation
        let y_true_owned = y_true.to_owned();
        let y_pred_owned = y_pred.to_owned();

        // Calculate metrics
        let mae = mean_absolute_error(&y_true_owned, &y_pred_owned)?;
        let mse = mean_squared_error(&y_true_owned, &y_pred_owned)?;
        let rmse = root_mean_squared_error(&y_true_owned, &y_pred_owned)?;
        let r2 = r2_score(&y_true_owned, &y_pred_owned)?;

        // Add metrics
        self.report.metrics.extend(vec![
            MetricData {
                name: "Mean Absolute Error".to_string(),
                value: mae,
                description: "Average absolute difference between predicted and true values"
                    .to_string(),
                interpretation: Some(self.interpret_mae(mae)),
                confidence_interval: None,
            },
            MetricData {
                name: "Mean Squared Error".to_string(),
                value: mse,
                description: "Average squared difference between predicted and true values"
                    .to_string(),
                interpretation: Some(self.interpret_mse(mse)),
                confidence_interval: None,
            },
            MetricData {
                name: "Root Mean Squared Error".to_string(),
                value: rmse,
                description: "Square root of the mean squared error".to_string(),
                interpretation: Some(self.interpret_rmse(rmse)),
                confidence_interval: None,
            },
            MetricData {
                name: "R² Score".to_string(),
                value: r2,
                description: "Coefficient of determination".to_string(),
                interpretation: Some(self.interpret_r2(r2)),
                confidence_interval: None,
            },
        ]);

        Ok(self)
    }

    /// Add a custom metric
    pub fn add_metric(mut self, metric: MetricData) -> Self {
        self.report.metrics.push(metric);
        self
    }

    /// Add a table
    pub fn add_table(mut self, table: LatexTable) -> Self {
        self.report.tables.push(table);
        self
    }

    /// Add a figure
    pub fn add_figure(mut self, figure: LatexFigure) -> Self {
        self.report.figures.push(figure);
        self
    }

    /// Set custom LaTeX configuration
    pub fn config(mut self, config: LatexConfig) -> Self {
        self.report.config = config;
        self
    }

    /// Build the final report
    pub fn build(self) -> MetricsResult<LatexReport> {
        Ok(self.report)
    }

    // Helper methods for applying templates
    fn apply_template(&mut self, template: ReportTemplate) {
        match template {
            ReportTemplate::Classification => {
                self.report.title = "Classification Performance Report".to_string();
                self.report.abstract_text = Some(
                    "This report provides a comprehensive evaluation of classification model performance \
                     using multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC. \
                     The analysis includes confusion matrix visualization and detailed metric interpretation.".to_string()
                );
                self.report.sections = vec![
                    ReportSection {
                        title: "Executive Summary".to_string(),
                        content: "This section provides a high-level overview of the model's classification performance.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Detailed Metrics Analysis".to_string(),
                        content: "Comprehensive analysis of classification metrics with mathematical formulations.".to_string(),
                        subsections: vec![
                            ReportSection {
                                title: "Confusion Matrix".to_string(),
                                content: "The confusion matrix provides detailed view of correct and incorrect predictions for each class.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Precision and Recall".to_string(),
                                content: "Analysis of precision (positive predictive value) and recall (sensitivity) for each class.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "F1-Score and ROC Analysis".to_string(),
                                content: "Harmonic mean of precision and recall, plus receiver operating characteristic analysis.".to_string(),
                                subsections: Vec::new(),
                            },
                        ],
                    },
                    ReportSection {
                        title: "Statistical Significance".to_string(),
                        content: "Statistical analysis including confidence intervals and significance tests.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Recommendations".to_string(),
                        content: "Key performance indicators are summarized below.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Confusion Matrix".to_string(),
                        content: "The confusion matrix shows the distribution of predictions across classes.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Interpretation".to_string(),
                        content: "Detailed interpretation of the results and recommendations for model improvement.".to_string(),
                        subsections: Vec::new(),
                    },
                ];
            }
            ReportTemplate::Regression => {
                self.report.title = "Regression Performance Report".to_string();
                self.report.abstract_text = Some(
                    "This report provides a comprehensive evaluation of regression model performance \
                     using metrics such as MAE, MSE, RMSE, R², and advanced statistical measures. \
                     The analysis includes residual analysis, error distribution, and model diagnostics.".to_string()
                );
                self.report.sections = vec![
                    ReportSection {
                        title: "Executive Summary".to_string(),
                        content: "High-level overview of regression model performance and key findings.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Error Analysis".to_string(),
                        content: "Comprehensive analysis of prediction errors and residuals.".to_string(),
                        subsections: vec![
                            ReportSection {
                                title: "Mean Absolute Error (MAE)".to_string(),
                                content: "MAE measures the average magnitude of errors without considering direction.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Mean Squared Error (MSE) and RMSE".to_string(),
                                content: "MSE penalizes larger errors more heavily; RMSE is in the same units as target variable.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Residual Analysis".to_string(),
                                content: "Analysis of residual patterns to assess model assumptions and identify issues.".to_string(),
                                subsections: Vec::new(),
                            },
                        ],
                    },
                    ReportSection {
                        title: "Model Fit Quality".to_string(),
                        content: "Assessment of how well the model explains variance in the target variable.".to_string(),
                        subsections: vec![
                            ReportSection {
                                title: "R-squared (R²)".to_string(),
                                content: "Coefficient of determination measuring proportion of variance explained by model.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Adjusted R-squared".to_string(),
                                content: "Modified R² that accounts for number of predictors in the model.".to_string(),
                                subsections: Vec::new(),
                            },
                        ],
                    },
                    ReportSection {
                        title: "Advanced Diagnostics".to_string(),
                        content: "Advanced regression diagnostics and robustness analysis.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Recommendations".to_string(),
                        content: "Model improvement recommendations based on performance analysis.".to_string(),
                        subsections: Vec::new(),
                    },
                ];
            }
            ReportTemplate::Clustering => {
                self.report.title = "Clustering Evaluation Report".to_string();
                self.report.abstract_text = Some(
                    "This report provides a comprehensive evaluation of clustering results using \
                     internal and external validation metrics including silhouette analysis, \
                     Davies-Bouldin index, Calinski-Harabasz index, and adjusted rand index."
                        .to_string(),
                );
                self.report.sections = vec![
                    ReportSection {
                        title: "Executive Summary".to_string(),
                        content: "High-level overview of clustering quality and key findings.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Internal Validation Metrics".to_string(),
                        content: "Metrics that evaluate clustering quality without external ground truth.".to_string(),
                        subsections: vec![
                            ReportSection {
                                title: "Silhouette Analysis".to_string(),
                                content: "Measures how well each point fits within its cluster compared to other clusters.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Davies-Bouldin Index".to_string(),
                                content: "Measures average similarity between clusters, where lower values indicate better clustering.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Calinski-Harabasz Index".to_string(),
                                content: "Variance ratio criterion measuring cluster separation and compactness.".to_string(),
                                subsections: Vec::new(),
                            },
                        ],
                    },
                    ReportSection {
                        title: "External Validation Metrics".to_string(),
                        content: "Metrics that compare clustering results with ground truth labels.".to_string(),
                        subsections: vec![
                            ReportSection {
                                title: "Adjusted Rand Index".to_string(),
                                content: "Measures similarity between clustering and ground truth, adjusted for chance.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Mutual Information".to_string(),
                                content: "Information-theoretic measure of agreement between clusterings.".to_string(),
                                subsections: Vec::new(),
                            },
                        ],
                    },
                    ReportSection {
                        title: "Cluster Stability Analysis".to_string(),
                        content: "Assessment of clustering stability across different parameter settings and data samples.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Recommendations".to_string(),
                        content: "Recommendations for optimal cluster number and algorithm parameters.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "External Validation".to_string(),
                        content: "Metrics comparing clustering results to ground truth labels."
                            .to_string(),
                        subsections: Vec::new(),
                    },
                ];
            }
            ReportTemplate::MultiObjective => {
                self.report.title = "Multi-Objective Model Comparison Report".to_string();
                self.report.abstract_text = Some(
                    "This report presents a multi-objective evaluation comparing multiple models \
                     across various performance metrics, including Pareto frontier analysis, \
                     TOPSIS ranking, and trade-off analysis."
                        .to_string(),
                );
                self.report.sections = vec![
                    ReportSection {
                        title: "Executive Summary".to_string(),
                        content: "Overview of model comparison results and recommendations."
                            .to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Pareto Frontier Analysis".to_string(),
                        content: "Analysis of non-dominated solutions in the objective space."
                            .to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "TOPSIS Ranking".to_string(),
                        content: "Multi-criteria decision analysis using TOPSIS methodology."
                            .to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Trade-off Analysis".to_string(),
                        content: "Analysis of trade-offs between competing objectives.".to_string(),
                        subsections: Vec::new(),
                    },
                ];
            }
            ReportTemplate::TimeSeries => {
                self.report.title = "Time Series Forecasting Evaluation Report".to_string();
                self.report.abstract_text = Some(
                    "This report provides comprehensive evaluation of time series forecasting models \
                     using specialized metrics such as MASE, sMAPE, directional accuracy, \
                     and seasonal decomposition analysis.".to_string()
                );
                self.report.sections = vec![
                    ReportSection {
                        title: "Executive Summary".to_string(),
                        content: "Overview of forecasting performance and key insights.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Forecast Accuracy Metrics".to_string(),
                        content: "Time series specific accuracy measures.".to_string(),
                        subsections: vec![
                            ReportSection {
                                title: "Mean Absolute Scaled Error (MASE)".to_string(),
                                content: "Scale-independent measure comparing to naive seasonal forecast.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Symmetric MAPE (sMAPE)".to_string(),
                                content: "Symmetric version of MAPE avoiding bias issues.".to_string(),
                                subsections: Vec::new(),
                            },
                            ReportSection {
                                title: "Directional Accuracy".to_string(),
                                content: "Proportion of correctly predicted direction changes.".to_string(),
                                subsections: Vec::new(),
                            },
                        ],
                    },
                    ReportSection {
                        title: "Seasonal Decomposition Analysis".to_string(),
                        content: "Analysis of seasonal patterns and trend components.".to_string(),
                        subsections: Vec::new(),
                    },
                    ReportSection {
                        title: "Forecast Diagnostics".to_string(),
                        content: "Diagnostic analysis of forecast residuals and patterns.".to_string(),
                        subsections: Vec::new(),
                    },
                ];
            }
            ReportTemplate::Custom => {
                // Custom template loaded from file - default structure
                self.report.sections = vec![ReportSection {
                    title: "Custom Analysis".to_string(),
                    content: "Custom metric analysis and results.".to_string(),
                    subsections: Vec::new(),
                }];
            }
        }
    }

    // Helper methods for creating tables
    fn create_confusion_matrix_table(&self, cm: &Array2<usize>) -> LatexTable {
        let n_classes = cm.nrows();
        let mut headers = vec!["True\\Predicted".to_string()];
        for i in 0..n_classes {
            headers.push(format!("Class {}", i));
        }

        let mut rows = Vec::new();
        for i in 0..n_classes {
            let mut row = vec![format!("Class {}", i)];
            for j in 0..n_classes {
                row.push(cm[[i, j]].to_string());
            }
            rows.push(row);
        }

        LatexTable {
            caption: "Confusion Matrix".to_string(),
            label: "tab:confusion_matrix".to_string(),
            headers,
            rows,
            alignment: format!("c|{}", "c|".repeat(n_classes)),
        }
    }

    // Interpretation methods
    fn interpret_accuracy(&self, accuracy: f64) -> String {
        match accuracy {
            x if x >= 0.95 => "Excellent performance".to_string(),
            x if x >= 0.90 => "Very good performance".to_string(),
            x if x >= 0.80 => "Good performance".to_string(),
            x if x >= 0.70 => "Moderate performance".to_string(),
            _ => "Poor performance, model needs improvement".to_string(),
        }
    }

    fn interpret_precision(&self, precision: f64) -> String {
        match precision {
            x if x >= 0.90 => "High precision, few false positives".to_string(),
            x if x >= 0.70 => "Moderate precision".to_string(),
            _ => "Low precision, many false positives".to_string(),
        }
    }

    fn interpret_recall(&self, recall: f64) -> String {
        match recall {
            x if x >= 0.90 => "High recall, few false negatives".to_string(),
            x if x >= 0.70 => "Moderate recall".to_string(),
            _ => "Low recall, many false negatives".to_string(),
        }
    }

    fn interpret_f1(&self, f1: f64) -> String {
        match f1 {
            x if x >= 0.90 => "Excellent balance between precision and recall".to_string(),
            x if x >= 0.70 => "Good balance between precision and recall".to_string(),
            _ => "Poor balance, consider adjusting model threshold".to_string(),
        }
    }

    fn interpret_mae(&self, _mae: f64) -> String {
        "Lower values indicate better performance".to_string()
    }

    fn interpret_mse(&self, _mse: f64) -> String {
        "Lower values indicate better performance".to_string()
    }

    fn interpret_rmse(&self, _rmse: f64) -> String {
        "Lower values indicate better performance, same units as target variable".to_string()
    }

    fn interpret_r2(&self, r2: f64) -> String {
        match r2 {
            x if x >= 0.90 => "Excellent model fit".to_string(),
            x if x >= 0.70 => "Good model fit".to_string(),
            x if x >= 0.50 => "Moderate model fit".to_string(),
            x if x >= 0.30 => "Weak model fit".to_string(),
            _ => "Poor model fit, consider different approach".to_string(),
        }
    }
}

impl LatexReport {
    /// Generate LaTeX code from the report
    #[cfg(all(feature = "latex", feature = "serde"))]
    pub fn to_latex(&self) -> MetricsResult<String> {
        // Direct LaTeX generation - get_document_template() already formats everything
        Ok(self.get_document_template())
    }

    /// Generate LaTeX code from the report (without template engine)
    #[cfg(all(feature = "latex", not(feature = "serde")))]
    pub fn to_latex(&self) -> MetricsResult<String> {
        // Direct LaTeX generation without template engine
        Ok(self.get_document_template())
    }

    #[cfg(not(feature = "latex"))]
    pub fn to_latex(&self) -> MetricsResult<String> {
        Err(MetricsError::InvalidInput(
            "LaTeX export requires the 'latex' feature to be enabled".to_string(),
        ))
    }

    /// Save LaTeX code to file
    pub fn save_latex<P: AsRef<Path>>(&self, path: P) -> MetricsResult<()> {
        let latex_code = self.to_latex()?;
        fs::write(path, latex_code)
            .map_err(|e| MetricsError::InvalidInput(format!("File write error: {}", e)))?;
        Ok(())
    }

    /// Compile LaTeX to PDF
    pub fn to_pdf<P: AsRef<Path>>(&self, output_path: P) -> MetricsResult<()> {
        // Ensure output directory exists
        if !self.config.output_dir.exists() {
            fs::create_dir_all(&self.config.output_dir).map_err(|e| {
                MetricsError::InvalidInput(format!("Directory creation error: {}", e))
            })?;
        }

        // Generate LaTeX file
        let tex_path = self.config.output_dir.join("report.tex");
        self.save_latex(&tex_path)?;

        // Compile with pdflatex
        let output = self.run_pdflatex(&tex_path)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MetricsError::InvalidInput(format!(
                "pdflatex failed: {}",
                stderr
            )));
        }

        // Move PDF to desired location
        let pdf_path = self.config.output_dir.join("report.pdf");
        if pdf_path.exists() {
            fs::copy(&pdf_path, output_path)
                .map_err(|e| MetricsError::InvalidInput(format!("File copy error: {}", e)))?;
        }

        // Clean up auxiliary files if requested
        if self.config.cleanup_aux_files {
            self.cleanup_latex_files()?;
        }

        Ok(())
    }

    fn get_document_template(&self) -> String {
        format!(
            r#"\documentclass[{font_size},{paper_size}]{{{document_class}}}

{packages}

{geometry}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\begin{{document}}

\maketitle

{table_of_contents}

{abstract_section}

{sections}

{metrics_section}

{tables_section}

{figures_section}

\end{{document}}
"#,
            font_size = self.config.font_size,
            paper_size = self.config.paper_size,
            document_class = self.config.document_class,
            packages = self.generate_packages(),
            geometry = r"\geometry{margin=1in}",
            title = self.title,
            author = self.author,
            date = self.date,
            table_of_contents = if self.config.table_of_contents {
                "\\tableofcontents\\newpage"
            } else {
                ""
            },
            abstract_section = self.generate_abstract_section(),
            sections = self.generate_sections(),
            metrics_section = self.generate_metrics_section(),
            tables_section = self.generate_tables_section(),
            figures_section = self.generate_figures_section(),
        )
    }

    fn generate_packages(&self) -> String {
        self.config
            .packages
            .iter()
            .map(|package| format!("\\usepackage{{{}}}", package))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_abstract_section(&self) -> String {
        match &self.abstract_text {
            Some(abstract_text) => format!(
                "\\begin{{abstract}}\n{}\n\\end{{abstract}}\n\\newpage\n",
                abstract_text
            ),
            None => String::new(),
        }
    }

    fn generate_sections(&self) -> String {
        self.sections
            .iter()
            .map(|section| {
                format!(
                    "\\section{{{}}}\n{}\n\n{}",
                    section.title,
                    section.content,
                    self.generate_subsections(&section.subsections)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_subsections(&self, subsections: &[ReportSection]) -> String {
        subsections
            .iter()
            .map(|subsection| {
                format!(
                    "\\subsection{{{}}}\n{}\n\n{}",
                    subsection.title,
                    subsection.content,
                    self.generate_subsections(&subsection.subsections)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_metrics_section(&self) -> String {
        if self.metrics.is_empty() {
            return String::new();
        }

        let mut content = String::from("\\section{Metrics Summary}\n\n");

        for metric in &self.metrics {
            content.push_str(&format!(
                "\\subsection{{{}}}\n\\textbf{{Value:}} {:.4}\\\\\n\\textbf{{Description:}} {}\\\\\n",
                metric.name, metric.value, metric.description
            ));

            if let Some(interpretation) = &metric.interpretation {
                content.push_str(&format!(
                    "\\textbf{{Interpretation:}} {}\\\\\n",
                    interpretation
                ));
            }

            if let Some((lower, upper)) = metric.confidence_interval {
                content.push_str(&format!(
                    "\\textbf{{95\\% Confidence Interval:}} [{:.4}, {:.4}]\\\\\n",
                    lower, upper
                ));
            }

            content.push('\n');
        }

        content
    }

    fn generate_tables_section(&self) -> String {
        if self.tables.is_empty() {
            return String::new();
        }

        let mut content = String::new();

        for table in &self.tables {
            content.push_str(&format!(
                "\\begin{{table}}[{}]\n\\centering\n\\begin{{tabular}}{{{}}}\n\\toprule\n",
                "htbp", table.alignment
            ));

            // Headers
            content.push_str(&table.headers.join(" & "));
            content.push_str(" \\\\\n\\midrule\n");

            // Rows
            for row in &table.rows {
                content.push_str(&row.join(" & "));
                content.push_str(" \\\\\n");
            }

            content.push_str(&format!(
                "\\bottomrule\n\\end{{tabular}}\n\\caption{{{}}}\n\\label{{{}}}\n\\end{{table}}\n\n",
                table.caption, table.label
            ));
        }

        content
    }

    fn generate_figures_section(&self) -> String {
        if self.figures.is_empty() {
            return String::new();
        }

        let mut content = String::new();

        for figure in &self.figures {
            content.push_str(&format!(
                "\\begin{{figure}}[{}]\n\\centering\n\\includegraphics[width={}]{{{}}}\n\\caption{{{}}}\n\\label{{{}}}\n\\end{{figure}}\n\n",
                figure.position, figure.width, figure.file_path, figure.caption, figure.label
            ));
        }

        content
    }

    fn run_pdflatex(&self, tex_path: &Path) -> MetricsResult<Output> {
        Command::new("pdflatex")
            .arg("-output-directory")
            .arg(&self.config.output_dir)
            .arg("-interaction=nonstopmode")
            .arg(tex_path)
            .output()
            .map_err(|e| MetricsError::InvalidInput(format!("pdflatex execution error: {}", e)))
    }

    fn cleanup_latex_files(&self) -> MetricsResult<()> {
        let extensions = &["aux", "log", "out", "toc"];

        for extension in extensions {
            let file_path = self.config.output_dir.join(format!("report.{}", extension));
            if file_path.exists() {
                fs::remove_file(file_path)
                    .map_err(|e| MetricsError::InvalidInput(format!("Cleanup error: {}", e)))?;
            }
        }

        Ok(())
    }
}

impl Default for LatexReportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_latex_report_builder() {
        let builder = LatexReportBuilder::new()
            .title("Test Report")
            .author("Test Author")
            .template(ReportTemplate::Classification);

        let report = builder.build().unwrap();
        assert_eq!(report.title, "Test Report");
        assert_eq!(report.author, "Test Author");
        assert!(!report.sections.is_empty());
    }

    #[test]
    fn test_add_classification_metrics() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 1];

        let report = LatexReportBuilder::new()
            .add_classification_metrics(&y_true.view(), &y_pred.view())
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(report.metrics.len(), 4); // accuracy, precision, recall, f1
        assert_eq!(report.tables.len(), 1); // confusion matrix
    }

    #[test]
    fn test_add_regression_metrics() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 2.1, 2.9, 3.9, 5.1];

        let report = LatexReportBuilder::new()
            .add_regression_metrics(&y_true.view(), &y_pred.view())
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(report.metrics.len(), 4); // mae, mse, rmse, r2
    }

    #[cfg(feature = "latex")]
    #[test]
    fn test_latex_generation() {
        let report = LatexReportBuilder::new()
            .title("Test Report")
            .build()
            .unwrap();

        let latex_code = report.to_latex().unwrap();
        assert!(latex_code.contains("\\documentclass"));
        assert!(latex_code.contains("Test Report"));
        assert!(latex_code.contains("\\begin{document}"));
        assert!(latex_code.contains("\\end{document}"));
    }

    #[test]
    fn test_metric_interpretations() {
        let builder = LatexReportBuilder::new();

        assert_eq!(builder.interpret_accuracy(0.95), "Excellent performance");
        assert_eq!(builder.interpret_accuracy(0.85), "Good performance");
        assert_eq!(
            builder.interpret_accuracy(0.65),
            "Poor performance, model needs improvement"
        );

        assert_eq!(builder.interpret_r2(0.95), "Excellent model fit");
        assert_eq!(
            builder.interpret_r2(0.25),
            "Poor model fit, consider different approach"
        );
    }
}
