//! Demonstration of advanced sklears-metrics features
//!
//! This example showcases:
//! 1. LaTeX report generation with charts (ROC curves, bar charts, scatter plots)
//! 2. Distributed metrics computation with different strategies
//! 3. Display utilities for ROC/PR/DET curves
//!
//! Run with: `cargo run --example advanced_features_demo --features latex,parallel`

use scirs2_core::ndarray::{array, Array1};
use sklears_metrics::{classification, regression, MetricsResult};

#[cfg(feature = "latex")]
use sklears_metrics::latex_export::{LatexReportBuilder, ReportTemplate};

#[cfg(all(feature = "parallel", not(target_os = "macos")))]
use sklears_metrics::distributed_metrics::{ComputeStrategy, DistributedMetricsComputer};

fn main() -> MetricsResult<()> {
    println!("=== Advanced Features Demo ===\n");

    // Example classification data (multiclass)
    let y_true_class = array![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
    let y_pred_class = array![0, 2, 1, 0, 0, 1, 0, 1, 2, 1];

    // Binary classification data for ROC/PR/DET curves
    #[allow(unused_variables)]
    let y_true_binary = array![0, 1, 0, 0, 1, 1, 0, 1, 0, 1];
    #[allow(unused_variables)]
    let y_score = array![0.9, 0.3, 0.6, 0.85, 0.7, 0.5, 0.95, 0.8, 0.4, 0.3];

    // Example regression data
    let y_true_reg = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_pred_reg = array![1.1, 2.1, 2.9, 3.9, 5.1, 5.9, 7.2, 7.8, 9.1, 10.2];

    // Demo 1: Classification metrics
    demo_classification_metrics(&y_true_class, &y_pred_class)?;

    // Demo 2: Regression metrics
    demo_regression_metrics(&y_true_reg, &y_pred_reg)?;

    // Demo 3: Display utilities (ROC/PR/DET curves)
    #[cfg(feature = "latex")]
    demo_display_utilities(&y_true_binary, &y_score)?;

    // Demo 4: LaTeX report generation with charts
    #[cfg(feature = "latex")]
    demo_latex_reports(&y_true_class, &y_pred_class, &y_true_reg, &y_pred_reg)?;

    // Demo 5: Distributed metrics computation
    #[cfg(all(feature = "parallel", not(target_os = "macos")))]
    demo_distributed_metrics(&y_true_class, &y_pred_class, &y_true_reg, &y_pred_reg)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_classification_metrics(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> MetricsResult<()> {
    println!("1. Classification Metrics:");
    println!(
        "   - Accuracy: {:.4}",
        classification::accuracy_score(y_true, y_pred)?
    );

    let unique_labels: Vec<i32> = y_true
        .iter()
        .chain(y_pred.iter())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for &label in &unique_labels {
        let prec = classification::precision_score(y_true, y_pred, Some(label))?;
        let rec = classification::recall_score(y_true, y_pred, Some(label))?;
        let f1 = classification::f1_score(y_true, y_pred, Some(label))?;
        println!(
            "   - Class {}: Precision={:.4}, Recall={:.4}, F1={:.4}",
            label, prec, rec, f1
        );
    }
    println!();
    Ok(())
}

fn demo_regression_metrics(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<()> {
    println!("2. Regression Metrics:");
    println!(
        "   - MAE: {:.4}",
        regression::mean_absolute_error(y_true, y_pred)?
    );
    println!(
        "   - MSE: {:.4}",
        regression::mean_squared_error(y_true, y_pred)?
    );
    println!(
        "   - RMSE: {:.4}",
        regression::root_mean_squared_error(y_true, y_pred)?
    );
    println!("   - R²: {:.4}", regression::r2_score(y_true, y_pred)?);
    println!();
    Ok(())
}

#[cfg(feature = "latex")]
fn demo_display_utilities(y_true: &Array1<i32>, y_score: &Array1<f64>) -> MetricsResult<()> {
    use sklears_metrics::classification::{
        DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    };

    println!("3. Display Utilities:");

    // ROC Curve
    let roc_display =
        RocCurveDisplay::from_predictions(y_true, y_score, None, Some("Model A".to_string()))?;
    println!(
        "   - ROC Curve created with AUC: {:.4}",
        roc_display.roc_auc.unwrap_or(0.0)
    );

    // Precision-Recall Curve
    let pr_display = PrecisionRecallDisplay::from_predictions(
        y_true,
        y_score,
        None,
        Some("Model A".to_string()),
    )?;
    println!(
        "   - PR Curve created with AP: {:.4}",
        pr_display.average_precision.unwrap_or(0.0)
    );

    // DET Curve
    let _det_display =
        DetCurveDisplay::from_predictions(y_true, y_score, None, Some("Model A".to_string()))?;
    println!("   - DET Curve created");
    println!();
    Ok(())
}

#[cfg(feature = "latex")]
fn demo_latex_reports(
    y_true_class: &Array1<i32>,
    y_pred_class: &Array1<i32>,
    y_true_reg: &Array1<f64>,
    y_pred_reg: &Array1<f64>,
) -> MetricsResult<()> {
    println!("4. LaTeX Report Generation:");

    // Create classification report
    let class_report = LatexReportBuilder::new()
        .template(ReportTemplate::Classification)
        .title("Classification Model Performance Report")
        .author("Advanced Metrics Demo")
        .add_classification_metrics(&y_true_class.view(), &y_pred_class.view())?
        // Add ROC curve chart
        .add_roc_curve(
            vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vec![0.0, 0.5, 0.7, 0.85, 0.95, 1.0],
            0.82,
        )
        // Add metrics comparison bar chart
        .add_metrics_bar_chart(
            vec![
                "Accuracy".to_string(),
                "Precision".to_string(),
                "Recall".to_string(),
                "F1".to_string(),
            ],
            vec![0.7, 0.68, 0.72, 0.70],
            "Score".to_string(),
        )
        .build()?;

    println!(
        "   - Classification report created with {} metrics",
        class_report.metrics.len()
    );
    println!(
        "   - {} tables, {} charts",
        class_report.tables.len(),
        class_report.charts.len()
    );

    // Create regression report
    let reg_report = LatexReportBuilder::new()
        .template(ReportTemplate::Regression)
        .title("Regression Model Performance Report")
        .author("Advanced Metrics Demo")
        .add_regression_metrics(&y_true_reg.view(), &y_pred_reg.view())?
        // Add scatter plot with trend line
        .add_scatter_plot(
            y_true_reg.to_vec(),
            y_pred_reg.to_vec(),
            "True Values".to_string(),
            "Predicted Values".to_string(),
            true, // Show trend line
        )
        .build()?;

    println!(
        "   - Regression report created with {} metrics",
        reg_report.metrics.len()
    );
    println!("   - {} charts", reg_report.charts.len());

    // Generate LaTeX code (feature-gated)
    #[cfg(feature = "latex")]
    {
        let latex_code = class_report.to_latex()?;
        println!(
            "   - LaTeX code generated ({} characters)",
            latex_code.len()
        );

        // Uncomment to save reports:
        // class_report.save_latex("classification_report.tex")?;
        // reg_report.save_latex("regression_report.tex")?;
        // class_report.to_pdf("classification_report.pdf")?;  // Requires pdflatex
    }

    println!();
    Ok(())
}

#[cfg(all(feature = "parallel", not(target_os = "macos")))]
fn demo_distributed_metrics(
    y_true_class: &Array1<i32>,
    y_pred_class: &Array1<i32>,
    y_true_reg: &Array1<f64>,
    y_pred_reg: &Array1<f64>,
) -> MetricsResult<()> {
    println!("5. Distributed Metrics Computation:");

    // Data Parallel Strategy
    println!("   a) Data Parallel Strategy:");
    let results = DistributedMetricsComputer::new()?
        .strategy(ComputeStrategy::DataParallel)
        .compute_classification_metrics(&y_true_class.view(), &y_pred_class.view())?;
    println!("      - Computation time: {:?}", results.computation_time);
    println!(
        "      - Load balance efficiency: {:.2}%",
        results.load_balance_efficiency * 100.0
    );
    println!(
        "      - Accuracy: {:.4}",
        results.metrics.get("accuracy").unwrap_or(&0.0)
    );

    // Model Parallel Strategy
    println!("   b) Model Parallel Strategy:");
    let results = DistributedMetricsComputer::new()?
        .strategy(ComputeStrategy::ModelParallel)
        .compute_classification_metrics(&y_true_class.view(), &y_pred_class.view())?;
    println!("      - Computation time: {:?}", results.computation_time);
    println!("      - Metrics computed: {}", results.metrics.len());
    for (name, value) in &results.metrics {
        println!("        * {}: {:.4}", name, value);
    }

    // Pipeline Parallel Strategy
    println!("   c) Pipeline Parallel Strategy:");
    let results = DistributedMetricsComputer::new()?
        .strategy(ComputeStrategy::PipelineParallel)
        .compute_classification_metrics(&y_true_class.view(), &y_pred_class.view())?;
    println!("      - Computation time: {:?}", results.computation_time);
    println!(
        "      - Load balance efficiency: {:.2}%",
        results.load_balance_efficiency * 100.0
    );

    // Hybrid Strategy
    println!("   d) Hybrid Strategy (Data + Model Parallel):");
    let results = DistributedMetricsComputer::new()?
        .strategy(ComputeStrategy::Hybrid)
        .compute_classification_metrics(&y_true_class.view(), &y_pred_class.view())?;
    println!("      - Computation time: {:?}", results.computation_time);
    println!("      - Metrics computed: {}", results.metrics.len());

    // Regression with different strategies
    println!("   e) Regression with Data Parallel:");
    let results = DistributedMetricsComputer::new()?
        .strategy(ComputeStrategy::DataParallel)
        .compute_regression_metrics(&y_true_reg.view(), &y_pred_reg.view())?;
    println!("      - Computation time: {:?}", results.computation_time);
    println!(
        "      - MAE: {:.4}",
        results.metrics.get("mae").unwrap_or(&f64::NAN)
    );
    println!(
        "      - R²: {:.4}",
        results.metrics.get("r2").unwrap_or(&f64::NAN)
    );

    println!();
    Ok(())
}
