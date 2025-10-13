use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Calculate classification score based on metric name
pub fn calculate_classification_score(
    predictions: &Array1<Int>,
    y_true: &Array1<Int>,
    metric: &str,
) -> Result<Float> {
    if predictions.len() != y_true.len() {
        return Err(SklearsError::InvalidInput(
            "Predictions and true values must have the same length".to_string(),
        ));
    }

    match metric.to_lowercase().as_str() {
        "accuracy" => Ok(accuracy_score(predictions, y_true)),
        "precision" => precision_score(predictions, y_true),
        "recall" => recall_score(predictions, y_true),
        "f1" => f1_score(predictions, y_true),
        "balanced_accuracy" => balanced_accuracy_score(predictions, y_true),
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown classification metric: {}",
            metric
        ))),
    }
}

/// Calculate regression score based on metric name
pub fn calculate_regression_score(
    predictions: &Array1<Float>,
    y_true: &Array1<Float>,
    metric: &str,
) -> Result<Float> {
    if predictions.len() != y_true.len() {
        return Err(SklearsError::InvalidInput(
            "Predictions and true values must have the same length".to_string(),
        ));
    }

    match metric.to_lowercase().as_str() {
        "neg_mean_squared_error" | "mse" => Ok(-mean_squared_error(predictions, y_true)),
        "neg_mean_absolute_error" | "mae" => Ok(-mean_absolute_error(predictions, y_true)),
        "r2" => r2_score(predictions, y_true),
        "neg_root_mean_squared_error" | "rmse" => Ok(-root_mean_squared_error(predictions, y_true)),
        "explained_variance" => explained_variance_score(predictions, y_true),
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown regression metric: {}",
            metric
        ))),
    }
}

/// Calculate accuracy score
pub fn accuracy_score(predictions: &Array1<Int>, y_true: &Array1<Int>) -> Float {
    let correct = predictions
        .iter()
        .zip(y_true.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count();
    correct as Float / predictions.len() as Float
}

/// Calculate precision score (macro average for multiclass)
pub fn precision_score(predictions: &Array1<Int>, y_true: &Array1<Int>) -> Result<Float> {
    let confusion_matrix = create_confusion_matrix(predictions, y_true);
    let mut classes: Vec<Int> = confusion_matrix.keys().map(|(actual, _)| *actual).collect();
    classes.sort();
    classes.dedup();

    if classes.len() <= 1 {
        return Ok(0.0);
    }

    let mut precisions = Vec::new();

    for &class in &classes {
        let tp = confusion_matrix.get(&(class, class)).copied().unwrap_or(0) as Float;

        let mut fp = 0.0;
        for &other_class in &classes {
            if other_class != class {
                fp += confusion_matrix
                    .get(&(other_class, class))
                    .copied()
                    .unwrap_or(0) as Float;
            }
        }

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        precisions.push(precision);
    }

    Ok(precisions.iter().sum::<Float>() / precisions.len() as Float)
}

/// Calculate recall score (macro average for multiclass)
pub fn recall_score(predictions: &Array1<Int>, y_true: &Array1<Int>) -> Result<Float> {
    let confusion_matrix = create_confusion_matrix(predictions, y_true);
    let mut classes: Vec<Int> = confusion_matrix.keys().map(|(actual, _)| *actual).collect();
    classes.sort();
    classes.dedup();

    if classes.len() <= 1 {
        return Ok(0.0);
    }

    let mut recalls = Vec::new();

    for &class in &classes {
        let tp = confusion_matrix.get(&(class, class)).copied().unwrap_or(0) as Float;

        let mut fn_count = 0.0;
        for &other_class in &classes {
            if other_class != class {
                fn_count += confusion_matrix
                    .get(&(class, other_class))
                    .copied()
                    .unwrap_or(0) as Float;
            }
        }

        let recall = if tp + fn_count > 0.0 {
            tp / (tp + fn_count)
        } else {
            0.0
        };
        recalls.push(recall);
    }

    Ok(recalls.iter().sum::<Float>() / recalls.len() as Float)
}

/// Calculate F1 score
pub fn f1_score(predictions: &Array1<Int>, y_true: &Array1<Int>) -> Result<Float> {
    let precision = precision_score(predictions, y_true)?;
    let recall = recall_score(predictions, y_true)?;

    if precision + recall > 0.0 {
        Ok(2.0 * precision * recall / (precision + recall))
    } else {
        Ok(0.0)
    }
}

/// Calculate balanced accuracy score
pub fn balanced_accuracy_score(predictions: &Array1<Int>, y_true: &Array1<Int>) -> Result<Float> {
    recall_score(predictions, y_true)
}

/// Create confusion matrix as a HashMap
fn create_confusion_matrix(
    predictions: &Array1<Int>,
    y_true: &Array1<Int>,
) -> HashMap<(Int, Int), usize> {
    let mut matrix = HashMap::new();

    for (&pred, &actual) in predictions.iter().zip(y_true.iter()) {
        *matrix.entry((actual, pred)).or_insert(0) += 1;
    }

    matrix
}

/// Calculate mean squared error
pub fn mean_squared_error(predictions: &Array1<Float>, y_true: &Array1<Float>) -> Float {
    predictions
        .iter()
        .zip(y_true.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<Float>()
        / predictions.len() as Float
}

/// Calculate mean absolute error
pub fn mean_absolute_error(predictions: &Array1<Float>, y_true: &Array1<Float>) -> Float {
    predictions
        .iter()
        .zip(y_true.iter())
        .map(|(&pred, &actual)| (pred - actual).abs())
        .sum::<Float>()
        / predictions.len() as Float
}

/// Calculate root mean squared error
pub fn root_mean_squared_error(predictions: &Array1<Float>, y_true: &Array1<Float>) -> Float {
    mean_squared_error(predictions, y_true).sqrt()
}

/// Calculate R² score
pub fn r2_score(predictions: &Array1<Float>, y_true: &Array1<Float>) -> Result<Float> {
    let y_mean = y_true.iter().sum::<Float>() / y_true.len() as Float;

    let ss_res = predictions
        .iter()
        .zip(y_true.iter())
        .map(|(&pred, &actual)| (actual - pred).powi(2))
        .sum::<Float>();

    let ss_tot = y_true
        .iter()
        .map(|&actual| (actual - y_mean).powi(2))
        .sum::<Float>();

    if ss_tot == 0.0 {
        // Handle the case where y is constant
        if ss_res == 0.0 {
            Ok(1.0) // Perfect prediction
        } else {
            Ok(0.0) // Imperfect prediction of constant target
        }
    } else {
        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Calculate explained variance score
pub fn explained_variance_score(
    predictions: &Array1<Float>,
    y_true: &Array1<Float>,
) -> Result<Float> {
    let y_mean = y_true.iter().sum::<Float>() / y_true.len() as Float;
    let pred_mean = predictions.iter().sum::<Float>() / predictions.len() as Float;

    let var_y = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum::<Float>() / y_true.len() as Float;

    let var_residual = predictions
        .iter()
        .zip(y_true.iter())
        .map(|(&pred, &actual)| (actual - pred).powi(2))
        .sum::<Float>()
        / predictions.len() as Float;

    if var_y == 0.0 {
        Ok(0.0)
    } else {
        Ok(1.0 - var_residual / var_y)
    }
}

/// Calculate custom scoring function
pub fn custom_score<F>(predictions: &Array1<Float>, y_true: &Array1<Float>, scorer: F) -> Float
where
    F: Fn(Float, Float) -> Float,
{
    predictions
        .iter()
        .zip(y_true.iter())
        .map(|(&pred, &actual)| scorer(pred, actual))
        .sum::<Float>()
        / predictions.len() as Float
}

/// Classification metrics summary
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// accuracy
    pub accuracy: Float,
    /// precision
    pub precision: Float,
    /// recall
    pub recall: Float,
    /// f1
    pub f1: Float,
    /// balanced_accuracy
    pub balanced_accuracy: Float,
    /// confusion_matrix
    pub confusion_matrix: HashMap<(Int, Int), usize>,
}

impl ClassificationMetrics {
    pub fn compute(predictions: &Array1<Int>, y_true: &Array1<Int>) -> Result<Self> {
        Ok(Self {
            accuracy: accuracy_score(predictions, y_true),
            precision: precision_score(predictions, y_true)?,
            recall: recall_score(predictions, y_true)?,
            f1: f1_score(predictions, y_true)?,
            balanced_accuracy: balanced_accuracy_score(predictions, y_true)?,
            confusion_matrix: create_confusion_matrix(predictions, y_true),
        })
    }

    pub fn print_summary(&self) {
        println!("Classification Metrics:");
        println!("  Accuracy: {:.4}", self.accuracy);
        println!("  Precision: {:.4}", self.precision);
        println!("  Recall: {:.4}", self.recall);
        println!("  F1 Score: {:.4}", self.f1);
        println!("  Balanced Accuracy: {:.4}", self.balanced_accuracy);
    }
}

/// Regression metrics summary
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    /// mse
    pub mse: Float,
    /// mae
    pub mae: Float,
    /// rmse
    pub rmse: Float,
    /// r2
    pub r2: Float,
    /// explained_variance
    pub explained_variance: Float,
}

impl RegressionMetrics {
    pub fn compute(predictions: &Array1<Float>, y_true: &Array1<Float>) -> Result<Self> {
        Ok(Self {
            mse: mean_squared_error(predictions, y_true),
            mae: mean_absolute_error(predictions, y_true),
            rmse: root_mean_squared_error(predictions, y_true),
            r2: r2_score(predictions, y_true)?,
            explained_variance: explained_variance_score(predictions, y_true)?,
        })
    }

    pub fn print_summary(&self) {
        println!("Regression Metrics:");
        println!("  MSE: {:.4}", self.mse);
        println!("  MAE: {:.4}", self.mae);
        println!("  RMSE: {:.4}", self.rmse);
        println!("  R²: {:.4}", self.r2);
        println!("  Explained Variance: {:.4}", self.explained_variance);
    }
}

/// Calculate log loss for classification
pub fn log_loss(y_true: &Array1<Int>, y_prob: &Array2<Float>) -> Result<Float> {
    if y_true.len() != y_prob.nrows() {
        return Err(SklearsError::InvalidInput(
            "Number of samples must match".to_string(),
        ));
    }

    let eps = 1e-15; // Small epsilon to avoid log(0)
    let mut loss = 0.0;

    for (i, &true_class) in y_true.iter().enumerate() {
        let prob = y_prob[(i, true_class as usize)].max(eps).min(1.0 - eps);
        loss -= prob.ln();
    }

    Ok(loss / y_true.len() as Float)
}

/// Calculate area under ROC curve (simplified binary case)
pub fn roc_auc_score(y_true: &Array1<Int>, y_scores: &Array1<Float>) -> Result<Float> {
    if y_true.len() != y_scores.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Count positive and negative samples
    let n_pos = y_true.iter().filter(|&&y| y == 1).count();
    let n_neg = y_true.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Ok(0.5); // No discrimination possible
    }

    // Create pairs and sort by score
    let mut pairs: Vec<(Float, Int)> = y_scores
        .iter()
        .zip(y_true.iter())
        .map(|(&score, &label)| (score, label))
        .collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate AUC using trapezoidal rule approximation
    let mut auc = 0.0;
    let mut tp = 0;
    let mut fp = 0;

    for &(_, label) in &pairs {
        if label == 1 {
            tp += 1;
        } else {
            fp += 1;
            auc += tp as Float; // Add area of rectangle
        }
    }

    Ok(auc / (n_pos * n_neg) as Float)
}

/// Calculate mean absolute percentage error
pub fn mean_absolute_percentage_error(
    predictions: &Array1<Float>,
    y_true: &Array1<Float>,
) -> Float {
    let mut mape = 0.0;
    let mut valid_count = 0;

    for (&pred, &actual) in predictions.iter().zip(y_true.iter()) {
        if actual != 0.0 {
            mape += ((actual - pred) / actual).abs();
            valid_count += 1;
        }
    }

    if valid_count > 0 {
        100.0 * mape / valid_count as Float
    } else {
        0.0
    }
}

/// Calculate median absolute error
pub fn median_absolute_error(predictions: &Array1<Float>, y_true: &Array1<Float>) -> Float {
    let mut errors: Vec<Float> = predictions
        .iter()
        .zip(y_true.iter())
        .map(|(&pred, &actual)| (pred - actual).abs())
        .collect();

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if errors.is_empty() {
        0.0
    } else if errors.len() % 2 == 0 {
        let mid = errors.len() / 2;
        (errors[mid - 1] + errors[mid]) / 2.0
    } else {
        errors[errors.len() / 2]
    }
}
