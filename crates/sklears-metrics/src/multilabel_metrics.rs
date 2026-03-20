//! Multi-label classification metrics
//!
//! This module contains metrics specifically designed for multi-label classification
//! problems where each instance can belong to multiple classes simultaneously.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array2;

/// Multi-label exact match ratio (also called subset accuracy)
///
/// Computes the fraction of samples where all labels are correctly predicted.
/// This is the strictest multi-label metric - all labels must be predicted correctly.
///
/// # Arguments
/// * `y_true` - True binary label matrix (n_samples x n_labels)
/// * `y_pred` - Predicted binary label matrix (n_samples x n_labels)
///
/// # Returns
/// Exact match ratio
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::multilabel_metrics::multilabel_exact_match_ratio;
///
/// let y_true = array![[1, 0, 1], [0, 1, 0], [1, 1, 0]];
/// let y_pred = array![[1, 0, 1], [0, 1, 1], [1, 1, 0]];
/// let ratio = multilabel_exact_match_ratio(&y_true, &y_pred).unwrap();
/// println!("Exact match ratio: {:.3}", ratio);
/// ```
pub fn multilabel_exact_match_ratio(
    y_true: &Array2<i32>,
    y_pred: &Array2<i32>,
) -> MetricsResult<f64> {
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_pred.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.nrows();
    let mut exact_matches = 0;

    for i in 0..n_samples {
        let true_row = y_true.row(i);
        let pred_row = y_pred.row(i);

        if true_row.iter().zip(pred_row.iter()).all(|(t, p)| t == p) {
            exact_matches += 1;
        }
    }

    Ok(exact_matches as f64 / n_samples as f64)
}

/// Multi-label accuracy (also called Jaccard similarity for multi-label)
///
/// Computes the fraction of correctly predicted labels for each sample,
/// then returns the average across all samples. This considers the intersection
/// over union of true and predicted labels.
///
/// # Arguments
/// * `y_true` - True binary label matrix (n_samples x n_labels)
/// * `y_pred` - Predicted binary label matrix (n_samples x n_labels)
///
/// # Returns
/// Multi-label accuracy
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::multilabel_metrics::multilabel_accuracy_score;
///
/// let y_true = array![[1, 0, 1], [0, 1, 0]];
/// let y_pred = array![[1, 0, 0], [0, 1, 1]];
/// let accuracy = multilabel_accuracy_score(&y_true, &y_pred).unwrap();
/// println!("Multi-label accuracy: {:.3}", accuracy);
/// ```
pub fn multilabel_accuracy_score(y_true: &Array2<i32>, y_pred: &Array2<i32>) -> MetricsResult<f64> {
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_pred.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.nrows();
    let mut total_accuracy = 0.0;

    for i in 0..n_samples {
        let true_row = y_true.row(i);
        let pred_row = y_pred.row(i);

        let intersection: i32 = true_row
            .iter()
            .zip(pred_row.iter())
            .map(|(t, p)| if *t == 1 && *p == 1 { 1 } else { 0 })
            .sum();

        let union: i32 = true_row
            .iter()
            .zip(pred_row.iter())
            .map(|(t, p)| if *t == 1 || *p == 1 { 1 } else { 0 })
            .sum();

        if union > 0 {
            total_accuracy += intersection as f64 / union as f64;
        } else {
            // If both true and pred are all zeros, consider it as perfect match
            total_accuracy += 1.0;
        }
    }

    Ok(total_accuracy / n_samples as f64)
}

/// Multi-label Hamming loss
///
/// Computes the fraction of incorrectly predicted labels across all samples and labels.
/// This is the most lenient multi-label metric. Lower values are better.
///
/// # Arguments
/// * `y_true` - True binary label matrix (n_samples x n_labels)
/// * `y_pred` - Predicted binary label matrix (n_samples x n_labels)
///
/// # Returns
/// Hamming loss (0 is perfect, 1 is worst possible)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::multilabel_metrics::multilabel_hamming_loss;
///
/// let y_true = array![[1, 0, 1], [0, 1, 0]];
/// let y_pred = array![[1, 0, 0], [0, 1, 1]];
/// let loss = multilabel_hamming_loss(&y_true, &y_pred).unwrap();
/// println!("Hamming loss: {:.3}", loss);
/// ```
pub fn multilabel_hamming_loss(y_true: &Array2<i32>, y_pred: &Array2<i32>) -> MetricsResult<f64> {
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_pred.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let total_elements = y_true.len();
    let incorrect_predictions = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t != p)
        .count();

    Ok(incorrect_predictions as f64 / total_elements as f64)
}

/// Multi-label ranking loss
///
/// Computes the average number of label pairs that are incorrectly ordered
/// given the predicted scores. This metric evaluates how well the ranking
/// of labels matches the true labels.
///
/// # Arguments
/// * `y_true` - True binary label matrix (n_samples x n_labels)
/// * `y_score` - Predicted score matrix (n_samples x n_labels)
///
/// # Returns
/// Ranking loss (0 is perfect, 1 is worst possible)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::multilabel_metrics::multilabel_ranking_loss;
///
/// let y_true = array![[1, 0, 1], [0, 1, 0]];
/// let y_score = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.3]];
/// let loss = multilabel_ranking_loss(&y_true, &y_score).unwrap();
/// println!("Ranking loss: {:.3}", loss);
/// ```
pub fn multilabel_ranking_loss(y_true: &Array2<i32>, y_score: &Array2<f64>) -> MetricsResult<f64> {
    if y_true.nrows() != y_score.nrows() || y_true.ncols() != y_score.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_score.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.nrows();
    let mut total_loss = 0.0;
    let mut total_pairs = 0;

    for i in 0..n_samples {
        let true_row = y_true.row(i);
        let score_row = y_score.row(i);

        // Count relevant and irrelevant labels
        let relevant_labels: Vec<usize> = true_row
            .iter()
            .enumerate()
            .filter(|(_, &val)| val == 1)
            .map(|(idx, _)| idx)
            .collect();

        let irrelevant_labels: Vec<usize> = true_row
            .iter()
            .enumerate()
            .filter(|(_, &val)| val == 0)
            .map(|(idx, _)| idx)
            .collect();

        if relevant_labels.is_empty() || irrelevant_labels.is_empty() {
            continue;
        }

        // Count mis-ordered pairs
        let mut sample_loss = 0;
        for &rel_idx in &relevant_labels {
            for &irrel_idx in &irrelevant_labels {
                if score_row[rel_idx] <= score_row[irrel_idx] {
                    sample_loss += 1;
                }
            }
        }

        total_loss += sample_loss as f64;
        total_pairs += relevant_labels.len() * irrelevant_labels.len();
    }

    if total_pairs == 0 {
        Ok(0.0)
    } else {
        Ok(total_loss / total_pairs as f64)
    }
}

/// Average precision score for multi-label classification
///
/// Computes the average precision for each sample and returns the mean.
/// This metric evaluates the precision at each threshold and averages them.
///
/// # Arguments
/// * `y_true` - True binary label matrix (n_samples x n_labels)
/// * `y_score` - Predicted score matrix (n_samples x n_labels)
///
/// # Returns
/// Mean average precision
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::multilabel_metrics::multilabel_average_precision_score;
///
/// let y_true = array![[1, 0, 1], [0, 1, 0]];
/// let y_score = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.3]];
/// let ap = multilabel_average_precision_score(&y_true, &y_score).unwrap();
/// println!("Average precision: {:.3}", ap);
/// ```
pub fn multilabel_average_precision_score(
    y_true: &Array2<i32>,
    y_score: &Array2<f64>,
) -> MetricsResult<f64> {
    if y_true.nrows() != y_score.nrows() || y_true.ncols() != y_score.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_score.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.nrows();
    let mut total_ap = 0.0;
    let mut valid_samples = 0;

    for i in 0..n_samples {
        let true_row = y_true.row(i);
        let score_row = y_score.row(i);

        // Create pairs of (score, true_label) and sort by score descending
        let mut score_label_pairs: Vec<(f64, i32)> = score_row
            .iter()
            .zip(true_row.iter())
            .map(|(&score, &label)| (score, label))
            .collect();

        score_label_pairs
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_relevant = true_row.iter().filter(|&&x| x == 1).count();
        if n_relevant == 0 {
            continue;
        }

        let mut tp = 0;
        let mut ap = 0.0;

        for (rank, (_, true_label)) in score_label_pairs.iter().enumerate() {
            if *true_label == 1 {
                tp += 1;
                ap += tp as f64 / (rank + 1) as f64;
            }
        }

        total_ap += ap / n_relevant as f64;
        valid_samples += 1;
    }

    if valid_samples == 0 {
        Ok(0.0)
    } else {
        Ok(total_ap / valid_samples as f64)
    }
}
