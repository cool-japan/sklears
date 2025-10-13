use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Compute Area Under the Curve (AUC) using the trapezoidal rule
///
/// # Arguments
/// * `x` - Monotonically increasing x values
/// * `y` - Corresponding y values
///
/// # Returns
/// The area under the curve
pub fn auc(x: &Array1<f64>, y: &Array1<f64>) -> MetricsResult<f64> {
    if x.len() != y.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        });
    }

    if x.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "At least 2 points are required to compute AUC".to_string(),
        ));
    }

    // Check that x is monotonically increasing
    for i in 1..x.len() {
        if x[i] < x[i - 1] {
            return Err(MetricsError::InvalidInput(
                "x values must be monotonically increasing".to_string(),
            ));
        }
    }

    // Compute area using trapezoidal rule
    let mut area = 0.0;
    for i in 1..x.len() {
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }

    Ok(area)
}

/// Compute average precision score
///
/// Average precision summarizes a precision-recall curve as the weighted mean of precisions
/// achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
///
/// # Arguments
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probability estimates of the positive class)
///
/// # Returns
/// Average precision score
pub fn average_precision_score(y_true: &Array1<i32>, y_score: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check if y_true is binary
    let unique_labels: Vec<i32> = {
        let mut labels = y_true.to_vec();
        labels.sort_unstable();
        labels.dedup();
        labels
    };

    if unique_labels.len() != 2 || !unique_labels.contains(&0) || !unique_labels.contains(&1) {
        return Err(MetricsError::InvalidInput(
            "y_true must be binary (containing only 0 and 1)".to_string(),
        ));
    }

    let (precision, recall, _) = precision_recall_curve(y_true, y_score)?;

    // Compute average precision as area under precision-recall curve
    // Note: recall values are in increasing order
    let mut ap = 0.0;
    for i in 1..precision.len() {
        let recall_diff = recall[i] - recall[i - 1];
        if recall_diff > 0.0 {
            ap += recall_diff * precision[i];
        }
    }

    Ok(ap)
}

/// Coverage error
///
/// The coverage error is the average number of labels that have to be included
/// in the final prediction such that all true labels are predicted.
///
/// # Arguments
/// * `y_true` - True binary labels (n_samples, n_labels)
/// * `y_score` - Target scores (n_samples, n_labels)
///
/// # Returns
/// Coverage error
pub fn coverage_error(y_true: &Array2<i32>, y_score: &Array2<f64>) -> MetricsResult<f64> {
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_score.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    let mut coverage_sum = 0.0;

    for i in 0..n_samples {
        let true_labels = y_true.row(i);
        let scores = y_score.row(i);

        // Get indices sorted by scores in descending order
        let mut indices: Vec<usize> = (0..n_labels).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));

        // Find the maximum rank among true labels
        let mut max_rank = 0;
        for (rank, &idx) in indices.iter().enumerate() {
            if true_labels[idx] == 1 {
                max_rank = rank + 1; // 1-indexed
            }
        }

        coverage_sum += max_rank as f64;
    }

    Ok(coverage_sum / n_samples as f64)
}

/// Discounted Cumulative Gain (DCG) at rank k
///
/// # Arguments
/// * `y_true` - True relevance scores
/// * `y_score` - Predicted scores
/// * `k` - Rank position (None for all positions)
///
/// # Returns
/// DCG score
pub fn dcg_score(
    y_true: &Array1<f64>,
    y_score: &Array1<f64>,
    k: Option<usize>,
) -> MetricsResult<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Get indices sorted by predicted scores in descending order
    let mut indices: Vec<usize> = (0..y_score.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(Ordering::Equal)
    });

    let k = k.unwrap_or(indices.len()).min(indices.len());
    let mut dcg = 0.0;

    for i in 0..k {
        let relevance = y_true[indices[i]];
        let discount = (i + 2) as f64; // i+2 because log2(1) = 0
        dcg += (2f64.powf(relevance) - 1.0) / discount.log2();
    }

    Ok(dcg)
}

/// Normalized Discounted Cumulative Gain (NDCG) at rank k
///
/// # Arguments
/// * `y_true` - True relevance scores
/// * `y_score` - Predicted scores
/// * `k` - Rank position (None for all positions)
///
/// # Returns
/// NDCG score
pub fn ndcg_score(
    y_true: &Array1<f64>,
    y_score: &Array1<f64>,
    k: Option<usize>,
) -> MetricsResult<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Compute DCG
    let dcg = dcg_score(y_true, y_score, k)?;

    // Compute ideal DCG (using perfect ranking)
    let ideal_dcg = dcg_score(y_true, y_true, k)?;

    if ideal_dcg == 0.0 {
        return Ok(0.0);
    }

    Ok(dcg / ideal_dcg)
}

/// Compute ROC curve
///
/// # Arguments
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probability estimates of the positive class)
///
/// # Returns
/// * `fpr` - False positive rates
/// * `tpr` - True positive rates
/// * `thresholds` - Thresholds used to compute fpr and tpr
pub fn roc_curve(
    y_true: &Array1<i32>,
    y_score: &Array1<f64>,
) -> MetricsResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check if y_true is binary
    let unique_labels: Vec<i32> = {
        let mut labels = y_true.to_vec();
        labels.sort_unstable();
        labels.dedup();
        labels
    };

    if unique_labels.len() != 2 || !unique_labels.contains(&0) || !unique_labels.contains(&1) {
        return Err(MetricsError::InvalidInput(
            "y_true must be binary (containing only 0 and 1)".to_string(),
        ));
    }

    // Sort by scores in descending order
    let mut indices: Vec<usize> = (0..y_score.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(Ordering::Equal)
    });

    // Count positives and negatives
    let n_pos = y_true.iter().filter(|&&x| x == 1).count();
    let n_neg = y_true.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Err(MetricsError::InvalidInput(
            "Both positive and negative samples are required".to_string(),
        ));
    }

    let mut tpr_vec = vec![0.0];
    let mut fpr_vec = vec![0.0];
    let mut thresholds_vec = vec![y_score[indices[0]] + 1.0];

    let mut tp = 0;
    let mut fp = 0;
    let mut prev_score = y_score[indices[0]] + 1.0;

    for &idx in &indices {
        if y_score[idx] != prev_score {
            tpr_vec.push(tp as f64 / n_pos as f64);
            fpr_vec.push(fp as f64 / n_neg as f64);
            thresholds_vec.push(y_score[idx]);
            prev_score = y_score[idx];
        }

        if y_true[idx] == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    // Add final point
    tpr_vec.push(1.0);
    fpr_vec.push(1.0);
    thresholds_vec.push(y_score[indices[indices.len() - 1]] - 1.0);

    Ok((
        Array1::from_vec(fpr_vec),
        Array1::from_vec(tpr_vec),
        Array1::from_vec(thresholds_vec),
    ))
}

/// Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
///
/// # Arguments
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probability estimates of the positive class)
///
/// # Returns
/// ROC AUC score
pub fn roc_auc_score(y_true: &Array1<i32>, y_score: &Array1<f64>) -> MetricsResult<f64> {
    let (fpr, tpr, _) = roc_curve(y_true, y_score)?;
    auc(&fpr, &tpr)
}

/// Compute precision-recall curve
///
/// # Arguments
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probability estimates of the positive class)
///
/// # Returns
/// * `precision` - Precision values
/// * `recall` - Recall values
/// * `thresholds` - Thresholds used to compute precision and recall
pub fn precision_recall_curve(
    y_true: &Array1<i32>,
    y_score: &Array1<f64>,
) -> MetricsResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check if y_true is binary
    let unique_labels: Vec<i32> = {
        let mut labels = y_true.to_vec();
        labels.sort_unstable();
        labels.dedup();
        labels
    };

    if unique_labels.len() != 2 || !unique_labels.contains(&0) || !unique_labels.contains(&1) {
        return Err(MetricsError::InvalidInput(
            "y_true must be binary (containing only 0 and 1)".to_string(),
        ));
    }

    // Sort by scores in descending order
    let mut indices: Vec<usize> = (0..y_score.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(Ordering::Equal)
    });

    // Count positives
    let n_pos = y_true.iter().filter(|&&x| x == 1).count();

    if n_pos == 0 {
        return Err(MetricsError::InvalidInput(
            "No positive samples found".to_string(),
        ));
    }

    let mut precision_vec = Vec::new();
    let mut recall_vec = Vec::new();
    let mut thresholds_vec = Vec::new();

    let mut tp = 0;
    let mut fp = 0;
    let mut prev_score = y_score[indices[0]] + 1.0;

    for (i, &idx) in indices.iter().enumerate() {
        if y_score[idx] != prev_score && i > 0 {
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                1.0
            };
            let recall = tp as f64 / n_pos as f64;

            precision_vec.push(precision);
            recall_vec.push(recall);
            thresholds_vec.push(y_score[idx]);
            prev_score = y_score[idx];
        }

        if y_true[idx] == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    // Add final point
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        1.0
    };
    let recall = tp as f64 / n_pos as f64;
    precision_vec.push(precision);
    recall_vec.push(recall);

    // Add boundary point
    recall_vec.push(0.0);
    precision_vec.push(1.0);

    // Reverse to have recall in increasing order
    precision_vec.reverse();
    recall_vec.reverse();

    Ok((
        Array1::from_vec(precision_vec),
        Array1::from_vec(recall_vec),
        Array1::from_vec(thresholds_vec),
    ))
}

/// Multi-class averaging strategies for metrics
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Average {
    /// Calculate metrics for each label and return their unweighted mean
    Macro,
    /// Calculate metrics globally by counting total true positives, false negatives etc
    Micro,
    /// Calculate metrics for each label and return their weighted mean by support
    Weighted,
    /// Return None (used for binary classification)
    None,
}

/// Compute multi-class ROC AUC score using one-vs-rest approach
///
/// # Arguments
/// * `y_true` - True class labels (0, 1, 2, ...)
/// * `y_score` - Prediction probabilities for each class (n_samples x n_classes)
/// * `average` - Averaging strategy ('macro', 'weighted', or None for per-class scores)
/// * `multi_class` - Strategy for multi-class ('ovr' for one-vs-rest, 'ovo' for one-vs-one)
///
/// # Returns
/// ROC AUC score(s)
pub fn roc_auc_score_multiclass<T: PartialEq + Copy + Ord + std::hash::Hash>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
    average: Option<Average>,
    multi_class: &str,
) -> MetricsResult<f64> {
    if y_true.len() != y_score.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_score.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if multi_class != "ovr" && multi_class != "ovo" {
        return Err(MetricsError::InvalidParameter(
            "multi_class must be 'ovr' or 'ovo'".to_string(),
        ));
    }

    // Get unique classes
    let mut classes = BTreeSet::new();
    for &label in y_true.iter() {
        classes.insert(label);
    }
    let classes: Vec<T> = classes.into_iter().collect();
    let n_classes = classes.len();

    if n_classes < 2 {
        return Err(MetricsError::InvalidInput(
            "At least 2 classes are required".to_string(),
        ));
    }

    // Binary case
    if n_classes == 2 {
        if y_score.ncols() == 1 {
            // Single column, use as scores for positive class
            let y_binary: Array1<i32> = y_true
                .iter()
                .map(|&label| if label == classes[1] { 1 } else { 0 })
                .collect();
            return roc_auc_score(&y_binary, &y_score.column(0).to_owned());
        } else if y_score.ncols() == 2 {
            // Two columns, use second column as scores for positive class
            let y_binary: Array1<i32> = y_true
                .iter()
                .map(|&label| if label == classes[1] { 1 } else { 0 })
                .collect();
            return roc_auc_score(&y_binary, &y_score.column(1).to_owned());
        } else {
            return Err(MetricsError::InvalidInput(
                "For binary classification, y_score should have 1 or 2 columns".to_string(),
            ));
        }
    }

    if y_score.ncols() != n_classes {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len(), n_classes],
            actual: vec![y_score.nrows(), y_score.ncols()],
        });
    }

    match multi_class {
        "ovr" => roc_auc_ovr(y_true, y_score, &classes, average),
        "ovo" => roc_auc_ovo(y_true, y_score, &classes, average),
        _ => unreachable!(),
    }
}

/// One-vs-Rest ROC AUC computation
fn roc_auc_ovr<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
    classes: &[T],
    average: Option<Average>,
) -> MetricsResult<f64> {
    let mut class_scores = Vec::new();
    let mut class_counts = Vec::new();

    for (i, &class) in classes.iter().enumerate() {
        // Create binary labels (current class vs all others)
        let y_binary: Array1<i32> = y_true
            .iter()
            .map(|&label| if label == class { 1 } else { 0 })
            .collect();

        // Count samples for this class
        let class_count = y_binary.iter().filter(|&&x| x == 1).count();

        // Skip classes with no positive samples
        if class_count == 0 || class_count == y_true.len() {
            continue;
        }

        let score_column = y_score.column(i).to_owned();
        let auc_score = roc_auc_score(&y_binary, &score_column)?;

        class_scores.push(auc_score);
        class_counts.push(class_count);
    }

    if class_scores.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No valid classes for ROC AUC computation".to_string(),
        ));
    }

    // Apply averaging
    match average.unwrap_or(Average::Macro) {
        Average::Macro => {
            let sum: f64 = class_scores.iter().sum();
            Ok(sum / class_scores.len() as f64)
        }
        Average::Weighted => {
            let total_weight: usize = class_counts.iter().sum();
            let weighted_sum: f64 = class_scores
                .iter()
                .zip(class_counts.iter())
                .map(|(score, count)| score * (*count as f64))
                .sum();
            Ok(weighted_sum / total_weight as f64)
        }
        Average::Micro => {
            // For micro averaging in multiclass OvR, we concatenate all binary problems
            // This is complex and rarely used, so we default to macro for now
            let sum: f64 = class_scores.iter().sum();
            Ok(sum / class_scores.len() as f64)
        }
        Average::None => {
            // Return mean for now (would typically return array for each class)
            let sum: f64 = class_scores.iter().sum();
            Ok(sum / class_scores.len() as f64)
        }
    }
}

/// One-vs-One ROC AUC computation
fn roc_auc_ovo<T: PartialEq + Copy + Ord>(
    y_true: &Array1<T>,
    y_score: &Array2<f64>,
    classes: &[T],
    average: Option<Average>,
) -> MetricsResult<f64> {
    let n_classes = classes.len();
    let mut pairwise_scores = Vec::new();
    let mut pairwise_counts = Vec::new();

    for i in 0..n_classes {
        for j in (i + 1)..n_classes {
            let class_i = classes[i];
            let class_j = classes[j];

            // Find samples that belong to either class i or class j
            let mut binary_indices = Vec::new();
            let mut y_binary_vec = Vec::new();
            let mut score_diff_vec = Vec::new();

            for (idx, &label) in y_true.iter().enumerate() {
                if label == class_i {
                    binary_indices.push(idx);
                    y_binary_vec.push(0); // class i = 0
                    score_diff_vec.push(y_score[[idx, i]] - y_score[[idx, j]]);
                } else if label == class_j {
                    binary_indices.push(idx);
                    y_binary_vec.push(1); // class j = 1
                    score_diff_vec.push(y_score[[idx, i]] - y_score[[idx, j]]);
                }
            }

            if y_binary_vec.is_empty()
                || y_binary_vec.iter().all(|&x| x == 0)
                || y_binary_vec.iter().all(|&x| x == 1)
            {
                continue;
            }

            let y_binary = Array1::from_vec(y_binary_vec);
            let score_diff = Array1::from_vec(score_diff_vec);

            let auc_score = roc_auc_score(&y_binary, &score_diff)?;
            pairwise_scores.push(auc_score);
            pairwise_counts.push(binary_indices.len());
        }
    }

    if pairwise_scores.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No valid class pairs for OvO ROC AUC computation".to_string(),
        ));
    }

    // Apply averaging
    match average.unwrap_or(Average::Macro) {
        Average::Macro => {
            let sum: f64 = pairwise_scores.iter().sum();
            Ok(sum / pairwise_scores.len() as f64)
        }
        Average::Weighted => {
            let total_weight: usize = pairwise_counts.iter().sum();
            let weighted_sum: f64 = pairwise_scores
                .iter()
                .zip(pairwise_counts.iter())
                .map(|(score, count)| score * (*count as f64))
                .sum();
            Ok(weighted_sum / total_weight as f64)
        }
        Average::Micro | Average::None => {
            // Same as macro for OvO
            let sum: f64 = pairwise_scores.iter().sum();
            Ok(sum / pairwise_scores.len() as f64)
        }
    }
}

/// Compute Area Under the Precision-Recall Curve (PR AUC)
///
/// # Arguments
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probability estimates of the positive class)
///
/// # Returns
/// PR AUC score
pub fn precision_recall_auc_score(
    y_true: &Array1<i32>,
    y_score: &Array1<f64>,
) -> MetricsResult<f64> {
    let (precision, recall, _) = precision_recall_curve(y_true, y_score)?;

    // Sort by recall to ensure monotonic increasing order
    let mut recall_precision_pairs: Vec<_> = recall.iter().zip(precision.iter()).collect();
    recall_precision_pairs
        .sort_by(|a, b| a.0.partial_cmp(b.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_recall: Array1<f64> = recall_precision_pairs.iter().map(|(r, _)| **r).collect();
    let sorted_precision: Array1<f64> = recall_precision_pairs.iter().map(|(_, p)| **p).collect();

    auc(&sorted_recall, &sorted_precision)
}

/// Mean Average Precision (MAP)
///
/// Computes the mean average precision for multi-label ranking or information retrieval.
/// For each query/sample, calculates the average precision and then takes the mean.
///
/// # Arguments
///
/// * `y_true` - True binary labels (2D array where each row is a sample)
/// * `y_score` - Prediction scores (2D array where each row is a sample)
///
/// # Returns
///
/// Mean average precision score
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::ranking::mean_average_precision;
///
/// let y_true = array![[1, 0, 1], [0, 1, 1]];
/// let y_score = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.7]];
/// let map_score = mean_average_precision(&y_true, &y_score).unwrap();
/// println!("MAP: {:.4}", map_score);
/// ```
pub fn mean_average_precision(y_true: &Array2<i32>, y_score: &Array2<f64>) -> MetricsResult<f64> {
    if y_true.shape() != y_score.shape() {
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

    for i in 0..n_samples {
        let y_true_row = y_true.row(i);
        let y_score_row = y_score.row(i);

        // Convert to 1D arrays
        let y_true_1d = Array1::from_iter(y_true_row.iter().cloned());
        let y_score_1d = Array1::from_iter(y_score_row.iter().cloned());

        // Validate binary labels
        for &label in y_true_1d.iter() {
            if label != 0 && label != 1 {
                return Err(MetricsError::InvalidInput(
                    "y_true must contain only 0 and 1".to_string(),
                ));
            }
        }

        // Create pairs of (score, label) and sort by score in descending order
        let mut score_label_pairs: Vec<(f64, i32)> = y_score_1d
            .iter()
            .zip(y_true_1d.iter())
            .map(|(&score, &label)| (score, label))
            .collect();

        score_label_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Calculate average precision for this sample
        let mut relevant_count = 0;
        let mut precision_sum = 0.0;
        let total_relevant: i32 = y_true_1d.iter().sum();

        if total_relevant == 0 {
            // No relevant items, AP = 0
            continue;
        }

        for (rank, (_score, label)) in score_label_pairs.iter().enumerate() {
            if *label == 1 {
                relevant_count += 1;
                let precision_at_k = relevant_count as f64 / (rank + 1) as f64;
                precision_sum += precision_at_k;
            }
        }

        let average_precision = precision_sum / total_relevant as f64;
        total_ap += average_precision;
    }

    Ok(total_ap / n_samples as f64)
}

/// Mean Reciprocal Rank (MRR)
///
/// Computes the mean reciprocal rank for ranking evaluation.
/// For each query/sample, finds the rank of the first relevant item and takes the reciprocal.
///
/// # Arguments
///
/// * `y_true` - True binary labels (2D array where each row is a sample)
/// * `y_score` - Prediction scores (2D array where each row is a sample)
///
/// # Returns
///
/// Mean reciprocal rank score
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::ranking::mean_reciprocal_rank;
///
/// let y_true = array![[1, 0, 0], [0, 1, 0]];
/// let y_score = array![[0.9, 0.1, 0.2], [0.2, 0.9, 0.1]];
/// let mrr_score = mean_reciprocal_rank(&y_true, &y_score).unwrap();
/// println!("MRR: {:.4}", mrr_score);
/// ```
pub fn mean_reciprocal_rank(y_true: &Array2<i32>, y_score: &Array2<f64>) -> MetricsResult<f64> {
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: y_true.shape().to_vec(),
            actual: y_score.shape().to_vec(),
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.nrows();
    let mut total_rr = 0.0;

    for i in 0..n_samples {
        let y_true_row = y_true.row(i);
        let y_score_row = y_score.row(i);

        // Convert to 1D arrays
        let y_true_1d = Array1::from_iter(y_true_row.iter().cloned());
        let y_score_1d = Array1::from_iter(y_score_row.iter().cloned());

        // Validate binary labels
        for &label in y_true_1d.iter() {
            if label != 0 && label != 1 {
                return Err(MetricsError::InvalidInput(
                    "y_true must contain only 0 and 1".to_string(),
                ));
            }
        }

        // Create pairs of (score, label, original_index) and sort by score in descending order
        let mut score_label_pairs: Vec<(f64, i32, usize)> = y_score_1d
            .iter()
            .zip(y_true_1d.iter())
            .enumerate()
            .map(|(idx, (&score, &label))| (score, label, idx))
            .collect();

        score_label_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Find the rank of the first relevant item
        let mut reciprocal_rank = 0.0;
        for (rank, (_score, label, _idx)) in score_label_pairs.iter().enumerate() {
            if *label == 1 {
                reciprocal_rank = 1.0 / (rank + 1) as f64;
                break;
            }
        }

        total_rr += reciprocal_rank;
    }

    Ok(total_rr / n_samples as f64)
}

// =============================================================================
// Cost-Sensitive Evaluation Metrics
// =============================================================================

/// Cost matrix for cost-sensitive classification
///
/// Represents the cost of misclassifying class i as class j.
/// cost_matrix[i][j] is the cost of predicting class j when true class is i.
#[derive(Debug, Clone)]
pub struct CostMatrix {
    /// Cost matrix where element [i,j] is cost of predicting j when true class is i
    pub costs: Array2<f64>,
    /// Class labels corresponding to matrix indices
    pub classes: Vec<i32>,
}

impl CostMatrix {
    /// Create a new cost matrix
    ///
    /// # Arguments
    /// * `costs` - Cost matrix (n_classes x n_classes)
    /// * `classes` - Class labels in order
    pub fn new(costs: Array2<f64>, classes: Vec<i32>) -> MetricsResult<Self> {
        let n_classes = classes.len();
        if costs.shape() != [n_classes, n_classes] {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![n_classes, n_classes],
                actual: costs.shape().to_vec(),
            });
        }

        // Ensure diagonal is zero (no cost for correct classification)
        for i in 0..n_classes {
            if costs[[i, i]] != 0.0 {
                return Err(MetricsError::InvalidInput(
                    "Diagonal elements of cost matrix should be 0".to_string(),
                ));
            }
        }

        Ok(Self { costs, classes })
    }

    /// Create a uniform cost matrix where all misclassifications have the same cost
    pub fn uniform(classes: Vec<i32>, misclassification_cost: f64) -> Self {
        let n_classes = classes.len();
        let mut costs = Array2::from_elem((n_classes, n_classes), misclassification_cost);

        // Set diagonal to zero
        for i in 0..n_classes {
            costs[[i, i]] = 0.0;
        }

        Self { costs, classes }
    }

    /// Get the cost of predicting `predicted` when true class is `actual`
    pub fn get_cost(&self, actual: i32, predicted: i32) -> MetricsResult<f64> {
        let actual_idx = self
            .classes
            .iter()
            .position(|&x| x == actual)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Class {} not found", actual)))?;
        let predicted_idx = self
            .classes
            .iter()
            .position(|&x| x == predicted)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Class {} not found", predicted)))?;

        Ok(self.costs[[actual_idx, predicted_idx]])
    }
}

/// Compute cost-sensitive accuracy
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `cost_matrix` - Cost matrix for misclassifications
///
/// # Returns
/// Cost-sensitive accuracy (1 - normalized total cost)
pub fn cost_sensitive_accuracy(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    cost_matrix: &CostMatrix,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let total_cost = total_cost(y_true, y_pred, cost_matrix)?;
    let max_cost = maximum_cost(y_true, cost_matrix)?;

    if max_cost == 0.0 {
        return Ok(1.0); // All classifications are correct
    }

    Ok(1.0 - (total_cost / max_cost))
}

/// Compute total cost of misclassifications
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `cost_matrix` - Cost matrix for misclassifications
///
/// # Returns
/// Total cost of all misclassifications
pub fn total_cost(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    cost_matrix: &CostMatrix,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut total = 0.0;
    for (actual, predicted) in y_true.iter().zip(y_pred.iter()) {
        total += cost_matrix.get_cost(*actual, *predicted)?;
    }

    Ok(total)
}

/// Compute average cost per sample
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `cost_matrix` - Cost matrix for misclassifications
///
/// # Returns
/// Average cost per sample
pub fn average_cost(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    cost_matrix: &CostMatrix,
) -> MetricsResult<f64> {
    let total = total_cost(y_true, y_pred, cost_matrix)?;
    Ok(total / y_true.len() as f64)
}

/// Compute maximum possible cost (worst case scenario)
///
/// # Arguments
/// * `y_true` - True class labels
/// * `cost_matrix` - Cost matrix for misclassifications
///
/// # Returns
/// Maximum possible total cost
fn maximum_cost(y_true: &Array1<i32>, cost_matrix: &CostMatrix) -> MetricsResult<f64> {
    let mut max_total = 0.0;

    for &actual in y_true.iter() {
        let actual_idx = cost_matrix
            .classes
            .iter()
            .position(|&x| x == actual)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Class {} not found", actual)))?;

        // Find the maximum cost for misclassifying this true class
        let row = cost_matrix.costs.row(actual_idx);
        let max_cost_for_class = row
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(&0.0);

        max_total += max_cost_for_class;
    }

    Ok(max_total)
}

/// Compute cost-sensitive confusion matrix
///
/// Returns a confusion matrix where each cell contains the total cost
/// of those misclassifications, not just the count.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted class labels
/// * `cost_matrix` - Cost matrix for misclassifications
///
/// # Returns
/// Cost-weighted confusion matrix
pub fn cost_confusion_matrix(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    cost_matrix: &CostMatrix,
) -> MetricsResult<Array2<f64>> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_classes = cost_matrix.classes.len();
    let mut cost_cm = Array2::zeros((n_classes, n_classes));

    for (actual, predicted) in y_true.iter().zip(y_pred.iter()) {
        let actual_idx = cost_matrix
            .classes
            .iter()
            .position(|&x| x == *actual)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Class {} not found", actual)))?;
        let predicted_idx = cost_matrix
            .classes
            .iter()
            .position(|&x| x == *predicted)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Class {} not found", predicted)))?;

        let cost = cost_matrix.costs[[actual_idx, predicted_idx]];
        cost_cm[[actual_idx, predicted_idx]] += cost;
    }

    Ok(cost_cm)
}

/// Compute cost curve for different classification thresholds
///
/// Cost curves plot the normalized expected cost vs. probability threshold
/// for binary classification with different cost ratios.
///
/// # Arguments
/// * `y_true` - True binary labels (0, 1)
/// * `y_prob` - Predicted probabilities for positive class
/// * `cost_fp` - Cost of false positive
/// * `cost_fn` - Cost of false negative
/// * `n_thresholds` - Number of thresholds to evaluate
///
/// # Returns
/// (thresholds, costs) - Arrays of thresholds and corresponding normalized costs
pub fn cost_curve(
    y_true: &Array1<i32>,
    y_prob: &Array1<f64>,
    cost_fp: f64,
    cost_fn: f64,
    n_thresholds: Option<usize>,
) -> MetricsResult<(Array1<f64>, Array1<f64>)> {
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_prob.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate binary labels
    for &label in y_true.iter() {
        if label != 0 && label != 1 {
            return Err(MetricsError::InvalidInput(
                "y_true must contain only 0 and 1 for binary classification".to_string(),
            ));
        }
    }

    let n_thresholds = n_thresholds.unwrap_or(100);
    let mut thresholds = Vec::with_capacity(n_thresholds);
    let mut costs = Vec::with_capacity(n_thresholds);

    // Generate thresholds from 0 to 1
    for i in 0..n_thresholds {
        let threshold = i as f64 / (n_thresholds - 1) as f64;
        thresholds.push(threshold);

        // Make predictions based on threshold
        let y_pred: Array1<i32> = y_prob
            .iter()
            .map(|&prob| if prob >= threshold { 1 } else { 0 })
            .collect();

        // Count false positives and false negatives
        let mut fp = 0;
        let mut fn_count = 0;

        for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
            match (true_label, pred_label) {
                (0, 1) => fp += 1,       // False positive
                (1, 0) => fn_count += 1, // False negative
                _ => {}                  // True positive or true negative
            }
        }

        // Calculate normalized cost
        let total_cost = (fp as f64 * cost_fp) + (fn_count as f64 * cost_fn);
        let max_possible_cost = (y_true.iter().filter(|&&x| x == 0).count() as f64 * cost_fp)
            + (y_true.iter().filter(|&&x| x == 1).count() as f64 * cost_fn);

        let normalized_cost = if max_possible_cost > 0.0 {
            total_cost / max_possible_cost
        } else {
            0.0
        };

        costs.push(normalized_cost);
    }

    Ok((Array1::from_vec(thresholds), Array1::from_vec(costs)))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_auc() {
        let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let area = auc(&x, &y).unwrap();
        assert!((area - 0.5).abs() < 1e-6);

        // Test rectangle
        let x = array![0.0, 1.0];
        let y = array![1.0, 1.0];
        let area = auc(&x, &y).unwrap();
        assert!((area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_roc_auc_score() {
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.4, 0.35, 0.8];
        let score = roc_auc_score(&y_true, &y_score).unwrap();
        assert!((score - 0.75).abs() < 1e-6);

        // Perfect classifier
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.0, 0.0, 1.0, 1.0];
        let score = roc_auc_score(&y_true, &y_score).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_precision_score() {
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.4, 0.35, 0.8];
        let score = average_precision_score(&y_true, &y_score).unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_dcg_ndcg_score() {
        let y_true = array![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
        let y_score = array![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let dcg = dcg_score(&y_true, &y_score, None).unwrap();
        assert!(dcg > 0.0);

        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!(ndcg > 0.0 && ndcg <= 1.0);

        // Test with k parameter
        let dcg_k3 = dcg_score(&y_true, &y_score, Some(3)).unwrap();
        assert!(dcg_k3 > 0.0 && dcg_k3 <= dcg);
    }

    #[test]
    fn test_coverage_error() {
        let y_true = array![[1, 0, 0], [0, 1, 1], [0, 0, 1]];
        let y_score = array![[0.9, 0.1, 0.2], [0.1, 0.9, 0.8], [0.2, 0.3, 0.9]];

        let error = coverage_error(&y_true, &y_score).unwrap();
        assert!(error >= 1.0);
    }

    #[test]
    fn test_precision_recall_curve() {
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.4, 0.35, 0.8];

        let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();

        // Check dimensions
        assert_eq!(precision.len(), recall.len());
        // Thresholds are shorter because of the added boundary point
        assert!(thresholds.len() < precision.len());

        // Check boundary conditions
        assert!((recall[0] - 0.0).abs() < 1e-6);
        assert!((precision[precision.len() - 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_roc_curve() {
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.4, 0.35, 0.8];

        let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score).unwrap();

        // Check dimensions
        assert_eq!(fpr.len(), tpr.len());
        assert_eq!(fpr.len(), thresholds.len());

        // Check boundary conditions
        assert!((fpr[0] - 0.0).abs() < 1e-6);
        assert!((tpr[0] - 0.0).abs() < 1e-6);
        assert!((fpr[fpr.len() - 1] - 1.0).abs() < 1e-6);
        assert!((tpr[tpr.len() - 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_error_cases() {
        // Test shape mismatch
        let y_true = array![0, 1, 1];
        let y_score = array![0.5, 0.7];
        assert!(roc_auc_score(&y_true, &y_score).is_err());

        // Test empty input
        let y_true = array![];
        let y_score = array![];
        assert!(roc_auc_score(&y_true, &y_score).is_err());

        // Test non-binary labels
        let y_true = array![0, 1, 2];
        let y_score = array![0.5, 0.7, 0.9];
        assert!(roc_auc_score(&y_true, &y_score).is_err());

        // Test non-monotonic x for AUC
        let x = array![0.0, 0.5, 0.3, 1.0];
        let y = array![0.0, 0.5, 0.3, 1.0];
        assert!(auc(&x, &y).is_err());
    }

    #[test]
    fn test_precision_recall_auc_score() {
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.4, 0.35, 0.8];
        let score = precision_recall_auc_score(&y_true, &y_score).unwrap();
        assert!(score > 0.0 && score <= 1.0);

        // Good classifier
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.2, 0.8, 0.9];
        let score = precision_recall_auc_score(&y_true, &y_score).unwrap();
        // Good classifiers should have high precision-recall AUC
        assert!(score > 0.5 && score <= 1.0);
    }

    #[test]
    fn test_roc_auc_score_multiclass() {
        // Multi-class test case
        let y_true = array![0, 0, 1, 1, 2, 2];
        let y_score = array![
            [0.8, 0.1, 0.1], // class 0
            [0.7, 0.2, 0.1], // class 0
            [0.1, 0.8, 0.1], // class 1
            [0.2, 0.7, 0.1], // class 1
            [0.1, 0.1, 0.8], // class 2
            [0.1, 0.2, 0.7]  // class 2
        ];

        // Test one-vs-rest with macro averaging
        let score_ovr_macro =
            roc_auc_score_multiclass(&y_true, &y_score, Some(Average::Macro), "ovr").unwrap();
        assert!(score_ovr_macro > 0.0 && score_ovr_macro <= 1.0);

        // Test one-vs-rest with weighted averaging
        let score_ovr_weighted =
            roc_auc_score_multiclass(&y_true, &y_score, Some(Average::Weighted), "ovr").unwrap();
        assert!(score_ovr_weighted > 0.0 && score_ovr_weighted <= 1.0);

        // Test one-vs-one
        let score_ovo =
            roc_auc_score_multiclass(&y_true, &y_score, Some(Average::Macro), "ovo").unwrap();
        assert!(score_ovo >= 0.0 && score_ovo <= 1.0);

        // Binary case with 2 columns
        let y_true_binary = array![0, 0, 1, 1];
        let y_score_binary = array![[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]];
        let score_binary =
            roc_auc_score_multiclass(&y_true_binary, &y_score_binary, Some(Average::Macro), "ovr")
                .unwrap();
        assert!(score_binary > 0.0 && score_binary <= 1.0);

        // Binary case with 1 column
        let y_score_binary_1col = array![[0.2], [0.3], [0.7], [0.8]];
        let score_binary_1col = roc_auc_score_multiclass(
            &y_true_binary,
            &y_score_binary_1col,
            Some(Average::Macro),
            "ovr",
        )
        .unwrap();
        assert!(score_binary_1col > 0.0 && score_binary_1col <= 1.0);
    }

    #[test]
    fn test_roc_auc_multiclass_errors() {
        let y_true = array![0, 1, 2];
        let y_score = array![[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]];

        // Test invalid multi_class parameter
        assert!(
            roc_auc_score_multiclass(&y_true, &y_score, Some(Average::Macro), "invalid").is_err()
        );

        // Test shape mismatch
        let y_score_wrong = array![[0.5, 0.5], [0.3, 0.7]];
        assert!(
            roc_auc_score_multiclass(&y_true, &y_score_wrong, Some(Average::Macro), "ovr").is_err()
        );

        // Test insufficient classes
        let y_true_single = array![0, 0, 0];
        let y_score_single = array![[1.0], [1.0], [1.0]];
        assert!(roc_auc_score_multiclass(
            &y_true_single,
            &y_score_single,
            Some(Average::Macro),
            "ovr"
        )
        .is_err());
    }

    #[test]
    fn test_mean_average_precision() {
        // Test case with 2 samples, 3 items each
        let y_true = array![[1, 0, 1], [0, 1, 1]];
        let y_score = array![[0.9, 0.1, 0.8], [0.2, 0.9, 0.7]];

        let map_score = mean_average_precision(&y_true, &y_score).unwrap();
        assert!(map_score >= 0.0 && map_score <= 1.0);

        // Perfect ranking should give MAP = 1.0
        let y_true_perfect = array![[1, 1, 0], [1, 1, 0]];
        let y_score_perfect = array![[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]];
        let map_perfect = mean_average_precision(&y_true_perfect, &y_score_perfect).unwrap();
        assert!((map_perfect - 1.0).abs() < 1e-6);

        // Worst ranking should give low MAP
        let y_score_worst = array![[0.1, 0.2, 0.9], [0.1, 0.2, 0.9]];
        let map_worst = mean_average_precision(&y_true_perfect, &y_score_worst).unwrap();
        assert!(map_worst < map_perfect);

        // Test error cases
        let y_true_wrong_shape = array![[1, 0], [0, 1]];
        assert!(mean_average_precision(&y_true_wrong_shape, &y_score).is_err());

        // Test invalid labels
        let y_true_invalid = array![[1, 2, 0]];
        let y_score_single = array![[0.5, 0.3, 0.2]];
        assert!(mean_average_precision(&y_true_invalid, &y_score_single).is_err());

        // Test empty input
        let y_true_empty = Array2::zeros((0, 0));
        let y_score_empty = Array2::zeros((0, 0));
        assert!(mean_average_precision(&y_true_empty, &y_score_empty).is_err());
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        // Test case where first item is relevant
        let y_true = array![[1, 0, 0], [0, 1, 0]];
        let y_score = array![[0.9, 0.1, 0.2], [0.2, 0.9, 0.1]];

        let mrr_score = mean_reciprocal_rank(&y_true, &y_score).unwrap();
        assert!((mrr_score - 1.0).abs() < 1e-6); // Should be 1.0 since relevant items are ranked first

        // Test case where relevant item is second
        let y_true_second = array![[1, 0, 0]];
        let y_score_second = array![[0.5, 0.9, 0.1]]; // relevant item (index 0) ranked second
        let mrr_second = mean_reciprocal_rank(&y_true_second, &y_score_second).unwrap();
        assert!((mrr_second - 0.5).abs() < 1e-6); // 1/2 = 0.5

        // Test case where relevant item is third
        let y_true_third = array![[1, 0, 0]];
        let y_score_third = array![[0.1, 0.9, 0.8]]; // relevant item (index 0) ranked third
        let mrr_third = mean_reciprocal_rank(&y_true_third, &y_score_third).unwrap();
        assert!((mrr_third - 1.0 / 3.0).abs() < 1e-6); // 1/3

        // Test case with no relevant items
        let y_true_none = array![[0, 0, 0]];
        let y_score_none = array![[0.9, 0.5, 0.1]];
        let mrr_none = mean_reciprocal_rank(&y_true_none, &y_score_none).unwrap();
        assert_eq!(mrr_none, 0.0);

        // Test multiple samples
        let y_true_multi = array![[1, 0, 0], [0, 0, 1]];
        let y_score_multi = array![[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]];
        let mrr_multi = mean_reciprocal_rank(&y_true_multi, &y_score_multi).unwrap();
        assert!((mrr_multi - 1.0).abs() < 1e-6); // Both relevant items ranked first: (1 + 1) / 2 = 1.0

        // Test error cases
        let y_true_wrong_shape = array![[1, 0], [0, 1]];
        assert!(mean_reciprocal_rank(&y_true_wrong_shape, &y_score).is_err());

        // Test invalid labels
        let y_true_invalid = array![[1, 2, 0]];
        let y_score_single = array![[0.5, 0.3, 0.2]];
        assert!(mean_reciprocal_rank(&y_true_invalid, &y_score_single).is_err());

        // Test empty input
        let y_true_empty = Array2::zeros((0, 0));
        let y_score_empty = Array2::zeros((0, 0));
        assert!(mean_reciprocal_rank(&y_true_empty, &y_score_empty).is_err());
    }

    #[test]
    fn test_cost_matrix() {
        // Test uniform cost matrix
        let classes = vec![0, 1, 2];
        let cost_matrix = CostMatrix::uniform(classes.clone(), 1.0);

        assert_eq!(cost_matrix.get_cost(0, 0).unwrap(), 0.0); // Correct classification
        assert_eq!(cost_matrix.get_cost(0, 1).unwrap(), 1.0); // Misclassification
        assert_eq!(cost_matrix.get_cost(1, 2).unwrap(), 1.0); // Misclassification

        // Test custom cost matrix
        let costs = array![[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0]];
        let custom_cost_matrix = CostMatrix::new(costs, classes).unwrap();

        assert_eq!(custom_cost_matrix.get_cost(0, 1).unwrap(), 2.0);
        assert_eq!(custom_cost_matrix.get_cost(1, 2).unwrap(), 3.0);
        assert_eq!(custom_cost_matrix.get_cost(2, 0).unwrap(), 2.0);

        // Test error cases
        let invalid_costs = array![[1.0, 2.0], [1.0, 0.0]]; // Non-zero diagonal
        assert!(CostMatrix::new(invalid_costs, vec![0, 1]).is_err());

        let wrong_shape = array![[0.0, 1.0]]; // Wrong shape
        assert!(CostMatrix::new(wrong_shape, vec![0, 1]).is_err());
    }

    #[test]
    fn test_cost_sensitive_accuracy() {
        let y_true = array![0, 0, 1, 1, 2, 2];
        let y_pred = array![0, 1, 1, 2, 2, 0]; // Some misclassifications

        // Uniform cost matrix
        let cost_matrix = CostMatrix::uniform(vec![0, 1, 2], 1.0);
        let accuracy = cost_sensitive_accuracy(&y_true, &y_pred, &cost_matrix).unwrap();

        // Perfect predictions would have accuracy 1.0, worst would have accuracy 0.0
        assert!(accuracy >= 0.0 && accuracy <= 1.0);

        // Test perfect predictions
        let y_pred_perfect = array![0, 0, 1, 1, 2, 2];
        let perfect_accuracy =
            cost_sensitive_accuracy(&y_true, &y_pred_perfect, &cost_matrix).unwrap();
        assert_eq!(perfect_accuracy, 1.0);

        // Test custom cost matrix with higher penalty for certain misclassifications
        let costs = array![[0.0, 5.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let custom_cost_matrix = CostMatrix::new(costs, vec![0, 1, 2]).unwrap();
        let custom_accuracy =
            cost_sensitive_accuracy(&y_true, &y_pred, &custom_cost_matrix).unwrap();

        // Should be different due to different costs
        assert!(custom_accuracy >= 0.0 && custom_accuracy <= 1.0);
    }

    #[test]
    fn test_total_cost() {
        let y_true = array![0, 0, 1, 1];
        let y_pred = array![0, 1, 1, 0]; // 2 misclassifications

        let cost_matrix = CostMatrix::uniform(vec![0, 1], 2.0);
        let total = total_cost(&y_true, &y_pred, &cost_matrix).unwrap();
        assert_eq!(total, 4.0); // 2 misclassifications * 2.0 cost each

        let avg = average_cost(&y_true, &y_pred, &cost_matrix).unwrap();
        assert_eq!(avg, 1.0); // 4.0 total cost / 4 samples
    }

    #[test]
    fn test_cost_confusion_matrix() {
        let y_true = array![0, 0, 1, 1];
        let y_pred = array![0, 1, 1, 0]; // 2 correct, 2 misclassifications

        let costs = array![[0.0, 3.0], [2.0, 0.0]]; // Different costs for different misclassifications
        let cost_matrix = CostMatrix::new(costs, vec![0, 1]).unwrap();

        let cost_cm = cost_confusion_matrix(&y_true, &y_pred, &cost_matrix).unwrap();

        // cost_cm[i][j] = total cost of predicting j when true class is i
        assert_eq!(cost_cm[[0, 0]], 0.0); // 1 correct prediction of class 0, cost = 0
        assert_eq!(cost_cm[[0, 1]], 3.0); // 1 misclassification: true=0, pred=1, cost = 3.0
        assert_eq!(cost_cm[[1, 0]], 2.0); // 1 misclassification: true=1, pred=0, cost = 2.0
        assert_eq!(cost_cm[[1, 1]], 0.0); // 1 correct prediction of class 1, cost = 0
    }

    #[test]
    fn test_cost_curve() {
        let y_true = array![0, 0, 1, 1];
        let y_prob = array![0.1, 0.4, 0.6, 0.9];

        let cost_fp = 1.0; // Cost of false positive
        let cost_fn = 2.0; // Cost of false negative (higher penalty)

        let (thresholds, costs) = cost_curve(&y_true, &y_prob, cost_fp, cost_fn, Some(11)).unwrap();

        assert_eq!(thresholds.len(), 11);
        assert_eq!(costs.len(), 11);

        // All costs should be normalized between 0 and 1
        for &cost in costs.iter() {
            assert!(cost >= 0.0 && cost <= 1.0);
        }

        // Check boundary conditions
        assert_eq!(thresholds[0], 0.0);
        assert_eq!(thresholds[10], 1.0);
    }

    #[test]
    fn test_cost_sensitive_errors() {
        let y_true = array![0, 1];
        let y_pred = array![0]; // Wrong length
        let cost_matrix = CostMatrix::uniform(vec![0, 1], 1.0);

        assert!(cost_sensitive_accuracy(&y_true, &y_pred, &cost_matrix).is_err());
        assert!(total_cost(&y_true, &y_pred, &cost_matrix).is_err());
        assert!(cost_confusion_matrix(&y_true, &y_pred, &cost_matrix).is_err());

        // Test empty inputs
        let y_true_empty = array![];
        let y_pred_empty = array![];
        assert!(cost_sensitive_accuracy(&y_true_empty, &y_pred_empty, &cost_matrix).is_err());

        // Test cost curve with non-binary labels
        let y_true_multiclass = array![0, 1, 2];
        let y_prob = array![0.1, 0.5, 0.9];
        assert!(cost_curve(&y_true_multiclass, &y_prob, 1.0, 1.0, None).is_err());

        // Test unknown class in cost matrix
        let y_true_unknown = array![0, 3]; // Class 3 not in cost matrix
        let y_pred_unknown = array![0, 1];
        assert!(cost_sensitive_accuracy(&y_true_unknown, &y_pred_unknown, &cost_matrix).is_err());
    }
}
