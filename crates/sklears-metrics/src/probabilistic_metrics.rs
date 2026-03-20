//! Probabilistic classification metrics
//!
//! This module contains metrics that evaluate the quality of probabilistic predictions
//! including calibration metrics, scoring rules, and divergences.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeSet, HashMap};

/// Type alias for reliability diagram output: (bin_edges, mean_predicted, mean_observed, counts)
type ReliabilityDiagramOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

/// Logarithmic loss (cross-entropy loss) for probabilistic predictions
///
/// Computes the cross-entropy loss for multi-class classification problems.
/// This is one of the most common probabilistic scoring rules.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_proba` - Predicted class probabilities (n_samples x n_classes)
/// * `eps` - Small value to clip probabilities to avoid log(0)
///
/// # Returns
/// Logarithmic loss (lower is better)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use sklears_metrics::probabilistic_metrics::log_loss;
///
/// let y_true = array![0, 1, 0];
/// let y_proba = Array2::from_shape_vec((3, 2), vec![
///     0.9, 0.1,  // Sample 0: P(class=0)=0.9, P(class=1)=0.1
///     0.2, 0.8,  // Sample 1: P(class=0)=0.2, P(class=1)=0.8
///     0.7, 0.3   // Sample 2: P(class=0)=0.7, P(class=1)=0.3
/// ]).unwrap();
/// let loss = log_loss(&y_true, &y_proba, None).unwrap();
/// println!("Log loss: {:.4}", loss);
/// ```
pub fn log_loss(
    y_true: &Array1<i32>,
    y_proba: &Array2<f64>,
    eps: Option<f64>,
) -> MetricsResult<f64> {
    let eps = eps.unwrap_or(1e-15);

    if y_true.len() != y_proba.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let n_samples = y_true.len() as f64;
    let n_classes = y_proba.ncols();

    // Clip probabilities to avoid log(0)
    let clipped = y_proba.mapv(|p| p.max(eps).min(1.0 - eps));

    let mut loss = 0.0;
    for (i, &true_label) in y_true.iter().enumerate() {
        if true_label < 0 || true_label >= n_classes as i32 {
            return Err(MetricsError::InvalidParameter(format!(
                "Label {true_label} out of range [0, {n_classes})"
            )));
        }
        loss -= clipped[[i, true_label as usize]].ln();
    }

    Ok(loss / n_samples)
}

/// Brier score loss for probability estimates
///
/// The Brier score measures the accuracy of probabilistic predictions for binary
/// classification problems. It is the mean squared difference between predicted
/// probabilities and the actual outcomes.
///
/// # Arguments
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_proba` - Predicted probabilities for the positive class
/// * `pos_label` - Positive class label (default: 1)
///
/// # Returns
/// Brier score loss (lower is better)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::probabilistic_metrics::brier_score_loss;
///
/// let y_true = array![0, 1, 1, 0];
/// let y_proba = array![0.1, 0.9, 0.8, 0.2];
/// let loss = brier_score_loss(&y_true, &y_proba, None).unwrap();
/// println!("Brier score: {:.4}", loss);
/// ```
pub fn brier_score_loss(
    y_true: &Array1<i32>,
    y_proba: &Array1<f64>,
    pos_label: Option<i32>,
) -> MetricsResult<f64> {
    if y_true.len() != y_proba.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check that probabilities are in [0, 1]
    if y_proba.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err(MetricsError::InvalidParameter(
            "Probabilities must be in [0, 1]".to_string(),
        ));
    }

    let pos_label = pos_label.unwrap_or(1);

    let loss = y_true
        .iter()
        .zip(y_proba.iter())
        .map(|(&y, &p)| {
            let y_binary = if y == pos_label { 1.0 } else { 0.0 };
            (p - y_binary).powi(2)
        })
        .sum::<f64>();

    Ok(loss / y_true.len() as f64)
}

/// Expected Calibration Error (ECE)
///
/// Measures the difference between predicted confidences and actual accuracies.
/// The prediction space is divided into bins, and the ECE is the weighted average
/// of the absolute difference between confidence and accuracy in each bin.
///
/// # Arguments
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_proba` - Predicted probabilities for the positive class
/// * `n_bins` - Number of bins to divide the probability range [0, 1]
///
/// # Returns
/// Expected calibration error (lower is better)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::probabilistic_metrics::expected_calibration_error;
///
/// let y_true = array![1, 0, 1, 0];
/// let y_proba = array![0.9, 0.1, 0.8, 0.2];
/// let ece = expected_calibration_error(&y_true, &y_proba, 5).unwrap();
/// println!("Expected Calibration Error: {:.4}", ece);
/// ```
pub fn expected_calibration_error(
    y_true: &Array1<i32>,
    y_proba: &Array1<f64>,
    n_bins: usize,
) -> MetricsResult<f64> {
    if y_true.len() != y_proba.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if n_bins == 0 {
        return Err(MetricsError::InvalidParameter(
            "Number of bins must be positive".to_string(),
        ));
    }

    // Validate binary labels
    for &label in y_true.iter() {
        if label != 0 && label != 1 {
            return Err(MetricsError::InvalidParameter(
                "Labels must be binary (0 or 1)".to_string(),
            ));
        }
    }

    // Validate probabilities
    for &prob in y_proba.iter() {
        if !(0.0..=1.0).contains(&prob) {
            return Err(MetricsError::InvalidParameter(
                "Probabilities must be in range [0, 1]".to_string(),
            ));
        }
    }

    let bin_size = 1.0 / n_bins as f64;
    let mut ece = 0.0;
    let n_samples = y_true.len() as f64;

    for i in 0..n_bins {
        let bin_lower = i as f64 * bin_size;
        let bin_upper = (i + 1) as f64 * bin_size;

        // Find samples in this bin
        let mut bin_samples = Vec::new();
        let mut bin_true_labels = Vec::new();

        for (j, &prob) in y_proba.iter().enumerate() {
            if (prob > bin_lower && prob <= bin_upper) || (i == 0 && prob == 0.0) {
                bin_samples.push(prob);
                bin_true_labels.push(y_true[j]);
            }
        }

        if !bin_samples.is_empty() {
            // Calculate confidence (average predicted probability in bin)
            let confidence = bin_samples.iter().sum::<f64>() / bin_samples.len() as f64;

            // Calculate accuracy (fraction of correct predictions in bin)
            let accuracy =
                bin_true_labels.iter().sum::<i32>() as f64 / bin_true_labels.len() as f64;

            // Weight by the number of samples in the bin
            let weight = bin_samples.len() as f64 / n_samples;

            // Add to ECE
            ece += weight * (confidence - accuracy).abs();
        }
    }

    Ok(ece)
}

/// Reliability Diagram Data
///
/// Computes the data needed to create a reliability diagram (calibration plot).
/// Returns bin boundaries, accuracies, confidences, and sample counts for each bin.
///
/// # Arguments
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_proba` - Predicted probabilities for the positive class
/// * `n_bins` - Number of bins to divide the probability range [0, 1]
///
/// # Returns
/// A tuple containing:
/// - `bin_boundaries` - Bin boundaries (length n_bins + 1)
/// - `bin_accuracies` - Accuracy in each bin (length n_bins)
/// - `bin_confidences` - Average confidence in each bin (length n_bins)
/// - `bin_counts` - Number of samples in each bin (length n_bins)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::probabilistic_metrics::reliability_diagram;
///
/// let y_true = array![1, 0, 1, 0, 1, 1, 0, 0];
/// let y_proba = array![0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.4];
/// let (boundaries, accuracies, confidences, counts) =
///     reliability_diagram(&y_true, &y_proba, 5).unwrap();
/// ```
pub fn reliability_diagram(
    y_true: &Array1<i32>,
    y_proba: &Array1<f64>,
    n_bins: usize,
) -> MetricsResult<ReliabilityDiagramOutput> {
    if y_true.len() != y_proba.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if n_bins == 0 {
        return Err(MetricsError::InvalidParameter(
            "Number of bins must be positive".to_string(),
        ));
    }

    // Validate binary labels
    for &label in y_true.iter() {
        if label != 0 && label != 1 {
            return Err(MetricsError::InvalidParameter(
                "Labels must be binary (0 or 1)".to_string(),
            ));
        }
    }

    // Validate probabilities
    for &prob in y_proba.iter() {
        if !(0.0..=1.0).contains(&prob) {
            return Err(MetricsError::InvalidParameter(
                "Probabilities must be in range [0, 1]".to_string(),
            ));
        }
    }

    let bin_size = 1.0 / n_bins as f64;

    // Create bin boundaries
    let mut bin_boundaries = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bin_boundaries.push(i as f64 * bin_size);
    }

    let mut bin_accuracies = Vec::with_capacity(n_bins);
    let mut bin_confidences = Vec::with_capacity(n_bins);
    let mut bin_counts = Vec::with_capacity(n_bins);

    for i in 0..n_bins {
        let bin_lower = bin_boundaries[i];
        let bin_upper = bin_boundaries[i + 1];

        // Find samples in this bin
        let mut bin_samples = Vec::new();
        let mut bin_true_labels = Vec::new();

        for (j, &prob) in y_proba.iter().enumerate() {
            if (prob > bin_lower && prob <= bin_upper) || (i == 0 && prob == 0.0) {
                bin_samples.push(prob);
                bin_true_labels.push(y_true[j]);
            }
        }

        let count = bin_samples.len();
        bin_counts.push(count);

        if count > 0 {
            // Calculate confidence (average predicted probability in bin)
            let confidence = bin_samples.iter().sum::<f64>() / count as f64;
            bin_confidences.push(confidence);

            // Calculate accuracy (fraction of correct predictions in bin)
            let accuracy = bin_true_labels.iter().sum::<i32>() as f64 / count as f64;
            bin_accuracies.push(accuracy);
        } else {
            // Empty bin - use bin midpoint as confidence, 0 as accuracy
            bin_confidences.push((bin_lower + bin_upper) / 2.0);
            bin_accuracies.push(0.0);
        }
    }

    Ok((bin_boundaries, bin_accuracies, bin_confidences, bin_counts))
}

/// Spherical score
///
/// A proper scoring rule that evaluates probabilistic predictions by computing
/// the probability assigned to the true class normalized by the L2 norm of the
/// probability vector.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_proba` - Predicted class probabilities (n_samples x n_classes)
/// * `normalize` - Whether to normalize by sample weights
/// * `sample_weight` - Optional sample weights
///
/// # Returns
/// Spherical score (higher is better)
pub fn spherical_score(
    y_true: &Array1<i32>,
    y_proba: &Array2<f64>,
    normalize: bool,
    sample_weight: Option<&Array1<f64>>,
) -> MetricsResult<f64> {
    if y_true.len() != y_proba.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate that y_proba contains valid probabilities
    for row in y_proba.rows() {
        let sum: f64 = row.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(MetricsError::InvalidParameter(
                "Each row in y_proba must sum to 1.0".to_string(),
            ));
        }
        for &prob in row.iter() {
            if !(0.0..=1.0).contains(&prob) {
                return Err(MetricsError::InvalidParameter(
                    "All probabilities must be between 0 and 1".to_string(),
                ));
            }
        }
    }

    // Get unique classes and create mapping
    let mut unique_classes = BTreeSet::new();
    for &label in y_true.iter() {
        unique_classes.insert(label);
    }
    let classes: Vec<i32> = unique_classes.into_iter().collect();

    if classes.len() != y_proba.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len(), classes.len()],
            actual: vec![y_proba.nrows(), y_proba.ncols()],
        });
    }

    let class_to_index: HashMap<i32, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, &class)| (class, i))
        .collect();

    let mut total_score = 0.0;
    let mut total_weight = 0.0;

    for (i, &true_label) in y_true.iter().enumerate() {
        let true_class_idx = class_to_index[&true_label];
        let proba_row = y_proba.row(i);

        // Calculate L2 norm of probability vector
        let l2_norm: f64 = proba_row.iter().map(|&p| p * p).sum::<f64>().sqrt();

        if l2_norm == 0.0 {
            return Err(MetricsError::InvalidParameter(
                "Probability vector cannot have zero norm".to_string(),
            ));
        }

        // Spherical score: p_i / ||p||_2 where p_i is the probability for the true class
        let score = proba_row[true_class_idx] / l2_norm;

        let weight = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        total_score += score * weight;
        total_weight += weight;
    }

    if normalize {
        Ok(total_score / total_weight)
    } else {
        Ok(total_score)
    }
}

/// Quadratic score (Brier score)
///
/// A proper scoring rule that evaluates probabilistic predictions using
/// the quadratic scoring rule: 2*p_i - ||p||_2^2 where p_i is the probability
/// for the true class.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_proba` - Predicted class probabilities (n_samples x n_classes)
/// * `normalize` - Whether to normalize by sample weights
/// * `sample_weight` - Optional sample weights
///
/// # Returns
/// Quadratic score (higher is better)
pub fn quadratic_score(
    y_true: &Array1<i32>,
    y_proba: &Array2<f64>,
    normalize: bool,
    sample_weight: Option<&Array1<f64>>,
) -> MetricsResult<f64> {
    if y_true.len() != y_proba.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_proba.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Validate that y_proba contains valid probabilities
    for row in y_proba.rows() {
        let sum: f64 = row.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(MetricsError::InvalidParameter(
                "Each row in y_proba must sum to 1.0".to_string(),
            ));
        }
        for &prob in row.iter() {
            if !(0.0..=1.0).contains(&prob) {
                return Err(MetricsError::InvalidParameter(
                    "All probabilities must be between 0 and 1".to_string(),
                ));
            }
        }
    }

    // Get unique classes and create mapping
    let mut unique_classes = BTreeSet::new();
    for &label in y_true.iter() {
        unique_classes.insert(label);
    }
    let classes: Vec<i32> = unique_classes.into_iter().collect();

    if classes.len() != y_proba.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len(), classes.len()],
            actual: vec![y_proba.nrows(), y_proba.ncols()],
        });
    }

    let class_to_index: HashMap<i32, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, &class)| (class, i))
        .collect();

    let mut total_score = 0.0;
    let mut total_weight = 0.0;

    for (i, &true_label) in y_true.iter().enumerate() {
        let true_class_idx = class_to_index[&true_label];
        let proba_row = y_proba.row(i);

        // Quadratic score: 2*p_i - ||p||_2^2 where p_i is the probability for the true class
        let p_true = proba_row[true_class_idx];
        let squared_norm: f64 = proba_row.iter().map(|&p| p * p).sum();
        let score = 2.0 * p_true - squared_norm;

        let weight = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        total_score += score * weight;
        total_weight += weight;
    }

    if normalize {
        Ok(total_score / total_weight)
    } else {
        Ok(total_score)
    }
}

/// Cross-entropy loss
///
/// Computes the cross-entropy between true and predicted probability distributions.
/// This is similar to log_loss but expects full probability distributions.
///
/// # Arguments
/// * `y_true` - True class labels
/// * `y_pred` - Predicted probability distributions (n_samples x n_classes)
/// * `epsilon` - Small value to clip probabilities
/// * `normalize` - Whether to normalize by sample weights
/// * `sample_weight` - Optional sample weights
///
/// # Returns
/// Cross-entropy loss (lower is better)
pub fn cross_entropy(
    y_true: &Array1<i32>,
    y_pred: &Array2<f64>,
    epsilon: Option<f64>,
    normalize: bool,
    sample_weight: Option<&Array1<f64>>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.nrows()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let eps = epsilon.unwrap_or(1e-15);

    // Validate that y_pred contains valid probabilities
    for row in y_pred.rows() {
        let sum: f64 = row.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(MetricsError::InvalidParameter(
                "Each row in y_pred must sum to 1.0".to_string(),
            ));
        }
        for &prob in row.iter() {
            if !(0.0..=1.0).contains(&prob) {
                return Err(MetricsError::InvalidParameter(
                    "All probabilities must be between 0 and 1".to_string(),
                ));
            }
        }
    }

    // Get unique classes and create mapping
    let mut unique_classes = BTreeSet::new();
    for &label in y_true.iter() {
        unique_classes.insert(label);
    }
    let classes: Vec<i32> = unique_classes.into_iter().collect();

    if classes.len() != y_pred.ncols() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len(), classes.len()],
            actual: vec![y_pred.nrows(), y_pred.ncols()],
        });
    }

    let class_to_index: HashMap<i32, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, &class)| (class, i))
        .collect();

    let mut total_loss = 0.0;
    let mut total_weight = 0.0;

    for (i, &true_label) in y_true.iter().enumerate() {
        let true_class_idx = class_to_index[&true_label];
        let pred_prob = y_pred[[i, true_class_idx]];

        // Clip probability to avoid log(0)
        let clipped_prob = pred_prob.max(eps).min(1.0 - eps);
        let loss = -clipped_prob.ln();

        let weight = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        total_loss += loss * weight;
        total_weight += weight;
    }

    if normalize {
        Ok(total_loss / total_weight)
    } else {
        Ok(total_loss)
    }
}

/// Kullback-Leibler (KL) divergence
///
/// Computes the KL divergence between true and predicted probability distributions.
/// KL divergence is not symmetric and measures how much the predicted distribution
/// diverges from the true distribution.
///
/// # Arguments
/// * `p_true` - True probability distributions (n_samples x n_classes)
/// * `p_pred` - Predicted probability distributions (n_samples x n_classes)
/// * `epsilon` - Small value to clip probabilities
/// * `normalize` - Whether to normalize by sample weights
/// * `sample_weight` - Optional sample weights
///
/// # Returns
/// KL divergence (lower is better, 0 means identical distributions)
pub fn kl_divergence(
    p_true: &Array2<f64>,
    p_pred: &Array2<f64>,
    epsilon: Option<f64>,
    normalize: bool,
    sample_weight: Option<&Array1<f64>>,
) -> MetricsResult<f64> {
    if p_true.shape() != p_pred.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: p_true.shape().to_vec(),
            actual: p_pred.shape().to_vec(),
        });
    }

    if p_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let eps = epsilon.unwrap_or(1e-15);

    // Validate that both probability matrices contain valid probabilities
    for (i, (row_true, row_pred)) in p_true
        .rows()
        .into_iter()
        .zip(p_pred.rows().into_iter())
        .enumerate()
    {
        let sum_true: f64 = row_true.sum();
        let sum_pred: f64 = row_pred.sum();

        if (sum_true - 1.0).abs() > 1e-6 || (sum_pred - 1.0).abs() > 1e-6 {
            return Err(MetricsError::InvalidParameter(format!(
                "Each row must sum to 1.0, but row {} has sums: true={:.6}, pred={:.6}",
                i, sum_true, sum_pred
            )));
        }

        for (&p_t, &p_p) in row_true.iter().zip(row_pred.iter()) {
            if !(0.0..=1.0).contains(&p_t) || !(0.0..=1.0).contains(&p_p) {
                return Err(MetricsError::InvalidParameter(
                    "All probabilities must be between 0 and 1".to_string(),
                ));
            }
        }
    }

    let mut total_kl = 0.0;
    let mut total_weight = 0.0;

    for (i, (row_true, row_pred)) in p_true
        .rows()
        .into_iter()
        .zip(p_pred.rows().into_iter())
        .enumerate()
    {
        let mut sample_kl = 0.0;

        for (&p_t, &p_p) in row_true.iter().zip(row_pred.iter()) {
            if p_t > 0.0 {
                // Clip predicted probabilities to avoid log(0)
                let clipped_p_p = p_p.max(eps);
                sample_kl += p_t * (p_t / clipped_p_p).ln();
            }
        }

        let weight = sample_weight.map(|w| w[i]).unwrap_or(1.0);
        total_kl += sample_kl * weight;
        total_weight += weight;
    }

    if normalize {
        Ok(total_kl / total_weight)
    } else {
        Ok(total_kl)
    }
}

/// Jensen-Shannon (JS) divergence
///
/// Computes the JS divergence between true and predicted probability distributions.
/// JS divergence is symmetric and is the square root of the JS distance.
/// It is bounded between 0 and 1.
///
/// # Arguments
/// * `p_true` - True probability distributions (n_samples x n_classes)
/// * `p_pred` - Predicted probability distributions (n_samples x n_classes)
/// * `epsilon` - Small value to clip probabilities
/// * `normalize` - Whether to normalize by sample weights
/// * `sample_weight` - Optional sample weights
///
/// # Returns
/// JS divergence (lower is better, 0 means identical distributions)
pub fn js_divergence(
    p_true: &Array2<f64>,
    p_pred: &Array2<f64>,
    epsilon: Option<f64>,
    normalize: bool,
    sample_weight: Option<&Array1<f64>>,
) -> MetricsResult<f64> {
    if p_true.shape() != p_pred.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: p_true.shape().to_vec(),
            actual: p_pred.shape().to_vec(),
        });
    }

    if p_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Jensen-Shannon divergence is symmetric: JS(P,Q) = 0.5 * KL(P,M) + 0.5 * KL(Q,M)
    // where M = 0.5 * (P + Q)
    let mut p_mean = Array2::zeros(p_true.dim());

    for ((i, j), &p_t) in p_true.indexed_iter() {
        let p_p = p_pred[[i, j]];
        p_mean[[i, j]] = 0.5 * (p_t + p_p);
    }

    // Calculate KL(P, M) and KL(Q, M)
    let kl_pm = kl_divergence(p_true, &p_mean, epsilon, normalize, sample_weight)?;
    let kl_qm = kl_divergence(p_pred, &p_mean, epsilon, normalize, sample_weight)?;

    // JS divergence is the average of the two KL divergences
    Ok(0.5 * (kl_pm + kl_qm))
}
