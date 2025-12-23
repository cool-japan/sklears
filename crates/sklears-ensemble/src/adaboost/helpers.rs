//! Helper functions for AdaBoost ensemble methods

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;

/// Helper function to convert Float labels to i32 for decision tree
pub(crate) fn convert_labels_to_i32(y: &Array1<Float>) -> Array1<i32> {
    y.mapv(|v| v as i32)
}

/// Helper function to convert i32 predictions back to Float
pub(crate) fn convert_predictions_to_float(y: &Array1<i32>) -> Array1<Float> {
    y.mapv(|v| v as Float)
}

/// Estimate class probabilities for SAMME.R using confidence from decision stumps
pub(crate) fn estimate_probabilities(
    y_pred: &Array1<Float>,
    classes: &Array1<Float>,
    n_samples: usize,
    n_classes: usize,
) -> Array2<Float> {
    let mut probs = Array2::<Float>::zeros((n_samples, n_classes));

    if n_classes == 2 {
        for i in 0..n_samples {
            let pred_class = y_pred[i];
            let base_confidence = 0.75;
            let logit_confidence = ((base_confidence / (1.0 - base_confidence)) as Float)
                .ln()
                .abs();

            if pred_class == classes[0] {
                let p0 = 1.0 / (1.0 + (-logit_confidence).exp());
                probs[[i, 0]] = p0;
                probs[[i, 1]] = 1.0 - p0;
            } else {
                let p1 = 1.0 / (1.0 + (-logit_confidence).exp());
                probs[[i, 1]] = p1;
                probs[[i, 0]] = 1.0 - p1;
            }
        }
    } else {
        let temperature = 2.0;
        for i in 0..n_samples {
            let pred_class = y_pred[i];
            let mut logits = Array1::<Float>::zeros(n_classes);

            for (j, &class) in classes.iter().enumerate() {
                if class == pred_class {
                    logits[j] = 1.0 / temperature;
                } else {
                    logits[j] = -0.5 / temperature;
                }
            }

            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Array1<Float> = logits.mapv(|l| (l - max_logit).exp());
            let sum_exp = exp_logits.sum();

            for j in 0..n_classes {
                probs[[i, j]] = exp_logits[j] / sum_exp;
                probs[[i, j]] = probs[[i, j]].max(1e-7);
            }
        }
    }

    probs
}

/// Estimate binary class probabilities for Real AdaBoost
pub(crate) fn estimate_binary_probabilities(
    y_pred: &Array1<Float>,
    classes: &Array1<Float>,
) -> Array2<Float> {
    let n_samples = y_pred.len();
    let mut probs = Array2::<Float>::zeros((n_samples, 2));

    for i in 0..n_samples {
        let pred_class = y_pred[i];
        let base_confidence = 0.8;

        if pred_class == classes[0] {
            probs[[i, 0]] = base_confidence;
            probs[[i, 1]] = 1.0 - base_confidence;
        } else {
            probs[[i, 0]] = 1.0 - base_confidence;
            probs[[i, 1]] = base_confidence;
        }
    }

    probs
}
