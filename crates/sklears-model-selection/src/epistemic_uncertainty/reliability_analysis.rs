use super::uncertainty_types::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

pub fn compute_reliability_diagram(
    confidences: &Array1<f64>,
    accuracies: &Array1<f64>,
    n_bins: usize,
) -> ReliabilityDiagram {
    let bin_boundaries = Array1::<f64>::linspace(0.0, 1.0, n_bins + 1);
    let mut bin_accuracies = Array1::<f64>::zeros(n_bins);
    let mut bin_confidences = Array1::<f64>::zeros(n_bins);
    let mut bin_counts = Array1::<usize>::zeros(n_bins);

    for i in 0..n_bins {
        let lower_bound = bin_boundaries[i];
        let upper_bound = bin_boundaries[i + 1];

        let mut bin_acc_sum = 0.0;
        let mut bin_conf_sum = 0.0;
        let mut count = 0;

        for j in 0..confidences.len() {
            let conf = confidences[j];
            if conf > lower_bound && conf <= upper_bound {
                bin_acc_sum += accuracies[j];
                bin_conf_sum += conf;
                count += 1;
            }
        }

        if count > 0 {
            bin_accuracies[i] = bin_acc_sum / count as f64;
            bin_confidences[i] = bin_conf_sum / count as f64;
        }
        bin_counts[i] = count;
    }

    let expected_calibration_error = compute_expected_calibration_error(
        &bin_accuracies,
        &bin_confidences,
        &bin_counts,
        confidences.len(),
    );

    let maximum_calibration_error = bin_accuracies
        .iter()
        .zip(bin_confidences.iter())
        .map(|(&acc, &conf)| (acc - conf).abs())
        .fold(0.0, |max, x| max.max(x));

    ReliabilityDiagram {
        bin_boundaries,
        bin_accuracies,
        bin_confidences,
        bin_counts,
        expected_calibration_error,
        maximum_calibration_error,
    }
}

pub fn compute_expected_calibration_error(
    bin_accuracies: &Array1<f64>,
    bin_confidences: &Array1<f64>,
    bin_counts: &Array1<usize>,
    total_samples: usize,
) -> f64 {
    let mut ece = 0.0;

    for i in 0..bin_accuracies.len() {
        if bin_counts[i] > 0 {
            let weight = bin_counts[i] as f64 / total_samples as f64;
            let bin_error = (bin_accuracies[i] - bin_confidences[i]).abs();
            ece += weight * bin_error;
        }
    }

    ece
}

pub fn compute_coverage_probability(
    predictions: &Array1<f64>,
    prediction_intervals: &Array2<f64>,
    y_true: &Array1<f64>,
) -> f64 {
    let mut covered = 0;
    let total = predictions.len();

    for i in 0..total {
        let lower_bound = prediction_intervals[[i, 0]];
        let upper_bound = prediction_intervals[[i, 1]];
        let true_value = y_true[i];

        if true_value >= lower_bound && true_value <= upper_bound {
            covered += 1;
        }
    }

    covered as f64 / total as f64
}

pub fn compute_prediction_interval_score(
    prediction_intervals: &Array2<f64>,
    y_true: &Array1<f64>,
    alpha: f64,
) -> f64 {
    let mut total_score = 0.0;
    let n_samples = y_true.len();

    for i in 0..n_samples {
        let lower = prediction_intervals[[i, 0]];
        let upper = prediction_intervals[[i, 1]];
        let true_val = y_true[i];

        let interval_width = upper - lower;
        let penalty = if true_val < lower {
            2.0 / alpha * (lower - true_val)
        } else if true_val > upper {
            2.0 / alpha * (true_val - upper)
        } else {
            0.0
        };

        total_score += interval_width + penalty;
    }

    total_score / n_samples as f64
}

pub fn compute_continuous_ranked_probability_score(
    predictions: &Array1<f64>,
    uncertainties: &Array1<f64>,
    y_true: &Array1<f64>,
) -> f64 {
    let mut total_crps = 0.0;
    let n_samples = predictions.len();

    for i in 0..n_samples {
        let pred = predictions[i];
        let std_dev = uncertainties[i].sqrt();
        let true_val = y_true[i];

        // Approximate CRPS for normal distribution
        let z = (true_val - pred) / std_dev;
        let phi_z = normal_cdf(z);
        let pdf_z = normal_pdf(z);

        let crps =
            std_dev * (z * (2.0 * phi_z - 1.0) + 2.0 * pdf_z - 1.0 / std::f64::consts::PI.sqrt());
        total_crps += crps;
    }

    total_crps / n_samples as f64
}

pub fn compute_sharpness(uncertainties: &Array1<f64>) -> f64 {
    uncertainties.mean().unwrap_or(0.0)
}

pub fn compute_reliability_score(calibration_error: f64, sharpness: f64) -> f64 {
    let normalized_sharpness = sharpness / (1.0 + sharpness);
    1.0 - calibration_error - 0.1 * normalized_sharpness
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn normal_pdf(x: f64) -> f64 {
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    (-0.5 * x * x).exp() / sqrt_2pi
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
