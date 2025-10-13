use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

#[derive(Debug, Clone)]
pub enum CalibrationMethod {
    /// PlattScaling
    PlattScaling,
    /// IsotonicRegression
    IsotonicRegression,
    /// TemperatureScaling
    TemperatureScaling,
    /// HistogramBinning
    HistogramBinning {
        n_bins: usize,
    },

    None,
}

pub fn apply_temperature_scaling(logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
    logits.mapv(|x| x / temperature)
}

pub fn compute_calibration_error(
    confidences: &Array1<f64>,
    accuracies: &Array1<f64>,
    n_bins: usize,
) -> f64 {
    let bin_boundaries = Array1::<f64>::linspace(0.0, 1.0, n_bins + 1);
    let mut calibration_error = 0.0;
    let n_samples = confidences.len();

    for i in 0..n_bins {
        let lower_bound = bin_boundaries[i];
        let upper_bound = bin_boundaries[i + 1];

        let mask: Vec<bool> = confidences
            .iter()
            .map(|&conf| conf > lower_bound && conf <= upper_bound)
            .collect();

        let bin_size = mask.iter().filter(|&&m| m).count();
        if bin_size > 0 {
            let bin_accuracy: f64 = mask
                .iter()
                .zip(accuracies.iter())
                .filter(|(&m, _)| m)
                .map(|(_, &acc)| acc)
                .sum::<f64>()
                / bin_size as f64;

            let bin_confidence: f64 = mask
                .iter()
                .zip(confidences.iter())
                .filter(|(&m, _)| m)
                .map(|(_, &conf)| conf)
                .sum::<f64>()
                / bin_size as f64;

            calibration_error +=
                (bin_size as f64 / n_samples as f64) * (bin_confidence - bin_accuracy).abs();
        }
    }

    calibration_error
}

pub fn platt_scaling(
    scores: &Array1<f64>,
    labels: &Array1<f64>,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let mut a = 1.0;
    let mut b = 0.0;

    for _ in 0..100 {
        let mut gradient_a = 0.0;
        let mut gradient_b = 0.0;
        let mut hessian_aa = 0.0;
        let mut hessian_ab = 0.0;
        let mut hessian_bb = 0.0;

        for (i, (&score, &label)) in scores.iter().zip(labels.iter()).enumerate() {
            let z = a * score + b;
            let p = 1.0 / (1.0 + (-z).exp());

            let error = p - label;
            gradient_a += error * score;
            gradient_b += error;

            let weight = p * (1.0 - p);
            hessian_aa += weight * score * score;
            hessian_ab += weight * score;
            hessian_bb += weight;
        }

        let det = hessian_aa * hessian_bb - hessian_ab * hessian_ab;
        if det.abs() < 1e-10 {
            break;
        }

        let delta_a = -(hessian_bb * gradient_a - hessian_ab * gradient_b) / det;
        let delta_b = -(hessian_aa * gradient_b - hessian_ab * gradient_a) / det;

        a += delta_a;
        b += delta_b;

        if (delta_a.abs() + delta_b.abs()) < 1e-6 {
            break;
        }
    }

    Ok((a, b))
}
