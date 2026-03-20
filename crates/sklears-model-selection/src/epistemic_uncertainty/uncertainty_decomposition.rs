use scirs2_core::ndarray::{Array1, Array2};
// use scirs2_core::numeric::Float;

#[derive(Debug, Clone)]
pub enum UncertaintyDecompositionMethod {
    /// VarianceDecomposition
    VarianceDecomposition,
    /// InformationTheoretic
    InformationTheoretic,
    /// BayesianDecomposition
    BayesianDecomposition,
    /// PredictiveEntropy
    PredictiveEntropy,
    /// MutualInformation
    MutualInformation,
}

pub fn decompose_variance_uncertainty(
    predictions: &Array2<f64>,
    mean_prediction: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    let n_samples = predictions.nrows();
    let n_predictions = predictions.ncols();

    let total_variance = predictions.var_axis(scirs2_core::ndarray::Axis(0), 0.0);

    let mut epistemic_uncertainty = Array1::<f64>::zeros(n_predictions);
    let mut aleatoric_uncertainty = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let pred_col = predictions.column(i);
        let mean_val = mean_prediction[i];

        let model_variance = pred_col
            .iter()
            .map(|&x| (x - mean_val).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        let within_model_variance = total_variance[i] - model_variance;

        epistemic_uncertainty[i] = model_variance;
        aleatoric_uncertainty[i] = within_model_variance.max(0.0);
    }

    Ok((epistemic_uncertainty, aleatoric_uncertainty))
}

pub fn information_theoretic_decomposition(
    predictions: &Array2<f64>,
    prediction_probs: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    let n_samples = predictions.nrows();
    let n_predictions = predictions.ncols();

    let mut epistemic_uncertainty = Array1::<f64>::zeros(n_predictions);
    let mut aleatoric_uncertainty = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let prob_col = prediction_probs.column(i);

        let mean_entropy = prob_col
            .iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum::<f64>()
            / n_samples as f64;

        let entropy_of_mean = {
            let mean_p = prob_col.mean().unwrap_or(0.0);
            if mean_p > 0.0 {
                -mean_p * mean_p.ln()
            } else {
                0.0
            }
        };

        epistemic_uncertainty[i] = entropy_of_mean;
        aleatoric_uncertainty[i] = mean_entropy;
    }

    Ok((epistemic_uncertainty, aleatoric_uncertainty))
}
