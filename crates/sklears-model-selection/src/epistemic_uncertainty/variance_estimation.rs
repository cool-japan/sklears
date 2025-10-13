use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

pub fn heteroskedastic_regression_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    n_ensemble: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let base_pred = input_row.sum() / input_row.len() as f64;
        let input_dependent_var = estimate_input_dependent_variance(&input_row.to_owned());

        predictions[i] = base_pred;
        variance_estimates[i] = input_dependent_var;
        uncertainties[i] = input_dependent_var;
        noise_estimates[i] = input_dependent_var.sqrt() * 0.1;
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

pub fn mixture_density_network_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    n_components: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let (pred, var) = mixture_prediction(&input_row.to_owned(), n_components);

        predictions[i] = pred;
        variance_estimates[i] = var;
        uncertainties[i] = var;
        noise_estimates[i] = var.sqrt() * 0.05;
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

pub fn quantile_regression_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    quantiles: &[f64],
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let quantile_preds = quantiles
            .iter()
            .map(|&q| quantile_prediction(&input_row.to_owned(), q))
            .collect::<Vec<_>>();

        let median_pred = quantile_preds[quantile_preds.len() / 2];
        let iqr = quantile_preds.last().unwrap() - quantile_preds.first().unwrap();
        let variance = (iqr / 1.35).powi(2); // Approximation

        predictions[i] = median_pred;
        variance_estimates[i] = variance;
        uncertainties[i] = variance;
        noise_estimates[i] = iqr * 0.1;
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

pub fn parametric_uncertainty_estimation<E>(
    models: &[E],
    x: &Array2<f64>,
    distribution: &str,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let (pred, var) = parametric_prediction(&input_row.to_owned(), distribution);

        predictions[i] = pred;
        variance_estimates[i] = var;
        uncertainties[i] = var;
        noise_estimates[i] = var.sqrt() * 0.2;
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

pub fn input_dependent_noise_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    noise_model: &str,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let base_pred = input_row.sum() / input_row.len() as f64;
        let noise_level = estimate_noise_level(&input_row.to_owned(), noise_model);

        predictions[i] = base_pred;
        variance_estimates[i] = noise_level.powi(2);
        uncertainties[i] = noise_level.powi(2);
        noise_estimates[i] = noise_level;
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

pub fn residual_based_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    y_true: Option<&Array1<f64>>,
    window_size: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let base_pred = input_row.sum() / input_row.len() as f64;

        let residual_variance = match y_true {
            Some(y) => compute_local_residual_variance(i, y, window_size),
            None => 0.1, // Default uncertainty
        };

        predictions[i] = base_pred;
        variance_estimates[i] = residual_variance;
        uncertainties[i] = residual_variance;
        noise_estimates[i] = residual_variance.sqrt();
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

pub fn ensemble_aleatoric_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    n_models: usize,
    noise_estimation: &str,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);
    let mut variance_estimates = Array1::<f64>::zeros(n_predictions);
    let mut noise_estimates = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let base_pred = input_row.sum() / input_row.len() as f64;
        let ensemble_var =
            estimate_ensemble_noise(&input_row.to_owned(), n_models, noise_estimation);

        predictions[i] = base_pred;
        variance_estimates[i] = ensemble_var;
        uncertainties[i] = ensemble_var;
        noise_estimates[i] = ensemble_var.sqrt() * 0.5;
    }

    Ok((
        predictions,
        uncertainties,
        variance_estimates,
        noise_estimates,
    ))
}

fn estimate_input_dependent_variance(input: &Array1<f64>) -> f64 {
    let mean_input = input.mean().unwrap_or(1.0);
    0.01 + 0.1 * mean_input.abs()
}

fn mixture_prediction(input: &Array1<f64>, n_components: usize) -> (f64, f64) {
    let base_pred = input.sum() / input.len() as f64;
    let variance = 0.05 + 0.02 * n_components as f64;
    (base_pred, variance)
}

fn quantile_prediction(input: &Array1<f64>, quantile: f64) -> f64 {
    let base_pred = input.sum() / input.len() as f64;
    base_pred + (quantile - 0.5) * 0.2
}

fn parametric_prediction(input: &Array1<f64>, distribution: &str) -> (f64, f64) {
    let base_pred = input.sum() / input.len() as f64;
    let variance = match distribution {
        "normal" => 0.1,
        "gamma" => 0.15,
        "beta" => 0.08,
        _ => 0.1,
    };
    (base_pred, variance)
}

fn estimate_noise_level(input: &Array1<f64>, noise_model: &str) -> f64 {
    let input_magnitude = input.iter().map(|x| x.abs()).sum::<f64>();
    match noise_model {
        "linear" => 0.01 + 0.05 * input_magnitude,
        "quadratic" => 0.01 + 0.02 * input_magnitude.powi(2),
        _ => 0.1,
    }
}

fn compute_local_residual_variance(index: usize, y_true: &Array1<f64>, window_size: usize) -> f64 {
    let start = index.saturating_sub(window_size / 2);
    let end = (index + window_size / 2).min(y_true.len());

    let window_values = &y_true.slice(scirs2_core::ndarray::s![start..end]);
    window_values.var(0.0)
}

fn estimate_ensemble_noise(input: &Array1<f64>, n_models: usize, noise_estimation: &str) -> f64 {
    let base_noise = input.var(0.0);
    match noise_estimation {
        "average" => base_noise / n_models as f64,
        "maximum" => base_noise,
        _ => base_noise * 0.5,
    }
}
