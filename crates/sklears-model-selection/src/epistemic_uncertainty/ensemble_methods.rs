use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::rand_prelude::StdRng;
use scirs2_core::random::Random;

pub fn deep_ensemble_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    n_models: usize,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let effective_models = n_models.min(models.len());
    let mut all_predictions = Array2::<f64>::zeros((effective_models, n_predictions));

    for model_idx in 0..effective_models {
        let predictions = ensemble_model_predictions(&models[model_idx], x)?;
        for (i, pred) in predictions.iter().enumerate() {
            all_predictions[[model_idx, i]] = *pred;
        }
    }

    let mean_predictions = all_predictions
        .mean_axis(scirs2_core::ndarray::Axis(0))
        .unwrap();
    let uncertainties = all_predictions.var_axis(scirs2_core::ndarray::Axis(0), 0.0);

    Ok((mean_predictions, uncertainties))
}

pub fn bootstrap_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    n_bootstrap: usize,
    sample_ratio: f64,
    rng: &mut Random<StdRng>,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let sample_size = (n_predictions as f64 * sample_ratio) as usize;
    let mut all_predictions = Array2::<f64>::zeros((n_bootstrap, n_predictions));

    for bootstrap_idx in 0..n_bootstrap {
        let bootstrap_indices = generate_bootstrap_indices(n_predictions, sample_size, rng);
        let bootstrap_predictions = bootstrap_model_predictions(&models[0], x, &bootstrap_indices)?;

        for (i, pred) in bootstrap_predictions.iter().enumerate() {
            all_predictions[[bootstrap_idx, i]] = *pred;
        }
    }

    let mean_predictions = all_predictions
        .mean_axis(scirs2_core::ndarray::Axis(0))
        .unwrap();
    let uncertainties = all_predictions.var_axis(scirs2_core::ndarray::Axis(0), 0.0);

    Ok((mean_predictions, uncertainties))
}

pub fn gaussian_process_uncertainty<E>(
    models: &[E],
    x: &Array2<f64>,
    kernel_type: &str,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);
    let mut uncertainties = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        let (pred, var) =
            gp_prediction_with_uncertainty(&models[0], &input_row.to_owned(), kernel_type)?;
        predictions[i] = pred;
        uncertainties[i] = var;
    }

    Ok((predictions, uncertainties))
}

fn ensemble_model_predictions<E>(
    model: &E,
    x: &Array2<f64>,
) -> Result<Array1<f64>, Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let input_row = x.row(i);
        predictions[i] = simulate_model_prediction(model, &input_row.to_owned())?;
    }

    Ok(predictions)
}

fn bootstrap_model_predictions<E>(
    model: &E,
    x: &Array2<f64>,
    indices: &[usize],
) -> Result<Array1<f64>, Box<dyn std::error::Error>>
where
    E: Clone,
{
    let n_predictions = x.nrows();
    let mut predictions = Array1::<f64>::zeros(n_predictions);

    for i in 0..n_predictions {
        let sample_idx = indices[i % indices.len()];
        let input_row = x.row(sample_idx);
        predictions[i] = simulate_model_prediction(model, &input_row.to_owned())?;
    }

    Ok(predictions)
}

fn generate_bootstrap_indices(
    n_total: usize,
    sample_size: usize,
    rng: &mut Random<StdRng>,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        indices.push(rng.gen_range(0..n_total));
    }
    indices
}

fn simulate_model_prediction<E>(
    model: &E,
    input: &Array1<f64>,
) -> Result<f64, Box<dyn std::error::Error>>
where
    E: Clone,
{
    let sum = input.sum();
    Ok(sum / input.len() as f64)
}

fn gp_prediction_with_uncertainty<E>(
    model: &E,
    input: &Array1<f64>,
    kernel_type: &str,
) -> Result<(f64, f64), Box<dyn std::error::Error>>
where
    E: Clone,
{
    let prediction = input.sum() / input.len() as f64;
    let uncertainty = match kernel_type {
        "rbf" => 0.1,
        "matern" => 0.12,
        "linear" => 0.08,
        _ => 0.1,
    };
    Ok((prediction, uncertainty))
}
