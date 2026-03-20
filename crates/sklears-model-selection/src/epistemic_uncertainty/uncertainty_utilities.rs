use super::aleatoric_quantifier::AleatoricUncertaintyQuantifier;
use super::epistemic_quantifier::EpistemicUncertaintyQuantifier;
use super::uncertainty_config::*;
use super::uncertainty_quantifier::UncertaintyQuantifier;
use super::uncertainty_results::*;
use scirs2_core::ndarray::{Array1, Array2};

pub fn quantify_epistemic_uncertainty<E, P>(
    models: &[E],
    x: &Array2<f64>,
    y_true: Option<&Array1<f64>>,
    config: Option<EpistemicUncertaintyConfig>,
) -> Result<EpistemicUncertaintyResult, Box<dyn std::error::Error>>
where
    E: Clone,
    P: Clone,
{
    let quantifier = match config {
        Some(cfg) => EpistemicUncertaintyQuantifier::with_config(cfg),
        None => EpistemicUncertaintyQuantifier::new(),
    };

    quantifier.quantify::<E, P>(models, x, y_true)
}

pub fn quantify_aleatoric_uncertainty<E, P>(
    models: &[E],
    x: &Array2<f64>,
    y_true: Option<&Array1<f64>>,
    config: Option<AleatoricUncertaintyConfig>,
) -> Result<AleatoricUncertaintyResult, Box<dyn std::error::Error>>
where
    E: Clone,
    P: Clone,
{
    let quantifier = match config {
        Some(cfg) => AleatoricUncertaintyQuantifier::with_config(cfg),
        None => AleatoricUncertaintyQuantifier::new(),
    };

    quantifier.quantify::<E, P>(models, x, y_true)
}

pub fn quantify_uncertainty<E, P>(
    models: &[E],
    x: &Array2<f64>,
    y_true: Option<&Array1<f64>>,
    config: Option<UncertaintyQuantificationConfig>,
) -> Result<UncertaintyQuantificationResult, Box<dyn std::error::Error>>
where
    E: Clone,
    P: Clone,
{
    let quantifier = match config {
        Some(cfg) => UncertaintyQuantifier::with_config(cfg),
        None => UncertaintyQuantifier::new(),
    };

    quantifier.quantify::<E, P>(models, x, y_true)
}
