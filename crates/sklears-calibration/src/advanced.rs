//! Advanced calibration methods for sophisticated probability calibration
//!
//! This module provides advanced calibration techniques that go beyond traditional
//! methods like Platt scaling and isotonic regression. It includes:
//!
//! - **Conformal Prediction**: Provides prediction intervals with finite-sample guarantees
//! - **Bayesian Methods**: Model averaging, variational inference, MCMC, and hierarchical approaches
//! - **Non-parametric Methods**: Gaussian processes and Dirichlet processes
//! - **Domain-specific Methods**: Time series, regression, ranking, and survival analysis calibration
//!
//! These methods are particularly useful for:
//! - Applications requiring uncertainty quantification
//! - Non-stationary environments where traditional calibration fails
//! - Complex models where simple parametric calibration is insufficient
//! - Safety-critical applications requiring reliability guarantees
//!
//! # Examples
//!
//! ```rust
//! use sklears_calibration::advanced::ConformalAdapter;
//! use sklears_calibration::conformal::ConformalMethod;
//! use sklears_calibration::CalibrationEstimator;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create a conformal calibrator
//! let mut calibrator = ConformalAdapter::new(ConformalMethod::Split, 0.1);
//!
//! // Fit on calibration data
//! let probabilities = Array1::from(vec![0.1, 0.7, 0.9, 0.3]);
//! let targets = Array1::from(vec![0, 1, 1, 0]);
//! calibrator.fit(&probabilities, &targets)?;
//!
//! // Predict calibrated probabilities
//! let test_probs = Array1::from(vec![0.6, 0.8]);
//! let calibrated = calibrator.predict_proba(&test_probs)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

use crate::{
    bayesian::{
        BayesianModelAveragingCalibrator, DirichletProcessCalibrator, GPKernelType,
        HierarchicalBayesianCalibrator, MCMCCalibrator, NonParametricGPCalibrator,
        VariationalInferenceCalibrator,
    },
    bbq::BBQCalibrator,
    binary::SigmoidCalibrator,
    conformal,
    domain_specific::{
        RankingCalibrator, RegressionCalibrator, SurvivalCalibrator, TimeSeriesCalibrator,
    },
    histogram::HistogramBinningCalibrator,
    isotonic::IsotonicCalibrator,
    temperature::TemperatureScalingCalibrator,
    CalibrationEstimator,
};

/// Adapter to make conformal predictors compatible with CalibrationEstimator trait
///
/// The `ConformalAdapter` enables the use of conformal prediction methods within
/// the standard calibration framework. Conformal prediction provides finite-sample
/// validity guarantees for prediction intervals, making it particularly valuable
/// for uncertainty quantification.
///
/// # Supported Conformal Methods
///
/// - **Split Conformal**: Uses a held-out calibration set
/// - **Cross Conformal**: Cross-validation based approach
/// - **Jackknife+**: Leave-one-out based method with improved performance
///
/// # Parameters
///
/// - `method`: The conformal prediction method to use
/// - `alpha`: Miscoverage level (e.g., 0.1 for 90% coverage)
///
/// # Example
///
/// ```rust
/// use sklears_calibration::advanced::ConformalAdapter;
/// use sklears_calibration::conformal::ConformalMethod;
/// use sklears_calibration::CalibrationEstimator;
/// use scirs2_core::ndarray::Array1;
///
/// let mut calibrator = ConformalAdapter::new(ConformalMethod::Split, 0.1);
/// let probabilities = Array1::from(vec![0.1, 0.7, 0.9, 0.3]);
/// let targets = Array1::from(vec![0, 1, 1, 0]);
/// calibrator.fit(&probabilities, &targets)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ConformalAdapter {
    /// Method-specific parameters
    method: conformal::ConformalMethod,
    /// Alpha level (miscoverage)
    alpha: Float,
    /// Fitted conformal predictor
    predictor: Option<conformal::ConformalPredictor>,
}

impl ConformalAdapter {
    /// Create a new conformal adapter
    ///
    /// # Arguments
    ///
    /// * `method` - The conformal prediction method to use
    /// * `alpha` - Miscoverage level (e.g., 0.1 for 90% coverage)
    ///
    /// # Returns
    ///
    /// A new `ConformalAdapter` instance
    pub fn new(method: conformal::ConformalMethod, alpha: Float) -> Self {
        Self {
            method,
            alpha,
            predictor: None,
        }
    }
}

impl CalibrationEstimator for ConformalAdapter {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Convert binary targets to continuous for conformal prediction
        let targets: Array1<Float> = y_true.mapv(|y| y as Float);

        // Use absolute residual score by default
        let conformity_score = Box::new(conformal::AbsoluteResidualScore);

        let mut predictor =
            conformal::ConformalPredictor::new(self.method.clone(), conformity_score)
                .alpha(self.alpha);

        predictor.fit(probabilities, &targets)?;
        self.predictor = Some(predictor);

        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if let Some(ref predictor) = self.predictor {
            let result = predictor.predict(probabilities)?;

            // For calibration compatibility, return the midpoint of intervals as probabilities
            let midpoints: Vec<Float> = (0..result.intervals.nrows())
                .map(|i| {
                    let lower = result.intervals[[i, 0]];
                    let upper = result.intervals[[i, 1]];
                    (lower + upper) / 2.0
                })
                .collect();

            Ok(Array1::from(midpoints))
        } else {
            Err(SklearsError::InvalidData {
                reason: "Conformal predictor not fitted".to_string(),
            })
        }
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Train conformal split calibrators for multi-class problems
///
/// Uses the split conformal prediction method to create calibrators for each class.
/// The split method divides data into training and calibration sets.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused for split method)
/// * `alpha` - Miscoverage level
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_conformal_split_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    alpha: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train conformal split calibrator
        let mut calibrator = ConformalAdapter::new(conformal::ConformalMethod::Split, alpha);
        calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train conformal cross-validation calibrators for multi-class problems
///
/// Uses the cross-conformal prediction method which performs cross-validation
/// to obtain more robust calibration.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `alpha` - Miscoverage level
/// * `n_folds` - Number of cross-validation folds
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_conformal_cross_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    alpha: Float,
    n_folds: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train conformal cross-validation calibrator
        let mut calibrator = ConformalAdapter::new(
            conformal::ConformalMethod::CrossConformal { n_folds },
            alpha,
        );
        calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train conformal jackknife+ calibrators for multi-class problems
///
/// Uses the jackknife+ method which provides improved finite-sample coverage
/// guarantees compared to standard conformal methods.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `alpha` - Miscoverage level
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_conformal_jackknife_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    alpha: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train conformal jackknife+ calibrator
        let mut calibrator =
            ConformalAdapter::new(conformal::ConformalMethod::JackknifeePlus, alpha);
        calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train Bayesian model averaging calibrators for multi-class problems
///
/// Combines multiple different calibration methods using Bayesian model averaging
/// to provide more robust calibration that accounts for model uncertainty.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `n_models` - Number of models to include in the ensemble
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_bayesian_model_averaging_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_models: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Create individual models for Bayesian model averaging
        let mut individual_models: Vec<Box<dyn CalibrationEstimator>> = Vec::new();

        // Add different calibration models to the ensemble
        individual_models.push(Box::new(SigmoidCalibrator::new()));
        individual_models.push(Box::new(IsotonicCalibrator::new()));
        if n_models > 2 {
            individual_models.push(Box::new(HistogramBinningCalibrator::new(10)));
        }
        if n_models > 3 {
            individual_models.push(Box::new(TemperatureScalingCalibrator::new()));
        }
        if n_models > 4 {
            individual_models.push(Box::new(BBQCalibrator::new(2, 10)));
        }

        // Trim to requested number of models
        individual_models.truncate(n_models);

        // Train Bayesian model averaging calibrator
        let mut bma_calibrator = BayesianModelAveragingCalibrator::new(individual_models);
        bma_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(bma_calibrator));
    }

    Ok(calibrators)
}

/// Train variational inference calibrators for multi-class problems
///
/// Uses variational inference to approximate the posterior distribution over
/// calibration parameters, providing uncertainty estimates for the calibration itself.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `learning_rate` - Learning rate for variational optimization
/// * `n_samples` - Number of samples for variational approximation
/// * `max_iter` - Maximum number of optimization iterations
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_variational_inference_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    learning_rate: Float,
    n_samples: usize,
    max_iter: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train variational inference calibrator
        let mut vi_calibrator =
            VariationalInferenceCalibrator::with_config(learning_rate, n_samples, max_iter);
        vi_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(vi_calibrator));
    }

    Ok(calibrators)
}

/// Train MCMC calibrators for multi-class problems
///
/// Uses Markov Chain Monte Carlo methods to sample from the posterior distribution
/// over calibration parameters, providing full Bayesian treatment of uncertainty.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `n_samples` - Number of MCMC samples
/// * `burn_in` - Number of burn-in samples to discard
/// * `step_size` - Step size for MCMC sampling
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_mcmc_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_samples: usize,
    burn_in: usize,
    step_size: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train MCMC calibrator
        let mut mcmc_calibrator = MCMCCalibrator::with_config(n_samples, burn_in, step_size);
        mcmc_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(mcmc_calibrator));
    }

    Ok(calibrators)
}

/// Train hierarchical Bayesian calibrators for multi-class problems
///
/// Uses hierarchical Bayesian modeling to share information across groups,
/// particularly useful when data can be naturally grouped (e.g., by domain, time period).
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_hierarchical_bayesian_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Create default groups for hierarchical modeling (could be extended)
        let groups = (0..class_probas.len())
            .map(|idx| format!("group_{}", idx % 3)) // Simple grouping strategy
            .collect();

        // Train hierarchical Bayesian calibrator
        let mut hb_calibrator = HierarchicalBayesianCalibrator::new().with_groups(groups);
        hb_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(hb_calibrator));
    }

    Ok(calibrators)
}

/// Train Dirichlet process calibrators for multi-class problems
///
/// Uses Dirichlet process mixture models for non-parametric calibration,
/// automatically determining the appropriate model complexity.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `concentration` - Concentration parameter for the Dirichlet process
/// * `max_clusters` - Maximum number of clusters to consider
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_dirichlet_process_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    concentration: Float,
    max_clusters: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut calibrator = DirichletProcessCalibrator::new()
            .with_concentration(concentration)
            .with_max_clusters(max_clusters);

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

/// Train non-parametric Gaussian process calibrators for multi-class problems
///
/// Uses Gaussian processes with advanced kernels for flexible, non-parametric calibration
/// that can capture complex relationships between predictions and true probabilities.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `kernel_type` - Type of kernel to use ("spectral_mixture", "neural_network", "periodic", "compositional")
/// * `n_inducing` - Number of inducing points for sparse GP approximation
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_nonparametric_gp_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    kernel_type: String,
    n_inducing: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let kernel = match kernel_type.as_str() {
            "spectral_mixture" => GPKernelType::SpectralMixture,
            "neural_network" => GPKernelType::NeuralNetwork,
            "periodic" => GPKernelType::Periodic,
            "compositional" => GPKernelType::Compositional,
            _ => GPKernelType::SpectralMixture, // Default
        };

        let mut calibrator = NonParametricGPCalibrator::new()
            .with_kernel(kernel)
            .with_inducing_points(n_inducing);

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

/// Train time series calibrators for multi-class problems
///
/// Specialized calibration for time series data that accounts for temporal dependencies
/// and concept drift in the relationship between predictions and true probabilities.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `window_size` - Size of the temporal window
/// * `temporal_decay` - Decay factor for temporal weighting
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_time_series_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    window_size: usize,
    temporal_decay: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train time series calibrator
        let mut ts_calibrator =
            TimeSeriesCalibrator::new(window_size).with_temporal_decay(temporal_decay);
        ts_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(ts_calibrator));
    }

    Ok(calibrators)
}

/// Train regression calibrators for multi-class problems
///
/// Uses advanced regression techniques for calibration, with optional distributional
/// modeling that captures the full distribution of calibrated probabilities.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `distributional` - Whether to use distributional calibration
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_regression_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    distributional: bool,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train regression calibrator
        let mut reg_calibrator = if distributional {
            RegressionCalibrator::new().with_distributional_calibration()
        } else {
            RegressionCalibrator::new()
        };
        reg_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(reg_calibrator));
    }

    Ok(calibrators)
}

/// Train ranking calibrators for multi-class problems
///
/// Specialized calibration for ranking problems that considers the relative ordering
/// of predictions and can use listwise learning approaches.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `ranking_weight` - Weight for ranking loss component
/// * `listwise` - Whether to use listwise calibration
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_ranking_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    ranking_weight: Float,
    listwise: bool,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train ranking calibrator
        let mut ranking_calibrator = RankingCalibrator::new().with_ranking_weight(ranking_weight);

        if listwise {
            ranking_calibrator = ranking_calibrator.with_listwise_calibration(10);
        }

        ranking_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(ranking_calibrator));
    }

    Ok(calibrators)
}

/// Train survival calibrators for multi-class problems
///
/// Specialized calibration for survival analysis that handles censored data
/// and time-to-event predictions.
///
/// # Arguments
///
/// * `probabilities` - Predicted probabilities for each class
/// * `y` - True class labels
/// * `classes` - Unique class labels
/// * `_cv` - Cross-validation parameter (unused)
/// * `time_points` - Time points for survival analysis
/// * `handle_censoring` - Whether to handle censored observations
///
/// # Returns
///
/// Vector of trained calibrators, one per class
pub fn train_survival_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    time_points: Vec<Float>,
    handle_censoring: bool,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train survival calibrator
        let mut survival_calibrator = SurvivalCalibrator::new(time_points.clone());

        if handle_censoring {
            survival_calibrator = survival_calibrator.with_censoring_handling();
            // Set default censoring indicators (all events observed)
            let censoring = Array1::from(vec![1; class_probas.len()]);
            survival_calibrator.set_censoring_indicators(censoring);
        }

        survival_calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(survival_calibrator));
    }

    Ok(calibrators)
}
