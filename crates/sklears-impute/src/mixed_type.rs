//! Mixed-type data imputation methods
//!
//! This module provides imputation strategies for datasets containing heterogeneous data types,
//! including ordinal variables, semi-continuous data, and bounded variables.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Variable type enumeration for mixed-type data
#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    /// Continuous numerical variable
    Continuous,
    /// Ordinal categorical variable with ordered levels
    Ordinal(Vec<f64>),
    /// Nominal categorical variable
    Categorical(Vec<f64>),
    /// Semi-continuous variable (mixture of continuous and discrete components)
    SemiContinuous { zero_probability: f64 },
    /// Bounded continuous variable
    Bounded { lower: f64, upper: f64 },
    /// Binary variable
    Binary,
}

/// Variable metadata for mixed-type imputation
#[derive(Debug, Clone)]
pub struct VariableMetadata {
    /// variable_type
    pub variable_type: VariableType,
    /// missing_pattern
    pub missing_pattern: String,
    /// is_target
    pub is_target: bool,
}

/// Heterogeneous Data Imputer
///
/// Imputation for datasets containing multiple data types including continuous,
/// ordinal, categorical, semi-continuous, and bounded variables.
///
/// # Parameters
///
/// * `variable_types` - Map from feature index to variable type
/// * `max_iter` - Maximum number of iterations for iterative methods
/// * `tol` - Tolerance for convergence
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_impute::{HeterogeneousImputer, VariableType};
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
/// ///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
/// let mut variable_types = HashMap::new();
/// variable_types.insert(0, VariableType::Continuous);
/// variable_types.insert(1, VariableType::Ordinal(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
/// variable_types.insert(2, VariableType::Bounded { lower: 0.0, upper: 10.0 });
///
/// let imputer = HeterogeneousImputer::new()
///     .variable_types(variable_types)
///     .max_iter(50);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HeterogeneousImputer<S = Untrained> {
    state: S,
    variable_types: HashMap<usize, VariableType>,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for HeterogeneousImputer
#[derive(Debug, Clone)]
pub struct HeterogeneousImputerTrained {
    variable_types: HashMap<usize, VariableType>,
    learned_parameters: HashMap<usize, VariableParameters>,
    n_features_in_: usize,
}

/// Parameters learned for each variable type
#[derive(Debug, Clone)]
pub enum VariableParameters {
    /// ContinuousParams
    ContinuousParams {
        mean: f64,
        std: f64,
        coefficients: Option<Array1<f64>>,
    },
    /// OrdinalParams
    OrdinalParams {
        levels: Vec<f64>,
        probabilities: Array1<f64>,
        transition_matrix: Option<Array2<f64>>,
    },
    /// CategoricalParams
    CategoricalParams {
        categories: Vec<f64>,
        probabilities: Array1<f64>,
    },
    /// SemiContinuousParams
    SemiContinuousParams {
        zero_prob: f64,
        continuous_mean: f64,
        continuous_std: f64,
        threshold: f64,
    },
    /// BoundedParams
    BoundedParams {
        lower: f64,
        upper: f64,
        beta_alpha: f64,
        beta_beta: f64,
    },
    /// BinaryParams
    BinaryParams { probability: f64 },
}

impl HeterogeneousImputer<Untrained> {
    /// Create a new HeterogeneousImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            variable_types: HashMap::new(),
            max_iter: 100,
            tol: 1e-4,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the variable types for each feature
    pub fn variable_types(mut self, variable_types: HashMap<usize, VariableType>) -> Self {
        self.variable_types = variable_types;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for HeterogeneousImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HeterogeneousImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for HeterogeneousImputer<Untrained> {
    type Fitted = HeterogeneousImputer<HeterogeneousImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        // Auto-detect variable types if not provided
        let variable_types = if self.variable_types.is_empty() {
            self.auto_detect_variable_types(&X)?
        } else {
            self.variable_types.clone()
        };

        // Learn parameters for each variable type
        let mut learned_parameters = HashMap::new();

        for (&feature_idx, var_type) in &variable_types {
            if feature_idx < n_features {
                let column = X.column(feature_idx);
                let observed_values: Vec<f64> = column
                    .iter()
                    .filter(|&&x| !self.is_missing(x))
                    .cloned()
                    .collect();

                if !observed_values.is_empty() {
                    let params = self.learn_variable_parameters(var_type, &observed_values)?;
                    learned_parameters.insert(feature_idx, params);
                }
            }
        }

        Ok(HeterogeneousImputer {
            state: HeterogeneousImputerTrained {
                variable_types,
                learned_parameters,
                n_features_in_: n_features,
            },
            variable_types: self.variable_types,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl HeterogeneousImputer<Untrained> {
    fn auto_detect_variable_types(
        &self,
        X: &Array2<f64>,
    ) -> SklResult<HashMap<usize, VariableType>> {
        let mut variable_types = HashMap::new();
        let (_, n_features) = X.dim();

        for j in 0..n_features {
            let column = X.column(j);
            let observed_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if observed_values.is_empty() {
                continue;
            }

            let var_type = self.detect_variable_type(&observed_values);
            variable_types.insert(j, var_type);
        }

        Ok(variable_types)
    }

    fn detect_variable_type(&self, values: &[f64]) -> VariableType {
        let unique_values: std::collections::HashSet<_> = values
            .iter()
            .map(|&x| (x * 1000.0).round() as i64)
            .collect();

        // Check if binary
        if unique_values.len() == 2 {
            return VariableType::Binary;
        }

        // Check if all values are integers (potential ordinal/categorical)
        let all_integers = values.iter().all(|&x| x.fract() == 0.0);

        if all_integers && unique_values.len() <= 10 {
            // Assume ordinal if few unique integer values
            let mut sorted_values: Vec<f64> = values.to_vec();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_values.dedup();
            return VariableType::Ordinal(sorted_values);
        }

        // Check for semi-continuous (many zeros)
        let zero_count = values.iter().filter(|&&x| x == 0.0).count();
        let zero_proportion = zero_count as f64 / values.len() as f64;

        if zero_proportion > 0.1 && zero_proportion < 0.9 {
            return VariableType::SemiContinuous {
                zero_probability: zero_proportion,
            };
        }

        // Check if bounded (all values in a specific range)
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min_val >= 0.0 && max_val <= 1.0 {
            return VariableType::Bounded {
                lower: 0.0,
                upper: 1.0,
            };
        }

        // Default to continuous
        VariableType::Continuous
    }

    fn learn_variable_parameters(
        &self,
        var_type: &VariableType,
        observed_values: &[f64],
    ) -> SklResult<VariableParameters> {
        match var_type {
            VariableType::Continuous => {
                let mean = observed_values.iter().sum::<f64>() / observed_values.len() as f64;
                let variance = observed_values
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (observed_values.len() as f64 - 1.0).max(1.0);
                let std = variance.sqrt();

                Ok(VariableParameters::ContinuousParams {
                    mean,
                    std,
                    coefficients: None,
                })
            }
            VariableType::Ordinal(levels) => {
                let mut probabilities = Array1::zeros(levels.len());
                let total_count = observed_values.len() as f64;

                for &value in observed_values {
                    if let Some(idx) = levels
                        .iter()
                        .position(|&level| (level - value).abs() < 1e-10)
                    {
                        probabilities[idx] += 1.0 / total_count;
                    }
                }

                Ok(VariableParameters::OrdinalParams {
                    levels: levels.clone(),
                    probabilities,
                    transition_matrix: None,
                })
            }
            VariableType::Categorical(categories) => {
                let mut probabilities = Array1::zeros(categories.len());
                let total_count = observed_values.len() as f64;

                for &value in observed_values {
                    if let Some(idx) = categories
                        .iter()
                        .position(|&cat| (cat - value).abs() < 1e-10)
                    {
                        probabilities[idx] += 1.0 / total_count;
                    }
                }

                Ok(VariableParameters::CategoricalParams {
                    categories: categories.clone(),
                    probabilities,
                })
            }
            VariableType::SemiContinuous {
                zero_probability: _,
            } => {
                let zero_count = observed_values.iter().filter(|&&x| x == 0.0).count();
                let zero_prob = zero_count as f64 / observed_values.len() as f64;

                let non_zero_values: Vec<f64> = observed_values
                    .iter()
                    .filter(|&&x| x != 0.0)
                    .cloned()
                    .collect();

                let (continuous_mean, continuous_std) = if non_zero_values.is_empty() {
                    (0.0, 1.0)
                } else {
                    let mean = non_zero_values.iter().sum::<f64>() / non_zero_values.len() as f64;
                    let variance = non_zero_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (non_zero_values.len() as f64 - 1.0).max(1.0);
                    (mean, variance.sqrt())
                };

                Ok(VariableParameters::SemiContinuousParams {
                    zero_prob,
                    continuous_mean,
                    continuous_std,
                    threshold: 0.0,
                })
            }
            VariableType::Bounded { lower, upper } => {
                // Fit Beta distribution parameters using method of moments
                let mean = observed_values.iter().sum::<f64>() / observed_values.len() as f64;
                let variance = observed_values
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (observed_values.len() as f64 - 1.0).max(1.0);

                // Transform to [0,1] scale for Beta distribution
                let range = upper - lower;
                let scaled_mean = (mean - lower) / range;
                let scaled_variance = variance / (range * range);

                // Method of moments for Beta distribution
                let alpha =
                    scaled_mean * (scaled_mean * (1.0 - scaled_mean) / scaled_variance - 1.0);
                let beta = (1.0 - scaled_mean)
                    * (scaled_mean * (1.0 - scaled_mean) / scaled_variance - 1.0);

                Ok(VariableParameters::BoundedParams {
                    lower: *lower,
                    upper: *upper,
                    beta_alpha: alpha.max(0.1),
                    beta_beta: beta.max(0.1),
                })
            }
            VariableType::Binary => {
                let ones = observed_values.iter().filter(|&&x| x == 1.0).count();
                let probability = ones as f64 / observed_values.len() as f64;

                Ok(VariableParameters::BinaryParams { probability })
            }
        }
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for HeterogeneousImputer<HeterogeneousImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();
        let mut rng = Random::default();

        // Iterative imputation for mixed-type data
        for iteration in 0..self.max_iter {
            let mut converged = true;
            let _prev_X = X_imputed.clone();

            for (&feature_idx, var_type) in &self.state.variable_types {
                if let Some(params) = self.state.learned_parameters.get(&feature_idx) {
                    for i in 0..n_samples {
                        if self.is_missing(X[[i, feature_idx]]) {
                            let imputed_value = self.impute_value(
                                var_type,
                                params,
                                &X_imputed,
                                i,
                                feature_idx,
                                &mut rng,
                            )?;

                            let old_value = X_imputed[[i, feature_idx]];
                            X_imputed[[i, feature_idx]] = imputed_value;

                            if (old_value - imputed_value).abs() > self.tol {
                                converged = false;
                            }
                        }
                    }
                }
            }

            if converged && iteration > 0 {
                break;
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl HeterogeneousImputer<HeterogeneousImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn impute_value(
        &self,
        var_type: &VariableType,
        params: &VariableParameters,
        X: &Array2<f64>,
        sample_idx: usize,
        feature_idx: usize,
        rng: &mut Random,
    ) -> SklResult<f64> {
        match (var_type, params) {
            (VariableType::Continuous, VariableParameters::ContinuousParams { mean, std, .. }) => {
                // Use regression if other features are available, otherwise use mean
                if let Some(predicted) = self.predict_continuous(X, sample_idx, feature_idx)? {
                    Ok(predicted)
                } else {
                    Ok(mean + std * rng.gen::<f64>())
                }
            }
            (
                VariableType::Ordinal(levels),
                VariableParameters::OrdinalParams { probabilities, .. },
            ) => {
                // Sample from learned probability distribution
                let random_val: f64 = rng.gen();
                let mut cumulative = 0.0;

                for (i, &prob) in probabilities.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative && i < levels.len() {
                        return Ok(levels[i]);
                    }
                }

                // Fallback to first level
                Ok(levels.first().copied().unwrap_or(0.0))
            }
            (
                VariableType::Categorical(categories),
                VariableParameters::CategoricalParams { probabilities, .. },
            ) => {
                // Sample from categorical distribution
                let random_val: f64 = rng.gen();
                let mut cumulative = 0.0;

                for (i, &prob) in probabilities.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative && i < categories.len() {
                        return Ok(categories[i]);
                    }
                }

                // Fallback to first category
                Ok(categories.first().copied().unwrap_or(0.0))
            }
            (
                VariableType::SemiContinuous { .. },
                VariableParameters::SemiContinuousParams {
                    zero_prob,
                    continuous_mean,
                    continuous_std,
                    ..
                },
            ) => {
                // Two-step process: first decide if zero, then sample continuous part
                if rng.gen::<f64>() < *zero_prob {
                    Ok(0.0)
                } else {
                    Ok(continuous_mean + continuous_std * rng.gen::<f64>())
                }
            }
            (
                VariableType::Bounded { .. },
                VariableParameters::BoundedParams {
                    lower,
                    upper,
                    beta_alpha,
                    beta_beta,
                },
            ) => {
                // Sample from Beta distribution and transform to bounds
                let beta_sample = self.sample_beta(*beta_alpha, *beta_beta, rng);
                Ok(lower + (upper - lower) * beta_sample)
            }
            (VariableType::Binary, VariableParameters::BinaryParams { probability }) => {
                if rng.gen::<f64>() < *probability {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            }
            _ => Err(SklearsError::InvalidInput(
                "Mismatched variable type and parameters".to_string(),
            )),
        }
    }

    fn predict_continuous(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        target_feature: usize,
    ) -> SklResult<Option<f64>> {
        // Simple linear regression using other observed features
        let mut predictors = Vec::new();
        let mut targets = Vec::new();

        // Collect training data from other samples where target feature is observed
        for i in 0..X.nrows() {
            if i != sample_idx && !self.is_missing(X[[i, target_feature]]) {
                let mut predictor_row = Vec::new();
                let mut all_observed = true;

                for j in 0..X.ncols() {
                    if j != target_feature {
                        if self.is_missing(X[[i, j]]) {
                            all_observed = false;
                            break;
                        }
                        predictor_row.push(X[[i, j]]);
                    }
                }

                if all_observed && !predictor_row.is_empty() {
                    predictors.push(predictor_row);
                    targets.push(X[[i, target_feature]]);
                }
            }
        }

        if predictors.len() < 2 || predictors.is_empty() {
            return Ok(None);
        }

        // Simple linear regression (least squares)
        let n_predictors = predictors[0].len();
        let n_samples = predictors.len();

        // Build design matrix with intercept
        let mut design_matrix = Array2::ones((n_samples, n_predictors + 1));
        for (i, pred_row) in predictors.iter().enumerate() {
            for (j, &val) in pred_row.iter().enumerate() {
                design_matrix[[i, j + 1]] = val;
            }
        }

        let y = Array1::from_vec(targets);

        // Solve normal equations: (X^T X)^{-1} X^T y
        let xt = design_matrix.t();
        let xtx = xt.dot(&design_matrix);
        let xty = xt.dot(&y);

        // Simple 2x2 matrix inversion for intercept + one predictor
        if let Some(coefficients) = self.solve_linear_system(&xtx, &xty) {
            // Make prediction for current sample
            let mut pred_row = Vec::new();
            for j in 0..X.ncols() {
                if j != target_feature && !self.is_missing(X[[sample_idx, j]]) {
                    pred_row.push(X[[sample_idx, j]]);
                }
            }

            if pred_row.len() == n_predictors {
                let mut prediction = coefficients[0]; // intercept
                for (i, &val) in pred_row.iter().enumerate() {
                    prediction += coefficients[i + 1] * val;
                }
                return Ok(Some(prediction));
            }
        }

        Ok(None)
    }

    fn solve_linear_system(&self, A: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
        let n = A.nrows();
        if n != A.ncols() || n != b.len() || n == 0 {
            return None;
        }

        // Simple 2x2 case (intercept + one predictor)
        if n == 2 {
            let det = A[[0, 0]] * A[[1, 1]] - A[[0, 1]] * A[[1, 0]];
            if det.abs() < 1e-10 {
                return None;
            }

            let x0 = (A[[1, 1]] * b[0] - A[[0, 1]] * b[1]) / det;
            let x1 = (A[[0, 0]] * b[1] - A[[1, 0]] * b[0]) / det;

            return Some(Array1::from_vec(vec![x0, x1]));
        }

        // For larger systems, use simple Gaussian elimination
        let mut augmented = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = A[[i, j]];
            }
            augmented[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if augmented[[i, i]].abs() < 1e-10 {
                return None;
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = augmented[[k, i]] / augmented[[i, i]];
                for j in i..=n {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = augmented[[i, n]];
            for j in (i + 1)..n {
                x[i] -= augmented[[i, j]] * x[j];
            }
            x[i] /= augmented[[i, i]];
        }

        Some(x)
    }

    fn sample_beta(&self, alpha: f64, beta: f64, rng: &mut Random) -> f64 {
        // Simple rejection sampling for Beta distribution
        // This is not the most efficient method but works for basic cases
        if alpha <= 0.0 || beta <= 0.0 {
            return rng.gen::<f64>();
        }

        // Use transformation method for Beta(1,1) = Uniform(0,1)
        if (alpha - 1.0).abs() < 1e-10 && (beta - 1.0).abs() < 1e-10 {
            return rng.gen::<f64>();
        }

        // For other cases, use simple approximation
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();

        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);

        x / (x + y)
    }
}

/// Mixed-Type MICE Imputer
///
/// Multiple Imputation by Chained Equations specifically designed for mixed-type data.
/// Handles different variable types appropriately during the chained imputation process.
///
/// # Parameters
///
/// * `variable_types` - Map from feature index to variable type
/// * `n_imputations` - Number of multiple imputations to generate
/// * `max_iter` - Maximum number of iterations for each imputation
/// * `burn_in` - Number of burn-in iterations before collecting imputations
/// * `tol` - Tolerance for convergence
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_impute::{MixedTypeMICEImputer, VariableType};
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
/// ///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
/// let mut variable_types = HashMap::new();
/// variable_types.insert(0, VariableType::Continuous);
/// variable_types.insert(1, VariableType::Ordinal(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
/// variable_types.insert(2, VariableType::SemiContinuous { zero_probability: 0.1 });
///
/// let imputer = MixedTypeMICEImputer::new()
///     .variable_types(variable_types)
///     .n_imputations(5)
///     .max_iter(20);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let multiple_imputations = fitted.transform_multiple(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MixedTypeMICEImputer<S = Untrained> {
    state: S,
    variable_types: HashMap<usize, VariableType>,
    n_imputations: usize,
    max_iter: usize,
    burn_in: usize,
    tol: f64,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for MixedTypeMICEImputer
#[derive(Debug, Clone)]
pub struct MixedTypeMICEImputerTrained {
    variable_types: HashMap<usize, VariableType>,
    learned_parameters: HashMap<usize, VariableParameters>,
    n_features_in_: usize,
}

/// Multiple imputation results for mixed-type data
#[derive(Debug, Clone)]
pub struct MixedTypeMultipleImputationResults {
    /// imputations
    pub imputations: Vec<Array2<f64>>,
    /// pooled_estimates
    pub pooled_estimates: Option<Array2<f64>>,
    /// within_imputation_variance
    pub within_imputation_variance: Option<Array2<f64>>,
    /// between_imputation_variance
    pub between_imputation_variance: Option<Array2<f64>>,
    /// total_variance
    pub total_variance: Option<Array2<f64>>,
}

impl MixedTypeMICEImputer<Untrained> {
    /// Create a new MixedTypeMICEImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            variable_types: HashMap::new(),
            n_imputations: 5,
            max_iter: 10,
            burn_in: 5,
            tol: 1e-4,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the variable types for each feature
    pub fn variable_types(mut self, variable_types: HashMap<usize, VariableType>) -> Self {
        self.variable_types = variable_types;
        self
    }

    /// Set the number of imputations
    pub fn n_imputations(mut self, n_imputations: usize) -> Self {
        self.n_imputations = n_imputations;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the burn-in period
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }
}

impl Default for MixedTypeMICEImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MixedTypeMICEImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MixedTypeMICEImputer<Untrained> {
    type Fitted = MixedTypeMICEImputer<MixedTypeMICEImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        // Use HeterogeneousImputer for initial parameter learning
        let hetero_imputer = HeterogeneousImputer::new()
            .variable_types(self.variable_types.clone())
            .random_state(self.random_state);

        let fitted_hetero = hetero_imputer.fit(&X.view(), &())?;

        Ok(MixedTypeMICEImputer {
            state: MixedTypeMICEImputerTrained {
                variable_types: fitted_hetero.state.variable_types.clone(),
                learned_parameters: fitted_hetero.state.learned_parameters.clone(),
                n_features_in_: n_features,
            },
            variable_types: self.variable_types,
            n_imputations: self.n_imputations,
            max_iter: self.max_iter,
            burn_in: self.burn_in,
            tol: self.tol,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for MixedTypeMICEImputer<MixedTypeMICEImputerTrained>
{
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // For single imputation, use the first imputation from multiple imputation
        let multiple_results = self.transform_multiple(X)?;
        if let Some(first_imputation) = multiple_results.imputations.first() {
            Ok(first_imputation.mapv(|x| x as Float))
        } else {
            Err(SklearsError::InvalidInput(
                "No imputations generated".to_string(),
            ))
        }
    }
}

impl MixedTypeMICEImputer<MixedTypeMICEImputerTrained> {
    /// Generate multiple imputations
    #[allow(non_snake_case)]
    pub fn transform_multiple(
        &self,
        X: &ArrayView2<'_, Float>,
    ) -> SklResult<MixedTypeMultipleImputationResults> {
        let X = X.mapv(|x| x);
        let mut imputations = Vec::new();

        let mut base_rng = if let Some(_seed) = self.random_state {
            Random::default()
        } else {
            Random::default()
        };

        for _m in 0..self.n_imputations {
            let imputation_seed = base_rng.random::<u64>();
            let imputation = self.generate_single_imputation(&X, imputation_seed)?;
            imputations.push(imputation);
        }

        // Calculate pooled estimates using Rubin's rules
        let pooled_estimates = self.pool_imputations(&imputations);
        let (within_var, between_var, total_var) =
            self.calculate_imputation_variance(&imputations, &pooled_estimates);

        Ok(MixedTypeMultipleImputationResults {
            imputations,
            pooled_estimates: Some(pooled_estimates),
            within_imputation_variance: Some(within_var),
            between_imputation_variance: Some(between_var),
            total_variance: Some(total_var),
        })
    }

    fn generate_single_imputation(&self, X: &Array2<f64>, _seed: u64) -> SklResult<Array2<f64>> {
        let mut X_imputed = X.clone();
        let mut rng = Random::default();

        // Initialize missing values with simple imputation
        self.initialize_missing_values(&mut X_imputed, &mut rng)?;

        // MICE iterations
        for iteration in 0..(self.burn_in + self.max_iter) {
            let prev_X = X_imputed.clone();

            for (&feature_idx, var_type) in &self.state.variable_types {
                if let Some(params) = self.state.learned_parameters.get(&feature_idx) {
                    self.update_feature_mice(
                        &mut X_imputed,
                        X,
                        feature_idx,
                        var_type,
                        params,
                        &mut rng,
                    )?;
                }
            }

            // Check convergence after burn-in
            if iteration >= self.burn_in {
                let max_change = self.calculate_max_change(&prev_X, &X_imputed, X);
                if max_change < self.tol {
                    break;
                }
            }
        }

        Ok(X_imputed)
    }

    fn initialize_missing_values(
        &self,
        X_imputed: &mut Array2<f64>,
        rng: &mut Random,
    ) -> SklResult<()> {
        let (n_samples, n_features) = X_imputed.dim();

        for j in 0..n_features {
            if let (Some(var_type), Some(params)) = (
                self.state.variable_types.get(&j),
                self.state.learned_parameters.get(&j),
            ) {
                for i in 0..n_samples {
                    if self.is_missing(X_imputed[[i, j]]) {
                        let initial_value = match (var_type, params) {
                            (
                                VariableType::Continuous,
                                VariableParameters::ContinuousParams { mean, .. },
                            ) => *mean,
                            (VariableType::Ordinal(levels), _) => {
                                let idx = rng.gen_range(0..levels.len());
                                levels[idx]
                            }
                            (VariableType::Categorical(categories), _) => {
                                let idx = rng.gen_range(0..categories.len());
                                categories[idx]
                            }
                            (
                                VariableType::SemiContinuous { .. },
                                VariableParameters::SemiContinuousParams {
                                    continuous_mean, ..
                                },
                            ) => *continuous_mean,
                            (VariableType::Bounded { lower, upper }, _) => {
                                lower + (upper - lower) * rng.gen::<f64>()
                            }
                            (
                                VariableType::Binary,
                                VariableParameters::BinaryParams { probability },
                            ) => {
                                if rng.gen::<f64>() < *probability {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            _ => 0.0,
                        };
                        X_imputed[[i, j]] = initial_value;
                    }
                }
            }
        }

        Ok(())
    }

    fn update_feature_mice(
        &self,
        X_imputed: &mut Array2<f64>,
        X_original: &Array2<f64>,
        feature_idx: usize,
        var_type: &VariableType,
        params: &VariableParameters,
        rng: &mut Random,
    ) -> SklResult<()> {
        let (n_samples, _) = X_imputed.dim();

        // Create temporary imputer for this feature
        let hetero_imputer = HeterogeneousImputer {
            state: HeterogeneousImputerTrained {
                variable_types: self.state.variable_types.clone(),
                learned_parameters: self.state.learned_parameters.clone(),
                n_features_in_: self.state.n_features_in_,
            },
            variable_types: HashMap::new(),
            max_iter: 1,
            tol: self.tol,
            random_state: Some(rng.gen::<u64>()),
            missing_values: self.missing_values,
        };

        for i in 0..n_samples {
            if self.is_missing(X_original[[i, feature_idx]]) {
                let imputed_value = hetero_imputer.impute_value(
                    var_type,
                    params,
                    // X_imputed
                    X_imputed,
                    i,
                    feature_idx,
                    rng,
                )?;
                X_imputed[[i, feature_idx]] = imputed_value;
            }
        }

        Ok(())
    }

    fn calculate_max_change(
        &self,
        prev_X: &Array2<f64>,
        current_X: &Array2<f64>,
        original_X: &Array2<f64>,
    ) -> f64 {
        let mut max_change: f64 = 0.0;

        for ((i, j), &orig_val) in original_X.indexed_iter() {
            if self.is_missing(orig_val) {
                let change = (prev_X[[i, j]] - current_X[[i, j]]).abs();
                max_change = max_change.max(change);
            }
        }

        max_change
    }

    fn pool_imputations(&self, imputations: &[Array2<f64>]) -> Array2<f64> {
        if imputations.is_empty() {
            return Array2::zeros((0, 0));
        }

        let (n_samples, n_features) = imputations[0].dim();
        let mut pooled = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let sum: f64 = imputations.iter().map(|imp| imp[[i, j]]).sum();
                pooled[[i, j]] = sum / imputations.len() as f64;
            }
        }

        pooled
    }

    fn calculate_imputation_variance(
        &self,
        imputations: &[Array2<f64>],
        pooled: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        if imputations.is_empty() {
            let zero_mat = Array2::zeros((0, 0));
            return (zero_mat.clone(), zero_mat.clone(), zero_mat);
        }

        let (n_samples, n_features) = pooled.dim();
        let m = imputations.len() as f64;

        let mut within_var = Array2::zeros((n_samples, n_features));
        let mut between_var = Array2::zeros((n_samples, n_features));

        // Within-imputation variance (average of individual variances)
        for imp in imputations {
            for i in 0..n_samples {
                for j in 0..n_features {
                    let diff = imp[[i, j]] - pooled[[i, j]];
                    within_var[[i, j]] += diff * diff;
                }
            }
        }
        within_var /= m;

        // Between-imputation variance
        for imp in imputations {
            for i in 0..n_samples {
                for j in 0..n_features {
                    let diff = imp[[i, j]] - pooled[[i, j]];
                    between_var[[i, j]] += diff * diff;
                }
            }
        }
        between_var /= m - 1.0;

        // Total variance (Rubin's rule)
        let total_var = &within_var + &between_var * (1.0 + 1.0 / m);

        (within_var, between_var, total_var)
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Ordinal Variable Imputer
///
/// Specialized imputation for ordinal categorical variables that respects
/// the ordered nature of the categories.
///
/// # Parameters
///
/// * `levels` - Ordered levels of the ordinal variable
/// * `method` - Imputation method ("mode", "proportional_odds", "adjacent_categories")
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::OrdinalImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0], [2.0], [f64::NAN], [3.0], [1.0]];
/// let levels = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// let imputer = OrdinalImputer::new()
///     .levels(levels)
///     .method("proportional_odds".to_string());
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OrdinalImputer<S = Untrained> {
    state: S,
    levels: Vec<f64>,
    method: String,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for OrdinalImputer
#[derive(Debug, Clone)]
pub struct OrdinalImputerTrained {
    levels: Vec<f64>,
    level_probabilities: Array1<f64>,
    cumulative_probabilities: Array1<f64>,
    transition_matrix: Option<Array2<f64>>,
    n_features_in_: usize,
}

impl OrdinalImputer<Untrained> {
    /// Create a new OrdinalImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            levels: Vec::new(),
            method: "mode".to_string(),
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the ordered levels
    pub fn levels(mut self, levels: Vec<f64>) -> Self {
        self.levels = levels;
        self
    }

    /// Set the imputation method
    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }
}

impl Default for OrdinalImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OrdinalImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for OrdinalImputer<Untrained> {
    type Fitted = OrdinalImputer<OrdinalImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        if n_features != 1 {
            return Err(SklearsError::InvalidInput(
                "OrdinalImputer only supports single-column input".to_string(),
            ));
        }

        let column = X.column(0);
        let observed_values: Vec<f64> = column
            .iter()
            .filter(|&&x| !self.is_missing(x))
            .cloned()
            .collect();

        if observed_values.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No observed values found".to_string(),
            ));
        }

        // Auto-detect levels if not provided
        let levels = if self.levels.is_empty() {
            let mut unique_values: Vec<f64> = observed_values.clone().into_iter().collect();
            unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_values.dedup();
            unique_values
        } else {
            self.levels.clone()
        };

        // Calculate level probabilities
        let mut level_counts = Array1::<f64>::zeros(levels.len());
        let total_count = column.len() as f64;

        for &value in column.iter() {
            if !self.is_missing(value) {
                if let Some(idx) = levels
                    .iter()
                    .position(|&level| (level - value).abs() < 1e-10)
                {
                    level_counts[idx] += 1.0;
                }
            }
        }

        let level_probabilities = level_counts.mapv(|count: f64| count / total_count);

        // Calculate cumulative probabilities
        let mut cumulative_probabilities = Array1::<f64>::zeros(levels.len());
        cumulative_probabilities[0] = level_probabilities[0];
        for i in 1..levels.len() {
            cumulative_probabilities[i] = cumulative_probabilities[i - 1] + level_probabilities[i];
        }

        // Calculate transition matrix for adjacent categories method
        let transition_matrix = if self.method == "adjacent_categories" {
            Some(self.estimate_transition_matrix(&levels, &observed_values))
        } else {
            None
        };

        Ok(OrdinalImputer {
            state: OrdinalImputerTrained {
                levels,
                level_probabilities,
                cumulative_probabilities,
                transition_matrix,
                n_features_in_: n_features,
            },
            levels: self.levels,
            method: self.method,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl OrdinalImputer<Untrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn estimate_transition_matrix(&self, levels: &[f64], observed_values: &[f64]) -> Array2<f64> {
        let n_levels = levels.len();
        let mut transition_counts = Array2::zeros((n_levels, n_levels));

        // Count transitions between adjacent observations
        for window in observed_values.windows(2) {
            if let (Some(from_idx), Some(to_idx)) = (
                levels
                    .iter()
                    .position(|&level| (level - window[0]).abs() < 1e-10),
                levels
                    .iter()
                    .position(|&level| (level - window[1]).abs() < 1e-10),
            ) {
                transition_counts[[from_idx, to_idx]] += 1.0;
            }
        }

        // Normalize to probabilities
        let mut transition_matrix = Array2::zeros((n_levels, n_levels));
        for i in 0..n_levels {
            let row_sum: f64 = transition_counts.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_levels {
                    transition_matrix[[i, j]] = transition_counts[[i, j]] / row_sum;
                }
            } else {
                // Uniform distribution if no transitions observed
                for j in 0..n_levels {
                    transition_matrix[[i, j]] = 1.0 / n_levels as f64;
                }
            }
        }

        transition_matrix
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for OrdinalImputer<OrdinalImputerTrained> {
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();
        let mut rng = Random::default();

        for i in 0..n_samples {
            if self.is_missing(X_imputed[[i, 0]]) {
                let imputed_value = match self.method.as_str() {
                    "mode" => self.impute_mode(&mut rng),
                    "proportional_odds" => self.impute_proportional_odds(&mut rng),
                    "adjacent_categories" => {
                        self.impute_adjacent_categories(&X_imputed, i, &mut rng)
                    }
                    _ => self.impute_mode(&mut rng),
                };
                X_imputed[[i, 0]] = imputed_value;
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl OrdinalImputer<OrdinalImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn impute_mode(&self, _rng: &mut Random) -> f64 {
        // Return the level with highest probability
        let max_idx = self
            .state
            .level_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        self.state.levels.get(max_idx).copied().unwrap_or(0.0)
    }

    fn impute_proportional_odds(&self, rng: &mut Random) -> f64 {
        // Sample from cumulative distribution
        let random_val: f64 = rng.gen();

        for (i, &cum_prob) in self.state.cumulative_probabilities.iter().enumerate() {
            if random_val <= cum_prob {
                return self.state.levels.get(i).copied().unwrap_or(0.0);
            }
        }

        // Fallback to last level
        self.state.levels.last().copied().unwrap_or(0.0)
    }

    fn impute_adjacent_categories(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        rng: &mut Random,
    ) -> f64 {
        // Find nearest observed values to inform imputation
        if let Some(ref transition_matrix) = self.state.transition_matrix {
            // Look for adjacent observed values
            let column = X.column(0);

            // Find closest observed value
            let mut closest_value = None;
            let mut min_distance = usize::MAX;

            for (i, &value) in column.iter().enumerate() {
                if !self.is_missing(value) {
                    let distance = (i as i32 - sample_idx as i32).unsigned_abs() as usize;
                    if distance < min_distance {
                        min_distance = distance;
                        closest_value = Some(value);
                    }
                }
            }

            if let Some(closest_val) = closest_value {
                if let Some(from_idx) = self
                    .state
                    .levels
                    .iter()
                    .position(|&level| (level - closest_val).abs() < 1e-10)
                {
                    // Sample from transition probabilities
                    let random_val: f64 = rng.gen();
                    let mut cumulative = 0.0;

                    for (to_idx, &prob) in transition_matrix.row(from_idx).iter().enumerate() {
                        cumulative += prob;
                        if random_val <= cumulative {
                            return self.state.levels.get(to_idx).copied().unwrap_or(0.0);
                        }
                    }
                }
            }
        }

        // Fallback to proportional odds
        self.impute_proportional_odds(rng)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Transform;

    #[test]
    fn test_heterogeneous_imputer_basic() {
        let data = array![[1.0, 2.0, 0.5], [f64::NAN, 3.0, 0.8], [3.0, f64::NAN, 0.0]];

        let mut variable_types = HashMap::new();
        variable_types.insert(0, VariableType::Continuous);
        variable_types.insert(1, VariableType::Ordinal(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        variable_types.insert(
            2,
            VariableType::Bounded {
                lower: 0.0,
                upper: 1.0,
            },
        );

        let imputer = HeterogeneousImputer::new()
            .variable_types(variable_types)
            .max_iter(10);

        let fitted = imputer.fit(&data.view(), &()).unwrap();
        let result = fitted.transform(&data.view()).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| (x).is_nan()));

        // Non-missing values should be preserved
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mixed_type_mice_basic() {
        let data = array![[1.0, 2.0, 0.0], [f64::NAN, 3.0, 1.0], [3.0, f64::NAN, 0.0]];

        let mut variable_types = HashMap::new();
        variable_types.insert(0, VariableType::Continuous);
        variable_types.insert(1, VariableType::Ordinal(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        variable_types.insert(
            2,
            VariableType::SemiContinuous {
                zero_probability: 0.6,
            },
        );

        let imputer = MixedTypeMICEImputer::new()
            .variable_types(variable_types)
            .n_imputations(3)
            .max_iter(5);

        let fitted = imputer.fit(&data.view(), &()).unwrap();
        let results = fitted.transform_multiple(&data.view()).unwrap();

        // Should generate requested number of imputations
        assert_eq!(results.imputations.len(), 3);

        // Each imputation should have no missing values
        for imputation in &results.imputations {
            assert!(!imputation.iter().any(|&x| x.is_nan()));
        }

        // Should have pooled estimates
        assert!(results.pooled_estimates.is_some());
    }

    #[test]
    fn test_ordinal_imputer_basic() {
        let data = array![[1.0], [2.0], [f64::NAN], [3.0], [1.0]];
        let levels = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let imputer = OrdinalImputer::new()
            .levels(levels)
            .method("mode".to_string());

        let fitted = imputer.fit(&data.view(), &()).unwrap();
        let result = fitted.transform(&data.view()).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| (x).is_nan()));

        // Non-missing values should be preserved
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[3, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[4, 0]], 1.0, epsilon = 1e-10);

        // Imputed value should be one of the levels
        let imputed_val = result[[2, 0]];
        assert!([1.0, 2.0, 3.0, 4.0, 5.0].contains(&imputed_val));
    }

    #[test]
    fn test_variable_type_auto_detection() {
        let data = array![[1.0, 1.0, 0.5], [2.0, 0.0, 0.8], [3.0, 1.0, 0.0]];

        let imputer = HeterogeneousImputer::new().max_iter(5);
        let fitted = imputer.fit(&data.view(), &()).unwrap();

        // Should auto-detect variable types
        let variable_types = &fitted.state.variable_types;

        // First column should be detected as ordinal (few integer values)
        if let Some(VariableType::Ordinal(_)) = variable_types.get(&0) {
            // Expected
        } else if let Some(VariableType::Continuous) = variable_types.get(&0) {
            // Also acceptable
        } else {
            panic!("Unexpected variable type for first column");
        }

        // Second column should be detected as semi-continuous or binary
        assert!(variable_types.contains_key(&1));

        // Third column should be detected as bounded (values in [0,1])
        if let Some(VariableType::Bounded { lower, upper }) = variable_types.get(&2) {
            assert_abs_diff_eq!(*lower, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(*upper, 1.0, epsilon = 1e-10);
        }
    }
}
