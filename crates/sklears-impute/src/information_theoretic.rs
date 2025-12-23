//! Information-theoretic imputation methods
//!
//! This module provides imputation strategies based on information theory,
//! including mutual information, entropy, minimum description length, and maximum entropy methods.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Mutual Information Imputer
///
/// Uses mutual information to select the most informative features for imputing each missing value.
/// This method identifies the features that are most dependent on the target feature and uses
/// them for prediction.
///
/// # Parameters
///
/// * `k_features` - Number of most informative features to use for imputation
/// * `discretization_bins` - Number of bins for discretizing continuous variables for MI calculation
/// * `imputation_method` - Method to use for final imputation ('knn', 'regression', 'mean')
/// * `mi_estimation_method` - Method for MI estimation ('histogram', 'kde')
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::MutualInformationImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = MutualInformationImputer::new()
///     .k_features(2)
///     .discretization_bins(10);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MutualInformationImputer<S = Untrained> {
    state: S,
    k_features: usize,
    discretization_bins: usize,
    imputation_method: String,
    mi_estimation_method: String,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for MutualInformationImputer
#[derive(Debug, Clone)]
pub struct MutualInformationImputerTrained {
    feature_mi_rankings: HashMap<usize, Vec<usize>>,
    feature_means_: Array1<f64>,
    feature_stds_: Array1<f64>,
    bin_edges_: Vec<Array1<f64>>,
    n_features_in_: usize,
}

/// Entropy-based Imputer
///
/// Uses entropy-based methods to impute missing values by maximizing the entropy
/// of the completed dataset while respecting observed constraints.
///
/// # Parameters
///
/// * `max_iter` - Maximum number of iterations for optimization
/// * `tol` - Tolerance for convergence
/// * `regularization` - Regularization parameter for entropy maximization
/// * `constraint_weight` - Weight for observed data constraints
/// * `random_state` - Random state for reproducibility
#[derive(Debug, Clone)]
pub struct EntropyImputer<S = Untrained> {
    state: S,
    max_iter: usize,
    tol: f64,
    regularization: f64,
    constraint_weight: f64,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for EntropyImputer
#[derive(Debug, Clone)]
pub struct EntropyImputerTrained {
    feature_distributions_: Vec<Array1<f64>>,
    feature_means_: Array1<f64>,
    feature_stds_: Array1<f64>,
    entropy_weights_: Array2<f64>,
    n_features_in_: usize,
}

/// Minimum Description Length Imputer
///
/// Uses the Minimum Description Length (MDL) principle to select the simplest
/// model that adequately describes the data for imputation purposes.
///
/// # Parameters
///
/// * `model_complexity_penalty` - Penalty for model complexity in MDL calculation
/// * `max_models` - Maximum number of models to consider
/// * `validation_fraction` - Fraction of data to use for model validation
/// * `random_state` - Random state for reproducibility
#[derive(Debug, Clone)]
pub struct MDLImputer<S = Untrained> {
    state: S,
    model_complexity_penalty: f64,
    max_models: usize,
    validation_fraction: f64,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for MDLImputer
#[derive(Debug, Clone)]
pub struct MDLImputerTrained {
    selected_models_: HashMap<usize, MDLModel>,
    feature_means_: Array1<f64>,
    feature_stds_: Array1<f64>,
    n_features_in_: usize,
}

/// MDL Model representation
#[derive(Debug, Clone)]
pub struct MDLModel {
    model_type: String,
    parameters: Array1<f64>,
    description_length: f64,
    predictor_features: Vec<usize>,
}

/// Maximum Entropy Imputer
///
/// Imputes missing values by finding the maximum entropy distribution
/// that is consistent with the observed data constraints.
///
/// # Parameters
///
/// * `lagrange_multipliers` - Number of Lagrange multipliers for constraints
/// * `max_iter` - Maximum number of iterations for optimization
/// * `tol` - Tolerance for convergence
/// * `constraint_types` - Types of constraints to enforce ('mean', 'variance', 'covariance')
/// * `random_state` - Random state for reproducibility
#[derive(Debug, Clone)]
pub struct MaxEntropyImputer<S = Untrained> {
    state: S,
    lagrange_multipliers: usize,
    max_iter: usize,
    tol: f64,
    constraint_types: Vec<String>,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for MaxEntropyImputer
#[derive(Debug, Clone)]
pub struct MaxEntropyImputerTrained {
    lagrange_params_: Array1<f64>,
    constraint_values_: Array1<f64>,
    feature_means_: Array1<f64>,
    feature_stds_: Array1<f64>,
    covariance_matrix_: Array2<f64>,
    n_features_in_: usize,
}

// MutualInformationImputer implementation

impl MutualInformationImputer<Untrained> {
    /// Create a new MutualInformationImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            k_features: 3,
            discretization_bins: 10,
            imputation_method: "knn".to_string(),
            mi_estimation_method: "histogram".to_string(),
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the number of most informative features to use
    pub fn k_features(mut self, k_features: usize) -> Self {
        self.k_features = k_features;
        self
    }

    /// Set the number of discretization bins
    pub fn discretization_bins(mut self, bins: usize) -> Self {
        self.discretization_bins = bins;
        self
    }

    /// Set the imputation method
    pub fn imputation_method(mut self, method: String) -> Self {
        self.imputation_method = method;
        self
    }

    /// Set the MI estimation method
    pub fn mi_estimation_method(mut self, method: String) -> Self {
        self.mi_estimation_method = method;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
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

impl Default for MutualInformationImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MutualInformationImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MutualInformationImputer<Untrained> {
    type Fitted = MutualInformationImputer<MutualInformationImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        // Compute feature statistics
        let feature_means = compute_feature_means(&X, self.missing_values);
        let feature_stds = compute_feature_stds(&X, &feature_means, self.missing_values);

        // Create bin edges for discretization
        let bin_edges = compute_bin_edges(&X, self.discretization_bins, self.missing_values)?;

        // Compute mutual information rankings for each feature
        let mut feature_mi_rankings = HashMap::new();

        for target_feature in 0..n_features {
            let mut mi_scores = Vec::new();

            for other_feature in 0..n_features {
                if other_feature != target_feature {
                    let mi_score = compute_mutual_information(
                        &X,
                        target_feature,
                        other_feature,
                        &bin_edges,
                        self.missing_values,
                    )?;
                    mi_scores.push((mi_score, other_feature));
                }
            }

            // Sort by MI score (descending)
            mi_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Take top k features
            let top_features: Vec<usize> = mi_scores
                .into_iter()
                .take(self.k_features)
                .map(|(_, idx)| idx)
                .collect();

            feature_mi_rankings.insert(target_feature, top_features);
        }

        Ok(MutualInformationImputer {
            state: MutualInformationImputerTrained {
                feature_mi_rankings,
                feature_means_: feature_means,
                feature_stds_: feature_stds,
                bin_edges_: bin_edges,
                n_features_in_: n_features,
            },
            k_features: self.k_features,
            discretization_bins: self.discretization_bins,
            imputation_method: self.imputation_method,
            mi_estimation_method: self.mi_estimation_method,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for MutualInformationImputer<MutualInformationImputerTrained>
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

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    // Get the most informative features for this target feature
                    if let Some(informative_features) = self.state.feature_mi_rankings.get(&j) {
                        let imputed_value = match self.imputation_method.as_str() {
                            "knn" => {
                                self.impute_with_knn(&X_imputed, i, j, informative_features)?
                            }
                            "regression" => {
                                self.impute_with_regression(&X_imputed, i, j, informative_features)?
                            }
                            _ => self.state.feature_means_[j], // Fallback to mean
                        };
                        X_imputed[[i, j]] = imputed_value;
                    } else {
                        X_imputed[[i, j]] = self.state.feature_means_[j];
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl MutualInformationImputer<MutualInformationImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn impute_with_knn(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        feature_idx: usize,
        informative_features: &[usize],
    ) -> SklResult<f64> {
        let k = 3; // Number of neighbors
        let mut distances = Vec::new();

        let target_sample = X.row(sample_idx);

        for other_idx in 0..X.nrows() {
            if other_idx != sample_idx && !self.is_missing(X[[other_idx, feature_idx]]) {
                let other_sample = X.row(other_idx);

                // Compute distance using only informative features
                let mut distance = 0.0;
                let mut valid_features = 0;

                for &feat_idx in informative_features {
                    if !self.is_missing(target_sample[feat_idx])
                        && !self.is_missing(other_sample[feat_idx])
                    {
                        let diff = target_sample[feat_idx] - other_sample[feat_idx];
                        distance += diff * diff;
                        valid_features += 1;
                    }
                }

                if valid_features > 0 {
                    distance = (distance / valid_features as f64).sqrt();
                    distances.push((distance, other_idx));
                }
            }
        }

        if distances.is_empty() {
            return Ok(self.state.feature_means_[feature_idx]);
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k_nearest = distances.into_iter().take(k);

        // Weighted average of k nearest neighbors
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (distance, neighbor_idx) in k_nearest {
            let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
            weighted_sum += weight * X[[neighbor_idx, feature_idx]];
            total_weight += weight;
        }

        Ok(if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            self.state.feature_means_[feature_idx]
        })
    }

    fn impute_with_regression(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        feature_idx: usize,
        informative_features: &[usize],
    ) -> SklResult<f64> {
        // Simple linear regression using informative features
        let mut train_X = Vec::new();
        let mut train_y = Vec::new();

        // Collect training data
        for i in 0..X.nrows() {
            if i != sample_idx && !self.is_missing(X[[i, feature_idx]]) {
                let mut all_informative_observed = true;
                for &feat_idx in informative_features {
                    if self.is_missing(X[[i, feat_idx]]) {
                        all_informative_observed = false;
                        break;
                    }
                }

                if all_informative_observed {
                    let features: Vec<f64> = informative_features
                        .iter()
                        .map(|&idx| X[[i, idx]])
                        .collect();
                    train_X.push(features);
                    train_y.push(X[[i, feature_idx]]);
                }
            }
        }

        if train_X.is_empty() || train_X[0].is_empty() {
            return Ok(self.state.feature_means_[feature_idx]);
        }

        // Fit simple linear regression (using normal equations)
        let n = train_X.len();
        let p = train_X[0].len();

        // Create design matrix
        let mut design_matrix = Array2::ones((n, p + 1)); // +1 for intercept
        for (i, row) in train_X.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                design_matrix[[i, j + 1]] = val;
            }
        }

        let y = Array1::from_vec(train_y);

        // Solve normal equations: β = (X^T X)^(-1) X^T y
        let xt = design_matrix.t();
        let xtx = xt.dot(&design_matrix);
        let xty = xt.dot(&y);

        // Simple pseudo-inverse for small matrices
        let beta = solve_linear_system(&xtx, &xty)?;

        // Predict for target sample
        let mut prediction = beta[0]; // intercept
        for (i, &feat_idx) in informative_features.iter().enumerate() {
            let value = if self.is_missing(X[[sample_idx, feat_idx]]) {
                self.state.feature_means_[feat_idx]
            } else {
                X[[sample_idx, feat_idx]]
            };
            prediction += beta[i + 1] * value;
        }

        Ok(prediction)
    }
}

// EntropyImputer implementation

impl EntropyImputer<Untrained> {
    /// Create a new EntropyImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_iter: 100,
            tol: 1e-6,
            regularization: 0.01,
            constraint_weight: 1.0,
            random_state: None,
            missing_values: f64::NAN,
        }
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

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the constraint weight
    pub fn constraint_weight(mut self, weight: f64) -> Self {
        self.constraint_weight = weight;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
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

impl Default for EntropyImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EntropyImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for EntropyImputer<Untrained> {
    type Fitted = EntropyImputer<EntropyImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        // Compute feature statistics
        let feature_means = compute_feature_means(&X, self.missing_values);
        let feature_stds = compute_feature_stds(&X, &feature_means, self.missing_values);

        // Estimate feature distributions for entropy calculation
        let mut feature_distributions = Vec::new();
        let n_bins = 20;

        for j in 0..n_features {
            let column = X.column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if !valid_values.is_empty() {
                let histogram = compute_histogram(&valid_values, n_bins);
                feature_distributions.push(histogram);
            } else {
                feature_distributions.push(Array1::ones(n_bins) / n_bins as f64);
            }
        }

        // Initialize entropy weights
        let entropy_weights = Array2::eye(n_features) * self.regularization;

        Ok(EntropyImputer {
            state: EntropyImputerTrained {
                feature_distributions_: feature_distributions,
                feature_means_: feature_means,
                feature_stds_: feature_stds,
                entropy_weights_: entropy_weights,
                n_features_in_: n_features,
            },
            max_iter: self.max_iter,
            tol: self.tol,
            regularization: self.regularization,
            constraint_weight: self.constraint_weight,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for EntropyImputer<EntropyImputerTrained> {
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

        let mut rng = Random::default();

        let mut X_imputed = X.clone();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    // Sample from the learned distribution for this feature
                    let distribution = &self.state.feature_distributions_[j];
                    let sampled_value = sample_from_distribution(
                        distribution,
                        self.state.feature_means_[j],
                        self.state.feature_stds_[j],
                        &mut rng,
                    )?;
                    X_imputed[[i, j]] = sampled_value;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl EntropyImputer<EntropyImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

// Helper functions

fn compute_feature_means(X: &Array2<f64>, missing_values: f64) -> Array1<f64> {
    let (_, n_features) = X.dim();
    let mut means = Array1::zeros(n_features);

    let is_missing_nan = missing_values.is_nan();

    for j in 0..n_features {
        let column = X.column(j);
        let valid_values: Vec<f64> = column
            .iter()
            .filter(|&&x| {
                if is_missing_nan {
                    !x.is_nan()
                } else {
                    (x - missing_values).abs() >= f64::EPSILON
                }
            })
            .cloned()
            .collect();

        means[j] = if valid_values.is_empty() {
            0.0
        } else {
            valid_values.iter().sum::<f64>() / valid_values.len() as f64
        };
    }

    means
}

fn compute_feature_stds(X: &Array2<f64>, means: &Array1<f64>, missing_values: f64) -> Array1<f64> {
    let (_, n_features) = X.dim();
    let mut stds = Array1::ones(n_features);

    let is_missing_nan = missing_values.is_nan();

    for j in 0..n_features {
        let column = X.column(j);
        let valid_values: Vec<f64> = column
            .iter()
            .filter(|&&x| {
                if is_missing_nan {
                    !x.is_nan()
                } else {
                    (x - missing_values).abs() >= f64::EPSILON
                }
            })
            .cloned()
            .collect();

        if valid_values.len() > 1 {
            let variance = valid_values
                .iter()
                .map(|&x| (x - means[j]).powi(2))
                .sum::<f64>()
                / (valid_values.len() - 1) as f64;
            stds[j] = variance.sqrt().max(1e-8);
        }
    }

    stds
}

fn compute_bin_edges(
    X: &Array2<f64>,
    n_bins: usize,
    missing_values: f64,
) -> SklResult<Vec<Array1<f64>>> {
    let (_, n_features) = X.dim();
    let mut bin_edges = Vec::new();

    let is_missing_nan = missing_values.is_nan();

    for j in 0..n_features {
        let column = X.column(j);
        let mut valid_values: Vec<f64> = column
            .iter()
            .filter(|&&x| {
                if is_missing_nan {
                    !x.is_nan()
                } else {
                    (x - missing_values).abs() >= f64::EPSILON
                }
            })
            .cloned()
            .collect();

        if valid_values.is_empty() {
            // Default bin edges
            bin_edges.push(Array1::linspace(0.0, 1.0, n_bins + 1));
        } else {
            valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min_val = valid_values[0];
            let max_val = valid_values[valid_values.len() - 1];

            if (max_val - min_val).abs() < f64::EPSILON {
                // All values are the same
                let edges = Array1::from_vec(vec![min_val - 0.5, max_val + 0.5]);
                bin_edges.push(edges);
            } else {
                let edges = Array1::linspace(min_val, max_val, n_bins + 1);
                bin_edges.push(edges);
            }
        }
    }

    Ok(bin_edges)
}

fn compute_mutual_information(
    X: &Array2<f64>,
    feature1: usize,
    feature2: usize,
    bin_edges: &[Array1<f64>],
    missing_values: f64,
) -> SklResult<f64> {
    let is_missing_nan = missing_values.is_nan();

    // Collect valid pairs
    let mut valid_pairs = Vec::new();
    for i in 0..X.nrows() {
        let val1 = X[[i, feature1]];
        let val2 = X[[i, feature2]];

        let is_val1_missing = if is_missing_nan {
            val1.is_nan()
        } else {
            (val1 - missing_values).abs() < f64::EPSILON
        };

        let is_val2_missing = if is_missing_nan {
            val2.is_nan()
        } else {
            (val2 - missing_values).abs() < f64::EPSILON
        };

        if !is_val1_missing && !is_val2_missing {
            valid_pairs.push((val1, val2));
        }
    }

    if valid_pairs.len() < 2 {
        return Ok(0.0);
    }

    // Discretize values
    let edges1 = &bin_edges[feature1];
    let edges2 = &bin_edges[feature2];

    let mut joint_counts = HashMap::new();
    let mut marginal1_counts = HashMap::new();
    let mut marginal2_counts = HashMap::new();

    for (val1, val2) in valid_pairs.iter() {
        let bin1 = discretize_value(*val1, edges1);
        let bin2 = discretize_value(*val2, edges2);

        *joint_counts.entry((bin1, bin2)).or_insert(0) += 1;
        *marginal1_counts.entry(bin1).or_insert(0) += 1;
        *marginal2_counts.entry(bin2).or_insert(0) += 1;
    }

    let n_samples = valid_pairs.len() as f64;

    // Compute mutual information
    let mut mi = 0.0;
    for ((bin1, bin2), &joint_count) in joint_counts.iter() {
        if joint_count > 0 {
            let p_joint = joint_count as f64 / n_samples;
            let p_marg1 = marginal1_counts[bin1] as f64 / n_samples;
            let p_marg2 = marginal2_counts[bin2] as f64 / n_samples;

            if p_joint > 0.0 && p_marg1 > 0.0 && p_marg2 > 0.0 {
                mi += p_joint * (p_joint / (p_marg1 * p_marg2)).ln();
            }
        }
    }

    Ok(mi)
}

fn discretize_value(value: f64, bin_edges: &Array1<f64>) -> usize {
    for i in 0..bin_edges.len() - 1 {
        if value >= bin_edges[i] && value < bin_edges[i + 1] {
            return i;
        }
    }
    // For values at the maximum edge
    if value >= bin_edges[bin_edges.len() - 1] {
        return bin_edges.len() - 2;
    }
    0
}

fn solve_linear_system(A: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
    let n = A.nrows();
    if n != A.ncols() || n != b.len() {
        return Err(SklearsError::InvalidInput(
            "Matrix dimensions mismatch".to_string(),
        ));
    }

    // Simple Gaussian elimination (for small matrices)
    let mut aug_matrix = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug_matrix[[i, j]] = A[[i, j]];
        }
        aug_matrix[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if aug_matrix[[k, i]].abs() > aug_matrix[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug_matrix[[i, j]];
                aug_matrix[[i, j]] = aug_matrix[[max_row, j]];
                aug_matrix[[max_row, j]] = temp;
            }
        }

        // Make diagonal element 1
        let diag = aug_matrix[[i, i]];
        if diag.abs() < 1e-10 {
            // Add small regularization
            aug_matrix[[i, i]] += 1e-6;
        }

        for j in i..=n {
            aug_matrix[[i, j]] /= aug_matrix[[i, i]];
        }

        // Eliminate column
        for k in i + 1..n {
            let factor = aug_matrix[[k, i]];
            for j in i..=n {
                aug_matrix[[k, j]] -= factor * aug_matrix[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug_matrix[[i, n]];
        for j in i + 1..n {
            x[i] -= aug_matrix[[i, j]] * x[j];
        }
    }

    Ok(x)
}

fn compute_histogram(values: &[f64], n_bins: usize) -> Array1<f64> {
    if values.is_empty() {
        return Array1::ones(n_bins) / n_bins as f64;
    }

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut histogram = Array1::zeros(n_bins);

    if (max_val - min_val).abs() < f64::EPSILON {
        // All values are the same
        histogram[n_bins / 2] = 1.0;
        return histogram;
    }

    let bin_width = (max_val - min_val) / n_bins as f64;

    for &value in values {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        histogram[bin_idx] += 1.0;
    }

    // Normalize to probability distribution
    let total = histogram.sum();
    if total > 0.0 {
        histogram /= total;
    } else {
        histogram.fill(1.0 / n_bins as f64);
    }

    histogram
}

fn sample_from_distribution(
    distribution: &Array1<f64>,
    mean: f64,
    std: f64,
    rng: &mut impl Rng,
) -> SklResult<f64> {
    // Sample from the histogram distribution and then map to original scale
    let cumsum: Vec<f64> = distribution
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();

    let random_val = rng.gen();

    let bin_idx = cumsum
        .iter()
        .position(|&x| x >= random_val)
        .unwrap_or(distribution.len() - 1);

    // Convert bin index to value (assuming bins are centered around mean ± 3*std)
    let n_bins = distribution.len();
    let bin_width = 6.0 * std / n_bins as f64;
    let bin_center = mean - 3.0 * std + (bin_idx as f64 + 0.5) * bin_width;

    // Add some noise within the bin
    let noise = rng.gen_range(-bin_width / 2.0..bin_width / 2.0);

    Ok(bin_center + noise)
}

/// Information Gain Imputer
///
/// Uses information gain to select the most informative features for imputing missing values.
/// Information gain measures the reduction in entropy achieved by partitioning the data
/// based on a feature's values, commonly used in decision tree algorithms.
///
/// # Parameters
///
/// * `k_features` - Number of features with highest information gain to use for imputation
/// * `discretization_bins` - Number of bins for discretizing continuous variables
/// * `imputation_method` - Final imputation method ('tree', 'knn', 'regression', 'mode')
/// * `min_samples_split` - Minimum samples required to split a node (for tree method)
/// * `max_depth` - Maximum depth for decision tree imputation
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::InformationGainImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = InformationGainImputer::new()
///     .k_features(2)
///     .discretization_bins(5);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct InformationGainImputer<S = Untrained> {
    state: S,
    k_features: usize,
    discretization_bins: usize,
    imputation_method: String,
    min_samples_split: usize,
    max_depth: usize,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for InformationGainImputer
#[derive(Debug, Clone)]
pub struct InformationGainImputerTrained {
    feature_ig_rankings: HashMap<usize, Vec<(usize, f64)>>, // (feature_idx, info_gain)
    feature_means_: Array1<f64>,
    feature_stds_: Array1<f64>,
    bin_edges_: Vec<Array1<f64>>,
    imputation_trees_: HashMap<usize, DecisionNode>,
    n_features_in_: usize,
}

/// Decision tree node for information gain imputation
#[derive(Debug, Clone)]
pub struct DecisionNode {
    /// feature_idx
    pub feature_idx: Option<usize>,
    /// threshold
    pub threshold: Option<f64>,
    /// left
    pub left: Option<Box<DecisionNode>>,
    /// right
    pub right: Option<Box<DecisionNode>>,
    /// value
    pub value: Option<f64>,
    /// is_leaf
    pub is_leaf: bool,
}

impl InformationGainImputer<Untrained> {
    /// Create a new InformationGainImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            k_features: 5,
            discretization_bins: 10,
            imputation_method: "tree".to_string(),
            min_samples_split: 10,
            max_depth: 5,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the number of features with highest information gain to use
    pub fn k_features(mut self, k_features: usize) -> Self {
        self.k_features = k_features;
        self
    }

    /// Set the number of discretization bins
    pub fn discretization_bins(mut self, bins: usize) -> Self {
        self.discretization_bins = bins;
        self
    }

    /// Set the imputation method
    pub fn imputation_method(mut self, method: String) -> Self {
        self.imputation_method = method;
        self
    }

    /// Set the minimum samples required to split
    pub fn min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples;
        self
    }

    /// Set the maximum tree depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
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

impl Default for InformationGainImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for InformationGainImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for InformationGainImputer<Untrained> {
    type Fitted = InformationGainImputer<InformationGainImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let mut rng = Random::default();

        // Compute feature statistics
        let feature_means = compute_feature_means(&X, self.missing_values);
        let feature_stds = compute_feature_stds(&X, &feature_means, self.missing_values);

        // Create bin edges for discretization
        let bin_edges = compute_bin_edges(&X, self.discretization_bins, self.missing_values)?;

        // Compute information gain rankings for each feature
        let mut feature_ig_rankings = HashMap::new();
        let mut imputation_trees = HashMap::new();

        for target_feature in 0..n_features {
            // Check if this feature has missing values
            let has_missing = (0..n_samples).any(|i| self.is_missing(X[[i, target_feature]]));

            if has_missing {
                let mut ig_scores = Vec::new();

                // Compute information gain for each predictor feature
                for predictor_feature in 0..n_features {
                    if predictor_feature != target_feature {
                        let ig_score = compute_information_gain(
                            &X,
                            target_feature,
                            predictor_feature,
                            &bin_edges,
                            self.missing_values,
                        )?;
                        ig_scores.push((predictor_feature, ig_score));
                    }
                }

                // Sort by information gain (descending)
                ig_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Take top k features
                let top_features: Vec<(usize, f64)> =
                    ig_scores.into_iter().take(self.k_features).collect();

                feature_ig_rankings.insert(target_feature, top_features.clone());

                // Build decision tree for this target feature if using tree method
                if self.imputation_method == "tree" {
                    let predictor_features: Vec<usize> =
                        top_features.iter().map(|(idx, _)| *idx).collect();
                    let tree = build_decision_tree(
                        &X,
                        target_feature,
                        &predictor_features,
                        self.min_samples_split,
                        self.max_depth,
                        self.missing_values,
                        &mut rng,
                    )?;
                    imputation_trees.insert(target_feature, tree);
                }
            }
        }

        Ok(InformationGainImputer {
            state: InformationGainImputerTrained {
                feature_ig_rankings,
                feature_means_: feature_means,
                feature_stds_: feature_stds,
                bin_edges_: bin_edges,
                imputation_trees_: imputation_trees,
                n_features_in_: n_features,
            },
            k_features: self.k_features,
            discretization_bins: self.discretization_bins,
            imputation_method: self.imputation_method,
            min_samples_split: self.min_samples_split,
            max_depth: self.max_depth,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for InformationGainImputer<InformationGainImputerTrained>
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

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    // Get the top informative features for this target feature
                    if let Some(top_features) = self.state.feature_ig_rankings.get(&j) {
                        let imputed_value = match self.imputation_method.as_str() {
                            "tree" => {
                                if let Some(tree) = self.state.imputation_trees_.get(&j) {
                                    predict_with_tree(tree, &X_imputed.row(i).to_owned())?
                                } else {
                                    self.state.feature_means_[j] // Fallback
                                }
                            }
                            "knn" => impute_with_knn(
                                &X_imputed,
                                i,
                                j,
                                top_features,
                                3,
                                self.missing_values,
                            )?,
                            "mode" => impute_with_mode(&X, j, self.missing_values),
                            _ => self.state.feature_means_[j], // Default to mean
                        };

                        X_imputed[[i, j]] = imputed_value;
                    } else {
                        // No information gain data available, use mean
                        X_imputed[[i, j]] = self.state.feature_means_[j];
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl InformationGainImputer<InformationGainImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    /// Get the information gain rankings for each feature
    pub fn get_feature_importances(&self) -> &HashMap<usize, Vec<(usize, f64)>> {
        &self.state.feature_ig_rankings
    }
}

// Information gain helper functions

/// Compute information gain for feature selection
fn compute_information_gain(
    X: &Array2<f64>,
    target_feature: usize,
    predictor_feature: usize,
    bin_edges: &[Array1<f64>],
    missing_values: f64,
) -> SklResult<f64> {
    let (n_samples, _) = X.dim();
    let is_missing_nan = missing_values.is_nan();

    // Collect valid samples (both features observed)
    let mut valid_samples = Vec::new();
    for i in 0..n_samples {
        let target_val = X[[i, target_feature]];
        let pred_val = X[[i, predictor_feature]];

        let target_missing = if is_missing_nan {
            target_val.is_nan()
        } else {
            (target_val - missing_values).abs() < f64::EPSILON
        };
        let pred_missing = if is_missing_nan {
            pred_val.is_nan()
        } else {
            (pred_val - missing_values).abs() < f64::EPSILON
        };

        if !target_missing && !pred_missing {
            valid_samples.push((target_val, pred_val));
        }
    }

    if valid_samples.len() < 10 {
        return Ok(0.0); // Not enough samples for reliable information gain
    }

    // Discretize the target and predictor features
    let target_discretized = discretize_values(
        &valid_samples.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        &bin_edges[target_feature],
    );
    let predictor_discretized = discretize_values(
        &valid_samples.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
        &bin_edges[predictor_feature],
    );

    // Compute entropy of target feature
    let target_entropy = compute_entropy(&target_discretized);

    // Compute conditional entropy
    let mut conditional_entropy = 0.0;
    let predictor_counts = count_values(&predictor_discretized);
    let total_samples = valid_samples.len() as f64;

    for (&pred_value, &pred_count) in &predictor_counts {
        let pred_prob = pred_count as f64 / total_samples;

        // Get target values for this predictor value
        let mut target_subset = Vec::new();
        for (i, &pred_val) in predictor_discretized.iter().enumerate() {
            if pred_val == pred_value {
                target_subset.push(target_discretized[i]);
            }
        }

        if !target_subset.is_empty() {
            let subset_entropy = compute_entropy(&target_subset);
            conditional_entropy += pred_prob * subset_entropy;
        }
    }

    let information_gain = target_entropy - conditional_entropy;
    Ok(information_gain.max(0.0)) // Ensure non-negative
}

/// Discretize continuous values using bin edges
fn discretize_values(values: &[f64], bin_edges: &Array1<f64>) -> Vec<usize> {
    values
        .iter()
        .map(|&value| {
            for (i, &edge) in bin_edges.iter().enumerate().skip(1) {
                if value <= edge {
                    return i - 1;
                }
            }
            bin_edges.len() - 2 // Last bin
        })
        .collect()
}

/// Compute entropy of discrete values
fn compute_entropy(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let counts = count_values(values);
    let total = values.len() as f64;
    let mut entropy = 0.0;

    for &count in counts.values() {
        if count > 0 {
            let prob = count as f64 / total;
            entropy -= prob * prob.log2();
        }
    }

    entropy
}

/// Count occurrences of each value
fn count_values<T: Eq + std::hash::Hash + Copy>(values: &[T]) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for &value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
}

/// Build a decision tree for imputation
fn build_decision_tree(
    X: &Array2<f64>,
    target_feature: usize,
    predictor_features: &[usize],
    min_samples_split: usize,
    max_depth: usize,
    missing_values: f64,
    rng: &mut impl Rng,
) -> SklResult<DecisionNode> {
    // Collect training samples
    let (n_samples, _) = X.dim();
    let is_missing_nan = missing_values.is_nan();
    let mut training_samples = Vec::new();

    for i in 0..n_samples {
        let target_val = X[[i, target_feature]];
        let target_missing = if is_missing_nan {
            target_val.is_nan()
        } else {
            (target_val - missing_values).abs() < f64::EPSILON
        };

        if !target_missing {
            // Check if all predictor features are observed
            let all_observed = predictor_features.iter().all(|&j| {
                let val = X[[i, j]];
                if is_missing_nan {
                    !val.is_nan()
                } else {
                    (val - missing_values).abs() >= f64::EPSILON
                }
            });

            if all_observed {
                let mut features = Vec::new();
                for &j in predictor_features {
                    features.push(X[[i, j]]);
                }
                training_samples.push((features, target_val));
            }
        }
    }

    if training_samples.is_empty() {
        return Ok(DecisionNode {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(0.0),
            is_leaf: true,
        });
    }

    build_tree_recursive(
        &training_samples,
        predictor_features,
        min_samples_split,
        max_depth,
        0,
        rng,
    )
}

/// Recursively build decision tree
fn build_tree_recursive(
    samples: &[(Vec<f64>, f64)],
    feature_indices: &[usize],
    min_samples_split: usize,
    max_depth: usize,
    current_depth: usize,
    _rng: &mut impl Rng,
) -> SklResult<DecisionNode> {
    // Base case: create leaf node
    if samples.len() < min_samples_split || current_depth >= max_depth || samples.is_empty() {
        let mean_value = if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(|(_, target)| *target).sum::<f64>() / samples.len() as f64
        };

        return Ok(DecisionNode {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(mean_value),
            is_leaf: true,
        });
    }

    // Find best split
    let mut best_gain = 0.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for (feat_idx, &_global_feat_idx) in feature_indices.iter().enumerate() {
        // Try multiple thresholds
        let feature_values: Vec<f64> = samples
            .iter()
            .map(|(features, _)| features[feat_idx])
            .collect();
        let mut unique_values = feature_values.clone();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_values.dedup();

        for i in 0..unique_values.len().saturating_sub(1) {
            let threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;
            let gain = compute_split_gain(samples, feat_idx, threshold);

            if gain > best_gain {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = threshold;
            }
        }
    }

    // If no good split found, create leaf
    if best_gain <= 0.0 {
        let mean_value =
            samples.iter().map(|(_, target)| *target).sum::<f64>() / samples.len() as f64;
        return Ok(DecisionNode {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(mean_value),
            is_leaf: true,
        });
    }

    // Split samples
    let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
        .iter()
        .partition(|(features, _)| features[best_feature] <= best_threshold);

    // Convert to owned values
    let left_samples_owned: Vec<(Vec<f64>, f64)> = left_samples.into_iter().cloned().collect();
    let right_samples_owned: Vec<(Vec<f64>, f64)> = right_samples.into_iter().cloned().collect();

    // Recursively build subtrees
    let left_child = build_tree_recursive(
        &left_samples_owned,
        feature_indices,
        min_samples_split,
        max_depth,
        current_depth + 1,
        _rng,
    )?;
    let right_child = build_tree_recursive(
        &right_samples_owned,
        feature_indices,
        min_samples_split,
        max_depth,
        current_depth + 1,
        _rng,
    )?;

    Ok(DecisionNode {
        feature_idx: Some(feature_indices[best_feature]),
        threshold: Some(best_threshold),
        left: Some(Box::new(left_child)),
        right: Some(Box::new(right_child)),
        value: None,
        is_leaf: false,
    })
}

/// Compute gain from a split
fn compute_split_gain(samples: &[(Vec<f64>, f64)], feature_idx: usize, threshold: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
        .iter()
        .partition(|(features, _)| features[feature_idx] <= threshold);

    if left_samples.is_empty() || right_samples.is_empty() {
        return 0.0;
    }

    let total_variance = compute_variance(
        &samples
            .iter()
            .map(|(_, target)| *target)
            .collect::<Vec<_>>(),
    );
    let left_variance = compute_variance(
        &left_samples
            .iter()
            .map(|(_, target)| *target)
            .collect::<Vec<_>>(),
    );
    let right_variance = compute_variance(
        &right_samples
            .iter()
            .map(|(_, target)| *target)
            .collect::<Vec<_>>(),
    );

    let left_weight = left_samples.len() as f64 / samples.len() as f64;
    let right_weight = right_samples.len() as f64 / samples.len() as f64;

    total_variance - (left_weight * left_variance + right_weight * right_variance)
}

/// Compute variance of values
fn compute_variance(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance
}

/// Predict using decision tree
fn predict_with_tree(tree: &DecisionNode, features: &Array1<f64>) -> SklResult<f64> {
    if tree.is_leaf {
        return Ok(tree.value.unwrap_or(0.0));
    }

    if let (Some(feature_idx), Some(threshold)) = (tree.feature_idx, tree.threshold) {
        if feature_idx < features.len() {
            if features[feature_idx] <= threshold {
                if let Some(ref left) = tree.left {
                    return predict_with_tree(left, features);
                }
            } else if let Some(ref right) = tree.right {
                return predict_with_tree(right, features);
            }
        }
    }

    Ok(0.0) // Fallback
}

/// KNN imputation using top informative features
fn impute_with_knn(
    X: &Array2<f64>,
    sample_idx: usize,
    target_feature: usize,
    top_features: &[(usize, f64)],
    k: usize,
    missing_values: f64,
) -> SklResult<f64> {
    let (n_samples, _) = X.dim();
    let is_missing_nan = missing_values.is_nan();
    let predictor_features: Vec<usize> = top_features.iter().map(|(idx, _)| *idx).collect();

    let mut distances = Vec::new();

    for i in 0..n_samples {
        if i == sample_idx {
            continue;
        }

        // Check if target is observed and predictors are observed
        let target_val = X[[i, target_feature]];
        let target_missing = if is_missing_nan {
            target_val.is_nan()
        } else {
            (target_val - missing_values).abs() < f64::EPSILON
        };

        if !target_missing {
            let all_predictors_observed = predictor_features.iter().all(|&j| {
                let val = X[[i, j]];
                if is_missing_nan {
                    !val.is_nan()
                } else {
                    (val - missing_values).abs() >= f64::EPSILON
                }
            });

            if all_predictors_observed {
                // Compute distance using only the top informative features
                let mut distance = 0.0;
                let mut valid_features = 0;

                for &feature_idx in &predictor_features {
                    let val1 = X[[sample_idx, feature_idx]];
                    let val2 = X[[i, feature_idx]];

                    let val1_missing = if is_missing_nan {
                        val1.is_nan()
                    } else {
                        (val1 - missing_values).abs() < f64::EPSILON
                    };

                    if !val1_missing {
                        distance += (val1 - val2).powi(2);
                        valid_features += 1;
                    }
                }

                if valid_features > 0 {
                    distance = (distance / valid_features as f64).sqrt();
                    distances.push((distance, target_val));
                }
            }
        }
    }

    if distances.is_empty() {
        return Ok(0.0);
    }

    // Sort by distance and take k nearest
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let k_neighbors = distances.into_iter().take(k).collect::<Vec<_>>();

    // Return weighted average
    if k_neighbors.is_empty() {
        Ok(0.0)
    } else {
        let sum: f64 = k_neighbors.iter().map(|(_, val)| *val).sum();
        Ok(sum / k_neighbors.len() as f64)
    }
}

/// Mode imputation
fn impute_with_mode(X: &Array2<f64>, target_feature: usize, missing_values: f64) -> f64 {
    let (n_samples, _) = X.dim();
    let is_missing_nan = missing_values.is_nan();
    let mut value_counts = HashMap::new();

    for i in 0..n_samples {
        let val = X[[i, target_feature]];
        let is_missing = if is_missing_nan {
            val.is_nan()
        } else {
            (val - missing_values).abs() < f64::EPSILON
        };

        if !is_missing {
            // Discretize for mode calculation
            let discrete_val = (val * 10.0).round() / 10.0; // Round to 1 decimal place
            *value_counts.entry(discrete_val as i64).or_insert(0) += 1;
        }
    }

    if value_counts.is_empty() {
        return 0.0;
    }

    // Find mode
    let mode = value_counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(&val, _)| val as f64)
        .unwrap_or(0.0);

    mode
}
