//! Hot-Deck Imputation for Categorical Data
//!
//! Hot-deck imputation replaces missing values with observed values from similar
//! records (donors). This method is particularly suitable for categorical data
//! where statistical interpolation doesn't make sense.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Hot-Deck Imputation for Categorical Data
///
/// Hot-deck imputation replaces missing values with observed values from similar
/// records (donors). This method is particularly suitable for categorical data
/// where statistical interpolation doesn't make sense.
///
/// # Parameters
///
/// * `n_donors` - Number of donor records to consider
/// * `distance_metric` - Distance metric for finding similar records
/// * `categorical_features` - Indices of categorical features
/// * `missing_values` - The placeholder for missing values
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::HotDeckImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 2.0, 1.0], [2.0, f64::NAN, 3.0]];
///
/// let imputer = HotDeckImputer::new()
///     .n_donors(3)
///     .categorical_features(vec![0, 2]);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HotDeckImputer<S = Untrained> {
    state: S,
    n_donors: usize,
    distance_metric: String,
    categorical_features: Vec<usize>,
    missing_values: f64,
    random_state: Option<u64>,
}

/// Trained state for HotDeckImputer
#[derive(Debug, Clone)]
pub struct HotDeckImputerTrained {
    donor_pool_: Array2<f64>,
    category_mappings_: HashMap<usize, HashMap<String, usize>>,
    reverse_mappings_: HashMap<usize, Vec<String>>,
    n_features_in_: usize,
}

impl HotDeckImputer<Untrained> {
    /// Create a new HotDeckImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_donors: 5,
            distance_metric: "hamming".to_string(),
            categorical_features: Vec::new(),
            missing_values: f64::NAN,
            random_state: None,
        }
    }

    /// Set the number of donor records
    pub fn n_donors(mut self, n_donors: usize) -> Self {
        self.n_donors = n_donors;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, distance_metric: String) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    /// Set the categorical features indices
    pub fn categorical_features(mut self, categorical_features: Vec<usize>) -> Self {
        self.categorical_features = categorical_features;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
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

impl Default for HotDeckImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HotDeckImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for HotDeckImputer<Untrained> {
    type Fitted = HotDeckImputer<HotDeckImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        // If no categorical features specified, assume all are categorical
        let categorical_features = if self.categorical_features.is_empty() {
            (0..n_features).collect()
        } else {
            self.categorical_features.clone()
        };

        // Extract complete records for donor pool
        let mut complete_rows = Vec::new();
        for i in 0..n_samples {
            let mut is_complete = true;
            for j in 0..n_features {
                if self.is_missing(X[[i, j]]) {
                    is_complete = false;
                    break;
                }
            }
            if is_complete {
                complete_rows.push(i);
            }
        }

        if complete_rows.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No complete records found for donor pool".to_string(),
            ));
        }

        // Create donor pool
        let mut donor_pool = Array2::zeros((complete_rows.len(), n_features));
        for (new_i, &orig_i) in complete_rows.iter().enumerate() {
            for j in 0..n_features {
                donor_pool[[new_i, j]] = X[[orig_i, j]];
            }
        }

        // Create category mappings for categorical features
        let mut category_mappings = HashMap::new();
        let mut reverse_mappings = HashMap::new();

        for &feature_idx in &categorical_features {
            let mut categories = HashSet::new();
            for i in 0..n_samples {
                if !self.is_missing(X[[i, feature_idx]]) {
                    categories.insert(format!("{}", X[[i, feature_idx]] as i32));
                }
            }

            let mut categories_vec: Vec<String> = categories.into_iter().collect();
            categories_vec.sort();

            let mapping: HashMap<String, usize> = categories_vec
                .iter()
                .enumerate()
                .map(|(i, cat)| (cat.clone(), i))
                .collect();

            category_mappings.insert(feature_idx, mapping);
            reverse_mappings.insert(feature_idx, categories_vec);
        }

        Ok(HotDeckImputer {
            state: HotDeckImputerTrained {
                donor_pool_: donor_pool,
                category_mappings_: category_mappings,
                reverse_mappings_: reverse_mappings,
                n_features_in_: n_features,
            },
            n_donors: self.n_donors,
            distance_metric: self.distance_metric,
            categorical_features,
            missing_values: self.missing_values,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for HotDeckImputer<HotDeckImputerTrained> {
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
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    let donors = self.find_donors(&X_imputed, i, j)?;
                    if !donors.is_empty() {
                        // Randomly select one of the top donors
                        let selected_idx = rng.gen_range(0..donors.len());
                        let selected_donor = &donors[selected_idx];
                        X_imputed[[i, j]] = self.state.donor_pool_[[*selected_donor, j]];
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl HotDeckImputer<HotDeckImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn find_donors(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        _target_feature: usize,
    ) -> SklResult<Vec<usize>> {
        let n_donors = self.state.donor_pool_.nrows();
        let mut distances: Vec<(f64, usize)> = Vec::new();

        for donor_idx in 0..n_donors {
            let distance =
                self.compute_distance(X.row(sample_idx), self.state.donor_pool_.row(donor_idx))?;
            distances.push((distance, donor_idx));
        }

        // Sort by distance and take the closest donors
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let selected_donors: Vec<usize> = distances
            .iter()
            .take(self.n_donors)
            .map(|(_, idx)| *idx)
            .collect();

        Ok(selected_donors)
    }

    fn compute_distance(
        &self,
        record1: ArrayView1<f64>,
        record2: ArrayView1<f64>,
    ) -> SklResult<f64> {
        match self.distance_metric.as_str() {
            "hamming" => Ok(self.hamming_distance(record1, record2)),
            "euclidean" => Ok(self.euclidean_distance(record1, record2)),
            _ => Ok(self.hamming_distance(record1, record2)), // Default to hamming
        }
    }

    fn hamming_distance(&self, record1: ArrayView1<f64>, record2: ArrayView1<f64>) -> f64 {
        let mut distance = 0.0;
        let mut valid_comparisons = 0;

        for i in 0..record1.len() {
            if !self.is_missing(record1[i]) && !self.is_missing(record2[i]) {
                valid_comparisons += 1;
                if self.categorical_features.contains(&i) {
                    // For categorical features, check exact match
                    if (record1[i] as i32) != (record2[i] as i32) {
                        distance += 1.0;
                    }
                } else {
                    // For numerical features, use threshold
                    if (record1[i] - record2[i]).abs() > 0.1 {
                        distance += 1.0;
                    }
                }
            }
        }

        if valid_comparisons > 0 {
            distance / valid_comparisons as f64
        } else {
            f64::INFINITY
        }
    }

    fn euclidean_distance(&self, record1: ArrayView1<f64>, record2: ArrayView1<f64>) -> f64 {
        let mut distance = 0.0;
        let mut valid_comparisons = 0;

        for i in 0..record1.len() {
            if !self.is_missing(record1[i]) && !self.is_missing(record2[i]) {
                valid_comparisons += 1;
                if self.categorical_features.contains(&i) {
                    // For categorical features, use 0/1 distance
                    if (record1[i] as i32) != (record2[i] as i32) {
                        distance += 1.0;
                    }
                } else {
                    // For numerical features, use squared difference
                    distance += (record1[i] - record2[i]).powi(2);
                }
            }
        }

        if valid_comparisons > 0 {
            (distance / valid_comparisons as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }
}
