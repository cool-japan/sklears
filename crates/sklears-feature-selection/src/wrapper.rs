//! Wrapper-based feature selection methods

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Score, Trained, Transform, Untrained},
    types::Float,
};
// use sklears_model_selection::{CrossValidator, KFold}; // Removed dependency

/// Simple cross-validator trait replacement
pub trait CrossValidator: Send + Sync {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
}

/// Simple KFold implementation replacement
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl CrossValidator for KFold {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            use scirs2_core::rand_prelude::SliceRandom;
            use scirs2_core::random::{rngs::StdRng, SeedableRng};
            let mut rng = if let Some(seed) = self.random_state {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::seed_from_u64(42)
            };
            indices.shuffle(&mut rng);
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / self.n_splits;

        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices[0..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}
use std::marker::PhantomData;

/// Trait for target arrays that can be indexed
pub trait IndexableTarget: Clone {
    /// Get elements at the specified indices
    fn select(&self, indices: &[usize]) -> Self;
}

// Implement for Array1
impl<T: Clone> IndexableTarget for Array1<T> {
    fn select(&self, indices: &[usize]) -> Self {
        self.select(Axis(0), indices)
    }
}

/// Trait for estimators that can provide feature importance scores
pub trait FeatureImportance {
    /// Get feature importance scores
    /// Higher scores indicate more important features
    fn feature_importances(&self) -> SklResult<Array1<Float>>;
}

/// Trait for estimators that have linear coefficients
pub trait HasCoefficients {
    /// Get the coefficients
    fn coef(&self) -> SklResult<Array1<Float>>;
}

// Implement FeatureImportance for estimators with coefficients
impl<T> FeatureImportance for T
where
    T: HasCoefficients,
{
    fn feature_importances(&self) -> SklResult<Array1<Float>> {
        // Use absolute values of coefficients as importance scores
        self.coef().map(|coef| coef.mapv(|x| x.abs()))
    }
}

/// Recursive Feature Elimination
pub struct RFE<E, State = Untrained> {
    estimator: E,
    n_features_to_select: Option<usize>,
    step: f64,
    state: PhantomData<State>,
    // Trained state
    support_: Option<Array1<bool>>,
    ranking_: Option<Array1<usize>>,
    n_features_: Option<usize>,
}

impl<E: Clone> RFE<E, Untrained> {
    /// Create a new RFE selector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            n_features_to_select: None,
            step: 1.0,
            state: PhantomData,
            support_: None,
            ranking_: None,
            n_features_: None,
        }
    }

    /// Set the number of features to select
    pub fn n_features_to_select(mut self, n: usize) -> Self {
        self.n_features_to_select = Some(n);
        self
    }

    /// Set the step size for feature elimination
    /// If step >= 1, remove step features at each iteration
    /// If 0 < step < 1, remove step * n_features features at each iteration
    pub fn step(mut self, step: f64) -> Self {
        if step <= 0.0 {
            panic!("step must be > 0");
        }
        self.step = step;
        self
    }
}

impl<E> Estimator for RFE<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for RFE<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: FeatureImportance + Send + Sync,
    Y: Clone + Send + Sync,
{
    type Fitted = RFE<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();
        let n_features_to_select = self.n_features_to_select.unwrap_or(n_features / 2);

        if n_features_to_select > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "n_features_to_select ({}) must be <= n_features ({})",
                n_features_to_select, n_features
            )));
        }

        // Initialize support mask and ranking
        let mut support = Array1::from_elem(n_features, true);
        let mut ranking = Array1::ones(n_features);

        let mut current_n_features = n_features;
        let mut current_step = 1;

        while current_n_features > n_features_to_select {
            // Select features that are currently supported
            let features_mask: Vec<usize> = (0..n_features).filter(|&i| support[i]).collect();

            // Create subset of X with selected features
            let x_subset = x.select(Axis(1), &features_mask);

            // Fit estimator on subset
            let fitted_estimator = self.estimator.clone().fit(&x_subset, y)?;

            // Get feature importances
            let importances = fitted_estimator.feature_importances()?;

            // Determine how many features to eliminate
            let n_features_to_remove = if self.step >= 1.0 {
                self.step as usize
            } else {
                ((current_n_features as f64) * self.step).max(1.0) as usize
            };
            let n_features_to_remove =
                n_features_to_remove.min(current_n_features - n_features_to_select);

            // Find indices of least important features
            let mut importance_indices: Vec<(usize, Float)> = importances
                .iter()
                .enumerate()
                .map(|(i, &imp)| (i, imp))
                .collect();
            importance_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Eliminate least important features
            for i in 0..n_features_to_remove {
                let feature_idx = features_mask[importance_indices[i].0];
                support[feature_idx] = false;
                ranking[feature_idx] = current_step;
            }

            current_n_features -= n_features_to_remove;
            current_step += 1;
        }

        // Normalize ranking: selected features (still with rank 1) should stay rank 1
        // Features eliminated later should have better (lower) ranks than those eliminated earlier
        // The current ranking has: selected features = 1, first eliminated = 1, later eliminated = 2, etc.
        // We need to ensure selected features keep rank 1, and adjust others accordingly
        let max_rank = *ranking.iter().max().unwrap();
        for i in 0..n_features {
            if !support[i] {
                // For eliminated features, invert the ranking so later eliminated (higher step) get better rank
                ranking[i] = max_rank + 2 - ranking[i];
            }
            // Selected features already have rank 1, so we don't change them
        }

        Ok(RFE {
            estimator: self.estimator,
            n_features_to_select: self.n_features_to_select,
            step: self.step,
            state: PhantomData,
            support_: Some(support),
            ranking_: Some(ranking),
            n_features_: Some(n_features),
        })
    }
}

impl<E> Transform<Array2<Float>> for RFE<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let support = self.support_.as_ref().unwrap();
        let selected_features: Vec<usize> = (0..support.len()).filter(|&i| support[i]).collect();

        Ok(x.select(Axis(1), &selected_features))
    }
}

impl<E> RFE<E, Trained> {
    /// Get the support mask
    pub fn support(&self) -> &Array1<bool> {
        self.support_.as_ref().unwrap()
    }

    /// Get the feature ranking
    pub fn ranking(&self) -> &Array1<usize> {
        self.ranking_.as_ref().unwrap()
    }
}

/// Recursive Feature Elimination with Cross-Validation
pub struct RFECV<E, State = Untrained> {
    estimator: E,
    step: f64,
    min_features_to_select: usize,
    cv: Option<Box<dyn CrossValidator>>,
    n_jobs: Option<i32>,
    state: PhantomData<State>,
    // Trained state
    support_: Option<Array1<bool>>,
    ranking_: Option<Array1<usize>>,
    n_features_: Option<usize>,
    cv_results_: Option<RFECVResults>,
}

#[derive(Debug, Clone)]
pub struct RFECVResults {
    pub mean_test_scores: Array1<Float>,
    pub std_test_scores: Array1<Float>,
    pub n_features: Array1<usize>,
}

impl<E: Clone> RFECV<E, Untrained> {
    /// Create a new RFECV selector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            step: 1.0,
            min_features_to_select: 1,
            cv: None,
            n_jobs: None,
            state: PhantomData,
            support_: None,
            ranking_: None,
            n_features_: None,
            cv_results_: None,
        }
    }

    /// Set the minimum number of features to consider
    pub fn min_features_to_select(mut self, min_features: usize) -> Self {
        if min_features < 1 {
            panic!("min_features_to_select must be >= 1");
        }
        self.min_features_to_select = min_features;
        self
    }

    /// Set the step size for feature elimination
    pub fn step(mut self, step: f64) -> Self {
        if step <= 0.0 {
            panic!("step must be > 0");
        }
        self.step = step;
        self
    }

    /// Set the cross-validation strategy
    pub fn cv(mut self, cv: Box<dyn CrossValidator>) -> Self {
        self.cv = Some(cv);
        self
    }
}

impl<E> Estimator for RFECV<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for RFECV<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: FeatureImportance + Score<Array2<Float>, Y, Float = f64> + Send + Sync,
    Y: IndexableTarget + Send + Sync,
{
    type Fitted = RFECV<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();

        if self.min_features_to_select > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "min_features_to_select ({}) must be <= n_features ({})",
                self.min_features_to_select, n_features
            )));
        }

        // Use default CV if not provided
        let default_cv = KFold::new(5);
        let cv: &dyn CrossValidator = self
            .cv
            .as_ref()
            .map(|cv| cv.as_ref())
            .unwrap_or(&default_cv);

        // Initialize tracking variables
        let mut mean_scores = Vec::new();
        let mut std_scores = Vec::new();
        let mut n_features_list = Vec::new();

        // Try different numbers of features
        let mut current_n_features = n_features;
        let mut support = Array1::from_elem(n_features, true);
        let mut ranking = Array1::ones(n_features);
        let mut current_step = 1;

        while current_n_features >= self.min_features_to_select {
            // Select features that are currently supported
            let features_mask: Vec<usize> = (0..n_features).filter(|&i| support[i]).collect();

            // Create subset of X with selected features
            let x_subset = x.select(Axis(1), &features_mask);

            // Perform cross-validation
            let splits = cv.split(x.nrows());
            let mut fold_scores = Vec::new();

            for (train_idx, test_idx) in splits {
                let x_train = x_subset.select(Axis(0), &train_idx);
                let x_test = x_subset.select(Axis(0), &test_idx);
                let y_train = y.select(&train_idx);
                let y_test = y.select(&test_idx);

                let fitted = self.estimator.clone().fit(&x_train, &y_train)?;
                let score = fitted.score(&x_test, &y_test)?;
                fold_scores.push(score);
            }

            // Keep only finite scores to avoid propagating NaNs from degenerate folds
            fold_scores.retain(|score| score.is_finite());

            if fold_scores.is_empty() {
                mean_scores.push(f64::NEG_INFINITY);
                std_scores.push(0.0);
            } else {
                // Calculate mean and std of scores
                let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                let variance = fold_scores
                    .iter()
                    .map(|&score| (score - mean_score).powi(2))
                    .sum::<f64>()
                    / fold_scores.len() as f64;
                let std_score = variance.sqrt();

                mean_scores.push(mean_score);
                std_scores.push(std_score);
            }
            n_features_list.push(current_n_features);

            // If we've reached min features, stop
            if current_n_features == self.min_features_to_select {
                break;
            }

            // Fit estimator on full subset to get importances for elimination
            let fitted_estimator = self.estimator.clone().fit(&x_subset, y)?;
            let importances = fitted_estimator.feature_importances()?;

            // Determine how many features to eliminate
            let n_features_to_remove = if self.step >= 1.0 {
                self.step as usize
            } else {
                ((current_n_features as f64) * self.step).max(1.0) as usize
            };
            let n_features_to_remove =
                n_features_to_remove.min(current_n_features - self.min_features_to_select);

            if n_features_to_remove == 0 {
                break;
            }

            // Find indices of least important features
            let mut importance_indices: Vec<(usize, Float)> = importances
                .iter()
                .enumerate()
                .map(|(i, &imp)| (i, imp))
                .collect();
            importance_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Eliminate least important features
            for i in 0..n_features_to_remove {
                let feature_idx = features_mask[importance_indices[i].0];
                support[feature_idx] = false;
                ranking[feature_idx] = current_step;
            }

            current_n_features -= n_features_to_remove;
            current_step += 1;
        }

        // Find optimal number of features
        use std::cmp::Ordering;

        let best_idx = mean_scores
            .iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .max_by(
                |(idx_a, score_a), (idx_b, score_b)| match score_a.total_cmp(score_b) {
                    Ordering::Equal => n_features_list[*idx_a].cmp(&n_features_list[*idx_b]),
                    other => other,
                },
            )
            .or_else(|| {
                mean_scores
                    .iter()
                    .enumerate()
                    .max_by(|(idx_a, score_a), (idx_b, score_b)| {
                        match (score_a.is_nan(), score_b.is_nan()) {
                            (true, true) => n_features_list[*idx_a].cmp(&n_features_list[*idx_b]),
                            (true, false) => Ordering::Less,
                            (false, true) => Ordering::Greater,
                            (false, false) => match score_a.total_cmp(score_b) {
                                Ordering::Equal => {
                                    n_features_list[*idx_a].cmp(&n_features_list[*idx_b])
                                }
                                other => other,
                            },
                        }
                    })
            })
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                SklearsError::FitError(
                    "RFECV could not compute any valid cross-validation scores".to_string(),
                )
            })?;
        let optimal_n_features = n_features_list[best_idx];

        // Update support to reflect optimal number of features
        support = Array1::from_elem(n_features, false);
        let sorted_ranking: Vec<(usize, usize)> =
            ranking.iter().enumerate().map(|(i, &r)| (i, r)).collect();
        let mut sorted_by_rank = sorted_ranking;
        sorted_by_rank.sort_by_key(|&(_, r)| r);

        for i in 0..optimal_n_features {
            support[sorted_by_rank[i].0] = true;
        }

        // Normalize ranking: selected features (still with rank 1) should stay rank 1
        // Features eliminated later should have better (lower) ranks than those eliminated earlier
        // The current ranking has: selected features = 1, first eliminated = 1, later eliminated = 2, etc.
        // We need to ensure selected features keep rank 1, and adjust others accordingly
        let max_rank = *ranking.iter().max().unwrap();
        for i in 0..n_features {
            if !support[i] {
                // For eliminated features, invert the ranking so later eliminated (higher step) get better rank
                ranking[i] = max_rank + 2 - ranking[i];
            }
            // Selected features already have rank 1, so we don't change them
        }

        let cv_results = RFECVResults {
            mean_test_scores: Array1::from_vec(mean_scores),
            std_test_scores: Array1::from_vec(std_scores),
            n_features: Array1::from_vec(n_features_list),
        };

        Ok(RFECV {
            estimator: self.estimator,
            step: self.step,
            min_features_to_select: self.min_features_to_select,
            cv: self.cv,
            n_jobs: self.n_jobs,
            state: PhantomData,
            support_: Some(support),
            ranking_: Some(ranking),
            n_features_: Some(n_features),
            cv_results_: Some(cv_results),
        })
    }
}

impl<E> Transform<Array2<Float>> for RFECV<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let support = self.support_.as_ref().unwrap();
        let selected_features: Vec<usize> = (0..support.len()).filter(|&i| support[i]).collect();

        Ok(x.select(Axis(1), &selected_features))
    }
}

impl<E> RFECV<E, Trained> {
    /// Get the support mask
    pub fn support(&self) -> &Array1<bool> {
        self.support_.as_ref().unwrap()
    }

    /// Get the feature ranking
    pub fn ranking(&self) -> &Array1<usize> {
        self.ranking_.as_ref().unwrap()
    }

    /// Get cross-validation results
    pub fn cv_results(&self) -> &RFECVResults {
        self.cv_results_.as_ref().unwrap()
    }
}

/// Select features from any estimator with feature importances
#[derive(Debug, Clone)]
pub struct SelectFromModel<E, State = Untrained> {
    estimator: E,
    threshold: Option<f64>,
    max_features: Option<usize>,
    state: PhantomData<State>,
    // Trained state
    estimator_: Option<E>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    threshold_: Option<f64>,
}

impl<E: Clone> SelectFromModel<E, Untrained> {
    /// Create a new SelectFromModel selector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            threshold: None,
            max_features: None,
            state: PhantomData,
            estimator_: None,
            selected_features_: None,
            n_features_: None,
            threshold_: None,
        }
    }

    /// Set the threshold for feature selection
    /// If None, will use median of feature importances
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set the maximum number of features to select
    pub fn max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }
}

impl<E> Estimator for SelectFromModel<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for SelectFromModel<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: FeatureImportance + Send + Sync,
    Y: Clone + Send + Sync,
{
    type Fitted = SelectFromModel<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();

        // Fit the estimator
        let fitted_estimator = self.estimator.clone().fit(x, y)?;

        // Get feature importances
        let importances = fitted_estimator.feature_importances()?;

        // Determine threshold
        let threshold = if let Some(thresh) = self.threshold {
            thresh
        } else {
            // Use median as default threshold
            let mut sorted_importances = importances.to_vec();
            sorted_importances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_importances[sorted_importances.len() / 2]
        };

        // Select features above threshold
        let mut selected_features: Vec<usize> = (0..n_features)
            .filter(|&i| importances[i] > threshold)
            .collect();

        // Apply max_features limit if specified
        if let Some(max_feat) = self.max_features {
            if selected_features.len() > max_feat {
                // Sort by importance and take top max_feat
                selected_features
                    .sort_by(|&a, &b| importances[b].partial_cmp(&importances[a]).unwrap());
                selected_features.truncate(max_feat);
                selected_features.sort(); // Restore original order
            }
        }

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(format!(
                "No features selected with threshold={}",
                threshold
            )));
        }

        Ok(SelectFromModel {
            estimator: self.estimator.clone(),
            threshold: self.threshold,
            max_features: self.max_features,
            state: PhantomData,
            estimator_: Some(self.estimator),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            threshold_: Some(threshold),
        })
    }
}

impl<E> Transform<Array2<Float>> for SelectFromModel<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let k = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, k));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl<E> SelectFromModel<E, Trained> {
    /// Get the support mask
    pub fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    /// Get the threshold used for selection
    pub fn threshold(&self) -> f64 {
        self.threshold_.unwrap()
    }
}

/// Sequential Feature Selection (forward or backward)
pub struct SequentialFeatureSelector<E, State = Untrained> {
    estimator: E,
    n_features_to_select: Option<usize>,
    direction: String,
    cv_folds: usize,
    state: PhantomData<State>,
    // Trained state
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl<E: Clone> SequentialFeatureSelector<E, Untrained> {
    /// Create a new SequentialFeatureSelector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            n_features_to_select: None,
            direction: "forward".to_string(),
            cv_folds: 5,
            state: PhantomData,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the number of features to select
    pub fn n_features_to_select(mut self, n: usize) -> Self {
        self.n_features_to_select = Some(n);
        self
    }

    /// Set the direction ("forward", "backward", or "bidirectional")
    pub fn direction(mut self, direction: &str) -> Self {
        if direction != "forward" && direction != "backward" && direction != "bidirectional" {
            panic!("direction must be 'forward', 'backward', or 'bidirectional'");
        }
        self.direction = direction.to_string();
        self
    }

    /// Set the number of CV folds
    pub fn cv(mut self, cv_folds: usize) -> Self {
        self.cv_folds = cv_folds;
        self
    }
}

impl<E> Estimator for SequentialFeatureSelector<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for SequentialFeatureSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: Score<Array2<Float>, Y, Float = f64> + Send + Sync,
    Y: IndexableTarget + Send + Sync,
{
    type Fitted = SequentialFeatureSelector<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();
        let n_features_to_select = self.n_features_to_select.unwrap_or(n_features / 2);

        if n_features_to_select > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "n_features_to_select ({}) must be <= n_features ({})",
                n_features_to_select, n_features
            )));
        }

        // Use KFold cross-validation
        let cv = KFold::new(self.cv_folds);

        let selected_features = match self.direction.as_str() {
            "forward" => self.forward_selection(x, y, n_features_to_select, &cv)?,
            "backward" => self.backward_elimination(x, y, n_features_to_select, &cv)?,
            "bidirectional" => self.bidirectional_selection(x, y, n_features_to_select, &cv)?,
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Invalid direction specified".to_string(),
                ))
            }
        };

        Ok(SequentialFeatureSelector {
            estimator: self.estimator,
            n_features_to_select: self.n_features_to_select,
            direction: self.direction,
            cv_folds: self.cv_folds,
            state: PhantomData,
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl<E> SequentialFeatureSelector<E, Untrained> {
    fn forward_selection<Y>(
        &self,
        x: &Array2<Float>,
        y: &Y,
        n_features_to_select: usize,
        cv: &KFold,
    ) -> SklResult<Vec<usize>>
    where
        E: Clone + Fit<Array2<Float>, Y>,
        E::Fitted: Score<Array2<Float>, Y, Float = f64>,
        Y: IndexableTarget,
    {
        let n_features = x.ncols();
        let mut selected = Vec::new();
        let mut remaining: Vec<usize> = (0..n_features).collect();

        while selected.len() < n_features_to_select {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_feature = None;

            for &candidate in &remaining {
                let mut current_features = selected.clone();
                current_features.push(candidate);
                current_features.sort();

                let score = self.evaluate_features(x, y, &current_features, cv)?;
                if score > best_score {
                    best_score = score;
                    best_feature = Some(candidate);
                }
            }

            if let Some(feature) = best_feature {
                selected.push(feature);
                remaining.retain(|&f| f != feature);
            } else {
                break;
            }
        }

        selected.sort();
        Ok(selected)
    }

    fn backward_elimination<Y>(
        &self,
        x: &Array2<Float>,
        y: &Y,
        n_features_to_select: usize,
        cv: &KFold,
    ) -> SklResult<Vec<usize>>
    where
        E: Clone + Fit<Array2<Float>, Y>,
        E::Fitted: Score<Array2<Float>, Y, Float = f64>,
        Y: IndexableTarget,
    {
        let n_features = x.ncols();
        let mut selected: Vec<usize> = (0..n_features).collect();

        while selected.len() > n_features_to_select {
            let mut best_score = f64::NEG_INFINITY;
            let mut worst_feature = None;

            for &candidate in &selected {
                let mut current_features = selected.clone();
                current_features.retain(|&f| f != candidate);

                let score = self.evaluate_features(x, y, &current_features, cv)?;
                if score > best_score {
                    best_score = score;
                    worst_feature = Some(candidate);
                }
            }

            if let Some(feature) = worst_feature {
                selected.retain(|&f| f != feature);
            } else {
                break;
            }
        }

        selected.sort();
        Ok(selected)
    }

    fn bidirectional_selection<Y>(
        &self,
        x: &Array2<Float>,
        y: &Y,
        n_features_to_select: usize,
        cv: &KFold,
    ) -> SklResult<Vec<usize>>
    where
        E: Clone + Fit<Array2<Float>, Y>,
        E::Fitted: Score<Array2<Float>, Y, Float = f64>,
        Y: IndexableTarget,
    {
        #[derive(Debug, Clone)]
        enum Action {
            Add(usize),
            Remove(usize),
            Swap(usize, usize), // (add, remove)
        }

        let n_features = x.ncols();
        let mut selected = Vec::new();
        let mut remaining: Vec<usize> = (0..n_features).collect();

        // Start with no features and add/remove until we reach the target
        while selected.len() != n_features_to_select {
            let mut best_action = None;
            let mut best_score = f64::NEG_INFINITY;

            // If we haven't reached the target, consider adding features
            if selected.len() < n_features_to_select {
                for &candidate in &remaining {
                    let mut test_features = selected.clone();
                    test_features.push(candidate);
                    test_features.sort();

                    let score = self.evaluate_features(x, y, &test_features, cv)?;
                    if score > best_score {
                        best_score = score;
                        best_action = Some(Action::Add(candidate));
                    }
                }
            }

            // If we have too many features, consider removing features
            if selected.len() > n_features_to_select && !selected.is_empty() {
                for &candidate in &selected {
                    let mut test_features = selected.clone();
                    test_features.retain(|&f| f != candidate);

                    let score = self.evaluate_features(x, y, &test_features, cv)?;
                    if score > best_score {
                        best_score = score;
                        best_action = Some(Action::Remove(candidate));
                    }
                }
            }

            // If we're at exactly the right number, consider swapping features
            if selected.len() == n_features_to_select && !remaining.is_empty() {
                for &add_candidate in &remaining {
                    for &remove_candidate in &selected {
                        let mut test_features = selected.clone();
                        test_features.retain(|&f| f != remove_candidate);
                        test_features.push(add_candidate);
                        test_features.sort();

                        let score = self.evaluate_features(x, y, &test_features, cv)?;
                        if score > best_score {
                            best_score = score;
                            best_action = Some(Action::Swap(add_candidate, remove_candidate));
                        }
                    }
                }
            }

            // Apply the best action
            match best_action {
                Some(Action::Add(feature)) => {
                    selected.push(feature);
                    remaining.retain(|&f| f != feature);
                }
                Some(Action::Remove(feature)) => {
                    selected.retain(|&f| f != feature);
                    remaining.push(feature);
                    remaining.sort();
                }
                Some(Action::Swap(add_feature, remove_feature)) => {
                    selected.retain(|&f| f != remove_feature);
                    selected.push(add_feature);
                    remaining.retain(|&f| f != add_feature);
                    remaining.push(remove_feature);
                    remaining.sort();
                }
                None => break, // No improvement found, stop
            }
        }

        selected.sort();
        Ok(selected)
    }

    fn evaluate_features<Y>(
        &self,
        x: &Array2<Float>,
        y: &Y,
        features: &[usize],
        cv: &KFold,
    ) -> SklResult<f64>
    where
        E: Clone + Fit<Array2<Float>, Y>,
        E::Fitted: Score<Array2<Float>, Y, Float = f64>,
        Y: IndexableTarget,
    {
        if features.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }

        let x_subset = x.select(Axis(1), features);
        let splits = cv.split(x.nrows());
        let mut scores = Vec::new();

        for (train_idx, test_idx) in splits {
            let x_train = x_subset.select(Axis(0), &train_idx);
            let x_test = x_subset.select(Axis(0), &test_idx);
            let y_train = y.select(&train_idx);
            let y_test = y.select(&test_idx);

            let fitted = self.estimator.clone().fit(&x_train, &y_train)?;
            let score = fitted.score(&x_test, &y_test)?;
            scores.push(score);
        }

        let sum: f64 = scores.iter().sum();
        Ok(sum / scores.len() as f64)
    }
}

impl<E> Transform<Array2<Float>> for SequentialFeatureSelector<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let k = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, k));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl<E> SequentialFeatureSelector<E, Trained> {
    /// Get the support mask
    pub fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    /// Get the selected features
    pub fn get_feature_names_out(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::{thread_rng, Rng};
    use sklears_core::types::Array1;

    // Mock estimator for testing
    #[derive(Clone)]
    struct MockLinearEstimator {
        coef_: Option<Array1<Float>>,
    }

    impl MockLinearEstimator {
        fn new() -> Self {
            Self { coef_: None }
        }

        fn simple_solve(&self, a: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
            let n = a.nrows();
            if n != a.ncols() {
                return Err(SklearsError::FitError("Matrix must be square".to_string()));
            }
            if n != b.len() {
                return Err(SklearsError::FitError("Dimension mismatch".to_string()));
            }

            let mut aug = Array2::zeros((n, n + 1));
            for i in 0..n {
                for j in 0..n {
                    aug[[i, j]] = a[[i, j]];
                }
                aug[[i, n]] = b[i];
            }

            // Forward elimination with partial pivoting
            for k in 0..n {
                let mut pivot_row = k;
                for i in (k + 1)..n {
                    if aug[[i, k]].abs() > aug[[pivot_row, k]].abs() {
                        pivot_row = i;
                    }
                }

                if pivot_row != k {
                    for j in 0..=n {
                        let temp = aug[[k, j]];
                        aug[[k, j]] = aug[[pivot_row, j]];
                        aug[[pivot_row, j]] = temp;
                    }
                }

                if aug[[k, k]].abs() < 1e-12 {
                    return Err(SklearsError::FitError(
                        "Matrix is singular or ill-conditioned".to_string(),
                    ));
                }

                for i in (k + 1)..n {
                    let factor = aug[[i, k]] / aug[[k, k]];
                    for j in k..=n {
                        aug[[i, j]] -= factor * aug[[k, j]];
                    }
                }
            }

            // Back substitution
            let mut x = Array1::zeros(n);
            for i in (0..n).rev() {
                let mut sum = aug[[i, n]];
                for j in (i + 1)..n {
                    sum -= aug[[i, j]] * x[j];
                }
                x[i] = sum / aug[[i, i]];
            }

            Ok(x)
        }
    }

    impl Estimator for MockLinearEstimator {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<Array2<Float>, Array1<Float>> for MockLinearEstimator {
        type Fitted = MockLinearEstimator;

        fn fit(mut self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
            // Simple linear regression using normal equation with regularization
            let xtx = x.t().dot(x);
            let xty = x.t().dot(y);

            // Add small regularization to avoid singular matrix
            let n_features = x.ncols();
            let mut xtx_reg = xtx.clone();
            for i in 0..n_features {
                xtx_reg[[i, i]] += 1e-5;
            }

            // Simple solve using iterative method
            let coef = self.simple_solve(&xtx_reg, &xty)?;

            self.coef_ = Some(coef);
            Ok(self)
        }
    }

    impl Score<Array2<Float>, Array1<Float>> for MockLinearEstimator {
        type Float = f64;

        fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<f64> {
            let coef = self.coef_.as_ref().ok_or(SklearsError::NotFitted {
                operation: "score".to_string(),
            })?;
            let predictions = x.dot(coef);

            // R² score
            let ss_res = (&predictions - y).mapv(|x| x * x).sum();
            let y_mean = y.mean().unwrap_or(0.0);
            let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

            if ss_tot == 0.0 {
                Ok(1.0)
            } else {
                Ok(1.0 - (ss_res / ss_tot))
            }
        }
    }

    impl HasCoefficients for MockLinearEstimator {
        fn coef(&self) -> SklResult<Array1<Float>> {
            self.coef_.clone().ok_or(SklearsError::NotFitted {
                operation: "coef".to_string(),
            })
        }
    }

    #[test]
    fn test_rfe_basic() {
        // Create a simple dataset where first two features are most important
        let x = array![
            [1.0, 2.0, 0.1, 0.0],
            [2.0, 4.0, 0.2, 0.1],
            [3.0, 6.0, 0.1, 0.0],
            [4.0, 8.0, 0.3, 0.1],
            [5.0, 10.0, 0.2, 0.0],
        ];
        let y = array![5.0, 10.0, 15.0, 20.0, 25.0]; // y = x0 + 2*x1

        let estimator = MockLinearEstimator::new();
        let rfe = RFE::new(estimator).n_features_to_select(2).step(1.0);

        let fitted_rfe = rfe.fit(&x, &y).unwrap();

        // Check that the first two features are selected
        let support = fitted_rfe.support();
        println!("Support: {:?}", support);

        // Check ranking
        let ranking = fitted_rfe.ranking();
        println!("Ranking: {:?}", ranking);

        // The features with highest absolute coefficients should be selected
        // In our test data, y = 1*x0 + 2*x1 + small*x2 + small*x3
        // So x1 should definitely be selected (coef=2), and x0 (coef=1)
        // The support array indicates which features are selected
        let n_selected = support.iter().filter(|&&x| x).count();
        assert_eq!(n_selected, 2);

        // Selected features should have rank 1
        for i in 0..4 {
            if support[i] {
                assert_eq!(ranking[i], 1, "Selected feature {} should have rank 1", i);
            } else {
                assert!(
                    ranking[i] > 1,
                    "Non-selected feature {} should have rank > 1",
                    i
                );
            }
        }

        // Test transform
        let x_transformed = fitted_rfe.transform(&x).unwrap();
        assert_eq!(x_transformed.shape(), &[5, 2]);
    }

    #[test]
    fn test_rfe_step_fraction() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let estimator = MockLinearEstimator::new();
        let rfe = RFE::new(estimator).n_features_to_select(2).step(0.5); // Remove 50% of features at each step

        let fitted_rfe = rfe.fit(&x, &y).unwrap();

        // Should have selected 2 features
        let support = fitted_rfe.support();
        let n_selected = support.iter().filter(|&&x| x).count();
        assert_eq!(n_selected, 2);
    }

    #[test]
    #[should_panic(expected = "step must be > 0")]
    fn test_rfe_invalid_step() {
        let estimator = MockLinearEstimator::new();
        let _rfe = RFE::new(estimator).step(0.0);
    }

    #[test]
    fn test_rfe_too_many_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let estimator = MockLinearEstimator::new();
        let rfe = RFE::new(estimator).n_features_to_select(5);

        let result = rfe.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfecv_basic() {
        // Create a dataset where first two features are informative
        let mut x = Array2::<Float>::zeros((20, 5));
        let mut y = Array1::<Float>::zeros(20);

        for i in 0..20 {
            x[[i, 0]] = i as Float;
            x[[i, 1]] = (i * 2) as Float;
            x[[i, 2]] = thread_rng().gen::<Float>(); // noise
            x[[i, 3]] = thread_rng().gen::<Float>(); // noise
            x[[i, 4]] = thread_rng().gen::<Float>(); // noise

            y[i] = x[[i, 0]] + 2.0 * x[[i, 1]] + 0.1 * thread_rng().gen::<Float>();
        }

        let estimator = MockLinearEstimator::new();
        let rfecv = RFECV::new(estimator).min_features_to_select(1).step(1.0);

        let fitted_rfecv = rfecv.fit(&x, &y).unwrap();

        // Check that CV results were computed
        let cv_results = fitted_rfecv.cv_results();
        assert!(cv_results.mean_test_scores.len() > 0);
        assert_eq!(
            cv_results.mean_test_scores.len(),
            cv_results.std_test_scores.len()
        );
        assert_eq!(
            cv_results.mean_test_scores.len(),
            cv_results.n_features.len()
        );

        // The optimal number of features should be at least 2 (the informative ones)
        let support = fitted_rfecv.support();
        let n_selected = support.iter().filter(|&&x| x).count();
        assert!(n_selected >= 2);
    }

    #[test]
    #[should_panic(expected = "min_features_to_select must be >= 1")]
    fn test_rfecv_invalid_min_features() {
        let estimator = MockLinearEstimator::new();
        let _rfecv = RFECV::new(estimator).min_features_to_select(0);
    }

    #[test]
    fn test_rfecv_custom_cv() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let estimator = MockLinearEstimator::new();
        let cv = Box::new(KFold::new(3));
        let rfecv = RFECV::new(estimator).min_features_to_select(1).cv(cv);

        let fitted_rfecv = rfecv.fit(&x, &y).unwrap();

        // Should have computed CV scores
        let cv_results = fitted_rfecv.cv_results();
        assert!(cv_results.mean_test_scores.len() > 0);
    }

    #[test]
    fn test_select_from_model() {
        let x = array![
            [1.0, 2.0, 0.1, 0.0],
            [2.0, 4.0, 0.2, 0.1],
            [3.0, 6.0, 0.1, 0.0],
            [4.0, 8.0, 0.3, 0.1],
            [5.0, 10.0, 0.2, 0.0],
        ];
        let y = array![5.0, 10.0, 15.0, 20.0, 25.0];

        let estimator = MockLinearEstimator::new();
        let selector = SelectFromModel::new(estimator).threshold(0.5);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        // Should select features with high coefficients
        let support = fitted_selector.get_support().unwrap();
        let n_selected = support.iter().filter(|&&x| x).count();
        assert!(n_selected >= 1);

        let x_transformed = fitted_selector.transform(&x).unwrap();
        assert_eq!(x_transformed.ncols(), n_selected);
    }

    // Note: SequentialFeatureSelector tests commented out due to complex generics
    // They would need more sophisticated trait bounds

    // #[test]
    // fn test_sequential_feature_selector_forward() {
    //     // Test would go here once generics are properly handled
    // }
}
