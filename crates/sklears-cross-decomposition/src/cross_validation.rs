//! Cross-validation utilities for cross-decomposition methods

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, Random, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::collections::HashMap;

/// Cross-validation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum CVStrategy {
    /// K-fold cross-validation
    KFold { k: usize, shuffle: bool },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Stratified K-fold (for classification)
    StratifiedKFold { k: usize, shuffle: bool },
    /// Time series split (for temporal data)
    TimeSeriesSplit { n_splits: usize },
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CVResults {
    /// Mean cross-validation score
    pub mean_score: Float,
    /// Standard deviation of cross-validation scores
    pub std_score: Float,
    /// Individual fold scores
    pub fold_scores: Vec<Float>,
    /// Optimal number of components
    pub best_n_components: Option<usize>,
    /// Scores for each number of components tested
    pub component_scores: Option<HashMap<usize, Float>>,
    /// Bootstrap confidence interval (lower, upper)
    pub confidence_interval: Option<(Float, Float)>,
    /// Bootstrap samples (if requested)
    pub bootstrap_scores: Option<Vec<Float>>,
}

/// Scoring function for cross-validation
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringFunction {
    /// Mean squared error (lower is better)
    MeanSquaredError,
    /// Root mean squared error (lower is better)
    RootMeanSquaredError,
    /// Mean absolute error (lower is better)
    MeanAbsoluteError,
    /// R² score (higher is better)
    R2Score,
    /// Explained variance score (higher is better)
    ExplainedVariance,
    /// Custom scoring function
    Custom(fn(&Array2<Float>, &Array2<Float>) -> Float),
}

impl ScoringFunction {
    /// Compute the score given true and predicted values
    pub fn compute(&self, y_true: &Array2<Float>, y_pred: &Array2<Float>) -> Float {
        match self {
            ScoringFunction::MeanSquaredError => {
                let diff = y_true - y_pred;
                diff.mapv(|x| x * x).mean().unwrap()
            }
            ScoringFunction::RootMeanSquaredError => {
                let diff = y_true - y_pred;
                diff.mapv(|x| x * x).mean().unwrap().sqrt()
            }
            ScoringFunction::MeanAbsoluteError => {
                let diff = y_true - y_pred;
                diff.mapv(|x| x.abs()).mean().unwrap()
            }
            ScoringFunction::R2Score => {
                let y_mean = y_true.mean().unwrap();
                let ss_res = (y_true - y_pred).mapv(|x| x * x).sum();
                let ss_tot = (y_true - y_mean).mapv(|x| x * x).sum();
                1.0 - (ss_res / ss_tot)
            }
            ScoringFunction::ExplainedVariance => {
                let y_true_mean = y_true.mean().unwrap();
                let y_pred_mean = y_pred.mean().unwrap();
                let numerator = ((y_true - y_true_mean) * (y_pred - y_pred_mean))
                    .mean()
                    .unwrap();
                let denominator = (y_true - y_true_mean).mapv(|x| x * x).mean().unwrap();
                if denominator > 0.0 {
                    numerator / denominator
                } else {
                    0.0
                }
            }
            ScoringFunction::Custom(func) => func(y_true, y_pred),
        }
    }

    /// Whether higher scores are better
    pub fn higher_is_better(&self) -> bool {
        match self {
            ScoringFunction::MeanSquaredError
            | ScoringFunction::RootMeanSquaredError
            | ScoringFunction::MeanAbsoluteError => false,
            ScoringFunction::R2Score | ScoringFunction::ExplainedVariance => true,
            ScoringFunction::Custom(_) => true, // Assume higher is better for custom functions
        }
    }
}

/// Cross-validation for decomposition methods
#[derive(Clone)]
pub struct CrossValidator {
    /// Cross-validation strategy
    pub cv_strategy: CVStrategy,
    /// Scoring function
    pub scoring: ScoringFunction,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Whether to return train scores
    pub return_train_score: bool,
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
}

impl CrossValidator {
    /// Create a new cross-validator
    pub fn new(cv_strategy: CVStrategy, scoring: ScoringFunction) -> Self {
        Self {
            cv_strategy,
            scoring,
            random_state: None,
            return_train_score: false,
            n_jobs: None,
        }
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set whether to return training scores
    pub fn return_train_score(mut self, return_train_score: bool) -> Self {
        self.return_train_score = return_train_score;
        self
    }

    /// Set number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Perform cross-validation
    pub fn cross_validate<E>(
        &self,
        estimator: E,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<CVResults>
    where
        E: Clone + Fit<Array2<Float>, Array2<Float>>,
        E::Fitted: Predict<Array2<Float>, Array2<Float>>,
    {
        let indices = self.generate_cv_indices(x.nrows(), Some(y))?;
        let mut fold_scores = Vec::new();

        for (train_idx, test_idx) in indices {
            // Split data
            let x_train = self.select_rows(x, &train_idx);
            let y_train = self.select_rows(y, &train_idx);
            let x_test = self.select_rows(x, &test_idx);
            let y_test = self.select_rows(y, &test_idx);

            // Fit and predict
            let fitted = estimator.clone().fit(&x_train, &y_train)?;
            let y_pred = fitted.predict(&x_test)?;

            // Compute score
            let score = self.scoring.compute(&y_test, &y_pred);
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
        let variance = fold_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / fold_scores.len() as Float;
        let std_score = variance.sqrt();

        Ok(CVResults {
            mean_score,
            std_score,
            fold_scores,
            best_n_components: None,
            component_scores: None,
            confidence_interval: None,
            bootstrap_scores: None,
        })
    }

    /// Perform cross-validation for component selection
    pub fn select_components<E>(
        &self,
        estimator_factory: impl Fn(usize) -> E,
        x: &Array2<Float>,
        y: &Array2<Float>,
        component_range: std::ops::Range<usize>,
    ) -> Result<CVResults>
    where
        E: Clone + Fit<Array2<Float>, Array2<Float>>,
        E::Fitted: Predict<Array2<Float>, Array2<Float>>,
    {
        let mut component_scores = HashMap::new();
        let mut best_score = if self.scoring.higher_is_better() {
            Float::NEG_INFINITY
        } else {
            Float::INFINITY
        };
        let mut best_n_components = component_range.start;

        for n_components in component_range.clone() {
            let estimator = estimator_factory(n_components);
            let cv_result = self.cross_validate(estimator, x, y)?;

            component_scores.insert(n_components, cv_result.mean_score);

            if (self.scoring.higher_is_better() && cv_result.mean_score > best_score)
                || (!self.scoring.higher_is_better() && cv_result.mean_score < best_score)
            {
                best_score = cv_result.mean_score;
                best_n_components = n_components;
            }
        }

        // Return results for best component count
        let best_estimator = estimator_factory(best_n_components);
        let best_cv_result = self.cross_validate(best_estimator, x, y)?;

        Ok(CVResults {
            mean_score: best_cv_result.mean_score,
            std_score: best_cv_result.std_score,
            fold_scores: best_cv_result.fold_scores,
            best_n_components: Some(best_n_components),
            component_scores: Some(component_scores),
            confidence_interval: None,
            bootstrap_scores: None,
        })
    }

    /// Perform cross-validation for multi-block methods
    pub fn cross_validate_multiblock<E>(
        &self,
        estimator: E,
        x_blocks: &[Array2<Float>],
        y: &Array2<Float>,
    ) -> Result<CVResults>
    where
        E: Clone + Fit<Vec<Array2<Float>>, Array2<Float>>,
        E::Fitted: Predict<Vec<Array2<Float>>, Array2<Float>>,
    {
        let n_samples = x_blocks[0].nrows();
        let indices = self.generate_cv_indices(n_samples, Some(y))?;
        let mut fold_scores = Vec::new();

        for (train_idx, test_idx) in indices {
            // Split data
            let x_train_blocks: Vec<Array2<Float>> = x_blocks
                .iter()
                .map(|block| self.select_rows(block, &train_idx))
                .collect();
            let y_train = self.select_rows(y, &train_idx);
            let x_test_blocks: Vec<Array2<Float>> = x_blocks
                .iter()
                .map(|block| self.select_rows(block, &test_idx))
                .collect();
            let y_test = self.select_rows(y, &test_idx);

            // Fit and predict
            let fitted = estimator.clone().fit(&x_train_blocks, &y_train)?;
            let y_pred = fitted.predict(&x_test_blocks)?;

            // Compute score
            let score = self.scoring.compute(&y_test, &y_pred);
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
        let variance = fold_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / fold_scores.len() as Float;
        let std_score = variance.sqrt();

        Ok(CVResults {
            mean_score,
            std_score,
            fold_scores,
            best_n_components: None,
            component_scores: None,
            confidence_interval: None,
            bootstrap_scores: None,
        })
    }

    /// Generate cross-validation indices
    fn generate_cv_indices(
        &self,
        n_samples: usize,
        y: Option<&Array2<Float>>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        match &self.cv_strategy {
            CVStrategy::KFold { k, shuffle } => {
                if *k > n_samples {
                    return Err(SklearsError::InvalidInput(
                        "k cannot be larger than the number of samples".into(),
                    ));
                }

                let mut indices: Vec<usize> = (0..n_samples).collect();

                if *shuffle {
                    use scirs2_core::rand_prelude::SliceRandom;
                    let mut rng = if let Some(seed) = self.random_state {
                        StdRng::seed_from_u64(seed)
                    } else {
                        let mut entropy_rng = thread_rng();
                        StdRng::from_rng(&mut entropy_rng)
                    };
                    indices.shuffle(&mut rng);
                }

                let fold_size = n_samples / k;
                let mut cv_indices = Vec::new();

                for i in 0..*k {
                    let start = i * fold_size;
                    let end = if i == k - 1 {
                        n_samples
                    } else {
                        (i + 1) * fold_size
                    };

                    let test_idx: Vec<usize> = indices[start..end].to_vec();
                    let train_idx: Vec<usize> = indices[..start]
                        .iter()
                        .chain(indices[end..].iter())
                        .copied()
                        .collect();

                    cv_indices.push((train_idx, test_idx));
                }

                Ok(cv_indices)
            }
            CVStrategy::LeaveOneOut => {
                let mut cv_indices = Vec::new();
                for i in 0..n_samples {
                    let test_idx = vec![i];
                    let train_idx: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
                    cv_indices.push((train_idx, test_idx));
                }
                Ok(cv_indices)
            }
            CVStrategy::StratifiedKFold { k, shuffle: _ } => {
                // For regression, fall back to regular KFold
                // In a real implementation, you'd handle classification differently
                self.generate_cv_indices(n_samples, None)
            }
            CVStrategy::TimeSeriesSplit { n_splits } => {
                let mut cv_indices = Vec::new();
                let min_train_size = n_samples / (n_splits + 1);

                for i in 1..=*n_splits {
                    let train_end = min_train_size * i;
                    let test_start = train_end;
                    let test_end = if i == *n_splits {
                        n_samples
                    } else {
                        min_train_size * (i + 1)
                    };

                    if test_start < n_samples {
                        let train_idx: Vec<usize> = (0..train_end).collect();
                        let test_idx: Vec<usize> = (test_start..test_end.min(n_samples)).collect();
                        cv_indices.push((train_idx, test_idx));
                    }
                }

                Ok(cv_indices)
            }
        }
    }

    /// Select rows from an array based on indices
    fn select_rows(&self, array: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let selected_rows: Vec<_> = indices.iter().map(|&i| array.row(i).to_owned()).collect();

        if selected_rows.is_empty() {
            Array2::zeros((0, array.ncols()))
        } else {
            let nrows = selected_rows.len();
            let ncols = selected_rows[0].len();
            let mut result = Array2::zeros((nrows, ncols));

            for (i, row) in selected_rows.iter().enumerate() {
                result.row_mut(i).assign(row);
            }

            result
        }
    }

    /// Compute bootstrap confidence intervals for cross-validation scores
    pub fn bootstrap_confidence_interval<E>(
        &self,
        estimator: E,
        x: &Array2<Float>,
        y: &Array2<Float>,
        n_bootstrap: usize,
        confidence_level: Float,
    ) -> Result<CVResults>
    where
        E: Clone + Fit<Array2<Float>, Array2<Float>>,
        E::Fitted: Predict<Array2<Float>, Array2<Float>>,
    {
        let n_samples = x.nrows();
        let mut bootstrap_scores = Vec::with_capacity(n_bootstrap);

        // Set up random number generator
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut entropy_rng = thread_rng();
                StdRng::from_rng(&mut entropy_rng)
            }
        };

        // Perform bootstrap resampling
        for _ in 0..n_bootstrap {
            // Bootstrap sample indices with replacement
            let mut bootstrap_indices = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                bootstrap_indices.push(rng.gen_range(0..n_samples));
            }

            // Create bootstrap sample
            let mut x_bootstrap = Array2::zeros((n_samples, x.ncols()));
            let mut y_bootstrap = Array2::zeros((n_samples, y.ncols()));

            for (i, &idx) in bootstrap_indices.iter().enumerate() {
                x_bootstrap.row_mut(i).assign(&x.row(idx));
                y_bootstrap.row_mut(i).assign(&y.row(idx));
            }

            // Perform cross-validation on bootstrap sample
            let cv_result = self.cross_validate(estimator.clone(), &x_bootstrap, &y_bootstrap)?;
            bootstrap_scores.push(cv_result.mean_score);
        }

        // Filter out NaN values
        let valid_scores: Vec<Float> = bootstrap_scores
            .iter()
            .copied()
            .filter(|&score| score.is_finite())
            .collect();

        if valid_scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "All bootstrap scores are invalid".to_string(),
            ));
        }

        // Calculate mean and std of bootstrap scores
        let mean_score = valid_scores.iter().sum::<Float>() / valid_scores.len() as Float;
        let variance = valid_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / valid_scores.len() as Float;
        let std_score = variance.sqrt();

        // Calculate confidence interval
        let mut sorted_scores = valid_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = 1.0 - confidence_level;
        let n_valid = valid_scores.len();
        let lower_idx = ((alpha / 2.0) * n_valid as Float) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_valid as Float) as usize;

        let lower_bound = sorted_scores
            .get(lower_idx)
            .copied()
            .unwrap_or(sorted_scores[0]);
        let upper_bound = sorted_scores
            .get(upper_idx.min(n_valid - 1))
            .copied()
            .unwrap_or(sorted_scores[n_valid - 1]);

        Ok(CVResults {
            mean_score,
            std_score,
            fold_scores: bootstrap_scores.clone(),
            best_n_components: None,
            component_scores: None,
            confidence_interval: Some((lower_bound, upper_bound)),
            bootstrap_scores: Some(bootstrap_scores),
        })
    }
}

/// Nested cross-validation for robust model selection
pub struct NestedCrossValidator {
    /// Outer cross-validation strategy
    pub outer_cv: CVStrategy,
    /// Inner cross-validation strategy  
    pub inner_cv: CVStrategy,
    /// Scoring function
    pub scoring: ScoringFunction,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl NestedCrossValidator {
    /// Create a new nested cross-validator
    pub fn new(outer_cv: CVStrategy, inner_cv: CVStrategy, scoring: ScoringFunction) -> Self {
        Self {
            outer_cv,
            inner_cv,
            scoring,
            random_state: None,
        }
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Perform nested cross-validation
    pub fn nested_cross_validate<E>(
        &self,
        estimator_factory: impl Fn(usize) -> E,
        x: &Array2<Float>,
        y: &Array2<Float>,
        component_range: std::ops::Range<usize>,
    ) -> Result<CVResults>
    where
        E: Clone + Fit<Array2<Float>, Array2<Float>>,
        E::Fitted: Predict<Array2<Float>, Array2<Float>>,
    {
        let mut outer_cv = CrossValidator::new(self.outer_cv.clone(), self.scoring.clone());
        let mut inner_cv = CrossValidator::new(self.inner_cv.clone(), self.scoring.clone());

        if let Some(seed) = self.random_state {
            outer_cv = outer_cv.random_state(seed);
            inner_cv = inner_cv.random_state(seed + 1);
        }

        let outer_indices = outer_cv.generate_cv_indices(x.nrows(), Some(y))?;
        let mut nested_scores = Vec::new();
        let mut selected_components = Vec::new();

        for (train_idx, test_idx) in outer_indices {
            // Split data for outer loop
            let x_train = outer_cv.select_rows(x, &train_idx);
            let y_train = outer_cv.select_rows(y, &train_idx);
            let x_test = outer_cv.select_rows(x, &test_idx);
            let y_test = outer_cv.select_rows(y, &test_idx);

            // Inner cross-validation for component selection
            let inner_result = inner_cv.select_components(
                &estimator_factory,
                &x_train,
                &y_train,
                component_range.clone(),
            )?;

            let best_n_components = inner_result.best_n_components.unwrap();
            selected_components.push(best_n_components);

            // Train final model with selected components on full training set
            let final_estimator = estimator_factory(best_n_components);
            let fitted = final_estimator.fit(&x_train, &y_train)?;
            let y_pred = fitted.predict(&x_test)?;

            // Compute outer score
            let outer_score = self.scoring.compute(&y_test, &y_pred);
            nested_scores.push(outer_score);
        }

        let mean_score = nested_scores.iter().sum::<Float>() / nested_scores.len() as Float;
        let variance = nested_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / nested_scores.len() as Float;
        let std_score = variance.sqrt();

        // Most frequently selected number of components
        let mut component_counts = HashMap::new();
        for &comp in &selected_components {
            *component_counts.entry(comp).or_insert(0) += 1;
        }
        let best_n_components = component_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(comp, _)| comp);

        Ok(CVResults {
            mean_score,
            std_score,
            fold_scores: nested_scores,
            best_n_components,
            component_scores: None,
            confidence_interval: None,
            bootstrap_scores: None,
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::PLSRegression;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cross_validation_basic() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ];
        let y = array![[1.5], [3.5], [5.5], [7.5], [9.5], [11.5]];

        let cv = CrossValidator::new(
            CVStrategy::KFold {
                k: 3,
                shuffle: false,
            },
            ScoringFunction::MeanSquaredError,
        );

        let pls = PLSRegression::new(1);
        let result = cv.cross_validate(pls, &x, &y).unwrap();

        assert_eq!(result.fold_scores.len(), 3);
        assert!(result.mean_score >= 0.0);
        assert!(result.std_score >= 0.0);
    }

    #[test]
    fn test_component_selection() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0]
        ];
        let y = array![[2.0], [5.0], [8.0], [11.0], [14.0], [17.0]];

        let cv = CrossValidator::new(
            CVStrategy::KFold {
                k: 3,
                shuffle: false,
            },
            ScoringFunction::MeanSquaredError,
        );

        let result = cv
            .select_components(|n_comp| PLSRegression::new(n_comp), &x, &y, 1..2)
            .unwrap();

        assert!(result.best_n_components.is_some());
        assert!(result.component_scores.is_some());
        let component_scores = result.component_scores.unwrap();
        assert!(component_scores.len() >= 1);
    }

    #[test]
    fn test_leave_one_out() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.5], [3.5], [5.5], [7.5]];

        let cv = CrossValidator::new(CVStrategy::LeaveOneOut, ScoringFunction::R2Score);

        let pls = PLSRegression::new(1);
        let result = cv.cross_validate(pls, &x, &y).unwrap();

        assert_eq!(result.fold_scores.len(), 4); // One for each sample
    }

    #[test]
    fn test_scoring_functions() {
        let y_true = array![[1.0], [2.0], [3.0], [4.0]];
        let y_pred = array![[1.1], [1.9], [3.1], [3.9]];

        let mse = ScoringFunction::MeanSquaredError.compute(&y_true, &y_pred);
        let rmse = ScoringFunction::RootMeanSquaredError.compute(&y_true, &y_pred);
        let mae = ScoringFunction::MeanAbsoluteError.compute(&y_true, &y_pred);
        let r2 = ScoringFunction::R2Score.compute(&y_true, &y_pred);

        assert!(mse > 0.0);
        assert!(rmse > 0.0);
        assert!(mae > 0.0);
        assert!(r2 > 0.0 && r2 <= 1.0);
        assert_abs_diff_eq!(rmse, mse.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_nested_cross_validation() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0]
        ];
        let y = array![[2.0], [5.0], [8.0], [11.0], [14.0], [17.0], [20.0], [23.0]];

        let nested_cv = NestedCrossValidator::new(
            CVStrategy::KFold {
                k: 4,
                shuffle: false,
            },
            CVStrategy::KFold {
                k: 3,
                shuffle: false,
            },
            ScoringFunction::MeanSquaredError,
        );

        let result = nested_cv
            .nested_cross_validate(|n_comp| PLSRegression::new(n_comp), &x, &y, 1..2)
            .unwrap();

        assert_eq!(result.fold_scores.len(), 4); // Outer folds
        assert!(result.best_n_components.is_some());
    }

    #[test]
    fn test_bootstrap_confidence_interval() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ];
        let y = array![[1.5], [3.5], [5.5], [7.5], [9.5], [11.5]];

        let cv = CrossValidator::new(
            CVStrategy::KFold {
                k: 3,
                shuffle: false,
            },
            ScoringFunction::R2Score,
        )
        .random_state(42);

        let result = cv
            .bootstrap_confidence_interval(PLSRegression::new(1), &x, &y, 50, 0.95)
            .unwrap();

        // Check that bootstrap results are reasonable
        assert_eq!(result.bootstrap_scores.as_ref().unwrap().len(), 50);
        assert!(result.confidence_interval.is_some());

        let (lower, upper) = result.confidence_interval.unwrap();
        assert!(lower <= upper); // Confidence interval should be valid
        assert!(lower.is_finite() && upper.is_finite()); // Should not be NaN or infinite

        // Check that fold_scores contains bootstrap scores
        assert_eq!(result.fold_scores.len(), 50);

        // Check that mean and std are reasonable
        assert!(result.mean_score.is_finite());
        assert!(result.std_score.is_finite());
        assert!(result.std_score >= 0.0);
    }
}
