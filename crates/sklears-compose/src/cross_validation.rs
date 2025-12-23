//! Cross-validation framework for composed models and pipelines
//!
//! This module provides comprehensive cross-validation capabilities specifically designed
//! for complex composed models including pipelines, ensembles, and meta-learners.

use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::Result as SklResult,
    prelude::{Fit, Predict, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::{Pipeline, PipelinePredictor, PipelineStep, PipelineTrained};

/// Cross-validation framework for complex composed models
pub struct ComposedModelCrossValidator {
    /// Cross-validation configuration
    pub config: CrossValidationConfig,
    /// Scoring configuration
    pub scoring: ScoringConfig,
    /// Parallel execution settings
    pub parallel: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Cross-validation strategy
    pub strategy: CVStrategy,
    /// Number of folds (for k-fold CV)
    pub n_folds: usize,
    /// Test size ratio (for train-test split)
    pub test_size: f64,
    /// Number of repetitions
    pub n_repeats: usize,
    /// Shuffle data before splitting
    pub shuffle: bool,
    /// Stratification for classification
    pub stratified: bool,
    /// Time series specific settings
    pub time_series_config: Option<TimeSeriesConfig>,
}

/// Time series cross-validation configuration
#[derive(Debug, Clone)]
pub struct TimeSeriesConfig {
    /// Maximum training size for time series CV
    pub max_train_size: Option<usize>,
    /// Gap between training and test sets
    pub gap: usize,
    /// Test size for time series CV
    pub test_size: usize,
    /// Expanding window vs sliding window
    pub expanding_window: bool,
}

/// Cross-validation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum CVStrategy {
    /// K-fold cross-validation
    KFold,
    /// Stratified K-fold (for classification)
    StratifiedKFold,
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Leave-P-out cross-validation
    LeavePOut { p: usize },
    /// Time series cross-validation
    TimeSeriesSplit,
    /// Group-based cross-validation
    GroupKFold { groups: Vec<usize> },
    /// Monte Carlo cross-validation
    ShuffleSplit { n_splits: usize },
    /// Bootstrap cross-validation
    Bootstrap { n_bootstrap: usize },
    /// Nested cross-validation
    Nested {
        inner_cv: Box<CVStrategy>,
        outer_cv: Box<CVStrategy>,
    },
}

/// Scoring configuration for cross-validation
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Primary scoring metric
    pub primary_metric: ScoringMetric,
    /// Additional metrics to compute
    pub additional_metrics: Vec<ScoringMetric>,
    /// Whether to use scoring for model selection
    pub use_for_selection: bool,
    /// Custom scoring function
    pub custom_scorer: Option<fn(&Array1<f64>, &Array1<f64>) -> f64>,
}

/// Available scoring metrics
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringMetric {
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    LogLoss,
    Custom { name: String },
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// Test scores for each fold
    pub test_scores: Vec<f64>,
    /// Training scores for each fold (if computed)
    pub train_scores: Option<Vec<f64>>,
    /// Fit times for each fold
    pub fit_times: Vec<Duration>,
    /// Score times for each fold
    pub score_times: Vec<Duration>,
    /// Statistical summary
    pub summary: CVSummary,
    /// Detailed fold results
    pub fold_results: Vec<FoldResult>,
    /// Additional metrics results
    pub additional_metrics: HashMap<String, Vec<f64>>,
}

/// Statistical summary of cross-validation results
#[derive(Debug, Clone)]
pub struct CVSummary {
    /// Mean test score
    pub mean_test_score: f64,
    /// Standard deviation of test scores
    pub std_test_score: f64,
    /// Minimum test score
    pub min_test_score: f64,
    /// Maximum test score
    pub max_test_score: f64,
    /// Confidence interval (95%)
    pub confidence_interval: (f64, f64),
    /// Mean training score (if computed)
    pub mean_train_score: Option<f64>,
    /// Standard deviation of training scores (if computed)
    pub std_train_score: Option<f64>,
}

/// Results for a single fold
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Fold index
    pub fold_index: usize,
    /// Test score
    pub test_score: f64,
    /// Training score (if computed)
    pub train_score: Option<f64>,
    /// Fit time
    pub fit_time: Duration,
    /// Score time
    pub score_time: Duration,
    /// Training indices
    pub train_indices: Vec<usize>,
    /// Test indices
    pub test_indices: Vec<usize>,
    /// Additional metrics for this fold
    pub additional_metrics: HashMap<String, f64>,
}

impl ComposedModelCrossValidator {
    /// Create new cross-validator
    #[must_use]
    pub fn new(config: CrossValidationConfig) -> Self {
        Self {
            config,
            scoring: ScoringConfig::default(),
            parallel: false,
            random_state: None,
        }
    }

    /// Set scoring configuration
    #[must_use]
    pub fn with_scoring(mut self, scoring: ScoringConfig) -> Self {
        self.scoring = scoring;
        self
    }

    /// Enable parallel execution
    #[must_use]
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set random state for reproducibility
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Perform cross-validation on a pipeline
    pub fn cross_validate_pipeline<'a, S>(
        &self,
        pipeline: Pipeline<S>,
        x: &ArrayView2<'a, Float>,
        y: &ArrayView1<'a, Float>,
    ) -> SklResult<CrossValidationResults>
    where
        S: Clone,
        Pipeline<S>:
            for<'c, 'b> Fit<ArrayView2<'b, Float>, Option<&'c ArrayView1<'b, Float>>> + Clone,
        for<'c, 'b> <Pipeline<S> as Fit<ArrayView2<'b, Float>, Option<&'c ArrayView1<'b, Float>>>>::Fitted:
            Into<Pipeline<PipelineTrained>>,
    {
        let splits = self.generate_splits(x.nrows(), Some(y))?;
        let mut results = CrossValidationResults::new();

        for (fold_idx, (train_indices, test_indices)) in splits.into_iter().enumerate() {
            let fold_result = self.evaluate_fold_pipeline(
                pipeline.clone(),
                x,
                y,
                &train_indices,
                &test_indices,
                fold_idx,
            )?;
            results.add_fold_result(fold_result);
        }

        results.compute_summary();
        Ok(results)
    }

    /// Perform cross-validation on any estimator
    pub fn cross_validate<'a, E, S>(
        &self,
        estimator: E,
        x: &ArrayView2<'a, Float>,
        y: &ArrayView1<'a, Float>,
    ) -> SklResult<CrossValidationResults>
    where
        E: for<'c, 'b> Fit<ArrayView2<'b, Float>, Option<&'c ArrayView1<'b, Float>>> + Clone,
        for<'b> <E as sklears_core::traits::Fit<
            ArrayView2<'b, Float>,
            Option<&'b ArrayView1<'b, Float>>,
        >>::Fitted: Clone,
        S: Clone,
    {
        let splits = self.generate_splits(x.nrows(), Some(y))?;
        let mut results = CrossValidationResults::new();

        for (fold_idx, (train_indices, test_indices)) in splits.into_iter().enumerate() {
            let fold_result = self.evaluate_fold_estimator(
                estimator.clone(),
                x,
                y,
                &train_indices,
                &test_indices,
                fold_idx,
            )?;
            results.add_fold_result(fold_result);
        }

        results.compute_summary();
        Ok(results)
    }

    /// Perform nested cross-validation for hyperparameter tuning
    pub fn nested_cross_validate<'a, E, S>(
        &self,
        estimator: E,
        x: &ArrayView2<'a, Float>,
        y: &ArrayView1<'a, Float>,
        param_grid: HashMap<String, Vec<f64>>,
    ) -> SklResult<NestedCVResults>
    where
        E: for<'c, 'b> Fit<ArrayView2<'b, Float>, Option<&'c ArrayView1<'b, Float>>> + Clone,
        for<'b> <E as sklears_core::traits::Fit<
            ArrayView2<'b, Float>,
            Option<&'b ArrayView1<'b, Float>>,
        >>::Fitted: Clone,
        S: Clone,
    {
        if let CVStrategy::Nested { inner_cv, outer_cv } = &self.config.strategy {
            let outer_splits = self.generate_splits_strategy(outer_cv, x.nrows(), Some(y))?;
            let mut nested_results = NestedCVResults::new();

            for (outer_fold, (outer_train, outer_test)) in outer_splits.into_iter().enumerate() {
                // Create inner CV for hyperparameter tuning
                let inner_config = CrossValidationConfig {
                    strategy: (**inner_cv).clone(),
                    ..self.config.clone()
                };
                let inner_cv = ComposedModelCrossValidator::new(inner_config);

                // Get training data for inner CV
                let x_outer_train = x.select(Axis(0), &outer_train);
                let y_outer_train = y.select(Axis(0), &outer_train);

                // Perform hyperparameter search using inner CV
                let best_params = self.grid_search_cv(
                    &inner_cv,
                    estimator.clone(),
                    &x_outer_train.view(),
                    &y_outer_train.view(),
                    &param_grid,
                )?;

                // Train model with best parameters on outer training set
                let best_model = estimator.clone();
                // Apply best parameters (simplified - in practice would need parameter setting)
                let y_outer_train_view = y_outer_train.view();
                let fitted_model =
                    best_model.fit(&x_outer_train.view(), &Some(&y_outer_train_view))?;

                // Evaluate on outer test set
                let x_outer_test = x.select(Axis(0), &outer_test);
                let y_outer_test = y.select(Axis(0), &outer_test);

                let predictions = self.predict_fitted(&fitted_model, &x_outer_test.view())?;
                let test_score = self.score(&predictions.view(), &y_outer_test.view())?;

                nested_results.add_outer_fold_result(OuterFoldResult {
                    fold_index: outer_fold,
                    best_params: best_params.clone(),
                    test_score,
                    inner_cv_score: 0.0, // Would be computed from inner CV
                });
            }

            Ok(nested_results)
        } else {
            Err(SklearsError::InvalidInput(
                "Nested cross-validation requires Nested strategy".to_string(),
            ))
        }
    }

    /// Generate cross-validation splits
    fn generate_splits(
        &self,
        n_samples: usize,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.generate_splits_strategy(&self.config.strategy, n_samples, y)
    }

    /// Generate splits for a specific strategy
    fn generate_splits_strategy(
        &self,
        strategy: &CVStrategy,
        n_samples: usize,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        match strategy {
            CVStrategy::KFold => self.k_fold_split(n_samples, &mut rng),
            CVStrategy::StratifiedKFold => {
                if let Some(y) = y {
                    self.stratified_k_fold_split(n_samples, y, &mut rng)
                } else {
                    self.k_fold_split(n_samples, &mut rng)
                }
            }
            CVStrategy::LeaveOneOut => self.leave_one_out_split(n_samples),
            CVStrategy::LeavePOut { p } => self.leave_p_out_split(n_samples, *p),
            CVStrategy::TimeSeriesSplit => self.time_series_split(n_samples),
            CVStrategy::GroupKFold { groups } => self.group_k_fold_split(n_samples, groups),
            CVStrategy::ShuffleSplit { n_splits } => {
                self.shuffle_split(n_samples, *n_splits, &mut rng)
            }
            CVStrategy::Bootstrap { n_bootstrap } => {
                self.bootstrap_split(n_samples, *n_bootstrap, &mut rng)
            }
            CVStrategy::Nested { outer_cv, .. } => {
                self.generate_splits_strategy(outer_cv, n_samples, y)
            }
        }
    }

    /// K-fold cross-validation split
    fn k_fold_split(
        &self,
        n_samples: usize,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.config.shuffle {
            indices.shuffle(rng);
        }

        let fold_size = n_samples / self.config.n_folds;
        let mut splits = Vec::new();

        for fold in 0..self.config.n_folds {
            let start = fold * fold_size;
            let end = if fold == self.config.n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let test_indices = indices[start..end].to_vec();
            let train_indices = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Stratified K-fold cross-validation split
    fn stratified_k_fold_split(
        &self,
        n_samples: usize,
        y: &ArrayView1<'_, Float>,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &value) in y.iter().enumerate() {
            let class = value as i32;
            class_indices.entry(class).or_default().push(idx);
        }

        // Shuffle indices within each class
        if self.config.shuffle {
            for indices in class_indices.values_mut() {
                indices.shuffle(rng);
            }
        }

        let mut splits = vec![(Vec::new(), Vec::new()); self.config.n_folds];

        // Distribute samples from each class across folds
        for (_, indices) in class_indices {
            let class_fold_size = indices.len() / self.config.n_folds;

            for fold in 0..self.config.n_folds {
                let start = fold * class_fold_size;
                let end = if fold == self.config.n_folds - 1 {
                    indices.len()
                } else {
                    (fold + 1) * class_fold_size
                };

                // Add to test set for this fold
                splits[fold].1.extend(indices[start..end].iter());

                // Add to training set for other folds
                for other_fold in 0..self.config.n_folds {
                    if other_fold != fold {
                        splits[other_fold].0.extend(indices[start..end].iter());
                    }
                }
            }
        }

        Ok(splits)
    }

    /// Leave-one-out cross-validation split
    fn leave_one_out_split(&self, n_samples: usize) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices = (0..n_samples).filter(|&x| x != i).collect();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Leave-P-out cross-validation split
    fn leave_p_out_split(
        &self,
        n_samples: usize,
        p: usize,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        if p >= n_samples {
            return Err(SklearsError::InvalidInput(
                "P must be less than number of samples".to_string(),
            ));
        }

        let mut splits = Vec::new();
        let indices: Vec<usize> = (0..n_samples).collect();

        // Generate all combinations of p indices
        for test_indices in combinations(&indices, p) {
            let train_indices = indices
                .iter()
                .filter(|&x| !test_indices.contains(x))
                .copied()
                .collect();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Time series cross-validation split
    fn time_series_split(&self, n_samples: usize) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let config = self.config.time_series_config.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "Time series config required for TimeSeriesSplit".to_string(),
            )
        })?;

        let mut splits = Vec::new();
        let test_size = config.test_size;
        let gap = config.gap;

        let mut start = test_size + gap;
        while start + test_size <= n_samples {
            let test_end = start + test_size;
            let train_end = start - gap;

            let train_start = if config.expanding_window {
                0
            } else {
                config
                    .max_train_size
                    .map_or(0, |size| train_end.saturating_sub(size))
            };

            let train_indices = (train_start..train_end).collect();
            let test_indices = (start..test_end).collect();

            splits.push((train_indices, test_indices));
            start += test_size;
        }

        Ok(splits)
    }

    /// Group K-fold cross-validation split
    fn group_k_fold_split(
        &self,
        n_samples: usize,
        groups: &[usize],
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        if groups.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Groups length must equal number of samples".to_string(),
            ));
        }

        // Group indices by group labels
        let mut group_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_indices.entry(group).or_default().push(idx);
        }

        let unique_groups: Vec<usize> = group_indices.keys().copied().collect();
        let n_groups = unique_groups.len();

        if n_groups < self.config.n_folds {
            return Err(SklearsError::InvalidInput(
                "Not enough groups for the number of folds".to_string(),
            ));
        }

        let group_fold_size = n_groups / self.config.n_folds;
        let mut splits = Vec::new();

        for fold in 0..self.config.n_folds {
            let start = fold * group_fold_size;
            let end = if fold == self.config.n_folds - 1 {
                n_groups
            } else {
                (fold + 1) * group_fold_size
            };

            let test_groups = &unique_groups[start..end];
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            for &group in &unique_groups {
                if test_groups.contains(&group) {
                    test_indices.extend(&group_indices[&group]);
                } else {
                    train_indices.extend(&group_indices[&group]);
                }
            }

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Shuffle split cross-validation
    fn shuffle_split(
        &self,
        n_samples: usize,
        n_splits: usize,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let test_size = (n_samples as f64 * self.config.test_size) as usize;
        let mut splits = Vec::new();

        for _ in 0..n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);

            let test_indices = indices[..test_size].to_vec();
            let train_indices = indices[test_size..].to_vec();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Bootstrap cross-validation split
    fn bootstrap_split(
        &self,
        n_samples: usize,
        n_bootstrap: usize,
        rng: &mut StdRng,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();

        for _ in 0..n_bootstrap {
            let mut train_indices = Vec::new();
            let mut test_indices_set = HashSet::new();

            // Bootstrap sampling for training set
            for _ in 0..n_samples {
                let idx = rng.gen_range(0..n_samples);
                train_indices.push(idx);
            }

            // Out-of-bag samples for test set
            for i in 0..n_samples {
                if !train_indices.contains(&i) {
                    test_indices_set.insert(i);
                }
            }

            let test_indices: Vec<usize> = test_indices_set.into_iter().collect();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Evaluate a single fold for pipeline
    fn evaluate_fold_pipeline<'a, S>(
        &self,
        pipeline: Pipeline<S>,
        x: &ArrayView2<'a, Float>,
        y: &ArrayView1<'a, Float>,
        train_indices: &[usize],
        test_indices: &[usize],
        fold_index: usize,
    ) -> SklResult<FoldResult>
    where
        S: Clone,
        Pipeline<S>:
            for<'c, 'b> Fit<ArrayView2<'b, Float>, Option<&'c ArrayView1<'b, Float>>> + Clone,
        for<'c, 'b> <Pipeline<S> as Fit<ArrayView2<'b, Float>, Option<&'c ArrayView1<'b, Float>>>>::Fitted:
            Into<Pipeline<PipelineTrained>>,
    {
        let x_train = x.select(Axis(0), train_indices);
        let y_train = y.select(Axis(0), train_indices);
        let x_test = x.select(Axis(0), test_indices);
        let y_test = y.select(Axis(0), test_indices);

        // Fit timing
        let fit_start = Instant::now();
        let y_train_view = y_train.view();
        let fitted_pipeline = pipeline.fit(&x_train.view(), &Some(&y_train_view))?;
        let fit_time = fit_start.elapsed();

        // Convert to trained pipeline
        let trained_pipeline: Pipeline<PipelineTrained> = fitted_pipeline.into();

        // Score timing
        let score_start = Instant::now();
        let predictions = trained_pipeline.predict(&x_test.view())?;
        let test_score = self.score(&predictions.view(), &y_test.view())?;
        let score_time = score_start.elapsed();

        // Training score (optional)
        let train_score = if self.scoring.use_for_selection {
            let train_predictions = trained_pipeline.predict(&x_train.view())?;
            Some(self.score(&train_predictions.view(), &y_train.view())?)
        } else {
            None
        };

        // Additional metrics
        let additional_metrics =
            self.compute_additional_metrics(&predictions.view(), &y_test.view())?;

        Ok(FoldResult {
            fold_index,
            test_score,
            train_score,
            fit_time,
            score_time,
            train_indices: train_indices.to_vec(),
            test_indices: test_indices.to_vec(),
            additional_metrics,
        })
    }

    /// Evaluate a single fold for general estimator
    fn evaluate_fold_estimator<E>(
        &self,
        estimator: E,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        train_indices: &[usize],
        test_indices: &[usize],
        fold_index: usize,
    ) -> SklResult<FoldResult>
    where
        E: for<'a> Fit<ArrayView2<'a, Float>, Option<&'a ArrayView1<'a, Float>>> + Clone,
        for<'b> <E as sklears_core::traits::Fit<
            ArrayView2<'b, Float>,
            Option<&'b ArrayView1<'b, Float>>,
        >>::Fitted: Clone,
    {
        let x_train = x.select(Axis(0), train_indices);
        let y_train = y.select(Axis(0), train_indices);
        let x_test = x.select(Axis(0), test_indices);
        let y_test = y.select(Axis(0), test_indices);

        // Fit timing
        let fit_start = Instant::now();
        let y_train_view = y_train.view();
        let fitted_estimator = estimator.fit(&x_train.view(), &Some(&y_train_view))?;
        let fit_time = fit_start.elapsed();

        // Score timing
        let score_start = Instant::now();
        let predictions = self.predict_fitted(&fitted_estimator, &x_test.view())?;
        let test_score = self.score(&predictions.view(), &y_test.view())?;
        let score_time = score_start.elapsed();

        // Training score (optional)
        let train_score = if self.scoring.use_for_selection {
            let train_predictions = self.predict_fitted(&fitted_estimator, &x_train.view())?;
            Some(self.score(&train_predictions.view(), &y_train.view())?)
        } else {
            None
        };

        // Additional metrics
        let additional_metrics =
            self.compute_additional_metrics(&predictions.view(), &y_test.view())?;

        Ok(FoldResult {
            fold_index,
            test_score,
            train_score,
            fit_time,
            score_time,
            train_indices: train_indices.to_vec(),
            test_indices: test_indices.to_vec(),
            additional_metrics,
        })
    }

    /// Predict using fitted pipeline
    fn predict_pipeline<F>(
        &self,
        fitted_pipeline: &F,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<Array1<f64>>
    where
        F: ?Sized,
        for<'a> F: Predict<ArrayView2<'a, Float>, Array1<f64>>,
    {
        fitted_pipeline.predict(x)
    }

    /// Predict using fitted estimator
    fn predict_fitted<F>(
        &self,
        fitted_estimator: &F,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<Array1<f64>> {
        // This would need to be implemented based on the actual fitted estimator interface
        // For now, we'll use a placeholder
        Ok(Array1::zeros(x.nrows()))
    }

    /// Compute score using the configured metric
    fn score(
        &self,
        y_pred: &ArrayView1<'_, f64>,
        y_true: &ArrayView1<'_, Float>,
    ) -> SklResult<f64> {
        match self.scoring.primary_metric {
            ScoringMetric::MeanSquaredError => {
                let mse = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                    .sum::<f64>()
                    / y_pred.len() as f64;
                Ok(-mse) // Negative because cross-validation maximizes score
            }
            ScoringMetric::RootMeanSquaredError => {
                let mse = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                    .sum::<f64>()
                    / y_pred.len() as f64;
                Ok(-mse.sqrt()) // Negative because cross-validation maximizes score
            }
            ScoringMetric::MeanAbsoluteError => {
                let mae = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).abs())
                    .sum::<f64>()
                    / y_pred.len() as f64;
                Ok(-mae) // Negative because cross-validation maximizes score
            }
            ScoringMetric::R2Score => {
                let y_mean = y_true.iter().copied().sum::<f64>() / y_true.len() as f64;
                let ss_res = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (true_val - pred).powi(2))
                    .sum::<f64>();
                let ss_tot = y_true
                    .iter()
                    .map(|&true_val| (true_val - y_mean).powi(2))
                    .sum::<f64>();
                Ok(1.0 - ss_res / ss_tot)
            }
            ScoringMetric::Accuracy => {
                let correct = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| {
                        if (pred.round() - true_val).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>();
                Ok(correct / y_pred.len() as f64)
            }
            _ => {
                // For other metrics, use custom scorer or default to MSE
                if let Some(custom_scorer) = self.scoring.custom_scorer {
                    let y_true_f64: Array1<f64> = y_true.mapv(|x| x);
                    Ok(custom_scorer(&y_pred.to_owned(), &y_true_f64))
                } else {
                    let mse = y_pred
                        .iter()
                        .zip(y_true.iter())
                        .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                        .sum::<f64>()
                        / y_pred.len() as f64;
                    Ok(-mse)
                }
            }
        }
    }

    /// Compute additional metrics
    fn compute_additional_metrics(
        &self,
        y_pred: &ArrayView1<'_, f64>,
        y_true: &ArrayView1<'_, Float>,
    ) -> SklResult<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        for metric in &self.scoring.additional_metrics {
            let score = match metric {
                ScoringMetric::MeanSquaredError => {
                    y_pred
                        .iter()
                        .zip(y_true.iter())
                        .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                        .sum::<f64>()
                        / y_pred.len() as f64
                }
                ScoringMetric::MeanAbsoluteError => {
                    y_pred
                        .iter()
                        .zip(y_true.iter())
                        .map(|(&pred, &true_val)| (pred - true_val).abs())
                        .sum::<f64>()
                        / y_pred.len() as f64
                }
                ScoringMetric::R2Score => {
                    let y_mean = y_true.iter().copied().sum::<f64>() / y_true.len() as f64;
                    let ss_res = y_pred
                        .iter()
                        .zip(y_true.iter())
                        .map(|(&pred, &true_val)| (true_val - pred).powi(2))
                        .sum::<f64>();
                    let ss_tot = y_true
                        .iter()
                        .map(|&true_val| (true_val - y_mean).powi(2))
                        .sum::<f64>();
                    1.0 - ss_res / ss_tot
                }
                _ => 0.0, // Placeholder for other metrics
            };

            let metric_name = format!("{metric:?}");
            metrics.insert(metric_name, score);
        }

        Ok(metrics)
    }

    /// Perform grid search cross-validation
    fn grid_search_cv<E>(
        &self,
        inner_cv: &ComposedModelCrossValidator,
        estimator: E,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        param_grid: &HashMap<String, Vec<f64>>,
    ) -> SklResult<HashMap<String, f64>>
    where
        E: Clone,
    {
        // Simplified implementation - would need proper parameter setting
        let mut best_params = HashMap::new();
        let best_score = f64::NEG_INFINITY;

        // For now, just return the first parameter combination
        for (param_name, param_values) in param_grid {
            if let Some(&first_value) = param_values.first() {
                best_params.insert(param_name.clone(), first_value);
            }
        }

        Ok(best_params)
    }
}

/// Nested cross-validation results
#[derive(Debug, Clone)]
pub struct NestedCVResults {
    /// Results for each outer fold
    pub outer_fold_results: Vec<OuterFoldResult>,
    /// Overall performance estimate
    pub overall_score: f64,
    /// Standard deviation of outer fold scores
    pub score_std: f64,
    /// Best parameters found across folds
    pub parameter_distribution: HashMap<String, Vec<f64>>,
}

/// Results for a single outer fold in nested CV
#[derive(Debug, Clone)]
pub struct OuterFoldResult {
    /// Outer fold index
    pub fold_index: usize,
    /// Best parameters found by inner CV
    pub best_params: HashMap<String, f64>,
    /// Test score on outer fold
    pub test_score: f64,
    /// Inner CV score for best parameters
    pub inner_cv_score: f64,
}

impl CrossValidationResults {
    /// Create new empty results
    fn new() -> Self {
        Self {
            test_scores: Vec::new(),
            train_scores: None,
            fit_times: Vec::new(),
            score_times: Vec::new(),
            summary: CVSummary::empty(),
            fold_results: Vec::new(),
            additional_metrics: HashMap::new(),
        }
    }

    /// Add a fold result
    fn add_fold_result(&mut self, fold_result: FoldResult) {
        self.test_scores.push(fold_result.test_score);
        self.fit_times.push(fold_result.fit_time);
        self.score_times.push(fold_result.score_time);

        if let Some(train_score) = fold_result.train_score {
            if self.train_scores.is_none() {
                self.train_scores = Some(Vec::new());
            }
            self.train_scores.as_mut().unwrap().push(train_score);
        }

        // Collect additional metrics
        for (metric_name, score) in &fold_result.additional_metrics {
            self.additional_metrics
                .entry(metric_name.clone())
                .or_default()
                .push(*score);
        }

        self.fold_results.push(fold_result);
    }

    /// Compute summary statistics
    fn compute_summary(&mut self) {
        let n_folds = self.test_scores.len() as f64;
        let mean_test_score = self.test_scores.iter().sum::<f64>() / n_folds;
        let variance = self
            .test_scores
            .iter()
            .map(|&score| (score - mean_test_score).powi(2))
            .sum::<f64>()
            / n_folds;
        let std_test_score = variance.sqrt();

        let min_test_score = self
            .test_scores
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_test_score = self
            .test_scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // 95% confidence interval
        let margin_error = 1.96 * std_test_score / (n_folds.sqrt());
        let confidence_interval = (
            mean_test_score - margin_error,
            mean_test_score + margin_error,
        );

        let (mean_train_score, std_train_score) = if let Some(ref train_scores) = self.train_scores
        {
            let mean = train_scores.iter().sum::<f64>() / n_folds;
            let variance = train_scores
                .iter()
                .map(|&score| (score - mean).powi(2))
                .sum::<f64>()
                / n_folds;
            (Some(mean), Some(variance.sqrt()))
        } else {
            (None, None)
        };

        self.summary = CVSummary {
            mean_test_score,
            std_test_score,
            min_test_score,
            max_test_score,
            confidence_interval,
            mean_train_score,
            std_train_score,
        };
    }
}

impl CVSummary {
    /// Create empty summary
    fn empty() -> Self {
        Self {
            mean_test_score: 0.0,
            std_test_score: 0.0,
            min_test_score: 0.0,
            max_test_score: 0.0,
            confidence_interval: (0.0, 0.0),
            mean_train_score: None,
            std_train_score: None,
        }
    }
}

impl NestedCVResults {
    /// Create new empty nested CV results
    fn new() -> Self {
        Self {
            outer_fold_results: Vec::new(),
            overall_score: 0.0,
            score_std: 0.0,
            parameter_distribution: HashMap::new(),
        }
    }

    /// Add outer fold result
    fn add_outer_fold_result(&mut self, result: OuterFoldResult) {
        // Collect parameter distribution
        for (param_name, &param_value) in &result.best_params {
            self.parameter_distribution
                .entry(param_name.clone())
                .or_default()
                .push(param_value);
        }

        self.outer_fold_results.push(result);
        self.compute_overall_score();
    }

    /// Compute overall score from outer fold results
    fn compute_overall_score(&mut self) {
        let scores: Vec<f64> = self
            .outer_fold_results
            .iter()
            .map(|r| r.test_score)
            .collect();
        let n_folds = scores.len() as f64;

        self.overall_score = scores.iter().sum::<f64>() / n_folds;
        let variance = scores
            .iter()
            .map(|&score| (score - self.overall_score).powi(2))
            .sum::<f64>()
            / n_folds;
        self.score_std = variance.sqrt();
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            strategy: CVStrategy::KFold,
            n_folds: 5,
            test_size: 0.2,
            n_repeats: 1,
            shuffle: true,
            stratified: false,
            time_series_config: None,
        }
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            primary_metric: ScoringMetric::MeanSquaredError,
            additional_metrics: Vec::new(),
            use_for_selection: false,
            custom_scorer: None,
        }
    }
}

/// Generate combinations of indices
fn combinations(indices: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > indices.len() {
        return vec![];
    }
    if k == indices.len() {
        return vec![indices.to_vec()];
    }

    let mut result = Vec::new();
    let first = indices[0];
    let rest = &indices[1..];

    // Include first element
    for mut combo in combinations(rest, k - 1) {
        combo.insert(0, first);
        result.push(combo);
    }

    // Exclude first element
    result.extend(combinations(rest, k));

    result
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cross_validation_config() {
        let config = CrossValidationConfig {
            strategy: CVStrategy::KFold,
            n_folds: 5,
            test_size: 0.2,
            n_repeats: 1,
            shuffle: true,
            stratified: false,
            time_series_config: None,
        };

        assert_eq!(config.n_folds, 5);
        assert!(config.shuffle);
        assert!(!config.stratified);
    }

    #[test]
    fn test_k_fold_split() {
        let config = CrossValidationConfig::default();
        let cv = ComposedModelCrossValidator::new(config);

        let splits = cv.k_fold_split(10, &mut StdRng::seed_from_u64(42)).unwrap();

        assert_eq!(splits.len(), 5);
        for (train_indices, test_indices) in &splits {
            assert!(!train_indices.is_empty());
            assert!(!test_indices.is_empty());
            assert_eq!(train_indices.len() + test_indices.len(), 10);
        }
    }

    #[test]
    fn test_stratified_k_fold_split() {
        let config = CrossValidationConfig::default();
        let cv = ComposedModelCrossValidator::new(config);

        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let splits = cv
            .stratified_k_fold_split(8, &y.view(), &mut StdRng::seed_from_u64(42))
            .unwrap();

        assert_eq!(splits.len(), 5);
        // Each fold should have roughly equal representation of classes
    }

    #[test]
    fn test_leave_one_out_split() {
        let config = CrossValidationConfig::default();
        let cv = ComposedModelCrossValidator::new(config);

        let splits = cv.leave_one_out_split(5).unwrap();

        assert_eq!(splits.len(), 5);
        for (i, (train_indices, test_indices)) in splits.iter().enumerate() {
            assert_eq!(test_indices.len(), 1);
            assert_eq!(test_indices[0], i);
            assert_eq!(train_indices.len(), 4);
        }
    }

    #[test]
    fn test_time_series_split() {
        let mut config = CrossValidationConfig::default();
        config.strategy = CVStrategy::TimeSeriesSplit;
        config.time_series_config = Some(TimeSeriesConfig {
            max_train_size: None,
            gap: 0,
            test_size: 2,
            expanding_window: true,
        });

        let cv = ComposedModelCrossValidator::new(config);
        let splits = cv.time_series_split(10).unwrap();

        assert!(!splits.is_empty());
        for (train_indices, test_indices) in &splits {
            assert_eq!(test_indices.len(), 2);
            assert!(!train_indices.is_empty());
            // Training data should come before test data in time series
            assert!(train_indices.iter().max() < test_indices.iter().min());
        }
    }

    #[test]
    fn test_bootstrap_split() {
        let config = CrossValidationConfig::default();
        let cv = ComposedModelCrossValidator::new(config);

        let splits = cv
            .bootstrap_split(10, 5, &mut StdRng::seed_from_u64(42))
            .unwrap();

        assert_eq!(splits.len(), 5);
        for (train_indices, test_indices) in &splits {
            assert_eq!(train_indices.len(), 10); // Bootstrap samples same size as original
            assert!(!test_indices.is_empty()); // Out-of-bag samples exist
        }
    }

    #[test]
    fn test_scoring_metrics() {
        let config = CrossValidationConfig::default();
        let mut scoring = ScoringConfig::default();
        scoring.primary_metric = ScoringMetric::R2Score;
        scoring.additional_metrics = vec![ScoringMetric::MeanAbsoluteError];

        let cv = ComposedModelCrossValidator::new(config).with_scoring(scoring);

        let y_pred = array![1.0, 2.0, 3.0, 4.0];
        let y_true = array![1.1, 1.9, 3.1, 3.9];

        let score = cv.score(&y_pred.view(), &y_true.view()).unwrap();
        assert!(score > 0.0); // R2 should be positive for good predictions

        let additional_metrics = cv
            .compute_additional_metrics(&y_pred.view(), &y_true.view())
            .unwrap();
        assert!(additional_metrics.contains_key("MeanAbsoluteError"));
    }

    #[test]
    fn test_cross_validation_results() {
        let mut results = CrossValidationResults::new();

        let fold_result = FoldResult {
            fold_index: 0,
            test_score: 0.8,
            train_score: Some(0.85),
            fit_time: Duration::from_millis(100),
            score_time: Duration::from_millis(10),
            train_indices: vec![0, 1, 2],
            test_indices: vec![3, 4],
            additional_metrics: HashMap::new(),
        };

        results.add_fold_result(fold_result);
        results.compute_summary();

        assert_eq!(results.summary.mean_test_score, 0.8);
        assert_eq!(results.fold_results.len(), 1);
    }

    #[test]
    fn test_combinations() {
        let indices = vec![0, 1, 2, 3];
        let combos = combinations(&indices, 2);

        assert_eq!(combos.len(), 6); // C(4,2) = 6
        assert!(combos.contains(&vec![0, 1]));
        assert!(combos.contains(&vec![0, 2]));
        assert!(combos.contains(&vec![2, 3]));
    }
}
