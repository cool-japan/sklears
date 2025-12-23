//! Multi-task Elastic Net with built-in cross-validation
//!
//! This module implements multi-task elastic net regression with automatic
//! hyperparameter selection using cross-validation.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
// use sklears_model_selection::{CrossValidator, KFold}; // Temporarily disabled
use crate::lasso_cv::KFold;

use crate::multi_task_elastic_net::MultiTaskElasticNet;

/// Configuration for MultiTaskElasticNetCV
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNetCVConfig {
    /// The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
    /// l1_ratio = 0 corresponds to L2 penalty (Ridge)
    /// l1_ratio = 1 corresponds to L1 penalty (Lasso)
    /// If a list, will test all values in cross-validation
    pub l1_ratios: Option<Vec<Float>>,
    /// List of alpha values to try
    /// If None, uses an automatic range
    pub alphas: Option<Vec<Float>>,
    /// Number of alphas along the regularization path
    pub n_alphas: usize,
    /// Number of folds for cross-validation
    pub cv: usize,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for optimization
    pub tol: Float,
    /// Number of jobs to run in parallel (-1 for all cores)
    pub n_jobs: Option<i32>,
    /// Verbosity level
    pub verbose: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for MultiTaskElasticNetCVConfig {
    fn default() -> Self {
        Self {
            l1_ratios: None,
            alphas: None,
            n_alphas: 100,
            cv: 5,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            n_jobs: None,
            verbose: 0,
            random_state: None,
        }
    }
}

/// Multi-task Elastic Net with built-in cross-validation
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNetCV<State = Untrained> {
    config: MultiTaskElasticNetCVConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    alpha_: Option<Float>,
    l1_ratio_: Option<Float>,
    cv_alphas_: Option<Vec<Float>>,
    cv_l1_ratios_: Option<Vec<Float>>,
    cv_scores_: Option<HashMap<String, Vec<Float>>>,
    best_score_: Option<Float>,
    n_features_: Option<usize>,
    n_tasks_: Option<usize>,
}

impl MultiTaskElasticNetCV {
    /// Create a new MultiTaskElasticNetCV with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for MultiTaskElasticNetCV
    pub fn builder() -> MultiTaskElasticNetCVBuilder {
        MultiTaskElasticNetCVBuilder::new()
    }
}

impl Default for MultiTaskElasticNetCV {
    fn default() -> Self {
        Self {
            config: MultiTaskElasticNetCVConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            l1_ratio_: None,
            cv_alphas_: None,
            cv_l1_ratios_: None,
            cv_scores_: None,
            best_score_: None,
            n_features_: None,
            n_tasks_: None,
        }
    }
}

/// Builder for MultiTaskElasticNetCV
pub struct MultiTaskElasticNetCVBuilder {
    config: MultiTaskElasticNetCVConfig,
}

impl MultiTaskElasticNetCVBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MultiTaskElasticNetCVConfig::default(),
        }
    }

    /// Set the l1_ratio values to try
    pub fn l1_ratios(mut self, l1_ratios: Vec<Float>) -> Self {
        self.config.l1_ratios = Some(l1_ratios);
        self
    }

    /// Set the list of alpha values to try
    pub fn alphas(mut self, alphas: Vec<Float>) -> Self {
        self.config.alphas = Some(alphas);
        self
    }

    /// Set the number of alphas
    pub fn n_alphas(mut self, n_alphas: usize) -> Self {
        self.config.n_alphas = n_alphas;
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv(mut self, cv: usize) -> Self {
        self.config.cv = cv;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.config.n_jobs = Some(n_jobs);
        self
    }

    /// Set the verbosity level
    pub fn verbose(mut self, verbose: usize) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Build the MultiTaskElasticNetCV
    pub fn build(self) -> MultiTaskElasticNetCV {
        MultiTaskElasticNetCV {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            l1_ratio_: None,
            cv_alphas_: None,
            cv_l1_ratios_: None,
            cv_scores_: None,
            best_score_: None,
            n_features_: None,
            n_tasks_: None,
        }
    }
}

impl Default for MultiTaskElasticNetCVBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTaskElasticNetCV {
    type Config = MultiTaskElasticNetCVConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for MultiTaskElasticNetCV<Trained> {
    type Config = MultiTaskElasticNetCVConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array2<Float>> for MultiTaskElasticNetCV {
    type Fitted = MultiTaskElasticNetCV<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        // Validate input
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let _n_samples = x.nrows();
        let n_features = x.ncols();
        let n_tasks = y.ncols();

        // Get l1_ratio values
        let l1_ratios = self
            .config
            .l1_ratios
            .clone()
            .unwrap_or_else(|| vec![0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]);

        // Get alpha values
        let alphas = self.config.alphas.clone().unwrap_or_else(|| {
            // Generate alphas on log scale
            let mut values = Vec::new();
            let alpha_max: Float = 1.0; // This could be computed based on data
            let alpha_min = alpha_max * 0.0001;

            for i in 0..self.config.n_alphas {
                let ratio = i as Float / (self.config.n_alphas - 1) as Float;
                let log_alpha = (1.0 - ratio) * alpha_max.ln() + ratio * alpha_min.ln();
                values.push(log_alpha.exp());
            }
            values.reverse(); // Start with largest alpha
            values
        });

        // Create cross-validation splitter
        let cv = KFold::new(self.config.cv)
            .shuffle(true)
            .random_state(Some(self.config.random_state.unwrap_or(42)));

        // Initialize scores storage
        let mut cv_scores: HashMap<String, Vec<Float>> = HashMap::new();
        let mut best_score = Float::NEG_INFINITY;
        let mut best_alpha = alphas[0];
        let mut best_l1_ratio = l1_ratios[0];

        // Try each combination of l1_ratio and alpha
        for &l1_ratio in &l1_ratios {
            for &alpha in &alphas {
                let key = format!("{:.6}_{:.6}", l1_ratio, alpha);

                // Create model with current parameters
                let model = MultiTaskElasticNet::new()
                    .alpha(alpha)
                    .l1_ratio(l1_ratio)
                    .fit_intercept(self.config.fit_intercept)
                    .max_iter(self.config.max_iter)
                    .tol(self.config.tol);

                // Perform manual cross-validation for multi-task
                let splits = cv.split(x.nrows(), None);
                let mut fold_scores = Vec::new();

                for (train_idx, test_idx) in splits {
                    // Extract train and test sets
                    let x_train = x.select(Axis(0), &train_idx);
                    let y_train = y.select(Axis(0), &train_idx);
                    let x_test = x.select(Axis(0), &test_idx);
                    let y_test = y.select(Axis(0), &test_idx);

                    // Fit model on train set
                    let fitted = model.clone().fit(&x_train, &y_train)?;

                    // Predict on test set
                    let y_pred = fitted.predict(&x_test)?;

                    // Calculate R² score for each task and average
                    let mut task_scores = Vec::new();
                    for task in 0..y_test.ncols() {
                        let y_true_task = y_test.column(task);
                        let y_pred_task = y_pred.column(task);

                        // Calculate R² score
                        let ss_tot = y_true_task
                            .mapv(|v| (v - y_true_task.mean().unwrap()).powi(2))
                            .sum();
                        let ss_res = (&y_true_task - &y_pred_task).mapv(|v| v.powi(2)).sum();
                        let r2 = if ss_tot == 0.0 {
                            0.0
                        } else {
                            1.0 - ss_res / ss_tot
                        };
                        task_scores.push(r2);
                    }

                    // Average R² across tasks
                    let mean_task_score =
                        task_scores.iter().sum::<Float>() / task_scores.len() as Float;
                    fold_scores.push(mean_task_score);
                }

                let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;

                if self.config.verbose > 0 {
                    let variance = fold_scores
                        .iter()
                        .map(|&s| (s - mean_score).powi(2))
                        .sum::<Float>()
                        / fold_scores.len() as Float;
                    let std_score = variance.sqrt();
                    println!(
                        "l1_ratio={:.3}, alpha={:.6e}, mean score={:.4}, std={:.4}",
                        l1_ratio, alpha, mean_score, std_score
                    );
                }

                cv_scores.insert(key, fold_scores);

                if mean_score > best_score {
                    best_score = mean_score;
                    best_alpha = alpha;
                    best_l1_ratio = l1_ratio;
                }
            }
        }

        // Refit on entire dataset with best parameters
        let best_model = MultiTaskElasticNet::new()
            .alpha(best_alpha)
            .l1_ratio(best_l1_ratio)
            .fit_intercept(self.config.fit_intercept)
            .max_iter(self.config.max_iter)
            .tol(self.config.tol);
        let trained_model = best_model.fit(x, y)?;

        // Extract coefficients and intercept
        let coef_ = trained_model.coef().clone();
        let intercept_ = trained_model.intercept().clone();

        Ok(MultiTaskElasticNetCV {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef_),
            intercept_: Some(intercept_),
            alpha_: Some(best_alpha),
            l1_ratio_: Some(best_l1_ratio),
            cv_alphas_: Some(alphas),
            cv_l1_ratios_: Some(l1_ratios),
            cv_scores_: Some(cv_scores),
            best_score_: Some(best_score),
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
        })
    }
}

impl Predict<Array2<Float>, Array2<Float>> for MultiTaskElasticNetCV<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features != self.n_features_.unwrap() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features_.unwrap(),
                n_features
            )));
        }

        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();

        // Compute predictions: Y = X @ W.T + intercept
        let mut predictions = x.dot(&coef.t());

        // Add intercept
        for i in 0..n_samples {
            let mut row = predictions.row_mut(i);
            row += intercept;
        }

        Ok(predictions)
    }
}

impl MultiTaskElasticNetCV<Trained> {
    /// Get the best alpha value found
    pub fn best_alpha(&self) -> Float {
        self.alpha_.unwrap()
    }

    /// Get the best l1_ratio value found
    pub fn best_l1_ratio(&self) -> Float {
        self.l1_ratio_.unwrap()
    }

    /// Get the coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> &Array1<Float> {
        self.intercept_.as_ref().unwrap()
    }

    /// Get all cross-validation scores
    pub fn cv_scores(&self) -> &HashMap<String, Vec<Float>> {
        self.cv_scores_.as_ref().unwrap()
    }

    /// Get the best cross-validation score
    pub fn best_score(&self) -> Float {
        self.best_score_.unwrap()
    }

    /// Get the alpha values that were tried
    pub fn alphas(&self) -> &Vec<Float> {
        self.cv_alphas_.as_ref().unwrap()
    }

    /// Get the l1_ratio values that were tried
    pub fn l1_ratios(&self) -> &Vec<Float> {
        self.cv_l1_ratios_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_task_elastic_net_cv_basic() {
        // Create multi-task regression data
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];

        // Two tasks
        let y = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];

        let model = MultiTaskElasticNetCV::builder()
            .cv(3)
            .alphas(vec![0.1, 1.0, 10.0])
            .l1_ratios(vec![0.5, 0.9])
            .build();

        let trained = model.fit(&x, &y).unwrap();

        // Check that we found best parameters
        assert!(trained.alphas().contains(&trained.best_alpha()));
        assert!(trained.l1_ratios().contains(&trained.best_l1_ratio()));

        // Check predictions shape
        let predictions = trained.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[6, 2]);

        // Check coefficients shape
        assert_eq!(trained.coef().shape(), &[2, 2]); // 2 features, 2 tasks
        assert_eq!(trained.intercept().len(), 2); // 2 tasks
    }

    #[test]
    fn test_multi_task_elastic_net_cv_default_alphas() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0],
            [8.0, 9.0, 10.0],
        ];

        // Three tasks
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0],
            [8.0, 9.0, 10.0],
        ];

        // Don't specify alphas or l1_ratios, should use defaults
        let model = MultiTaskElasticNetCV::builder().cv(3).n_alphas(10).build();

        let trained = model.fit(&x, &y).unwrap();

        // Should have generated 10 alphas and 7 default l1_ratios
        assert_eq!(trained.alphas().len(), 10);
        assert_eq!(trained.l1_ratios().len(), 7);

        // Check coefficients shape
        assert_eq!(trained.coef().shape(), &[3, 3]); // 3 features, 3 tasks
    }

    #[test]
    fn test_multi_task_elastic_net_cv_single_l1_ratio() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5],];

        // Test with single l1_ratio (pure Lasso)
        let model = MultiTaskElasticNetCV::builder()
            .cv(3)
            .l1_ratios(vec![1.0])
            .alphas(vec![0.01, 0.1, 1.0])
            .build();

        let trained = model.fit(&x, &y).unwrap();

        // Should have l1_ratio = 1.0
        assert_eq!(trained.best_l1_ratio(), 1.0);

        // Predictions should work
        let predictions = trained.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[5, 2]);
    }
}
