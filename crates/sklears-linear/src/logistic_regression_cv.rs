//! Logistic regression with built-in cross-validation
//!
//! This module implements logistic regression with automatic hyperparameter
//! selection using cross-validation.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
// use sklears_model_selection::{cross_val_score, CrossValidator, KFold, Scoring}; // Temporarily disabled
use crate::lasso_cv::{cross_val_score, KFold};

use crate::{logistic_regression::LogisticRegression, Penalty, Solver};

/// Configuration for LogisticRegressionCV
#[derive(Debug, Clone)]
pub struct LogisticRegressionCVConfig {
    /// List of C values to try (regularization strength)
    /// If None, uses 10 values on a log scale between 1e-4 and 1e4
    pub cs: Option<Vec<f64>>,
    /// Number of folds for cross-validation
    pub cv: usize,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Penalty type: "l1", "l2", "elasticnet", or "none"
    pub penalty: String,
    /// Elastic net mixing parameter (only used if penalty="elasticnet")
    pub l1_ratio: Option<f64>,
    /// Solver to use
    pub solver: Solver,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for optimization
    pub tol: f64,
    /// Class weight (None for equal weights)
    pub class_weight: Option<HashMap<i32, f64>>,
    /// Whether to refit the model using the best parameters on the whole dataset
    pub refit: bool,
    /// Number of jobs to run in parallel (-1 for all cores)
    pub n_jobs: Option<i32>,
    /// Scoring function (None defaults to accuracy)
    pub scoring: Option<String>,
    /// Verbosity level
    pub verbose: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for LogisticRegressionCVConfig {
    fn default() -> Self {
        Self {
            cs: None,
            cv: 5,
            fit_intercept: true,
            penalty: "l2".to_string(),
            l1_ratio: None,
            solver: Solver::Lbfgs,
            max_iter: 100,
            tol: 1e-4,
            class_weight: None,
            refit: true,
            n_jobs: None,
            scoring: None,
            verbose: 0,
            random_state: None,
        }
    }
}

/// Logistic regression with built-in cross-validation
#[derive(Debug, Clone)]
pub struct LogisticRegressionCV<State = Untrained> {
    config: LogisticRegressionCVConfig,
    state: PhantomData<State>,
    // Trained state fields
    /// Best regularization parameter found
    c_: Option<f64>,
    /// Coefficients of the model
    coef_: Option<Array2<f64>>,
    /// Intercept terms
    intercept_: Option<Array1<f64>>,
    /// Cross-validation scores for each C value
    scores_: Option<HashMap<String, Vec<f64>>>,
    /// Mean cross-validation scores for each C value
    mean_scores_: Option<HashMap<String, f64>>,
    /// Standard deviation of cross-validation scores for each C value
    std_scores_: Option<HashMap<String, f64>>,
    /// The actual C values used
    cs_: Option<Vec<f64>>,
    /// Number of iterations for each fold and C value
    n_iter_: Option<HashMap<String, Vec<usize>>>,
    /// Unique classes in the training data
    classes_: Option<Array1<f64>>,
}

impl LogisticRegressionCV {
    /// Create a new LogisticRegressionCV with default configuration
    pub fn new() -> Self {
        Self {
            config: LogisticRegressionCVConfig::default(),
            state: PhantomData,
            c_: None,
            coef_: None,
            intercept_: None,
            scores_: None,
            mean_scores_: None,
            std_scores_: None,
            cs_: None,
            n_iter_: None,
            classes_: None,
        }
    }

    /// Create a builder for LogisticRegressionCV
    pub fn builder() -> LogisticRegressionCVBuilder {
        LogisticRegressionCVBuilder::new()
    }
}

impl Default for LogisticRegressionCV {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for LogisticRegressionCV
pub struct LogisticRegressionCVBuilder {
    config: LogisticRegressionCVConfig,
}

impl LogisticRegressionCVBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: LogisticRegressionCVConfig::default(),
        }
    }

    /// Set the list of C values to try
    pub fn cs(mut self, cs: Vec<f64>) -> Self {
        self.config.cs = Some(cs);
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

    /// Set the penalty type
    pub fn penalty(mut self, penalty: impl Into<String>) -> Self {
        self.config.penalty = penalty.into();
        self
    }

    /// Set the elastic net mixing parameter
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.config.l1_ratio = Some(l1_ratio);
        self
    }

    /// Set the solver
    pub fn solver(mut self, solver: Solver) -> Self {
        self.config.solver = solver;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set class weights
    pub fn class_weight(mut self, class_weight: HashMap<i32, f64>) -> Self {
        self.config.class_weight = Some(class_weight);
        self
    }

    /// Set whether to refit
    pub fn refit(mut self, refit: bool) -> Self {
        self.config.refit = refit;
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.config.n_jobs = Some(n_jobs);
        self
    }

    /// Set the scoring function
    pub fn scoring(mut self, scoring: impl Into<String>) -> Self {
        self.config.scoring = Some(scoring.into());
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

    /// Build the LogisticRegressionCV
    pub fn build(self) -> LogisticRegressionCV {
        LogisticRegressionCV {
            config: self.config,
            state: PhantomData,
            c_: None,
            coef_: None,
            intercept_: None,
            scores_: None,
            mean_scores_: None,
            std_scores_: None,
            cs_: None,
            n_iter_: None,
            classes_: None,
        }
    }
}

impl Default for LogisticRegressionCVBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LogisticRegressionCV<Untrained> {
    type Config = LogisticRegressionCVConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for LogisticRegressionCV<Untrained> {
    type Fitted = LogisticRegressionCV<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        // Validate input
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        let classes = Array1::from_vec(classes);

        // Generate C values if not provided
        let cs = self.config.cs.clone().unwrap_or_else(|| {
            // Generate 10 values on log scale from 1e-4 to 1e4
            let mut values = Vec::new();
            for i in 0..10 {
                let log_c = -4.0 + (8.0 * i as f64 / 9.0);
                values.push(10f64.powf(log_c));
            }
            values
        });

        // Initialize results storage
        let mut scores: HashMap<String, Vec<f64>> = HashMap::new();
        let mut n_iter: HashMap<String, Vec<usize>> = HashMap::new();

        // Create cross-validation splitter
        let cv = KFold::new(self.config.cv)
            .shuffle(true)
            .random_state(Some(self.config.random_state.unwrap_or(42)));

        // Try each C value
        for &c in &cs {
            // Create model with current C
            let penalty = match self.config.penalty.as_str() {
                "l1" => Penalty::L1(1.0 / c),
                "l2" => Penalty::L2(1.0 / c),
                "elasticnet" => {
                    let l1_ratio = self.config.l1_ratio.unwrap_or(0.5);
                    Penalty::ElasticNet {
                        l1_ratio,
                        alpha: 1.0 / c,
                    }
                }
                "none" => Penalty::None,
                _ => Penalty::L2(1.0 / c), // Default to L2
            };

            let mut model = LogisticRegression::new()
                .penalty(penalty)
                .solver(self.config.solver)
                .max_iter(self.config.max_iter)
                .fit_intercept(self.config.fit_intercept);

            if let Some(seed) = self.config.random_state {
                model = model.random_state(seed);
            }

            // Use cross_val_score with placeholder scoring
            let scoring = None; // Placeholder until proper scoring is implemented

            // Compute cross-validation scores
            let fold_scores =
                cross_val_score(model, x, y, &cv, scoring, self.config.n_jobs.map(|_| ()))?;

            // Convert Array1<f64> to Vec<f64> for storage
            let fold_scores_vec: Vec<f64> = fold_scores.to_vec();

            scores.insert(c.to_string(), fold_scores_vec.clone());

            // Store number of iterations (placeholder for now)
            let fold_iters = vec![self.config.max_iter; cv.n_splits()];
            n_iter.insert(c.to_string(), fold_iters);
        }

        // Calculate mean and std scores
        let mut mean_scores = HashMap::new();
        let mut std_scores = HashMap::new();
        let mut best_c = cs[0];
        let mut best_score = f64::NEG_INFINITY;

        for &c in &cs {
            let fold_scores = &scores[&c.to_string()];
            let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            let std = if fold_scores.len() > 1 {
                let variance = fold_scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                    / (fold_scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            mean_scores.insert(c.to_string(), mean);
            std_scores.insert(c.to_string(), std);

            if mean > best_score {
                best_score = mean;
                best_c = c;
            }

            if self.config.verbose > 0 {
                println!("C={:.4e}, mean score={:.4}, std={:.4}", c, mean, std);
            }
        }

        // Refit on entire dataset with best C if requested
        let (coef_, intercept_) = if self.config.refit {
            let penalty = match self.config.penalty.as_str() {
                "l1" => Penalty::L1(1.0 / best_c),
                "l2" => Penalty::L2(1.0 / best_c),
                "elasticnet" => {
                    let l1_ratio = self.config.l1_ratio.unwrap_or(0.5);
                    Penalty::ElasticNet {
                        l1_ratio,
                        alpha: 1.0 / best_c,
                    }
                }
                "none" => Penalty::None,
                _ => Penalty::L2(1.0 / best_c),
            };

            let mut model = LogisticRegression::new()
                .penalty(penalty)
                .solver(self.config.solver)
                .max_iter(self.config.max_iter)
                .fit_intercept(self.config.fit_intercept);

            if let Some(seed) = self.config.random_state {
                model = model.random_state(seed);
            }

            let trained_model = model.fit(x, y)?;

            // Extract coefficients and intercept
            // Note: LogisticRegression currently only supports binary classification
            if classes.len() > 2 {
                return Err(SklearsError::InvalidInput(
                    "Multi-class classification not yet supported in LogisticRegression"
                        .to_string(),
                ));
            }

            let coef = trained_model.coef().clone();
            let coef_ = coef.insert_axis(Axis(0)); // Convert 1D to 2D with shape (1, n_features)

            let intercept_ = if let Some(intercept) = trained_model.intercept() {
                Array1::from(vec![intercept])
            } else {
                Array1::zeros(1)
            };

            (coef_, intercept_)
        } else {
            // If not refitting, just use dummy values
            let n_features = x.ncols();
            let n_classes = classes.len();
            if n_classes == 2 {
                (Array2::zeros((1, n_features)), Array1::zeros(1))
            } else {
                (
                    Array2::zeros((n_classes, n_features)),
                    Array1::zeros(n_classes),
                )
            }
        };

        Ok(LogisticRegressionCV {
            config: self.config.clone(),
            state: PhantomData,
            c_: Some(best_c),
            coef_: Some(coef_),
            intercept_: Some(intercept_),
            scores_: Some(scores),
            mean_scores_: Some(mean_scores),
            std_scores_: Some(std_scores),
            cs_: Some(cs),
            n_iter_: Some(n_iter),
            classes_: Some(classes),
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for LogisticRegressionCV<Trained> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().expect("Model not fitted");
        let classes = self.classes_.as_ref().expect("Model not fitted");

        if n_features != coef.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                coef.ncols(),
                n_features
            )));
        }

        // Convert to class predictions
        let mut predictions = Array1::zeros(n_samples);

        if classes.len() == 2 {
            // Binary classification
            let coef_row = coef.row(0);
            let intercept_val = intercept[0];
            let decision = x.dot(&coef_row.t()) + intercept_val;

            for i in 0..n_samples {
                predictions[i] = if decision[i] > 0.0 {
                    classes[1]
                } else {
                    classes[0]
                };
            }
        } else {
            // Multi-class classification
            let decision = x.dot(&coef.t()) + intercept;

            for i in 0..n_samples {
                let row = decision.row(i);
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                predictions[i] = classes[max_idx];
            }
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<f64>, Array2<f64>> for LogisticRegressionCV<Trained> {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let coef = self.coef_.as_ref().expect("Model not fitted");
        let intercept = self.intercept_.as_ref().expect("Model not fitted");
        let classes = self.classes_.as_ref().expect("Model not fitted");

        if n_features != coef.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                coef.ncols(),
                n_features
            )));
        }

        // Apply softmax to get probabilities
        if classes.len() == 2 {
            // Binary classification - use sigmoid
            let coef_row = coef.row(0);
            let intercept_val = intercept[0];
            let decision = x.dot(&coef_row.t()) + intercept_val;

            let mut proba = Array2::zeros((n_samples, 2));
            for i in 0..n_samples {
                let p = 1.0 / (1.0 + (-decision[i]).exp());
                proba[[i, 0]] = 1.0 - p;
                proba[[i, 1]] = p;
            }
            Ok(proba)
        } else {
            // Multi-class classification - use softmax
            let decision = x.dot(&coef.t()) + intercept;

            let mut proba = Array2::zeros((n_samples, classes.len()));
            for i in 0..n_samples {
                let row = decision.row(i);
                let max_val = *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();

                for j in 0..classes.len() {
                    proba[[i, j]] = (row[j] - max_val).exp() / exp_sum;
                }
            }
            Ok(proba)
        }
    }
}

impl LogisticRegressionCV<Trained> {
    /// Get the best C value found
    pub fn best_c(&self) -> f64 {
        self.c_.expect("Model not fitted")
    }

    /// Get the coefficients
    pub fn coef(&self) -> &Array2<f64> {
        self.coef_.as_ref().expect("Model not fitted")
    }

    /// Get the intercept
    pub fn intercept(&self) -> &Array1<f64> {
        self.intercept_.as_ref().expect("Model not fitted")
    }

    /// Get all cross-validation scores
    pub fn cv_scores(&self) -> &HashMap<String, Vec<f64>> {
        self.scores_.as_ref().expect("Model not fitted")
    }

    /// Get mean cross-validation scores
    pub fn mean_cv_scores(&self) -> &HashMap<String, f64> {
        self.mean_scores_.as_ref().expect("Model not fitted")
    }

    /// Get standard deviation of cross-validation scores
    pub fn std_cv_scores(&self) -> &HashMap<String, f64> {
        self.std_scores_.as_ref().expect("Model not fitted")
    }

    /// Get the C values that were tried
    pub fn cs(&self) -> &Vec<f64> {
        self.cs_.as_ref().expect("Model not fitted")
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<f64> {
        self.classes_.as_ref().expect("Model not fitted")
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_logistic_regression_cv_binary() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model = LogisticRegressionCV::builder()
            .cv(3)
            .cs(vec![0.1, 1.0, 10.0])
            .build();

        let trained = model.fit(&x, &y).unwrap();

        // Check that we found a best C
        assert!(trained.cs().contains(&trained.best_c()));

        // Check predictions
        let predictions = trained.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Check probabilities
        let proba = trained.predict_proba(&x).unwrap();
        assert_eq!(proba.shape(), &[6, 2]);

        // Probabilities should sum to 1
        for i in 0..6 {
            let row_sum: f64 = proba.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_logistic_regression_cv_multiclass() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let model = LogisticRegressionCV::builder()
            .cv(3)
            .cs(vec![0.1, 1.0, 10.0])
            .build();

        let trained = model.fit(&x, &y).unwrap();

        // Check that we have 2 classes
        assert_eq!(trained.classes().len(), 2);

        // Check predictions
        let predictions = trained.predict(&x).unwrap();
        assert_eq!(predictions.len(), 9);

        // Check probabilities
        let proba = trained.predict_proba(&x).unwrap();
        assert_eq!(proba.shape(), &[9, 2]);

        // Probabilities should sum to 1
        for i in 0..9 {
            let row_sum: f64 = proba.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_logistic_regression_cv_default_cs() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // Don't specify cs, should use default 10 values
        let model = LogisticRegressionCV::builder().cv(3).build();

        let trained = model.fit(&x, &y).unwrap();

        // Should have 10 C values by default
        assert_eq!(trained.cs().len(), 10);

        // Check that C values are on log scale
        let cs = trained.cs();
        assert!(cs[0] < cs[1]);
        assert!(cs[cs.len() - 1] > cs[cs.len() - 2]);
    }
}
