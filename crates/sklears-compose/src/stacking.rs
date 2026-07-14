//! Stacking ensembles
//!
//! Standard stacked generalization over [`PipelinePredictor`] base learners
//! plus a meta-learner: base learners are fitted on **out-of-fold** (k-fold
//! cross-validated) predictions of the training data to build meta-features,
//! avoiding the leakage that would result from training the meta-learner on
//! in-sample predictions the base learners had already memorized. The base
//! learners are then refit on the *full* training data for use at prediction
//! time, and the meta-learner is fitted on the out-of-fold meta-features
//! against the true targets.
//!
//! This mirrors the real, standard `sklearn.ensemble.StackingRegressor`
//! scheme. It is intentionally a separate implementation from
//! `sklears-ensemble`'s `StackingClassifier`/`StackingRegressor`, which use
//! different trait bounds (`sklears_ensemble` estimators, not
//! `Box<dyn PipelinePredictor>`) and cannot be re-exported as-is here.
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_compose::{StackingEnsemble, MockPredictor};
//!
//! let stacking = StackingEnsemble::builder()
//!     .base_learner("ridge", Box::new(ridge_model))
//!     .base_learner("tree", Box::new(tree_model))
//!     .cv_folds(5)
//!     .build();
//!
//! let fitted = stacking.fit(&x.view(), &y.view())?;
//! let predictions = fitted.predict(&x_test.view())?;
//! ```

use scirs2_core::linalg::lstsq_ndarray;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

use crate::PipelinePredictor;

/// Stacking ensemble over [`PipelinePredictor`] base learners plus a
/// meta-learner.
///
/// # Type Parameters
///
/// * `S` - State type ([`Untrained`] or [`StackingEnsembleTrained`])
#[derive(Debug)]
pub struct StackingEnsemble<S = Untrained> {
    state: S,
    /// Named base learners
    base_learners: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Meta-learner combining the base learners' out-of-fold predictions.
    /// When `None`, a built-in ordinary-least-squares learner is used.
    meta_learner: Option<Box<dyn PipelinePredictor>>,
    /// Number of folds used to generate out-of-fold meta-features
    cv_folds: usize,
    /// When true, the meta-learner also receives the original input features
    /// alongside the base learners' out-of-fold predictions (sklearn calls
    /// this `passthrough`).
    passthrough: bool,
    /// Random state controlling the k-fold shuffle, for reproducibility
    random_state: Option<u64>,
    /// Verbose output flag
    verbose: bool,
}

/// Trained state for [`StackingEnsemble`], produced by [`Fit::fit`].
pub struct StackingEnsembleTrained {
    /// Base learners, refit on the *full* training data (used at prediction
    /// time).
    fitted_base_learners: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Meta-learner, fitted on out-of-fold meta-features.
    fitted_meta_learner: Box<dyn PipelinePredictor>,
    /// Whether the meta-learner also consumes the original features.
    passthrough: bool,
    /// Number of input features seen during fitting.
    n_features_in: usize,
    /// The out-of-fold meta-feature matrix actually used to fit the
    /// meta-learner (exposed for introspection/testing: it should differ from
    /// what the final, fit-on-everything base learners would predict
    /// in-sample, proving the meta-learner was not trained on leaked data).
    oof_meta_features: Array2<Float>,
}

impl std::fmt::Debug for StackingEnsembleTrained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackingEnsembleTrained")
            .field("n_base_learners", &self.fitted_base_learners.len())
            .field("passthrough", &self.passthrough)
            .field("n_features_in", &self.n_features_in)
            .field("oof_meta_features_shape", &self.oof_meta_features.dim())
            .finish()
    }
}

impl StackingEnsemble<Untrained> {
    /// Create a new stacking ensemble builder
    #[must_use]
    pub fn builder() -> StackingEnsembleBuilder {
        StackingEnsembleBuilder::new()
    }

    /// Create a new stacking ensemble with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            base_learners: Vec::new(),
            meta_learner: None,
            cv_folds: 5,
            passthrough: false,
            random_state: None,
            verbose: false,
        }
    }

    /// Add a base learner
    #[must_use]
    pub fn add_base_learner(mut self, name: &str, learner: Box<dyn PipelinePredictor>) -> Self {
        self.base_learners.push((name.to_string(), learner));
        self
    }

    /// Set the meta-learner (defaults to a built-in OLS learner if unset)
    #[must_use]
    pub fn set_meta_learner(mut self, learner: Box<dyn PipelinePredictor>) -> Self {
        self.meta_learner = Some(learner);
        self
    }
}

impl Default for StackingEnsemble<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> StackingEnsemble<S> {
    /// Get base learner names
    #[must_use]
    pub fn base_learner_names(&self) -> Vec<&str> {
        self.base_learners
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get number of base learners
    #[must_use]
    pub fn n_base_learners(&self) -> usize {
        self.base_learners.len()
    }

    /// Get number of cross-validation folds configured
    #[must_use]
    pub fn cv_folds(&self) -> usize {
        self.cv_folds
    }

    /// Get whether passthrough is enabled
    #[must_use]
    pub fn passthrough(&self) -> bool {
        self.passthrough
    }
}

impl Estimator for StackingEnsemble<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for StackingEnsemble<Untrained> {
    type Fitted = StackingEnsemble<StackingEnsembleTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Fitted> {
        if self.base_learners.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "base_learners".to_string(),
                reason: "StackingEnsemble requires at least one base learner".to_string(),
            });
        }
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(),
                y.len()
            )));
        }
        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }
        if self.cv_folds < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "cv_folds".to_string(),
                reason: "cv_folds must be at least 2".to_string(),
            });
        }

        let n_samples = x.nrows();
        let n_learners = self.base_learners.len();
        let cv_folds = self.cv_folds.clamp(2, n_samples.max(2));

        // 1. Out-of-fold meta-features: every base learner's prediction for a
        //    sample comes from a clone that never saw that sample during its
        //    own fitting, so the meta-learner is never trained on leaked,
        //    in-sample-memorized predictions.
        let mut oof_meta_features = Array2::<Float>::zeros((n_samples, n_learners));
        let folds = k_fold_indices(n_samples, cv_folds, self.random_state);
        for (train_idx, test_idx) in &folds {
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            let x_train = x.select(Axis(0), train_idx);
            let y_train = y.select(Axis(0), train_idx);
            let x_test = x.select(Axis(0), test_idx);

            for (learner_idx, (_, learner)) in self.base_learners.iter().enumerate() {
                let mut fold_learner = learner.clone_predictor();
                fold_learner.fit(&x_train.view(), &y_train.view())?;
                let preds = fold_learner.predict(&x_test.view())?;
                for (row_pos, &sample_idx) in test_idx.iter().enumerate() {
                    oof_meta_features[[sample_idx, learner_idx]] = preds[row_pos];
                }
            }
        }

        // 2. Refit each base learner on the FULL training data: these are the
        //    models genuinely used to produce meta-features at prediction
        //    time on new data.
        let mut fitted_base_learners = Vec::with_capacity(n_learners);
        for (name, mut learner) in self.base_learners {
            learner.fit(x, y)?;
            fitted_base_learners.push((name, learner));
        }

        // 3. Optionally passthrough the original features, then fit the
        //    meta-learner on the out-of-fold meta-features.
        let meta_input = if self.passthrough {
            concatenate_columns(&oof_meta_features, x)
        } else {
            oof_meta_features.clone()
        };

        let mut meta_learner = self
            .meta_learner
            .unwrap_or_else(|| Box::new(OlsMetaLearner::default()));
        meta_learner.fit(&meta_input.view(), y)?;

        Ok(StackingEnsemble {
            state: StackingEnsembleTrained {
                fitted_base_learners,
                fitted_meta_learner: meta_learner,
                passthrough: self.passthrough,
                n_features_in: x.ncols(),
                oof_meta_features,
            },
            base_learners: Vec::new(),
            meta_learner: None,
            cv_folds: self.cv_folds,
            passthrough: self.passthrough,
            random_state: self.random_state,
            verbose: self.verbose,
        })
    }
}

impl StackingEnsemble<StackingEnsembleTrained> {
    /// Predict using the fitted stacking ensemble.
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }

        let base_predictions = self.collect_base_predictions(x)?;
        let meta_features = build_design_matrix(&base_predictions, x.nrows());
        let meta_input = if self.state.passthrough {
            concatenate_columns(&meta_features, x)
        } else {
            meta_features
        };

        self.state.fitted_meta_learner.predict(&meta_input.view())
    }

    /// Collect predictions from every fitted base learner.
    fn collect_base_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
        self.state
            .fitted_base_learners
            .iter()
            .map(|(_, learner)| learner.predict(x))
            .collect()
    }

    /// Get the fitted base learners.
    #[must_use]
    pub fn base_learners(&self) -> &[(String, Box<dyn PipelinePredictor>)] {
        &self.state.fitted_base_learners
    }

    /// Get the fitted meta-learner.
    #[must_use]
    pub fn meta_learner(&self) -> &dyn PipelinePredictor {
        &*self.state.fitted_meta_learner
    }

    /// Get the out-of-fold meta-feature matrix the meta-learner was fitted
    /// on.
    #[must_use]
    pub fn oof_meta_features(&self) -> &Array2<Float> {
        &self.state.oof_meta_features
    }

    /// Get the number of input features seen during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.state.n_features_in
    }
}

/// Build `(train_indices, test_indices)` pairs for k-fold cross-validation
/// (shuffle + contiguous chunk split). Mirrors the private algorithm used by
/// `cross_validation.rs`; duplicated locally since that helper isn't public.
fn k_fold_indices(
    n_samples: usize,
    n_folds: usize,
    random_state: Option<u64>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    if n_samples == 0 {
        return Vec::new();
    }
    let n_folds = n_folds.clamp(2, n_samples.max(2));

    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut thread_rng()),
    };

    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let fold_size = n_samples / n_folds;
    let mut splits = Vec::with_capacity(n_folds);
    for fold in 0..n_folds {
        let start = fold * fold_size;
        let end = if fold + 1 == n_folds {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        let test_indices = indices[start..end].to_vec();
        let train_indices: Vec<usize> = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();

        splits.push((train_indices, test_indices));
    }

    splits
}

/// Stack per-learner prediction vectors as columns of a design matrix.
fn build_design_matrix(predictions: &[Array1<Float>], n_samples: usize) -> Array2<Float> {
    let n_learners = predictions.len();
    let mut design = Array2::<Float>::zeros((n_samples, n_learners));
    for (col, pred) in predictions.iter().enumerate() {
        for (row, &value) in pred.iter().enumerate() {
            design[[row, col]] = value;
        }
    }
    design
}

/// Horizontally concatenate two matrices with the same number of rows.
fn concatenate_columns(a: &Array2<Float>, b: &ArrayView2<'_, Float>) -> Array2<Float> {
    let n_samples = a.nrows();
    let mut out = Array2::<Float>::zeros((n_samples, a.ncols() + b.ncols()));
    out.slice_mut(s![.., ..a.ncols()]).assign(a);
    out.slice_mut(s![.., a.ncols()..]).assign(b);
    out
}

/// Minimal built-in ordinary-least-squares meta-learner used as
/// [`StackingEnsemble`]'s default when the caller doesn't supply their own
/// meta-learner via `.set_meta_learner(...)`. Fits an intercept plus one
/// coefficient per meta-feature column via the normal equations (through
/// `scirs2_core::linalg::lstsq_ndarray`) — the same closed-form solve used by
/// `model_fusion::learn_linear_weights`.
#[derive(Debug, Clone, Default)]
struct OlsMetaLearner {
    /// `coefficients[0]` is the intercept; `coefficients[1..]` are the
    /// per-column weights.
    coefficients: Option<Array1<Float>>,
}

impl PipelinePredictor for OlsMetaLearner {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        let coef = self
            .coefficients
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut preds = Array1::<Float>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut sum = coef[0];
            for j in 0..x.ncols() {
                sum += x[[i, j]] * coef[j + 1];
            }
            preds[i] = sum;
        }
        Ok(preds)
    }

    fn fit(&mut self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Design matrix with a bias/intercept column of ones prepended.
        let mut design = Array2::<Float>::ones((n_samples, n_features + 1));
        for i in 0..n_samples {
            for j in 0..n_features {
                design[[i, j + 1]] = x[[i, j]];
            }
        }

        let target: Array1<Float> = y.to_owned();
        let coef = lstsq_ndarray(&design, &target).map_err(|e| {
            SklearsError::NumericalError(format!("OLS meta-learner fit failed: {e}"))
        })?;

        self.coefficients = Some(coef);
        Ok(())
    }

    fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
        Box::new(self.clone())
    }
}

/// Builder for [`StackingEnsemble`]
#[derive(Debug)]
pub struct StackingEnsembleBuilder {
    base_learners: Vec<(String, Box<dyn PipelinePredictor>)>,
    meta_learner: Option<Box<dyn PipelinePredictor>>,
    cv_folds: usize,
    passthrough: bool,
    random_state: Option<u64>,
    verbose: bool,
}

impl StackingEnsembleBuilder {
    /// Create new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_learners: Vec::new(),
            meta_learner: None,
            cv_folds: 5,
            passthrough: false,
            random_state: None,
            verbose: false,
        }
    }

    /// Add a base learner
    #[must_use]
    pub fn base_learner(mut self, name: &str, learner: Box<dyn PipelinePredictor>) -> Self {
        self.base_learners.push((name.to_string(), learner));
        self
    }

    /// Set the meta-learner (defaults to a built-in OLS learner if unset)
    #[must_use]
    pub fn meta_learner(mut self, learner: Box<dyn PipelinePredictor>) -> Self {
        self.meta_learner = Some(learner);
        self
    }

    /// Set the number of cross-validation folds used to build out-of-fold
    /// meta-features
    #[must_use]
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Enable passing the original input features through to the
    /// meta-learner alongside the base learners' predictions
    #[must_use]
    pub fn passthrough(mut self, enable: bool) -> Self {
        self.passthrough = enable;
        self
    }

    /// Set random state controlling the k-fold shuffle
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set verbose flag
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the [`StackingEnsemble`]
    #[must_use]
    pub fn build(self) -> StackingEnsemble<Untrained> {
        StackingEnsemble {
            state: Untrained,
            base_learners: self.base_learners,
            meta_learner: self.meta_learner,
            cv_folds: self.cv_folds,
            passthrough: self.passthrough,
            random_state: self.random_state,
            verbose: self.verbose,
        }
    }
}

impl Default for StackingEnsembleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockPredictor;

    /// Test-only predictor computing a genuine (stateless) function of `x`:
    /// `x[:, 0] + bias`. Because it's a real function of the input rather
    /// than memorized training data, it behaves identically whether called
    /// in-fold or out-of-fold, which keeps the "beats any single base
    /// learner" test's arithmetic simple and exact.
    #[derive(Debug, Clone)]
    struct BiasedLinearPredictor {
        bias: Float,
    }

    impl PipelinePredictor for BiasedLinearPredictor {
        fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
            Ok(x.column(0).mapv(|v| v + self.bias))
        }

        fn fit(&mut self, _x: &ArrayView2<'_, Float>, _y: &ArrayView1<'_, Float>) -> SklResult<()> {
            Ok(())
        }

        fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_stacking_ensemble_builder() {
        let stacking = StackingEnsemble::builder()
            .base_learner("a", Box::new(MockPredictor::new()))
            .base_learner("b", Box::new(MockPredictor::new()))
            .cv_folds(4)
            .passthrough(true)
            .build();

        assert_eq!(stacking.n_base_learners(), 2);
        assert_eq!(stacking.cv_folds(), 4);
        assert!(stacking.passthrough());
        assert_eq!(stacking.base_learner_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_fit_requires_at_least_one_base_learner() {
        let x = Array2::<Float>::zeros((4, 1));
        let y = Array1::<Float>::zeros(4);
        let stacking = StackingEnsemble::builder().build();
        assert!(stacking.fit(&x.view(), &y.view()).is_err());
    }

    /// Core correctness test: a toy dataset where each individual base
    /// learner is systematically biased (`y + 10` and `y - 10`), but the
    /// trained meta-learner can combine them to recover `y` almost exactly —
    /// something a hard-coded / fake fusion could never do.
    #[test]
    fn test_stacking_beats_any_single_base_learner() {
        let n = 20;
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|v| v as Float).collect())
            .expect("valid shape");
        let y = Array1::from_vec((0..n).map(|v| v as Float).collect());

        let stacking = StackingEnsemble::builder()
            .base_learner("high", Box::new(BiasedLinearPredictor { bias: 10.0 }))
            .base_learner("low", Box::new(BiasedLinearPredictor { bias: -10.0 }))
            .cv_folds(5)
            .random_state(42)
            .build();

        let fitted = stacking
            .fit(&x.view(), &y.view())
            .expect("stacking fit should succeed");

        let stacked_preds = fitted.predict(&x.view()).expect("predict should succeed");
        let stacked_mse: Float = stacked_preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<Float>()
            / n as Float;

        for (_, learner) in fitted.base_learners() {
            let solo_preds = learner.predict(&x.view()).expect("predict should succeed");
            let solo_mse: Float = solo_preds
                .iter()
                .zip(y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<Float>()
                / n as Float;
            assert!(
                stacked_mse < solo_mse,
                "stacked mse ({stacked_mse}) should beat solo base learner mse ({solo_mse})"
            );
        }

        // The optimal combination (0.5*high + 0.5*low) exactly reproduces y.
        assert!(
            stacked_mse < 1e-6,
            "stacking should nearly perfectly recover y, got mse={stacked_mse}"
        );
    }

    /// Verifies meta-features are genuinely derived from out-of-fold
    /// predictions rather than leaked in-sample ones. `MockPredictor::fit`
    /// sets its intercept to `mean(y_train)`, which depends on exactly which
    /// rows it was trained on — so a real out-of-fold prediction must differ
    /// from what the *final* (fit-on-everything) model would have predicted
    /// in-sample for at least one held-out fold.
    #[test]
    fn test_meta_features_are_out_of_fold_not_leaked() {
        let n = 10;
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|v| v as Float).collect())
            .expect("valid shape");
        let y = Array1::from_vec(vec![
            1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0,
        ]);

        let stacking = StackingEnsemble::builder()
            .base_learner("m", Box::new(MockPredictor::new()))
            .cv_folds(5)
            .random_state(3)
            .build();

        let fitted = stacking
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");

        // What a LEAKED (in-sample) computation would have produced: predict
        // with the final (fit-on-everything) base learner directly.
        let leaked = fitted.base_learners()[0]
            .1
            .predict(&x.view())
            .expect("predict should succeed");
        let oof = fitted.oof_meta_features();

        let differs = (0..n).any(|i| (oof[[i, 0]] - leaked[i]).abs() > 1e-9);
        assert!(
            differs,
            "out-of-fold meta-features must differ from in-sample/leaked predictions"
        );
    }

    #[test]
    fn test_passthrough_includes_original_features() {
        let n = 12;
        let x = Array2::from_shape_vec((n, 2), (0..n * 2).map(|v| v as Float).collect())
            .expect("valid shape");
        let y = Array1::from_vec((0..n).map(|v| v as Float).collect());

        let stacking = StackingEnsemble::builder()
            .base_learner("m", Box::new(MockPredictor::new()))
            .cv_folds(3)
            .passthrough(true)
            .random_state(1)
            .build();

        let fitted = stacking
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");
        let preds = fitted.predict(&x.view()).expect("predict should succeed");
        assert_eq!(preds.len(), n);
        assert!(preds.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_default_meta_learner_is_used_when_unset() {
        let n = 8;
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|v| v as Float).collect())
            .expect("valid shape");
        let y = Array1::from_vec((0..n).map(|v| v as Float * 3.0).collect());

        let stacking = StackingEnsemble::builder()
            .base_learner("a", Box::new(BiasedLinearPredictor { bias: 0.0 }))
            .cv_folds(4)
            .random_state(5)
            .build();

        let fitted = stacking
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");
        // The OLS default should learn to scale "x" by ~3 to reproduce y=3x.
        let preds = fitted.predict(&x.view()).expect("predict should succeed");
        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1e-6, "expected {t}, got {p}");
        }
    }
}
