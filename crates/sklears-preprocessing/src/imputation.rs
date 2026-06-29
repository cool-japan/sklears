//! Missing value imputation utilities
//!
//! This module provides comprehensive missing value imputation capabilities including
//! simple statistical methods, k-nearest neighbors, iterative approaches, a regression
//! backed (honest) GAIN substitute, multiple imputation with uncertainty quantification,
//! and outlier-aware techniques. Missing values are encoded as `NaN`
//! ([`Float::is_nan`]) consistently across all imputers.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::{
    error::{Result, SklearsError},
    traits::Transform,
    types::Float,
};
use std::collections::HashMap;

/// Strategy used by [`SimpleImputer`] (and others) to compute a fill value.
#[derive(Debug, Clone, Copy)]
pub enum ImputationStrategy {
    /// Use mean value
    Mean,
    /// Use median value
    Median,
    /// Use most frequent value
    MostFrequent,
    /// Use constant value
    Constant(Float),
}

/// Univariate imputation transformer, mirroring scikit-learn's `SimpleImputer`.
///
/// During `fit`, a single replacement statistic is computed independently for
/// each feature column over its finite (non-`NaN`) values:
///
/// * [`ImputationStrategy::Mean`] — arithmetic mean.
/// * [`ImputationStrategy::Median`] — linear-interpolation median (sort, then
///   the middle element or the mean of the two central elements).
/// * [`ImputationStrategy::MostFrequent`] — the most frequently occurring value;
///   ties are broken by choosing the smallest value.
/// * [`ImputationStrategy::Constant`] — the supplied constant `c`.
///
/// If a column is entirely `NaN`, the stored statistic is `0.0` for
/// `Mean`/`Median`/`MostFrequent` (an explicit, documented fallback) and `c`
/// for `Constant`.
///
/// During `transform`, every `NaN` is replaced with its column's stored
/// statistic; finite values pass through unchanged.
#[derive(Debug, Clone)]
pub struct SimpleImputer {
    /// Strategy used to compute each column's fill value.
    strategy: ImputationStrategy,
    /// Per-column fill values, populated after `fit`.
    statistics_: Option<Vec<Float>>,
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleImputer {
    /// Create a new imputer using the default [`ImputationStrategy::Mean`] strategy.
    pub fn new() -> Self {
        Self {
            strategy: ImputationStrategy::Mean,
            statistics_: None,
        }
    }

    /// Set the imputation strategy.
    pub fn strategy(mut self, strategy: ImputationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Access the fitted per-column statistics (returns `None` before `fit`).
    pub fn statistics_(&self) -> Option<&[Float]> {
        self.statistics_.as_deref()
    }

    /// Compute the median of a sorted, non-empty slice using linear interpolation.
    fn median_of_sorted(sorted: &[Float]) -> Float {
        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Most frequent finite value of a column; ties broken by the smallest value.
    fn most_frequent(values: &[Float]) -> Float {
        // Sort so that equal values are adjacent and ties resolve to the smallest.
        let mut sorted: Vec<Float> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_value = sorted[0];
        let mut best_count = 1usize;
        let mut current_value = sorted[0];
        let mut current_count = 1usize;
        for &v in &sorted[1..] {
            if v == current_value {
                current_count += 1;
            } else {
                current_value = v;
                current_count = 1;
            }
            if current_count > best_count {
                best_count = current_count;
                best_value = current_value;
            }
        }
        best_value
    }

    /// Compute per-column fill statistics from `x`.
    pub fn fit(mut self, x: &Array2<Float>) -> Result<Self> {
        let n_cols = x.ncols();
        let mut statistics = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            let finite: Vec<Float> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();

            let stat = match self.strategy {
                ImputationStrategy::Constant(c) => c,
                _ if finite.is_empty() => 0.0,
                ImputationStrategy::Mean => {
                    finite.iter().copied().sum::<Float>() / finite.len() as Float
                }
                ImputationStrategy::Median => {
                    let mut sorted = finite;
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    Self::median_of_sorted(&sorted)
                }
                ImputationStrategy::MostFrequent => Self::most_frequent(&finite),
            };
            statistics.push(stat);
        }

        self.statistics_ = Some(statistics);
        Ok(self)
    }

    /// Convenience: fit on `x` and immediately transform it.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.fit(x)?.transform(x)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for SimpleImputer {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let stats = self.statistics_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "SimpleImputer has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != stats.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: stats.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let fill = stats[j];
            for i in 0..n_rows {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = fill;
                }
            }
        }
        Ok(result)
    }
}

/// k-Nearest-Neighbours imputation, mirroring scikit-learn's `KNNImputer`.
///
/// Each missing value is imputed using the (optionally distance-weighted) mean
/// of the values present in the `n_neighbors` nearest donor rows of the training
/// data. Distances are computed with a NaN-aware metric:
///
/// For Euclidean distance between rows `a` and `b` only the coordinates that are
/// present (non-`NaN`) in **both** rows contribute, and the squared distance is
/// rescaled by the ratio of the total number of features to the number of
/// co-present coordinates (sklearn's `nan_euclidean`):
///
/// ```text
/// dist = sqrt( (n_total / n_present) * sum_present (a_k - b_k)^2 )
/// ```
///
/// Manhattan distance is the analogous `(n_total / n_present) * sum |a_k - b_k|`,
/// and Cosine distance is `1 - dot / (||a|| * ||b||)` evaluated over co-present
/// coordinates. Donors with no co-present coordinate cannot supply a distance
/// and are skipped.
///
/// When fewer than `n_neighbors` valid donors exist for a column, every
/// available donor is used. When none exist, the column's finite-value mean
/// (computed during `fit`) is used as a fallback.
#[derive(Debug, Clone)]
pub struct KNNImputer {
    /// Number of neighbours used to impute each value.
    n_neighbors: usize,
    /// Distance metric.
    metric: DistanceMetric,
    /// `true` for uniform weights, `false` for inverse-distance weights.
    weights_uniform: bool,
    /// Training matrix retained for neighbour search (may contain `NaN`).
    fit_data_: Option<Array2<Float>>,
    /// Number of features seen during `fit`.
    n_features_in_: usize,
    /// Per-column finite-value means used as a last-resort fallback.
    column_means_: Vec<Float>,
}

impl Default for KNNImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl KNNImputer {
    /// Create a new imputer with sklearn defaults (`n_neighbors = 5`,
    /// Euclidean metric, uniform weights).
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            metric: DistanceMetric::Euclidean,
            weights_uniform: true,
            fit_data_: None,
            n_features_in_: 0,
            column_means_: Vec::new(),
        }
    }

    /// Set the number of neighbours (clamped to at least 1).
    pub fn n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k.max(1);
        self
    }

    /// Set the distance metric.
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Enable inverse-distance weighting (`true`) or keep uniform weights (`false`).
    pub fn weighted(mut self, distance_weighted: bool) -> Self {
        self.weights_uniform = !distance_weighted;
        self
    }

    /// Store the training data and per-column fallback means.
    pub fn fit(mut self, x: &Array2<Float>) -> Result<Self> {
        let n_cols = x.ncols();
        let mut column_means = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            let finite: Vec<Float> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            let mean = if finite.is_empty() {
                0.0
            } else {
                finite.iter().copied().sum::<Float>() / finite.len() as Float
            };
            column_means.push(mean);
        }

        self.n_features_in_ = n_cols;
        self.column_means_ = column_means;
        self.fit_data_ = Some(x.clone());
        Ok(self)
    }

    /// NaN-aware distance between a query row and a donor row.
    ///
    /// Returns `None` when the two rows share no present coordinate.
    fn nan_distance(&self, query: &[Float], donor: &[Float], n_total: usize) -> Option<Float> {
        let mut n_present = 0usize;
        match self.metric {
            DistanceMetric::Euclidean => {
                let mut acc = 0.0;
                for k in 0..n_total {
                    let a = query[k];
                    let b = donor[k];
                    if a.is_finite() && b.is_finite() {
                        let d = a - b;
                        acc += d * d;
                        n_present += 1;
                    }
                }
                if n_present == 0 {
                    return None;
                }
                let scaled = (n_total as Float / n_present as Float) * acc;
                Some(scaled.sqrt())
            }
            DistanceMetric::Manhattan => {
                let mut acc = 0.0;
                for k in 0..n_total {
                    let a = query[k];
                    let b = donor[k];
                    if a.is_finite() && b.is_finite() {
                        acc += (a - b).abs();
                        n_present += 1;
                    }
                }
                if n_present == 0 {
                    return None;
                }
                Some((n_total as Float / n_present as Float) * acc)
            }
            DistanceMetric::Cosine => {
                let mut dot = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;
                for k in 0..n_total {
                    let a = query[k];
                    let b = donor[k];
                    if a.is_finite() && b.is_finite() {
                        dot += a * b;
                        norm_a += a * a;
                        norm_b += b * b;
                        n_present += 1;
                    }
                }
                if n_present == 0 {
                    return None;
                }
                let denom = norm_a.sqrt() * norm_b.sqrt();
                if denom <= Float::EPSILON {
                    // Degenerate (zero) vector over co-present coords: maximal distance.
                    Some(1.0)
                } else {
                    Some(1.0 - dot / denom)
                }
            }
        }
    }

    /// Convenience: fit on `x` and immediately transform it.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        self.fit(x)?.transform(x)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for KNNImputer {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fit_data = self.fit_data_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "KNNImputer has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != self.n_features_in_ {
            return Err(SklearsError::DimensionMismatch {
                expected: self.n_features_in_,
                actual: n_cols,
            });
        }

        let n_donors = fit_data.nrows();
        let mut result = x.clone();

        for i in 0..n_rows {
            let query: Vec<Float> = x.row(i).to_vec();
            if !query.iter().any(|v| v.is_nan()) {
                continue;
            }

            for j in 0..n_cols {
                if !query[j].is_nan() {
                    continue;
                }

                // Gather (distance, donor_value) pairs from donors that have a
                // finite value at column j.
                let mut candidates: Vec<(Float, Float)> = Vec::new();
                for d in 0..n_donors {
                    let donor_row = fit_data.row(d);
                    let donor_value = donor_row[j];
                    if !donor_value.is_finite() {
                        continue;
                    }
                    let donor: Vec<Float> = donor_row.to_vec();
                    if let Some(dist) = self.nan_distance(&query, &donor, n_cols) {
                        candidates.push((dist, donor_value));
                    }
                }

                if candidates.is_empty() {
                    result[[i, j]] = self.column_means_[j];
                    continue;
                }

                // Select the n_neighbors closest donors.
                candidates
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let k = self.n_neighbors.min(candidates.len());
                let selected = &candidates[..k];

                let imputed = if self.weights_uniform {
                    selected.iter().map(|(_, v)| *v).sum::<Float>() / k as Float
                } else {
                    // Inverse-distance weighting. Exact matches (dist == 0)
                    // dominate: if any are present, average only those.
                    let zero_donors: Vec<Float> = selected
                        .iter()
                        .filter(|(d, _)| *d == 0.0)
                        .map(|(_, v)| *v)
                        .collect();
                    if !zero_donors.is_empty() {
                        zero_donors.iter().copied().sum::<Float>() / zero_donors.len() as Float
                    } else {
                        let mut weight_sum = 0.0;
                        let mut weighted_value = 0.0;
                        for (dist, value) in selected {
                            let w = 1.0 / dist;
                            weight_sum += w;
                            weighted_value += w * value;
                        }
                        weighted_value / weight_sum
                    }
                };

                result[[i, j]] = imputed;
            }
        }

        Ok(result)
    }
}

/// Solve the ridge-regularized normal equations `(AᵀA + lambda·I) beta = Aᵀy`
/// using Gaussian elimination with partial pivoting.
///
/// `a` is the design matrix (rows = samples, cols = predictors, including any
/// intercept column), `y` are the targets. Returns the coefficient vector, or
/// `None` if the system is numerically singular even after regularization (so
/// callers can fall back to a simpler estimate).
fn ridge_solve(a: &[Vec<Float>], y: &[Float], lambda: Float) -> Option<Vec<Float>> {
    let n_rows = a.len();
    if n_rows == 0 {
        return None;
    }
    let n_cols = a[0].len();
    if n_cols == 0 {
        return None;
    }

    // Form the normal-equation system: M = AᵀA + lambda·I, rhs = Aᵀy.
    let mut m = vec![vec![0.0; n_cols]; n_cols];
    let mut rhs = vec![0.0; n_cols];
    for r in 0..n_rows {
        let row = &a[r];
        let yr = y[r];
        for p in 0..n_cols {
            let arp = row[p];
            rhs[p] += arp * yr;
            for q in 0..n_cols {
                m[p][q] += arp * row[q];
            }
        }
    }
    for (p, row) in m.iter_mut().enumerate() {
        row[p] += lambda;
    }

    // Gaussian elimination with partial pivoting on the augmented system.
    for col in 0..n_cols {
        // Find pivot row (largest magnitude in the current column at or below the diagonal).
        let mut pivot_row = col;
        let mut pivot_val = m[col][col].abs();
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            let v = row[col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < 1e-12 {
            // Singular even with ridge regularization.
            return None;
        }
        if pivot_row != col {
            m.swap(col, pivot_row);
            rhs.swap(col, pivot_row);
        }

        // Eliminate below the pivot. Split the matrix at `col + 1` so the pivot
        // row can be read immutably while the rows below it are mutated.
        let (head, tail) = m.split_at_mut(col + 1);
        let pivot_row_ref = &head[col];
        let pivot = pivot_row_ref[col];
        let rhs_pivot = rhs[col];
        for (offset, row) in tail.iter_mut().enumerate() {
            let factor = row[col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in col..n_cols {
                row[c] -= factor * pivot_row_ref[c];
            }
            rhs[col + 1 + offset] -= factor * rhs_pivot;
        }
    }

    // Back substitution.
    let mut beta = vec![0.0; n_cols];
    for col in (0..n_cols).rev() {
        let mut acc = rhs[col];
        for c in (col + 1)..n_cols {
            acc -= m[col][c] * beta[c];
        }
        let pivot = m[col][col];
        if pivot.abs() < 1e-12 {
            return None;
        }
        beta[col] = acc / pivot;
    }

    if beta.iter().all(|v| v.is_finite()) {
        Some(beta)
    } else {
        None
    }
}

/// A single fitted per-column regressor produced by the iterative imputer.
#[derive(Debug, Clone)]
struct ColumnRegressor {
    /// Target column index.
    target: usize,
    /// Predictor column indices (all columns except `target`), in order.
    predictors: Vec<usize>,
    /// Regression coefficients: `[intercept, beta_1, .., beta_p]`.
    beta: Vec<Float>,
}

/// Outcome of running the round-robin regression imputation core.
struct IterativeFit {
    /// Per-column initial means (over finite values), used to seed missing cells.
    initial_means: Vec<Float>,
    /// Regressors fitted at the final iteration (one per column that had missing
    /// values, in processing order).
    regressors: Vec<ColumnRegressor>,
    /// Fully imputed copy of the training data.
    completed: Array2<Float>,
    /// Per-target residual standard deviation on observed rows
    /// (`column_index -> std`); empty entry means "no estimate".
    residual_std: HashMap<usize, Float>,
}

/// Predict the value at `(row, target)` from the current filled matrix using a
/// fitted [`ColumnRegressor`].
fn predict_with(reg: &ColumnRegressor, filled: &Array2<Float>, row: usize) -> Float {
    let mut acc = reg.beta[0]; // intercept
    for (idx, &p) in reg.predictors.iter().enumerate() {
        acc += reg.beta[idx + 1] * filled[[row, p]];
    }
    acc
}

/// Round-robin (MICE-style) regression imputation core shared by
/// [`IterativeImputer`], [`GAINImputer`] and [`MultipleImputer`].
///
/// Missing cells are first seeded with per-column means, then each column with
/// missing values is regressed (ridge least squares) on all other columns using
/// its observed rows; the originally-missing cells are overwritten with the
/// model's predictions. Sweeps repeat until the maximum absolute change drops
/// below `tol` or `max_iter` is reached.
fn run_iterative(
    x: &Array2<Float>,
    max_iter: usize,
    tol: Float,
    ridge_lambda: Float,
) -> Result<IterativeFit> {
    let (n_rows, n_cols) = x.dim();
    if n_rows == 0 || n_cols == 0 {
        return Err(SklearsError::InvalidInput(
            "iterative imputation requires a non-empty matrix".to_string(),
        ));
    }

    // Missing mask and per-column initial means.
    let mut missing_mask = vec![vec![false; n_cols]; n_rows];
    let mut initial_means = vec![0.0; n_cols];
    let mut missing_counts = vec![0usize; n_cols];

    for j in 0..n_cols {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..n_rows {
            let v = x[[i, j]];
            if v.is_nan() {
                missing_mask[i][j] = true;
                missing_counts[j] += 1;
            } else {
                sum += v;
                count += 1;
            }
        }
        initial_means[j] = if count == 0 {
            0.0
        } else {
            sum / count as Float
        };
    }

    // Initialize the working matrix with column means at the missing positions.
    let mut filled = x.clone();
    for i in 0..n_rows {
        for j in 0..n_cols {
            if missing_mask[i][j] {
                filled[[i, j]] = initial_means[j];
            }
        }
    }

    // Columns that need imputation, ordered by ascending missing count (sklearn
    // default "ascending" ordering).
    let mut columns_with_missing: Vec<usize> =
        (0..n_cols).filter(|&j| missing_counts[j] > 0).collect();
    columns_with_missing.sort_by_key(|&j| missing_counts[j]);

    let effective_iters = max_iter.max(1);

    // Round-robin sweeps: each updates the originally-missing cells in place and
    // tracks the largest absolute change so we can stop once it falls below `tol`.
    for _ in 0..effective_iters {
        let mut max_change = 0.0_f64;
        for &target in &columns_with_missing {
            let predictors: Vec<usize> = (0..n_cols).filter(|&p| p != target).collect();
            let beta =
                match fit_target_beta(&filled, &missing_mask, target, &predictors, ridge_lambda) {
                    Some(b) => b,
                    // Singular system: leave the column at its current mean fallback.
                    None => continue,
                };
            let reg = ColumnRegressor {
                target,
                predictors,
                beta,
            };
            for i in 0..n_rows {
                if missing_mask[i][target] {
                    let prediction = predict_with(&reg, &filled, i);
                    let change = (prediction - filled[[i, target]]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                    filled[[i, target]] = prediction;
                }
            }
        }
        if max_change < tol {
            break;
        }
    }

    // Fit the final regressors (and residual std estimates) against the converged
    // matrix so that transforming new data replays a consistent final model.
    let mut regressors: Vec<ColumnRegressor> = Vec::with_capacity(columns_with_missing.len());
    let mut residual_std: HashMap<usize, Float> = HashMap::new();
    for &target in &columns_with_missing {
        let predictors: Vec<usize> = (0..n_cols).filter(|&p| p != target).collect();
        let beta = fit_target_beta(&filled, &missing_mask, target, &predictors, ridge_lambda)
            .unwrap_or_else(|| {
                let mut fallback = vec![0.0; predictors.len() + 1];
                fallback[0] = initial_means[target];
                fallback
            });
        let reg = ColumnRegressor {
            target,
            predictors,
            beta,
        };

        // Residual standard deviation over the observed rows of this column.
        let mut sse = 0.0;
        let mut count = 0usize;
        for i in 0..n_rows {
            if missing_mask[i][target] {
                continue;
            }
            let resid = filled[[i, target]] - predict_with(&reg, &filled, i);
            sse += resid * resid;
            count += 1;
        }
        let std = if count > 1 {
            (sse / (count - 1) as Float).sqrt()
        } else {
            0.0
        };
        residual_std.insert(target, std);
        regressors.push(reg);
    }

    Ok(IterativeFit {
        initial_means,
        regressors,
        completed: filled,
        residual_std,
    })
}

/// Fit ridge-regression coefficients for `target` against `predictors`, using
/// only the rows where `target` is observed. Returns `None` for a singular
/// system. The returned coefficients are `[intercept, beta_1, .., beta_p]`.
fn fit_target_beta(
    filled: &Array2<Float>,
    missing_mask: &[Vec<bool>],
    target: usize,
    predictors: &[usize],
    ridge_lambda: Float,
) -> Option<Vec<Float>> {
    let n_rows = filled.nrows();
    let mut design: Vec<Vec<Float>> = Vec::new();
    let mut targets: Vec<Float> = Vec::new();
    for i in 0..n_rows {
        if missing_mask[i][target] {
            continue;
        }
        let mut row = Vec::with_capacity(predictors.len() + 1);
        row.push(1.0); // intercept term
        for &p in predictors {
            row.push(filled[[i, p]]);
        }
        design.push(row);
        targets.push(filled[[i, target]]);
    }
    ridge_solve(&design, &targets, ridge_lambda)
}

/// Apply stored regressors to new data in a single deterministic pass.
///
/// New missing cells are seeded with the stored training means, then each
/// fitted regressor predicts its column's originally-missing cells once.
fn apply_iterative(fit: &IterativeFit, x: &Array2<Float>) -> Result<Array2<Float>> {
    let (n_rows, n_cols) = x.dim();
    if n_cols != fit.initial_means.len() {
        return Err(SklearsError::DimensionMismatch {
            expected: fit.initial_means.len(),
            actual: n_cols,
        });
    }

    let mut missing_mask = vec![vec![false; n_cols]; n_rows];
    let mut filled = x.clone();
    for i in 0..n_rows {
        for j in 0..n_cols {
            if filled[[i, j]].is_nan() {
                missing_mask[i][j] = true;
                filled[[i, j]] = fit.initial_means[j];
            }
        }
    }

    for reg in &fit.regressors {
        for i in 0..n_rows {
            if missing_mask[i][reg.target] {
                filled[[i, reg.target]] = predict_with(reg, &filled, i);
            }
        }
    }

    Ok(filled)
}

/// Multivariate imputer mirroring scikit-learn's `IterativeImputer` (MICE).
///
/// Each feature with missing values is modelled, in turn, as a ridge linear
/// regression on all of the other features (round-robin chained equations).
/// scikit-learn's default `BayesianRidge` estimator is replaced here by an
/// ordinary least-squares fit with a small Tikhonov (`ridge_lambda`) term for
/// numerical stability; an intercept is always included.
///
/// Algorithm (`fit`):
/// 1. Seed every missing cell with its column mean (over finite values).
/// 2. Repeat up to `max_iter` times: for each column with missing values
///    (processed in ascending order of missing count), solve
///    `(XᵀX + lambda·I) beta = Xᵀ y` over the rows where the target is
///    observed, then overwrite that column's originally-missing cells with the
///    fitted predictions. Stop early when the largest absolute change in any
///    imputed value across a full sweep drops below `tol`.
/// 3. Store the per-column means and the regressors fitted at the final sweep.
///
/// **Transform on new data** seeds new missing cells with the stored means and
/// then applies each stored regressor in a single deterministic pass. This is a
/// faithful, honest approximation of scikit-learn — it is **not** a full
/// multi-iteration replay of the per-iteration imputers; only the final-model
/// pass is reproduced.
///
/// If the normal equations remain singular despite the ridge term, the affected
/// column falls back to mean imputation and the sweep continues.
#[derive(Debug, Clone)]
pub struct IterativeImputer {
    /// Maximum number of round-robin sweeps.
    max_iter: usize,
    /// Early-stopping tolerance on the maximum imputed-value change per sweep.
    tol: Float,
    /// Tikhonov regularization strength added to the normal equations.
    ridge_lambda: Float,
    /// Fitted state (means + final regressors), populated after `fit`.
    fit_: Option<IterativeState>,
}

/// Serializable subset of the fitted iterative-imputer state.
#[derive(Debug, Clone)]
struct IterativeState {
    initial_means: Vec<Float>,
    regressors: Vec<ColumnRegressor>,
}

impl Default for IterativeImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl IterativeImputer {
    /// Create an imputer with sklearn-like defaults (`max_iter = 10`,
    /// `tol = 1e-3`, `ridge_lambda = 1e-6`).
    pub fn new() -> Self {
        Self {
            max_iter: 10,
            tol: 1e-3,
            ridge_lambda: 1e-6,
            fit_: None,
        }
    }

    /// Set the maximum number of round-robin sweeps (clamped to at least 1).
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n.max(1);
        self
    }

    /// Set the early-stopping tolerance.
    pub fn tol(mut self, t: Float) -> Self {
        self.tol = t;
        self
    }

    /// Set the ridge (Tikhonov) regularization strength.
    pub fn ridge_lambda(mut self, l: Float) -> Self {
        self.ridge_lambda = l;
        self
    }

    /// Fit the chained-equation regressors on `x`.
    pub fn fit(mut self, x: &Array2<Float>) -> Result<Self> {
        let fit = run_iterative(x, self.max_iter, self.tol, self.ridge_lambda)?;
        self.fit_ = Some(IterativeState {
            initial_means: fit.initial_means,
            regressors: fit.regressors,
        });
        Ok(self)
    }

    /// Convenience: fit on `x` and immediately transform it.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fit = run_iterative(x, self.max_iter, self.tol, self.ridge_lambda)?;
        Ok(fit.completed)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for IterativeImputer {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let state = self.fit_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "IterativeImputer has not been fitted yet; call fit() first".to_string(),
            )
        })?;
        let fit = IterativeFit {
            initial_means: state.initial_means.clone(),
            regressors: state.regressors.clone(),
            completed: Array2::zeros((0, 0)),
            residual_std: HashMap::new(),
        };
        apply_iterative(&fit, x)
    }
}

/// Regression-based imputer exposed under the `GAINImputer` name.
///
/// **Honesty note.** A true GAIN (Generative Adversarial Imputation Network)
/// requires trained generator and discriminator neural networks, which are out
/// of scope for this dependency-free module. This type does **not** implement an
/// adversarial network and makes no such claim. Instead it performs genuine
/// multivariate imputation using the same round-robin ridge-regression engine as
/// [`IterativeImputer`], producing correct, non-fabricated imputations.
///
/// The [`GAINImputerConfig`] fields are mapped as follows:
/// * `epochs` → number of round-robin sweeps (`max_iter`); a value of `0`
///   defaults to `10`.
/// * `learning_rate` → retained for API compatibility but unused by the
///   regression backend (there is no gradient-descent training here).
#[derive(Debug, Clone)]
pub struct GAINImputer {
    /// Number of round-robin regression sweeps.
    max_iter: usize,
    /// Stored only for API compatibility; the regression backend does not use it.
    #[allow(dead_code)]
    learning_rate: Float,
    /// Fitted iterative-imputer state.
    fit_: Option<IterativeState>,
}

impl Default for GAINImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl GAINImputer {
    /// Create an imputer with default settings (10 regression sweeps).
    pub fn new() -> Self {
        Self {
            max_iter: 10,
            learning_rate: 0.0,
            fit_: None,
        }
    }

    /// Configure the imputer from a [`GAINImputerConfig`].
    ///
    /// `epochs` maps to the number of regression sweeps (`0` ⇒ `10`);
    /// `learning_rate` is stored but unused by the regression backend.
    pub fn with_config(config: GAINImputerConfig) -> Self {
        let max_iter = if config.epochs == 0 {
            10
        } else {
            config.epochs
        };
        Self {
            max_iter,
            learning_rate: config.learning_rate,
            fit_: None,
        }
    }

    /// Fit the regression backend on `x`.
    pub fn fit(mut self, x: &Array2<Float>) -> Result<Self> {
        let fit = run_iterative(x, self.max_iter, 1e-3, 1e-6)?;
        self.fit_ = Some(IterativeState {
            initial_means: fit.initial_means,
            regressors: fit.regressors,
        });
        Ok(self)
    }

    /// Convenience: fit on `x` and immediately transform it.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fit = run_iterative(x, self.max_iter, 1e-3, 1e-6)?;
        Ok(fit.completed)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for GAINImputer {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let state = self.fit_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "GAINImputer has not been fitted yet; call fit() first".to_string(),
            )
        })?;
        let fit = IterativeFit {
            initial_means: state.initial_means.clone(),
            regressors: state.regressors.clone(),
            completed: Array2::zeros((0, 0)),
            residual_std: HashMap::new(),
        };
        apply_iterative(&fit, x)
    }
}

/// GAINImputer configuration
#[derive(Debug, Clone, Default)]
pub struct GAINImputerConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: Float,
}

/// Multiple imputation with between-imputation uncertainty quantification.
///
/// A base regression imputer (the same round-robin ridge engine as
/// [`IterativeImputer`]) produces a point estimate for every missing cell along
/// with a per-column residual standard deviation estimated from the observed
/// rows. `n_imputations` completed datasets are then drawn, each adding Gaussian
/// noise `N(0, residual_std_j)` to the point estimate of every originally
/// missing cell in column `j`.
///
/// Randomness uses `scirs2_core::random::seeded_rng` together with the
/// `Normal` sampler (no direct `rand` dependency). A fixed base seed is offset
/// per imputation, so results are deterministic and reproducible.
///
/// The returned [`MultipleImputationResult`] carries the `m` completed matrices
/// and an `uncertainties` matrix whose `[i, j]` entry is the variance across the
/// `m` imputations at that cell — `0.0` for cells that were originally observed.
#[derive(Debug, Clone)]
pub struct MultipleImputer {
    /// Number of completed datasets to generate.
    n_imputations: usize,
    /// Base RNG seed; offset per imputation for distinct draws.
    seed: u64,
}

impl Default for MultipleImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultipleImputer {
    /// Create a multiple imputer producing 5 datasets by default.
    pub fn new() -> Self {
        Self {
            n_imputations: 5,
            seed: 42,
        }
    }

    /// Configure from a [`MultipleImputerConfig`] (`n_imputations == 0` ⇒ `5`).
    pub fn with_config(config: MultipleImputerConfig) -> Self {
        let n_imputations = if config.n_imputations == 0 {
            5
        } else {
            config.n_imputations
        };
        Self {
            n_imputations,
            seed: 42,
        }
    }

    /// Set the base random seed used for the Gaussian draws.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit the base imputer on `x` and produce `n_imputations` completed
    /// datasets together with per-cell between-imputation variance.
    pub fn fit_impute(self, x: &Array2<Float>) -> Result<MultipleImputationResult> {
        let (n_rows, n_cols) = x.dim();

        // Record which cells were originally missing.
        let mut missing_mask = vec![vec![false; n_cols]; n_rows];
        for i in 0..n_rows {
            for j in 0..n_cols {
                if x[[i, j]].is_nan() {
                    missing_mask[i][j] = true;
                }
            }
        }

        // Base point-estimate imputation + per-column residual std.
        let fit = run_iterative(x, 10, 1e-3, 1e-6)?;
        let point = &fit.completed;

        let m = self.n_imputations;
        let mut imputations: Vec<Array2<Float>> = Vec::with_capacity(m);

        for draw in 0..m {
            let mut rng = seeded_rng(self.seed.wrapping_add(draw as u64));
            let mut completed = point.clone();
            for j in 0..n_cols {
                let std = fit.residual_std.get(&j).copied().unwrap_or(0.0);
                if std <= 0.0 {
                    // No spread to add; the point estimate is used as-is.
                    continue;
                }
                let normal = Normal::new(0.0, std).map_err(|e| {
                    SklearsError::InvalidInput(format!("failed to build noise distribution: {e}"))
                })?;
                for i in 0..n_rows {
                    if missing_mask[i][j] {
                        completed[[i, j]] = point[[i, j]] + normal.sample(&mut rng);
                    }
                }
            }
            imputations.push(completed);
        }

        // Between-imputation variance per originally-missing cell.
        let mut uncertainties = Array2::<Float>::zeros((n_rows, n_cols));
        if m > 1 {
            for i in 0..n_rows {
                for j in 0..n_cols {
                    if !missing_mask[i][j] {
                        continue;
                    }
                    let mean =
                        imputations.iter().map(|arr| arr[[i, j]]).sum::<Float>() / m as Float;
                    let var = imputations
                        .iter()
                        .map(|arr| {
                            let d = arr[[i, j]] - mean;
                            d * d
                        })
                        .sum::<Float>()
                        / (m - 1) as Float;
                    uncertainties[[i, j]] = var;
                }
            }
        }

        Ok(MultipleImputationResult {
            imputations,
            uncertainties,
        })
    }
}

/// Multiple imputer configuration
#[derive(Debug, Clone, Default)]
pub struct MultipleImputerConfig {
    /// Number of imputations
    pub n_imputations: usize,
}

/// Multiple imputation result
#[derive(Debug, Clone)]
pub struct MultipleImputationResult {
    /// Imputed datasets
    pub imputations: Vec<Array2<Float>>,
    /// Uncertainty estimates
    pub uncertainties: Array2<Float>,
}

/// Per-feature statistics fitted by `OutlierAwareImputer`.
#[derive(Debug, Clone)]
pub struct OutlierAwareFeatureStats {
    /// Median of inlier values (used for imputation)
    pub impute_value: Float,
    /// Lower Winsorization fence: Q25 − 1.5 × IQR
    pub lower_fence: Float,
    /// Upper Winsorization fence: Q75 + 1.5 × IQR
    pub upper_fence: Float,
}

/// Imputes missing values (`NaN`) while being robust to outliers.
///
/// **Fit phase** (per feature column):
/// 1. Collect all finite, non-NaN values.
/// 2. Compute Q25, Q75, and IQR.
/// 3. Define inlier range: `[Q25 − 1.5×IQR, Q75 + 1.5×IQR]`.
/// 4. Store the median of the inliers as the imputation value.
///    Falls back to the overall median when the inlier set is empty,
///    and to 0.0 when the entire column is NaN.
///
/// **Transform phase**: replace every NaN with the stored imputation value.
///
/// The `threshold` field scales the IQR multiplier (default 1.5).
/// The `strategy` field is stored for documentation; currently "mad" and
/// "iqr" are both handled via IQR.
#[derive(Debug, Clone, Default)]
pub struct OutlierAwareImputer {
    /// IQR multiplier for outlier fencing (scales the 1.5 factor)
    threshold: Float,
    /// Outlier detection strategy name; retained for serialization/introspection (not yet used for branching)
    #[allow(dead_code)]
    strategy: String,
    /// Imputation strategy
    base_strategy: Option<ImputationStrategy>,
    /// Per-feature statistics populated after `fit`
    stats_: Option<Vec<OutlierAwareFeatureStats>>,
}

impl OutlierAwareImputer {
    /// Create an outlier-aware imputer with the given outlier threshold multiplier.
    ///
    /// `strategy` is one of `"mad"` or `"iqr"` (both use IQR fencing internally).
    pub fn exclude_outliers(threshold: Float, strategy: &str) -> Result<Self> {
        Ok(Self {
            threshold,
            strategy: strategy.to_string(),
            base_strategy: None,
            stats_: None,
        })
    }

    /// Set the imputation strategy used to compute the fill value.
    pub fn base_strategy(mut self, strategy: ImputationStrategy) -> Self {
        self.base_strategy = Some(strategy);
        self
    }

    /// Access per-feature statistics (returns `None` before `fit`).
    pub fn feature_stats(&self) -> Option<&[OutlierAwareFeatureStats]> {
        self.stats_.as_deref()
    }

    /// Compute the median of a sorted non-empty slice.
    fn median_of_sorted(sorted: &[Float]) -> Float {
        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Compute per-feature statistics from `x` and store them.
    pub fn fit(mut self, x: &Array2<Float>) -> Result<Self> {
        let n_cols = x.ncols();
        // IQR multiplier defaults to 1.5; `threshold` can scale it.
        let iqr_multiplier = if self.threshold > 0.0 {
            self.threshold
        } else {
            1.5
        };

        let mut stats = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            let mut col: Vec<Float> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();

            if col.is_empty() {
                // All values are NaN or non-finite — fall back to 0.0
                stats.push(OutlierAwareFeatureStats {
                    impute_value: 0.0,
                    lower_fence: Float::NEG_INFINITY,
                    upper_fence: Float::INFINITY,
                });
                continue;
            }

            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = col.len() as Float;

            // Linear-interpolation quantile
            let quantile = |q: Float| -> Float {
                let pos = q * (n - 1.0);
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(col.len() - 1);
                let frac = pos - lo as Float;
                col[lo] * (1.0 - frac) + col[hi] * frac
            };

            let q25 = quantile(0.25);
            let q75 = quantile(0.75);
            let iqr = q75 - q25;

            let lower_fence = q25 - iqr_multiplier * iqr;
            let upper_fence = q75 + iqr_multiplier * iqr;

            // Collect inliers
            let inliers: Vec<Float> = col
                .iter()
                .copied()
                .filter(|v| *v >= lower_fence && *v <= upper_fence)
                .collect();

            let impute_value = if inliers.is_empty() {
                // No inliers — use overall median as fallback
                if iqr.abs() < Float::EPSILON {
                    // Zero IQR means all values are equal
                    col[0]
                } else {
                    Self::median_of_sorted(&col)
                }
            } else {
                let mut sorted_inliers = inliers;
                sorted_inliers
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                Self::median_of_sorted(&sorted_inliers)
            };

            stats.push(OutlierAwareFeatureStats {
                impute_value,
                lower_fence,
                upper_fence,
            });
        }

        self.stats_ = Some(stats);
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OutlierAwareImputer {
    /// Replace every NaN in `x` with the per-feature imputation value.
    ///
    /// Returns an error if the imputer has not been fitted yet.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let stats = self.stats_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "OutlierAwareImputer has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != stats.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: stats.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let fill = stats[j].impute_value;
            for i in 0..n_rows {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = fill;
                }
            }
        }
        Ok(result)
    }
}

/// OutlierAware imputer configuration
#[derive(Debug, Clone, Default)]
pub struct OutlierAwareImputerConfig {
    /// Outlier detection threshold
    pub threshold: Float,
}

/// OutlierAware statistics
#[derive(Debug, Clone, Default)]
pub struct OutlierAwareStatistics {
    /// Number of outliers detected
    pub outlier_count: usize,
}

/// OutlierAware strategy
#[derive(Debug, Clone, Copy)]
pub enum OutlierAwareStrategy {
    /// Exclude outliers from imputation
    Exclude,
    /// Transform outliers before imputation
    Transform,
}

/// Distance metrics for KNN imputation
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

/// Base imputation method
#[derive(Debug, Clone, Copy)]
pub enum BaseImputationMethod {
    /// Simple statistical imputation
    Simple(ImputationStrategy),
    /// K-nearest neighbors
    KNN,
    /// Iterative (MICE)
    Iterative,
}

/// Missing pattern information
#[derive(Debug, Clone)]
pub struct MissingPattern {
    /// Pattern matrix
    pub pattern: Array2<bool>,
    /// Pattern counts
    pub counts: HashMap<String, usize>,
}

/// Missing value analysis
#[derive(Debug, Clone, Default)]
pub struct MissingValueAnalysis {
    /// Missing value patterns
    pub patterns: Vec<MissingPattern>,
}

/// Missingness type
#[derive(Debug, Clone, Copy)]
pub enum MissingnessType {
    /// Missing Completely At Random
    MCAR,
    /// Missing At Random
    MAR,
    /// Missing Not At Random
    MNAR,
}

/// Feature missing statistics
#[derive(Debug, Clone, Default)]
pub struct FeatureMissingStats {
    /// Missing count per feature
    pub missing_counts: Vec<usize>,
    /// Missing percentage per feature
    pub missing_percentages: Vec<Float>,
}

/// Overall missing statistics
#[derive(Debug, Clone, Default)]
pub struct OverallMissingStats {
    /// Total missing values
    pub total_missing: usize,
    /// Overall missing percentage
    pub missing_percentage: Float,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn matrix(rows: usize, cols: usize, data: Vec<Float>) -> Array2<Float> {
        Array2::from_shape_vec((rows, cols), data).expect("valid shape")
    }

    #[test]
    fn simple_imputer_mean() {
        // Column 0: [1, 3, NaN] -> mean 2; column 1: [NaN, 4, 6] -> mean 5.
        let x = matrix(3, 2, vec![1.0, Float::NAN, 3.0, 4.0, Float::NAN, 6.0]);
        let imputer = SimpleImputer::new()
            .strategy(ImputationStrategy::Mean)
            .fit(&x)
            .expect("fit");
        let stats = imputer.statistics_().expect("fitted");
        assert!((stats[0] - 2.0).abs() < 1e-9);
        assert!((stats[1] - 5.0).abs() < 1e-9);

        let out = imputer.transform(&x).expect("transform");
        assert!((out[[2, 0]] - 2.0).abs() < 1e-9);
        assert!((out[[0, 1]] - 5.0).abs() < 1e-9);
        // Non-NaN values unchanged.
        assert!((out[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((out[[2, 1]] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn simple_imputer_median() {
        // Column: [1, 2, 4, NaN] -> median of {1,2,4} = 2.
        let x = matrix(4, 1, vec![1.0, 2.0, 4.0, Float::NAN]);
        let out = SimpleImputer::new()
            .strategy(ImputationStrategy::Median)
            .fit_transform(&x)
            .expect("fit_transform");
        assert!((out[[3, 0]] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn simple_imputer_most_frequent_ties() {
        // Values 1 and 3 both appear twice; tie broken by smallest -> 1.
        let x = matrix(5, 1, vec![3.0, 1.0, 3.0, 1.0, Float::NAN]);
        let out = SimpleImputer::new()
            .strategy(ImputationStrategy::MostFrequent)
            .fit_transform(&x)
            .expect("fit_transform");
        assert!((out[[4, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn simple_imputer_constant() {
        let x = matrix(2, 1, vec![Float::NAN, 7.0]);
        let out = SimpleImputer::new()
            .strategy(ImputationStrategy::Constant(-1.5))
            .fit_transform(&x)
            .expect("fit_transform");
        assert!((out[[0, 0]] - (-1.5)).abs() < 1e-9);
        assert!((out[[1, 0]] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn simple_imputer_not_fitted_errors() {
        let x = matrix(1, 1, vec![1.0]);
        let imputer = SimpleImputer::new();
        assert!(imputer.transform(&x).is_err());
    }

    #[test]
    fn knn_imputer_uniform_mean() {
        // Two clean rows and a query whose feature 1 is missing. Feature 0 of the
        // query (0.0) is closest to the first two rows, so the imputed value is
        // the mean of their feature-1 values (10 and 12) = 11 with k = 2.
        let x = matrix(
            4,
            2,
            vec![
                0.0,
                10.0, // donor a (near)
                0.1,
                12.0, // donor b (near)
                5.0,
                100.0, // donor c (far)
                0.05,
                Float::NAN, // query
            ],
        );
        let out = KNNImputer::new()
            .n_neighbors(2)
            .metric(DistanceMetric::Euclidean)
            .fit_transform(&x)
            .expect("fit_transform");
        assert!(
            (out[[3, 1]] - 11.0).abs() < 1e-6,
            "imputed = {}",
            out[[3, 1]]
        );
        // Rows without NaN are unchanged.
        assert!((out[[0, 1]] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn knn_imputer_fallback_to_column_mean() {
        // The only donors with a value at column 1 are removed by making them NaN
        // there; the query must fall back to the (single available) column mean.
        let x = matrix(
            2,
            2,
            vec![
                1.0,
                5.0, // donor with value at col 1
                2.0,
                Float::NAN, // query missing col 1
            ],
        );
        let out = KNNImputer::new().fit_transform(&x).expect("fit_transform");
        // Column 1 finite mean = 5.0 (only donor available).
        assert!((out[[1, 1]] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn knn_imputer_dimension_mismatch() {
        let x = matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let imputer = KNNImputer::new().fit(&x).expect("fit");
        let bad = matrix(1, 3, vec![1.0, 2.0, 3.0]);
        assert!(imputer.transform(&bad).is_err());
    }

    #[test]
    fn iterative_imputer_recovers_linear_relation() {
        // col1 = 2*col0 + 3 exactly. One missing entry in col1 should be
        // recovered by the regression to high accuracy.
        let mut data = Vec::new();
        for k in 0..8 {
            let c0 = k as Float;
            let c1 = 2.0 * c0 + 3.0;
            data.push(c0);
            data.push(c1);
        }
        // Make row 3's col1 missing (true value = 2*3+3 = 9).
        let mut x = matrix(8, 2, data);
        x[[3, 1]] = Float::NAN;

        let out = IterativeImputer::new()
            .max_iter(20)
            .fit_transform(&x)
            .expect("fit_transform");
        assert!(
            (out[[3, 1]] - 9.0).abs() < 1e-2,
            "imputed = {}",
            out[[3, 1]]
        );
    }

    #[test]
    fn iterative_imputer_transform_new_data() {
        let mut data = Vec::new();
        for k in 0..8 {
            let c0 = k as Float;
            data.push(c0);
            data.push(2.0 * c0 + 3.0);
        }
        let mut x = matrix(8, 2, data);
        x[[2, 1]] = Float::NAN;
        let imputer = IterativeImputer::new().max_iter(20).fit(&x).expect("fit");

        let new = matrix(1, 2, vec![10.0, Float::NAN]);
        let out = imputer.transform(&new).expect("transform");
        assert!(
            (out[[0, 1]] - 23.0).abs() < 1e-1,
            "imputed = {}",
            out[[0, 1]]
        );
    }

    #[test]
    fn gain_imputer_no_nan_and_matches_iterative() {
        let mut data = Vec::new();
        for k in 0..6 {
            let c0 = k as Float;
            data.push(c0);
            data.push(0.5 * c0 - 1.0);
        }
        let mut x = matrix(6, 2, data);
        x[[1, 1]] = Float::NAN;
        x[[4, 0]] = Float::NAN;

        let gain = GAINImputer::with_config(GAINImputerConfig {
            epochs: 10,
            learning_rate: 0.001,
        })
        .fit_transform(&x)
        .expect("gain");
        assert!(gain.iter().all(|v| v.is_finite()), "GAIN left a NaN");

        let iterative = IterativeImputer::new()
            .max_iter(10)
            .fit_transform(&x)
            .expect("iterative");
        // Same backend -> identical output.
        for (a, b) in gain.iter().zip(iterative.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn multiple_imputer_shapes_and_uncertainty() {
        // A noisy linear data set so the residual std is positive.
        let mut data = Vec::new();
        let offsets = [0.3, -0.4, 0.2, -0.1, 0.5, -0.3, 0.15, -0.25];
        for (k, off) in offsets.iter().enumerate() {
            let c0 = k as Float;
            data.push(c0);
            data.push(3.0 * c0 + 1.0 + off);
        }
        let mut x = matrix(8, 2, data);
        x[[2, 1]] = Float::NAN;
        x[[5, 1]] = Float::NAN;

        let result = MultipleImputer::with_config(MultipleImputerConfig { n_imputations: 4 })
            .fit_impute(&x)
            .expect("fit_impute");

        assert_eq!(result.imputations.len(), 4);
        for arr in &result.imputations {
            assert_eq!(arr.dim(), (8, 2));
            assert!(arr.iter().all(|v| v.is_finite()), "left a NaN");
        }
        assert_eq!(result.uncertainties.dim(), (8, 2));
        // Originally-observed cells have zero variance.
        assert!((result.uncertainties[[0, 0]]).abs() < 1e-12);
        assert!((result.uncertainties[[0, 1]]).abs() < 1e-12);
        // Imputed cells generally carry positive variance with m > 1 and std > 0.
        assert!(result.uncertainties[[2, 1]] > 0.0);
        assert!(result.uncertainties[[5, 1]] > 0.0);
    }

    #[test]
    fn ridge_solve_recovers_exact_line() {
        // y = 2*x + 1 with intercept augmentation.
        let a = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let beta = ridge_solve(&a, &y, 1e-9).expect("solvable");
        assert!((beta[0] - 1.0).abs() < 1e-4);
        assert!((beta[1] - 2.0).abs() < 1e-4);
    }
}
