//! Metrics aggregation for combining multiple evaluation measures
//!
//! This module implements classical aggregation algorithms used to combine the
//! outputs of several feature-selection evaluators or rankers into a single
//! consensus. All routines are exact implementations of well-defined methods:
//!
//! * [`MetricsAggregator`] / [`WeightedAveraging`] — (weighted) arithmetic mean
//!   and median aggregation of scalar metric values.
//! * [`RankAggregation`] — Borda count and Kemeny-optimal rank aggregation.
//! * [`ConsensusMetrics`] — mean pairwise Kendall-tau agreement of a set of
//!   rankings.
//! * [`MultiCriteriaEvaluation`] — TOPSIS multi-criteria decision analysis.
//!
//! Ranking inputs follow the convention used throughout the crate: a ranking is
//! an ordered list of item indices where the element at position `p` is the item
//! placed at rank `p` (rank `0` is the best / most preferred item). Every ranking
//! passed to the rank-aggregation routines must be a permutation of the same item
//! set `{0, 1, ..., n-1}`.

use sklears_core::error::{Result as SklResult, SklearsError};

/// Aggregation strategy used by [`MetricsAggregator::aggregate_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Unweighted arithmetic mean of the metric values.
    Mean,
    /// Median of the metric values.
    Median,
    /// Weighted arithmetic mean (requires weights to be supplied).
    Weighted,
    /// Minimum metric value (pessimistic aggregation).
    Min,
    /// Maximum metric value (optimistic aggregation).
    Max,
}

/// Aggregator combining several scalar evaluation metrics into one value.
#[derive(Debug, Clone)]
pub struct MetricsAggregator;

impl MetricsAggregator {
    /// Aggregate metric values into a single scalar.
    ///
    /// When `weights` is `Some`, a weighted arithmetic mean is computed (the
    /// weights are normalised to sum to one); when it is `None`, the unweighted
    /// arithmetic mean is returned. Returns an error for empty input, mismatched
    /// weight lengths, or weights that are negative / sum to zero.
    pub fn aggregate_metrics(metrics: &[f64], weights: Option<&[f64]>) -> SklResult<f64> {
        match weights {
            Some(w) => WeightedAveraging::weighted_average(metrics, w),
            None => Self::aggregate_with(metrics, AggregationStrategy::Mean, None),
        }
    }

    /// Aggregate metric values using an explicit `AggregationStrategy`.
    ///
    /// `weights` is only consulted for `AggregationStrategy::Weighted`; for all
    /// other strategies it is ignored. Returns an error if `metrics` is empty, or
    /// if the weighted strategy is requested without valid weights.
    pub fn aggregate_with(
        metrics: &[f64],
        strategy: AggregationStrategy,
        weights: Option<&[f64]>,
    ) -> SklResult<f64> {
        if metrics.is_empty() {
            return Err(SklearsError::InvalidInput(
                "aggregate_metrics requires at least one metric value".to_string(),
            ));
        }

        match strategy {
            AggregationStrategy::Mean => Ok(metrics.iter().sum::<f64>() / metrics.len() as f64),
            AggregationStrategy::Median => Ok(median(metrics)),
            AggregationStrategy::Min => Ok(metrics.iter().copied().fold(f64::INFINITY, f64::min)),
            AggregationStrategy::Max => {
                Ok(metrics.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            }
            AggregationStrategy::Weighted => {
                let w = weights.ok_or_else(|| {
                    SklearsError::InvalidInput("weighted aggregation requires weights".to_string())
                })?;
                WeightedAveraging::weighted_average(metrics, w)
            }
        }
    }
}

/// Compute the median of a slice without mutating the caller's data.
fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Weighted averaging of metric values.
#[derive(Debug, Clone)]
pub struct WeightedAveraging;

impl WeightedAveraging {
    /// Compute the weighted arithmetic mean of `values` using `weights`.
    ///
    /// The weights are normalised internally so they sum to one, hence only their
    /// relative magnitudes matter. Returns an error when the inputs are empty, the
    /// lengths differ, any weight is negative or non-finite, or the weights sum to
    /// zero (in which case no meaningful average is defined).
    pub fn weighted_average(values: &[f64], weights: &[f64]) -> SklResult<f64> {
        if values.is_empty() {
            return Err(SklearsError::InvalidInput(
                "weighted_average requires at least one value".to_string(),
            ));
        }
        if values.len() != weights.len() {
            return Err(SklearsError::InvalidInput(format!(
                "weighted_average: values length ({}) must match weights length ({})",
                values.len(),
                weights.len()
            )));
        }

        let mut weight_sum = 0.0;
        for &w in weights {
            if !w.is_finite() {
                return Err(SklearsError::InvalidInput(
                    "weighted_average: weights must be finite".to_string(),
                ));
            }
            if w < 0.0 {
                return Err(SklearsError::InvalidInput(
                    "weighted_average: weights must be non-negative".to_string(),
                ));
            }
            weight_sum += w;
        }

        if weight_sum <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "weighted_average: sum of weights must be positive".to_string(),
            ));
        }

        let weighted_sum: f64 = values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum();

        Ok(weighted_sum / weight_sum)
    }
}

/// Rank aggregation methods producing a single consensus ranking.
#[derive(Debug, Clone)]
pub struct RankAggregation;

impl RankAggregation {
    /// Borda count rank aggregation.
    ///
    /// Each ranking awards `n - rank` points to the item at a given position
    /// (the best item, at rank `0`, receives `n` points; the worst receives `1`).
    /// Points are summed across all rankings and the consensus ranking lists items
    /// in order of decreasing total score. Ties are broken by smaller item index so
    /// the output is deterministic.
    ///
    /// Every ranking must be a permutation of `{0, ..., n-1}` for the same `n`.
    pub fn borda_count(rankings: &[Vec<usize>]) -> SklResult<Vec<usize>> {
        let n = validate_rankings(rankings)?;

        let mut scores = vec![0usize; n];
        for ranking in rankings {
            for (position, &item) in ranking.iter().enumerate() {
                // Item at position `position` (rank) receives `n - position` points.
                scores[item] += n - position;
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        // Sort by descending score, then ascending index for deterministic ties.
        order.sort_by(|&a, &b| scores[b].cmp(&scores[a]).then(a.cmp(&b)));
        Ok(order)
    }

    /// Kemeny-optimal rank aggregation.
    ///
    /// Returns the consensus ranking that minimises the total Kendall-tau distance
    /// (number of pairwise disagreements) to the input rankings. For `n <= 8` the
    /// optimum is found exactly by enumerating all `n!` permutations. For larger
    /// `n` an exhaustive search is intractable, so a documented local-search
    /// heuristic is used instead: starting from the Borda consensus, adjacent items
    /// are repeatedly swapped whenever the swap strictly reduces the total
    /// Kendall-tau cost, until no improving adjacent swap remains (a local Kemeny
    /// optimum). The result is therefore exact for small `n` and an approximation
    /// for large `n`.
    pub fn kemeny_optimal(rankings: &[Vec<usize>]) -> SklResult<Vec<usize>> {
        let n = validate_rankings(rankings)?;
        if n == 0 {
            return Ok(Vec::new());
        }

        // Pairwise preference matrix: prefer[i][j] = number of rankings placing i before j.
        let prefer = pairwise_preference_matrix(rankings, n);

        const EXACT_LIMIT: usize = 8;
        if n <= EXACT_LIMIT {
            Self::kemeny_exact(&prefer, n)
        } else {
            Self::kemeny_local_search(&prefer, n, rankings)
        }
    }

    /// Exhaustive Kemeny solver for small `n` (`n!` permutations).
    fn kemeny_exact(prefer: &[Vec<usize>], n: usize) -> SklResult<Vec<usize>> {
        let mut items: Vec<usize> = (0..n).collect();
        let mut best_order = items.clone();
        let mut best_cost = kemeny_cost(&best_order, prefer);

        // Heap's algorithm to iterate all permutations in place.
        let mut c = vec![0usize; n];
        // Evaluate the initial permutation already done above.
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    items.swap(0, i);
                } else {
                    items.swap(c[i], i);
                }
                let cost = kemeny_cost(&items, prefer);
                if cost < best_cost
                    || (cost == best_cost && lexicographically_smaller(&items, &best_order))
                {
                    best_cost = cost;
                    best_order = items.clone();
                }
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }

        Ok(best_order)
    }

    /// Local Kemeny search for large `n` using improving adjacent swaps.
    fn kemeny_local_search(
        prefer: &[Vec<usize>],
        n: usize,
        rankings: &[Vec<usize>],
    ) -> SklResult<Vec<usize>> {
        // Seed with the Borda consensus, which is a strong starting point.
        let mut order = Self::borda_count(rankings)?;

        let mut improved = true;
        while improved {
            improved = false;
            for pos in 0..n.saturating_sub(1) {
                let a = order[pos];
                let b = order[pos + 1];
                // Swapping adjacent a,b only changes the relative order of (a, b).
                // Cost contribution of a-before-b is the count preferring b-before-a.
                let cost_ab = prefer[b][a];
                let cost_ba = prefer[a][b];
                if cost_ba < cost_ab {
                    order.swap(pos, pos + 1);
                    improved = true;
                }
            }
        }

        Ok(order)
    }
}

/// Consensus measurement across a set of rankings.
#[derive(Debug, Clone)]
pub struct ConsensusMetrics;

impl ConsensusMetrics {
    /// Compute the consensus (agreement) of a set of rankings.
    ///
    /// The consensus is defined as the mean Kendall-tau correlation coefficient
    /// over all distinct pairs of rankings. For a pair, the coefficient is
    /// `(concordant - discordant) / (n choose 2)`, which equals `+1` for identical
    /// rankings and `-1` for exactly reversed rankings. The returned value lies in
    /// `[-1, 1]`; `1.0` denotes perfect agreement. A single ranking trivially has a
    /// consensus of `1.0`.
    pub fn compute_consensus(rankings: &[Vec<usize>]) -> SklResult<f64> {
        let n = validate_rankings(rankings)?;
        let m = rankings.len();
        if m == 1 || n <= 1 {
            return Ok(1.0);
        }

        // Position lookup: rank_of[r][item] = position of `item` in ranking `r`.
        let mut rank_of = vec![vec![0usize; n]; m];
        for (r, ranking) in rankings.iter().enumerate() {
            for (position, &item) in ranking.iter().enumerate() {
                rank_of[r][item] = position;
            }
        }

        let total_pairs = (n * (n - 1) / 2) as f64;
        let mut tau_sum = 0.0;
        let mut comparisons = 0usize;

        for r1 in 0..m {
            for r2 in (r1 + 1)..m {
                let mut concordant: i64 = 0;
                let mut discordant: i64 = 0;
                for i in 0..n {
                    for j in (i + 1)..n {
                        let a = rank_of[r1][i].cmp(&rank_of[r1][j]);
                        let b = rank_of[r2][i].cmp(&rank_of[r2][j]);
                        if a == b {
                            concordant += 1;
                        } else {
                            discordant += 1;
                        }
                    }
                }
                tau_sum += (concordant - discordant) as f64 / total_pairs;
                comparisons += 1;
            }
        }

        Ok(tau_sum / comparisons as f64)
    }
}

/// Multi-criteria decision analysis over a set of alternatives.
#[derive(Debug, Clone)]
pub struct MultiCriteriaEvaluation;

impl MultiCriteriaEvaluation {
    /// Evaluate alternatives using the TOPSIS method.
    ///
    /// `criteria_scores` is laid out as one inner vector per criterion, each of
    /// length `a` (the number of alternatives); `criteria_scores[c][i]` is the raw
    /// score of alternative `i` on criterion `c`. `criteria_weights` provides one
    /// weight per criterion. All criteria are treated as benefit criteria (larger
    /// is better).
    ///
    /// TOPSIS proceeds by: vector-normalising each criterion column, scaling by the
    /// normalised criterion weights, determining the positive- and negative-ideal
    /// solutions, and scoring each alternative by its relative closeness
    /// `d_neg / (d_pos + d_neg)`. The returned vector has one closeness score in
    /// `[0, 1]` per alternative; higher is better. Returns an error on empty input,
    /// inconsistent criterion lengths, or invalid weights.
    pub fn evaluate_multi_criteria(
        criteria_scores: &[Vec<f64>],
        criteria_weights: &[f64],
    ) -> SklResult<Vec<f64>> {
        let n_criteria = criteria_scores.len();
        if n_criteria == 0 {
            return Err(SklearsError::InvalidInput(
                "evaluate_multi_criteria requires at least one criterion".to_string(),
            ));
        }
        if criteria_weights.len() != n_criteria {
            return Err(SklearsError::InvalidInput(format!(
                "evaluate_multi_criteria: number of weights ({}) must equal number of criteria ({})",
                criteria_weights.len(),
                n_criteria
            )));
        }

        let n_alternatives = criteria_scores[0].len();
        if n_alternatives == 0 {
            return Err(SklearsError::InvalidInput(
                "evaluate_multi_criteria requires at least one alternative".to_string(),
            ));
        }
        for (c, column) in criteria_scores.iter().enumerate() {
            if column.len() != n_alternatives {
                return Err(SklearsError::InvalidInput(format!(
                    "evaluate_multi_criteria: criterion {} has {} alternatives, expected {}",
                    c,
                    column.len(),
                    n_alternatives
                )));
            }
        }

        // Normalise the weights to sum to one (also validates them).
        let mut weight_sum = 0.0;
        for &w in criteria_weights {
            if !w.is_finite() || w < 0.0 {
                return Err(SklearsError::InvalidInput(
                    "evaluate_multi_criteria: weights must be finite and non-negative".to_string(),
                ));
            }
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "evaluate_multi_criteria: sum of weights must be positive".to_string(),
            ));
        }
        let norm_weights: Vec<f64> = criteria_weights.iter().map(|&w| w / weight_sum).collect();

        // Vector normalisation per criterion, then apply the criterion weight.
        // weighted[c][i] = (score / column_norm) * weight.
        let mut weighted = vec![vec![0.0f64; n_alternatives]; n_criteria];
        for c in 0..n_criteria {
            let column_norm = criteria_scores[c]
                .iter()
                .map(|&v| v * v)
                .sum::<f64>()
                .sqrt();
            for i in 0..n_alternatives {
                let normalised = if column_norm > 0.0 {
                    criteria_scores[c][i] / column_norm
                } else {
                    // A constant-zero criterion contributes nothing; keep it at zero.
                    0.0
                };
                weighted[c][i] = normalised * norm_weights[c];
            }
        }

        // Positive- and negative-ideal solutions per criterion (benefit criteria).
        let mut ideal_best = vec![0.0f64; n_criteria];
        let mut ideal_worst = vec![0.0f64; n_criteria];
        for c in 0..n_criteria {
            ideal_best[c] = weighted[c]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            ideal_worst[c] = weighted[c].iter().copied().fold(f64::INFINITY, f64::min);
        }

        // Relative closeness of each alternative.
        let mut closeness = vec![0.0f64; n_alternatives];
        for i in 0..n_alternatives {
            let mut d_pos = 0.0;
            let mut d_neg = 0.0;
            for c in 0..n_criteria {
                let v = weighted[c][i];
                d_pos += (v - ideal_best[c]).powi(2);
                d_neg += (v - ideal_worst[c]).powi(2);
            }
            d_pos = d_pos.sqrt();
            d_neg = d_neg.sqrt();
            let denom = d_pos + d_neg;
            closeness[i] = if denom > 0.0 {
                d_neg / denom
            } else {
                // All alternatives identical on every criterion -> tie at 1.0.
                1.0
            };
        }

        Ok(closeness)
    }
}

/// Validate that `rankings` is non-empty and every ranking is a permutation of
/// `{0, ..., n-1}` for a common `n`. Returns the item count `n` on success.
fn validate_rankings(rankings: &[Vec<usize>]) -> SklResult<usize> {
    if rankings.is_empty() {
        return Err(SklearsError::InvalidInput(
            "rank aggregation requires at least one ranking".to_string(),
        ));
    }

    let n = rankings[0].len();
    for (idx, ranking) in rankings.iter().enumerate() {
        if ranking.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "ranking {} has length {} but expected {}",
                idx,
                ranking.len(),
                n
            )));
        }
        let mut seen = vec![false; n];
        for &item in ranking {
            if item >= n {
                return Err(SklearsError::InvalidInput(format!(
                    "ranking {idx} contains item index {item} out of range 0..{n}"
                )));
            }
            if seen[item] {
                return Err(SklearsError::InvalidInput(format!(
                    "ranking {idx} contains duplicate item index {item}"
                )));
            }
            seen[item] = true;
        }
    }

    Ok(n)
}

/// Build the pairwise preference matrix where `prefer[i][j]` counts how many
/// rankings place item `i` strictly before item `j`.
fn pairwise_preference_matrix(rankings: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut prefer = vec![vec![0usize; n]; n];
    for ranking in rankings {
        let mut rank_of = vec![0usize; n];
        for (position, &item) in ranking.iter().enumerate() {
            rank_of[item] = position;
        }
        for i in 0..n {
            for j in 0..n {
                if i != j && rank_of[i] < rank_of[j] {
                    prefer[i][j] += 1;
                }
            }
        }
    }
    prefer
}

/// Kendall-tau cost of `order` against the pairwise preference matrix: for every
/// ordered pair placed `i` before `j` in `order`, the cost is the number of input
/// rankings that disagree (i.e. that prefer `j` before `i`).
fn kemeny_cost(order: &[usize], prefer: &[Vec<usize>]) -> usize {
    let n = order.len();
    let mut cost = 0usize;
    for a in 0..n {
        for b in (a + 1)..n {
            let i = order[a];
            let j = order[b];
            // `i` is before `j` in `order`; disagreements are rankings preferring `j` before `i`.
            cost += prefer[j][i];
        }
    }
    cost
}

/// Lexicographic comparison used to break exact-Kemeny ties deterministically.
fn lexicographically_smaller(candidate: &[usize], current_best: &[usize]) -> bool {
    for (c, b) in candidate.iter().zip(current_best.iter()) {
        match c.cmp(b) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_average_basic() {
        // values 1,2,3 with weights 1,1,2 -> (1+2+6)/4 = 2.25
        let result =
            WeightedAveraging::weighted_average(&[1.0, 2.0, 3.0], &[1.0, 1.0, 2.0]).unwrap();
        assert!((result - 2.25).abs() < 1e-12);
    }

    #[test]
    fn weighted_average_normalises_weights() {
        // Scaling all weights must not change the result.
        let a = WeightedAveraging::weighted_average(&[4.0, 8.0], &[1.0, 3.0]).unwrap();
        let b = WeightedAveraging::weighted_average(&[4.0, 8.0], &[10.0, 30.0]).unwrap();
        // (4*1 + 8*3) / 4 = 28/4 = 7.0
        assert!((a - 7.0).abs() < 1e-12);
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn weighted_average_rejects_zero_weights() {
        assert!(WeightedAveraging::weighted_average(&[1.0, 2.0], &[0.0, 0.0]).is_err());
        assert!(WeightedAveraging::weighted_average(&[1.0, 2.0], &[-1.0, 1.0]).is_err());
        assert!(WeightedAveraging::weighted_average(&[], &[]).is_err());
        assert!(WeightedAveraging::weighted_average(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn aggregate_metrics_mean_and_weighted() {
        let mean = MetricsAggregator::aggregate_metrics(&[2.0, 4.0, 6.0], None).unwrap();
        assert!((mean - 4.0).abs() < 1e-12);

        let weighted =
            MetricsAggregator::aggregate_metrics(&[2.0, 4.0, 6.0], Some(&[1.0, 0.0, 0.0])).unwrap();
        assert!((weighted - 2.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_with_strategies() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!(
            (MetricsAggregator::aggregate_with(&v, AggregationStrategy::Median, None).unwrap()
                - 2.5)
                .abs()
                < 1e-12
        );
        assert!(
            (MetricsAggregator::aggregate_with(&v, AggregationStrategy::Min, None).unwrap() - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            (MetricsAggregator::aggregate_with(&v, AggregationStrategy::Max, None).unwrap() - 4.0)
                .abs()
                < 1e-12
        );
        assert!(
            MetricsAggregator::aggregate_with(&v, AggregationStrategy::Weighted, None).is_err()
        );
        assert!(MetricsAggregator::aggregate_with(&[], AggregationStrategy::Mean, None).is_err());
    }

    #[test]
    fn median_odd_and_even() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!((median(&[3.0, 1.0, 2.0, 4.0]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn borda_count_hand_computed() {
        // Three rankers over 3 items (0,1,2). Convention: position = rank, rank 0 best.
        // r1: 0,1,2  -> scores 0:3, 1:2, 2:1
        // r2: 1,0,2  -> scores 1:3, 0:2, 2:1
        // r3: 0,2,1  -> scores 0:3, 2:2, 1:1
        // Totals: 0: 3+2+3 = 8, 1: 2+3+1 = 6, 2: 1+1+2 = 4
        // Consensus order by descending score: [0, 1, 2]
        let rankings = vec![vec![0, 1, 2], vec![1, 0, 2], vec![0, 2, 1]];
        let consensus = RankAggregation::borda_count(&rankings).unwrap();
        assert_eq!(consensus, vec![0, 1, 2]);
    }

    #[test]
    fn borda_count_tie_breaks_by_index() {
        // Two rankers exactly reversed -> every item has identical Borda score.
        // Deterministic tie-break by ascending index gives [0,1,2].
        let rankings = vec![vec![0, 1, 2], vec![2, 1, 0]];
        let consensus = RankAggregation::borda_count(&rankings).unwrap();
        assert_eq!(consensus, vec![0, 1, 2]);
    }

    #[test]
    fn borda_count_rejects_bad_input() {
        assert!(RankAggregation::borda_count(&[]).is_err());
        // Not a permutation (duplicate).
        assert!(RankAggregation::borda_count(&[vec![0, 0, 1]]).is_err());
        // Out of range.
        assert!(RankAggregation::borda_count(&[vec![0, 1, 5]]).is_err());
        // Mismatched lengths.
        assert!(RankAggregation::borda_count(&[vec![0, 1, 2], vec![0, 1]]).is_err());
    }

    #[test]
    fn kemeny_optimal_three_items_exact() {
        // Classic example: rankings A>B>C, B>C>A, C>A>B form a Condorcet cycle.
        // Items A=0,B=1,C=2.
        // r1: 0,1,2 ; r2: 1,2,0 ; r3: 2,0,1
        // Every ordering has the same total Kendall-tau cost (3), so the exact
        // solver returns the lexicographically smallest optimum [0,1,2].
        let rankings = vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];
        let consensus = RankAggregation::kemeny_optimal(&rankings).unwrap();
        assert_eq!(consensus, vec![0, 1, 2]);
    }

    #[test]
    fn kemeny_optimal_clear_majority() {
        // Strong majority for order 0 < 1 < 2.
        // r1: 0,1,2 ; r2: 0,1,2 ; r3: 0,2,1
        // Majority pairwise: 0<1 (3-0), 0<2 (3-0), 1<2 (2-1). Optimal = [0,1,2].
        let rankings = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 2, 1]];
        let consensus = RankAggregation::kemeny_optimal(&rankings).unwrap();
        assert_eq!(consensus, vec![0, 1, 2]);
    }

    #[test]
    fn kemeny_optimal_recovers_unanimous() {
        // Unanimous rankings: the optimum is exactly that ranking, cost 0.
        let rankings = vec![vec![2, 0, 3, 1], vec![2, 0, 3, 1], vec![2, 0, 3, 1]];
        let consensus = RankAggregation::kemeny_optimal(&rankings).unwrap();
        assert_eq!(consensus, vec![2, 0, 3, 1]);
        // Cost against its own preference matrix must be zero.
        let prefer = pairwise_preference_matrix(&rankings, 4);
        assert_eq!(kemeny_cost(&consensus, &prefer), 0);
    }

    #[test]
    fn kemeny_local_search_large_n_unanimous() {
        // n = 10 > exact limit: the local search must still recover a unanimous order.
        let truth: Vec<usize> = vec![3, 7, 1, 9, 0, 5, 2, 8, 4, 6];
        let rankings = vec![truth.clone(), truth.clone(), truth.clone()];
        let consensus = RankAggregation::kemeny_optimal(&rankings).unwrap();
        assert_eq!(consensus, truth);
    }

    #[test]
    fn compute_consensus_identical_is_one() {
        let rankings = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        let c = ConsensusMetrics::compute_consensus(&rankings).unwrap();
        assert!((c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compute_consensus_reversed_is_minus_one() {
        let rankings = vec![vec![0, 1, 2, 3], vec![3, 2, 1, 0]];
        let c = ConsensusMetrics::compute_consensus(&rankings).unwrap();
        assert!((c + 1.0).abs() < 1e-12);
    }

    #[test]
    fn compute_consensus_single_ranking() {
        let c = ConsensusMetrics::compute_consensus(&[vec![0, 1, 2]]).unwrap();
        assert!((c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compute_consensus_one_swap() {
        // Two rankings differing by exactly one adjacent swap over 4 items.
        // n choose 2 = 6 pairs; 5 concordant, 1 discordant -> tau = 4/6 = 0.6667.
        let rankings = vec![vec![0, 1, 2, 3], vec![1, 0, 2, 3]];
        let c = ConsensusMetrics::compute_consensus(&rankings).unwrap();
        assert!((c - (4.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn topsis_prefers_dominant_alternative() {
        // Two criteria, three alternatives. Alternative 0 dominates on both.
        // criterion 0 scores: [10, 5, 1]; criterion 1 scores: [10, 6, 2]
        let criteria = vec![vec![10.0, 5.0, 1.0], vec![10.0, 6.0, 2.0]];
        let weights = vec![0.5, 0.5];
        let scores = MultiCriteriaEvaluation::evaluate_multi_criteria(&criteria, &weights).unwrap();
        assert_eq!(scores.len(), 3);
        // The dominant alternative is the positive ideal -> closeness 1.0.
        assert!((scores[0] - 1.0).abs() < 1e-9);
        // The dominated alternative is the negative ideal -> closeness 0.0.
        assert!(scores[2].abs() < 1e-9);
        // Ordering is strictly decreasing.
        assert!(scores[0] > scores[1] && scores[1] > scores[2]);
    }

    #[test]
    fn topsis_validates_inputs() {
        assert!(MultiCriteriaEvaluation::evaluate_multi_criteria(&[], &[]).is_err());
        // Mismatched weight count.
        assert!(
            MultiCriteriaEvaluation::evaluate_multi_criteria(&[vec![1.0, 2.0]], &[1.0, 1.0])
                .is_err()
        );
        // Inconsistent criterion lengths.
        assert!(MultiCriteriaEvaluation::evaluate_multi_criteria(
            &[vec![1.0, 2.0], vec![1.0]],
            &[1.0, 1.0]
        )
        .is_err());
        // Zero weights.
        assert!(
            MultiCriteriaEvaluation::evaluate_multi_criteria(&[vec![1.0, 2.0]], &[0.0]).is_err()
        );
    }
}
