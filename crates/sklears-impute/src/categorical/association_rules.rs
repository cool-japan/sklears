//! Association Rule Imputer
#![allow(non_snake_case)]
#![allow(dead_code)]
//!
//! Imputation using association rules discovered from categorical data.
//! Missing values are imputed based on frequent patterns and strong rules.
//!
//! # Note
//!
//! Not implemented in v0.1.0. `fit` and `transform` return
//! `Err(SklearsError::NotImplemented)`. Planned for v0.2.0.

use scirs2_core::ndarray::{Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Association Rule Imputer
///
/// Imputation using association rules discovered from categorical data.
/// Missing values are imputed based on frequent patterns and strong rules.
///
/// # Parameters
///
/// * `min_support` - Minimum support threshold for frequent itemsets
/// * `min_confidence` - Minimum confidence threshold for association rules
/// * `max_itemset_size` - Maximum size of itemsets to consider
/// * `missing_values` - The placeholder for missing values
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::AssociationRuleImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 2.0, 3.0], [1.0, f64::NAN, 3.0]];
///
/// let imputer = AssociationRuleImputer::new()
///     .min_support(0.3)
///     .min_confidence(0.7);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AssociationRuleImputer<S = Untrained> {
    state: S,
    min_support: f64,
    min_confidence: f64,
    max_itemset_size: usize,
    missing_values: f64,
    random_state: Option<u64>,
}

/// Trained state for AssociationRuleImputer
#[derive(Debug, Clone)]
pub struct AssociationRuleImputerTrained {
    rules_: Vec<AssociationRule>,
    frequent_values_: HashMap<usize, f64>,
    n_features_in_: usize,
}

/// Item in an itemset (feature, value)
pub type Item = (usize, f64);

/// Collection of items forming an itemset
pub type Itemset = Vec<Item>;

/// Simple association rule structure
#[derive(Debug, Clone)]
pub struct AssociationRule {
    antecedent: Vec<(usize, f64)>,
    consequent: (usize, f64),
    confidence: f64,
    support: f64,
}

impl AssociationRuleImputer<Untrained> {
    /// Create a new AssociationRuleImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            min_support: 0.1,
            min_confidence: 0.6,
            max_itemset_size: 3,
            missing_values: f64::NAN,
            random_state: None,
        }
    }

    /// Set the minimum support threshold
    pub fn min_support(mut self, min_support: f64) -> Self {
        self.min_support = min_support.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum confidence threshold
    pub fn min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum itemset size
    pub fn max_itemset_size(mut self, max_itemset_size: usize) -> Self {
        self.max_itemset_size = max_itemset_size;
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

impl Default for AssociationRuleImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AssociationRuleImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AssociationRuleImputer<Untrained> {
    type Fitted = AssociationRuleImputer<AssociationRuleImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input X must have at least one sample and one feature".to_string(),
            ));
        }

        // Use total number of rows as denominator for support (as per spec)
        let n_total = n_samples as f64;

        // ---------------------------------------------------------------
        // Step 1: Build transactions — each row contributes (feature, value)
        // pairs for every non-missing entry.
        // ---------------------------------------------------------------
        let mut transactions: Vec<Vec<Item>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut tx: Vec<Item> = Vec::new();
            for j in 0..n_features {
                let v = X[[i, j]];
                if !self.is_missing(v) {
                    tx.push((j, v));
                }
            }
            transactions.push(tx);
        }

        // ---------------------------------------------------------------
        // Step 2: Frequent 1-itemsets
        // ---------------------------------------------------------------
        // Count occurrences of each (feature, value) pair.
        // We use ordered_float-like stable key via integer bits for HashMap.
        // Two values are "equal" if |a-b| < 1e-9.
        // We canonicalise by rounding to 9 decimal places for the key.
        let canon = |v: f64| -> i64 {
            // Multiply by 1e9, round, cast to i64 — gives stable hash key.
            (v * 1_000_000_000.0).round() as i64
        };

        // HashMap<(feature, canon_value), (count, representative_f64)>
        let mut item_counts: HashMap<(usize, i64), (usize, f64)> = HashMap::new();
        for tx in &transactions {
            for &(feat, val) in tx {
                let key = (feat, canon(val));
                item_counts
                    .entry(key)
                    .and_modify(|e| e.0 += 1)
                    .or_insert((1, val));
            }
        }

        // Keep only items whose support >= min_support.
        // frequent_1: Vec<Item> — the canonical (feature, value) pairs.
        let mut frequent_1: Vec<Item> = Vec::new();
        let mut support_1: HashMap<(usize, i64), f64> = HashMap::new();
        for ((feat, cval), (cnt, val)) in &item_counts {
            let sup = *cnt as f64 / n_total;
            if sup >= self.min_support {
                frequent_1.push((*feat, *val));
                support_1.insert((*feat, *cval), sup);
            }
        }
        // Sort for deterministic join order
        frequent_1.sort_by(|a, b| a.0.cmp(&b.0).then(canon(a.1).cmp(&canon(b.1))));

        // ---------------------------------------------------------------
        // Step 3: Apriori — extend to k-itemsets for k = 2..=max_itemset_size
        // ---------------------------------------------------------------
        // We store frequent itemsets as sorted Vec<Item> together with support.
        // Level 0 = frequent 1-itemsets.
        let mut current_level: Vec<(Itemset, f64)> = frequent_1
            .iter()
            .map(|&item| {
                let sup = support_1[&(item.0, canon(item.1))];
                (vec![item], sup)
            })
            .collect();

        // All frequent itemsets (size >= 2) collected for rule generation.
        let mut frequent_large: Vec<(Itemset, f64)> = Vec::new();

        let max_k = self.max_itemset_size.max(2); // ensure we go at least to 2
        for _k in 2..=max_k {
            if current_level.is_empty() {
                break;
            }

            // Apriori candidate generation: join pairs sharing first k-2 items.
            #[allow(clippy::type_complexity)]
            let mut candidate_counts: HashMap<Vec<(usize, i64)>, (usize, Vec<Item>)> =
                HashMap::new();

            let level_len = current_level.len();
            for i in 0..level_len {
                for j in (i + 1)..level_len {
                    let (ref iset_i, _) = current_level[i];
                    let (ref iset_j, _) = current_level[j];
                    let k_prev = iset_i.len(); // = k-1

                    // The two itemsets must share the first k-2 items.
                    let prefix_len = if k_prev >= 2 { k_prev - 1 } else { 0 };
                    let prefix_match = iset_i[..prefix_len]
                        .iter()
                        .zip(iset_j[..prefix_len].iter())
                        .all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() < 1e-9);

                    if !prefix_match {
                        continue;
                    }

                    // Last items must differ (and iset_j's last item > iset_i's last item
                    // to avoid duplicates — we compare by (feature, canon_val)).
                    let last_i = iset_i[k_prev - 1];
                    let last_j = iset_j[k_prev - 1];
                    let li_key = (last_i.0, canon(last_i.1));
                    let lj_key = (last_j.0, canon(last_j.1));
                    if li_key >= lj_key {
                        continue;
                    }

                    // Build candidate by merging
                    let mut candidate: Vec<Item> = iset_i.clone();
                    candidate.push(last_j);

                    // Apriori pruning: every (k-1)-subset must be frequent.
                    // Build canonical key for candidate.
                    let cand_key: Vec<(usize, i64)> =
                        candidate.iter().map(|&(f, v)| (f, canon(v))).collect();

                    candidate_counts.entry(cand_key).or_insert((0, candidate));
                }
            }

            if candidate_counts.is_empty() {
                break;
            }

            // Count support for each candidate by scanning transactions.
            #[allow(clippy::type_complexity)]
            let candidate_support: Vec<(Vec<(usize, i64)>, (usize, Vec<Item>))> =
                candidate_counts.into_iter().collect();

            // Counting pass over transactions.
            let mut counts: Vec<usize> = vec![0; candidate_support.len()];
            for tx in &transactions {
                // Build a quick lookup set for this transaction.
                let tx_set: HashMap<(usize, i64), ()> =
                    tx.iter().map(|&(f, v)| ((f, canon(v)), ())).collect();
                for (ci, (ckey, _)) in candidate_support.iter().enumerate() {
                    if ckey.iter().all(|k| tx_set.contains_key(k)) {
                        counts[ci] += 1;
                    }
                }
            }

            // Keep frequent candidates, advance level.
            let mut next_level: Vec<(Itemset, f64)> = Vec::new();
            for (ci, (_, (_, itemset))) in candidate_support.iter().enumerate() {
                let sup = counts[ci] as f64 / n_total;
                if sup >= self.min_support {
                    next_level.push((itemset.clone(), sup));
                    frequent_large.push((itemset.clone(), sup));
                }
            }

            current_level = next_level;
        }

        // ---------------------------------------------------------------
        // Step 4: Generate association rules from frequent k-itemsets (k>=2)
        // ---------------------------------------------------------------
        // For each frequent itemset with k >= 2, for each item as consequent
        // (we take the item with the highest feature index as consequent,
        // per "last in sorted order"), compute confidence.
        // antecedent = itemset \ {consequent}.
        let mut rules: Vec<AssociationRule> = Vec::new();

        for (itemset, item_support) in &frequent_large {
            let k = itemset.len();
            if k < 2 {
                continue;
            }
            // Try each item as the consequent.
            for ci in 0..k {
                let consequent = itemset[ci];
                let antecedent: Vec<Item> = itemset
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx != ci)
                    .map(|(_, &item)| item)
                    .collect();

                // Look up support of antecedent.
                // For 1-item antecedents use support_1; for larger we need to
                // search frequent_large (or current_level snapshots).
                // We'll scan frequent_large + frequent_1 for a match.
                let ant_support = if antecedent.len() == 1 {
                    let a = antecedent[0];
                    support_1.get(&(a.0, canon(a.1))).copied()
                } else {
                    // Find in frequent_large
                    let ant_key: Vec<(usize, i64)> =
                        antecedent.iter().map(|&(f, v)| (f, canon(v))).collect();
                    frequent_large
                        .iter()
                        .find(|(iset, _)| {
                            if iset.len() != antecedent.len() {
                                return false;
                            }
                            let iset_key: Vec<(usize, i64)> =
                                iset.iter().map(|&(f, v)| (f, canon(v))).collect();
                            iset_key == ant_key
                        })
                        .map(|(_, s)| *s)
                };

                if let Some(ant_sup) = ant_support {
                    if ant_sup > 0.0 {
                        let confidence = item_support / ant_sup;
                        if confidence >= self.min_confidence {
                            rules.push(AssociationRule {
                                antecedent,
                                consequent,
                                confidence,
                                support: *item_support,
                            });
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Step 5: Compute fallback — most frequent value per feature.
        // ---------------------------------------------------------------
        let mut frequent_values: HashMap<usize, f64> = HashMap::new();
        for feat in 0..n_features {
            let mut best_count = 0usize;
            let mut best_val = 0.0f64;
            for ((f, _cval), (cnt, val)) in &item_counts {
                if *f == feat && *cnt > best_count {
                    best_count = *cnt;
                    best_val = *val;
                }
            }
            if best_count > 0 {
                frequent_values.insert(feat, best_val);
            }
        }

        Ok(AssociationRuleImputer {
            state: AssociationRuleImputerTrained {
                rules_: rules,
                frequent_values_: frequent_values,
                n_features_in_: n_features,
            },
            min_support: self.min_support,
            min_confidence: self.min_confidence,
            max_itemset_size: self.max_itemset_size,
            missing_values: self.missing_values,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for AssociationRuleImputer<AssociationRuleImputerTrained>
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

        let canon = |v: f64| -> i64 { (v * 1_000_000_000.0).round() as i64 };
        let items_equal = |a: f64, b: f64| -> bool { (a - b).abs() < 1e-9 };

        let mut X_out = X.clone();

        for i in 0..n_samples {
            // Collect observed items for this row.
            let observed: Vec<Item> = (0..n_features)
                .filter(|&j| !self.is_missing(X[[i, j]]))
                .map(|j| (j, X[[i, j]]))
                .collect();

            // Build a quick lookup: (feature, canon_val) -> present
            let obs_set: HashMap<(usize, i64), ()> =
                observed.iter().map(|&(f, v)| ((f, canon(v)), ())).collect();

            // Identify missing features.
            let missing_features: Vec<usize> = (0..n_features)
                .filter(|&j| self.is_missing(X[[i, j]]))
                .collect();

            for feat in missing_features {
                // Find the best rule whose consequent is (feat, *) and whose
                // antecedent is entirely contained in observed items.
                let best = self
                    .state
                    .rules_
                    .iter()
                    .filter(|rule| {
                        // Consequent must target `feat`
                        rule.consequent.0 == feat
                            // Antecedent must not contain `feat` itself
                            && rule.antecedent.iter().all(|&(f, _)| f != feat)
                            // Every antecedent item must appear in observed
                            && rule.antecedent.iter().all(|&(f, v)| {
                                obs_set.contains_key(&(f, canon(v)))
                                    || observed
                                        .iter()
                                        .any(|&(of, ov)| of == f && items_equal(ov, v))
                            })
                    })
                    .max_by(|a, b| {
                        a.confidence
                            .partial_cmp(&b.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                let imputed_val = if let Some(rule) = best {
                    rule.consequent.1
                } else {
                    // Fallback: most frequent value seen during fit
                    match self.state.frequent_values_.get(&feat) {
                        Some(&v) => v,
                        None => 0.0,
                    }
                };

                X_out[[i, feat]] = imputed_val;
            }
        }

        Ok(X_out)
    }
}

impl AssociationRuleImputer<AssociationRuleImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}
