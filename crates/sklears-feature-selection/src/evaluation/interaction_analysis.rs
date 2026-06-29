//! Feature interaction analysis for understanding feature relationships
//!
//! This module implements real, data-driven feature-interaction measures based on
//! information theory. All quantities are estimated from a feature matrix and a
//! target vector using histogram (equal-width binning) density estimators.
//!
//! The central quantity is the **interaction information** of a feature pair with
//! the target,
//!
//! ```text
//! II(X_i; X_j; Y) = I(X_i, X_j ; Y) - I(X_i ; Y) - I(X_j ; Y)
//! ```
//!
//! where `I(X_i, X_j ; Y)` is the mutual information between the *joint* variable
//! `(X_i, X_j)` and the target `Y`. A positive value indicates **synergy** (the two
//! features jointly carry more information about the target than the sum of their
//! individual contributions); a negative value indicates **redundancy** (the
//! features share information about the target). All mutual-information terms are
//! computed in nats (natural logarithm).
//!
//! Because these estimators require the underlying data, the public methods take a
//! feature matrix `features` (shape `n_samples x n_features`, columns are features)
//! and a target vector `target` (length `n_samples`), together with the indices of
//! the features to analyse. Continuous variables are discretised into a fixed number
//! of equal-width bins; this is a standard, documented MI estimator and the only
//! approximation involved.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Default number of equal-width bins used to discretise continuous variables.
const DEFAULT_BINS: usize = 10;

/// Feature interaction analysis producing an aggregate interaction measure.
#[derive(Debug, Clone)]
pub struct FeatureInteractionAnalysis;

impl FeatureInteractionAnalysis {
    /// Analyse the overall interaction strength among a set of features.
    ///
    /// Returns the mean magnitude of the pairwise interaction information
    /// `|II(X_i; X_j; Y)|` over all distinct feature pairs in `feature_indices`.
    /// This summarises how strongly the selected features interact (synergistically
    /// or redundantly) with respect to the target, irrespective of sign. A value of
    /// `0.0` means the features carry independent information about the target. When
    /// fewer than two features are supplied there are no pairs and the result is
    /// `0.0`.
    ///
    /// `features` has shape `n_samples x n_features` (columns are features) and
    /// `target` has length `n_samples`. Continuous variables are discretised into
    /// `DEFAULT_BINS` equal-width bins.
    pub fn analyze_interactions(
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> SklResult<f64> {
        let pairwise = PairwiseInteractions::compute_pairwise(features, target, feature_indices)?;
        if pairwise.is_empty() {
            return Ok(0.0);
        }
        let sum: f64 = pairwise.iter().map(|&(_, _, v)| v.abs()).sum();
        Ok(sum / pairwise.len() as f64)
    }
}

/// Pairwise feature interaction analysis.
#[derive(Debug, Clone)]
pub struct PairwiseInteractions;

impl PairwiseInteractions {
    /// Compute the pairwise interaction information of every distinct feature pair.
    ///
    /// For each pair `(i, j)` of feature indices, returns the tuple
    /// `(i, j, II(X_i; X_j; Y))` where `II` is the interaction information defined in
    /// the module documentation. Positive values indicate synergy, negative values
    /// indicate redundancy. The pairs are returned in lexicographic order of
    /// `(i, j)`.
    ///
    /// `features` has shape `n_samples x n_features`; `target` has length
    /// `n_samples`. Returns an error if any index is out of bounds, the data has no
    /// samples, or the feature/target lengths are inconsistent.
    pub fn compute_pairwise(
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> SklResult<Vec<(usize, usize, f64)>> {
        let n_samples = validate_data(features, target, feature_indices)?;
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "compute_pairwise requires at least one sample".to_string(),
            ));
        }

        // Pre-discretise the target and every requested feature once.
        let target_binned = discretize(target, DEFAULT_BINS);
        let mut feature_binned: Vec<(usize, Vec<usize>)> =
            Vec::with_capacity(feature_indices.len());
        for &idx in feature_indices {
            let column = features.column(idx);
            feature_binned.push((idx, discretize(&column, DEFAULT_BINS)));
        }

        let mut result = Vec::new();
        for a in 0..feature_binned.len() {
            for b in (a + 1)..feature_binned.len() {
                let (idx_a, ref bins_a) = feature_binned[a];
                let (idx_b, ref bins_b) = feature_binned[b];
                let ii = interaction_information(bins_a, bins_b, &target_binned, DEFAULT_BINS);
                let (lo, hi) = if idx_a <= idx_b {
                    (idx_a, idx_b)
                } else {
                    (idx_b, idx_a)
                };
                result.push((lo, hi, ii));
            }
        }
        result.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));
        Ok(result)
    }
}

/// Higher-order (three-way) interaction analysis.
#[derive(Debug, Clone)]
pub struct HigherOrderInteractions;

impl HigherOrderInteractions {
    /// Compute the mean magnitude of three-way interaction information.
    ///
    /// For each distinct triple of features `(i, j, k)` the three-way interaction
    /// information with the target is estimated as
    ///
    /// ```text
    /// II(X_i; X_j; X_k; Y) = I(X_i, X_j, X_k ; Y)
    ///                        - I(X_i, X_j ; Y) - I(X_i, X_k ; Y) - I(X_j, X_k ; Y)
    ///                        + I(X_i ; Y) + I(X_j ; Y) + I(X_k ; Y)
    /// ```
    ///
    /// which is the inclusion-exclusion form of the interaction information. The
    /// returned scalar is the mean of `|II|` over all triples; it is `0.0` when
    /// fewer than three features are supplied. Mutual-information terms use the same
    /// histogram estimator as the rest of the module.
    pub fn compute_higher_order(
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
        feature_indices: &[usize],
    ) -> SklResult<f64> {
        let n_samples = validate_data(features, target, feature_indices)?;
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "compute_higher_order requires at least one sample".to_string(),
            ));
        }
        if feature_indices.len() < 3 {
            return Ok(0.0);
        }

        let target_binned = discretize(target, DEFAULT_BINS);
        let binned: Vec<Vec<usize>> = feature_indices
            .iter()
            .map(|&idx| discretize(&features.column(idx), DEFAULT_BINS))
            .collect();

        let n = binned.len();
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let mi_ijk = mutual_information_joint3(
                        &binned[i],
                        &binned[j],
                        &binned[k],
                        &target_binned,
                        DEFAULT_BINS,
                    );
                    let mi_ij = mutual_information_joint(
                        &binned[i],
                        &binned[j],
                        &target_binned,
                        DEFAULT_BINS,
                    );
                    let mi_ik = mutual_information_joint(
                        &binned[i],
                        &binned[k],
                        &target_binned,
                        DEFAULT_BINS,
                    );
                    let mi_jk = mutual_information_joint(
                        &binned[j],
                        &binned[k],
                        &target_binned,
                        DEFAULT_BINS,
                    );
                    let mi_i = mutual_information_single(&binned[i], &target_binned, DEFAULT_BINS);
                    let mi_j = mutual_information_single(&binned[j], &target_binned, DEFAULT_BINS);
                    let mi_k = mutual_information_single(&binned[k], &target_binned, DEFAULT_BINS);

                    let ii = mi_ijk - mi_ij - mi_ik - mi_jk + mi_i + mi_j + mi_k;
                    sum += ii.abs();
                    count += 1;
                }
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(sum / count as f64)
        }
    }
}

/// Interaction strength between a single pair of features.
#[derive(Debug, Clone)]
pub struct InteractionStrength;

impl InteractionStrength {
    /// Compute the interaction strength between two specific features.
    ///
    /// Returns the absolute interaction information `|II(X_i; X_j; Y)|` for the pair
    /// `(feature1, feature2)` — a non-negative measure of how strongly the two
    /// features interact with respect to the target (regardless of synergy vs
    /// redundancy). Returns an error if either index is out of bounds or the data is
    /// empty.
    pub fn compute_strength(
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
        feature1: usize,
        feature2: usize,
    ) -> SklResult<f64> {
        let n_samples = validate_data(features, target, &[feature1, feature2])?;
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "compute_strength requires at least one sample".to_string(),
            ));
        }
        let bins_a = discretize(&features.column(feature1), DEFAULT_BINS);
        let bins_b = discretize(&features.column(feature2), DEFAULT_BINS);
        let target_binned = discretize(target, DEFAULT_BINS);
        let ii = interaction_information(&bins_a, &bins_b, &target_binned, DEFAULT_BINS);
        Ok(ii.abs())
    }
}

/// Synergy detection over feature pairs.
#[derive(Debug, Clone)]
pub struct SynergyDetection;

impl SynergyDetection {
    /// Detect synergistic feature pairs.
    ///
    /// Returns the list of feature pairs (each as a two-element `Vec`) whose
    /// interaction information with the target is strictly positive by more than
    /// `threshold`, i.e. pairs that are genuinely synergistic (carry jointly more
    /// information about the target than individually). Pairs are returned in
    /// lexicographic order of their indices. A `threshold` of `0.0` returns all
    /// strictly-synergistic pairs.
    pub fn detect_synergy(
        features: &ArrayView2<f64>,
        target: &ArrayView1<f64>,
        feature_indices: &[usize],
        threshold: f64,
    ) -> SklResult<Vec<Vec<usize>>> {
        let pairwise = PairwiseInteractions::compute_pairwise(features, target, feature_indices)?;
        let synergistic = pairwise
            .into_iter()
            .filter(|&(_, _, ii)| ii > threshold)
            .map(|(i, j, _)| vec![i, j])
            .collect();
        Ok(synergistic)
    }
}

/// Validate the feature matrix, target, and the requested feature indices.
/// Returns the number of samples on success.
fn validate_data(
    features: &ArrayView2<f64>,
    target: &ArrayView1<f64>,
    feature_indices: &[usize],
) -> SklResult<usize> {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    if target.len() != n_samples {
        return Err(SklearsError::InvalidInput(format!(
            "target length ({}) must equal number of samples ({})",
            target.len(),
            n_samples
        )));
    }

    for &idx in feature_indices {
        if idx >= n_features {
            return Err(SklearsError::InvalidInput(format!(
                "feature index {idx} out of range 0..{n_features}"
            )));
        }
    }

    Ok(n_samples)
}

/// Discretise a continuous variable into `n_bins` equal-width bins.
///
/// Returns a bin index in `0..n_bins` for every sample. A constant variable maps
/// every sample to bin `0`, yielding zero mutual information with anything (the
/// information-theoretically correct outcome for a degenerate feature).
fn discretize(values: &ArrayView1<f64>, n_bins: usize) -> Vec<usize> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values.iter() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let range = max - min;
    if range <= 0.0 || n_bins == 0 {
        return vec![0usize; values.len()];
    }

    let width = range / n_bins as f64;
    values
        .iter()
        .map(|&v| {
            let mut bin = ((v - min) / width) as usize;
            if bin >= n_bins {
                bin = n_bins - 1; // the maximum value falls into the last bin
            }
            bin
        })
        .collect()
}

/// Shannon entropy (in nats) of a discrete sequence of bin labels.
fn entropy(labels: &[usize], n_bins: usize) -> f64 {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; n_bins.max(1)];
    for &l in labels {
        counts[l] += 1;
    }
    let n_f = n as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n_f;
            h -= p * p.ln();
        }
    }
    h
}

/// Joint entropy (in nats) of two discrete sequences of bin labels.
fn joint_entropy2(a: &[usize], b: &[usize], n_bins: usize) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let nb = n_bins.max(1);
    let mut counts = vec![0usize; nb * nb];
    for (&x, &y) in a.iter().zip(b.iter()) {
        counts[x * nb + y] += 1;
    }
    let n_f = n as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n_f;
            h -= p * p.ln();
        }
    }
    h
}

/// Joint entropy (in nats) of three discrete sequences of bin labels.
fn joint_entropy3(a: &[usize], b: &[usize], c: &[usize], n_bins: usize) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    use std::collections::HashMap;
    let mut counts: HashMap<(usize, usize, usize), usize> = HashMap::new();
    for ((&x, &y), &z) in a.iter().zip(b.iter()).zip(c.iter()) {
        *counts.entry((x, y, z)).or_insert(0) += 1;
    }
    let _ = n_bins; // bin count not needed when using a sparse map
    let n_f = n as f64;
    let mut h = 0.0;
    for &cnt in counts.values() {
        let p = cnt as f64 / n_f;
        h -= p * p.ln();
    }
    h
}

/// Joint entropy (in nats) of four discrete sequences of bin labels.
fn joint_entropy4(a: &[usize], b: &[usize], c: &[usize], d: &[usize]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    use std::collections::HashMap;
    let mut counts: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
    for (((&w, &x), &y), &z) in a.iter().zip(b.iter()).zip(c.iter()).zip(d.iter()) {
        *counts.entry((w, x, y, z)).or_insert(0) += 1;
    }
    let n_f = n as f64;
    let mut h = 0.0;
    for &cnt in counts.values() {
        let p = cnt as f64 / n_f;
        h -= p * p.ln();
    }
    h
}

/// Mutual information `I(X ; Y)` of a single discrete feature and the target,
/// computed as `H(X) + H(Y) - H(X, Y)` (in nats).
fn mutual_information_single(feature: &[usize], target: &[usize], n_bins: usize) -> f64 {
    let h_x = entropy(feature, n_bins);
    let h_y = entropy(target, n_bins);
    let h_xy = joint_entropy2(feature, target, n_bins);
    (h_x + h_y - h_xy).max(0.0)
}

/// Mutual information `I(X_i, X_j ; Y)` between a feature pair and the target,
/// computed as `H(X_i, X_j) + H(Y) - H(X_i, X_j, Y)` (in nats).
fn mutual_information_joint(
    feature_a: &[usize],
    feature_b: &[usize],
    target: &[usize],
    n_bins: usize,
) -> f64 {
    let h_ab = joint_entropy2(feature_a, feature_b, n_bins);
    let h_y = entropy(target, n_bins);
    let h_aby = joint_entropy3(feature_a, feature_b, target, n_bins);
    (h_ab + h_y - h_aby).max(0.0)
}

/// Mutual information `I(X_i, X_j, X_k ; Y)` between a feature triple and target,
/// computed as `H(X_i, X_j, X_k) + H(Y) - H(X_i, X_j, X_k, Y)` (in nats).
fn mutual_information_joint3(
    feature_a: &[usize],
    feature_b: &[usize],
    feature_c: &[usize],
    target: &[usize],
    n_bins: usize,
) -> f64 {
    let h_abc = joint_entropy3(feature_a, feature_b, feature_c, n_bins);
    let h_y = entropy(target, n_bins);
    let h_abcy = joint_entropy4(feature_a, feature_b, feature_c, target);
    (h_abc + h_y - h_abcy).max(0.0)
}

/// Interaction information `II(X_i; X_j; Y) = I(X_i, X_j ; Y) - I(X_i ; Y) - I(X_j ; Y)`.
fn interaction_information(
    feature_a: &[usize],
    feature_b: &[usize],
    target: &[usize],
    n_bins: usize,
) -> f64 {
    let mi_joint = mutual_information_joint(feature_a, feature_b, target, n_bins);
    let mi_a = mutual_information_single(feature_a, target, n_bins);
    let mi_b = mutual_information_single(feature_b, target, n_bins);
    mi_joint - mi_a - mi_b
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn discretize_constant_is_single_bin() {
        let v = array![3.0, 3.0, 3.0];
        let bins = discretize(&v.view(), DEFAULT_BINS);
        assert_eq!(bins, vec![0, 0, 0]);
    }

    #[test]
    fn discretize_spreads_values() {
        // Range [0, 10), 10 bins of width 1.0: value v -> bin floor(v), max into last bin.
        let v = array![0.0, 1.0, 9.0, 10.0];
        let bins = discretize(&v.view(), DEFAULT_BINS);
        assert_eq!(bins, vec![0, 1, 9, 9]);
    }

    #[test]
    fn entropy_uniform_two_bins() {
        // Balanced binary labels -> entropy ln(2).
        let labels = vec![0, 1, 0, 1];
        let h = entropy(&labels, 2);
        assert!((h - 2.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn mutual_information_identity_equals_entropy() {
        // I(X; X) = H(X). For balanced binary X, that is ln(2).
        let x = vec![0, 1, 0, 1, 0, 1];
        let mi = mutual_information_single(&x, &x, 2);
        assert!((mi - 2.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn mutual_information_independent_is_zero() {
        // X and Y constructed to be independent: every (x,y) combination equally likely.
        let x = vec![0, 0, 1, 1];
        let y = vec![0, 1, 0, 1];
        let mi = mutual_information_single(&x, &y, 2);
        assert!(mi.abs() < 1e-12);
    }

    #[test]
    fn xor_pair_is_synergistic() {
        // Classic XOR: Y = X1 XOR X2. Each feature alone is independent of Y
        // (zero MI), but together they fully determine Y. Interaction information
        // II = I(X1,X2;Y) - I(X1;Y) - I(X2;Y) = ln(2) - 0 - 0 = ln(2) > 0 (synergy).
        // Rows: (x1, x2) over the 4 combinations; columns 0 and 1 are the features.
        let features =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let target = array![0.0, 1.0, 1.0, 0.0]; // XOR
        let pairwise =
            PairwiseInteractions::compute_pairwise(&features.view(), &target.view(), &[0, 1])
                .unwrap();
        assert_eq!(pairwise.len(), 1);
        let (i, j, ii) = pairwise[0];
        assert_eq!((i, j), (0, 1));
        // Synergy should be approximately ln(2).
        assert!(ii > 0.0);
        assert!((ii - 2.0f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn redundant_pair_is_negative() {
        // Two identical copies of a feature that perfectly predicts Y.
        // I(X1,X2;Y) = I(X1;Y) = I(X2;Y) = ln(2), so
        // II = ln(2) - ln(2) - ln(2) = -ln(2) < 0 (redundancy).
        let features =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let target = array![0.0, 0.0, 1.0, 1.0];
        let strength =
            InteractionStrength::compute_strength(&features.view(), &target.view(), 0, 1).unwrap();
        // |II| = ln(2).
        assert!((strength - 2.0f64.ln()).abs() < 1e-9);

        // detect_synergy must NOT flag a redundant pair as synergistic.
        let syn = SynergyDetection::detect_synergy(&features.view(), &target.view(), &[0, 1], 0.0)
            .unwrap();
        assert!(syn.is_empty());
    }

    #[test]
    fn detect_synergy_flags_xor() {
        let features =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let target = array![0.0, 1.0, 1.0, 0.0];
        let syn = SynergyDetection::detect_synergy(&features.view(), &target.view(), &[0, 1], 0.0)
            .unwrap();
        assert_eq!(syn, vec![vec![0, 1]]);
    }

    #[test]
    fn analyze_interactions_single_feature_is_zero() {
        let features = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let target = array![0.0, 1.0, 0.0];
        let v = FeatureInteractionAnalysis::analyze_interactions(
            &features.view(),
            &target.view(),
            &[0],
        )
        .unwrap();
        assert!(v.abs() < 1e-12);
    }

    #[test]
    fn analyze_interactions_matches_pairwise_mean_magnitude() {
        let features =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let target = array![0.0, 1.0, 1.0, 0.0];
        let agg = FeatureInteractionAnalysis::analyze_interactions(
            &features.view(),
            &target.view(),
            &[0, 1],
        )
        .unwrap();
        // Single pair -> aggregate equals |II| of that pair == ln(2).
        assert!((agg - 2.0f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn higher_order_requires_three_features() {
        let features =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let target = array![0.0, 1.0, 1.0, 0.0];
        let v = HigherOrderInteractions::compute_higher_order(
            &features.view(),
            &target.view(),
            &[0, 1],
        )
        .unwrap();
        assert!(v.abs() < 1e-12);
    }

    #[test]
    fn higher_order_three_way_xor() {
        // Y = X1 XOR X2 XOR X3 over all 8 combinations. The three-way interaction
        // information is non-zero; we assert it is computed and finite/positive in
        // magnitude (this is a genuine 3-way synergy).
        let mut data = Vec::new();
        let mut target_vals = Vec::new();
        for x1 in 0..2usize {
            for x2 in 0..2usize {
                for x3 in 0..2usize {
                    data.push(x1 as f64);
                    data.push(x2 as f64);
                    data.push(x3 as f64);
                    target_vals.push(((x1 ^ x2 ^ x3) as f64).to_owned());
                }
            }
        }
        let features = Array2::from_shape_vec((8, 3), data).unwrap();
        let target = scirs2_core::ndarray::Array1::from_vec(target_vals);
        let v = HigherOrderInteractions::compute_higher_order(
            &features.view(),
            &target.view(),
            &[0, 1, 2],
        )
        .unwrap();
        assert!(v.is_finite());
        assert!(v > 0.0);
    }

    #[test]
    fn validation_rejects_bad_input() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let target = array![0.0, 1.0, 0.0];
        // Out-of-range feature index.
        assert!(
            PairwiseInteractions::compute_pairwise(&features.view(), &target.view(), &[0, 5])
                .is_err()
        );
        // Mismatched target length.
        let bad_target = array![0.0, 1.0];
        assert!(PairwiseInteractions::compute_pairwise(
            &features.view(),
            &bad_target.view(),
            &[0, 1]
        )
        .is_err());
    }
}
