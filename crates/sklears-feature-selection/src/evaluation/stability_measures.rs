//! Stability measures for feature selection consistency
//!
//! This module implements comprehensive stability measures to evaluate feature selection consistency
//! across different data subsets, random initializations, and parameter settings.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::Array2;
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StabilityError {
    #[error("Feature sets must have the same length")]
    FeatureSetLengthMismatch,
    #[error("Invalid feature index: {0}")]
    InvalidFeatureIndex(usize),
    #[error("Insufficient data for stability analysis")]
    InsufficientData,
}

/// Jaccard similarity coefficient for feature set overlap
#[derive(Debug, Clone)]
pub struct JaccardSimilarity;

impl JaccardSimilarity {
    /// Compute Jaccard similarity between two feature sets
    pub fn compute(set1: &[usize], set2: &[usize]) -> Result<f64> {
        let s1: HashSet<_> = set1.iter().collect();
        let s2: HashSet<_> = set2.iter().collect();

        let intersection = s1.intersection(&s2).count() as f64;
        let union = s1.union(&s2).count() as f64;

        if union == 0.0 {
            return Ok(1.0); // Both sets are empty
        }

        Ok(intersection / union)
    }

    /// Compute average Jaccard similarity for multiple feature sets
    pub fn average_similarity(feature_sets: &[Vec<usize>]) -> Result<f64> {
        if feature_sets.len() < 2 {
            return Err(SklearsError::FitError(
                "At least two feature sets required".to_string(),
            ));
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..feature_sets.len() {
            for j in (i + 1)..feature_sets.len() {
                total_similarity += Self::compute(&feature_sets[i], &feature_sets[j])?;
                comparisons += 1;
            }
        }

        Ok(total_similarity / comparisons as f64)
    }
}

/// Dice similarity coefficient (Sørensen–Dice coefficient)
#[derive(Debug, Clone)]
pub struct DiceSimilarity;

impl DiceSimilarity {
    /// Compute Dice similarity between two feature sets
    pub fn compute(set1: &[usize], set2: &[usize]) -> Result<f64> {
        let s1: HashSet<_> = set1.iter().collect();
        let s2: HashSet<_> = set2.iter().collect();

        let intersection = s1.intersection(&s2).count() as f64;
        let total_size = (s1.len() + s2.len()) as f64;

        if total_size == 0.0 {
            return Ok(1.0); // Both sets are empty
        }

        Ok(2.0 * intersection / total_size)
    }

    /// Compute average Dice similarity for multiple feature sets
    pub fn average_similarity(feature_sets: &[Vec<usize>]) -> Result<f64> {
        if feature_sets.len() < 2 {
            return Err(SklearsError::FitError(
                "At least two feature sets required".to_string(),
            ));
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..feature_sets.len() {
            for j in (i + 1)..feature_sets.len() {
                total_similarity += Self::compute(&feature_sets[i], &feature_sets[j])?;
                comparisons += 1;
            }
        }

        Ok(total_similarity / comparisons as f64)
    }
}

/// Overlap coefficient for asymmetric similarity measurement
#[derive(Debug, Clone)]
pub struct OverlapCoefficient;

impl OverlapCoefficient {
    /// Compute overlap coefficient between two feature sets
    pub fn compute(set1: &[usize], set2: &[usize]) -> Result<f64> {
        let s1: HashSet<_> = set1.iter().collect();
        let s2: HashSet<_> = set2.iter().collect();

        let intersection = s1.intersection(&s2).count() as f64;
        let min_size = std::cmp::min(s1.len(), s2.len()) as f64;

        if min_size == 0.0 {
            return Ok(1.0); // At least one set is empty
        }

        Ok(intersection / min_size)
    }

    /// Compute average overlap coefficient for multiple feature sets
    pub fn average_coefficient(feature_sets: &[Vec<usize>]) -> Result<f64> {
        if feature_sets.len() < 2 {
            return Err(SklearsError::FitError(
                "At least two feature sets required".to_string(),
            ));
        }

        let mut total_coefficient = 0.0;
        let mut comparisons = 0;

        for i in 0..feature_sets.len() {
            for j in (i + 1)..feature_sets.len() {
                total_coefficient += Self::compute(&feature_sets[i], &feature_sets[j])?;
                comparisons += 1;
            }
        }

        Ok(total_coefficient / comparisons as f64)
    }
}

/// Kuncheva's consistency index for stability measurement
#[derive(Debug, Clone)]
pub struct ConsistencyIndex;

impl ConsistencyIndex {
    /// Compute Kuncheva's consistency index between two feature sets
    ///
    /// The consistency index accounts for the probability of randomly selecting
    /// the same features and provides a normalized stability measure.
    pub fn compute(set1: &[usize], set2: &[usize], total_features: usize) -> Result<f64> {
        if total_features == 0 {
            return Err(SklearsError::FitError(
                "Total features must be positive".to_string(),
            ));
        }

        let s1: HashSet<_> = set1.iter().collect();
        let s2: HashSet<_> = set2.iter().collect();

        let k1 = s1.len() as f64;
        let k2 = s2.len() as f64;
        let r = s1.intersection(&s2).count() as f64;
        let n = total_features as f64;

        let expected_overlap = (k1 * k2) / n;
        let numerator = r - expected_overlap;
        let denominator = std::cmp::min(s1.len(), s2.len()) as f64 - expected_overlap;

        if denominator.abs() < 1e-10 {
            return Ok(1.0); // Perfect consistency when denominator approaches 0
        }

        Ok(numerator / denominator)
    }

    /// Compute average Kuncheva's consistency index for multiple feature sets
    pub fn average_consistency(feature_sets: &[Vec<usize>], total_features: usize) -> Result<f64> {
        if feature_sets.len() < 2 {
            return Err(SklearsError::FitError(
                "At least two feature sets required".to_string(),
            ));
        }

        let mut total_consistency = 0.0;
        let mut comparisons = 0;

        for i in 0..feature_sets.len() {
            for j in (i + 1)..feature_sets.len() {
                total_consistency +=
                    Self::compute(&feature_sets[i], &feature_sets[j], total_features)?;
                comparisons += 1;
            }
        }

        Ok(total_consistency / comparisons as f64)
    }
}

/// Comprehensive stability measures aggregator
#[derive(Debug, Clone)]
pub struct StabilityMeasures {
    pub jaccard_similarity: f64,
    pub dice_similarity: f64,
    pub overlap_coefficient: f64,
    pub consistency_index: f64,
    pub pairwise_stability: f64,
    pub relative_stability_index: f64,
}

impl StabilityMeasures {
    /// Compute all stability measures for a collection of feature sets
    pub fn compute(feature_sets: &[Vec<usize>], total_features: usize) -> Result<Self> {
        if feature_sets.len() < 2 {
            return Err(SklearsError::FitError(
                "At least two feature sets required".to_string(),
            ));
        }

        let jaccard_similarity = JaccardSimilarity::average_similarity(feature_sets)?;
        let dice_similarity = DiceSimilarity::average_similarity(feature_sets)?;
        let overlap_coefficient = OverlapCoefficient::average_coefficient(feature_sets)?;
        let consistency_index =
            ConsistencyIndex::average_consistency(feature_sets, total_features)?;
        let pairwise_stability = Self::compute_pairwise_stability(feature_sets)?;
        let relative_stability_index =
            Self::compute_relative_stability_index(feature_sets, total_features)?;

        Ok(Self {
            jaccard_similarity,
            dice_similarity,
            overlap_coefficient,
            consistency_index,
            pairwise_stability,
            relative_stability_index,
        })
    }

    /// Compute pairwise stability across all feature set pairs
    fn compute_pairwise_stability(feature_sets: &[Vec<usize>]) -> Result<f64> {
        if feature_sets.len() < 2 {
            return Ok(0.0);
        }

        let n_sets = feature_sets.len();
        let mut stability_matrix = Array2::zeros((n_sets, n_sets));

        for i in 0..n_sets {
            for j in 0..n_sets {
                if i != j {
                    stability_matrix[[i, j]] =
                        JaccardSimilarity::compute(&feature_sets[i], &feature_sets[j])?;
                } else {
                    stability_matrix[[i, j]] = 1.0;
                }
            }
        }

        // Compute average off-diagonal elements
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..n_sets {
            for j in 0..n_sets {
                if i != j {
                    sum += stability_matrix[[i, j]];
                    count += 1;
                }
            }
        }

        Ok(sum / count as f64)
    }

    /// Compute relative stability index accounting for feature set sizes
    fn compute_relative_stability_index(
        feature_sets: &[Vec<usize>],
        total_features: usize,
    ) -> Result<f64> {
        let average_set_size =
            feature_sets.iter().map(|s| s.len()).sum::<usize>() as f64 / feature_sets.len() as f64;
        let max_possible_overlap = average_set_size.min(total_features as f64 - average_set_size);

        if max_possible_overlap < 1e-10 {
            return Ok(0.0);
        }

        let observed_overlap =
            JaccardSimilarity::average_similarity(feature_sets)? * average_set_size;

        Ok(observed_overlap / max_possible_overlap)
    }

    /// Generate a stability report with interpretation
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Feature Selection Stability Report ===\n\n");

        report.push_str(&format!(
            "Jaccard Similarity: {:.4}\n",
            self.jaccard_similarity
        ));
        report.push_str(&format!("  Interpretation: {}\n", self.interpret_jaccard()));

        report.push_str(&format!("\nDice Similarity: {:.4}\n", self.dice_similarity));
        report.push_str(&format!("  Interpretation: {}\n", self.interpret_dice()));

        report.push_str(&format!(
            "\nOverlap Coefficient: {:.4}\n",
            self.overlap_coefficient
        ));
        report.push_str(&format!("  Interpretation: {}\n", self.interpret_overlap()));

        report.push_str(&format!(
            "\nConsistency Index: {:.4}\n",
            self.consistency_index
        ));
        report.push_str(&format!(
            "  Interpretation: {}\n",
            self.interpret_consistency()
        ));

        report.push_str(&format!(
            "\nPairwise Stability: {:.4}\n",
            self.pairwise_stability
        ));
        report.push_str(&format!(
            "  Interpretation: {}\n",
            self.interpret_pairwise()
        ));

        report.push_str(&format!(
            "\nRelative Stability Index: {:.4}\n",
            self.relative_stability_index
        ));
        report.push_str(&format!(
            "  Interpretation: {}\n",
            self.interpret_relative()
        ));

        report.push_str(&format!(
            "\nOverall Assessment: {}\n",
            self.overall_assessment()
        ));

        report
    }

    fn interpret_jaccard(&self) -> &'static str {
        match self.jaccard_similarity {
            x if x >= 0.8 => "Excellent stability - feature sets are highly consistent",
            x if x >= 0.6 => "Good stability - reasonable consistency in feature selection",
            x if x >= 0.4 => "Moderate stability - some variability in feature selection",
            x if x >= 0.2 => "Poor stability - high variability in feature selection",
            _ => "Very poor stability - feature selection is highly inconsistent",
        }
    }

    fn interpret_dice(&self) -> &'static str {
        match self.dice_similarity {
            x if x >= 0.8 => "Excellent overlap between feature sets",
            x if x >= 0.6 => "Good overlap between feature sets",
            x if x >= 0.4 => "Moderate overlap between feature sets",
            x if x >= 0.2 => "Poor overlap between feature sets",
            _ => "Very poor overlap between feature sets",
        }
    }

    fn interpret_overlap(&self) -> &'static str {
        match self.overlap_coefficient {
            x if x >= 0.8 => "High subset consistency - smaller sets are well contained",
            x if x >= 0.6 => "Good subset consistency",
            x if x >= 0.4 => "Moderate subset consistency",
            x if x >= 0.2 => "Poor subset consistency",
            _ => "Very poor subset consistency",
        }
    }

    fn interpret_consistency(&self) -> &'static str {
        match self.consistency_index {
            x if x >= 0.8 => "Excellent consistency above random chance",
            x if x >= 0.6 => "Good consistency above random chance",
            x if x >= 0.4 => "Moderate consistency above random chance",
            x if x >= 0.0 => "Some consistency above random chance",
            _ => "Consistency below random chance - concerning",
        }
    }

    fn interpret_pairwise(&self) -> &'static str {
        match self.pairwise_stability {
            x if x >= 0.8 => "Excellent pairwise stability across all comparisons",
            x if x >= 0.6 => "Good pairwise stability",
            x if x >= 0.4 => "Moderate pairwise stability",
            x if x >= 0.2 => "Poor pairwise stability",
            _ => "Very poor pairwise stability",
        }
    }

    fn interpret_relative(&self) -> &'static str {
        match self.relative_stability_index {
            x if x >= 0.8 => "Excellent relative stability considering set sizes",
            x if x >= 0.6 => "Good relative stability",
            x if x >= 0.4 => "Moderate relative stability",
            x if x >= 0.2 => "Poor relative stability",
            _ => "Very poor relative stability",
        }
    }

    fn overall_assessment(&self) -> &'static str {
        let average = (self.jaccard_similarity
            + self.dice_similarity
            + self.overlap_coefficient
            + self.consistency_index
            + self.pairwise_stability
            + self.relative_stability_index)
            / 6.0;

        match average {
            x if x >= 0.8 => "EXCELLENT: Feature selection is highly stable and reliable",
            x if x >= 0.6 => "GOOD: Feature selection shows good stability",
            x if x >= 0.4 => {
                "MODERATE: Feature selection has moderate stability - consider parameter tuning"
            }
            x if x >= 0.2 => "POOR: Feature selection is unstable - review methodology",
            _ => "CRITICAL: Feature selection is highly unstable - major concerns",
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let set1 = vec![0, 1, 2];
        let set2 = vec![1, 2, 3];
        let similarity = JaccardSimilarity::compute(&set1, &set2).unwrap();
        assert!((similarity - 0.5).abs() < 1e-10);

        // Test identical sets
        let similarity = JaccardSimilarity::compute(&set1, &set1).unwrap();
        assert!((similarity - 1.0).abs() < 1e-10);

        // Test empty sets
        let empty1: Vec<usize> = vec![];
        let empty2: Vec<usize> = vec![];
        let similarity = JaccardSimilarity::compute(&empty1, &empty2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dice_similarity() {
        let set1 = vec![0, 1, 2];
        let set2 = vec![1, 2, 3];
        let similarity = DiceSimilarity::compute(&set1, &set2).unwrap();
        assert!((similarity - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_coefficient() {
        let set1 = vec![0, 1, 2];
        let set2 = vec![1, 2, 3, 4];
        let coefficient = OverlapCoefficient::compute(&set1, &set2).unwrap();
        assert!((coefficient - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_consistency_index() {
        let set1 = vec![0, 1, 2];
        let set2 = vec![1, 2, 3];
        let total_features = 10;
        let consistency = ConsistencyIndex::compute(&set1, &set2, total_features).unwrap();
        assert!(consistency > -1.0 && consistency <= 1.0);
    }

    #[test]
    fn test_stability_measures() {
        let feature_sets = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 2, 4]];
        let total_features = 10;

        let measures = StabilityMeasures::compute(&feature_sets, total_features).unwrap();

        assert!(measures.jaccard_similarity >= 0.0 && measures.jaccard_similarity <= 1.0);
        assert!(measures.dice_similarity >= 0.0 && measures.dice_similarity <= 1.0);
        assert!(measures.overlap_coefficient >= 0.0 && measures.overlap_coefficient <= 1.0);
        assert!(measures.pairwise_stability >= 0.0 && measures.pairwise_stability <= 1.0);

        let report = measures.report();
        assert!(report.contains("Stability Report"));
        assert!(report.contains("Overall Assessment"));
    }

    #[test]
    fn test_average_similarities() {
        let feature_sets = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4]];

        let jaccard = JaccardSimilarity::average_similarity(&feature_sets).unwrap();
        let dice = DiceSimilarity::average_similarity(&feature_sets).unwrap();
        let overlap = OverlapCoefficient::average_coefficient(&feature_sets).unwrap();

        assert!(jaccard >= 0.0 && jaccard <= 1.0);
        assert!(dice >= 0.0 && dice <= 1.0);
        assert!(overlap >= 0.0 && overlap <= 1.0);
    }
}
