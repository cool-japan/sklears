//! Comparative analysis tools for benchmarking and evaluation
//!
//! This module provides statistical tools for comparing different baseline models
//! and machine learning algorithms. It includes significance testing, effect size
//! computation, and various statistical comparison methods.
//!
//! The module includes:
//! - Statistical significance testing (t-test, Wilcoxon, permutation tests)
//! - Effect size computation (Cohen's d, Cliff's delta, etc.)
//! - Confidence interval comparisons
//! - Bayesian comparison methods
//! - Performance reporting utilities

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Statistical significance test types
#[derive(Debug, Clone, Copy)]
pub enum SignificanceTest {
    /// Two-sample t-test (assumes normality)
    TTest,
    /// Welch's t-test (unequal variances)
    WelchTTest,
    /// Wilcoxon rank-sum test (non-parametric)
    WilcoxonRankSum,
    /// Mann-Whitney U test (equivalent to Wilcoxon rank-sum)
    MannWhitneyU,
    /// Permutation test (non-parametric)
    PermutationTest { n_permutations: usize },
    /// Bootstrap test
    BootstrapTest { n_bootstrap: usize },
}

/// Effect size measures
#[derive(Debug, Clone, Copy)]
pub enum EffectSizeMeasure {
    /// Cohen's d (standardized mean difference)
    CohensD,
    /// Glass's delta (uses control group standard deviation)
    GlassDelta,
    /// Hedges' g (bias-corrected Cohen's d)
    HedgesG,
    /// Cliff's delta (non-parametric effect size)
    CliffsDelta,
    /// Common language effect size
    CommonLanguageEffect,
    /// Probability of superiority
    ProbabilityOfSuperiority,
}

/// Confidence interval types
#[derive(Debug, Clone, Copy)]
pub enum ConfidenceIntervalType {
    /// Normal approximation
    Normal,
    /// Bootstrap percentile method
    BootstrapPercentile { n_bootstrap: usize },
    /// Bootstrap bias-corrected and accelerated (BCa)
    BootstrapBCa { n_bootstrap: usize },
    /// Student's t-distribution
    TDistribution,
}

/// Results of a statistical significance test
#[derive(Debug, Clone)]
pub struct SignificanceTestResult {
    /// test_type
    pub test_type: SignificanceTest,
    /// statistic
    pub statistic: f64,
    /// p_value
    pub p_value: f64,
    /// effect_size
    pub effect_size: f64,
    /// effect_size_measure
    pub effect_size_measure: EffectSizeMeasure,
    /// confidence_interval
    pub confidence_interval: (f64, f64),
    /// sample_sizes
    pub sample_sizes: (usize, usize),
    /// is_significant
    pub is_significant: bool,
    /// alpha_level
    pub alpha_level: f64,
}

/// Results of effect size computation
#[derive(Debug, Clone)]
pub struct EffectSizeResult {
    /// measure
    pub measure: EffectSizeMeasure,
    /// value
    pub value: f64,
    /// confidence_interval
    pub confidence_interval: (f64, f64),
    /// interpretation
    pub interpretation: EffectSizeInterpretation,
    /// sample_sizes
    pub sample_sizes: (usize, usize),
}

/// Effect size interpretation according to conventional thresholds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// Negligible
    Negligible,
    /// Small
    Small,
    /// Medium
    Medium,
    /// Large
    Large,
    /// VeryLarge
    VeryLarge,
}

/// Model comparison result
#[derive(Debug, Clone)]
pub struct ModelComparisonResult {
    /// model_names
    pub model_names: Vec<String>,
    /// performance_scores
    pub performance_scores: Vec<f64>,
    /// pairwise_comparisons
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    /// ranking
    pub ranking: Vec<usize>, // Indices of models sorted by performance
    /// best_model_index
    pub best_model_index: usize,
    /// statistical_summary
    pub statistical_summary: StatisticalSummary,
}

/// Pairwise comparison between two models
#[derive(Debug, Clone)]
pub struct PairwiseComparison {
    /// model_a_index
    pub model_a_index: usize,
    /// model_b_index
    pub model_b_index: usize,
    /// significance_test
    pub significance_test: SignificanceTestResult,
    /// bayes_factor
    pub bayes_factor: Option<f64>,
    /// practical_significance
    pub practical_significance: bool,
}

/// Statistical summary of model comparisons
#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    /// mean_scores
    pub mean_scores: Vec<f64>,
    /// std_scores
    pub std_scores: Vec<f64>,
    /// confidence_intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// overall_best_model
    pub overall_best_model: usize,
    /// significantly_different_pairs
    pub significantly_different_pairs: Vec<(usize, usize)>,
    /// effect_sizes
    pub effect_sizes: HashMap<(usize, usize), f64>,
}

/// Main comparative analysis engine
pub struct ComparativeAnalyzer {
    alpha_level: f64,
    random_state: Option<u64>,
    correction_method: MultipleComparisonCorrection,
}

/// Multiple comparison correction methods
#[derive(Debug, Clone, Copy)]
pub enum MultipleComparisonCorrection {
    None,
    /// Bonferroni
    Bonferroni,
    /// Holm
    Holm,
    /// BenjaminiHochberg
    BenjaminiHochberg,
    /// BenjaminiYekutieli
    BenjaminiYekutieli,
}

impl ComparativeAnalyzer {
    /// Create a new comparative analyzer
    pub fn new() -> Self {
        Self {
            alpha_level: 0.05,
            random_state: None,
            correction_method: MultipleComparisonCorrection::Holm,
        }
    }

    /// Set significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha_level = alpha;
        self
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set multiple comparison correction method
    pub fn with_correction(mut self, correction: MultipleComparisonCorrection) -> Self {
        self.correction_method = correction;
        self
    }

    /// Perform statistical significance test between two groups
    pub fn significance_test(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        test_type: SignificanceTest,
        effect_size_measure: EffectSizeMeasure,
    ) -> Result<SignificanceTestResult, SklearsError> {
        let statistic = self.compute_test_statistic(group_a, group_b, test_type)?;
        let p_value = self.compute_p_value(group_a, group_b, test_type, statistic)?;
        let effect_size = self.compute_effect_size(group_a, group_b, effect_size_measure)?;
        let confidence_interval = self.compute_confidence_interval(
            group_a,
            group_b,
            ConfidenceIntervalType::Normal,
            0.95,
        )?;

        let corrected_alpha = self.apply_correction(self.alpha_level, 1);
        let is_significant = p_value < corrected_alpha;

        Ok(SignificanceTestResult {
            test_type,
            statistic,
            p_value,
            effect_size,
            effect_size_measure,
            confidence_interval,
            sample_sizes: (group_a.len(), group_b.len()),
            is_significant,
            alpha_level: corrected_alpha,
        })
    }

    /// Compute effect size between two groups
    pub fn effect_size(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        measure: EffectSizeMeasure,
    ) -> Result<EffectSizeResult, SklearsError> {
        let value = self.compute_effect_size(group_a, group_b, measure)?;
        let confidence_interval =
            self.effect_size_confidence_interval(group_a, group_b, measure)?;
        let interpretation = self.interpret_effect_size(value, measure);

        Ok(EffectSizeResult {
            measure,
            value,
            confidence_interval,
            interpretation,
            sample_sizes: (group_a.len(), group_b.len()),
        })
    }

    /// Compare multiple models using cross-validation scores
    pub fn compare_models(
        &self,
        model_names: Vec<String>,
        cv_scores: Vec<Array1<f64>>,
    ) -> Result<ModelComparisonResult, SklearsError> {
        if model_names.len() != cv_scores.len() {
            return Err(SklearsError::InvalidInput(
                "Number of model names must match number of score arrays".to_string(),
            ));
        }

        let performance_scores: Vec<f64> = cv_scores
            .iter()
            .map(|scores| scores.mean().unwrap_or(0.0))
            .collect();

        // Rank models by performance
        let mut ranking: Vec<usize> = (0..model_names.len()).collect();
        ranking.sort_by(|&a, &b| {
            performance_scores[b]
                .partial_cmp(&performance_scores[a])
                .unwrap()
        });
        let best_model_index = ranking[0];

        // Pairwise comparisons
        let mut pairwise_comparisons = Vec::new();
        let n_models = model_names.len();
        let n_comparisons = n_models * (n_models - 1) / 2;

        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let significance_test = self.significance_test(
                    &cv_scores[i],
                    &cv_scores[j],
                    SignificanceTest::WilcoxonRankSum,
                    EffectSizeMeasure::CliffsDelta,
                )?;

                let bayes_factor = self.compute_bayes_factor(&cv_scores[i], &cv_scores[j])?;
                let practical_significance = significance_test.effect_size.abs() > 0.2; // Small effect size threshold

                pairwise_comparisons.push(PairwiseComparison {
                    model_a_index: i,
                    model_b_index: j,
                    significance_test,
                    bayes_factor: Some(bayes_factor),
                    practical_significance,
                });
            }
        }

        // Apply multiple comparison correction
        let corrected_comparisons =
            self.apply_multiple_comparison_correction(pairwise_comparisons, n_comparisons);

        // Statistical summary
        let statistical_summary =
            self.compute_statistical_summary(&cv_scores, &corrected_comparisons)?;

        Ok(ModelComparisonResult {
            model_names,
            performance_scores,
            pairwise_comparisons: corrected_comparisons,
            ranking,
            best_model_index,
            statistical_summary,
        })
    }

    fn compute_test_statistic(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        test_type: SignificanceTest,
    ) -> Result<f64, SklearsError> {
        match test_type {
            SignificanceTest::TTest | SignificanceTest::WelchTTest => self.compute_t_statistic(
                group_a,
                group_b,
                matches!(test_type, SignificanceTest::WelchTTest),
            ),
            SignificanceTest::WilcoxonRankSum | SignificanceTest::MannWhitneyU => {
                self.compute_rank_sum_statistic(group_a, group_b)
            }
            SignificanceTest::PermutationTest { n_permutations } => {
                self.compute_permutation_statistic(group_a, group_b, n_permutations)
            }
            SignificanceTest::BootstrapTest { n_bootstrap } => {
                self.compute_bootstrap_statistic(group_a, group_b, n_bootstrap)
            }
        }
    }

    fn compute_t_statistic(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        welch: bool,
    ) -> Result<f64, SklearsError> {
        let mean_a = group_a.mean().unwrap_or(0.0);
        let mean_b = group_b.mean().unwrap_or(0.0);
        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;

        let var_a = group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
        let var_b = group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

        let t_statistic = if welch {
            // Welch's t-test (unequal variances)
            let se = (var_a / n_a + var_b / n_b).sqrt();
            (mean_a - mean_b) / se
        } else {
            // Student's t-test (equal variances)
            let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
            let se = pooled_var.sqrt() * (1.0 / n_a + 1.0 / n_b).sqrt();
            (mean_a - mean_b) / se
        };

        Ok(t_statistic)
    }

    fn compute_rank_sum_statistic(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        let mut combined: Vec<(f64, usize)> = group_a.iter().map(|&x| (x, 0)).collect();
        combined.extend(group_b.iter().map(|&x| (x, 1)));
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut rank_sum_a = 0.0;
        for (rank, (_, group)) in combined.iter().enumerate() {
            if *group == 0 {
                rank_sum_a += (rank + 1) as f64;
            }
        }

        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;
        let expected_rank_sum = n_a * (n_a + n_b + 1.0) / 2.0;
        let variance = n_a * n_b * (n_a + n_b + 1.0) / 12.0;

        let z_statistic = (rank_sum_a - expected_rank_sum) / variance.sqrt();
        Ok(z_statistic)
    }

    fn compute_permutation_statistic(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        n_permutations: usize,
    ) -> Result<f64, SklearsError> {
        let observed_diff = group_a.mean().unwrap_or(0.0) - group_b.mean().unwrap_or(0.0);

        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::seed_from_u64(0)
        };

        let mut combined: Vec<f64> = group_a.iter().chain(group_b.iter()).cloned().collect();
        let n_a = group_a.len();

        let mut extreme_count = 0;
        for _ in 0..n_permutations {
            // Shuffle the combined data
            for i in (1..combined.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                combined.swap(i, j);
            }

            let perm_mean_a = combined[..n_a].iter().sum::<f64>() / n_a as f64;
            let perm_mean_b = combined[n_a..].iter().sum::<f64>() / (combined.len() - n_a) as f64;
            let perm_diff = perm_mean_a - perm_mean_b;

            if perm_diff.abs() >= observed_diff.abs() {
                extreme_count += 1;
            }
        }

        let p_value = extreme_count as f64 / n_permutations as f64;
        Ok(p_value) // Return p-value directly for permutation test
    }

    fn compute_bootstrap_statistic(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        n_bootstrap: usize,
    ) -> Result<f64, SklearsError> {
        let observed_diff = group_a.mean().unwrap_or(0.0) - group_b.mean().unwrap_or(0.0);

        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::seed_from_u64(0)
        };

        let mut bootstrap_diffs = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            // Bootstrap sample from each group
            let boot_a: Vec<f64> = (0..group_a.len())
                .map(|_| group_a[rng.gen_range(0..group_a.len())])
                .collect();
            let boot_b: Vec<f64> = (0..group_b.len())
                .map(|_| group_b[rng.gen_range(0..group_b.len())])
                .collect();

            let boot_mean_a = boot_a.iter().sum::<f64>() / boot_a.len() as f64;
            let boot_mean_b = boot_b.iter().sum::<f64>() / boot_b.len() as f64;
            bootstrap_diffs.push(boot_mean_a - boot_mean_b);
        }

        bootstrap_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let std_bootstrap = {
            let mean_bootstrap = bootstrap_diffs.iter().sum::<f64>() / bootstrap_diffs.len() as f64;
            let variance = bootstrap_diffs
                .iter()
                .map(|&x| (x - mean_bootstrap).powi(2))
                .sum::<f64>()
                / (bootstrap_diffs.len() - 1) as f64;
            variance.sqrt()
        };

        Ok(observed_diff / std_bootstrap)
    }

    fn compute_p_value(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        test_type: SignificanceTest,
        statistic: f64,
    ) -> Result<f64, SklearsError> {
        match test_type {
            SignificanceTest::TTest => {
                let df = group_a.len() + group_b.len() - 2;
                self.t_distribution_p_value(statistic, df)
            }
            SignificanceTest::WelchTTest => {
                let n_a = group_a.len() as f64;
                let n_b = group_b.len() as f64;
                let var_a = group_a
                    .iter()
                    .map(|&x| (x - group_a.mean().unwrap_or(0.0)).powi(2))
                    .sum::<f64>()
                    / (n_a - 1.0);
                let var_b = group_b
                    .iter()
                    .map(|&x| (x - group_b.mean().unwrap_or(0.0)).powi(2))
                    .sum::<f64>()
                    / (n_b - 1.0);

                // Welch-Satterthwaite equation for degrees of freedom
                let s_a2_n_a = var_a / n_a;
                let s_b2_n_b = var_b / n_b;
                let numerator = (s_a2_n_a + s_b2_n_b).powi(2);
                let denominator = s_a2_n_a.powi(2) / (n_a - 1.0) + s_b2_n_b.powi(2) / (n_b - 1.0);
                let df = (numerator / denominator).floor() as usize;

                self.t_distribution_p_value(statistic, df)
            }
            SignificanceTest::WilcoxonRankSum | SignificanceTest::MannWhitneyU => {
                self.normal_distribution_p_value(statistic)
            }
            SignificanceTest::PermutationTest { .. } => {
                Ok(statistic) // statistic is already the p-value for permutation test
            }
            SignificanceTest::BootstrapTest { .. } => self.normal_distribution_p_value(statistic),
        }
    }

    fn compute_effect_size(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        measure: EffectSizeMeasure,
    ) -> Result<f64, SklearsError> {
        match measure {
            EffectSizeMeasure::CohensD => self.cohens_d(group_a, group_b),
            EffectSizeMeasure::GlassDelta => self.glass_delta(group_a, group_b),
            EffectSizeMeasure::HedgesG => self.hedges_g(group_a, group_b),
            EffectSizeMeasure::CliffsDelta => self.cliffs_delta(group_a, group_b),
            EffectSizeMeasure::CommonLanguageEffect => {
                self.common_language_effect(group_a, group_b)
            }
            EffectSizeMeasure::ProbabilityOfSuperiority => {
                self.probability_of_superiority(group_a, group_b)
            }
        }
    }

    fn cohens_d(&self, group_a: &Array1<f64>, group_b: &Array1<f64>) -> Result<f64, SklearsError> {
        let mean_a = group_a.mean().unwrap_or(0.0);
        let mean_b = group_b.mean().unwrap_or(0.0);
        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;

        let var_a = group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
        let var_b = group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

        let pooled_std = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0)).sqrt();

        Ok((mean_a - mean_b) / pooled_std)
    }

    fn glass_delta(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        let mean_a = group_a.mean().unwrap_or(0.0);
        let mean_b = group_b.mean().unwrap_or(0.0);
        let n_b = group_b.len() as f64;

        let var_b = group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);
        let std_b = var_b.sqrt();

        Ok((mean_a - mean_b) / std_b)
    }

    fn hedges_g(&self, group_a: &Array1<f64>, group_b: &Array1<f64>) -> Result<f64, SklearsError> {
        let cohens_d = self.cohens_d(group_a, group_b)?;
        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;

        // Bias correction factor
        let df = n_a + n_b - 2.0;
        let correction = 1.0 - 3.0 / (4.0 * df - 1.0);

        Ok(cohens_d * correction)
    }

    fn cliffs_delta(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        let mut dominance_count = 0;
        let total_comparisons = group_a.len() * group_b.len();

        for &a in group_a.iter() {
            for &b in group_b.iter() {
                if a > b {
                    dominance_count += 1;
                } else if a < b {
                    dominance_count -= 1;
                }
                // Ties contribute 0
            }
        }

        Ok(dominance_count as f64 / total_comparisons as f64)
    }

    fn common_language_effect(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        let mut superiority_count = 0;
        let total_comparisons = group_a.len() * group_b.len();

        for &a in group_a.iter() {
            for &b in group_b.iter() {
                if a > b {
                    superiority_count += 1;
                }
            }
        }

        Ok(superiority_count as f64 / total_comparisons as f64)
    }

    fn probability_of_superiority(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        // Same as common language effect for continuous data
        self.common_language_effect(group_a, group_b)
    }

    fn compute_confidence_interval(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        ci_type: ConfidenceIntervalType,
        confidence_level: f64,
    ) -> Result<(f64, f64), SklearsError> {
        match ci_type {
            ConfidenceIntervalType::Normal => {
                let mean_diff = group_a.mean().unwrap_or(0.0) - group_b.mean().unwrap_or(0.0);
                let n_a = group_a.len() as f64;
                let n_b = group_b.len() as f64;
                let mean_a = group_a.mean().unwrap_or(0.0);
                let mean_b = group_b.mean().unwrap_or(0.0);

                let var_a =
                    group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
                let var_b =
                    group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);
                let se = (var_a / n_a + var_b / n_b).sqrt();

                let z_critical = self.normal_quantile((1.0 + confidence_level) / 2.0);
                let margin_of_error = z_critical * se;

                Ok((mean_diff - margin_of_error, mean_diff + margin_of_error))
            }
            ConfidenceIntervalType::TDistribution => {
                let mean_diff = group_a.mean().unwrap_or(0.0) - group_b.mean().unwrap_or(0.0);
                let n_a = group_a.len() as f64;
                let n_b = group_b.len() as f64;
                let df = (n_a + n_b - 2.0) as usize;
                let mean_a = group_a.mean().unwrap_or(0.0);
                let mean_b = group_b.mean().unwrap_or(0.0);

                let var_a =
                    group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
                let var_b =
                    group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);
                let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
                let se = pooled_var.sqrt() * (1.0 / n_a + 1.0 / n_b).sqrt();

                let t_critical = self.t_quantile((1.0 + confidence_level) / 2.0, df);
                let margin_of_error = t_critical * se;

                Ok((mean_diff - margin_of_error, mean_diff + margin_of_error))
            }
            _ => {
                // For bootstrap methods, use normal approximation as fallback
                self.compute_confidence_interval(
                    group_a,
                    group_b,
                    ConfidenceIntervalType::Normal,
                    confidence_level,
                )
            }
        }
    }

    fn effect_size_confidence_interval(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
        measure: EffectSizeMeasure,
    ) -> Result<(f64, f64), SklearsError> {
        // Simplified confidence interval for effect sizes
        // In practice, this would use bootstrap or analytical methods
        let effect_size = self.compute_effect_size(group_a, group_b, measure)?;
        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;

        // Rough approximation for demonstration
        let se = match measure {
            EffectSizeMeasure::CohensD | EffectSizeMeasure::HedgesG => {
                ((n_a + n_b) / (n_a * n_b) + effect_size.powi(2) / (2.0 * (n_a + n_b))).sqrt()
            }
            _ => 0.1, // Simplified for other measures
        };

        let z_critical = self.normal_quantile(0.975); // 95% CI
        let margin_of_error = z_critical * se;

        Ok((effect_size - margin_of_error, effect_size + margin_of_error))
    }

    fn interpret_effect_size(
        &self,
        value: f64,
        measure: EffectSizeMeasure,
    ) -> EffectSizeInterpretation {
        let abs_value = value.abs();

        match measure {
            EffectSizeMeasure::CohensD | EffectSizeMeasure::HedgesG => {
                if abs_value < 0.2 {
                    EffectSizeInterpretation::Negligible
                } else if abs_value < 0.5 {
                    EffectSizeInterpretation::Small
                } else if abs_value < 0.8 {
                    EffectSizeInterpretation::Medium
                } else if abs_value < 1.2 {
                    EffectSizeInterpretation::Large
                } else {
                    EffectSizeInterpretation::VeryLarge
                }
            }
            EffectSizeMeasure::CliffsDelta => {
                if abs_value < 0.147 {
                    EffectSizeInterpretation::Negligible
                } else if abs_value < 0.33 {
                    EffectSizeInterpretation::Small
                } else if abs_value < 0.474 {
                    EffectSizeInterpretation::Medium
                } else {
                    EffectSizeInterpretation::Large
                }
            }
            _ => {
                // Generic interpretation
                if abs_value < 0.1 {
                    EffectSizeInterpretation::Negligible
                } else if abs_value < 0.3 {
                    EffectSizeInterpretation::Small
                } else if abs_value < 0.5 {
                    EffectSizeInterpretation::Medium
                } else {
                    EffectSizeInterpretation::Large
                }
            }
        }
    }

    fn compute_bayes_factor(
        &self,
        group_a: &Array1<f64>,
        group_b: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        // Simplified Bayes factor calculation using t-test approximation
        let t_stat = self.compute_t_statistic(group_a, group_b, false)?;
        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;
        let df = n_a + n_b - 2.0;

        // BIC approximation for Bayes factor
        let bic_null = (n_a + n_b) * (1.0 + t_stat.powi(2) / df).ln();
        let bic_alt = bic_null - df.ln();
        let log_bf = (bic_null - bic_alt) / 2.0;

        Ok(log_bf.exp())
    }

    fn apply_correction(&self, alpha: f64, n_comparisons: usize) -> f64 {
        match self.correction_method {
            MultipleComparisonCorrection::None => alpha,
            MultipleComparisonCorrection::Bonferroni => alpha / n_comparisons as f64,
            MultipleComparisonCorrection::Holm => alpha, // Applied during ranking
            MultipleComparisonCorrection::BenjaminiHochberg => alpha, // Applied during ranking
            MultipleComparisonCorrection::BenjaminiYekutieli => alpha, // Applied during ranking
        }
    }

    fn apply_multiple_comparison_correction(
        &self,
        mut comparisons: Vec<PairwiseComparison>,
        n_comparisons: usize,
    ) -> Vec<PairwiseComparison> {
        match self.correction_method {
            MultipleComparisonCorrection::None | MultipleComparisonCorrection::Bonferroni => {
                // Already applied in individual tests
                comparisons
            }
            MultipleComparisonCorrection::Holm => {
                // Sort p-values and apply Holm correction
                comparisons.sort_by(|a, b| {
                    a.significance_test
                        .p_value
                        .partial_cmp(&b.significance_test.p_value)
                        .unwrap()
                });
                for (i, comparison) in comparisons.iter_mut().enumerate() {
                    let corrected_alpha = self.alpha_level / (n_comparisons - i) as f64;
                    comparison.significance_test.is_significant =
                        comparison.significance_test.p_value < corrected_alpha;
                    comparison.significance_test.alpha_level = corrected_alpha;
                }
                comparisons
            }
            MultipleComparisonCorrection::BenjaminiHochberg => {
                // FDR correction
                comparisons.sort_by(|a, b| {
                    a.significance_test
                        .p_value
                        .partial_cmp(&b.significance_test.p_value)
                        .unwrap()
                });
                for (i, comparison) in comparisons.iter_mut().enumerate() {
                    let corrected_alpha = self.alpha_level * (i + 1) as f64 / n_comparisons as f64;
                    comparison.significance_test.is_significant =
                        comparison.significance_test.p_value <= corrected_alpha;
                    comparison.significance_test.alpha_level = corrected_alpha;
                }
                comparisons
            }
            MultipleComparisonCorrection::BenjaminiYekutieli => {
                // FDR correction for dependent tests
                let c_factor: f64 = (1..=n_comparisons).map(|i| 1.0 / i as f64).sum();
                comparisons.sort_by(|a, b| {
                    a.significance_test
                        .p_value
                        .partial_cmp(&b.significance_test.p_value)
                        .unwrap()
                });
                for (i, comparison) in comparisons.iter_mut().enumerate() {
                    let corrected_alpha =
                        self.alpha_level * (i + 1) as f64 / (n_comparisons as f64 * c_factor);
                    comparison.significance_test.is_significant =
                        comparison.significance_test.p_value <= corrected_alpha;
                    comparison.significance_test.alpha_level = corrected_alpha;
                }
                comparisons
            }
        }
    }

    fn compute_statistical_summary(
        &self,
        cv_scores: &[Array1<f64>],
        comparisons: &[PairwiseComparison],
    ) -> Result<StatisticalSummary, SklearsError> {
        let mean_scores: Vec<f64> = cv_scores
            .iter()
            .map(|scores| scores.mean().unwrap_or(0.0))
            .collect();
        let std_scores: Vec<f64> = cv_scores
            .iter()
            .enumerate()
            .map(|(i, scores)| {
                let mean = mean_scores[i];
                let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (scores.len() - 1) as f64;
                variance.sqrt()
            })
            .collect();

        let confidence_intervals: Vec<(f64, f64)> = cv_scores
            .iter()
            .enumerate()
            .map(|(i, scores)| {
                let mean = mean_scores[i];
                let std = std_scores[i];
                let n = scores.len() as f64;
                let se = std / n.sqrt();
                let t_critical = self.t_quantile(0.975, scores.len() - 1);
                let margin = t_critical * se;
                (mean - margin, mean + margin)
            })
            .collect();

        let overall_best_model = mean_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let significantly_different_pairs: Vec<(usize, usize)> = comparisons
            .iter()
            .filter(|comp| comp.significance_test.is_significant)
            .map(|comp| (comp.model_a_index, comp.model_b_index))
            .collect();

        let effect_sizes: HashMap<(usize, usize), f64> = comparisons
            .iter()
            .map(|comp| {
                (
                    (comp.model_a_index, comp.model_b_index),
                    comp.significance_test.effect_size,
                )
            })
            .collect();

        Ok(StatisticalSummary {
            mean_scores,
            std_scores,
            confidence_intervals,
            overall_best_model,
            significantly_different_pairs,
            effect_sizes,
        })
    }

    // Helper functions for statistical distributions
    fn normal_quantile(&self, p: f64) -> f64 {
        // Beasley-Springer-Moro algorithm approximation
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383_577_518_672_69e2,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    }

    fn t_quantile(&self, p: f64, df: usize) -> f64 {
        // Simplified t-distribution quantile approximation
        let z = self.normal_quantile(p);
        let df_f = df as f64;

        if df >= 30 {
            // For large df, t-distribution approaches normal
            return z;
        }

        // Hill's approximation for t-distribution quantile
        let a = 4.0 * df_f;
        let c = (a + z * z) / a;
        let correction = 1.0 + (z * z + 1.0) / a;

        z * c.sqrt() * correction
    }

    fn t_distribution_p_value(&self, t: f64, df: usize) -> Result<f64, SklearsError> {
        // Simplified two-tailed p-value calculation
        // This would use a proper t-distribution CDF in practice
        let abs_t = t.abs();

        if df >= 30 {
            return self.normal_distribution_p_value(abs_t);
        }

        // Rough approximation for smaller df
        let p = if abs_t > 3.0 {
            0.001
        } else if abs_t > 2.0 {
            0.05
        } else if abs_t > 1.0 {
            0.3
        } else {
            0.6
        };

        Ok(p)
    }

    fn normal_distribution_p_value(&self, z: f64) -> Result<f64, SklearsError> {
        // Two-tailed p-value using normal distribution
        let abs_z = z.abs();

        // Approximate normal CDF complement
        let p = if abs_z > 6.0 {
            0.000000001
        } else {
            let t = 1.0 / (1.0 + 0.2316419 * abs_z);
            let d = 0.3989423 * (-abs_z * abs_z / 2.0).exp();
            d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        };

        Ok(2.0 * p) // Two-tailed
    }
}

impl Default for ComparativeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for generating comparison reports
pub struct ComparisonReporter;

impl ComparisonReporter {
    /// Generate a comprehensive comparison report
    pub fn generate_report(comparison: &ModelComparisonResult) -> String {
        let mut report = String::new();

        report.push_str("# Model Comparison Report\n\n");

        // Overall ranking
        report.push_str("## Overall Ranking\n");
        for (rank, &model_idx) in comparison.ranking.iter().enumerate() {
            let model_name = &comparison.model_names[model_idx];
            let score = comparison.performance_scores[model_idx];
            report.push_str(&format!(
                "{}. {} (Score: {:.4})\n",
                rank + 1,
                model_name,
                score
            ));
        }

        // Statistical summary
        report.push_str("\n## Statistical Summary\n");
        for (i, model_name) in comparison.model_names.iter().enumerate() {
            let mean = comparison.statistical_summary.mean_scores[i];
            let std = comparison.statistical_summary.std_scores[i];
            let (ci_low, ci_high) = comparison.statistical_summary.confidence_intervals[i];
            report.push_str(&format!(
                "- {}: Mean = {:.4} ± {:.4}, 95% CI = [{:.4}, {:.4}]\n",
                model_name, mean, std, ci_low, ci_high
            ));
        }

        // Pairwise comparisons
        report.push_str("\n## Significant Pairwise Differences\n");
        for pairwise_comp in &comparison.pairwise_comparisons {
            if pairwise_comp.significance_test.is_significant {
                let model_a = &comparison.model_names[pairwise_comp.model_a_index];
                let model_b = &comparison.model_names[pairwise_comp.model_b_index];
                let p_value = pairwise_comp.significance_test.p_value;
                let effect_size = pairwise_comp.significance_test.effect_size;

                report.push_str(&format!(
                    "- {} vs {}: p = {:.4}, Effect Size = {:.4}\n",
                    model_a, model_b, p_value, effect_size
                ));
            }
        }

        report
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_significance_test() {
        let analyzer = ComparativeAnalyzer::new();
        let group_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = analyzer
            .significance_test(
                &group_a,
                &group_b,
                SignificanceTest::TTest,
                EffectSizeMeasure::CohensD,
            )
            .unwrap();

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(!result.statistic.is_nan());
        assert!(!result.effect_size.is_nan());
    }

    #[test]
    fn test_effect_size_cohens_d() {
        let analyzer = ComparativeAnalyzer::new();
        let group_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = analyzer
            .effect_size(&group_a, &group_b, EffectSizeMeasure::CohensD)
            .unwrap();

        assert!(result.value < 0.0); // group_a has lower mean
                                     // Check the actual Cohen's d value - it should be around -1.26 for this data
        assert!(result.value.abs() > 0.5); // Should be at least medium effect
                                           // The effect size should be meaningful
        assert!(!matches!(
            result.interpretation,
            EffectSizeInterpretation::Negligible
        ));
    }

    #[test]
    fn test_cliffs_delta() {
        let analyzer = ComparativeAnalyzer::new();
        let group_a = array![1.0, 2.0, 3.0];
        let group_b = array![4.0, 5.0, 6.0];

        let delta = analyzer.cliffs_delta(&group_a, &group_b).unwrap();
        assert_eq!(delta, -1.0); // All values in group_a are less than group_b
    }

    #[test]
    fn test_model_comparison() {
        let analyzer = ComparativeAnalyzer::new();
        let model_names = vec!["Model A".to_string(), "Model B".to_string()];
        let cv_scores = vec![
            array![0.8, 0.82, 0.78, 0.81, 0.79],
            array![0.75, 0.77, 0.73, 0.76, 0.74],
        ];

        let result = analyzer.compare_models(model_names, cv_scores).unwrap();

        assert_eq!(result.model_names.len(), 2);
        assert_eq!(result.performance_scores.len(), 2);
        assert_eq!(result.ranking[0], 0); // Model A should be ranked first
        assert_eq!(result.best_model_index, 0);
    }

    #[test]
    fn test_permutation_test() {
        let analyzer = ComparativeAnalyzer::new().with_random_state(42);
        let group_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = analyzer
            .significance_test(
                &group_a,
                &group_b,
                SignificanceTest::PermutationTest {
                    n_permutations: 1000,
                },
                EffectSizeMeasure::CohensD,
            )
            .unwrap();

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_confidence_intervals() {
        let analyzer = ComparativeAnalyzer::new();
        let group_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let (lower, upper) = analyzer
            .compute_confidence_interval(
                &group_a,
                &group_b,
                ConfidenceIntervalType::TDistribution,
                0.95,
            )
            .unwrap();

        assert!(lower < upper);
        // The mean difference should be negative (group_a has lower mean)
        let mean_diff = 3.0 - 4.0; // Expected difference
        assert!(lower < mean_diff && mean_diff < upper); // CI should contain the true difference
    }
}
