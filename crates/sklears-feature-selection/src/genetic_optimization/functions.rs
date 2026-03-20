//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::IndexableTarget;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Score},
    types::Float,
};
/// Generate a random chromosome (binary vector representing feature selection)
pub(crate) fn generate_random_chromosome(
    n_features: usize,
    min_features: usize,
    max_features: usize,
    rng: &mut StdRng,
) -> Vec<bool> {
    let mut chromosome = vec![false; n_features];
    let n_selected = rng.gen_range(min_features..max_features.min(n_features + 1));
    let mut selected_indices: Vec<usize> = (0..n_features).collect();
    for _ in 0..n_selected {
        let idx = rng.gen_range(0..selected_indices.len());
        let feature_idx = selected_indices.swap_remove(idx);
        chromosome[feature_idx] = true;
    }
    chromosome
}
/// Tournament selection
pub(crate) fn tournament_selection<'a>(
    population: &'a [Vec<bool>],
    fitness_scores: &[f64],
    tournament_size: usize,
    rng: &mut StdRng,
) -> &'a Vec<bool> {
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fitness = fitness_scores[best_idx];
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..population.len());
        if fitness_scores[idx] > best_fitness {
            best_idx = idx;
            best_fitness = fitness_scores[idx];
        }
    }
    &population[best_idx]
}
/// Uniform crossover
pub(crate) fn uniform_crossover(
    parent1: &[bool],
    parent2: &[bool],
    rng: &mut StdRng,
) -> (Vec<bool>, Vec<bool>) {
    let mut child1 = Vec::with_capacity(parent1.len());
    let mut child2 = Vec::with_capacity(parent1.len());
    for i in 0..parent1.len() {
        if rng.gen::<bool>() {
            child1.push(parent1[i]);
            child2.push(parent2[i]);
        } else {
            child1.push(parent2[i]);
            child2.push(parent1[i]);
        }
    }
    (child1, child2)
}
/// Flip mutation
pub(crate) fn flip_mutation(
    chromosome: &mut [bool],
    min_features: usize,
    max_features: usize,
    rng: &mut StdRng,
) {
    let flip_idx = rng.gen_range(0..chromosome.len());
    chromosome[flip_idx] = !chromosome[flip_idx];
    let new_count = chromosome.iter().filter(|&&x| x).count();
    if new_count < min_features {
        let need_to_add = min_features - new_count;
        let false_indices: Vec<usize> = chromosome
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if !val { Some(i) } else { None })
            .collect();
        for _ in 0..need_to_add {
            if !false_indices.is_empty() {
                let idx = rng.gen_range(0..false_indices.len());
                chromosome[false_indices[idx]] = true;
            }
        }
    } else if new_count > max_features {
        let need_to_remove = new_count - max_features;
        let true_indices: Vec<usize> = chromosome
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val { Some(i) } else { None })
            .collect();
        for _ in 0..need_to_remove {
            if !true_indices.is_empty() {
                let idx = rng.gen_range(0..true_indices.len());
                chromosome[true_indices[idx]] = false;
            }
        }
    }
}
/// Cross-validate score for a given feature subset
pub(crate) fn cross_validate_score<E, Y>(
    estimator: &E,
    x: &Array2<Float>,
    y: &Y,
    cv_folds: usize,
) -> SklResult<f64>
where
    E: Clone + Fit<Array2<Float>, Y>,
    E::Fitted: Score<Array2<Float>, Y>,
    Y: Clone + IndexableTarget,
    <E::Fitted as Score<Array2<Float>, Y>>::Float: Into<f64>,
{
    let n_samples = x.nrows();
    let cv = KFold::new(cv_folds).shuffle(true);
    let mut scores = Vec::new();
    for (train_idx, test_idx) in cv.split(n_samples) {
        let x_train = x.select(Axis(0), &train_idx);
        let y_train = y.select(&train_idx);
        let x_test = x.select(Axis(0), &test_idx);
        let y_test = y.select(&test_idx);
        let fitted = estimator.clone().fit(&x_train, &y_train)?;
        let score = fitted.score(&x_test, &y_test)?;
        scores.push(score);
    }
    let mut sum = 0.0;
    for score in &scores {
        sum += (*score).into();
    }
    Ok(sum / scores.len() as f64)
}
/// Trait for objective functions in multi-objective optimization
pub trait ObjectiveFunction: Send + Sync + std::fmt::Debug {
    /// Evaluate the objective for a given feature subset
    /// Returns higher values for better solutions (maximization)
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64>;
    /// Get the name of this objective
    fn name(&self) -> &str;
    /// Whether this is a minimization objective (will be converted to maximization)
    fn is_minimization(&self) -> bool {
        false
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction>;
}
impl Clone for Box<dyn ObjectiveFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
/// Helper function to compute Pearson correlation between two arrays
pub(crate) fn compute_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
