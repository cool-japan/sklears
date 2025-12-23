//! # GeneticSelector - Trait Implementations
//!
//! This module contains trait implementations for `GeneticSelector`.
//!
//! ## Implemented Traits
//!
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//! - `SelectorMixin`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use crate::ensemble_selectors::extract_features;
use crate::{base::SelectorMixin, IndexableTarget};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Score, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

impl<E> Estimator for GeneticSelector<E, Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for GeneticSelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: Score<Array2<Float>, Y> + Send + Sync,
    Y: Clone + IndexableTarget + Send + Sync,
    <E::Fitted as Score<Array2<Float>, Y>>::Float: Into<f64>,
{
    type Fitted = GeneticSelector<E, Trained>;
    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples < self.cv_folds {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least cv_folds".to_string(),
            ));
        }
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42)
        };
        let max_features = self.max_features.unwrap_or(n_features);
        if self.min_features > max_features {
            return Err(SklearsError::InvalidInput(
                "min_features cannot be greater than max_features".to_string(),
            ));
        }
        let mut population = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            let chromosome =
                generate_random_chromosome(n_features, self.min_features, max_features, &mut rng);
            population.push(chromosome);
        }
        let mut best_score = f64::NEG_INFINITY;
        let mut best_features = Vec::new();
        for _ in 0..self.generations {
            let mut fitness_scores = Vec::with_capacity(self.population_size);
            for chromosome in &population {
                let features: Vec<usize> = chromosome
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
                    .collect();
                if features.is_empty() {
                    fitness_scores.push(f64::NEG_INFINITY);
                    continue;
                }
                let x_subset = extract_features(x, &features)?;
                let score = cross_validate_score(&self.estimator, &x_subset, y, self.cv_folds)?;
                fitness_scores.push(score);
                if score > best_score {
                    best_score = score;
                    best_features = features;
                }
            }
            let mut new_population = Vec::with_capacity(self.population_size);
            let best_idx = fitness_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            new_population.push(population[best_idx].clone());
            while new_population.len() < self.population_size {
                let parent1 = tournament_selection(
                    &population,
                    &fitness_scores,
                    self.tournament_size,
                    &mut rng,
                );
                let parent2 = tournament_selection(
                    &population,
                    &fitness_scores,
                    self.tournament_size,
                    &mut rng,
                );
                let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_rate {
                    uniform_crossover(parent1, parent2, &mut rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };
                if rng.gen::<f64>() < self.mutation_rate {
                    flip_mutation(&mut child1, self.min_features, max_features, &mut rng);
                }
                if rng.gen::<f64>() < self.mutation_rate {
                    flip_mutation(&mut child2, self.min_features, max_features, &mut rng);
                }
                new_population.push(child1.to_vec());
                if new_population.len() < self.population_size {
                    new_population.push(child2.to_vec());
                }
            }
            population = new_population;
        }
        if best_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid feature subset found during genetic algorithm".to_string(),
            ));
        }
        Ok(GeneticSelector {
            estimator: self.estimator,
            population_size: self.population_size,
            generations: self.generations,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            tournament_size: self.tournament_size,
            min_features: self.min_features,
            max_features: self.max_features,
            cv_folds: self.cv_folds,
            random_state: self.random_state,
            state: PhantomData,
            best_features_: Some(best_features),
            best_score_: Some(best_score),
            n_features_: Some(n_features),
        })
    }
}

impl<E> Transform<Array2<Float>> for GeneticSelector<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;
        let selected_features = self.best_features_.as_ref().unwrap();
        extract_features(x, selected_features)
    }
}

impl<E> SelectorMixin for GeneticSelector<E, Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.best_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            support[idx] = true;
        }
        Ok(support)
    }
    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.best_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}
