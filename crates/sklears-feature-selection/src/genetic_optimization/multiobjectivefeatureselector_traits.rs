//! # MultiObjectiveFeatureSelector - Trait Implementations
//!
//! This module contains trait implementations for `MultiObjectiveFeatureSelector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Fit`
//! - `Fit`
//! - `Transform`
//! - `SelectorMixin`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::base::SelectorMixin;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
};
use std::marker::PhantomData;

impl Default for MultiObjectiveFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<f64>, Array1<i32>> for MultiObjectiveFeatureSelector<Untrained> {
    type Fitted = MultiObjectiveFeatureSelector<Trained>;
    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;
        let n_features = x.ncols();
        if self.objectives.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No objectives added".to_string(),
            ));
        }
        let (pareto_front, best_solutions, objective_values) =
            self.optimize(x, Some(y), None, n_features)?;
        Ok(MultiObjectiveFeatureSelector {
            objectives: self.objectives,
            optimization_method: self.optimization_method,
            population_size: self.population_size,
            n_generations: self.n_generations,
            crossover_prob: self.crossover_prob,
            mutation_prob: self.mutation_prob,
            random_state: self.random_state,
            state: PhantomData,
            pareto_front_: Some(pareto_front),
            best_solutions_: Some(best_solutions),
            objective_values_: Some(objective_values),
            n_features_: Some(n_features),
        })
    }
}

impl Fit<Array2<f64>, Array1<f64>> for MultiObjectiveFeatureSelector<Untrained> {
    type Fitted = MultiObjectiveFeatureSelector<Trained>;
    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;
        let n_features = x.ncols();
        if self.objectives.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No objectives added".to_string(),
            ));
        }
        let (pareto_front, best_solutions, objective_values) =
            self.optimize(x, None, Some(y), n_features)?;
        Ok(MultiObjectiveFeatureSelector {
            objectives: self.objectives,
            optimization_method: self.optimization_method,
            population_size: self.population_size,
            n_generations: self.n_generations,
            crossover_prob: self.crossover_prob,
            mutation_prob: self.mutation_prob,
            random_state: self.random_state,
            state: PhantomData,
            pareto_front_: Some(pareto_front),
            best_solutions_: Some(best_solutions),
            objective_values_: Some(objective_values),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<f64>> for MultiObjectiveFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = self.n_features_.unwrap();
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }
        let selected = self.balanced_solution().unwrap_or(&[]);
        let n_samples = x.nrows();
        let n_selected = selected.len();
        let mut x_new = Array2::<f64>::zeros((n_samples, n_selected));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }
        Ok(x_new)
    }
}

impl SelectorMixin for MultiObjectiveFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.balanced_solution().unwrap_or(&[]);
        let n_features = self.n_features_.unwrap();
        let mut support = vec![false; n_features];
        for &idx in selected_features {
            support[idx] = true;
        }
        Ok(Array1::from(support))
    }
    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected = self.balanced_solution().unwrap_or(&[]);
        Ok(indices
            .iter()
            .filter_map(|&idx| selected.iter().position(|&f| f == idx))
            .collect())
    }
}
