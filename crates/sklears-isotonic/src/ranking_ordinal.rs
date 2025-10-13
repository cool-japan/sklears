//! Ranking and Ordinal Data Models for Isotonic Regression
//!
//! This module provides isotonic regression models specifically designed for ranking
//! and ordinal data, including preference learning, pairwise comparisons, and tournament ranking.

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::{collections::HashMap, marker::PhantomData};

/// Ranking and ordinal regression methods
#[derive(Debug, Clone, Copy, PartialEq)]
/// RankingMethod
pub enum RankingMethod {
    /// Standard isotonic ranking - direct rank transformation
    IsotonicRanking,
    /// Ordinal regression with cumulative logits
    OrdinalRegression,
    /// Preference learning using pairwise comparisons
    PreferenceLearning,
    /// Tournament-style ranking system
    TournamentRanking,
    /// Bradley-Terry model for pairwise comparisons
    BradleyTerry,
}

/// Preference learning approach
#[derive(Debug, Clone, Copy, PartialEq)]
/// PreferenceMethod
pub enum PreferenceMethod {
    /// Simple pairwise preference ranking
    Pairwise,
    /// Ranking SVM approach
    RankingSVM,
    /// ListNet approach for list-wise ranking
    ListNet,
    /// RankBoost ensemble method
    RankBoost,
}

/// Tournament ranking system type
#[derive(Debug, Clone, Copy, PartialEq)]
/// TournamentType
pub enum TournamentType {
    /// Round-robin tournament (all pairs)
    RoundRobin,
    /// Single-elimination tournament
    SingleElimination,
    /// Swiss-system tournament
    Swiss,
    /// Elo rating system
    Elo,
}

/// Isotonic ranking model for ordinal and ranking data
#[derive(Debug, Clone)]
/// IsotonicRankingModel
pub struct IsotonicRankingModel<State> {
    method: RankingMethod,
    preference_method: Option<PreferenceMethod>,
    tournament_type: Option<TournamentType>,
    regularization: Float,
    max_iter: usize,
    tolerance: Float,
    fitted_scores: Option<Array1<Float>>,
    fitted_ranks: Option<Array1<usize>>,
    ordinal_thresholds: Option<Array1<Float>>,
    pairwise_weights: Option<Array2<Float>>,
    _state: PhantomData<State>,
}

impl IsotonicRankingModel<Untrained> {
    /// Create a new isotonic ranking model
    pub fn new() -> Self {
        Self {
            method: RankingMethod::IsotonicRanking,
            preference_method: None,
            tournament_type: None,
            regularization: 0.01,
            max_iter: 1000,
            tolerance: 1e-6,
            fitted_scores: None,
            fitted_ranks: None,
            ordinal_thresholds: None,
            pairwise_weights: None,
            _state: PhantomData,
        }
    }

    /// Set the ranking method
    pub fn method(mut self, method: RankingMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the preference learning method (for PreferenceLearning)
    pub fn preference_method(mut self, method: PreferenceMethod) -> Self {
        self.preference_method = Some(method);
        self
    }

    /// Set the tournament type (for TournamentRanking)
    pub fn tournament_type(mut self, tournament_type: TournamentType) -> Self {
        self.tournament_type = Some(tournament_type);
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: Float) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Estimator for IsotonicRankingModel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for IsotonicRankingModel<Untrained> {
    type Fitted = IsotonicRankingModel<Trained>;

    fn fit(mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Input and target arrays must have same length".to_string(),
            ));
        }

        let (fitted_scores, fitted_ranks, ordinal_thresholds, pairwise_weights) = match self.method
        {
            RankingMethod::IsotonicRanking => self.fit_isotonic_ranking(x, y)?,
            RankingMethod::OrdinalRegression => self.fit_ordinal_regression(x, y)?,
            RankingMethod::PreferenceLearning => self.fit_preference_learning(x, y)?,
            RankingMethod::TournamentRanking => self.fit_tournament_ranking(x, y)?,
            RankingMethod::BradleyTerry => self.fit_bradley_terry(x, y)?,
        };

        Ok(IsotonicRankingModel {
            method: self.method,
            preference_method: self.preference_method,
            tournament_type: self.tournament_type,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            fitted_scores,
            fitted_ranks,
            ordinal_thresholds,
            pairwise_weights,
            _state: PhantomData,
        })
    }
}

impl IsotonicRankingModel<Untrained> {
    /// Fit isotonic ranking model
    fn fit_isotonic_ranking(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        // Sort data by x values
        let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let sorted_x: Array1<Float> = sorted_indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Array1<Float> = sorted_indices.iter().map(|&i| y[i]).collect();

        // Apply isotonic regression to get monotonic scores
        let mut isotonic_y = sorted_y.clone();
        self.pool_adjacent_violators(&mut isotonic_y)?;

        // Convert scores to ranks
        let ranks = self.scores_to_ranks(&isotonic_y);

        // Map back to original order
        let mut final_scores = Array1::zeros(x.len());
        let mut final_ranks = Array1::zeros(x.len());

        for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            final_scores[orig_idx] = isotonic_y[new_idx];
            final_ranks[orig_idx] = ranks[new_idx];
        }

        Ok((Some(final_scores), Some(final_ranks), None, None))
    }

    /// Fit ordinal regression model
    fn fit_ordinal_regression(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        // Determine unique ordinal levels
        let mut unique_levels: Vec<Float> = y.to_vec();
        unique_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_levels.dedup();

        if unique_levels.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 ordinal levels for ordinal regression".to_string(),
            ));
        }

        // Initialize thresholds between levels
        let mut thresholds = Array1::zeros(unique_levels.len() - 1);
        for i in 0..thresholds.len() {
            thresholds[i] = (unique_levels[i] + unique_levels[i + 1]) / 2.0;
        }

        // Iterative optimization for ordinal regression
        let mut scores = Array1::zeros(x.len());

        for iter in 0..self.max_iter {
            let old_scores = scores.clone();

            // Update scores using isotonic regression within each ordinal level
            for (level_idx, &level) in unique_levels.iter().enumerate() {
                let level_mask: Vec<bool> = y.iter().map(|&val| val == level).collect();
                let level_indices: Vec<usize> = level_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &mask)| mask)
                    .map(|(i, _)| i)
                    .collect();

                if level_indices.len() > 1 {
                    let level_x: Array1<Float> = level_indices.iter().map(|&i| x[i]).collect();
                    let level_scores: Array1<Float> =
                        level_indices.iter().map(|&i| scores[i]).collect();

                    // Apply isotonic constraint within level
                    let mut sorted_indices: Vec<usize> = (0..level_x.len()).collect();
                    sorted_indices.sort_by(|&i, &j| level_x[i].partial_cmp(&level_x[j]).unwrap());

                    let mut isotonic_scores: Array1<Float> =
                        sorted_indices.iter().map(|&i| level_scores[i]).collect();
                    self.pool_adjacent_violators(&mut isotonic_scores)?;

                    // Update original scores
                    for (sort_idx, &orig_level_idx) in sorted_indices.iter().enumerate() {
                        let orig_idx = level_indices[orig_level_idx];
                        scores[orig_idx] = isotonic_scores[sort_idx];
                    }
                }
            }

            // Check convergence
            let change = (&scores - &old_scores).mapv(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), Some(thresholds), None))
    }

    /// Fit preference learning model
    fn fit_preference_learning(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let method = self.preference_method.unwrap_or(PreferenceMethod::Pairwise);

        match method {
            PreferenceMethod::Pairwise => self.fit_pairwise_preferences(x, y),
            PreferenceMethod::RankingSVM => self.fit_ranking_svm(x, y),
            PreferenceMethod::ListNet => self.fit_listnet(x, y),
            PreferenceMethod::RankBoost => self.fit_rankboost(x, y),
        }
    }

    /// Fit pairwise preference model
    fn fit_pairwise_preferences(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let n = x.len();
        let mut pairwise_weights = Array2::zeros((n, n));
        let mut scores = Array1::zeros(n);

        // Initialize scores with target values
        scores.assign(y);

        // Compute pairwise preference weights
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Preference strength based on score difference and input similarity
                    let score_diff = (y[i] - y[j]).abs();
                    let input_diff = (x[i] - x[j]).abs();
                    let weight = score_diff / (1.0 + input_diff);
                    pairwise_weights[[i, j]] = weight;
                }
            }
        }

        // Iterative refinement using pairwise constraints
        for iter in 0..self.max_iter {
            let old_scores = scores.clone();

            for i in 0..n {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for j in 0..n {
                    if i != j {
                        let weight = pairwise_weights[[i, j]];
                        if y[i] > y[j] && scores[i] <= scores[j] {
                            // Enforce preference constraint
                            weighted_sum += weight * (scores[j] + 0.1);
                        } else if y[i] < y[j] && scores[i] >= scores[j] {
                            weighted_sum += weight * (scores[j] - 0.1);
                        } else {
                            weighted_sum += weight * scores[j];
                        }
                        total_weight += weight;
                    }
                }

                if total_weight > 0.0 {
                    scores[i] = (1.0 - self.regularization) * scores[i]
                        + self.regularization * (weighted_sum / total_weight);
                }
            }

            // Apply isotonic constraint
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

            let mut isotonic_scores: Array1<Float> =
                sorted_indices.iter().map(|&i| scores[i]).collect();
            self.pool_adjacent_violators(&mut isotonic_scores)?;

            for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
                scores[orig_idx] = isotonic_scores[sort_idx];
            }

            // Check convergence
            let change = (&scores - &old_scores).mapv(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, Some(pairwise_weights)))
    }

    /// Fit ranking SVM model
    fn fit_ranking_svm(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        // Simplified ranking SVM using pairwise constraints
        let n = x.len();
        let mut scores = y.clone();

        // Generate preference pairs
        let mut preference_pairs = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if y[i] > y[j] {
                    preference_pairs.push((i, j, 1.0)); // i preferred over j
                } else if y[i] < y[j] {
                    preference_pairs.push((i, j, -1.0)); // j preferred over i
                }
            }
        }

        // Iterative optimization with SVM-like constraints
        for iter in 0..self.max_iter {
            let old_scores = scores.clone();

            for &(i, j, preference) in &preference_pairs {
                let margin = preference * (scores[i] - scores[j]);
                if margin < 1.0 {
                    // Adjust scores to satisfy margin constraint
                    let adjustment = self.regularization * (1.0 - margin) / 2.0;
                    scores[i] += preference * adjustment;
                    scores[j] -= preference * adjustment;
                }
            }

            // Apply isotonic constraint
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

            let mut isotonic_scores: Array1<Float> =
                sorted_indices.iter().map(|&i| scores[i]).collect();
            self.pool_adjacent_violators(&mut isotonic_scores)?;

            for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
                scores[orig_idx] = isotonic_scores[sort_idx];
            }

            // Check convergence
            let change = (&scores - &old_scores).mapv(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit ListNet model
    fn fit_listnet(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        // Simplified ListNet using permutation probability
        let n = x.len();
        let mut scores = y.clone();

        // Sort indices by target values to get ideal ranking
        let mut target_ranking: Vec<usize> = (0..n).collect();
        target_ranking.sort_by(|&i, &j| y[j].partial_cmp(&y[i]).unwrap()); // Descending order

        for iter in 0..self.max_iter {
            let old_scores = scores.clone();

            // Compute current ranking based on scores
            let mut current_ranking: Vec<usize> = (0..n).collect();
            current_ranking.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());

            // Update scores to better match target ranking
            for (target_pos, &item_idx) in target_ranking.iter().enumerate() {
                let current_pos = current_ranking.iter().position(|&x| x == item_idx).unwrap();
                let position_diff = target_pos as Float - current_pos as Float;
                scores[item_idx] += self.regularization * position_diff;
            }

            // Apply isotonic constraint based on input ordering
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

            let mut isotonic_scores: Array1<Float> =
                sorted_indices.iter().map(|&i| scores[i]).collect();
            self.pool_adjacent_violators(&mut isotonic_scores)?;

            for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
                scores[orig_idx] = isotonic_scores[sort_idx];
            }

            // Check convergence
            let change = (&scores - &old_scores).mapv(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit RankBoost model
    fn fit_rankboost(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        // Simplified RankBoost with adaptive weighting
        let n = x.len();
        let mut scores = y.clone();
        let mut example_weights: Array1<Float> = Array1::ones(n) / n as Float;

        // Generate preference pairs with weights
        let mut preference_pairs = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if y[i] != y[j] {
                    let preference = if y[i] > y[j] { 1.0 } else { -1.0 };
                    preference_pairs.push((i, j, preference));
                }
            }
        }

        for iter in 0..self.max_iter {
            let old_scores = scores.clone();

            // Compute ranking errors with current weights
            let mut total_error = 0.0;
            for &(i, j, preference) in &preference_pairs {
                let predicted_preference = if scores[i] > scores[j] { 1.0 } else { -1.0 };
                if predicted_preference != preference {
                    total_error += example_weights[i] + example_weights[j];
                }
            }

            if total_error > 0.0 {
                // Update example weights (boost incorrectly ranked examples)
                let boost_factor = (1.0 - total_error) / total_error;
                for &(i, j, preference) in &preference_pairs {
                    let predicted_preference = if scores[i] > scores[j] { 1.0 } else { -1.0 };
                    if predicted_preference != preference {
                        example_weights[i] *= boost_factor;
                        example_weights[j] *= boost_factor;
                    }
                }

                // Normalize weights
                let weight_sum = example_weights.sum();
                example_weights /= weight_sum;
            }

            // Update scores based on weighted preferences
            for i in 0..n {
                let mut score_adjustment = 0.0;
                let mut total_weight = 0.0;

                for &(pi, pj, preference) in &preference_pairs {
                    if pi == i {
                        score_adjustment += example_weights[i] * preference * 0.1;
                        total_weight += example_weights[i];
                    } else if pj == i {
                        score_adjustment -= example_weights[i] * preference * 0.1;
                        total_weight += example_weights[i];
                    }
                }

                if total_weight > 0.0 {
                    scores[i] += self.regularization * score_adjustment / total_weight;
                }
            }

            // Apply isotonic constraint
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

            let mut isotonic_scores: Array1<Float> =
                sorted_indices.iter().map(|&i| scores[i]).collect();
            self.pool_adjacent_violators(&mut isotonic_scores)?;

            for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
                scores[orig_idx] = isotonic_scores[sort_idx];
            }

            // Check convergence
            let change = (&scores - &old_scores).mapv(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit tournament ranking model
    fn fit_tournament_ranking(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let tournament_type = self.tournament_type.unwrap_or(TournamentType::RoundRobin);

        match tournament_type {
            TournamentType::RoundRobin => self.fit_round_robin_tournament(x, y),
            TournamentType::SingleElimination => self.fit_single_elimination_tournament(x, y),
            TournamentType::Swiss => self.fit_swiss_tournament(x, y),
            TournamentType::Elo => self.fit_elo_rating(x, y),
        }
    }

    /// Fit round-robin tournament model
    fn fit_round_robin_tournament(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let n = x.len();
        let mut scores: Array1<Float> = Array1::zeros(n);
        let mut wins: Array1<Float> = Array1::zeros(n);
        let mut games: Array1<Float> = Array1::zeros(n);

        // Simulate round-robin tournament
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    games[i] += 1.0;
                    games[j] += 1.0;

                    // Determine winner based on target values with some input influence
                    let prob_i_wins =
                        1.0 / (1.0 + (-0.1 * (y[i] - y[j] + 0.1 * (x[i] - x[j]))).exp());

                    if prob_i_wins > 0.5 {
                        wins[i] += 1.0;
                    } else {
                        wins[j] += 1.0;
                    }
                }
            }
        }

        // Calculate win rates as scores
        for i in 0..n {
            scores[i] = if games[i] > 0.0 {
                wins[i] / games[i]
            } else {
                0.0
            };
        }

        // Apply isotonic constraint
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let mut isotonic_scores: Array1<Float> =
            sorted_indices.iter().map(|&i| scores[i]).collect();
        self.pool_adjacent_violators(&mut isotonic_scores)?;

        for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            scores[orig_idx] = isotonic_scores[sort_idx];
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit single elimination tournament model
    fn fit_single_elimination_tournament(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let n = x.len();
        let mut scores = Array1::zeros(n);

        // Simulate single elimination tournament
        let mut participants: Vec<usize> = (0..n).collect();
        let mut round_scores = HashMap::new();

        for &i in &participants {
            round_scores.insert(i, 0.0);
        }

        let mut round = 1.0;
        while participants.len() > 1 {
            let mut next_round = Vec::new();

            for i in (0..participants.len()).step_by(2) {
                if i + 1 < participants.len() {
                    let p1 = participants[i];
                    let p2 = participants[i + 1];

                    // Winner determination
                    let prob_p1_wins = 1.0 / (1.0 + (-0.1 * (y[p1] - y[p2])).exp());
                    let winner = if prob_p1_wins > 0.5 { p1 } else { p2 };

                    *round_scores.get_mut(&winner).unwrap() += round;
                    next_round.push(winner);
                } else {
                    // Bye - advance automatically
                    let p1 = participants[i];
                    *round_scores.get_mut(&p1).unwrap() += round * 0.5;
                    next_round.push(p1);
                }
            }

            participants = next_round;
            round += 1.0;
        }

        // Set scores based on tournament performance
        for i in 0..n {
            scores[i] = *round_scores.get(&i).unwrap_or(&0.0);
        }

        // Apply isotonic constraint
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let mut isotonic_scores: Array1<Float> =
            sorted_indices.iter().map(|&i| scores[i]).collect();
        self.pool_adjacent_violators(&mut isotonic_scores)?;

        for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            scores[orig_idx] = isotonic_scores[sort_idx];
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit Swiss tournament model
    fn fit_swiss_tournament(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let n = x.len();
        let num_rounds = ((n as Float).log2().ceil() as usize).max(3);
        let mut scores: Array1<Float> = Array1::zeros(n);
        let mut points: Array1<Float> = Array1::zeros(n);

        // Swiss tournament simulation
        for round in 0..num_rounds {
            // Pair players with similar scores
            let mut players: Vec<usize> = (0..n).collect();
            players.sort_by(|&i, &j| points[j].partial_cmp(&points[i]).unwrap());

            for i in (0..players.len()).step_by(2) {
                if i + 1 < players.len() {
                    let p1 = players[i];
                    let p2 = players[i + 1];

                    // Determine winner
                    let strength_diff = y[p1] - y[p2] + 0.1 * (x[p1] - x[p2]);
                    let prob_p1_wins = 1.0 / (1.0 + (-strength_diff).exp());

                    if prob_p1_wins > 0.5 {
                        points[p1] += 1.0;
                    } else {
                        points[p2] += 1.0;
                    }
                }
            }
        }

        // Final scores based on tournament points
        scores.assign(&points);

        // Apply isotonic constraint
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let mut isotonic_scores: Array1<Float> =
            sorted_indices.iter().map(|&i| scores[i]).collect();
        self.pool_adjacent_violators(&mut isotonic_scores)?;

        for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            scores[orig_idx] = isotonic_scores[sort_idx];
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit Elo rating system
    fn fit_elo_rating(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let n = x.len();
        let mut elo_ratings = Array1::from_vec(vec![1500.0; n]); // Standard starting Elo
        let k_factor = 32.0; // Standard Elo K-factor

        // Simulate pairwise games for Elo rating updates
        let num_games = n * n; // Each pair plays multiple times

        for _game in 0..num_games {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        // Expected scores based on current Elo ratings
                        let expected_i = 1.0
                            / (1.0
                                + 10.0_f64.powf((elo_ratings[j] - elo_ratings[i]) as f64 / 400.0))
                                as Float;
                        let expected_j = 1.0 - expected_i;

                        // Actual game result based on target values
                        let strength_diff = y[i] - y[j] + 0.05 * (x[i] - x[j]);
                        let actual_i = if strength_diff > 0.0 { 1.0 } else { 0.0 };
                        let actual_j = 1.0 - actual_i;

                        // Update Elo ratings
                        elo_ratings[i] += k_factor * (actual_i - expected_i);
                        elo_ratings[j] += k_factor * (actual_j - expected_j);
                    }
                }
            }
        }

        let mut scores = elo_ratings;

        // Apply isotonic constraint
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let mut isotonic_scores: Array1<Float> =
            sorted_indices.iter().map(|&i| scores[i]).collect();
        self.pool_adjacent_violators(&mut isotonic_scores)?;

        for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            scores[orig_idx] = isotonic_scores[sort_idx];
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Fit Bradley-Terry model
    fn fit_bradley_terry(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Option<Array1<Float>>,
        Option<Array1<usize>>,
        Option<Array1<Float>>,
        Option<Array2<Float>>,
    )> {
        let n = x.len();
        let mut strength_params = Array1::ones(n);

        // Bradley-Terry model parameter estimation
        for iter in 0..self.max_iter {
            let old_params = strength_params.clone();

            for i in 0..n {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for j in 0..n {
                    if i != j {
                        // Observed comparison outcome
                        let i_wins = if y[i] > y[j] { 1.0 } else { 0.0 };
                        let j_wins = 1.0 - i_wins;

                        // Bradley-Terry probabilities
                        let prob_i_wins =
                            strength_params[i] / (strength_params[i] + strength_params[j]);

                        numerator += i_wins;
                        denominator += prob_i_wins;
                    }
                }

                if denominator > 0.0 {
                    let new_param = numerator / denominator;
                    strength_params[i] = if new_param > 0.0 { new_param } else { 1e-8 };
                } else {
                    strength_params[i] = 1e-8;
                }
            }

            // Normalize parameters
            let param_sum = strength_params.sum();
            if param_sum > 0.0 {
                strength_params /= param_sum / n as Float;
            }

            // Check convergence
            let change = (&strength_params - &old_params).mapv(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        let mut scores = strength_params;

        // Apply isotonic constraint
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let mut isotonic_scores: Array1<Float> =
            sorted_indices.iter().map(|&i| scores[i]).collect();
        self.pool_adjacent_violators(&mut isotonic_scores)?;

        for (sort_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            scores[orig_idx] = isotonic_scores[sort_idx];
        }

        let ranks = self.scores_to_ranks(&scores);

        Ok((Some(scores), Some(ranks), None, None))
    }

    /// Pool Adjacent Violators Algorithm for isotonic regression
    fn pool_adjacent_violators(&self, y: &mut Array1<Float>) -> Result<()> {
        let n = y.len();
        if n <= 1 {
            return Ok(());
        }

        let mut i = 0;
        while i < n - 1 {
            if y[i] > y[i + 1] {
                // Find the violating segment
                let mut j = i + 1;
                while j < n && y[i] > y[j] {
                    j += 1;
                }

                // Compute the average for the violating segment
                let sum: Float = y.slice(s![i..j]).sum();
                let avg = sum / (j - i) as Float;

                // Set all values in the segment to the average
                for k in i..j {
                    y[k] = avg;
                }

                // Backtrack to check for new violations
                if i > 0 {
                    i -= 1;
                } else {
                    i = j;
                }
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    /// Convert scores to ranks (1-based, with ties getting average rank)
    fn scores_to_ranks(&self, scores: &Array1<Float>) -> Array1<usize> {
        let n = scores.len();
        let mut indexed_scores: Vec<(usize, Float)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        // Sort by score in descending order
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut ranks = Array1::zeros(n);
        for (rank, &(original_index, _)) in indexed_scores.iter().enumerate() {
            ranks[original_index] = rank + 1; // 1-based ranking
        }

        ranks
    }
}

impl Predict<Array1<Float>, Array1<Float>> for IsotonicRankingModel<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_scores = self
            .fitted_scores
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "Model not fitted".to_string(),
            })?;

        // Simple linear interpolation for prediction
        let mut predictions = Array1::zeros(x.len());

        // Find the range of fitted data for extrapolation
        let min_score = fitted_scores.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_score = fitted_scores
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        for (i, &xi) in x.iter().enumerate() {
            // Simple interpolation/extrapolation
            if xi <= 0.0 {
                predictions[i] = min_score;
            } else if xi >= fitted_scores.len() as Float {
                predictions[i] = max_score;
            } else {
                let idx = (xi as usize).min(fitted_scores.len() - 1);
                predictions[i] = fitted_scores[idx];
            }
        }

        Ok(predictions)
    }
}

/// Convenience function for isotonic ranking regression
pub fn isotonic_ranking_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    method: RankingMethod,
) -> Result<(Array1<Float>, Array1<usize>)> {
    let model = IsotonicRankingModel::new().method(method);
    let fitted_model = model.fit(x, y)?;

    let scores = fitted_model.fitted_scores.unwrap();
    let ranks = fitted_model.fitted_ranks.unwrap();

    Ok((scores, ranks))
}

/// Convenience function for ordinal regression
pub fn ordinal_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
) -> Result<(Array1<Float>, Array1<usize>, Array1<Float>)> {
    let model = IsotonicRankingModel::new().method(RankingMethod::OrdinalRegression);
    let fitted_model = model.fit(x, y)?;

    let scores = fitted_model.fitted_scores.unwrap();
    let ranks = fitted_model.fitted_ranks.unwrap();
    let thresholds = fitted_model.ordinal_thresholds.unwrap();

    Ok((scores, ranks, thresholds))
}

/// Convenience function for preference learning
pub fn preference_learning(
    x: &Array1<Float>,
    y: &Array1<Float>,
    method: PreferenceMethod,
) -> Result<(Array1<Float>, Array1<usize>)> {
    let model = IsotonicRankingModel::new()
        .method(RankingMethod::PreferenceLearning)
        .preference_method(method);
    let fitted_model = model.fit(x, y)?;

    let scores = fitted_model.fitted_scores.unwrap();
    let ranks = fitted_model.fitted_ranks.unwrap();

    Ok((scores, ranks))
}

/// Convenience function for tournament ranking
pub fn tournament_ranking(
    x: &Array1<Float>,
    y: &Array1<Float>,
    tournament_type: TournamentType,
) -> Result<(Array1<Float>, Array1<usize>)> {
    let model = IsotonicRankingModel::new()
        .method(RankingMethod::TournamentRanking)
        .tournament_type(tournament_type);
    let fitted_model = model.fit(x, y)?;

    let scores = fitted_model.fitted_scores.unwrap();
    let ranks = fitted_model.fitted_ranks.unwrap();

    Ok((scores, ranks))
}

/// Convenience function for pairwise comparison using Bradley-Terry model
pub fn pairwise_comparison_bradley_terry(
    x: &Array1<Float>,
    y: &Array1<Float>,
) -> Result<(Array1<Float>, Array1<usize>)> {
    let model = IsotonicRankingModel::new().method(RankingMethod::BradleyTerry);
    let fitted_model = model.fit(x, y)?;

    let scores = fitted_model.fitted_scores.unwrap();
    let ranks = fitted_model.fitted_ranks.unwrap();

    Ok((scores, ranks))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_isotonic_ranking_basic() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 5.0, 4.0];

        let model = IsotonicRankingModel::new();
        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 5);

        // Check that predictions respect isotonic constraint
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_ordinal_regression() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 1.0, 2.0, 2.0, 3.0]; // Ordinal categories

        let (scores, ranks, thresholds) = ordinal_regression(&x, &y)?;

        assert_eq!(scores.len(), 5);
        assert_eq!(ranks.len(), 5);
        assert_eq!(thresholds.len(), 2); // 3 categories - 1

        // Check monotonicity of scores
        let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        for i in 0..sorted_indices.len() - 1 {
            let curr_idx = sorted_indices[i];
            let next_idx = sorted_indices[i + 1];
            assert!(scores[curr_idx] <= scores[next_idx]);
        }

        Ok(())
    }

    #[test]
    fn test_preference_learning_pairwise() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 3.0, 2.0, 4.0];

        let (scores, ranks) = preference_learning(&x, &y, PreferenceMethod::Pairwise)?;

        assert_eq!(scores.len(), 4);
        assert_eq!(ranks.len(), 4);

        // Check isotonic constraint
        let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        for i in 0..sorted_indices.len() - 1 {
            let curr_idx = sorted_indices[i];
            let next_idx = sorted_indices[i + 1];
            assert!(scores[curr_idx] <= scores[next_idx]);
        }

        Ok(())
    }

    #[test]
    fn test_tournament_ranking_round_robin() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 1.0, 4.0, 3.0];

        let (scores, ranks) = tournament_ranking(&x, &y, TournamentType::RoundRobin)?;

        assert_eq!(scores.len(), 4);
        assert_eq!(ranks.len(), 4);

        // Check that scores are non-negative (win rates should be >= 0)
        for &score in scores.iter() {
            assert!(score >= 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_bradley_terry_model() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 3.0, 2.0, 4.0];

        let (scores, ranks) = pairwise_comparison_bradley_terry(&x, &y)?;

        assert_eq!(scores.len(), 4);
        assert_eq!(ranks.len(), 4);

        // Check that scores are positive (Bradley-Terry strength parameters)
        for &score in scores.iter() {
            assert!(score > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_elo_rating() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 1.0, 4.0, 3.0];

        let (scores, ranks) = tournament_ranking(&x, &y, TournamentType::Elo)?;

        assert_eq!(scores.len(), 4);
        assert_eq!(ranks.len(), 4);

        // Elo ratings should be reasonable (typically 1000-2000 range after adjustment)
        for &score in scores.iter() {
            assert!(score > 0.0); // Should be positive after isotonic adjustment
        }

        Ok(())
    }

    #[test]
    fn test_ranking_methods_consistency() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect monotonic data

        // Test all ranking methods
        let methods = vec![
            RankingMethod::IsotonicRanking,
            RankingMethod::OrdinalRegression,
            RankingMethod::PreferenceLearning,
            RankingMethod::TournamentRanking,
            RankingMethod::BradleyTerry,
        ];

        for method in methods {
            let (scores, ranks) = isotonic_ranking_regression(&x, &y, method)?;

            assert_eq!(scores.len(), 5);
            assert_eq!(ranks.len(), 5);

            // For perfect monotonic data, scores should be monotonic
            for i in 0..scores.len() - 1 {
                assert!(
                    scores[i] <= scores[i + 1],
                    "Method {:?} failed monotonicity",
                    method
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_empty_input() {
        let x = array![];
        let y = array![];

        let model = IsotonicRankingModel::new();
        let result = model.fit(&x, &y);

        // Should handle empty input gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_point() -> Result<()> {
        let x = array![1.0];
        let y = array![2.0];

        let model = IsotonicRankingModel::new();
        let fitted = model.fit(&x, &y)?;
        let predictions = fitted.predict(&x)?;

        assert_eq!(predictions.len(), 1);
        Ok(())
    }

    #[test]
    fn test_preference_methods() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 1.0, 4.0, 3.0];

        let methods = vec![
            PreferenceMethod::Pairwise,
            PreferenceMethod::RankingSVM,
            PreferenceMethod::ListNet,
            PreferenceMethod::RankBoost,
        ];

        for method in methods {
            let (scores, ranks) = preference_learning(&x, &y, method)?;

            assert_eq!(scores.len(), 4);
            assert_eq!(ranks.len(), 4);

            // Check isotonic constraint
            let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
            sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

            for i in 0..sorted_indices.len() - 1 {
                let curr_idx = sorted_indices[i];
                let next_idx = sorted_indices[i + 1];
                assert!(
                    scores[curr_idx] <= scores[next_idx],
                    "Method {:?} failed isotonic constraint",
                    method
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_tournament_types() -> Result<()> {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let tournament_types = vec![
            TournamentType::RoundRobin,
            TournamentType::SingleElimination,
            TournamentType::Swiss,
            TournamentType::Elo,
        ];

        for tournament_type in tournament_types {
            let (scores, ranks) = tournament_ranking(&x, &y, tournament_type)?;

            assert_eq!(scores.len(), 4);
            assert_eq!(ranks.len(), 4);

            // Check that all ranks are unique for this simple case
            let mut unique_ranks = ranks.to_vec();
            unique_ranks.sort();
            unique_ranks.dedup();
            assert_eq!(
                unique_ranks.len(),
                4,
                "Tournament type {:?} should produce unique ranks for monotonic data",
                tournament_type
            );
        }

        Ok(())
    }
}
