//! Federated neighbor computation for privacy-preserving machine learning

use crate::{Distance, NeighborsError, NeighborsResult};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Privacy protection level for federated learning
#[derive(Debug, Clone, Copy)]
pub enum PrivacyLevel {
    /// No privacy protection (for testing only)
    None,
    /// Basic noise addition
    Basic,
    /// Advanced differential privacy
    Differential,
    /// Homomorphic encryption (placeholder)
    Homomorphic,
}

/// Noise generation strategy for differential privacy
#[derive(Debug, Clone, Copy)]
pub enum NoiseStrategy {
    /// Gaussian noise
    Gaussian,
    /// Laplacian noise
    Laplacian,
    /// Exponential noise
    Exponential,
}

/// Configuration for federated neighbor computation
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Privacy protection level
    pub privacy_level: PrivacyLevel,
    /// Noise generation strategy
    pub noise_strategy: NoiseStrategy,
    /// Privacy budget (epsilon for differential privacy)
    pub privacy_budget: Float,
    /// Distance metric to use
    pub distance: Distance,
    /// Number of rounds for iterative computation
    pub rounds: usize,
    /// Minimum number of participants required
    pub min_participants: usize,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            privacy_level: PrivacyLevel::Basic,
            noise_strategy: NoiseStrategy::Gaussian,
            privacy_budget: 1.0,
            distance: Distance::default(),
            rounds: 10,
            min_participants: 2,
        }
    }
}

/// Federated participant in the neighbor computation
#[derive(Debug, Clone)]
pub struct FederatedParticipant {
    /// Participant ID
    pub id: usize,
    /// Local training data
    pub data: Array2<Float>,
    /// Local training labels
    pub labels: Option<Array1<Int>>,
    /// Privacy budget remaining
    pub privacy_budget_remaining: Float,
    /// Number of queries processed
    pub queries_processed: usize,
}

impl FederatedParticipant {
    /// Create a new federated participant
    pub fn new(id: usize, data: Array2<Float>, labels: Option<Array1<Int>>) -> Self {
        Self {
            id,
            data,
            labels,
            privacy_budget_remaining: 1.0,
            queries_processed: 0,
        }
    }

    /// Compute local neighbors with privacy protection
    pub fn compute_local_neighbors(
        &mut self,
        query: &ArrayView1<Float>,
        k: usize,
        config: &FederatedConfig,
    ) -> NeighborsResult<Vec<(Float, usize)>> {
        if self.privacy_budget_remaining <= 0.0 {
            return Err(NeighborsError::InvalidInput(
                "Privacy budget exhausted".to_string(),
            ));
        }

        // Compute local distances
        let mut distances: Vec<(Float, usize)> = self
            .data
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(idx, sample)| {
                let distance = config.distance.calculate(query, &sample);
                (distance, idx)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(k);

        // Apply privacy protection
        let protected_distances = self.apply_privacy_protection(&distances, config)?;

        // Update privacy budget
        self.privacy_budget_remaining -= config.privacy_budget / config.rounds as Float;
        self.queries_processed += 1;

        Ok(protected_distances)
    }

    /// Apply privacy protection to distance computations
    fn apply_privacy_protection(
        &self,
        distances: &[(Float, usize)],
        config: &FederatedConfig,
    ) -> NeighborsResult<Vec<(Float, usize)>> {
        match config.privacy_level {
            PrivacyLevel::None => Ok(distances.to_vec()),
            PrivacyLevel::Basic => {
                // Add basic noise to distances
                let noise_scale = 0.1 * config.privacy_budget;
                Ok(distances
                    .iter()
                    .map(|(dist, idx)| {
                        let noise = self.generate_noise(noise_scale, config.noise_strategy);
                        (dist + noise, *idx)
                    })
                    .collect())
            }
            PrivacyLevel::Differential => {
                // Implement differential privacy with calibrated noise
                let sensitivity = self.compute_sensitivity(config);
                let noise_scale = sensitivity / config.privacy_budget;

                Ok(distances
                    .iter()
                    .map(|(dist, idx)| {
                        let noise = self.generate_noise(noise_scale, config.noise_strategy);
                        (dist + noise, *idx)
                    })
                    .collect())
            }
            PrivacyLevel::Homomorphic => {
                // Placeholder for homomorphic encryption
                // In practice, this would encrypt the distances
                Ok(distances.to_vec())
            }
        }
    }

    /// Generate noise according to the specified strategy
    fn generate_noise(&self, scale: Float, strategy: NoiseStrategy) -> Float {
        use scirs2_core::random::thread_rng;

        let mut rng = thread_rng();

        match strategy {
            NoiseStrategy::Gaussian => {
                // Simple Box-Muller transform for Gaussian noise
                let u1: Float = rng.random();
                let u2: Float = rng.random();
                let z0 = ((-2.0 as Float) * u1.ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                z0 * scale
            }
            NoiseStrategy::Laplacian => {
                // Simple Laplacian noise generation
                let u: Float = rng.random::<Float>() - 0.5;
                let sign = if u >= 0.0 { 1.0 } else { -1.0 };
                -sign * scale * ((1.0 as Float) - (2.0 as Float) * u.abs()).ln()
            }
            NoiseStrategy::Exponential => {
                // Exponential noise (symmetric)
                let u: Float = rng.random();
                let exp_sample = -scale * u.ln();
                if rng.random::<bool>() {
                    exp_sample
                } else {
                    -exp_sample
                }
            }
        }
    }

    /// Compute sensitivity for differential privacy
    fn compute_sensitivity(&self, config: &FederatedConfig) -> Float {
        // For k-NN queries, sensitivity depends on the distance metric
        // This is a simplified version - in practice, sensitivity analysis is more complex
        match config.distance {
            Distance::Euclidean => 2.0, // Maximum change when one point is added/removed
            Distance::Manhattan => 1.0,
            Distance::Cosine => 2.0,
            _ => 1.0, // Conservative estimate
        }
    }
}

/// Federated coordinator managing the neighbor computation
#[derive(Debug)]
pub struct FederatedNeighborCoordinator {
    config: FederatedConfig,
    participants: Vec<FederatedParticipant>,
    global_statistics: HashMap<String, Float>,
}

impl FederatedNeighborCoordinator {
    /// Create a new federated coordinator
    pub fn new(config: FederatedConfig) -> Self {
        Self {
            config,
            participants: Vec::new(),
            global_statistics: HashMap::new(),
        }
    }

    /// Add a participant to the federation
    pub fn add_participant(&mut self, participant: FederatedParticipant) -> NeighborsResult<()> {
        if self.participants.len() >= 100 {
            return Err(NeighborsError::InvalidInput(
                "Too many participants".to_string(),
            ));
        }

        self.participants.push(participant);
        Ok(())
    }

    /// Perform federated k-nearest neighbors search
    pub fn federated_kneighbors(
        &mut self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<(Array1<usize>, Array1<Float>)> {
        if self.participants.len() < self.config.min_participants {
            return Err(NeighborsError::InvalidInput(format!(
                "Need at least {} participants",
                self.config.min_participants
            )));
        }

        // Collect local results from all participants
        #[cfg(feature = "parallel")]
        let local_results: Vec<Vec<(Float, usize)>> = self
            .participants
            .par_iter_mut()
            .map(|participant| {
                participant
                    .compute_local_neighbors(query, k, &self.config)
                    .unwrap_or_else(|_| Vec::new())
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let local_results: Vec<Vec<(Float, usize)>> = self
            .participants
            .iter_mut()
            .map(|participant| {
                participant
                    .compute_local_neighbors(query, k, &self.config)
                    .unwrap_or_else(|_| Vec::new())
            })
            .collect();

        // Aggregate results using secure aggregation
        self.secure_aggregate(local_results, k)
    }

    /// Securely aggregate results from multiple participants
    fn secure_aggregate(
        &mut self,
        local_results: Vec<Vec<(Float, usize)>>,
        k: usize,
    ) -> NeighborsResult<(Array1<usize>, Array1<Float>)> {
        // Simple aggregation - in practice, this would use secure multi-party computation
        let mut all_neighbors: Vec<(Float, (usize, usize))> = Vec::new();

        for (participant_id, results) in local_results.into_iter().enumerate() {
            for (distance, local_idx) in results {
                // Convert local index to global index
                let global_idx = self.local_to_global_index(participant_id, local_idx);
                all_neighbors.push((distance, (global_idx, participant_id)));
            }
        }

        // Sort by distance and take k nearest
        all_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all_neighbors.truncate(k);

        // Extract indices and distances
        let distances = Array1::from_iter(all_neighbors.iter().map(|(d, _)| *d));
        let indices = Array1::from_iter(all_neighbors.iter().map(|(_, (idx, _))| *idx));

        // Update global statistics
        self.update_global_statistics(&all_neighbors);

        Ok((indices, distances))
    }

    /// Convert local participant index to global index
    fn local_to_global_index(&self, participant_id: usize, local_idx: usize) -> usize {
        let mut global_offset = 0;
        for i in 0..participant_id {
            global_offset += self.participants[i].data.nrows();
        }
        global_offset + local_idx
    }

    /// Update global statistics
    fn update_global_statistics(&mut self, neighbors: &[(Float, (usize, usize))]) {
        if neighbors.is_empty() {
            return;
        }

        let distances: Vec<Float> = neighbors.iter().map(|(d, _)| *d).collect();
        let mean_distance = distances.iter().sum::<Float>() / distances.len() as Float;
        let min_distance = distances.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_distance = distances.iter().fold(0.0_f64, |a, &b| a.max(b)) as Float;

        self.global_statistics
            .insert("mean_distance".to_string(), mean_distance);
        self.global_statistics
            .insert("min_distance".to_string(), min_distance);
        self.global_statistics
            .insert("max_distance".to_string(), max_distance);
        self.global_statistics.insert(
            "total_participants".to_string(),
            self.participants.len() as Float,
        );
    }

    /// Perform federated radius neighbors search
    pub fn federated_radius_neighbors(
        &mut self,
        query: &ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<(Vec<usize>, Vec<Float>)> {
        if self.participants.len() < self.config.min_participants {
            return Err(NeighborsError::InvalidInput(format!(
                "Need at least {} participants",
                self.config.min_participants
            )));
        }

        // Collect local radius neighbors from all participants
        let mut local_results = Vec::new();
        let distance_metric = self.config.distance.clone();
        for participant in &mut self.participants {
            if participant.privacy_budget_remaining <= 0.0 {
                local_results.push(Vec::new());
                continue;
            }

            let distances: Vec<(Float, usize)> = participant
                .data
                .axis_iter(Axis(0))
                .enumerate()
                .filter_map(|(idx, sample)| {
                    let distance = distance_metric.calculate(query, &sample);
                    if distance <= radius {
                        Some((distance, idx))
                    } else {
                        None
                    }
                })
                .collect();

            // Apply privacy protection
            let result = participant
                .apply_privacy_protection(&distances, &self.config)
                .unwrap_or_else(|_| Vec::new());
            local_results.push(result);
        }

        // Aggregate results
        let mut all_neighbors: Vec<(Float, usize)> = Vec::new();
        for (participant_id, results) in local_results.into_iter().enumerate() {
            for (distance, local_idx) in results {
                let global_idx = self.local_to_global_index(participant_id, local_idx);
                all_neighbors.push((distance, global_idx));
            }
        }

        // Sort by distance
        all_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let distances: Vec<Float> = all_neighbors.iter().map(|(d, _)| *d).collect();
        let indices: Vec<usize> = all_neighbors.iter().map(|(_, i)| *i).collect();

        Ok((indices, distances))
    }

    /// Compute local radius neighbors for a participant
    fn compute_local_radius_neighbors(
        &self,
        participant: &mut FederatedParticipant,
        query: &ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<Vec<(Float, usize)>> {
        if participant.privacy_budget_remaining <= 0.0 {
            return Ok(Vec::new());
        }

        let distances: Vec<(Float, usize)> = participant
            .data
            .axis_iter(Axis(0))
            .enumerate()
            .filter_map(|(idx, sample)| {
                let distance = self.config.distance.calculate(query, &sample);
                if distance <= radius {
                    Some((distance, idx))
                } else {
                    None
                }
            })
            .collect();

        // Apply privacy protection
        participant.apply_privacy_protection(&distances, &self.config)
    }

    /// Compute local radius neighbors for a participant (direct version to avoid borrowing issues)
    fn compute_local_radius_neighbors_direct(
        &self,
        participant: &mut FederatedParticipant,
        query: &ArrayView1<Float>,
        radius: Float,
    ) -> NeighborsResult<Vec<(Float, usize)>> {
        if participant.privacy_budget_remaining <= 0.0 {
            return Ok(Vec::new());
        }

        let distances: Vec<(Float, usize)> = participant
            .data
            .axis_iter(Axis(0))
            .enumerate()
            .filter_map(|(idx, sample)| {
                let distance = self.config.distance.calculate(query, &sample);
                if distance <= radius {
                    Some((distance, idx))
                } else {
                    None
                }
            })
            .collect();

        // Apply privacy protection
        participant.apply_privacy_protection(&distances, &self.config)
    }

    /// Get federation statistics
    pub fn get_federation_stats(&self) -> HashMap<String, Float> {
        let mut stats = self.global_statistics.clone();

        // Add participant statistics
        let total_samples: usize = self.participants.iter().map(|p| p.data.nrows()).sum();
        let total_queries: usize = self.participants.iter().map(|p| p.queries_processed).sum();
        let avg_privacy_remaining: Float = self
            .participants
            .iter()
            .map(|p| p.privacy_budget_remaining)
            .sum::<Float>()
            / self.participants.len() as Float;

        stats.insert("total_samples".to_string(), total_samples as Float);
        stats.insert("total_queries".to_string(), total_queries as Float);
        stats.insert("avg_privacy_remaining".to_string(), avg_privacy_remaining);
        stats.insert(
            "num_participants".to_string(),
            self.participants.len() as Float,
        );

        stats
    }

    /// Reset privacy budgets for all participants
    pub fn reset_privacy_budgets(&mut self) {
        for participant in &mut self.participants {
            participant.privacy_budget_remaining = 1.0;
            participant.queries_processed = 0;
        }
    }

    /// Remove participants with exhausted privacy budgets
    pub fn cleanup_exhausted_participants(&mut self) {
        self.participants
            .retain(|p| p.privacy_budget_remaining > 0.0);
    }
}

/// Trait for privacy-preserving neighbor computation protocols
pub trait PrivacyPreservingProtocol {
    /// Compute neighbors while preserving privacy
    fn private_neighbors(
        &mut self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<Vec<(Float, usize)>>;

    /// Check if privacy budget allows for more queries
    fn can_process_query(&self) -> bool;

    /// Get remaining privacy budget
    fn remaining_privacy_budget(&self) -> Float;
}

impl PrivacyPreservingProtocol for FederatedParticipant {
    fn private_neighbors(
        &mut self,
        query: &ArrayView1<Float>,
        k: usize,
    ) -> NeighborsResult<Vec<(Float, usize)>> {
        let config = FederatedConfig::default();
        self.compute_local_neighbors(query, k, &config)
    }

    fn can_process_query(&self) -> bool {
        self.privacy_budget_remaining > 0.0
    }

    fn remaining_privacy_budget(&self) -> Float {
        self.privacy_budget_remaining
    }
}

use scirs2_core::random::Rng;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_federated_participant_creation() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let labels = Some(array![0, 0, 1, 1]);

        let participant = FederatedParticipant::new(1, data, labels);
        assert_eq!(participant.id, 1);
        assert_eq!(participant.privacy_budget_remaining, 1.0);
        assert_eq!(participant.queries_processed, 0);
    }

    #[test]
    fn test_federated_coordinator() {
        let config = FederatedConfig::default();
        let mut coordinator = FederatedNeighborCoordinator::new(config);

        // Add participants
        let data1 = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let participant1 = FederatedParticipant::new(1, data1, Some(array![0, 0]));
        coordinator.add_participant(participant1).unwrap();

        let data2 = Array2::from_shape_vec((2, 2), vec![3.0, 3.0, 4.0, 4.0]).unwrap();
        let participant2 = FederatedParticipant::new(2, data2, Some(array![1, 1]));
        coordinator.add_participant(participant2).unwrap();

        // Test federated search
        let query = array![2.5, 2.5];
        let result = coordinator.federated_kneighbors(&query.view(), 2);
        assert!(result.is_ok());

        let (indices, distances) = result.unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_privacy_levels() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let mut participant = FederatedParticipant::new(1, data, None);

        let privacy_levels = [
            PrivacyLevel::None,
            PrivacyLevel::Basic,
            PrivacyLevel::Differential,
            PrivacyLevel::Homomorphic,
        ];

        for privacy_level in &privacy_levels {
            let config = FederatedConfig {
                privacy_level: *privacy_level,
                noise_strategy: NoiseStrategy::Gaussian,
                privacy_budget: 1.0,
                distance: Distance::default(),
                rounds: 10,
                min_participants: 2,
            };

            let query = array![2.5, 2.5];
            let result = participant.compute_local_neighbors(&query.view(), 2, &config);
            assert!(result.is_ok(), "Privacy level {:?} failed", privacy_level);
        }
    }

    #[test]
    fn test_noise_strategies() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let mut participant = FederatedParticipant::new(1, data, None);

        let noise_strategies = [
            NoiseStrategy::Gaussian,
            NoiseStrategy::Laplacian,
            NoiseStrategy::Exponential,
        ];

        for noise_strategy in &noise_strategies {
            let config = FederatedConfig {
                privacy_level: PrivacyLevel::Basic,
                noise_strategy: *noise_strategy,
                privacy_budget: 1.0,
                distance: Distance::default(),
                rounds: 10,
                min_participants: 2,
            };

            let query = array![2.5, 2.5];
            let result = participant.compute_local_neighbors(&query.view(), 2, &config);
            assert!(result.is_ok(), "Noise strategy {:?} failed", noise_strategy);
        }
    }

    #[test]
    fn test_privacy_budget_exhaustion() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let mut participant = FederatedParticipant::new(1, data, None);

        // Exhaust privacy budget
        participant.privacy_budget_remaining = 0.0;

        let config = FederatedConfig::default();
        let query = array![1.5, 1.5];
        let result = participant.compute_local_neighbors(&query.view(), 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_federated_radius_neighbors() {
        let config = FederatedConfig::default();
        let mut coordinator = FederatedNeighborCoordinator::new(config);

        // Add participants
        let data1 = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let participant1 = FederatedParticipant::new(1, data1, Some(array![0, 0]));
        coordinator.add_participant(participant1).unwrap();

        let data2 = Array2::from_shape_vec((2, 2), vec![5.0, 5.0, 6.0, 6.0]).unwrap();
        let participant2 = FederatedParticipant::new(2, data2, Some(array![1, 1]));
        coordinator.add_participant(participant2).unwrap();

        let query = array![1.5, 1.5];
        let result = coordinator.federated_radius_neighbors(&query.view(), 2.0);
        assert!(result.is_ok());

        let (indices, distances) = result.unwrap();
        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_federation_stats() {
        let config = FederatedConfig::default();
        let mut coordinator = FederatedNeighborCoordinator::new(config);

        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let participant = FederatedParticipant::new(1, data, Some(array![0, 1]));
        coordinator.add_participant(participant).unwrap();

        let stats = coordinator.get_federation_stats();
        assert!(stats.contains_key("total_samples"));
        assert!(stats.contains_key("num_participants"));
        assert_eq!(stats.get("num_participants").unwrap(), &1.0);
    }
}
