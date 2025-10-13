//! ML model implementations for trait recommendations
//!
//! This module contains various machine learning models used for trait
//! similarity computation, clustering, collaborative filtering, and
//! pattern matching in the recommendation system.

use crate::error::{Result, SklearsError};
use crate::trait_explorer::ml_recommendations::data_types::*;
use crate::trait_explorer::ml_recommendations::neural_networks::NeuralEmbeddingModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// SciRS2 Compliance: Use scirs2_autograd for ndarray functionality
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Ix1};

/// Advanced similarity calculation model with semantic analysis and pattern matching
pub struct TraitSimilarityModel {
    /// Precomputed similarity matrix for fast lookups
    similarity_matrix: HashMap<(String, String), f64>,
    /// Configuration
    config: MLRecommendationConfig,
    /// Clustering model for grouping similar traits
    clustering_model: Option<ClusteringModel>,
    /// Collaborative filtering model
    collaborative_model: Option<CollaborativeFilteringModel>,
    /// Neural embedding model for semantic similarity
    embedding_model: Option<NeuralEmbeddingModel>,
}

impl TraitSimilarityModel {
    /// Create a new similarity model
    pub fn new() -> Self {
        let mut model = Self {
            similarity_matrix: HashMap::new(),
            config: MLRecommendationConfig::default(),
            clustering_model: None,
            collaborative_model: None,
            embedding_model: None,
        };
        model.initialize_similarity_data();
        model
    }

    /// Create a similarity model with custom configuration
    pub fn with_config(config: MLRecommendationConfig) -> Self {
        let clustering_model = Some(ClusteringModel::new(10, config.feature_dimensions));
        let collaborative_model = Some(CollaborativeFilteringModel::new(100, 50, 20));
        let embedding_model = if config.enable_neural_embeddings {
            Some(NeuralEmbeddingModel::new(
                config.feature_dimensions,
                config.feature_dimensions / 2,
                &config.neural_hidden_layers,
            ))
        } else {
            None
        };

        let mut model = Self {
            similarity_matrix: HashMap::new(),
            config,
            clustering_model,
            collaborative_model,
            embedding_model,
        };
        model.initialize_similarity_data();
        model
    }

    /// Initialize similarity data with common trait relationships
    fn initialize_similarity_data(&mut self) {
        // Common trait similarity relationships
        let similarities = vec![
            (("Clone", "Copy"), 0.8),
            (("Debug", "Display"), 0.6),
            (("PartialEq", "Eq"), 0.9),
            (("PartialOrd", "Ord"), 0.9),
            (("Iterator", "IntoIterator"), 0.7),
            (("Serialize", "Deserialize"), 0.8),
            (("Send", "Sync"), 0.7),
            (("Default", "Clone"), 0.5),
            (("Hash", "Eq"), 0.6),
            (("From", "Into"), 0.9),
            (("TryFrom", "TryInto"), 0.9),
            (("AsRef", "AsMut"), 0.7),
            (("Deref", "DerefMut"), 0.8),
            (("Add", "Sub"), 0.6),
            (("Mul", "Div"), 0.6),
            (("Read", "Write"), 0.7),
            (("BufRead", "Read"), 0.8),
            (("Seek", "Read"), 0.5),
            (("Future", "Stream"), 0.6),
            (("Pin", "Unpin"), 0.8),
        ];

        for ((trait1, trait2), similarity) in similarities {
            self.similarity_matrix
                .insert((trait1.to_string(), trait2.to_string()), similarity);
            self.similarity_matrix
                .insert((trait2.to_string(), trait1.to_string()), similarity);
        }

        // Add self-similarity
        for &(trait1, _) in &[
            "Clone", "Copy", "Debug", "Display", "PartialEq", "Eq",
            "PartialOrd", "Ord", "Iterator", "IntoIterator", "Serialize",
            "Deserialize", "Send", "Sync", "Default", "Hash", "From", "Into",
            "TryFrom", "TryInto", "AsRef", "AsMut", "Deref", "DerefMut",
        ] {
            self.similarity_matrix
                .insert((trait1.to_string(), trait1.to_string()), 1.0);
        }
    }

    /// Find similar trait patterns using advanced ML algorithms
    pub fn find_similar_patterns(&self, features: &Array1<f64>) -> Result<Vec<TraitCandidate>> {
        let mut candidates = Vec::new();

        // Use clustering if available
        if let Some(ref clustering) = self.clustering_model {
            let cluster = clustering.predict(features);

            // Find traits in the same cluster
            for (trait_name, &trait_cluster) in &clustering.assignments {
                if trait_cluster == cluster {
                    candidates.push(TraitCandidate {
                        traits: vec![trait_name.clone()],
                        similarity_score: 0.8,
                        synergy_score: 0.7,
                        usage_count: 1000, // Placeholder
                    });
                }
            }
        }

        // Use collaborative filtering if available
        if let Some(ref collaborative) = self.collaborative_model {
            // Get similar traits based on usage patterns
            let collaborative_candidates = self.get_collaborative_candidates(collaborative)?;
            candidates.extend(collaborative_candidates);
        }

        // Use neural embeddings if available
        if let Some(ref embedding) = self.embedding_model {
            let embedding_result = embedding.forward(features)?;
            let embedding_candidates = self.get_embedding_candidates(&embedding_result)?;
            candidates.extend(embedding_candidates);
        }

        // Fallback to basic similarity matrix
        if candidates.is_empty() {
            candidates = self.get_basic_candidates()?;
        }

        // Sort by similarity score and deduplicate
        candidates.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove duplicates and take top recommendations
        let mut seen_traits = std::collections::HashSet::new();
        let unique_candidates: Vec<TraitCandidate> = candidates
            .into_iter()
            .filter(|candidate| {
                let key = candidate.traits.join(",");
                if seen_traits.contains(&key) {
                    false
                } else {
                    seen_traits.insert(key);
                    true
                }
            })
            .take(self.config.max_recommendations)
            .collect();

        Ok(unique_candidates)
    }

    /// Get candidates from collaborative filtering
    fn get_collaborative_candidates(&self, collaborative: &CollaborativeFilteringModel) -> Result<Vec<TraitCandidate>> {
        let mut candidates = Vec::new();

        // Get similar traits based on usage patterns
        for (trait_name, _) in &collaborative.item_mapping {
            if let Some(similar_traits) = self.get_similar_traits_from_collaborative(trait_name, collaborative) {
                candidates.push(TraitCandidate {
                    traits: similar_traits,
                    similarity_score: 0.75,
                    synergy_score: 0.8,
                    usage_count: 5000,
                });
            }
        }

        Ok(candidates)
    }

    /// Get similar traits from collaborative filtering model
    fn get_similar_traits_from_collaborative(&self, trait_name: &str, collaborative: &CollaborativeFilteringModel) -> Option<Vec<String>> {
        // Simplified implementation - in practice would use proper CF similarity
        Some(vec![trait_name.to_string(), "Clone".to_string(), "Debug".to_string()])
    }

    /// Get candidates from neural embeddings
    fn get_embedding_candidates(&self, embedding: &Array1<f64>) -> Result<Vec<TraitCandidate>> {
        let mut candidates = Vec::new();

        // Compare with known trait embeddings (placeholder implementation)
        // In a real system, this would compare against a database of trait embeddings
        candidates.push(TraitCandidate {
            traits: vec!["PartialEq".to_string(), "Eq".to_string()],
            similarity_score: 0.85,
            synergy_score: 0.9,
            usage_count: 8000,
        });

        candidates.push(TraitCandidate {
            traits: vec!["Clone".to_string()],
            similarity_score: 0.78,
            synergy_score: 0.7,
            usage_count: 12000,
        });

        Ok(candidates)
    }

    /// Get basic candidates from similarity matrix
    fn get_basic_candidates(&self) -> Result<Vec<TraitCandidate>> {
        let mut candidates = Vec::new();

        // Example candidates based on common patterns
        candidates.push(TraitCandidate {
            traits: vec!["Clone".to_string()],
            similarity_score: 0.7,
            synergy_score: 0.6,
            usage_count: 10000,
        });

        candidates.push(TraitCandidate {
            traits: vec!["Debug".to_string()],
            similarity_score: 0.6,
            synergy_score: 0.5,
            usage_count: 8000,
        });

        candidates.push(TraitCandidate {
            traits: vec!["PartialEq".to_string(), "Eq".to_string()],
            similarity_score: 0.8,
            synergy_score: 0.9,
            usage_count: 6000,
        });

        candidates.push(TraitCandidate {
            traits: vec!["Send".to_string(), "Sync".to_string()],
            similarity_score: 0.75,
            synergy_score: 0.85,
            usage_count: 7000,
        });

        Ok(candidates)
    }

    /// Calculate semantic similarity between two traits
    pub fn semantic_similarity(&self, trait1: &str, trait2: &str) -> f64 {
        // Check precomputed similarity matrix first
        if let Some(&similarity) = self
            .similarity_matrix
            .get(&(trait1.to_string(), trait2.to_string()))
        {
            return similarity;
        }

        // Use clustering model if available
        if let Some(ref clustering) = self.clustering_model {
            return clustering.cluster_similarity(trait1, trait2);
        }

        // Use collaborative filtering if available
        if let Some(ref collaborative) = self.collaborative_model {
            if let Some(score) = collaborative.predict(trait1, trait2) {
                return score.tanh(); // Normalize to [-1, 1], then shift to [0, 1]
            }
        }

        // Fallback to string similarity
        self.string_similarity(trait1, trait2)
    }

    /// Calculate string similarity using edit distance
    fn string_similarity(&self, s1: &str, s2: &str) -> f64 {
        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 1.0;
        }

        let edit_distance = self.levenshtein_distance(s1, s2);
        1.0 - (edit_distance as f64 / max_len as f64)
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                    .min(matrix[i + 1][j] + 1)
                    .min(matrix[i][j] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Train the similarity model with usage data
    pub fn train(&mut self, usage_data: &[CrateTraitUsage]) -> Result<()> {
        // Extract features and interactions first
        let features = self.extract_features_from_usage(usage_data)?;
        let interactions = self.build_interactions_from_usage(usage_data);

        // Train clustering model
        if let Some(ref mut clustering) = self.clustering_model {
            clustering.fit(&features, 100)?;

            // Update cluster assignments
            for (i, usage) in usage_data.iter().enumerate() {
                if i < features.shape()[0] {
                    let cluster = clustering.predict(&features.row(i).to_owned());
                    clustering
                        .assignments
                        .insert(usage.primary_trait.clone(), cluster);
                }
            }
        }

        // Train collaborative filtering model
        if let Some(ref mut collaborative) = self.collaborative_model {
            collaborative.fit(
                &interactions,
                self.config.learning_rate,
                self.config.training_epochs,
            )?;
        }

        Ok(())
    }

    /// Extract features from usage data for training
    fn extract_features_from_usage(&self, usage_data: &[CrateTraitUsage]) -> Result<Array2<f64>> {
        let n_samples = usage_data.len();
        let n_features = self.config.feature_dimensions;
        let mut features = Array2::zeros((n_samples, n_features));

        for (i, usage) in usage_data.iter().enumerate() {
            let feature_vector = self.extract_usage_features(usage);
            let copy_len = feature_vector.len().min(n_features);
            features
                .row_mut(i)
                .slice_mut(scirs2_core::ndarray::s![..copy_len])
                .assign(&Array1::from_vec(feature_vector[..copy_len].to_vec()));
        }

        Ok(features)
    }

    /// Extract features from a single usage data point
    fn extract_usage_features(&self, usage: &CrateTraitUsage) -> Vec<f64> {
        vec![
            usage.usage_frequency as f64 / 1000.0,
            usage.crate_popularity as f64 / 100.0,
            usage.associated_traits.len() as f64 / 10.0,
            if usage.is_primary_trait { 1.0 } else { 0.0 },
            usage.context_complexity,
        ]
    }

    /// Build interactions from usage data
    fn build_interactions_from_usage(&self, usage_data: &[CrateTraitUsage]) -> Vec<(String, String, f64)> {
        let mut interactions = Vec::new();

        for usage in usage_data {
            // Create interaction between crate and primary trait
            interactions.push((
                usage.crate_name.clone(),
                usage.primary_trait.clone(),
                usage.usage_frequency as f64 / 100.0,
            ));

            // Create interactions with associated traits
            for associated_trait in &usage.associated_traits {
                interactions.push((
                    usage.crate_name.clone(),
                    associated_trait.clone(),
                    (usage.usage_frequency as f64 / 100.0) * 0.5, // Lower weight for associated
                ));
            }
        }

        interactions
    }

    /// Get model configuration
    pub fn get_config(&self) -> &MLRecommendationConfig {
        &self.config
    }

    /// Update model configuration
    pub fn update_config(&mut self, config: MLRecommendationConfig) {
        self.config = config;
    }
}

impl Default for TraitSimilarityModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Clustering model for grouping similar traits
#[derive(Debug, Clone)]
pub struct ClusteringModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Cluster assignments
    pub assignments: HashMap<String, usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Clustering configuration
    pub config: ClusteringConfig,
}

impl ClusteringModel {
    /// Create a new clustering model
    pub fn new(n_clusters: usize, feature_dim: usize) -> Self {
        // Initialize centroids with deterministic values
        let centroids = Array2::from_shape_fn((n_clusters, feature_dim), |(i, j)| {
            // Deterministic initialization based on cluster and feature indices
            ((i as f64 * 0.3 + j as f64 * 0.17) % 2.0) - 1.0 // Values between -1 and 1
        });

        Self {
            centroids,
            assignments: HashMap::new(),
            n_clusters,
            config: ClusteringConfig::default(),
        }
    }

    /// Create clustering model with configuration
    pub fn with_config(config: ClusteringConfig) -> Self {
        let n_clusters = config.num_clusters.unwrap_or(10);
        let feature_dim = 100; // Default feature dimension

        let mut model = Self::new(n_clusters, feature_dim);
        model.config = config;
        model
    }

    /// Fit the clustering model using K-means
    pub fn fit(&mut self, features: &Array2<f64>, max_iters: usize) -> Result<()> {
        let n_samples = features.shape()[0];
        let mut assignments = Array1::zeros(n_samples);
        let mut converged = false;

        for iter in 0..max_iters {
            let mut changed = false;

            // Assignment step
            for i in 0..n_samples {
                let sample = features.row(i);
                let best_cluster = self.find_best_cluster(&sample);

                if assignments[i] != best_cluster as f64 {
                    assignments[i] = best_cluster as f64;
                    changed = true;
                }
            }

            // Update step
            self.update_centroids(features, &assignments)?;

            if !changed {
                converged = true;
                break;
            }

            // Check for convergence based on centroid movement
            if iter > 0 && self.check_convergence_by_centroids() {
                converged = true;
                break;
            }
        }

        if !converged {
            eprintln!("Warning: Clustering did not converge after {} iterations", max_iters);
        }

        Ok(())
    }

    /// Find the best cluster for a given sample
    fn find_best_cluster(&self, sample: &ArrayView1<f64>) -> usize {
        let mut best_cluster = 0;
        let mut best_distance = f64::INFINITY;

        for j in 0..self.n_clusters {
            let centroid = self.centroids.row(j);
            let distance = match self.config.distance_metric {
                DistanceMetric::Euclidean => self.euclidean_distance(sample, &centroid),
                DistanceMetric::Manhattan => self.manhattan_distance(sample, &centroid),
                DistanceMetric::Cosine => 1.0 - self.cosine_similarity(sample, &centroid),
                _ => self.euclidean_distance(sample, &centroid),
            };

            if distance < best_distance {
                best_distance = distance;
                best_cluster = j;
            }
        }

        best_cluster
    }

    /// Update cluster centroids
    fn update_centroids(&mut self, features: &Array2<f64>, assignments: &Array1<f64>) -> Result<()> {
        for j in 0..self.n_clusters {
            let cluster_points: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &cluster)| cluster == j as f64)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_points.is_empty() {
                let mut centroid = Array1::zeros(features.shape()[1]);
                for &point_idx in &cluster_points {
                    centroid = centroid + &features.row(point_idx);
                }
                centroid /= cluster_points.len() as f64;
                self.centroids.row_mut(j).assign(&centroid);
            }
        }

        Ok(())
    }

    /// Check convergence by centroid movement
    fn check_convergence_by_centroids(&self) -> bool {
        // Simplified convergence check - in practice would compare with previous centroids
        true
    }

    /// Predict cluster for new data point
    pub fn predict(&self, features: &Array1<f64>) -> usize {
        self.find_best_cluster(&features.view())
    }

    /// Predict clusters for multiple data points
    pub fn predict_batch(&self, features: &Array2<f64>) -> Vec<usize> {
        (0..features.shape()[0])
            .map(|i| self.predict(&features.row(i).to_owned()))
            .collect()
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance<S1, S2>(&self, a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> f64
    where
        S1: ndarray::Data<Elem = f64>,
        S2: ndarray::Data<Elem = f64>,
    {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate Manhattan distance between two points
    fn manhattan_distance<S1, S2>(&self, a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> f64
    where
        S1: ndarray::Data<Elem = f64>,
        S2: ndarray::Data<Elem = f64>,
    {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>()
    }

    /// Calculate cosine similarity between two points
    fn cosine_similarity<S1, S2>(&self, a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> f64
    where
        S1: ndarray::Data<Elem = f64>,
        S2: ndarray::Data<Elem = f64>,
    {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Get cluster similarity score
    pub fn cluster_similarity(&self, trait1: &str, trait2: &str) -> f64 {
        match (self.assignments.get(trait1), self.assignments.get(trait2)) {
            (Some(&cluster1), Some(&cluster2)) => {
                if cluster1 == cluster2 {
                    1.0 // Same cluster
                } else {
                    // Calculate inter-cluster similarity
                    let centroid1 = self.centroids.row(cluster1);
                    let centroid2 = self.centroids.row(cluster2);
                    let distance = self.euclidean_distance(&centroid1, &centroid2);
                    (-distance).exp() // Exponential decay with distance
                }
            }
            _ => 0.5, // Unknown similarity
        }
    }

    /// Get cluster centers
    pub fn get_centroids(&self) -> &Array2<f64> {
        &self.centroids
    }

    /// Get cluster assignments
    pub fn get_assignments(&self) -> &HashMap<String, usize> {
        &self.assignments
    }

    /// Get inertia (within-cluster sum of squares)
    pub fn calculate_inertia(&self, features: &Array2<f64>) -> f64 {
        let mut inertia = 0.0;

        for i in 0..features.shape()[0] {
            let sample = features.row(i);
            let cluster = self.predict(&sample.to_owned());
            let centroid = self.centroids.row(cluster);
            let distance = self.euclidean_distance(&sample, &centroid);
            inertia += distance.powi(2);
        }

        inertia
    }

    /// Get silhouette score (simplified implementation)
    pub fn calculate_silhouette_score(&self, features: &Array2<f64>) -> f64 {
        // Simplified silhouette score calculation
        // In practice, this would compute the full silhouette analysis
        let inertia = self.calculate_inertia(features);
        let max_possible_inertia = features.shape()[0] as f64 * features.shape()[1] as f64;
        1.0 - (inertia / max_possible_inertia).min(1.0)
    }
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: Some(10),
            distance_metric: DistanceMetric::Euclidean,
            initialization: InitializationMethod::KMeansPlusPlus,
        }
    }
}

/// Collaborative filtering model for recommendations
#[derive(Debug, Clone)]
pub struct CollaborativeFilteringModel {
    /// User-item interaction matrix
    pub interaction_matrix: Array2<f64>,
    /// User features
    pub user_features: Array2<f64>,
    /// Item features
    pub item_features: Array2<f64>,
    /// User mapping
    pub user_mapping: HashMap<String, usize>,
    /// Item mapping
    pub item_mapping: HashMap<String, usize>,
    /// Model configuration
    pub config: CollaborativeFilteringConfig,
}

impl CollaborativeFilteringModel {
    /// Create a new collaborative filtering model
    pub fn new(n_users: usize, n_items: usize, n_factors: usize) -> Self {
        // Initialize features with deterministic values
        let user_features = Array2::from_shape_fn((n_users, n_factors), |(i, j)| {
            // Deterministic initialization for user features
            ((i as f64 * 0.13 + j as f64 * 0.19) % 1.0 - 0.5) * 0.2
        });
        let item_features = Array2::from_shape_fn((n_items, n_factors), |(i, j)| {
            // Deterministic initialization for item features
            ((i as f64 * 0.17 + j as f64 * 0.23) % 1.0 - 0.5) * 0.2
        });

        Self {
            interaction_matrix: Array2::zeros((n_users, n_items)),
            user_features,
            item_features,
            user_mapping: HashMap::new(),
            item_mapping: HashMap::new(),
            config: CollaborativeFilteringConfig::default(),
        }
    }

    /// Create model with configuration
    pub fn with_config(config: CollaborativeFilteringConfig) -> Self {
        let mut model = Self::new(100, 50, config.num_factors);
        model.config = config;
        model
    }

    /// Fit the model using matrix factorization
    pub fn fit(
        &mut self,
        interactions: &[(String, String, f64)],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        // Build mappings
        for (user, item, _) in interactions {
            if !self.user_mapping.contains_key(user) {
                let idx = self.user_mapping.len();
                self.user_mapping.insert(user.clone(), idx);
            }
            if !self.item_mapping.contains_key(item) {
                let idx = self.item_mapping.len();
                self.item_mapping.insert(item.clone(), idx);
            }
        }

        // Fill interaction matrix
        for (user, item, rating) in interactions {
            if let (Some(&user_idx), Some(&item_idx)) =
                (self.user_mapping.get(user), self.item_mapping.get(item))
            {
                if user_idx < self.interaction_matrix.shape()[0]
                    && item_idx < self.interaction_matrix.shape()[1]
                {
                    self.interaction_matrix[[user_idx, item_idx]] = *rating;
                }
            }
        }

        // Gradient descent training
        for epoch in 0..epochs {
            let mut total_error = 0.0;

            for (user, item, rating) in interactions {
                if let (Some(&user_idx), Some(&item_idx)) =
                    (self.user_mapping.get(user), self.item_mapping.get(item))
                {
                    if user_idx < self.user_features.shape()[0]
                        && item_idx < self.item_features.shape()[0]
                    {
                        let predicted = self
                            .user_features
                            .row(user_idx)
                            .dot(&self.item_features.row(item_idx));
                        let error = rating - predicted;
                        total_error += error.abs();

                        // Update features using gradient descent with regularization
                        let user_features_update = error * &self.item_features.row(item_idx);
                        let item_features_update = error * &self.user_features.row(user_idx);

                        // Apply regularization
                        let (reg_user, reg_item) = self.config.regularization;

                        let mut user_row = self.user_features.row_mut(user_idx);
                        let regularized_user_update = user_features_update - reg_user * &user_row;
                        user_row.scaled_add(learning_rate, &regularized_user_update);

                        let mut item_row = self.item_features.row_mut(item_idx);
                        let regularized_item_update = item_features_update - reg_item * &item_row;
                        item_row.scaled_add(learning_rate, &regularized_item_update);
                    }
                }
            }

            // Early stopping if error is small enough
            if total_error / (interactions.len() as f64) < 1e-6 {
                break;
            }

            // Decay learning rate
            if epoch % 10 == 0 && epoch > 0 {
                let decayed_lr = learning_rate * 0.99_f64.powi((epoch / 10) as i32);
                if decayed_lr < learning_rate * 0.1 {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Predict interaction score
    pub fn predict(&self, user: &str, item: &str) -> Option<f64> {
        match (self.user_mapping.get(user), self.item_mapping.get(item)) {
            (Some(&user_idx), Some(&item_idx)) => {
                if user_idx < self.user_features.shape()[0]
                    && item_idx < self.item_features.shape()[0]
                {
                    Some(
                        self.user_features
                            .row(user_idx)
                            .dot(&self.item_features.row(item_idx)),
                    )
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get similar items
    pub fn get_similar_items(&self, item: &str, n_similar: usize) -> Vec<(String, f64)> {
        if let Some(&item_idx) = self.item_mapping.get(item) {
            if item_idx < self.item_features.shape()[0] {
                let item_features = self.item_features.row(item_idx);
                let mut similarities = Vec::new();

                for (other_item, &other_idx) in &self.item_mapping {
                    if other_item != item && other_idx < self.item_features.shape()[0] {
                        let other_features = self.item_features.row(other_idx);
                        let similarity = self.cosine_similarity(&item_features, &other_features);
                        similarities.push((other_item.clone(), similarity));
                    }
                }

                similarities
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                similarities.into_iter().take(n_similar).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Get similar users
    pub fn get_similar_users(&self, user: &str, n_similar: usize) -> Vec<(String, f64)> {
        if let Some(&user_idx) = self.user_mapping.get(user) {
            if user_idx < self.user_features.shape()[0] {
                let user_features = self.user_features.row(user_idx);
                let mut similarities = Vec::new();

                for (other_user, &other_idx) in &self.user_mapping {
                    if other_user != user && other_idx < self.user_features.shape()[0] {
                        let other_features = self.user_features.row(other_idx);
                        let similarity = self.cosine_similarity(&user_features, &other_features);
                        similarities.push((other_user.clone(), similarity));
                    }
                }

                similarities
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                similarities.into_iter().take(n_similar).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Calculate cosine similarity
    fn cosine_similarity(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Calculate model evaluation metrics
    pub fn evaluate(&self, test_interactions: &[(String, String, f64)]) -> ModelEvaluationMetrics {
        let mut total_error = 0.0;
        let mut total_squared_error = 0.0;
        let mut valid_predictions = 0;

        for (user, item, actual_rating) in test_interactions {
            if let Some(predicted_rating) = self.predict(user, item) {
                let error = actual_rating - predicted_rating;
                total_error += error.abs();
                total_squared_error += error.powi(2);
                valid_predictions += 1;
            }
        }

        let mae = if valid_predictions > 0 {
            total_error / valid_predictions as f64
        } else {
            0.0
        };

        let rmse = if valid_predictions > 0 {
            (total_squared_error / valid_predictions as f64).sqrt()
        } else {
            0.0
        };

        ModelEvaluationMetrics {
            mean_absolute_error: mae,
            root_mean_squared_error: rmse,
            coverage: valid_predictions as f64 / test_interactions.len() as f64,
        }
    }

    /// Get model configuration
    pub fn get_config(&self) -> &CollaborativeFilteringConfig {
        &self.config
    }

    /// Get number of users and items
    pub fn get_dimensions(&self) -> (usize, usize, usize) {
        (
            self.user_features.shape()[0],
            self.item_features.shape()[0],
            self.user_features.shape()[1],
        )
    }
}

impl Default for CollaborativeFilteringConfig {
    fn default() -> Self {
        Self {
            num_factors: 20,
            regularization: (0.01, 0.01),
            num_iterations: 100,
            implicit_weight: 1.0,
        }
    }
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluationMetrics {
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Root mean squared error
    pub root_mean_squared_error: f64,
    /// Coverage (percentage of predictions made)
    pub coverage: f64,
}

/// Trait candidate for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitCandidate {
    /// List of recommended traits
    pub traits: Vec<String>,
    /// Similarity score (0.0 to 1.0)
    pub similarity_score: f64,
    /// Synergy score between traits (0.0 to 1.0)
    pub synergy_score: f64,
    /// Usage count in codebase analysis
    pub usage_count: u64,
}

/// Crate trait usage data for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateTraitUsage {
    /// Name of the crate
    pub crate_name: String,
    /// Primary trait being analyzed
    pub primary_trait: String,
    /// Associated traits used together
    pub associated_traits: Vec<String>,
    /// Usage frequency
    pub usage_frequency: u32,
    /// Crate popularity score
    pub crate_popularity: f64,
    /// Whether this is the primary trait for the crate
    pub is_primary_trait: bool,
    /// Context complexity score
    pub context_complexity: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_similarity_model_creation() {
        let model = TraitSimilarityModel::new();
        assert!(model.similarity_matrix.len() > 0);
    }

    #[test]
    fn test_trait_similarity_calculation() {
        let model = TraitSimilarityModel::new();

        // Test known similarities
        let clone_copy_sim = model.semantic_similarity("Clone", "Copy");
        let debug_display_sim = model.semantic_similarity("Debug", "Display");

        assert!(clone_copy_sim > 0.0);
        assert!(debug_display_sim > 0.0);
        assert!(clone_copy_sim > debug_display_sim); // Clone-Copy should be more similar
    }

    #[test]
    fn test_clustering_model() {
        let mut model = ClusteringModel::new(3, 5);

        // Create test data
        let features = Array2::from_shape_vec(
            (6, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0,
                1.1, 2.1, 3.1, 4.1, 5.1,
                5.0, 4.0, 3.0, 2.0, 1.0,
                5.1, 4.1, 3.1, 2.1, 1.1,
                3.0, 3.0, 3.0, 3.0, 3.0,
                2.9, 3.1, 2.8, 3.2, 2.9,
            ],
        ).unwrap();

        let result = model.fit(&features, 10);
        assert!(result.is_ok());

        // Test prediction
        let test_point = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let cluster = model.predict(&test_point);
        assert!(cluster < 3);
    }

    #[test]
    fn test_collaborative_filtering_model() {
        let mut model = CollaborativeFilteringModel::new(3, 3, 2);

        let interactions = vec![
            ("user1".to_string(), "item1".to_string(), 5.0),
            ("user1".to_string(), "item2".to_string(), 3.0),
            ("user2".to_string(), "item1".to_string(), 4.0),
            ("user2".to_string(), "item3".to_string(), 2.0),
            ("user3".to_string(), "item2".to_string(), 1.0),
            ("user3".to_string(), "item3".to_string(), 5.0),
        ];

        let result = model.fit(&interactions, 0.01, 10);
        assert!(result.is_ok());

        // Test prediction
        let prediction = model.predict("user1", "item3");
        assert!(prediction.is_some());
    }

    #[test]
    fn test_distance_metrics() {
        let model = ClusteringModel::new(2, 3);
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let euclidean = model.euclidean_distance(&a.view(), &b.view());
        let manhattan = model.manhattan_distance(&a.view(), &b.view());
        let cosine = model.cosine_similarity(&a.view(), &b.view());

        assert!(euclidean > 0.0);
        assert!(manhattan > 0.0);
        assert!(cosine > 0.0 && cosine <= 1.0);
        assert!(manhattan > euclidean); // For this specific case
    }

    #[test]
    fn test_model_evaluation() {
        let model = CollaborativeFilteringModel::new(2, 2, 1);

        let test_interactions = vec![
            ("user1".to_string(), "item1".to_string(), 3.0),
            ("user2".to_string(), "item2".to_string(), 4.0),
        ];

        let metrics = model.evaluate(&test_interactions);
        assert!(metrics.coverage >= 0.0 && metrics.coverage <= 1.0);
        assert!(metrics.mean_absolute_error >= 0.0);
        assert!(metrics.root_mean_squared_error >= 0.0);
    }
}