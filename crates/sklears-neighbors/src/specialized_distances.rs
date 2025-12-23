//! Specialized distance metrics for various data types
//!
//! This module implements distance metrics for specialized data types including
//! strings, graphs, sets, and categorical data.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float as FloatTrait;
use sklears_core::types::Float;
use std::collections::{HashMap, HashSet};

/// String distance metrics
#[derive(Debug, Clone)]
pub enum StringDistance {
    /// Levenshtein (edit) distance
    Levenshtein,
    /// Hamming distance (for equal-length strings)
    Hamming,
    /// Damerau-Levenshtein distance
    DamerauLevenshtein,
    /// Jaro distance
    Jaro,
    /// Jaro-Winkler distance
    JaroWinkler { prefix_scale: Float },
    /// Longest Common Subsequence distance
    LCS,
}

impl StringDistance {
    /// Compute distance between two strings
    pub fn distance(&self, s1: &str, s2: &str) -> Float {
        match self {
            StringDistance::Levenshtein => self.levenshtein_distance(s1, s2),
            StringDistance::Hamming => self.hamming_distance(s1, s2),
            StringDistance::DamerauLevenshtein => self.damerau_levenshtein_distance(s1, s2),
            StringDistance::Jaro => self.jaro_distance(s1, s2),
            StringDistance::JaroWinkler { prefix_scale } => {
                self.jaro_winkler_distance(s1, s2, *prefix_scale)
            }
            StringDistance::LCS => self.lcs_distance(s1, s2),
        }
    }

    /// Compute Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> Float {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2 as Float;
        }
        if len2 == 0 {
            return len1 as Float;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1, // deletion
                        matrix[i][j - 1] + 1, // insertion
                    ),
                    matrix[i - 1][j - 1] + cost, // substitution
                );
            }
        }

        matrix[len1][len2] as Float
    }

    /// Compute Hamming distance (for equal-length strings)
    fn hamming_distance(&self, s1: &str, s2: &str) -> Float {
        if s1.len() != s2.len() {
            return Float::infinity(); // Invalid for different lengths
        }

        s1.chars()
            .zip(s2.chars())
            .map(|(c1, c2)| if c1 == c2 { 0.0 } else { 1.0 })
            .sum()
    }

    /// Compute Damerau-Levenshtein distance
    fn damerau_levenshtein_distance(&self, s1: &str, s2: &str) -> Float {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2 as Float;
        }
        if len2 == 0 {
            return len1 as Float;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut h = vec![vec![0; len2 + 2]; len1 + 2];
        let max_dist = len1 + len2;
        h[0][0] = max_dist;

        for i in 0..=len1 {
            h[i + 1][0] = max_dist;
            h[i + 1][1] = i;
        }
        for j in 0..=len2 {
            h[0][j + 1] = max_dist;
            h[1][j + 1] = j;
        }

        let mut char_array = HashMap::new();

        for i in 1..=len1 {
            let mut db = 0;
            for j in 1..=len2 {
                let k = *char_array.get(&s2_chars[j - 1]).unwrap_or(&0);
                let l = db;
                let mut cost = 1;
                if s1_chars[i - 1] == s2_chars[j - 1] {
                    cost = 0;
                    db = j;
                }

                h[i + 1][j + 1] = std::cmp::min(
                    std::cmp::min(
                        h[i][j] + cost,  // substitution
                        h[i + 1][j] + 1, // insertion
                    ),
                    std::cmp::min(
                        h[i][j + 1] + 1,                         // deletion
                        h[k][l] + (i - k - 1) + 1 + (j - l - 1), // transposition
                    ),
                );
            }
            char_array.insert(s1_chars[i - 1], i);
        }

        h[len1 + 1][len2 + 1] as Float
    }

    /// Compute Jaro distance
    fn jaro_distance(&self, s1: &str, s2: &str) -> Float {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 && len2 == 0 {
            return 0.0;
        }
        if len1 == 0 || len2 == 0 {
            return 1.0;
        }

        let match_window = std::cmp::max(len1, len2) / 2 - 1;
        if match_window < 1 {
            return if s1 == s2 { 0.0 } else { 1.0 };
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut s1_matches = vec![false; len1];
        let mut s2_matches = vec![false; len2];

        let mut matches = 0;

        // Find matches
        for i in 0..len1 {
            let start = i.saturating_sub(match_window);
            let end = std::cmp::min(i + match_window + 1, len2);

            for j in start..end {
                if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                    continue;
                }
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1;
                break;
            }
        }

        if matches == 0 {
            return 1.0;
        }

        // Find transpositions
        let mut transpositions = 0;
        let mut k = 0;
        for i in 0..len1 {
            if !s1_matches[i] {
                continue;
            }
            while !s2_matches[k] {
                k += 1;
            }
            if s1_chars[i] != s2_chars[k] {
                transpositions += 1;
            }
            k += 1;
        }

        let jaro = (matches as Float / len1 as Float
            + matches as Float / len2 as Float
            + (matches as Float - transpositions as Float / 2.0) / matches as Float)
            / 3.0;

        1.0 - jaro // Convert similarity to distance
    }

    /// Compute Jaro-Winkler distance
    fn jaro_winkler_distance(&self, s1: &str, s2: &str, prefix_scale: Float) -> Float {
        let jaro_dist = self.jaro_distance(s1, s2);

        if jaro_dist > 0.7 {
            return jaro_dist;
        }

        // Find common prefix up to 4 characters
        let mut prefix_len = 0;
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 0..std::cmp::min(4, std::cmp::min(s1_chars.len(), s2_chars.len())) {
            if s1_chars[i] == s2_chars[i] {
                prefix_len += 1;
            } else {
                break;
            }
        }

        let jaro_sim = 1.0 - jaro_dist;
        let jw_sim = jaro_sim + (prefix_len as Float * prefix_scale * (1.0 - jaro_sim));

        1.0 - jw_sim // Convert similarity to distance
    }

    /// Compute Longest Common Subsequence distance
    fn lcs_distance(&self, s1: &str, s2: &str) -> Float {
        let len1 = s1.len();
        let len2 = s2.len();

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 1..=len1 {
            for j in 1..=len2 {
                if s1_chars[i - 1] == s2_chars[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = std::cmp::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        let lcs_length = dp[len1][len2];
        (len1 + len2 - 2 * lcs_length) as Float
    }
}

/// Set-based distance metrics
#[derive(Debug, Clone)]
pub enum SetDistance {
    /// Jaccard distance: 1 - |A ∩ B| / |A ∪ B|
    Jaccard,
    /// Dice distance: 1 - 2|A ∩ B| / (|A| + |B|)
    Dice,
    /// Cosine distance for set-based data
    Cosine,
    /// Hamming distance for binary sets
    Hamming,
    /// Tanimoto distance (generalized Jaccard)
    Tanimoto,
}

impl SetDistance {
    /// Compute distance between two sets
    pub fn distance<T: Eq + std::hash::Hash + Clone>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> Float {
        match self {
            SetDistance::Jaccard => self.jaccard_distance(set1, set2),
            SetDistance::Dice => self.dice_distance(set1, set2),
            SetDistance::Cosine => self.cosine_distance(set1, set2),
            SetDistance::Hamming => self.hamming_distance(set1, set2),
            SetDistance::Tanimoto => self.tanimoto_distance(set1, set2),
        }
    }

    /// Compute Jaccard distance
    fn jaccard_distance<T: Eq + std::hash::Hash + Clone>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> Float {
        let intersection_size = set1.intersection(set2).count() as Float;
        let union_size = set1.union(set2).count() as Float;

        if union_size == 0.0 {
            0.0
        } else {
            1.0 - intersection_size / union_size
        }
    }

    /// Compute Dice distance
    fn dice_distance<T: Eq + std::hash::Hash + Clone>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> Float {
        let intersection_size = set1.intersection(set2).count() as Float;
        let total_size = (set1.len() + set2.len()) as Float;

        if total_size == 0.0 {
            0.0
        } else {
            1.0 - 2.0 * intersection_size / total_size
        }
    }

    /// Compute cosine distance for sets
    fn cosine_distance<T: Eq + std::hash::Hash + Clone>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> Float {
        let intersection_size = set1.intersection(set2).count() as Float;
        let norm1 = (set1.len() as Float).sqrt();
        let norm2 = (set2.len() as Float).sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            1.0
        } else {
            1.0 - intersection_size / (norm1 * norm2)
        }
    }

    /// Compute Hamming distance for sets (symmetric difference)
    fn hamming_distance<T: Eq + std::hash::Hash + Clone>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> Float {
        set1.symmetric_difference(set2).count() as Float
    }

    /// Compute Tanimoto distance (generalized Jaccard)
    fn tanimoto_distance<T: Eq + std::hash::Hash + Clone>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> Float {
        // For sets, Tanimoto is equivalent to Jaccard
        self.jaccard_distance(set1, set2)
    }

    /// Compute distance between binary vectors (treating them as sets)
    pub fn distance_binary(&self, vec1: &ArrayView1<Float>, vec2: &ArrayView1<Float>) -> Float {
        match self {
            SetDistance::Jaccard => self.jaccard_distance_binary(vec1, vec2),
            SetDistance::Dice => self.dice_distance_binary(vec1, vec2),
            SetDistance::Cosine => self.cosine_distance_binary(vec1, vec2),
            SetDistance::Hamming => self.hamming_distance_binary(vec1, vec2),
            SetDistance::Tanimoto => self.tanimoto_distance_binary(vec1, vec2),
        }
    }

    /// Jaccard distance for binary vectors
    fn jaccard_distance_binary(&self, vec1: &ArrayView1<Float>, vec2: &ArrayView1<Float>) -> Float {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for (&a, &b) in vec1.iter().zip(vec2.iter()) {
            let a_bin = if a > 0.0 { 1.0 } else { 0.0 };
            let b_bin = if b > 0.0 { 1.0 } else { 0.0 };

            intersection += a_bin * b_bin;
            union += if a_bin > 0.0 || b_bin > 0.0 { 1.0 } else { 0.0 };
        }

        if union == 0.0 {
            0.0
        } else {
            1.0 - intersection / union
        }
    }

    /// Dice distance for binary vectors
    fn dice_distance_binary(&self, vec1: &ArrayView1<Float>, vec2: &ArrayView1<Float>) -> Float {
        let mut intersection = 0.0;
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;

        for (&a, &b) in vec1.iter().zip(vec2.iter()) {
            let a_bin = if a > 0.0 { 1.0 } else { 0.0 };
            let b_bin = if b > 0.0 { 1.0 } else { 0.0 };

            intersection += a_bin * b_bin;
            sum_a += a_bin;
            sum_b += b_bin;
        }

        let total = sum_a + sum_b;
        if total == 0.0 {
            0.0
        } else {
            1.0 - 2.0 * intersection / total
        }
    }

    /// Cosine distance for binary vectors
    fn cosine_distance_binary(&self, vec1: &ArrayView1<Float>, vec2: &ArrayView1<Float>) -> Float {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (&a, &b) in vec1.iter().zip(vec2.iter()) {
            let a_bin = if a > 0.0 { 1.0 } else { 0.0 };
            let b_bin = if b > 0.0 { 1.0 } else { 0.0 };

            dot_product += a_bin * b_bin;
            norm_a += a_bin * a_bin;
            norm_b += b_bin * b_bin;
        }

        let norm_product = norm_a.sqrt() * norm_b.sqrt();
        if norm_product == 0.0 {
            1.0
        } else {
            1.0 - dot_product / norm_product
        }
    }

    /// Hamming distance for binary vectors
    fn hamming_distance_binary(&self, vec1: &ArrayView1<Float>, vec2: &ArrayView1<Float>) -> Float {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(&a, &b)| {
                let a_bin = if a > 0.0 { 1.0 } else { 0.0 };
                let b_bin = if b > 0.0 { 1.0 } else { 0.0 };
                if a_bin != b_bin {
                    1.0
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Tanimoto distance for binary vectors
    fn tanimoto_distance_binary(
        &self,
        vec1: &ArrayView1<Float>,
        vec2: &ArrayView1<Float>,
    ) -> Float {
        self.jaccard_distance_binary(vec1, vec2)
    }
}

/// Graph distance metrics for comparing graph structures
#[derive(Debug, Clone)]
pub enum GraphDistance {
    /// Edit distance between graphs
    EditDistance,
    /// Spectral distance using eigenvalues
    SpectralDistance,
    /// Random walk distance
    RandomWalkDistance,
    /// Vertex/edge overlap distance
    OverlapDistance,
}

/// Simple graph representation for distance computation
#[derive(Debug, Clone)]
pub struct SimpleGraph {
    /// Adjacency matrix
    pub adjacency: Array2<Float>,
    /// Node labels (optional)
    pub node_labels: Option<Vec<String>>,
    /// Edge weights (if different from adjacency)
    pub edge_weights: Option<Array2<Float>>,
}

impl SimpleGraph {
    /// Create a new graph from adjacency matrix
    pub fn new(adjacency: Array2<Float>) -> Self {
        Self {
            adjacency,
            node_labels: None,
            edge_weights: None,
        }
    }

    /// Set node labels
    pub fn with_node_labels(mut self, labels: Vec<String>) -> Self {
        self.node_labels = Some(labels);
        self
    }

    /// Set edge weights
    pub fn with_edge_weights(mut self, weights: Array2<Float>) -> Self {
        self.edge_weights = Some(weights);
        self
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.adjacency.nrows()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        let mut count = 0;
        for i in 0..self.adjacency.nrows() {
            for j in 0..self.adjacency.ncols() {
                if self.adjacency[[i, j]] > 0.0 {
                    count += 1;
                }
            }
        }
        count
    }
}

impl GraphDistance {
    /// Compute distance between two graphs
    pub fn distance(&self, graph1: &SimpleGraph, graph2: &SimpleGraph) -> Float {
        match self {
            GraphDistance::EditDistance => self.edit_distance(graph1, graph2),
            GraphDistance::SpectralDistance => self.spectral_distance(graph1, graph2),
            GraphDistance::RandomWalkDistance => self.random_walk_distance(graph1, graph2),
            GraphDistance::OverlapDistance => self.overlap_distance(graph1, graph2),
        }
    }

    /// Simplified graph edit distance
    fn edit_distance(&self, graph1: &SimpleGraph, graph2: &SimpleGraph) -> Float {
        // Simplified version: count differences in adjacency matrices
        let size_diff = (graph1.num_nodes() as i32 - graph2.num_nodes() as i32).abs() as Float;

        // Compute adjacency difference for common nodes
        let min_size = std::cmp::min(graph1.num_nodes(), graph2.num_nodes());
        let mut adj_diff = 0.0;

        for i in 0..min_size {
            for j in 0..min_size {
                adj_diff += (graph1.adjacency[[i, j]] - graph2.adjacency[[i, j]]).abs();
            }
        }

        size_diff + adj_diff
    }

    /// Spectral distance using eigenvalues of Laplacian
    fn spectral_distance(&self, graph1: &SimpleGraph, graph2: &SimpleGraph) -> Float {
        let laplacian1 = self.compute_laplacian(&graph1.adjacency);
        let laplacian2 = self.compute_laplacian(&graph2.adjacency);

        // Simplified: compare matrix norms (in practice, use eigenvalues)
        let norm1 = self.frobenius_norm(&laplacian1);
        let norm2 = self.frobenius_norm(&laplacian2);

        (norm1 - norm2).abs()
    }

    /// Random walk distance (simplified)
    fn random_walk_distance(&self, graph1: &SimpleGraph, graph2: &SimpleGraph) -> Float {
        // Simplified: compare transition matrices
        let trans1 = self.compute_transition_matrix(&graph1.adjacency);
        let trans2 = self.compute_transition_matrix(&graph2.adjacency);

        if trans1.shape() != trans2.shape() {
            return Float::infinity();
        }

        // Compute difference
        let mut diff = 0.0;
        for i in 0..trans1.nrows() {
            for j in 0..trans1.ncols() {
                diff += (trans1[[i, j]] - trans2[[i, j]]).abs();
            }
        }

        diff
    }

    /// Overlap distance based on common structure
    fn overlap_distance(&self, graph1: &SimpleGraph, graph2: &SimpleGraph) -> Float {
        let edges1 = graph1.num_edges() as Float;
        let edges2 = graph2.num_edges() as Float;

        // Count common edges (simplified)
        let min_size = std::cmp::min(graph1.num_nodes(), graph2.num_nodes());
        let mut common_edges = 0.0;

        for i in 0..min_size {
            for j in 0..min_size {
                if graph1.adjacency[[i, j]] > 0.0 && graph2.adjacency[[i, j]] > 0.0 {
                    common_edges += 1.0;
                }
            }
        }

        let total_edges = edges1 + edges2;
        if total_edges == 0.0 {
            0.0
        } else {
            1.0 - 2.0 * common_edges / total_edges
        }
    }

    /// Compute graph Laplacian
    fn compute_laplacian(&self, adjacency: &Array2<Float>) -> Array2<Float> {
        let n = adjacency.nrows();
        let mut laplacian = Array2::zeros((n, n));

        // Compute degree matrix and Laplacian
        for i in 0..n {
            let degree = adjacency.row(i).sum();
            laplacian[[i, i]] = degree;

            for j in 0..n {
                if i != j {
                    laplacian[[i, j]] = -adjacency[[i, j]];
                }
            }
        }

        laplacian
    }

    /// Compute transition matrix for random walks
    fn compute_transition_matrix(&self, adjacency: &Array2<Float>) -> Array2<Float> {
        let n = adjacency.nrows();
        let mut transition = adjacency.clone();

        // Normalize rows to get transition probabilities
        for i in 0..n {
            let row_sum = transition.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n {
                    transition[[i, j]] /= row_sum;
                }
            }
        }

        transition
    }

    /// Compute Frobenius norm of matrix
    fn frobenius_norm(&self, matrix: &Array2<Float>) -> Float {
        matrix.iter().map(|&x| x.powi(2)).sum::<Float>().sqrt()
    }
}

/// Categorical distance metrics
#[derive(Debug, Clone)]
pub enum CategoricalDistance {
    /// Simple matching distance (1 if different, 0 if same)
    SimpleMatching,
    /// Weighted matching with category weights
    WeightedMatching { weights: HashMap<String, Float> },
    /// Value difference metric (VDM)
    ValueDifferenceMetric,
    /// Overlap distance for categorical vectors
    Overlap,
}

impl CategoricalDistance {
    /// Compute distance between categorical values
    pub fn distance(&self, cat1: &str, cat2: &str) -> Float {
        match self {
            CategoricalDistance::SimpleMatching => {
                if cat1 == cat2 {
                    0.0
                } else {
                    1.0
                }
            }
            CategoricalDistance::WeightedMatching { weights } => {
                if cat1 == cat2 {
                    0.0
                } else {
                    let w1 = weights.get(cat1).unwrap_or(&1.0);
                    let w2 = weights.get(cat2).unwrap_or(&1.0);
                    (w1 + w2) / 2.0
                }
            }
            CategoricalDistance::ValueDifferenceMetric => {
                // Simplified VDM (would need class information in practice)
                if cat1 == cat2 {
                    0.0
                } else {
                    1.0
                }
            }
            CategoricalDistance::Overlap => {
                if cat1 == cat2 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    /// Compute distance between categorical vectors
    pub fn distance_vector(&self, vec1: &[String], vec2: &[String]) -> Float {
        if vec1.len() != vec2.len() {
            return Float::infinity();
        }

        vec1.iter()
            .zip(vec2.iter())
            .map(|(c1, c2)| self.distance(c1, c2))
            .sum::<Float>()
            / vec1.len() as Float
    }
}

/// Probabilistic distance metrics
#[derive(Debug, Clone)]
pub enum ProbabilisticDistance {
    /// Kullback-Leibler divergence
    KLDivergence,
    /// Jensen-Shannon divergence
    JSDivergence,
    /// Bhattacharyya distance
    Bhattacharyya,
    /// Hellinger distance
    Hellinger,
    /// Total variation distance
    TotalVariation,
    /// Wasserstein distance (simplified 1D version)
    Wasserstein,
}

impl ProbabilisticDistance {
    /// Compute distance between probability distributions
    pub fn distance(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        match self {
            ProbabilisticDistance::KLDivergence => self.kl_divergence(p, q),
            ProbabilisticDistance::JSDivergence => self.js_divergence(p, q),
            ProbabilisticDistance::Bhattacharyya => self.bhattacharyya_distance(p, q),
            ProbabilisticDistance::Hellinger => self.hellinger_distance(p, q),
            ProbabilisticDistance::TotalVariation => self.total_variation_distance(p, q),
            ProbabilisticDistance::Wasserstein => self.wasserstein_distance(p, q),
        }
    }

    /// Kullback-Leibler divergence
    fn kl_divergence(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > 0.0 && qi > 0.0 {
                    pi * (pi / qi).ln()
                } else if pi == 0.0 {
                    0.0
                } else {
                    Float::infinity()
                }
            })
            .sum()
    }

    /// Jensen-Shannon divergence
    fn js_divergence(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        let m: Array1<Float> = (p + q) / 2.0;
        let kl_pm = self.kl_divergence(p, &m.view());
        let kl_qm = self.kl_divergence(q, &m.view());
        (kl_pm + kl_qm) / 2.0
    }

    /// Bhattacharyya distance
    fn bhattacharyya_distance(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        let bc = p
            .iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| (pi * qi).sqrt())
            .sum::<Float>();
        -bc.ln()
    }

    /// Hellinger distance
    fn hellinger_distance(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        let sum_sq_diff = p
            .iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| (pi.sqrt() - qi.sqrt()).powi(2))
            .sum::<Float>();
        (sum_sq_diff / 2.0).sqrt()
    }

    /// Total variation distance
    fn total_variation_distance(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| (pi - qi).abs())
            .sum::<Float>()
            / 2.0
    }

    /// Simplified 1D Wasserstein distance
    fn wasserstein_distance(&self, p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
        // Simplified version: sort and compute L1 distance
        let mut p_sorted = p.to_vec();
        let mut q_sorted = q.to_vec();
        p_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        p_sorted
            .iter()
            .zip(q_sorted.iter())
            .map(|(&pi, &qi)| (pi - qi).abs())
            .sum::<Float>()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_string_distances() {
        let distance = StringDistance::Levenshtein;

        assert_eq!(distance.distance("kitten", "sitting"), 3.0);
        assert_eq!(distance.distance("hello", "hello"), 0.0);
        assert_eq!(distance.distance("", "abc"), 3.0);

        let hamming = StringDistance::Hamming;
        assert_eq!(hamming.distance("abc", "abd"), 1.0);

        let jaro = StringDistance::Jaro;
        let jaro_dist = jaro.distance("martha", "marhta");
        assert!(jaro_dist > 0.0 && jaro_dist < 0.2);
    }

    #[test]
    fn test_set_distances() {
        let set1: HashSet<i32> = [1, 2, 3].iter().cloned().collect();
        let set2: HashSet<i32> = [2, 3, 4].iter().cloned().collect();

        let jaccard = SetDistance::Jaccard;
        let dist = jaccard.distance(&set1, &set2);
        assert_abs_diff_eq!(dist, 0.5, epsilon = 1e-6); // |intersection| = 2, |union| = 4

        let dice = SetDistance::Dice;
        let dice_dist = dice.distance(&set1, &set2);
        assert_abs_diff_eq!(dice_dist, 1.0 - 4.0 / 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_set_distances_binary() {
        let vec1 = array![1.0, 1.0, 0.0, 1.0];
        let vec2 = array![1.0, 0.0, 1.0, 1.0];

        let jaccard = SetDistance::Jaccard;
        let dist = jaccard.distance_binary(&vec1.view(), &vec2.view());
        // Intersection: 2, Union: 4
        assert_abs_diff_eq!(dist, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_graph_distances() {
        let adj1 =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();

        let adj2 =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .unwrap();

        let graph1 = SimpleGraph::new(adj1);
        let graph2 = SimpleGraph::new(adj2);

        let edit_dist = GraphDistance::EditDistance;
        let dist = edit_dist.distance(&graph1, &graph2);
        assert!(dist > 0.0);

        let overlap_dist = GraphDistance::OverlapDistance;
        let overlap = overlap_dist.distance(&graph1, &graph2);
        assert!(overlap >= 0.0 && overlap <= 1.0);
    }

    #[test]
    fn test_categorical_distances() {
        let simple = CategoricalDistance::SimpleMatching;
        assert_eq!(simple.distance("cat", "cat"), 0.0);
        assert_eq!(simple.distance("cat", "dog"), 1.0);

        let mut weights = HashMap::new();
        weights.insert("cat".to_string(), 2.0);
        weights.insert("dog".to_string(), 3.0);

        let weighted = CategoricalDistance::WeightedMatching { weights };
        assert_eq!(weighted.distance("cat", "cat"), 0.0);
        assert_eq!(weighted.distance("cat", "dog"), 2.5);
    }

    #[test]
    fn test_probabilistic_distances() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];

        let kl = ProbabilisticDistance::KLDivergence;
        let kl_dist = kl.distance(&p.view(), &q.view());
        assert!(kl_dist >= 0.0);

        let js = ProbabilisticDistance::JSDivergence;
        let js_dist = js.distance(&p.view(), &q.view());
        assert!(js_dist >= 0.0);

        let hellinger = ProbabilisticDistance::Hellinger;
        let h_dist = hellinger.distance(&p.view(), &q.view());
        assert!(h_dist >= 0.0 && h_dist <= 1.0);

        let tv = ProbabilisticDistance::TotalVariation;
        let tv_dist = tv.distance(&p.view(), &q.view());
        assert!(tv_dist >= 0.0 && tv_dist <= 1.0);
    }

    #[test]
    fn test_bhattacharyya_distance() {
        let p = array![0.6, 0.4];
        let q = array![0.4, 0.6];

        let bhatt = ProbabilisticDistance::Bhattacharyya;
        let dist = bhatt.distance(&p.view(), &q.view());

        // Bhattacharyya coefficient = sqrt(0.6*0.4) + sqrt(0.4*0.6) = 2*sqrt(0.24)
        let expected_bc = 2.0 * (0.24_f64).sqrt();
        let expected_dist = -expected_bc.ln();

        assert_abs_diff_eq!(dist, expected_dist as Float, epsilon = 1e-6);
    }
}
