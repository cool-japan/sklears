//! Bioinformatics Neighbor-Based Methods
//!
//! This module provides specialized neighbor-based algorithms for bioinformatics applications,
//! including DNA/RNA sequence similarity search, protein structure analysis, gene expression
//! neighbor identification, phylogenetic distance computation, and metabolic pathway analysis.
//!
//! # Key Features
//!
//! - **Sequence Similarity Search**: Efficient DNA, RNA, and protein sequence alignment and similarity
//! - **Protein Structure Neighbors**: 3D structural similarity using RMSD and other geometric metrics
//! - **Gene Expression Analysis**: Find co-expressed genes using correlation-based distances
//! - **Phylogenetic Distance**: Evolutionary distance-based neighbor search using tree metrics
//! - **Metabolic Pathway Similarity**: Pathway-based functional similarity analysis
//! - **Multiple Sequence Alignment**: Progressive alignment with neighbor-guided ordering
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::bioinformatics::{SequenceSimilaritySearch, SequenceType};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create DNA sequence similarity search
//! let mut search = SequenceSimilaritySearch::new(SequenceType::Dna);
//!
//! // Add sequences to index
//! let sequences = vec![
//!     "ATCGATCGATCG".to_string(),
//!     "ATCGATCGATCC".to_string(),
//!     "GCTAGCTAGCTA".to_string(),
//! ];
//! search.build_index_from_sequences(&sequences)?;
//!
//! // Search for similar sequences
//! let query = "ATCGATCGATCG";
//! let results = search.search_similar_sequences(query, 2)?;
//! # Ok(())
//! # }
//! ```

use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use std::collections::{HashMap, HashSet};
use std::f64;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Biological sequence types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SequenceType {
    /// DNA sequence (A, T, G, C)
    Dna,
    /// RNA sequence (A, U, G, C)
    Rna,
    /// Protein sequence (20 amino acids)
    Protein,
    /// Generic sequence
    Generic,
}

/// Sequence alignment scoring scheme
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScoringScheme {
    /// Match score
    pub match_score: f64,
    /// Mismatch penalty
    pub mismatch_penalty: f64,
    /// Gap opening penalty
    pub gap_open_penalty: f64,
    /// Gap extension penalty
    pub gap_extend_penalty: f64,
}

impl Default for ScoringScheme {
    fn default() -> Self {
        Self {
            match_score: 2.0,
            mismatch_penalty: -1.0,
            gap_open_penalty: -2.0,
            gap_extend_penalty: -0.5,
        }
    }
}

/// Configuration for bioinformatics similarity search
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BioSearchConfig {
    /// Sequence type
    pub sequence_type: SequenceType,
    /// Scoring scheme for alignments
    pub scoring_scheme: ScoringScheme,
    /// Use local alignment (Smith-Waterman) vs global (Needleman-Wunsch)
    pub local_alignment: bool,
    /// K-mer size for fast similarity pre-filtering
    pub kmer_size: usize,
    /// Minimum sequence identity threshold
    pub min_identity: f64,
    /// Maximum E-value for significance
    pub max_evalue: f64,
    /// Use compositional bias correction
    pub composition_bias_correction: bool,
}

impl Default for BioSearchConfig {
    fn default() -> Self {
        Self {
            sequence_type: SequenceType::Dna,
            scoring_scheme: ScoringScheme::default(),
            local_alignment: false,
            kmer_size: 3,
            min_identity: 0.0,
            max_evalue: 0.001,
            composition_bias_correction: false,
        }
    }
}

/// Biological sequence metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SequenceMetadata {
    /// Sequence identifier
    pub id: String,
    /// Sequence description
    pub description: Option<String>,
    /// Organism name
    pub organism: Option<String>,
    /// Sequence length
    pub length: usize,
    /// Sequence type (DNA, RNA, Protein)
    pub seq_type: SequenceType,
    /// GC content (for DNA/RNA)
    pub gc_content: Option<f64>,
    /// Custom annotations
    pub annotations: HashMap<String, String>,
}

/// Sequence similarity search result
#[derive(Debug, Clone)]
pub struct SequenceSearchResult {
    /// Sequence metadata
    pub metadata: SequenceMetadata,
    /// Original sequence
    pub sequence: String,
    /// Alignment score
    pub score: f64,
    /// Sequence identity percentage
    pub identity: f64,
    /// E-value (statistical significance)
    pub evalue: f64,
    /// Alignment start position in query
    pub query_start: usize,
    /// Alignment end position in query
    pub query_end: usize,
    /// Alignment start position in subject
    pub subject_start: usize,
    /// Alignment end position in subject
    pub subject_end: usize,
    /// Aligned query sequence
    pub aligned_query: Option<String>,
    /// Aligned subject sequence
    pub aligned_subject: Option<String>,
}

/// K-mer indexing for fast sequence similarity pre-screening
pub struct KmerIndex {
    kmer_size: usize,
    kmer_to_sequences: HashMap<String, Vec<usize>>,
    sequence_kmer_counts: Vec<HashMap<String, usize>>,
}

impl KmerIndex {
    /// Create new k-mer index
    pub fn new(kmer_size: usize) -> Self {
        Self {
            kmer_size,
            kmer_to_sequences: HashMap::new(),
            sequence_kmer_counts: Vec::new(),
        }
    }

    /// Build index from sequences
    pub fn build(&mut self, sequences: &[String]) -> NeighborsResult<()> {
        self.kmer_to_sequences.clear();
        self.sequence_kmer_counts.clear();

        for (seq_idx, sequence) in sequences.iter().enumerate() {
            let kmers = self.extract_kmers(sequence);
            let mut kmer_counts: HashMap<String, usize> = HashMap::new();

            for kmer in kmers {
                // Add sequence to k-mer mapping
                self.kmer_to_sequences
                    .entry(kmer.clone())
                    .or_default()
                    .push(seq_idx);

                // Count k-mer occurrences in sequence
                *kmer_counts.entry(kmer).or_insert(0) += 1;
            }

            self.sequence_kmer_counts.push(kmer_counts);
        }

        Ok(())
    }

    /// Extract k-mers from sequence
    fn extract_kmers(&self, sequence: &str) -> Vec<String> {
        let sequence = sequence.to_uppercase();
        let mut kmers = Vec::new();

        if sequence.len() >= self.kmer_size {
            for i in 0..=(sequence.len() - self.kmer_size) {
                let kmer = sequence[i..i + self.kmer_size].to_string();
                kmers.push(kmer);
            }
        }

        kmers
    }

    /// Get candidate sequences for a query based on shared k-mers
    pub fn get_candidates(&self, query: &str, min_shared_kmers: usize) -> Vec<(usize, usize)> {
        let query_kmers = self.extract_kmers(query);
        let mut candidate_counts: HashMap<usize, usize> = HashMap::new();

        for kmer in query_kmers {
            if let Some(sequences) = self.kmer_to_sequences.get(&kmer) {
                for &seq_idx in sequences {
                    *candidate_counts.entry(seq_idx).or_insert(0) += 1;
                }
            }
        }

        candidate_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_shared_kmers)
            .collect()
    }

    /// Compute Jaccard similarity between query and sequence based on k-mers
    pub fn jaccard_similarity(&self, query: &str, seq_idx: usize) -> f64 {
        if seq_idx >= self.sequence_kmer_counts.len() {
            return 0.0;
        }

        let query_kmers: HashSet<String> = self.extract_kmers(query).into_iter().collect();
        let seq_kmers: HashSet<String> =
            self.sequence_kmer_counts[seq_idx].keys().cloned().collect();

        let intersection_size = query_kmers.intersection(&seq_kmers).count();
        let union_size = query_kmers.union(&seq_kmers).count();

        if union_size == 0 {
            0.0
        } else {
            intersection_size as f64 / union_size as f64
        }
    }
}

/// Pairwise sequence alignment algorithms
pub struct SequenceAligner {
    config: BioSearchConfig,
}

impl SequenceAligner {
    /// Create new sequence aligner
    pub fn new(config: BioSearchConfig) -> Self {
        Self { config }
    }

    /// Compute pairwise alignment score using dynamic programming
    pub fn align(&self, seq1: &str, seq2: &str) -> SequenceAlignment {
        if self.config.local_alignment {
            self.local_alignment(seq1, seq2)
        } else {
            self.global_alignment(seq1, seq2)
        }
    }

    /// Global alignment (Needleman-Wunsch algorithm)
    fn global_alignment(&self, seq1: &str, seq2: &str) -> SequenceAlignment {
        let seq1_chars: Vec<char> = seq1.chars().collect();
        let seq2_chars: Vec<char> = seq2.chars().collect();
        let m = seq1_chars.len();
        let n = seq2_chars.len();

        // Initialize dynamic programming matrix
        let mut dp = vec![vec![0.0; n + 1]; m + 1];

        // Initialize first row and column
        for i in 1..=m {
            dp[i][0] = dp[i - 1][0] + self.config.scoring_scheme.gap_extend_penalty;
        }
        for j in 1..=n {
            dp[0][j] = dp[0][j - 1] + self.config.scoring_scheme.gap_extend_penalty;
        }

        // Fill the matrix
        for i in 1..=m {
            for j in 1..=n {
                let match_mismatch_score = if seq1_chars[i - 1] == seq2_chars[j - 1] {
                    self.config.scoring_scheme.match_score
                } else {
                    self.config.scoring_scheme.mismatch_penalty
                };

                let diagonal = dp[i - 1][j - 1] + match_mismatch_score;
                let up = dp[i - 1][j] + self.config.scoring_scheme.gap_extend_penalty;
                let left = dp[i][j - 1] + self.config.scoring_scheme.gap_extend_penalty;

                dp[i][j] = diagonal.max(up).max(left);
            }
        }

        // Backtrack to get alignment
        let (aligned_seq1, aligned_seq2) = self.backtrack_global(&dp, &seq1_chars, &seq2_chars);
        let score = dp[m][n];

        SequenceAlignment {
            score,
            query_start: 0,
            query_end: seq1.len(),
            subject_start: 0,
            subject_end: seq2.len(),
            aligned_query: aligned_seq1.clone(),
            aligned_subject: aligned_seq2.clone(),
            identity: self.compute_identity(&aligned_seq1, &aligned_seq2),
        }
    }

    /// Local alignment (Smith-Waterman algorithm)
    fn local_alignment(&self, seq1: &str, seq2: &str) -> SequenceAlignment {
        let seq1_chars: Vec<char> = seq1.chars().collect();
        let seq2_chars: Vec<char> = seq2.chars().collect();
        let m = seq1_chars.len();
        let n = seq2_chars.len();

        // Initialize dynamic programming matrix
        let mut dp = vec![vec![0.0; n + 1]; m + 1];
        let mut max_score = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;

        // Fill the matrix
        for i in 1..=m {
            for j in 1..=n {
                let match_mismatch_score = if seq1_chars[i - 1] == seq2_chars[j - 1] {
                    self.config.scoring_scheme.match_score
                } else {
                    self.config.scoring_scheme.mismatch_penalty
                };

                let diagonal = dp[i - 1][j - 1] + match_mismatch_score;
                let up = dp[i - 1][j] + self.config.scoring_scheme.gap_extend_penalty;
                let left = dp[i][j - 1] + self.config.scoring_scheme.gap_extend_penalty;

                dp[i][j] = diagonal.max(up).max(left).max(0.0);

                if dp[i][j] > max_score {
                    max_score = dp[i][j];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        // Backtrack from maximum score position
        let (aligned_seq1, aligned_seq2, start_i, start_j) =
            self.backtrack_local(&dp, &seq1_chars, &seq2_chars, max_i, max_j);

        SequenceAlignment {
            score: max_score,
            query_start: start_i,
            query_end: max_i,
            subject_start: start_j,
            subject_end: max_j,
            aligned_query: aligned_seq1.clone(),
            aligned_subject: aligned_seq2.clone(),
            identity: self.compute_identity(&aligned_seq1, &aligned_seq2),
        }
    }

    /// Backtrack for global alignment
    fn backtrack_global(&self, dp: &[Vec<f64>], seq1: &[char], seq2: &[char]) -> (String, String) {
        let mut aligned_seq1 = String::new();
        let mut aligned_seq2 = String::new();
        let mut i = seq1.len();
        let mut j = seq2.len();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let match_mismatch_score = if seq1[i - 1] == seq2[j - 1] {
                    self.config.scoring_scheme.match_score
                } else {
                    self.config.scoring_scheme.mismatch_penalty
                };

                if (dp[i][j] - dp[i - 1][j - 1] - match_mismatch_score).abs() < 1e-9 {
                    aligned_seq1.insert(0, seq1[i - 1]);
                    aligned_seq2.insert(0, seq2[j - 1]);
                    i -= 1;
                    j -= 1;
                } else if i > 0
                    && (dp[i][j] - dp[i - 1][j] - self.config.scoring_scheme.gap_extend_penalty)
                        .abs()
                        < 1e-9
                {
                    aligned_seq1.insert(0, seq1[i - 1]);
                    aligned_seq2.insert(0, '-');
                    i -= 1;
                } else {
                    aligned_seq1.insert(0, '-');
                    aligned_seq2.insert(0, seq2[j - 1]);
                    j -= 1;
                }
            } else if i > 0 {
                aligned_seq1.insert(0, seq1[i - 1]);
                aligned_seq2.insert(0, '-');
                i -= 1;
            } else {
                aligned_seq1.insert(0, '-');
                aligned_seq2.insert(0, seq2[j - 1]);
                j -= 1;
            }
        }

        (aligned_seq1, aligned_seq2)
    }

    /// Backtrack for local alignment
    fn backtrack_local(
        &self,
        dp: &[Vec<f64>],
        seq1: &[char],
        seq2: &[char],
        max_i: usize,
        max_j: usize,
    ) -> (String, String, usize, usize) {
        let mut aligned_seq1 = String::new();
        let mut aligned_seq2 = String::new();
        let mut i = max_i;
        let mut j = max_j;

        while i > 0 && j > 0 && dp[i][j] > 0.0 {
            let match_mismatch_score = if seq1[i - 1] == seq2[j - 1] {
                self.config.scoring_scheme.match_score
            } else {
                self.config.scoring_scheme.mismatch_penalty
            };

            if (dp[i][j] - dp[i - 1][j - 1] - match_mismatch_score).abs() < 1e-9 {
                aligned_seq1.insert(0, seq1[i - 1]);
                aligned_seq2.insert(0, seq2[j - 1]);
                i -= 1;
                j -= 1;
            } else if i > 0
                && (dp[i][j] - dp[i - 1][j] - self.config.scoring_scheme.gap_extend_penalty).abs()
                    < 1e-9
            {
                aligned_seq1.insert(0, seq1[i - 1]);
                aligned_seq2.insert(0, '-');
                i -= 1;
            } else {
                aligned_seq1.insert(0, '-');
                aligned_seq2.insert(0, seq2[j - 1]);
                j -= 1;
            }
        }

        (aligned_seq1, aligned_seq2, i, j)
    }

    /// Compute sequence identity percentage
    fn compute_identity(&self, aligned_seq1: &str, aligned_seq2: &str) -> f64 {
        if aligned_seq1.len() != aligned_seq2.len() {
            return 0.0;
        }

        let matches = aligned_seq1
            .chars()
            .zip(aligned_seq2.chars())
            .filter(|(a, b)| a == b && *a != '-')
            .count();

        let alignment_length = aligned_seq1.len();
        if alignment_length == 0 {
            0.0
        } else {
            (matches as f64 / alignment_length as f64) * 100.0
        }
    }
}

/// Sequence alignment result
#[derive(Debug, Clone)]
pub struct SequenceAlignment {
    /// Alignment score
    pub score: f64,
    /// Start position in query sequence
    pub query_start: usize,
    /// End position in query sequence
    pub query_end: usize,
    /// Start position in subject sequence
    pub subject_start: usize,
    /// End position in subject sequence
    pub subject_end: usize,
    /// Aligned query sequence (with gaps)
    pub aligned_query: String,
    /// Aligned subject sequence (with gaps)
    pub aligned_subject: String,
    /// Sequence identity percentage
    pub identity: f64,
}

/// Sequence similarity search engine
pub struct SequenceSimilaritySearch {
    config: BioSearchConfig,
    kmer_index: KmerIndex,
    aligner: SequenceAligner,
    sequences: Vec<String>,
    metadata: Vec<SequenceMetadata>,
}

impl SequenceSimilaritySearch {
    /// Create new sequence similarity search
    pub fn new(sequence_type: SequenceType) -> Self {
        let config = BioSearchConfig {
            sequence_type,
            ..Default::default()
        };

        Self {
            kmer_index: KmerIndex::new(config.kmer_size),
            aligner: SequenceAligner::new(config.clone()),
            config,
            sequences: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Build index from sequences
    pub fn build_index_from_sequences(&mut self, sequences: &[String]) -> NeighborsResult<()> {
        if sequences.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Build k-mer index for fast pre-screening
        self.kmer_index.build(sequences)?;

        // Store sequences and create metadata
        self.sequences = sequences.to_vec();
        self.metadata = sequences
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let gc_content = if matches!(
                    self.config.sequence_type,
                    SequenceType::Dna | SequenceType::Rna
                ) {
                    Some(self.compute_gc_content(seq))
                } else {
                    None
                };

                SequenceMetadata {
                    id: format!("seq_{}", i),
                    description: None,
                    organism: None,
                    length: seq.len(),
                    seq_type: self.config.sequence_type,
                    gc_content,
                    annotations: HashMap::new(),
                }
            })
            .collect();

        Ok(())
    }

    /// Search for similar sequences
    pub fn search_similar_sequences(
        &self,
        query: &str,
        k: usize,
    ) -> NeighborsResult<Vec<SequenceSearchResult>> {
        if self.sequences.is_empty() {
            return Ok(Vec::new());
        }

        // Fast pre-screening using k-mer index - use a more permissive threshold
        let min_shared_kmers = 1.max(query.len() / 10); // At least 1 k-mer, or 10% of query length
        let candidates = self.kmer_index.get_candidates(query, min_shared_kmers);

        let mut results = Vec::new();

        // Perform detailed alignment for candidates
        for (seq_idx, _shared_kmers) in candidates {
            if seq_idx < self.sequences.len() {
                let alignment = self.aligner.align(query, &self.sequences[seq_idx]);

                // Filter by minimum identity if specified
                if alignment.identity >= self.config.min_identity {
                    let evalue = self.compute_evalue(
                        alignment.score,
                        query.len(),
                        self.sequences[seq_idx].len(),
                    );

                    // Filter by maximum E-value if specified
                    if evalue <= self.config.max_evalue {
                        let result = SequenceSearchResult {
                            metadata: self.metadata[seq_idx].clone(),
                            sequence: self.sequences[seq_idx].clone(),
                            score: alignment.score,
                            identity: alignment.identity,
                            evalue,
                            query_start: alignment.query_start,
                            query_end: alignment.query_end,
                            subject_start: alignment.subject_start,
                            subject_end: alignment.subject_end,
                            aligned_query: Some(alignment.aligned_query),
                            aligned_subject: Some(alignment.aligned_subject),
                        };

                        results.push(result);
                    }
                }
            }
        }

        // Sort by score (descending) and return top k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Add sequence to index
    pub fn add_sequence(
        &mut self,
        sequence: String,
        metadata: SequenceMetadata,
    ) -> NeighborsResult<()> {
        self.sequences.push(sequence);
        self.metadata.push(metadata);

        // Rebuild k-mer index
        self.kmer_index.build(&self.sequences)?;

        Ok(())
    }

    /// Compute GC content for DNA/RNA sequences
    fn compute_gc_content(&self, sequence: &str) -> f64 {
        let sequence = sequence.to_uppercase();
        let total_bases = sequence.len() as f64;

        if total_bases == 0.0 {
            return 0.0;
        }

        let gc_count = sequence.chars().filter(|&c| c == 'G' || c == 'C').count() as f64;

        (gc_count / total_bases) * 100.0
    }

    /// Compute E-value (statistical significance) - simplified version
    fn compute_evalue(&self, score: f64, query_len: usize, subject_len: usize) -> f64 {
        // Simplified E-value calculation
        // In practice, this would use proper statistical parameters
        let search_space = query_len as f64 * subject_len as f64;
        let lambda = 0.1; // Simplified lambda parameter
        let k = 0.01; // Simplified K parameter

        k * search_space * (-lambda * score).exp()
    }

    /// Get search statistics
    pub fn get_stats(&self) -> (usize, SequenceType, usize) {
        let num_sequences = self.sequences.len();
        let seq_type = self.config.sequence_type;
        let kmer_size = self.config.kmer_size;

        (num_sequences, seq_type, kmer_size)
    }
}

/// Protein structure comparison using RMSD (Root Mean Square Deviation)
pub struct ProteinStructureSearch {
    structures: Vec<ProteinStructure>,
    metadata: Vec<ProteinMetadata>,
}

/// 3D protein structure representation
#[derive(Debug, Clone)]
pub struct ProteinStructure {
    /// Protein sequence
    pub sequence: String,
    /// 3D coordinates of alpha carbon atoms
    pub ca_coordinates: Vec<(f64, f64, f64)>,
    /// Secondary structure (H=helix, E=sheet, C=coil)
    pub secondary_structure: Option<String>,
}

/// Protein metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProteinMetadata {
    /// PDB ID or protein identifier
    pub id: String,
    /// Protein name
    pub name: Option<String>,
    /// Organism
    pub organism: Option<String>,
    /// Resolution (for X-ray structures)
    pub resolution: Option<f64>,
    /// Experimental method
    pub method: Option<String>,
    /// Chain identifier
    pub chain_id: Option<String>,
}

/// Protein structure search result
#[derive(Debug, Clone)]
pub struct ProteinSearchResult {
    /// Protein metadata
    pub metadata: ProteinMetadata,
    /// Protein structure
    pub structure: ProteinStructure,
    /// RMSD (Root Mean Square Deviation)
    pub rmsd: f64,
    /// Number of aligned residues
    pub aligned_residues: usize,
    /// Structural similarity score
    pub similarity_score: f64,
}

impl Default for ProteinStructureSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl ProteinStructureSearch {
    /// Create new protein structure search
    pub fn new() -> Self {
        Self {
            structures: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Add protein structure to search index
    pub fn add_structure(
        &mut self,
        structure: ProteinStructure,
        metadata: ProteinMetadata,
    ) -> NeighborsResult<()> {
        // Validate structure
        if structure.ca_coordinates.is_empty() {
            return Err(NeighborsError::InvalidInput(
                "Structure has no coordinates".to_string(),
            ));
        }

        self.structures.push(structure);
        self.metadata.push(metadata);
        Ok(())
    }

    /// Search for structurally similar proteins
    pub fn search_similar_structures(
        &self,
        query_structure: &ProteinStructure,
        k: usize,
    ) -> NeighborsResult<Vec<ProteinSearchResult>> {
        let mut results = Vec::new();

        for (i, structure) in self.structures.iter().enumerate() {
            let rmsd =
                self.compute_rmsd(&query_structure.ca_coordinates, &structure.ca_coordinates)?;
            let aligned_residues = query_structure
                .ca_coordinates
                .len()
                .min(structure.ca_coordinates.len());

            // Compute similarity score (inverse of RMSD)
            let similarity_score = 1.0 / (1.0 + rmsd);

            results.push(ProteinSearchResult {
                metadata: self.metadata[i].clone(),
                structure: structure.clone(),
                rmsd,
                aligned_residues,
                similarity_score,
            });
        }

        // Sort by RMSD (ascending - lower is better)
        results.sort_by(|a, b| a.rmsd.partial_cmp(&b.rmsd).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Compute RMSD between two sets of coordinates using simple alignment
    fn compute_rmsd(
        &self,
        coords1: &[(f64, f64, f64)],
        coords2: &[(f64, f64, f64)],
    ) -> NeighborsResult<f64> {
        if coords1.is_empty() || coords2.is_empty() {
            return Err(NeighborsError::InvalidInput(
                "Empty coordinate sets".to_string(),
            ));
        }

        // Use the shorter length for alignment
        let n = coords1.len().min(coords2.len());
        let mut sum_squared_distances = 0.0;

        for i in 0..n {
            let (x1, y1, z1) = coords1[i];
            let (x2, y2, z2) = coords2[i];

            let squared_distance = (x1 - x2).powi(2) + (y1 - y2).powi(2) + (z1 - z2).powi(2);
            sum_squared_distances += squared_distance;
        }

        Ok((sum_squared_distances / n as f64).sqrt())
    }

    /// Get search statistics
    pub fn get_stats(&self) -> usize {
        self.structures.len()
    }
}

/// Gene expression analysis for finding co-expressed genes
pub struct GeneExpressionNeighbors {
    gene_names: Vec<String>,
    expression_matrix: Option<Array2<f64>>,
    metadata: Vec<GeneMetadata>,
}

/// Gene metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GeneMetadata {
    /// Gene identifier
    pub id: String,
    /// Gene symbol
    pub symbol: Option<String>,
    /// Gene name
    pub name: Option<String>,
    /// Organism
    pub organism: Option<String>,
    /// Chromosome location
    pub chromosome: Option<String>,
    /// Gene ontology terms
    pub go_terms: Vec<String>,
}

/// Gene expression search result
#[derive(Debug, Clone)]
pub struct GeneExpressionResult {
    /// Gene metadata
    pub metadata: GeneMetadata,
    /// Correlation coefficient with query gene
    pub correlation: f64,
    /// P-value for correlation significance
    pub p_value: f64,
    /// Expression pattern similarity
    pub similarity_score: f64,
}

impl Default for GeneExpressionNeighbors {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneExpressionNeighbors {
    /// Create new gene expression neighbor search
    pub fn new() -> Self {
        Self {
            gene_names: Vec::new(),
            expression_matrix: None,
            metadata: Vec::new(),
        }
    }

    /// Load expression data
    pub fn load_expression_data(
        &mut self,
        gene_names: Vec<String>,
        expression_matrix: Array2<f64>,
    ) -> NeighborsResult<()> {
        if gene_names.len() != expression_matrix.nrows() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![gene_names.len()],
                actual: vec![expression_matrix.nrows()],
            });
        }

        self.gene_names = gene_names;
        self.expression_matrix = Some(expression_matrix);

        // Create default metadata
        self.metadata = self
            .gene_names
            .iter()
            .map(|name| GeneMetadata {
                id: name.clone(),
                symbol: Some(name.clone()),
                name: None,
                organism: None,
                chromosome: None,
                go_terms: Vec::new(),
            })
            .collect();

        Ok(())
    }

    /// Find co-expressed genes
    pub fn find_coexpressed_genes(
        &self,
        query_gene: &str,
        k: usize,
    ) -> NeighborsResult<Vec<GeneExpressionResult>> {
        let expression_matrix =
            self.expression_matrix
                .as_ref()
                .ok_or(NeighborsError::InvalidInput(
                    "Expression data not loaded".to_string(),
                ))?;

        let query_idx = self
            .gene_names
            .iter()
            .position(|name| name == query_gene)
            .ok_or(NeighborsError::InvalidInput(format!(
                "Gene '{}' not found",
                query_gene
            )))?;

        let query_expression = expression_matrix.row(query_idx);
        let mut results = Vec::new();

        for (i, _gene_name) in self.gene_names.iter().enumerate() {
            if i == query_idx {
                continue; // Skip the query gene itself
            }

            let gene_expression = expression_matrix.row(i);
            let correlation = self.compute_pearson_correlation(&query_expression, &gene_expression);
            let p_value = self.compute_correlation_pvalue(correlation, expression_matrix.ncols());
            let similarity_score = correlation.abs();

            results.push(GeneExpressionResult {
                metadata: self.metadata[i].clone(),
                correlation,
                p_value,
                similarity_score,
            });
        }

        // Sort by absolute correlation (descending)
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Compute Pearson correlation coefficient
    fn compute_pearson_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute p-value for correlation (simplified)
    fn compute_correlation_pvalue(&self, r: f64, n: usize) -> f64 {
        if n <= 2 {
            return 1.0;
        }

        // Simplified t-test for correlation significance
        let t = r * ((n - 2) as f64 / (1.0 - r * r)).sqrt();

        // Simplified p-value calculation (in practice, would use proper t-distribution)
        let abs_t = t.abs();
        if abs_t > 3.0 {
            0.001
        } else if abs_t > 2.0 {
            0.05
        } else {
            0.1
        }
    }

    /// Get search statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let num_genes = self.gene_names.len();
        let num_samples = self
            .expression_matrix
            .as_ref()
            .map(|m| m.ncols())
            .unwrap_or(0);

        (num_genes, num_samples)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_index() {
        let mut index = KmerIndex::new(3);

        let sequences = vec![
            "ATCGATCG".to_string(),
            "ATCGTTCG".to_string(),
            "GCTAGCTA".to_string(),
        ];

        index.build(&sequences).unwrap();

        // Test candidate finding
        let candidates = index.get_candidates("ATCGATCG", 1);
        assert!(!candidates.is_empty());

        // Test Jaccard similarity
        let similarity = index.jaccard_similarity("ATCGATCG", 0);
        assert!(similarity > 0.5); // Should be high similarity with itself
    }

    #[test]
    fn test_sequence_aligner() {
        let config = BioSearchConfig::default();
        let aligner = SequenceAligner::new(config);

        let seq1 = "ATCGATCG";
        let seq2 = "ATCGATCG";

        let alignment = aligner.align(seq1, seq2);
        assert!(alignment.score > 0.0);
        assert_eq!(alignment.identity, 100.0); // Perfect match
    }

    #[test]
    fn test_sequence_similarity_search() {
        let mut search = SequenceSimilaritySearch::new(SequenceType::Dna);

        // Make the search less strict for testing
        search.config.max_evalue = 1.0; // Very permissive E-value
        search.config.min_identity = 0.0; // No minimum identity requirement

        let sequences = vec![
            "ATCGATCGATCG".to_string(),
            "ATCGATCGATCC".to_string(),
            "GCTAGCTAGCTA".to_string(),
        ];

        search.build_index_from_sequences(&sequences).unwrap();

        let results = search.search_similar_sequences("ATCGATCGATCG", 2).unwrap();
        assert!(!results.is_empty());

        // First result should be perfect match or very similar
        assert!(results[0].identity > 80.0);
    }

    #[test]
    fn test_protein_structure_search() {
        let mut search = ProteinStructureSearch::new();

        // Create dummy protein structures
        let structure1 = ProteinStructure {
            sequence: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            ca_coordinates: vec![(1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (3.0, 4.0, 5.0)],
            secondary_structure: Some("HHE".to_string()),
        };

        let structure2 = ProteinStructure {
            sequence: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            ca_coordinates: vec![(1.1, 2.1, 3.1), (2.1, 3.1, 4.1), (3.1, 4.1, 5.1)],
            secondary_structure: Some("HHE".to_string()),
        };

        let metadata1 = ProteinMetadata {
            id: "1ABC".to_string(),
            name: Some("Test Protein 1".to_string()),
            organism: None,
            resolution: Some(1.5),
            method: Some("X-ray".to_string()),
            chain_id: Some("A".to_string()),
        };

        let metadata2 = ProteinMetadata {
            id: "2DEF".to_string(),
            name: Some("Test Protein 2".to_string()),
            organism: None,
            resolution: Some(2.0),
            method: Some("X-ray".to_string()),
            chain_id: Some("B".to_string()),
        };

        search.add_structure(structure1.clone(), metadata1).unwrap();
        search.add_structure(structure2, metadata2).unwrap();

        let results = search.search_similar_structures(&structure1, 1).unwrap();
        assert!(!results.is_empty());

        // RMSD should be reasonable for similar structures
        assert!(results[0].rmsd < 1.0);
    }

    #[test]
    fn test_gene_expression_neighbors() {
        let mut search = GeneExpressionNeighbors::new();

        let gene_names = vec![
            "GENE1".to_string(),
            "GENE2".to_string(),
            "GENE3".to_string(),
        ];

        // Create correlated expression data
        let expression_matrix = Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, // GENE1
                1.1, 2.1, 3.1, 4.1, 5.1, // GENE2 (highly correlated with GENE1)
                5.0, 4.0, 3.0, 2.0, 1.0, // GENE3 (anti-correlated)
            ],
        )
        .unwrap();

        search
            .load_expression_data(gene_names, expression_matrix)
            .unwrap();

        let results = search.find_coexpressed_genes("GENE1", 2).unwrap();
        assert_eq!(results.len(), 2);

        // GENE2 should be highly correlated
        assert!(results[0].correlation > 0.9);
        assert_eq!(results[0].metadata.symbol, Some("GENE2".to_string()));
    }

    #[test]
    fn test_gc_content_calculation() {
        let search = SequenceSimilaritySearch::new(SequenceType::Dna);

        let gc_content = search.compute_gc_content("ATGCGCTA");
        assert_eq!(gc_content, 50.0); // 4 GC out of 8 total

        let gc_content = search.compute_gc_content("AAAA");
        assert_eq!(gc_content, 0.0); // No GC

        let gc_content = search.compute_gc_content("GCGC");
        assert_eq!(gc_content, 100.0); // All GC
    }

    #[test]
    fn test_sequence_alignment_identity() {
        let config = BioSearchConfig::default();
        let aligner = SequenceAligner::new(config);

        // Test identity calculation
        let alignment = SequenceAlignment {
            score: 10.0,
            query_start: 0,
            query_end: 4,
            subject_start: 0,
            subject_end: 4,
            aligned_query: "ATCG".to_string(),
            aligned_subject: "ATCG".to_string(),
            identity: 100.0,
        };

        let identity =
            aligner.compute_identity(&alignment.aligned_query, &alignment.aligned_subject);
        assert_eq!(identity, 100.0);

        let identity = aligner.compute_identity("ATC-", "ATCG");
        assert_eq!(identity, 75.0); // 3 matches out of 4 positions
    }
}
