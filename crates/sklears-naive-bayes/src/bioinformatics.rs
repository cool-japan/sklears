//! Bioinformatics specific Naive Bayes implementations
//!
//! This module provides specialized Naive Bayes classifiers for bioinformatics tasks,
//! including genomic sequence classification, protein structure prediction, phylogenetic
//! classification, gene expression analysis, and biomarker discovery.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::Rng;
use sklears_core::traits::Estimator;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BioinformaticsError {
    #[error("Invalid sequence: {0}")]
    InvalidSequence(String),
    #[error("Sequence length mismatch: expected {expected}, got {actual}")]
    SequenceLengthMismatch { expected: usize, actual: usize },
    #[error("Invalid nucleotide or amino acid: {0}")]
    InvalidResidue(char),
    #[error("Invalid k-mer size: {0}")]
    InvalidKmerSize(usize),
    #[error("Phylogenetic tree error: {0}")]
    PhylogeneticError(String),
    #[error("Gene expression data error: {0}")]
    GeneExpressionError(String),
    #[error("Insufficient data for analysis")]
    InsufficientData,
    #[error("Biomarker analysis error: {0}")]
    BiomarkerError(String),
}

type Result<T> = std::result::Result<T, BioinformaticsError>;

/// Sequence types for bioinformatics analysis
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceType {
    /// DNA
    DNA,
    /// RNA
    RNA,
    /// Protein
    Protein,
}

/// Nucleotide encoding for DNA/RNA sequences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nucleotide {
    /// A
    A = 0,
    /// T
    T = 1, // Also U for RNA
    /// G
    G = 2,
    /// C
    C = 3,
    /// N
    N = 4, // Unknown/ambiguous
}

impl Nucleotide {
    pub fn from_char(c: char, seq_type: &SequenceType) -> Result<Self> {
        match c.to_ascii_uppercase() {
            'A' => Ok(Nucleotide::A),
            'T' => {
                if *seq_type == SequenceType::RNA {
                    Err(BioinformaticsError::InvalidResidue(c))
                } else {
                    Ok(Nucleotide::T)
                }
            }
            'U' => {
                if *seq_type == SequenceType::RNA {
                    Ok(Nucleotide::T) // Treat U as T internally
                } else {
                    Err(BioinformaticsError::InvalidResidue(c))
                }
            }
            'G' => Ok(Nucleotide::G),
            'C' => Ok(Nucleotide::C),
            'N' | '-' => Ok(Nucleotide::N),
            _ => Err(BioinformaticsError::InvalidResidue(c)),
        }
    }

    pub fn to_index(self) -> usize {
        self as usize
    }
}

/// Amino acid encoding for protein sequences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AminoAcid {
    /// A
    A = 0,
    /// R
    R = 1,
    /// N
    N = 2,
    /// D
    D = 3,
    /// C
    C = 4,
    /// Q
    Q = 5,
    /// E
    E = 6,
    /// G
    G = 7,
    /// H
    H = 8,
    /// I
    I = 9,
    L = 10,
    K = 11,
    M = 12,
    F = 13,
    P = 14,
    S = 15,
    T = 16,
    W = 17,
    Y = 18,
    V = 19,
    X = 20, // Unknown/ambiguous
}

impl AminoAcid {
    pub fn from_char(c: char) -> Result<Self> {
        match c.to_ascii_uppercase() {
            'A' => Ok(AminoAcid::A),
            'R' => Ok(AminoAcid::R),
            'N' => Ok(AminoAcid::N),
            'D' => Ok(AminoAcid::D),
            'C' => Ok(AminoAcid::C),
            'Q' => Ok(AminoAcid::Q),
            'E' => Ok(AminoAcid::E),
            'G' => Ok(AminoAcid::G),
            'H' => Ok(AminoAcid::H),
            'I' => Ok(AminoAcid::I),
            'L' => Ok(AminoAcid::L),
            'K' => Ok(AminoAcid::K),
            'M' => Ok(AminoAcid::M),
            'F' => Ok(AminoAcid::F),
            'P' => Ok(AminoAcid::P),
            'S' => Ok(AminoAcid::S),
            'T' => Ok(AminoAcid::T),
            'W' => Ok(AminoAcid::W),
            'Y' => Ok(AminoAcid::Y),
            'V' => Ok(AminoAcid::V),
            'X' | '-' => Ok(AminoAcid::X),
            _ => Err(BioinformaticsError::InvalidResidue(c)),
        }
    }

    pub fn to_index(self) -> usize {
        self as usize
    }
}

/// Genomic sequence representation
#[derive(Debug, Clone)]
pub struct GenomicSequence {
    pub sequence: String,
    pub seq_type: SequenceType,
    pub metadata: SequenceMetadata,
}

#[derive(Debug, Clone)]
pub struct SequenceMetadata {
    pub id: String,
    pub description: String,
    pub organism: String,
    pub length: usize,
}

/// Configuration for genomic sequence classification
#[derive(Debug, Clone)]
pub struct GenomicNBConfig {
    pub kmer_size: usize,
    pub use_reverse_complement: bool,
    pub normalize_length: bool,
    pub sequence_type: SequenceType,
    pub smoothing_alpha: f64,
    pub max_sequence_length: Option<usize>,
}

impl Default for GenomicNBConfig {
    fn default() -> Self {
        Self {
            kmer_size: 3,
            use_reverse_complement: true,
            normalize_length: true,
            sequence_type: SequenceType::DNA,
            smoothing_alpha: 1.0,
            max_sequence_length: Some(10000),
        }
    }
}

/// Genomic sequence Naive Bayes classifier
#[derive(Debug, Clone)]
pub struct GenomicNaiveBayes {
    config: GenomicNBConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    kmer_log_probs: Vec<HashMap<String, f64>>, // K-mer log probabilities per class
    sequence_length_stats: Vec<(f64, f64)>,    // Mean and std of sequence lengths per class
    is_fitted: bool,
}

impl GenomicNaiveBayes {
    pub fn new(config: GenomicNBConfig) -> Self {
        if config.kmer_size == 0 {
            panic!("K-mer size must be greater than 0");
        }

        Self {
            config,
            classes: Array1::zeros(0),
            class_log_prior: Array1::zeros(0),
            kmer_log_probs: Vec::new(),
            sequence_length_stats: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, sequences: &[GenomicSequence], y: &Array1<i32>) -> Result<()> {
        if sequences.len() != y.len() {
            return Err(BioinformaticsError::InvalidSequence(
                "Number of sequences must match number of labels".to_string(),
            ));
        }

        // Validate sequences
        self.validate_sequences(sequences)?;

        // Extract unique classes
        let unique_classes: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        self.classes = Array1::from_vec(unique_classes);

        // Compute class priors
        self.class_log_prior = self.compute_class_log_prior(y)?;

        // Initialize storage
        let n_classes = self.classes.len();
        self.kmer_log_probs = vec![HashMap::new(); n_classes];
        self.sequence_length_stats = vec![(0.0, 0.0); n_classes];

        // Process sequences for each class
        for (class_idx, &class_label) in self.classes.iter().enumerate() {
            let class_sequences: Vec<&GenomicSequence> = sequences
                .iter()
                .zip(y.iter())
                .filter(|(_, &label)| label == class_label)
                .map(|(seq, _)| seq)
                .collect();

            if class_sequences.is_empty() {
                return Err(BioinformaticsError::InsufficientData);
            }

            // Compute k-mer frequencies
            self.kmer_log_probs[class_idx] =
                self.compute_class_kmer_probabilities(&class_sequences)?;

            // Compute sequence length statistics
            if self.config.normalize_length {
                self.sequence_length_stats[class_idx] = self.compute_length_stats(&class_sequences);
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, sequences: &[GenomicSequence]) -> Result<Array1<i32>> {
        if !self.is_fitted {
            return Err(BioinformaticsError::InsufficientData);
        }

        self.validate_sequences(sequences)?;

        let mut predictions = Array1::zeros(sequences.len());

        for (i, sequence) in sequences.iter().enumerate() {
            let log_probs = self.predict_log_proba_single(sequence)?;
            let best_class_idx = log_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            predictions[i] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }

    pub fn predict_proba(&self, sequences: &[GenomicSequence]) -> Result<Array2<f64>> {
        if !self.is_fitted {
            return Err(BioinformaticsError::InsufficientData);
        }

        self.validate_sequences(sequences)?;

        let mut probabilities = Array2::zeros((sequences.len(), self.classes.len()));

        for (i, sequence) in sequences.iter().enumerate() {
            let log_probs = self.predict_log_proba_single(sequence)?;

            // Convert log probabilities to probabilities
            let max_log_prob = log_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_probs: Vec<f64> = log_probs
                .iter()
                .map(|&p| (p - max_log_prob).exp())
                .collect();
            let sum_exp: f64 = exp_probs.iter().sum();

            for (j, &prob) in exp_probs.iter().enumerate() {
                probabilities[[i, j]] = prob / sum_exp;
            }
        }

        Ok(probabilities)
    }

    fn validate_sequences(&self, sequences: &[GenomicSequence]) -> Result<()> {
        for seq in sequences {
            if seq.seq_type != self.config.sequence_type {
                return Err(BioinformaticsError::InvalidSequence(format!(
                    "Expected {:?} sequence, got {:?}",
                    self.config.sequence_type, seq.seq_type
                )));
            }

            if let Some(max_len) = self.config.max_sequence_length {
                if seq.sequence.len() > max_len {
                    return Err(BioinformaticsError::InvalidSequence(format!(
                        "Sequence length {} exceeds maximum {}",
                        seq.sequence.len(),
                        max_len
                    )));
                }
            }

            if seq.sequence.len() < self.config.kmer_size {
                return Err(BioinformaticsError::InvalidSequence(format!(
                    "Sequence length {} is less than k-mer size {}",
                    seq.sequence.len(),
                    self.config.kmer_size
                )));
            }
        }
        Ok(())
    }

    fn predict_log_proba_single(&self, sequence: &GenomicSequence) -> Result<Array1<f64>> {
        let mut log_probs = self.class_log_prior.clone();

        // Extract k-mers from sequence
        let kmers = self.extract_kmers(&sequence.sequence)?;

        // Compute k-mer likelihoods for each class
        for (class_idx, class_kmer_probs) in self.kmer_log_probs.iter().enumerate() {
            let mut kmer_log_likelihood = 0.0;

            for kmer in &kmers {
                let kmer_log_prob = class_kmer_probs.get(kmer).copied().unwrap_or(-10.0); // Small probability for unseen k-mers
                kmer_log_likelihood += kmer_log_prob;
            }

            // Normalize by number of k-mers if length normalization is enabled
            if self.config.normalize_length && !kmers.is_empty() {
                kmer_log_likelihood /= kmers.len() as f64;
            }

            log_probs[class_idx] += kmer_log_likelihood;

            // Add sequence length likelihood if enabled
            if self.config.normalize_length {
                let (mean_len, std_len) = self.sequence_length_stats[class_idx];
                if std_len > 0.0 {
                    let length_log_likelihood = -0.5
                        * ((sequence.sequence.len() as f64 - mean_len) / std_len).powi(2)
                        - std_len.ln()
                        - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    log_probs[class_idx] += length_log_likelihood;
                }
            }
        }

        Ok(log_probs)
    }

    fn compute_class_log_prior(&self, y: &Array1<i32>) -> Result<Array1<f64>> {
        let n_samples = y.len() as f64;
        let mut class_counts = Array1::zeros(self.classes.len());

        for &label in y.iter() {
            for (i, &class) in self.classes.iter().enumerate() {
                if label == class {
                    class_counts[i] += 1.0;
                    break;
                }
            }
        }

        let class_priors = &class_counts / n_samples;
        let class_log_prior = class_priors.mapv(|p: f64| (p + 1e-10).ln());
        Ok(class_log_prior)
    }

    fn compute_class_kmer_probabilities(
        &self,
        sequences: &[&GenomicSequence],
    ) -> Result<HashMap<String, f64>> {
        let mut kmer_counts: HashMap<String, usize> = HashMap::new();
        let mut total_kmers = 0;

        for sequence in sequences {
            let kmers = self.extract_kmers(&sequence.sequence)?;
            for kmer in kmers {
                *kmer_counts.entry(kmer).or_insert(0) += 1;
                total_kmers += 1;
            }
        }

        // Convert counts to log probabilities with smoothing
        let mut kmer_log_probs = HashMap::new();
        let vocab_size = 4_usize.pow(self.config.kmer_size as u32); // 4^k possible k-mers

        for (kmer, count) in kmer_counts {
            let smoothed_count = count as f64 + self.config.smoothing_alpha;
            let smoothed_total =
                total_kmers as f64 + self.config.smoothing_alpha * vocab_size as f64;
            kmer_log_probs.insert(kmer, (smoothed_count / smoothed_total).ln());
        }

        Ok(kmer_log_probs)
    }

    fn compute_length_stats(&self, sequences: &[&GenomicSequence]) -> (f64, f64) {
        let lengths: Vec<f64> = sequences.iter().map(|s| s.sequence.len() as f64).collect();
        let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
        let variance =
            lengths.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / lengths.len() as f64;
        let std = variance.sqrt().max(1e-6); // Prevent division by zero
        (mean, std)
    }

    fn extract_kmers(&self, sequence: &str) -> Result<Vec<String>> {
        let mut kmers = Vec::new();

        if sequence.len() < self.config.kmer_size {
            return Ok(kmers);
        }

        // Extract forward k-mers
        for i in 0..=sequence.len() - self.config.kmer_size {
            let kmer = &sequence[i..i + self.config.kmer_size];
            if self.is_valid_kmer(kmer) {
                kmers.push(kmer.to_uppercase());
            }
        }

        // Extract reverse complement k-mers if enabled and sequence is DNA/RNA
        if self.config.use_reverse_complement
            && (self.config.sequence_type == SequenceType::DNA
                || self.config.sequence_type == SequenceType::RNA)
        {
            let rev_comp = self.reverse_complement(sequence)?;
            for i in 0..=rev_comp.len() - self.config.kmer_size {
                let kmer = &rev_comp[i..i + self.config.kmer_size];
                if self.is_valid_kmer(kmer) {
                    kmers.push(kmer.to_uppercase());
                }
            }
        }

        Ok(kmers)
    }

    fn is_valid_kmer(&self, kmer: &str) -> bool {
        match self.config.sequence_type {
            SequenceType::DNA => kmer
                .chars()
                .all(|c| matches!(c.to_ascii_uppercase(), 'A' | 'T' | 'G' | 'C')),
            SequenceType::RNA => kmer
                .chars()
                .all(|c| matches!(c.to_ascii_uppercase(), 'A' | 'U' | 'G' | 'C')),
            SequenceType::Protein => kmer.chars().all(|c| AminoAcid::from_char(c).is_ok()),
        }
    }

    fn reverse_complement(&self, sequence: &str) -> Result<String> {
        let mut rev_comp = String::with_capacity(sequence.len());

        for c in sequence.chars().rev() {
            let complement = match c.to_ascii_uppercase() {
                'A' => 'T',
                'T' => 'A',
                'U' => 'A',
                'G' => 'C',
                'C' => 'G',
                'N' => 'N',
                _ => return Err(BioinformaticsError::InvalidResidue(c)),
            };
            rev_comp.push(complement);
        }

        Ok(rev_comp)
    }
}

/// Protein structure prediction using Naive Bayes
#[derive(Debug, Clone)]
pub struct ProteinStructureNB {
    config: ProteinStructureConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    amino_acid_probs: Vec<Array1<f64>>, // AA probabilities per secondary structure class
    dipeptide_probs: Vec<HashMap<String, f64>>, // Dipeptide probabilities per class
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub struct ProteinStructureConfig {
    pub window_size: usize,
    pub use_dipeptides: bool,
    pub use_physicochemical_properties: bool,
    pub smoothing_alpha: f64,
}

impl Default for ProteinStructureConfig {
    fn default() -> Self {
        Self {
            window_size: 7,
            use_dipeptides: true,
            use_physicochemical_properties: true,
            smoothing_alpha: 1.0,
        }
    }
}

/// Protein secondary structure: Helix, Sheet, Coil
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecondaryStructure {
    /// Helix
    Helix = 0,
    /// Sheet
    Sheet = 1,
    /// Coil
    Coil = 2,
}

impl SecondaryStructure {
    pub fn from_char(c: char) -> Result<Self> {
        match c.to_ascii_uppercase() {
            'H' => Ok(SecondaryStructure::Helix),
            'E' => Ok(SecondaryStructure::Sheet),
            'C' | '-' => Ok(SecondaryStructure::Coil),
            _ => Err(BioinformaticsError::InvalidResidue(c)),
        }
    }

    pub fn to_index(self) -> usize {
        self as usize
    }
}

impl ProteinStructureNB {
    pub fn new(config: ProteinStructureConfig) -> Self {
        Self {
            config,
            classes: Array1::zeros(0),
            class_log_prior: Array1::zeros(0),
            amino_acid_probs: Vec::new(),
            dipeptide_probs: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, proteins: &[GenomicSequence], structures: &[String]) -> Result<()> {
        if proteins.len() != structures.len() {
            return Err(BioinformaticsError::InvalidSequence(
                "Number of proteins must match number of structures".to_string(),
            ));
        }

        // Validate that all sequences are proteins
        for protein in proteins {
            if protein.seq_type != SequenceType::Protein {
                return Err(BioinformaticsError::InvalidSequence(
                    "All sequences must be proteins".to_string(),
                ));
            }
        }

        // Convert structure strings to class labels
        let mut y_data = Vec::new();
        for (protein, structure) in proteins.iter().zip(structures.iter()) {
            if protein.sequence.len() != structure.len() {
                return Err(BioinformaticsError::SequenceLengthMismatch {
                    expected: protein.sequence.len(),
                    actual: structure.len(),
                });
            }

            for c in structure.chars() {
                let ss = SecondaryStructure::from_char(c)?;
                y_data.push(ss.to_index() as i32);
            }
        }

        let y = Array1::from_vec(y_data);

        // Extract unique classes (should be 0, 1, 2 for H, E, C)
        self.classes = Array1::from_vec(vec![0, 1, 2]);

        // Compute class priors
        self.class_log_prior = self.compute_class_log_prior(&y)?;

        // Initialize storage
        let n_classes = 3;
        self.amino_acid_probs = vec![Array1::zeros(21); n_classes]; // 20 AA + X
        self.dipeptide_probs = vec![HashMap::new(); n_classes];

        // Process sequences to build amino acid and dipeptide frequencies
        let mut aa_counts = vec![Array1::zeros(21); n_classes];

        let seq_idx = 0;
        for (protein, structure) in proteins.iter().zip(structures.iter()) {
            for (i, (aa_char, ss_char)) in
                protein.sequence.chars().zip(structure.chars()).enumerate()
            {
                let aa = AminoAcid::from_char(aa_char)?;
                let ss = SecondaryStructure::from_char(ss_char)?;
                let class_idx = ss.to_index();

                // Count amino acids
                aa_counts[class_idx][aa.to_index()] += 1.0;

                // Count dipeptides if enabled
                if self.config.use_dipeptides && i > 0 {
                    let prev_aa = protein.sequence.chars().nth(i - 1).unwrap();
                    let dipeptide = format!("{}{}", prev_aa, aa_char);
                    *self.dipeptide_probs[class_idx]
                        .entry(dipeptide)
                        .or_insert(0.0) += 1.0;
                }
            }
        }

        // Convert counts to probabilities
        for (class_idx, aa_count) in aa_counts.iter().enumerate().take(n_classes) {
            let total_aa = aa_count.sum();
            if total_aa > 0.0 {
                self.amino_acid_probs[class_idx] = aa_count.mapv(|count: f64| {
                    let smoothed_prob = (count + self.config.smoothing_alpha)
                        / (total_aa + self.config.smoothing_alpha * 21.0);
                    smoothed_prob.ln()
                });
            }

            // Normalize dipeptide probabilities
            let total_dipeptides: f64 = self.dipeptide_probs[class_idx].values().sum();
            if total_dipeptides > 0.0 {
                for prob in self.dipeptide_probs[class_idx].values_mut() {
                    *prob = ((*prob + self.config.smoothing_alpha)
                        / (total_dipeptides + self.config.smoothing_alpha * 400.0))
                        .ln();
                }
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, proteins: &[GenomicSequence]) -> Result<Vec<String>> {
        if !self.is_fitted {
            return Err(BioinformaticsError::InsufficientData);
        }

        let mut predictions = Vec::new();

        for protein in proteins {
            if protein.seq_type != SequenceType::Protein {
                return Err(BioinformaticsError::InvalidSequence(
                    "All sequences must be proteins".to_string(),
                ));
            }

            let mut structure = String::with_capacity(protein.sequence.len());

            for (i, aa_char) in protein.sequence.chars().enumerate() {
                let aa = AminoAcid::from_char(aa_char)?;
                let mut log_probs = self.class_log_prior.clone();

                // Add amino acid likelihood
                for class_idx in 0..3 {
                    log_probs[class_idx] += self.amino_acid_probs[class_idx][aa.to_index()];

                    // Add dipeptide likelihood if available
                    if self.config.use_dipeptides && i > 0 {
                        let prev_aa = protein.sequence.chars().nth(i - 1).unwrap();
                        let dipeptide = format!("{}{}", prev_aa, aa_char);
                        let dipeptide_log_prob = self.dipeptide_probs[class_idx]
                            .get(&dipeptide)
                            .copied()
                            .unwrap_or(-10.0); // Small probability for unseen dipeptides
                        log_probs[class_idx] += dipeptide_log_prob;
                    }
                }

                // Find the most likely secondary structure
                let best_class_idx = log_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;

                let ss_char = match best_class_idx {
                    0 => 'H',
                    1 => 'E',
                    2 => 'C',
                    _ => 'C',
                };
                structure.push(ss_char);
            }

            predictions.push(structure);
        }

        Ok(predictions)
    }

    fn compute_class_log_prior(&self, y: &Array1<i32>) -> Result<Array1<f64>> {
        let n_samples = y.len() as f64;
        let mut class_counts = Array1::zeros(3);

        for &label in y.iter() {
            if (0..3).contains(&label) {
                class_counts[label as usize] += 1.0;
            }
        }

        let class_priors = &class_counts / n_samples;
        let class_log_prior = class_priors.mapv(|p: f64| (p + 1e-10).ln());
        Ok(class_log_prior)
    }
}

/// Phylogenetic classification using Naive Bayes
#[derive(Debug, Clone)]
pub struct PhylogeneticNB {
    config: PhylogeneticConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    evolutionary_distance_stats: Vec<(f64, f64)>, // Mean and std per class
    substitution_matrices: Vec<Array2<f64>>,      // Substitution probabilities per class
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub struct PhylogeneticConfig {
    pub use_evolutionary_distance: bool,
    pub use_substitution_patterns: bool,
    pub alignment_required: bool,
    pub smoothing_alpha: f64,
}

impl Default for PhylogeneticConfig {
    fn default() -> Self {
        Self {
            use_evolutionary_distance: true,
            use_substitution_patterns: true,
            alignment_required: true,
            smoothing_alpha: 1.0,
        }
    }
}

/// Gene expression analysis using Naive Bayes
#[derive(Debug, Clone)]
pub struct GeneExpressionNB {
    config: GeneExpressionConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    gene_expression_stats: Vec<Array2<f64>>, // Mean and std per gene per class
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub struct GeneExpressionConfig {
    pub normalize_expression: bool,
    pub log_transform: bool,
    pub feature_selection: bool,
    pub num_features: usize,
    pub smoothing_alpha: f64,
}

impl Default for GeneExpressionConfig {
    fn default() -> Self {
        Self {
            normalize_expression: true,
            log_transform: true,
            feature_selection: true,
            num_features: 1000,
            smoothing_alpha: 1.0,
        }
    }
}

impl GeneExpressionNB {
    pub fn new(config: GeneExpressionConfig) -> Self {
        Self {
            config,
            classes: Array1::zeros(0),
            class_log_prior: Array1::zeros(0),
            gene_expression_stats: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, expression_data: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        if expression_data.nrows() != y.len() {
            return Err(BioinformaticsError::GeneExpressionError(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        // Extract unique classes
        let unique_classes: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        self.classes = Array1::from_vec(unique_classes);

        // Compute class priors
        self.class_log_prior = self.compute_class_log_prior(y)?;

        // Process expression data
        let processed_data = self.preprocess_expression_data(expression_data)?;

        // Compute statistics for each class
        let n_classes = self.classes.len();
        let n_genes = processed_data.ncols();
        self.gene_expression_stats = Vec::with_capacity(n_classes);

        for &class_label in self.classes.iter() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class_label).collect();
            let class_data = self.extract_class_data(&processed_data, &class_mask);

            if class_data.nrows() == 0 {
                return Err(BioinformaticsError::InsufficientData);
            }

            let mut stats = Array2::zeros((2, n_genes)); // mean and std for each gene

            for gene_idx in 0..n_genes {
                let gene_values = class_data.column(gene_idx);
                let mean = gene_values.mean().unwrap_or(0.0);
                let std = gene_values.std(0.0).max(1e-6);

                stats[[0, gene_idx]] = mean;
                stats[[1, gene_idx]] = std;
            }

            self.gene_expression_stats.push(stats);
        }

        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, expression_data: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.is_fitted {
            return Err(BioinformaticsError::InsufficientData);
        }

        let processed_data = self.preprocess_expression_data(expression_data)?;
        let mut predictions = Array1::zeros(processed_data.nrows());

        for (sample_idx, sample) in processed_data.axis_iter(Axis(0)).enumerate() {
            let mut log_probs = self.class_log_prior.clone();

            for (class_idx, class_stats) in self.gene_expression_stats.iter().enumerate() {
                let mut class_log_likelihood = 0.0;

                for (gene_idx, &expression_value) in sample.iter().enumerate() {
                    let mean = class_stats[[0, gene_idx]];
                    let std = class_stats[[1, gene_idx]];

                    // Gaussian likelihood
                    let log_likelihood = -0.5 * ((expression_value - mean) / std).powi(2)
                        - std.ln()
                        - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    class_log_likelihood += log_likelihood;
                }

                log_probs[class_idx] += class_log_likelihood;
            }

            let best_class_idx = log_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            predictions[sample_idx] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }

    fn preprocess_expression_data(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut processed = data.clone();

        // Log transform if enabled
        if self.config.log_transform {
            processed.mapv_inplace(|x| (x + 1.0).ln());
        }

        // Normalize if enabled
        if self.config.normalize_expression {
            for mut sample in processed.axis_iter_mut(Axis(0)) {
                let mean = sample.mean().unwrap_or(0.0);
                let std = sample.std(0.0).max(1e-6);
                sample.mapv_inplace(|x| (x - mean) / std);
            }
        }

        Ok(processed)
    }

    fn extract_class_data(&self, data: &Array2<f64>, mask: &[bool]) -> Array2<f64> {
        let selected_rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &selected)| selected)
            .map(|(idx, _)| idx)
            .collect();

        let mut class_data = Array2::zeros((selected_rows.len(), data.ncols()));
        for (new_row, &old_row) in selected_rows.iter().enumerate() {
            class_data.row_mut(new_row).assign(&data.row(old_row));
        }

        class_data
    }

    fn compute_class_log_prior(&self, y: &Array1<i32>) -> Result<Array1<f64>> {
        let n_samples = y.len() as f64;
        let mut class_counts = Array1::zeros(self.classes.len());

        for &label in y.iter() {
            for (i, &class) in self.classes.iter().enumerate() {
                if label == class {
                    class_counts[i] += 1.0;
                    break;
                }
            }
        }

        let class_priors = &class_counts / n_samples;
        let class_log_prior = class_priors.mapv(|p: f64| (p + 1e-10).ln());
        Ok(class_log_prior)
    }
}

/// Biomarker discovery using Naive Bayes
#[derive(Debug, Clone)]
pub struct BiomarkerDiscoveryNB {
    config: BiomarkerConfig,
    classes: Array1<i32>,
    class_log_prior: Array1<f64>,
    biomarker_stats: Vec<Array2<f64>>, // Statistics per biomarker per class
    selected_features: Vec<usize>,     // Indices of selected biomarkers
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub struct BiomarkerConfig {
    pub feature_selection_method: FeatureSelectionMethod,
    pub num_biomarkers: usize,
    pub significance_threshold: f64,
    pub smoothing_alpha: f64,
}

#[derive(Debug, Clone)]
pub enum FeatureSelectionMethod {
    /// TTest
    TTest,
    /// MutualInformation
    MutualInformation,
    /// ChiSquare
    ChiSquare,
    /// FoldChange
    FoldChange,
}

impl Default for BiomarkerConfig {
    fn default() -> Self {
        Self {
            feature_selection_method: FeatureSelectionMethod::TTest,
            num_biomarkers: 100,
            significance_threshold: 0.05,
            smoothing_alpha: 1.0,
        }
    }
}

/// Utility functions for bioinformatics
pub mod utils {
    use super::*;

    /// Create a test genomic sequence
    pub fn create_test_sequence(length: usize, seq_type: SequenceType) -> GenomicSequence {
        let mut rng = scirs2_core::random::thread_rng();
        let mut sequence = String::with_capacity(length);

        match seq_type {
            SequenceType::DNA => {
                let nucleotides = ['A', 'T', 'G', 'C'];
                for _ in 0..length {
                    sequence.push(nucleotides[rng.gen_range(0..4)]);
                }
            }
            SequenceType::RNA => {
                let nucleotides = ['A', 'U', 'G', 'C'];
                for _ in 0..length {
                    sequence.push(nucleotides[rng.gen_range(0..4)]);
                }
            }
            SequenceType::Protein => {
                let amino_acids = [
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                    'T', 'W', 'Y', 'V',
                ];
                for _ in 0..length {
                    sequence.push(amino_acids[rng.gen_range(0..20)]);
                }
            }
        }

        GenomicSequence {
            sequence,
            seq_type,
            metadata: SequenceMetadata {
                id: format!("test_seq_{}", rng.random::<u32>()),
                description: "Test sequence".to_string(),
                organism: "Test organism".to_string(),
                length,
            },
        }
    }

    /// Calculate GC content of a DNA/RNA sequence
    pub fn gc_content(sequence: &str) -> f64 {
        let gc_count = sequence
            .chars()
            .filter(|&c| c.eq_ignore_ascii_case(&'G') || c.eq_ignore_ascii_case(&'C'))
            .count();
        gc_count as f64 / sequence.len() as f64
    }

    /// Generate random gene expression data
    pub fn create_test_expression_data(n_samples: usize, n_genes: usize) -> Array2<f64> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut data = Array2::zeros((n_samples, n_genes));

        for i in 0..n_samples {
            for j in 0..n_genes {
                // Simulate log-normal expression values
                data[[i, j]] = (-rng.gen::<f64>().ln()).exp();
            }
        }

        data
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_nucleotide_encoding() {
        assert_eq!(
            Nucleotide::from_char('A', &SequenceType::DNA).unwrap(),
            Nucleotide::A
        );
        assert_eq!(
            Nucleotide::from_char('T', &SequenceType::DNA).unwrap(),
            Nucleotide::T
        );
        assert_eq!(
            Nucleotide::from_char('U', &SequenceType::RNA).unwrap(),
            Nucleotide::T
        );
        assert!(Nucleotide::from_char('U', &SequenceType::DNA).is_err());
    }

    #[test]
    fn test_amino_acid_encoding() {
        assert_eq!(AminoAcid::from_char('A').unwrap(), AminoAcid::A);
        assert_eq!(AminoAcid::from_char('R').unwrap(), AminoAcid::R);
        assert_eq!(AminoAcid::from_char('X').unwrap(), AminoAcid::X);
        assert!(AminoAcid::from_char('Z').is_err());
    }

    #[test]
    fn test_genomic_nb_creation() {
        let config = GenomicNBConfig::default();
        let nb = GenomicNaiveBayes::new(config);
        assert!(!nb.is_fitted);
    }

    #[test]
    fn test_genomic_nb_fit_predict() {
        let mut nb = GenomicNaiveBayes::new(GenomicNBConfig::default());

        // Create test sequences
        let seq1 = create_test_sequence(100, SequenceType::DNA);
        let seq2 = create_test_sequence(100, SequenceType::DNA);
        let seq3 = create_test_sequence(100, SequenceType::DNA);

        let sequences = vec![seq1, seq2, seq3];
        let labels = Array1::from_vec(vec![0, 1, 0]);

        nb.fit(&sequences, &labels).unwrap();
        assert!(nb.is_fitted);

        let predictions = nb.predict(&sequences).unwrap();
        assert_eq!(predictions.len(), 3);

        let probabilities = nb.predict_proba(&sequences).unwrap();
        assert_eq!(probabilities.dim(), (3, 2));

        // Check probabilities sum to 1
        for i in 0..3 {
            let row_sum: f64 = probabilities.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kmer_extraction() {
        let config = GenomicNBConfig {
            kmer_size: 3,
            use_reverse_complement: false,
            ..Default::default()
        };
        let nb = GenomicNaiveBayes::new(config);

        let kmers = nb.extract_kmers("ATGCGA").unwrap();
        assert_eq!(kmers.len(), 4); // ATGCGA has 4 3-mers: ATG, TGC, GCG, CGA
        assert!(kmers.contains(&"ATG".to_string()));
        assert!(kmers.contains(&"TGC".to_string()));
        assert!(kmers.contains(&"GCG".to_string()));
        assert!(kmers.contains(&"CGA".to_string()));
    }

    #[test]
    fn test_reverse_complement() {
        let nb = GenomicNaiveBayes::new(GenomicNBConfig::default());

        let rev_comp = nb.reverse_complement("ATGC").unwrap();
        assert_eq!(rev_comp, "GCAT");

        let rev_comp2 = nb.reverse_complement("AAAGGG").unwrap();
        assert_eq!(rev_comp2, "CCCTTT");
    }

    #[test]
    fn test_protein_structure_nb_creation() {
        let config = ProteinStructureConfig::default();
        let nb = ProteinStructureNB::new(config);
        assert!(!nb.is_fitted);
    }

    #[test]
    fn test_protein_structure_nb_fit_predict() {
        let mut nb = ProteinStructureNB::new(ProteinStructureConfig::default());

        // Create test protein sequences
        let protein1 = create_test_sequence(50, SequenceType::Protein);
        let protein2 = create_test_sequence(50, SequenceType::Protein);

        // Create corresponding secondary structures
        let structure1 = "H".repeat(25) + &"C".repeat(25);
        let structure2 = "E".repeat(25) + &"C".repeat(25);

        let proteins = vec![protein1, protein2];
        let structures = vec![structure1, structure2];

        nb.fit(&proteins, &structures).unwrap();
        assert!(nb.is_fitted);

        let predictions = nb.predict(&proteins).unwrap();
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].len(), 50);
        assert_eq!(predictions[1].len(), 50);
    }

    #[test]
    fn test_secondary_structure_encoding() {
        assert_eq!(
            SecondaryStructure::from_char('H').unwrap(),
            SecondaryStructure::Helix
        );
        assert_eq!(
            SecondaryStructure::from_char('E').unwrap(),
            SecondaryStructure::Sheet
        );
        assert_eq!(
            SecondaryStructure::from_char('C').unwrap(),
            SecondaryStructure::Coil
        );
        assert!(SecondaryStructure::from_char('X').is_err());
    }

    #[test]
    fn test_gene_expression_nb_creation() {
        let config = GeneExpressionConfig::default();
        let nb = GeneExpressionNB::new(config);
        assert!(!nb.is_fitted);
    }

    #[test]
    fn test_gene_expression_nb_fit_predict() {
        let mut nb = GeneExpressionNB::new(GeneExpressionConfig::default());

        // Create test expression data
        let expression_data = create_test_expression_data(20, 100);
        let labels = Array1::from_vec(vec![0; 10].into_iter().chain(vec![1; 10]).collect());

        nb.fit(&expression_data, &labels).unwrap();
        assert!(nb.is_fitted);

        let predictions = nb.predict(&expression_data).unwrap();
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_gc_content() {
        assert_abs_diff_eq!(gc_content("ATGC"), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(gc_content("AAAA"), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gc_content("GGCC"), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_test_sequence_creation() {
        let dna_seq = create_test_sequence(100, SequenceType::DNA);
        assert_eq!(dna_seq.sequence.len(), 100);
        assert_eq!(dna_seq.seq_type, SequenceType::DNA);
        assert!(dna_seq
            .sequence
            .chars()
            .all(|c| matches!(c, 'A' | 'T' | 'G' | 'C')));

        let protein_seq = create_test_sequence(50, SequenceType::Protein);
        assert_eq!(protein_seq.sequence.len(), 50);
        assert_eq!(protein_seq.seq_type, SequenceType::Protein);
    }

    #[test]
    fn test_expression_data_preprocessing() {
        let nb = GeneExpressionNB::new(GeneExpressionConfig::default());
        let data = create_test_expression_data(10, 20);

        let processed = nb.preprocess_expression_data(&data).unwrap();
        assert_eq!(processed.dim(), data.dim());

        // Check that normalization worked (each sample should have mean ~0 and std ~1)
        for sample in processed.axis_iter(Axis(0)) {
            let mean = sample.mean().unwrap();
            let std = sample.std(0.0);
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(std, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sequence_validation() {
        let config = GenomicNBConfig::default();
        let nb = GenomicNaiveBayes::new(config);

        let valid_seq = create_test_sequence(100, SequenceType::DNA);
        let invalid_seq = create_test_sequence(100, SequenceType::Protein);

        assert!(nb.validate_sequences(&[valid_seq]).is_ok());
        assert!(nb.validate_sequences(&[invalid_seq]).is_err());
    }

    #[test]
    fn test_error_handling() {
        let mut nb = GenomicNaiveBayes::new(GenomicNBConfig::default());
        let seq = create_test_sequence(10, SequenceType::DNA);
        let empty_labels = Array1::from_vec(vec![]);

        // Test mismatched dimensions
        let result = nb.fit(&vec![seq], &empty_labels);
        assert!(result.is_err());

        // Test prediction before fitting
        let prediction_result = nb.predict(&vec![]);
        assert!(prediction_result.is_err());
    }
}
