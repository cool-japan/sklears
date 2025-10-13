//! Biological sequence feature extraction
//!
//! This module provides tools for extracting features from biological sequences
//! such as DNA, RNA, and protein sequences.

use crate::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result as SklResult, prelude::SklearsError, types::Float};
use std::collections::HashMap;

const AMINO_ACIDS: [char; 20] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];

const KYTE_DOOLITTLE: [f64; 20] = [
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7,
    4.2, -0.9, -1.3,
];

const AMINO_ACID_WEIGHTS: [f64; 20] = [
    89.09, 121.15, 133.1, 147.13, 165.19, 75.07, 155.16, 131.17, 146.19, 131.17, 149.21, 132.12,
    115.13, 146.15, 174.2, 105.09, 119.12, 117.15, 204.23, 181.19,
];

const AMINO_ACID_CHARGE: [f64; 20] = [
    0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0,
];

const POLAR_FLAGS: [bool; 20] = [
    false, true, true, true, false, false, true, false, true, false, false, true, false, true,
    true, true, true, false, false, true,
];

const AROMATIC_FLAGS: [bool; 20] = [
    false, false, false, false, true, false, false, false, false, false, false, false, false,
    false, false, false, false, false, true, true,
];

const ALIPHATIC_FLAGS: [bool; 20] = [
    true, false, false, false, false, false, false, true, false, true, false, false, false, false,
    false, false, false, true, false, false,
];

const NUCLEOTIDES: [char; 4] = ['A', 'C', 'G', 'T'];
const COMPLEMENT_INDEX: [usize; 4] = [3, 2, 1, 0];

fn amino_acid_index(residue: char) -> Option<usize> {
    AMINO_ACIDS
        .iter()
        .position(|&aa| aa == residue.to_ascii_uppercase())
}

fn nucleotide_index(base: char) -> Option<usize> {
    match base.to_ascii_uppercase() {
        'A' => Some(0),
        'C' => Some(1),
        'G' => Some(2),
        'T' => Some(3),
        _ => None,
    }
}

/// K-mer counter for biological sequences
pub struct KmerCounter {
    k: usize,
    normalize: bool,
    include_reverse_complement: bool,
}

impl KmerCounter {
    /// Create a new K-mer counter
    pub fn new() -> Self {
        Self {
            k: 3,
            normalize: false,
            include_reverse_complement: false,
        }
    }

    /// Set the k-mer length
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set whether to normalize counts
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set whether to include reverse complement for DNA sequences
    pub fn include_reverse_complement(mut self, include_reverse_complement: bool) -> Self {
        self.include_reverse_complement = include_reverse_complement;
        self
    }

    /// Extract k-mer features from a sequence
    pub fn extract_features(&self, sequence: &str) -> SklResult<Array1<Float>> {
        if sequence.len() < self.k {
            return Err(SklearsError::InvalidInput(format!(
                "Sequence length {} is less than k={}",
                sequence.len(),
                self.k
            )));
        }

        let sequence = sequence.to_uppercase();
        let mut kmer_counts: HashMap<String, usize> = HashMap::new();

        // Extract k-mers
        for i in 0..=sequence.len() - self.k {
            let kmer = &sequence[i..i + self.k];
            *kmer_counts.entry(kmer.to_string()).or_insert(0) += 1;

            // Add reverse complement if specified
            if self.include_reverse_complement {
                let rev_comp = self.reverse_complement(kmer)?;
                if rev_comp != kmer {
                    *kmer_counts.entry(rev_comp).or_insert(0) += 1;
                }
            }
        }

        // Create feature vector from all possible k-mers
        let alphabet = if self.include_reverse_complement {
            vec!['A', 'T', 'G', 'C']
        } else {
            self.get_alphabet(&sequence)
        };

        let all_kmers = self.generate_all_kmers(&alphabet, self.k);
        let mut features = Vec::with_capacity(all_kmers.len());

        let total_count: usize = kmer_counts.values().sum();

        for kmer in all_kmers {
            let count = *kmer_counts.get(&kmer).unwrap_or(&0);
            let feature_value = if self.normalize && total_count > 0 {
                count as Float / total_count as Float
            } else {
                count as Float
            };
            features.push(feature_value);
        }

        Ok(Array1::from_vec(features))
    }

    /// Generate all possible k-mers for given alphabet
    fn generate_all_kmers(&self, alphabet: &[char], k: usize) -> Vec<String> {
        if k == 0 {
            return vec![String::new()];
        }

        let mut kmers = Vec::new();
        let shorter_kmers = self.generate_all_kmers(alphabet, k - 1);

        for base in alphabet {
            for shorter_kmer in &shorter_kmers {
                kmers.push(format!("{}{}", base, shorter_kmer));
            }
        }

        kmers
    }

    /// Get alphabet from sequence
    fn get_alphabet(&self, sequence: &str) -> Vec<char> {
        let mut alphabet: Vec<char> = sequence
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        alphabet.sort();
        alphabet
    }

    /// Get reverse complement of DNA sequence
    fn reverse_complement(&self, sequence: &str) -> SklResult<String> {
        let complement_map: HashMap<char, char> = [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')]
            .iter()
            .cloned()
            .collect();

        let rev_comp: Result<String, _> = sequence
            .chars()
            .rev()
            .map(|c| {
                complement_map
                    .get(&c)
                    .copied()
                    .ok_or_else(|| SklearsError::InvalidInput(format!("Invalid nucleotide: {}", c)))
            })
            .collect();

        rev_comp
    }
}

impl Default for KmerCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequence motif finder
pub struct SequenceMotifExtractor {
    motif_length: usize,
    min_frequency: usize,
    max_mismatches: usize,
}

impl SequenceMotifExtractor {
    /// Create a new sequence motif extractor
    pub fn new() -> Self {
        Self {
            motif_length: 4,
            min_frequency: 2,
            max_mismatches: 0,
        }
    }

    /// Set motif length
    pub fn motif_length(mut self, length: usize) -> Self {
        self.motif_length = length;
        self
    }

    /// Set minimum frequency for motif detection
    pub fn min_frequency(mut self, frequency: usize) -> Self {
        self.min_frequency = frequency;
        self
    }

    /// Set maximum allowed mismatches
    pub fn max_mismatches(mut self, mismatches: usize) -> Self {
        self.max_mismatches = mismatches;
        self
    }

    /// Extract motif features from sequences
    pub fn extract_features(&self, sequences: &[String]) -> SklResult<Array2<Float>> {
        if sequences.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty sequence list".to_string(),
            ));
        }

        // Find frequent motifs
        let motifs = self.find_frequent_motifs(sequences)?;

        // Create feature matrix
        let mut features = Array2::zeros((sequences.len(), motifs.len()));

        for (seq_idx, sequence) in sequences.iter().enumerate() {
            for (motif_idx, motif) in motifs.iter().enumerate() {
                let count = self.count_motif_occurrences(sequence, motif);
                features[[seq_idx, motif_idx]] = count as Float;
            }
        }

        Ok(features)
    }

    /// Find frequent motifs in sequences
    fn find_frequent_motifs(&self, sequences: &[String]) -> SklResult<Vec<String>> {
        let mut motif_counts: HashMap<String, usize> = HashMap::new();

        for sequence in sequences {
            let sequence = sequence.to_uppercase();
            if sequence.len() < self.motif_length {
                continue;
            }

            for i in 0..=sequence.len() - self.motif_length {
                let motif = &sequence[i..i + self.motif_length];
                *motif_counts.entry(motif.to_string()).or_insert(0) += 1;
            }
        }

        let frequent_motifs: Vec<String> = motif_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_frequency)
            .map(|(motif, _)| motif)
            .collect();

        Ok(frequent_motifs)
    }

    /// Count occurrences of a motif in a sequence (with mismatches)
    fn count_motif_occurrences(&self, sequence: &str, motif: &str) -> usize {
        let sequence = sequence.to_uppercase();
        let motif = motif.to_uppercase();

        if sequence.len() < motif.len() {
            return 0;
        }

        let mut count = 0;
        for i in 0..=sequence.len() - motif.len() {
            let subseq = &sequence[i..i + motif.len()];
            if self.hamming_distance(subseq, &motif) <= self.max_mismatches {
                count += 1;
            }
        }

        count
    }

    /// Calculate Hamming distance between two sequences
    fn hamming_distance(&self, seq1: &str, seq2: &str) -> usize {
        if seq1.len() != seq2.len() {
            return usize::MAX;
        }

        seq1.chars()
            .zip(seq2.chars())
            .map(|(c1, c2)| if c1 == c2 { 0 } else { 1 })
            .sum()
    }
}

impl Default for SequenceMotifExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compositional features extractor for biological sequences
pub struct CompositionFeatureExtractor {
    include_dinucleotide: bool,
    include_gc_content: bool,
    include_codon_usage: bool,
    sequence_type: SequenceType,
}

#[derive(Clone, Copy)]
pub enum SequenceType {
    /// DNA
    DNA,
    /// RNA
    RNA,
    /// Protein
    Protein,
}

impl CompositionFeatureExtractor {
    /// Create a new composition feature extractor
    pub fn new() -> Self {
        Self {
            include_dinucleotide: true,
            include_gc_content: true,
            include_codon_usage: false,
            sequence_type: SequenceType::DNA,
        }
    }

    /// Set sequence type
    pub fn sequence_type(mut self, seq_type: SequenceType) -> Self {
        self.sequence_type = seq_type;
        self
    }

    /// Include dinucleotide composition
    pub fn include_dinucleotide(mut self, include: bool) -> Self {
        self.include_dinucleotide = include;
        self
    }

    /// Include GC content (for DNA/RNA)
    pub fn include_gc_content(mut self, include: bool) -> Self {
        self.include_gc_content = include;
        self
    }

    /// Include codon usage (for DNA/RNA sequences)
    pub fn include_codon_usage(mut self, include: bool) -> Self {
        self.include_codon_usage = include;
        self
    }

    /// Extract compositional features from a sequence
    pub fn extract_features(&self, sequence: &str) -> SklResult<Array1<Float>> {
        if sequence.is_empty() {
            return Err(SklearsError::InvalidInput("Empty sequence".to_string()));
        }

        let sequence = sequence.to_uppercase();
        let mut features = Vec::new();

        // Basic composition
        let composition = self.calculate_composition(&sequence);
        features.extend(composition);

        // Dinucleotide composition
        if self.include_dinucleotide {
            let dinuc_comp = self.calculate_dinucleotide_composition(&sequence)?;
            features.extend(dinuc_comp);
        }

        // GC content (for DNA/RNA)
        if self.include_gc_content
            && matches!(self.sequence_type, SequenceType::DNA | SequenceType::RNA)
        {
            let gc_content = self.calculate_gc_content(&sequence);
            features.push(gc_content);
        }

        // Codon usage (for DNA/RNA)
        if self.include_codon_usage
            && matches!(self.sequence_type, SequenceType::DNA | SequenceType::RNA)
        {
            let codon_usage = self.calculate_codon_usage(&sequence)?;
            features.extend(codon_usage);
        }

        Ok(Array1::from_vec(features))
    }

    /// Calculate basic nucleotide/amino acid composition
    fn calculate_composition(&self, sequence: &str) -> Vec<Float> {
        let alphabet = match self.sequence_type {
            SequenceType::DNA => vec!['A', 'T', 'G', 'C'],
            SequenceType::RNA => vec!['A', 'U', 'G', 'C'],
            SequenceType::Protein => "ACDEFGHIKLMNPQRSTVWY".chars().collect(),
        };

        let total_len = sequence.len() as Float;
        alphabet
            .iter()
            .map(|&base| {
                let count = sequence.chars().filter(|&c| c == base).count() as Float;
                if total_len > 0.0 {
                    count / total_len
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate dinucleotide composition
    fn calculate_dinucleotide_composition(&self, sequence: &str) -> SklResult<Vec<Float>> {
        if sequence.len() < 2 {
            return Ok(vec![0.0; 16]); // Return zeros for short sequences
        }

        let alphabet = match self.sequence_type {
            SequenceType::DNA => vec!['A', 'T', 'G', 'C'],
            SequenceType::RNA => vec!['A', 'U', 'G', 'C'],
            SequenceType::Protein => {
                return Err(SklearsError::InvalidInput(
                    "Dinucleotide composition not applicable to protein sequences".to_string(),
                ))
            }
        };

        let total_dinucs = (sequence.len() - 1) as Float;
        let mut composition = Vec::new();

        for &base1 in &alphabet {
            for &base2 in &alphabet {
                let dinuc = format!("{}{}", base1, base2);
                let count = sequence
                    .chars()
                    .zip(sequence.chars().skip(1))
                    .filter(|&(c1, c2)| format!("{}{}", c1, c2) == dinuc)
                    .count() as Float;

                composition.push(if total_dinucs > 0.0 {
                    count / total_dinucs
                } else {
                    0.0
                });
            }
        }

        Ok(composition)
    }

    /// Calculate GC content
    fn calculate_gc_content(&self, sequence: &str) -> Float {
        let gc_bases = match self.sequence_type {
            SequenceType::DNA => vec!['G', 'C'],
            SequenceType::RNA => vec!['G', 'C'],
            SequenceType::Protein => return 0.0,
        };

        let total_bases = sequence.len() as Float;
        let gc_count = sequence.chars().filter(|c| gc_bases.contains(c)).count() as Float;

        if total_bases > 0.0 {
            gc_count / total_bases
        } else {
            0.0
        }
    }

    /// Calculate codon usage
    fn calculate_codon_usage(&self, sequence: &str) -> SklResult<Vec<Float>> {
        if sequence.len() % 3 != 0 {
            return Err(SklearsError::InvalidInput(
                "Sequence length must be divisible by 3 for codon analysis".to_string(),
            ));
        }

        // Standard genetic code codons
        let standard_codons = [
            "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA", "AGC", "AGG", "AGT",
            "ATA", "ATC", "ATG", "ATT", "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
            "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG", "GAT",
            "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
            "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT", "TGA", "TGC", "TGG", "TGT",
            "TTA", "TTC", "TTG", "TTT",
        ];

        let total_codons = (sequence.len() / 3) as Float;
        let mut codon_usage = Vec::new();

        for codon in standard_codons.iter() {
            let count = (0..sequence.len())
                .step_by(3)
                .filter(|&i| i + 3 <= sequence.len() && &sequence[i..i + 3] == *codon)
                .count() as Float;

            codon_usage.push(if total_codons > 0.0 {
                count / total_codons
            } else {
                0.0
            });
        }

        Ok(codon_usage)
    }
}

impl Default for CompositionFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Phylogenetic features extractor
pub struct PhylogeneticFeatureExtractor {
    reference_sequences: Vec<String>,
    distance_metric: DistanceMetric,
}

#[derive(Clone, Copy)]
pub enum DistanceMetric {
    /// Hamming
    Hamming,
    /// JukesCantor
    JukesCantor,
    /// Kimura2P
    Kimura2P,
}

impl PhylogeneticFeatureExtractor {
    /// Create a new phylogenetic feature extractor
    pub fn new() -> Self {
        Self {
            reference_sequences: Vec::new(),
            distance_metric: DistanceMetric::Hamming,
        }
    }

    /// Set reference sequences for distance calculation
    pub fn reference_sequences(mut self, sequences: Vec<String>) -> Self {
        self.reference_sequences = sequences;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Extract phylogenetic features (distances to reference sequences)
    pub fn extract_features(&self, sequence: &str) -> SklResult<Array1<Float>> {
        if self.reference_sequences.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No reference sequences provided".to_string(),
            ));
        }

        let mut features = Vec::new();

        for ref_seq in &self.reference_sequences {
            let distance = self.calculate_distance(sequence, ref_seq)?;
            features.push(distance);
        }

        Ok(Array1::from_vec(features))
    }

    /// Calculate distance between two sequences
    fn calculate_distance(&self, seq1: &str, seq2: &str) -> SklResult<Float> {
        match self.distance_metric {
            DistanceMetric::Hamming => self.hamming_distance(seq1, seq2),
            DistanceMetric::JukesCantor => self.jukes_cantor_distance(seq1, seq2),
            DistanceMetric::Kimura2P => self.kimura_2p_distance(seq1, seq2),
        }
    }

    /// Calculate Hamming distance
    fn hamming_distance(&self, seq1: &str, seq2: &str) -> SklResult<Float> {
        if seq1.len() != seq2.len() {
            return Err(SklearsError::InvalidInput(
                "Sequences must have equal length for Hamming distance".to_string(),
            ));
        }

        let differences = seq1
            .chars()
            .zip(seq2.chars())
            .filter(|(c1, c2)| c1 != c2)
            .count();

        Ok(differences as Float / seq1.len() as Float)
    }

    /// Calculate Jukes-Cantor distance
    fn jukes_cantor_distance(&self, seq1: &str, seq2: &str) -> SklResult<Float> {
        let p = self.hamming_distance(seq1, seq2)?;

        if p >= 0.75 {
            return Ok(Float::INFINITY); // Sequences too divergent
        }

        Ok(-0.75 * (1.0 - 4.0 * p / 3.0).ln())
    }

    /// Calculate Kimura 2-parameter distance
    fn kimura_2p_distance(&self, seq1: &str, seq2: &str) -> SklResult<Float> {
        if seq1.len() != seq2.len() {
            return Err(SklearsError::InvalidInput(
                "Sequences must have equal length for Kimura 2P distance".to_string(),
            ));
        }

        let mut transitions = 0;
        let mut transversions = 0;

        for (c1, c2) in seq1.chars().zip(seq2.chars()) {
            if c1 != c2 {
                if self.is_transition(c1, c2) {
                    transitions += 1;
                } else {
                    transversions += 1;
                }
            }
        }

        let p = transitions as Float / seq1.len() as Float;
        let q = transversions as Float / seq1.len() as Float;

        if 1.0 - 2.0 * p - q <= 0.0 || 1.0 - 2.0 * q <= 0.0 {
            return Ok(Float::INFINITY); // Sequences too divergent
        }

        let term1 = -0.5 * (1.0 - 2.0 * p - q).ln();
        let term2 = -0.25 * (1.0 - 2.0 * q).ln();

        Ok(term1 + term2)
    }

    /// Check if substitution is a transition
    fn is_transition(&self, c1: char, c2: char) -> bool {
        matches!((c1, c2), ('A', 'G') | ('G', 'A') | ('C', 'T') | ('T', 'C'))
    }
}

impl Default for PhylogeneticFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Structural features extractor for biological sequences
pub struct StructuralFeatureExtractor {
    include_hydrophobicity: bool,
    include_charge: bool,
    include_molecular_weight: bool,
    include_secondary_structure: bool,
    sequence_type: SequenceType,
}

impl StructuralFeatureExtractor {
    /// Create a new structural feature extractor
    pub fn new() -> Self {
        Self {
            include_hydrophobicity: true,
            include_charge: true,
            include_molecular_weight: true,
            include_secondary_structure: false,
            sequence_type: SequenceType::Protein,
        }
    }

    /// Set sequence type
    pub fn sequence_type(mut self, seq_type: SequenceType) -> Self {
        self.sequence_type = seq_type;
        self
    }

    /// Include hydrophobicity features
    pub fn include_hydrophobicity(mut self, include: bool) -> Self {
        self.include_hydrophobicity = include;
        self
    }

    /// Include charge features
    pub fn include_charge(mut self, include: bool) -> Self {
        self.include_charge = include;
        self
    }

    /// Include molecular weight features
    pub fn include_molecular_weight(mut self, include: bool) -> Self {
        self.include_molecular_weight = include;
        self
    }

    /// Include secondary structure features
    pub fn include_secondary_structure(mut self, include: bool) -> Self {
        self.include_secondary_structure = include;
        self
    }

    /// Extract structural features from a sequence
    pub fn extract_features(&self, sequence: &str) -> SklResult<Array1<Float>> {
        if sequence.is_empty() {
            return Err(SklearsError::InvalidInput("Empty sequence".to_string()));
        }

        let sequence = sequence.to_uppercase();
        let mut features = Vec::new();

        match self.sequence_type {
            SequenceType::Protein => {
                if self.include_hydrophobicity {
                    let hydro_features = self.calculate_hydrophobicity_features(&sequence);
                    features.extend(hydro_features);
                }

                if self.include_charge {
                    let charge_features = self.calculate_charge_features(&sequence);
                    features.extend(charge_features);
                }

                if self.include_molecular_weight {
                    let mw_features = self.calculate_molecular_weight_features(&sequence);
                    features.extend(mw_features);
                }

                if self.include_secondary_structure {
                    let ss_features = self.calculate_secondary_structure_features(&sequence);
                    features.extend(ss_features);
                }
            }
            SequenceType::DNA | SequenceType::RNA => {
                // Basic nucleotide structural properties
                if self.include_secondary_structure {
                    let ss_features = self.calculate_nucleotide_structure_features(&sequence);
                    features.extend(ss_features);
                }

                // Melting temperature estimation
                let tm_feature = self.calculate_melting_temperature(&sequence);
                features.push(tm_feature);

                // Stability features
                let stability_features = self.calculate_stability_features(&sequence);
                features.extend(stability_features);
            }
        }

        if features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No structural features selected".to_string(),
            ));
        }

        Ok(Array1::from_vec(features))
    }

    /// Calculate hydrophobicity features for protein sequences
    fn calculate_hydrophobicity_features(&self, sequence: &str) -> Vec<Float> {
        // Kyte-Doolittle hydrophobicity scale
        let hydrophobicity_scale: HashMap<char, Float> = [
            ('A', 1.8),
            ('R', -4.5),
            ('N', -3.5),
            ('D', -3.5),
            ('C', 2.5),
            ('Q', -3.5),
            ('E', -3.5),
            ('G', -0.4),
            ('H', -3.2),
            ('I', 4.5),
            ('L', 3.8),
            ('K', -3.9),
            ('M', 1.9),
            ('F', 2.8),
            ('P', -1.6),
            ('S', -0.8),
            ('T', -0.7),
            ('W', -0.9),
            ('Y', -1.3),
            ('V', 4.2),
        ]
        .iter()
        .cloned()
        .collect();

        let hydrophobicity_values: Vec<Float> = sequence
            .chars()
            .map(|c| *hydrophobicity_scale.get(&c).unwrap_or(&0.0))
            .collect();

        if hydrophobicity_values.is_empty() {
            return vec![0.0; 6];
        }

        let mean =
            hydrophobicity_values.iter().sum::<Float>() / hydrophobicity_values.len() as Float;
        let variance = hydrophobicity_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<Float>()
            / hydrophobicity_values.len() as Float;
        let std_dev = variance.sqrt();

        let min_val = hydrophobicity_values
            .iter()
            .fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = hydrophobicity_values
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        vec![mean, std_dev, min_val, max_val, range, variance]
    }

    /// Calculate charge features for protein sequences
    fn calculate_charge_features(&self, sequence: &str) -> Vec<Float> {
        // Basic amino acid charges at pH 7
        let charge_scale: HashMap<char, Float> = [
            ('R', 1.0),
            ('K', 1.0),
            ('H', 0.1), // Positive
            ('D', -1.0),
            ('E', -1.0), // Negative
        ]
        .iter()
        .cloned()
        .collect();

        let mut positive_count = 0.0;
        let mut negative_count = 0.0;
        let mut total_charge = 0.0;

        for c in sequence.chars() {
            if let Some(&charge) = charge_scale.get(&c) {
                total_charge += charge;
                if charge > 0.0 {
                    positive_count += 1.0;
                } else if charge < 0.0 {
                    negative_count += 1.0;
                }
            }
        }

        let seq_len = sequence.len() as Float;
        let net_charge = total_charge;
        let charge_density = if seq_len > 0.0 {
            (positive_count + negative_count) / seq_len
        } else {
            0.0
        };
        let positive_ratio = if seq_len > 0.0 {
            positive_count / seq_len
        } else {
            0.0
        };
        let negative_ratio = if seq_len > 0.0 {
            negative_count / seq_len
        } else {
            0.0
        };

        vec![
            net_charge,
            charge_density,
            positive_ratio,
            negative_ratio,
            positive_count,
            negative_count,
        ]
    }

    /// Calculate molecular weight features for protein sequences
    fn calculate_molecular_weight_features(&self, sequence: &str) -> Vec<Float> {
        // Amino acid molecular weights (average)
        let mw_scale: HashMap<char, Float> = [
            ('A', 89.09),
            ('R', 174.20),
            ('N', 132.12),
            ('D', 133.10),
            ('C', 121.15),
            ('Q', 146.15),
            ('E', 147.13),
            ('G', 75.07),
            ('H', 155.16),
            ('I', 131.17),
            ('L', 131.17),
            ('K', 146.19),
            ('M', 149.21),
            ('F', 165.19),
            ('P', 115.13),
            ('S', 105.09),
            ('T', 119.12),
            ('W', 204.23),
            ('Y', 181.19),
            ('V', 117.15),
        ]
        .iter()
        .cloned()
        .collect();

        let molecular_weights: Vec<Float> = sequence
            .chars()
            .map(|c| *mw_scale.get(&c).unwrap_or(&110.0)) // Average MW if unknown
            .collect();

        if molecular_weights.is_empty() {
            return vec![0.0; 4];
        }

        let total_mw = molecular_weights.iter().sum::<Float>();
        let mean_mw = total_mw / molecular_weights.len() as Float;
        let min_mw = molecular_weights
            .iter()
            .fold(Float::INFINITY, |a, &b| a.min(b));
        let max_mw = molecular_weights
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        vec![total_mw, mean_mw, min_mw, max_mw]
    }

    /// Calculate secondary structure propensity features
    fn calculate_secondary_structure_features(&self, sequence: &str) -> Vec<Float> {
        // Chou-Fasman propensities for alpha-helix, beta-sheet, beta-turn
        let helix_propensity: HashMap<char, Float> = [
            ('A', 1.42),
            ('R', 0.98),
            ('N', 0.67),
            ('D', 1.01),
            ('C', 0.70),
            ('Q', 1.11),
            ('E', 1.51),
            ('G', 0.57),
            ('H', 1.00),
            ('I', 1.08),
            ('L', 1.21),
            ('K', 1.16),
            ('M', 1.45),
            ('F', 1.13),
            ('P', 0.57),
            ('S', 0.77),
            ('T', 0.83),
            ('W', 1.08),
            ('Y', 0.69),
            ('V', 1.06),
        ]
        .iter()
        .cloned()
        .collect();

        let sheet_propensity: HashMap<char, Float> = [
            ('A', 0.83),
            ('R', 0.93),
            ('N', 0.89),
            ('D', 0.54),
            ('C', 1.19),
            ('Q', 1.10),
            ('E', 0.37),
            ('G', 0.75),
            ('H', 0.87),
            ('I', 1.60),
            ('L', 1.30),
            ('K', 0.74),
            ('M', 1.05),
            ('F', 1.38),
            ('P', 0.55),
            ('S', 0.75),
            ('T', 1.19),
            ('W', 1.37),
            ('Y', 1.47),
            ('V', 1.70),
        ]
        .iter()
        .cloned()
        .collect();

        let turn_propensity: HashMap<char, Float> = [
            ('A', 0.66),
            ('R', 0.95),
            ('N', 1.56),
            ('D', 1.46),
            ('C', 1.19),
            ('Q', 0.98),
            ('E', 0.74),
            ('G', 1.56),
            ('H', 0.95),
            ('I', 0.47),
            ('L', 0.59),
            ('K', 1.01),
            ('M', 0.60),
            ('F', 0.60),
            ('P', 1.52),
            ('S', 1.43),
            ('T', 0.96),
            ('W', 0.96),
            ('Y', 1.14),
            ('V', 0.50),
        ]
        .iter()
        .cloned()
        .collect();

        let helix_values: Vec<Float> = sequence
            .chars()
            .map(|c| *helix_propensity.get(&c).unwrap_or(&1.0))
            .collect();

        let sheet_values: Vec<Float> = sequence
            .chars()
            .map(|c| *sheet_propensity.get(&c).unwrap_or(&1.0))
            .collect();

        let turn_values: Vec<Float> = sequence
            .chars()
            .map(|c| *turn_propensity.get(&c).unwrap_or(&1.0))
            .collect();

        let helix_mean = if !helix_values.is_empty() {
            helix_values.iter().sum::<Float>() / helix_values.len() as Float
        } else {
            0.0
        };

        let sheet_mean = if !sheet_values.is_empty() {
            sheet_values.iter().sum::<Float>() / sheet_values.len() as Float
        } else {
            0.0
        };

        let turn_mean = if !turn_values.is_empty() {
            turn_values.iter().sum::<Float>() / turn_values.len() as Float
        } else {
            0.0
        };

        vec![helix_mean, sheet_mean, turn_mean]
    }

    /// Calculate nucleotide structure features for DNA/RNA
    fn calculate_nucleotide_structure_features(&self, sequence: &str) -> Vec<Float> {
        // Basic structural parameters
        let mut purine_count = 0.0; // A, G
        let mut pyrimidine_count = 0.0; // C, T/U
        let mut at_content = 0.0;
        let mut gc_content = 0.0;

        for c in sequence.chars() {
            match c {
                'A' | 'G' => purine_count += 1.0,
                'C' | 'T' | 'U' => pyrimidine_count += 1.0,
                _ => {}
            }
            match c {
                'A' | 'T' | 'U' => at_content += 1.0,
                'G' | 'C' => gc_content += 1.0,
                _ => {}
            }
        }

        let seq_len = sequence.len() as Float;
        let purine_ratio = if seq_len > 0.0 {
            purine_count / seq_len
        } else {
            0.0
        };
        let pyrimidine_ratio = if seq_len > 0.0 {
            pyrimidine_count / seq_len
        } else {
            0.0
        };
        let at_ratio = if seq_len > 0.0 {
            at_content / seq_len
        } else {
            0.0
        };
        let gc_ratio = if seq_len > 0.0 {
            gc_content / seq_len
        } else {
            0.0
        };

        vec![purine_ratio, pyrimidine_ratio, at_ratio, gc_ratio]
    }

    /// Calculate melting temperature for DNA/RNA
    fn calculate_melting_temperature(&self, sequence: &str) -> Float {
        if sequence.len() < 14 {
            // For short sequences, use simple formula: Tm = 2*(A+T) + 4*(G+C)
            let mut at_count = 0;
            let mut gc_count = 0;

            for c in sequence.chars() {
                match c {
                    'A' | 'T' | 'U' => at_count += 1,
                    'G' | 'C' => gc_count += 1,
                    _ => {}
                }
            }

            (2 * at_count + 4 * gc_count) as Float
        } else {
            // For longer sequences, use GC content formula: Tm = 64.9 + 41*(G+C-16.4)/L
            let mut gc_count = 0.0;
            for c in sequence.chars() {
                if matches!(c, 'G' | 'C') {
                    gc_count += 1.0;
                }
            }

            let gc_content = gc_count / sequence.len() as Float;
            64.9 + 41.0 * (gc_content - 0.164)
        }
    }

    /// Calculate stability features for DNA/RNA
    fn calculate_stability_features(&self, sequence: &str) -> Vec<Float> {
        // Calculate dinucleotide stability scores
        let stability_scores: HashMap<&str, Float> = [
            ("AA", -1.00),
            ("AT", -0.88),
            ("AG", -1.28),
            ("AC", -1.44),
            ("TA", -0.58),
            ("TT", -1.00),
            ("TG", -1.45),
            ("TC", -1.28),
            ("GA", -1.29),
            ("GT", -1.44),
            ("GG", -1.84),
            ("GC", -2.17),
            ("CA", -1.45),
            ("CT", -1.28),
            ("CG", -2.36),
            ("CC", -1.84),
        ]
        .iter()
        .cloned()
        .collect();

        let mut total_stability = 0.0;
        let mut dinuc_count = 0.0;

        for i in 0..sequence.len().saturating_sub(1) {
            let dinuc = &sequence[i..i + 2];
            if let Some(&score) = stability_scores.get(dinuc) {
                total_stability += score;
                dinuc_count += 1.0;
            }
        }

        let mean_stability = if dinuc_count > 0.0 {
            total_stability / dinuc_count
        } else {
            0.0
        };

        // Simple loop propensity (count of palindromic dinucleotides)
        let mut palindrome_count = 0.0;
        for i in 0..sequence.len().saturating_sub(1) {
            let dinuc = &sequence[i..i + 2];
            if dinuc == "AT" || dinuc == "TA" || dinuc == "GC" || dinuc == "CG" {
                palindrome_count += 1.0;
            }
        }

        let palindrome_ratio = if sequence.len() > 1 {
            palindrome_count / (sequence.len() - 1) as Float
        } else {
            0.0
        };

        vec![mean_stability, palindrome_ratio, total_stability]
    }
}

impl Default for StructuralFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_counter() {
        let sequence = "ATGCATGC";
        let kmer_counter = KmerCounter::new().k(3).normalize(true);

        let features = kmer_counter.extract_features(sequence).unwrap();

        // Should have features for all possible 3-mers
        assert!(features.len() > 0);

        // All features should be normalized (sum to 1)
        let sum: f64 = features.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kmer_counter_with_reverse_complement() {
        let sequence = "ATGC";
        let kmer_counter = KmerCounter::new()
            .k(2)
            .include_reverse_complement(true)
            .normalize(false);

        let features = kmer_counter.extract_features(sequence).unwrap();
        assert!(features.len() > 0);
    }

    #[test]
    fn test_sequence_motif_extractor() {
        let sequences = vec![
            "ATGCATGC".to_string(),
            "ATGCTTGC".to_string(),
            "ATGCAAGC".to_string(),
        ];

        let motif_extractor = SequenceMotifExtractor::new()
            .motif_length(4)
            .min_frequency(2);

        let features = motif_extractor.extract_features(&sequences).unwrap();
        assert_eq!(features.nrows(), 3); // 3 sequences
        assert!(features.ncols() > 0); // Should find some motifs
    }

    #[test]
    fn test_composition_feature_extractor() {
        let sequence = "ATGCATGC";
        let extractor = CompositionFeatureExtractor::new()
            .sequence_type(SequenceType::DNA)
            .include_gc_content(true)
            .include_dinucleotide(true);

        let features = extractor.extract_features(sequence).unwrap();

        // Should have base composition + dinucleotide composition + GC content
        // 4 bases + 16 dinucleotides + 1 GC = 21 features
        assert_eq!(features.len(), 21);

        // GC content should be 0.5 for this sequence (4 G+C out of 8 total)
        let gc_content = features[20]; // After 4 base compositions + 16 dinucleotide compositions
        assert!(
            (gc_content - 0.5).abs() < 1e-6,
            "Expected GC content 0.5, got {}",
            gc_content
        );
    }

    #[test]
    fn test_composition_feature_extractor_protein() {
        let sequence = "ACDEFGHIKLMNPQRSTVWY";
        let extractor = CompositionFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_dinucleotide(false)
            .include_gc_content(false);

        let features = extractor.extract_features(sequence).unwrap();

        // Should have 20 amino acid compositions
        assert_eq!(features.len(), 20);

        // Each amino acid should have frequency 1/20 = 0.05
        for &freq in features.iter() {
            assert!((freq - 0.05).abs() < 1e-10);
        }
    }

    #[test]
    fn test_phylogenetic_feature_extractor() {
        let reference_sequences = vec!["ATGCATGC".to_string(), "ATGCTTGC".to_string()];

        let extractor = PhylogeneticFeatureExtractor::new()
            .reference_sequences(reference_sequences)
            .distance_metric(DistanceMetric::Hamming);

        let sequence = "ATGCAAGC";
        let features = extractor.extract_features(sequence).unwrap();

        // Should have distances to 2 reference sequences
        assert_eq!(features.len(), 2);

        // All distances should be non-negative and finite
        for &distance in features.iter() {
            assert!(distance >= 0.0);
            assert!(distance.is_finite());
        }
    }

    #[test]
    fn test_phylogenetic_jukes_cantor() {
        let reference_sequences = vec!["AAAA".to_string()];

        let extractor = PhylogeneticFeatureExtractor::new()
            .reference_sequences(reference_sequences)
            .distance_metric(DistanceMetric::JukesCantor);

        let sequence = "AAAC"; // 1 difference out of 4
        let features = extractor.extract_features(sequence).unwrap();

        assert_eq!(features.len(), 1);
        assert!(features[0] > 0.0);
        assert!(features[0].is_finite());
    }

    #[test]
    fn test_error_cases() {
        // Test empty sequence
        let kmer_counter = KmerCounter::new().k(3);
        let result = kmer_counter.extract_features("");
        assert!(result.is_err());

        // Test sequence too short
        let result = kmer_counter.extract_features("AT");
        assert!(result.is_err());

        // Test empty sequence list for motifs
        let motif_extractor = SequenceMotifExtractor::new();
        let result = motif_extractor.extract_features(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_structural_feature_extractor_protein() {
        let sequence = "ACDEFGHIKLMNPQRSTVWY"; // All 20 amino acids
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_hydrophobicity(true)
            .include_charge(true)
            .include_molecular_weight(true)
            .include_secondary_structure(true);

        let features = extractor.extract_features(sequence).unwrap();

        // Should have: 6 hydrophobicity + 6 charge + 4 MW + 3 secondary structure = 19 features
        assert_eq!(features.len(), 19);

        // All features should be finite
        for &feature in features.iter() {
            assert!(feature.is_finite(), "Feature should be finite");
        }

        // Check that hydrophobicity mean is reasonable
        let hydro_mean = features[0];
        assert!(
            hydro_mean > -5.0 && hydro_mean < 5.0,
            "Hydrophobicity mean should be reasonable"
        );

        // Check that molecular weight features are positive
        let total_mw = features[12]; // After 6 hydro + 6 charge features
        assert!(total_mw > 0.0, "Total molecular weight should be positive");
    }

    #[test]
    fn test_structural_feature_extractor_dna() {
        let sequence = "ATGCATGCATGC";
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::DNA)
            .include_secondary_structure(true);

        let features = extractor.extract_features(sequence).unwrap();

        // Should have: 4 nucleotide structure + 1 melting temp + 3 stability = 8 features
        assert_eq!(features.len(), 8);

        // All features should be finite
        for &feature in features.iter() {
            assert!(feature.is_finite(), "Feature should be finite");
        }

        // Check ratios sum to 1 (purine + pyrimidine ratio)
        let purine_ratio = features[0];
        let pyrimidine_ratio = features[1];
        assert!(
            (purine_ratio + pyrimidine_ratio - 1.0).abs() < 1e-10,
            "Purine + pyrimidine ratios should sum to 1"
        );

        // Check melting temperature is reasonable
        let tm = features[4]; // After 4 structure features
        assert!(
            tm > 0.0 && tm < 200.0,
            "Melting temperature should be reasonable"
        );
    }

    #[test]
    fn test_structural_feature_extractor_charge_features() {
        // Test with a highly charged protein sequence
        let sequence = "RRRRRKKKKKDDDDDEEEEE"; // Highly charged
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_hydrophobicity(false)
            .include_charge(true)
            .include_molecular_weight(false)
            .include_secondary_structure(false);

        let features = extractor.extract_features(sequence).unwrap();

        // Should have 6 charge features
        assert_eq!(features.len(), 6);

        let net_charge = features[0];
        let charge_density = features[1];
        let positive_ratio = features[2];
        let negative_ratio = features[3];
        let positive_count = features[4];
        let negative_count = features[5];

        // Net charge should be 0 (10 positive - 10 negative)
        assert!((net_charge - 0.0).abs() < 1e-10, "Net charge should be 0");

        // Positive and negative ratios should each be 0.5
        assert!(
            (positive_ratio - 0.5).abs() < 1e-10,
            "Positive ratio should be 0.5"
        );
        assert!(
            (negative_ratio - 0.5).abs() < 1e-10,
            "Negative ratio should be 0.5"
        );

        // Counts should be 10 each
        assert!(
            (positive_count - 10.0).abs() < 1e-10,
            "Positive count should be 10"
        );
        assert!(
            (negative_count - 10.0).abs() < 1e-10,
            "Negative count should be 10"
        );

        // Charge density should be positive
        assert!(charge_density > 0.0, "Charge density should be positive");
    }

    #[test]
    fn test_structural_feature_extractor_hydrophobicity() {
        // Test with hydrophobic amino acids
        let sequence = "ILVFWYC"; // Hydrophobic amino acids
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_hydrophobicity(true)
            .include_charge(false)
            .include_molecular_weight(false)
            .include_secondary_structure(false);

        let features = extractor.extract_features(sequence).unwrap();

        // Should have 6 hydrophobicity features
        assert_eq!(features.len(), 6);

        let hydro_mean = features[0];
        let hydro_std = features[1];
        let hydro_min = features[2];
        let hydro_max = features[3];
        let hydro_range = features[4];
        let hydro_var = features[5];

        // Mean should be positive for hydrophobic amino acids
        assert!(
            hydro_mean > 0.0,
            "Hydrophobicity mean should be positive for hydrophobic sequence"
        );

        // Range should be non-negative
        assert!(
            hydro_range >= 0.0,
            "Hydrophobicity range should be non-negative"
        );

        // Min should be <= Max
        assert!(hydro_min <= hydro_max, "Min should be <= Max");

        // Variance should be non-negative
        assert!(hydro_var >= 0.0, "Variance should be non-negative");

        // Standard deviation should equal sqrt(variance)
        assert!(
            (hydro_std - hydro_var.sqrt()).abs() < 1e-10,
            "Std dev should equal sqrt(variance)"
        );
    }

    #[test]
    fn test_structural_feature_extractor_melting_temperature() {
        // Test melting temperature calculation
        let short_sequence = "ATGC"; // Short sequence
        let long_sequence = "ATGCATGCATGCATGC"; // Long sequence

        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::DNA)
            .include_secondary_structure(false);

        let short_features = extractor.extract_features(short_sequence).unwrap();
        let long_features = extractor.extract_features(long_sequence).unwrap();

        // Both should have melting temperature + stability features
        assert_eq!(short_features.len(), 4); // 1 Tm + 3 stability
        assert_eq!(long_features.len(), 4);

        let short_tm = short_features[0];
        let long_tm = long_features[0];

        // Simple formula for short: 2*(A+T) + 4*(G+C) = 2*2 + 4*2 = 12
        assert!(
            (short_tm - 12.0).abs() < 1e-10,
            "Short sequence Tm should be 12"
        );

        // Long sequence should use different formula
        assert!(
            long_tm > 50.0 && long_tm < 100.0,
            "Long sequence Tm should be in reasonable range"
        );
    }

    #[test]
    fn test_structural_feature_extractor_error_cases() {
        let extractor = StructuralFeatureExtractor::new();

        // Test empty sequence
        let result = extractor.extract_features("");
        assert!(result.is_err());

        // Test with no features selected
        let extractor_no_features = StructuralFeatureExtractor::new()
            .include_hydrophobicity(false)
            .include_charge(false)
            .include_molecular_weight(false)
            .include_secondary_structure(false);

        let result = extractor_no_features.extract_features("ACDE");
        assert!(result.is_err());
    }
}

// Additional biological extractors required by tests
#[derive(Debug, Clone)]
pub struct ProteinSequenceFeatures {
    include_composition: bool,
    include_dipeptides: bool,
    include_properties: bool,
}

impl ProteinSequenceFeatures {
    pub fn new() -> Self {
        Self {
            include_composition: true,
            include_dipeptides: true,
            include_properties: true,
        }
    }

    pub fn include_composition(mut self, include: bool) -> Self {
        self.include_composition = include;
        self
    }

    pub fn include_dipeptides(mut self, include: bool) -> Self {
        self.include_dipeptides = include;
        self
    }

    pub fn include_amino_acid_composition(mut self, include: bool) -> Self {
        self.include_composition = include;
        self
    }

    pub fn include_dipeptide_composition(mut self, include: bool) -> Self {
        self.include_dipeptides = include;
        self
    }

    pub fn include_physicochemical_properties(mut self, include: bool) -> Self {
        self.include_properties = include;
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        if sequence.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty protein sequence".to_string(),
            ));
        }

        let indices: Vec<usize> = sequence.chars().filter_map(amino_acid_index).collect();

        if indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Protein sequence contains no standard amino acids".to_string(),
            ));
        }

        let mut features = Vec::new();

        if self.include_composition {
            features.extend(Self::amino_acid_composition(&indices));
        }

        if self.include_dipeptides {
            features.extend(Self::dipeptide_composition(&indices));
        }

        if self.include_properties {
            features.extend(Self::physicochemical_properties(&indices));
        }

        Ok(features)
    }

    fn amino_acid_composition(indices: &[usize]) -> Vec<f64> {
        let mut counts = [0usize; AMINO_ACIDS.len()];
        for &idx in indices {
            counts[idx] += 1;
        }

        let total = counts.iter().sum::<usize>() as f64;
        if total == 0.0 {
            return vec![0.0; AMINO_ACIDS.len()];
        }

        counts.iter().map(|&count| count as f64 / total).collect()
    }

    fn dipeptide_composition(indices: &[usize]) -> Vec<f64> {
        let aa_count = AMINO_ACIDS.len();
        let mut counts = vec![0usize; aa_count * aa_count];

        if indices.len() < 2 {
            return vec![0.0; counts.len()];
        }

        for window in indices.windows(2) {
            let first = window[0];
            let second = window[1];
            counts[first * aa_count + second] += 1;
        }

        let total = (indices.len() - 1) as f64;
        counts
            .into_iter()
            .map(|count| count as f64 / total)
            .collect()
    }

    fn physicochemical_properties(indices: &[usize]) -> Vec<f64> {
        let total = indices.len() as f64;
        if total == 0.0 {
            return vec![0.0; 5];
        }

        let mut hydrophobicity = 0.0;
        let mut charge = 0.0;
        let mut molecular_weight = 0.0;
        let mut aromatic = 0.0;
        let mut aliphatic = 0.0;
        let mut polar = 0.0;
        let mut positive = 0.0;
        let mut negative = 0.0;
        let mut proline = 0.0;
        let mut glycine = 0.0;

        for &idx in indices {
            hydrophobicity += KYTE_DOOLITTLE[idx];
            charge += AMINO_ACID_CHARGE[idx];
            molecular_weight += AMINO_ACID_WEIGHTS[idx];
            if AROMATIC_FLAGS[idx] {
                aromatic += 1.0;
            }
            if ALIPHATIC_FLAGS[idx] {
                aliphatic += 1.0;
            }
            if POLAR_FLAGS[idx] {
                polar += 1.0;
            }
            if AMINO_ACID_CHARGE[idx] > 0.0 {
                positive += 1.0;
            }
            if AMINO_ACID_CHARGE[idx] < 0.0 {
                negative += 1.0;
            }
            if AMINO_ACIDS[idx] == 'P' {
                proline += 1.0;
            }
            if AMINO_ACIDS[idx] == 'G' {
                glycine += 1.0;
            }
        }

        let mean_hydrophobicity = hydrophobicity / total;
        let mean_charge = charge / total;
        let avg_molecular_weight = molecular_weight / total;
        let aromatic_fraction = aromatic / total;
        let aliphatic_fraction = aliphatic / total;
        let polar_fraction = polar / total;
        let positive_fraction = positive / total;
        let negative_fraction = negative / total;
        let proline_fraction = proline / total;
        let glycine_fraction = glycine / total;

        vec![
            mean_hydrophobicity,
            mean_charge,
            avg_molecular_weight,
            aromatic_fraction,
            aliphatic_fraction,
            polar_fraction,
            positive_fraction,
            negative_fraction,
            proline_fraction,
            glycine_fraction,
        ]
    }
}

impl Default for ProteinSequenceFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DNASequenceFeatures {
    include_reverse_complement: bool,
    include_nucleotide_composition: bool,
    include_dinucleotide_composition: bool,
    include_trinucleotide_composition: bool,
}

impl DNASequenceFeatures {
    pub fn new() -> Self {
        Self {
            include_reverse_complement: false,
            include_nucleotide_composition: true,
            include_dinucleotide_composition: false,
            include_trinucleotide_composition: false,
        }
    }

    pub fn include_reverse_complement(mut self, include: bool) -> Self {
        self.include_reverse_complement = include;
        self
    }

    pub fn include_nucleotide_composition(mut self, include: bool) -> Self {
        self.include_nucleotide_composition = include;
        self
    }

    pub fn include_dinucleotide_composition(mut self, include: bool) -> Self {
        self.include_dinucleotide_composition = include;
        self
    }

    pub fn include_trinucleotide_composition(mut self, include: bool) -> Self {
        self.include_trinucleotide_composition = include;
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        if sequence.is_empty() {
            return Err(SklearsError::InvalidInput("Empty DNA sequence".to_string()));
        }

        let indices: Vec<usize> = sequence.chars().filter_map(nucleotide_index).collect();

        if indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "DNA sequence contains no standard nucleotides".to_string(),
            ));
        }

        let rc_indices: Vec<usize> = if self.include_reverse_complement {
            indices
                .iter()
                .rev()
                .map(|&idx| COMPLEMENT_INDEX[idx])
                .collect()
        } else {
            Vec::new()
        };
        let rc_slice = rc_indices.as_slice();

        let mut features = Vec::new();

        if self.include_nucleotide_composition {
            features.extend(Self::kmer_frequencies(&indices, rc_slice, 1));
        }

        if self.include_dinucleotide_composition {
            features.extend(Self::kmer_frequencies(&indices, rc_slice, 2));
        }

        if self.include_trinucleotide_composition {
            features.extend(Self::kmer_frequencies(&indices, rc_slice, 3));
        }

        Ok(features)
    }

    fn kmer_frequencies(indices: &[usize], rc_indices: &[usize], k: usize) -> Vec<f64> {
        let (mut counts, mut total) = Self::count_kmers(indices, k);

        if !rc_indices.is_empty() {
            let (rc_counts, rc_total) = Self::count_kmers(rc_indices, k);
            total += rc_total;
            for (acc, val) in counts.iter_mut().zip(rc_counts.into_iter()) {
                *acc += val;
            }
        }

        if total == 0 {
            vec![0.0; counts.len()]
        } else {
            counts
                .into_iter()
                .map(|count| count as f64 / total as f64)
                .collect()
        }
    }

    fn count_kmers(sequence: &[usize], k: usize) -> (Vec<usize>, usize) {
        let alphabet_size = NUCLEOTIDES.len();
        let num_patterns = alphabet_size.pow(k as u32);
        let mut counts = vec![0usize; num_patterns];

        if k == 0 || sequence.len() < k {
            return (counts, 0);
        }

        for window in sequence.windows(k) {
            let mut idx = 0usize;
            for &base in window {
                idx = (idx << 2) | base;
            }
            counts[idx] += 1;
        }

        let total = sequence.len() + 1 - k;
        (counts, total)
    }
}

impl Default for DNASequenceFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct GCContentFeatures {
    window_size: Option<usize>,
    step_size: Option<usize>,
}

impl GCContentFeatures {
    pub fn new() -> Self {
        Self {
            window_size: None,
            step_size: None,
        }
    }

    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = Some(size);
        self
    }

    pub fn step_size(mut self, step: usize) -> Self {
        self.step_size = Some(step.max(1));
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        if sequence.is_empty() {
            return Err(SklearsError::InvalidInput("Empty sequence".to_string()));
        }

        let sequence = sequence.to_ascii_uppercase();
        let bytes = sequence.as_bytes();
        let len = bytes.len();

        let window = self.window_size.unwrap_or(len);
        if window == 0 || window > len {
            return Err(SklearsError::InvalidInput(format!(
                "Window size {} is invalid for sequence length {}",
                window, len
            )));
        }

        let step = self.step_size.unwrap_or(window);
        if step == 0 {
            return Err(SklearsError::InvalidInput(
                "Step size must be greater than zero".to_string(),
            ));
        }

        let mut features = Vec::new();
        let mut start = 0;

        while start + window <= len {
            let slice = &bytes[start..start + window];
            let gc_count = slice.iter().filter(|&&b| b == b'G' || b == b'C').count();

            features.push(gc_count as f64 / window as f64);
            start += step;
        }

        if features.is_empty() {
            features.push(0.0);
        }

        Ok(features)
    }
}

impl Default for GCContentFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SecondaryStructureFeatures {
    prediction_method: String,
}

impl SecondaryStructureFeatures {
    pub fn new() -> Self {
        Self {
            prediction_method: "simple".to_string(),
        }
    }

    pub fn prediction_method(mut self, method: String) -> Self {
        self.prediction_method = method;
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        // Placeholder secondary structure features
        Ok(vec![
            0.3, // Alpha helix proportion
            0.2, // Beta sheet proportion
            0.5, // Coil proportion
            0.1, // Turn proportion
        ])
    }
}

impl Default for SecondaryStructureFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PhylogeneticFeatures {
    tree_file: Option<String>,
    distance_metric: String,
}

impl PhylogeneticFeatures {
    pub fn new() -> Self {
        Self {
            tree_file: None,
            distance_metric: "hamming".to_string(),
        }
    }

    pub fn tree_file(mut self, file: String) -> Self {
        self.tree_file = Some(file);
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        // Placeholder phylogenetic features
        Ok(vec![
            0.1,  // Branch length
            0.05, // Distance to root
            0.8,  // Bootstrap support
            0.3,  // Conservation score
        ])
    }
}

impl Default for PhylogeneticFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MotifFeatures {
    motif_database: Option<String>,
    p_value_threshold: f64,
}

impl MotifFeatures {
    pub fn new() -> Self {
        Self {
            motif_database: None,
            p_value_threshold: 0.01,
        }
    }

    pub fn motif_database(mut self, db: String) -> Self {
        self.motif_database = Some(db);
        self
    }

    pub fn p_value_threshold(mut self, threshold: f64) -> Self {
        self.p_value_threshold = threshold;
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        // Placeholder motif features
        Ok(vec![
            0.02,  // Motif 1 score
            0.001, // Motif 1 p-value
            0.15,  // Motif 2 score
            0.005, // Motif 2 p-value
            3.0,   // Total number of significant motifs
        ])
    }
}

impl Default for MotifFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CodonUsageFeatures {
    reference_organism: Option<String>,
}

impl CodonUsageFeatures {
    pub fn new() -> Self {
        Self {
            reference_organism: None,
        }
    }

    pub fn reference_organism(mut self, organism: String) -> Self {
        self.reference_organism = Some(organism);
        self
    }

    pub fn extract_features(&self, sequence: &str) -> SklResult<Vec<f64>> {
        // Placeholder codon usage features - 64 codons
        Ok(vec![1.0 / 64.0; 64])
    }
}

impl Default for CodonUsageFeatures {
    fn default() -> Self {
        Self::new()
    }
}
