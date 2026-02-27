//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use super::types::{
    CompositionFeatureExtractor, DistanceMetric, KmerCounter, PhylogeneticFeatureExtractor,
    SequenceMotifExtractor, SequenceType, StructuralFeatureExtractor,
};

pub(super) const AMINO_ACIDS: [char; 20] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];
pub(super) const KYTE_DOOLITTLE: [f64; 20] = [
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7,
    4.2, -0.9, -1.3,
];
pub(super) const AMINO_ACID_WEIGHTS: [f64; 20] = [
    89.09, 121.15, 133.1, 147.13, 165.19, 75.07, 155.16, 131.17, 146.19, 131.17, 149.21, 132.12,
    115.13, 146.15, 174.2, 105.09, 119.12, 117.15, 204.23, 181.19,
];
pub(super) const AMINO_ACID_CHARGE: [f64; 20] = [
    0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0,
];
pub(super) const POLAR_FLAGS: [bool; 20] = [
    false, true, true, true, false, false, true, false, true, false, false, true, false, true,
    true, true, true, false, false, true,
];
pub(super) const AROMATIC_FLAGS: [bool; 20] = [
    false, false, false, false, true, false, false, false, false, false, false, false, false,
    false, false, false, false, false, true, true,
];
pub(super) const ALIPHATIC_FLAGS: [bool; 20] = [
    true, false, false, false, false, false, false, true, false, true, false, false, false, false,
    false, false, false, true, false, false,
];
pub(super) const NUCLEOTIDES: [char; 4] = ['A', 'C', 'G', 'T'];
pub(super) const COMPLEMENT_INDEX: [usize; 4] = [3, 2, 1, 0];
pub(super) fn amino_acid_index(residue: char) -> Option<usize> {
    AMINO_ACIDS
        .iter()
        .position(|&aa| aa == residue.to_ascii_uppercase())
}
pub(super) fn nucleotide_index(base: char) -> Option<usize> {
    match base.to_ascii_uppercase() {
        'A' => Some(0),
        'C' => Some(1),
        'G' => Some(2),
        'T' => Some(3),
        _ => None,
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
        assert!(features.len() > 0);
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
        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);
    }
    #[test]
    fn test_composition_feature_extractor() {
        let sequence = "ATGCATGC";
        let extractor = CompositionFeatureExtractor::new()
            .sequence_type(SequenceType::DNA)
            .include_gc_content(true)
            .include_dinucleotide(true);
        let features = extractor.extract_features(sequence).unwrap();
        assert_eq!(features.len(), 21);
        let gc_content = features[20];
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
        assert_eq!(features.len(), 20);
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
        assert_eq!(features.len(), 2);
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
        let sequence = "AAAC";
        let features = extractor.extract_features(sequence).unwrap();
        assert_eq!(features.len(), 1);
        assert!(features[0] > 0.0);
        assert!(features[0].is_finite());
    }
    #[test]
    fn test_error_cases() {
        let kmer_counter = KmerCounter::new().k(3);
        let result = kmer_counter.extract_features("");
        assert!(result.is_err());
        let result = kmer_counter.extract_features("AT");
        assert!(result.is_err());
        let motif_extractor = SequenceMotifExtractor::new();
        let result = motif_extractor.extract_features(&[]);
        assert!(result.is_err());
    }
    #[test]
    fn test_structural_feature_extractor_protein() {
        let sequence = "ACDEFGHIKLMNPQRSTVWY";
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_hydrophobicity(true)
            .include_charge(true)
            .include_molecular_weight(true)
            .include_secondary_structure(true);
        let features = extractor.extract_features(sequence).unwrap();
        assert_eq!(features.len(), 19);
        for &feature in features.iter() {
            assert!(feature.is_finite(), "Feature should be finite");
        }
        let hydro_mean = features[0];
        assert!(
            hydro_mean > -5.0 && hydro_mean < 5.0,
            "Hydrophobicity mean should be reasonable"
        );
        let total_mw = features[12];
        assert!(total_mw > 0.0, "Total molecular weight should be positive");
    }
    #[test]
    fn test_structural_feature_extractor_dna() {
        let sequence = "ATGCATGCATGC";
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::DNA)
            .include_secondary_structure(true);
        let features = extractor.extract_features(sequence).unwrap();
        assert_eq!(features.len(), 8);
        for &feature in features.iter() {
            assert!(feature.is_finite(), "Feature should be finite");
        }
        let purine_ratio = features[0];
        let pyrimidine_ratio = features[1];
        assert!(
            (purine_ratio + pyrimidine_ratio - 1.0).abs() < 1e-10,
            "Purine + pyrimidine ratios should sum to 1"
        );
        let tm = features[4];
        assert!(
            tm > 0.0 && tm < 200.0,
            "Melting temperature should be reasonable"
        );
    }
    #[test]
    fn test_structural_feature_extractor_charge_features() {
        let sequence = "RRRRRKKKKKDDDDDEEEEE";
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_hydrophobicity(false)
            .include_charge(true)
            .include_molecular_weight(false)
            .include_secondary_structure(false);
        let features = extractor.extract_features(sequence).unwrap();
        assert_eq!(features.len(), 6);
        let net_charge = features[0];
        let charge_density = features[1];
        let positive_ratio = features[2];
        let negative_ratio = features[3];
        let positive_count = features[4];
        let negative_count = features[5];
        assert!((net_charge - 0.0).abs() < 1e-10, "Net charge should be 0");
        assert!(
            (positive_ratio - 0.5).abs() < 1e-10,
            "Positive ratio should be 0.5"
        );
        assert!(
            (negative_ratio - 0.5).abs() < 1e-10,
            "Negative ratio should be 0.5"
        );
        assert!(
            (positive_count - 10.0).abs() < 1e-10,
            "Positive count should be 10"
        );
        assert!(
            (negative_count - 10.0).abs() < 1e-10,
            "Negative count should be 10"
        );
        assert!(charge_density > 0.0, "Charge density should be positive");
    }
    #[test]
    fn test_structural_feature_extractor_hydrophobicity() {
        let sequence = "ILVFWYC";
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::Protein)
            .include_hydrophobicity(true)
            .include_charge(false)
            .include_molecular_weight(false)
            .include_secondary_structure(false);
        let features = extractor.extract_features(sequence).unwrap();
        assert_eq!(features.len(), 6);
        let hydro_mean = features[0];
        let hydro_std = features[1];
        let hydro_min = features[2];
        let hydro_max = features[3];
        let hydro_range = features[4];
        let hydro_var = features[5];
        assert!(
            hydro_mean > 0.0,
            "Hydrophobicity mean should be positive for hydrophobic sequence"
        );
        assert!(
            hydro_range >= 0.0,
            "Hydrophobicity range should be non-negative"
        );
        assert!(hydro_min <= hydro_max, "Min should be <= Max");
        assert!(hydro_var >= 0.0, "Variance should be non-negative");
        assert!(
            (hydro_std - hydro_var.sqrt()).abs() < 1e-10,
            "Std dev should equal sqrt(variance)"
        );
    }
    #[test]
    fn test_structural_feature_extractor_melting_temperature() {
        let short_sequence = "ATGC";
        let long_sequence = "ATGCATGCATGCATGC";
        let extractor = StructuralFeatureExtractor::new()
            .sequence_type(SequenceType::DNA)
            .include_secondary_structure(false);
        let short_features = extractor.extract_features(short_sequence).unwrap();
        let long_features = extractor.extract_features(long_sequence).unwrap();
        assert_eq!(short_features.len(), 4);
        assert_eq!(long_features.len(), 4);
        let short_tm = short_features[0];
        let long_tm = long_features[0];
        assert!(
            (short_tm - 12.0).abs() < 1e-10,
            "Short sequence Tm should be 12"
        );
        assert!(
            long_tm > 50.0 && long_tm < 100.0,
            "Long sequence Tm should be in reasonable range"
        );
    }
    #[test]
    fn test_structural_feature_extractor_error_cases() {
        let extractor = StructuralFeatureExtractor::new();
        let result = extractor.extract_features("");
        assert!(result.is_err());
        let extractor_no_features = StructuralFeatureExtractor::new()
            .include_hydrophobicity(false)
            .include_charge(false)
            .include_molecular_weight(false)
            .include_secondary_structure(false);
        let result = extractor_no_features.extract_features("ACDE");
        assert!(result.is_err());
    }
}
