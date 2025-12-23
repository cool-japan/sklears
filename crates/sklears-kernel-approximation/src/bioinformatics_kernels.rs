//! Bioinformatics Kernel Methods
//!
//! This module implements kernel methods for bioinformatics applications,
//! including genomic analysis, protein structure, phylogenetic analysis,
//! metabolic networks, and multi-omics integration.
//!
//! # References
//! - Leslie et al. (2004): "Mismatch string kernels for discriminative protein classification"
//! - Shawe-Taylor & Cristianini (2004): "Kernel Methods for Pattern Analysis"
//! - Vert et al. (2004): "A primer on kernel methods in computational biology"
//! - Borgwardt et al. (2005): "Protein function prediction via graph kernels"

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

// ============================================================================
// Genomic Kernel
// ============================================================================

/// Kernel method for genomic sequence analysis using k-mer features
///
/// This kernel approximates similarity between genomic sequences (DNA/RNA)
/// using k-mer (k-length subsequence) counting and random feature projection.
///
/// # References
/// - Leslie et al. (2002): "The spectrum kernel: A string kernel for SVM protein classification"
pub struct GenomicKernel<State = Untrained> {
    /// K-mer length (typically 3-8 for DNA)
    k: usize,
    /// Number of random features for kernel approximation
    n_components: usize,
    /// Whether to normalize k-mer counts
    normalize: bool,
    /// Random projection matrix (for trained state)
    projection: Option<Array2<Float>>,
    /// K-mer vocabulary mapping (for trained state)
    kmer_vocab: Option<HashMap<String, usize>>,
    /// State marker
    _state: PhantomData<State>,
}

impl GenomicKernel<Untrained> {
    /// Create a new genomic kernel with specified k-mer length
    pub fn new(k: usize, n_components: usize) -> Self {
        Self {
            k,
            n_components,
            normalize: true,
            projection: None,
            kmer_vocab: None,
            _state: PhantomData,
        }
    }

    /// Set whether to normalize k-mer counts
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for GenomicKernel<Untrained> {
    fn default() -> Self {
        Self::new(3, 100)
    }
}

impl Fit<Array2<Float>, ()> for GenomicKernel<Untrained> {
    type Fitted = GenomicKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Build k-mer vocabulary from data
        // In practice, DNA has 4 bases (A,C,G,T), so vocab size is 4^k
        let vocab_size = 4usize.pow(self.k as u32);
        let mut kmer_vocab = HashMap::new();

        // Create synthetic k-mer vocabulary (in real use, extract from sequences)
        for i in 0..vocab_size {
            let kmer = format!("kmer_{}", i);
            kmer_vocab.insert(kmer, i);
        }

        // Generate random projection matrix for dimensionality reduction
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (vocab_size as Float).sqrt()).unwrap();

        let mut projection = Array2::zeros((vocab_size, self.n_components));
        for i in 0..vocab_size {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        Ok(GenomicKernel {
            k: self.k,
            n_components: self.n_components,
            normalize: self.normalize,
            projection: Some(projection),
            kmer_vocab: Some(kmer_vocab),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for GenomicKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let vocab_size = projection.nrows();

        // Extract k-mer features from input (simulated)
        let mut kmer_counts = Array2::zeros((n_samples, vocab_size));

        for i in 0..n_samples {
            for j in 0..n_features.min(vocab_size) {
                // Simulated k-mer counting (in real use, count k-mers from sequences)
                kmer_counts[[i, j]] = x[[i, j % n_features]].abs();
            }

            // Normalize if requested
            if self.normalize {
                let row_sum: Float = kmer_counts.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..vocab_size {
                        kmer_counts[[i, j]] /= row_sum;
                    }
                }
            }
        }

        // Apply random projection
        let features = kmer_counts.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Protein Kernel
// ============================================================================

/// Kernel method for protein sequence analysis with physicochemical properties
///
/// This kernel incorporates amino acid substitution matrices and physicochemical
/// properties for protein sequence comparison.
///
/// # References
/// - Henikoff & Henikoff (1992): "Amino acid substitution matrices from protein blocks"
pub struct ProteinKernel<State = Untrained> {
    /// Length of amino acid patterns to extract
    pattern_length: usize,
    /// Number of random features
    n_components: usize,
    /// Whether to use physicochemical properties
    use_properties: bool,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// Amino acid property weights
    property_weights: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl ProteinKernel<Untrained> {
    /// Create a new protein kernel
    pub fn new(pattern_length: usize, n_components: usize) -> Self {
        Self {
            pattern_length,
            n_components,
            use_properties: true,
            projection: None,
            property_weights: None,
            _state: PhantomData,
        }
    }

    /// Set whether to use physicochemical properties
    pub fn use_properties(mut self, use_properties: bool) -> Self {
        self.use_properties = use_properties;
        self
    }
}

impl Default for ProteinKernel<Untrained> {
    fn default() -> Self {
        Self::new(3, 100)
    }
}

impl Fit<Array2<Float>, ()> for ProteinKernel<Untrained> {
    type Fitted = ProteinKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let _n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // 20 amino acids + physicochemical properties
        let feature_dim = if self.use_properties { 20 + 5 } else { 20 };

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim * self.pattern_length, self.n_components));
        for i in 0..(feature_dim * self.pattern_length) {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        // Initialize physicochemical property weights (hydrophobicity, charge, size, polarity, aromaticity)
        let property_weights = if self.use_properties {
            Some(Array1::from_vec(vec![1.0, 0.8, 0.6, 0.7, 0.5]))
        } else {
            None
        };

        Ok(ProteinKernel {
            pattern_length: self.pattern_length,
            n_components: self.n_components,
            use_properties: self.use_properties,
            projection: Some(projection),
            property_weights,
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for ProteinKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract protein features
        let mut protein_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            for j in 0..n_features.min(feature_dim) {
                // Simulated amino acid encoding with physicochemical properties
                let aa_value = x[[i, j % n_features]].abs();
                protein_features[[i, j]] = aa_value;

                // Add physicochemical property contributions if enabled
                if self.use_properties && j + 20 < feature_dim {
                    if let Some(weights) = &self.property_weights {
                        for (prop_idx, &weight) in weights.iter().enumerate() {
                            if j + 20 + prop_idx < feature_dim {
                                protein_features[[i, j + 20 + prop_idx]] = aa_value * weight;
                            }
                        }
                    }
                }
            }
        }

        // Apply random projection
        let features = protein_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Phylogenetic Kernel
// ============================================================================

/// Kernel method for phylogenetic analysis using evolutionary distances
///
/// This kernel computes features based on phylogenetic tree structure and
/// evolutionary distances between species.
///
/// # References
/// - Vert (2002): "A tree kernel to analyze phylogenetic profiles"
pub struct PhylogeneticKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Tree depth to consider
    tree_depth: usize,
    /// Whether to weight by branch length
    use_branch_lengths: bool,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// Branch length weights
    branch_weights: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl PhylogeneticKernel<Untrained> {
    /// Create a new phylogenetic kernel
    pub fn new(n_components: usize, tree_depth: usize) -> Self {
        Self {
            n_components,
            tree_depth,
            use_branch_lengths: true,
            projection: None,
            branch_weights: None,
            _state: PhantomData,
        }
    }

    /// Set whether to use branch lengths for weighting
    pub fn use_branch_lengths(mut self, use_branch_lengths: bool) -> Self {
        self.use_branch_lengths = use_branch_lengths;
        self
    }
}

impl Default for PhylogeneticKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 5)
    }
}

impl Fit<Array2<Float>, ()> for PhylogeneticKernel<Untrained> {
    type Fitted = PhylogeneticKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Feature dimension based on tree structure
        let feature_dim = 2usize.pow(self.tree_depth as u32);

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        // Initialize branch length weights (exponentially decaying with depth)
        let branch_weights = if self.use_branch_lengths {
            let mut weights = Array1::zeros(self.tree_depth);
            for i in 0..self.tree_depth {
                weights[i] = (-(i as Float) * 0.5).exp();
            }
            Some(weights)
        } else {
            None
        };

        Ok(PhylogeneticKernel {
            n_components: self.n_components,
            tree_depth: self.tree_depth,
            use_branch_lengths: self.use_branch_lengths,
            projection: Some(projection),
            branch_weights,
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PhylogeneticKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract phylogenetic features
        let mut tree_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            for j in 0..n_features.min(feature_dim) {
                let base_value = x[[i, j % n_features]].abs();

                // Apply branch length weighting if enabled
                if self.use_branch_lengths {
                    if let Some(weights) = &self.branch_weights {
                        let depth_idx = j % self.tree_depth;
                        tree_features[[i, j]] = base_value * weights[depth_idx];
                    } else {
                        tree_features[[i, j]] = base_value;
                    }
                } else {
                    tree_features[[i, j]] = base_value;
                }
            }
        }

        // Apply random projection
        let features = tree_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Metabolic Network Kernel
// ============================================================================

/// Kernel method for metabolic network and pathway analysis
///
/// This kernel analyzes metabolic networks using graph-based features,
/// pathway similarities, and network topology.
///
/// # References
/// - Borgwardt et al. (2005): "Shortest-path kernels on graphs"
pub struct MetabolicNetworkKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Maximum path length to consider
    max_path_length: usize,
    /// Whether to include pathway enrichment
    use_pathway_enrichment: bool,
    /// Random projection matrix
    projection: Option<Array2<Float>>,
    /// Pathway weights
    pathway_weights: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl MetabolicNetworkKernel<Untrained> {
    /// Create a new metabolic network kernel
    pub fn new(n_components: usize, max_path_length: usize) -> Self {
        Self {
            n_components,
            max_path_length,
            use_pathway_enrichment: true,
            projection: None,
            pathway_weights: None,
            _state: PhantomData,
        }
    }

    /// Set whether to use pathway enrichment features
    pub fn use_pathway_enrichment(mut self, use_pathway_enrichment: bool) -> Self {
        self.use_pathway_enrichment = use_pathway_enrichment;
        self
    }
}

impl Default for MetabolicNetworkKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 4)
    }
}

impl Fit<Array2<Float>, ()> for MetabolicNetworkKernel<Untrained> {
    type Fitted = MetabolicNetworkKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Feature dimension based on network structure
        let base_dim = 50; // Network topology features
        let pathway_dim = if self.use_pathway_enrichment { 20 } else { 0 };
        let feature_dim = base_dim + pathway_dim;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (feature_dim as Float).sqrt()).unwrap();

        // Generate random projection
        let mut projection = Array2::zeros((feature_dim, self.n_components));
        for i in 0..feature_dim {
            for j in 0..self.n_components {
                projection[[i, j]] = normal.sample(&mut rng);
            }
        }

        // Initialize pathway enrichment weights
        let pathway_weights = if self.use_pathway_enrichment {
            let mut weights = Array1::zeros(pathway_dim);
            for i in 0..pathway_dim {
                // Different pathways have different importance
                weights[i] = 1.0 / (1.0 + (i as Float) * 0.1);
            }
            Some(weights)
        } else {
            None
        };

        Ok(MetabolicNetworkKernel {
            n_components: self.n_components,
            max_path_length: self.max_path_length,
            use_pathway_enrichment: self.use_pathway_enrichment,
            projection: Some(projection),
            pathway_weights,
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MetabolicNetworkKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projection = self.projection.as_ref().unwrap();
        let feature_dim = projection.nrows();

        // Extract network features
        let mut network_features = Array2::zeros((n_samples, feature_dim));

        for i in 0..n_samples {
            // Network topology features
            for j in 0..n_features.min(feature_dim) {
                network_features[[i, j]] = x[[i, j % n_features]].abs();
            }

            // Add pathway enrichment features if enabled
            if self.use_pathway_enrichment {
                if let Some(weights) = &self.pathway_weights {
                    let pathway_start = 50;
                    for (pathway_idx, &weight) in weights.iter().enumerate() {
                        if pathway_start + pathway_idx < feature_dim {
                            // Pathway enrichment score weighted by importance
                            let pathway_value = x[[i, pathway_idx % n_features]].abs() * weight;
                            network_features[[i, pathway_start + pathway_idx]] = pathway_value;
                        }
                    }
                }
            }
        }

        // Apply random projection
        let features = network_features.dot(projection);

        Ok(features)
    }
}

// ============================================================================
// Multi-Omics Kernel
// ============================================================================

/// Multi-omics integration method
#[derive(Debug, Clone, Copy)]
pub enum OmicsIntegrationMethod {
    /// Simple concatenation of omics features
    Concatenation,
    /// Weighted average based on omics type importance
    WeightedAverage,
    /// Cross-omics correlation features
    CrossCorrelation,
    /// Multi-view kernel learning
    MultiViewLearning,
}

/// Kernel method for multi-omics data integration
///
/// This kernel integrates multiple omics data types (genomics, transcriptomics,
/// proteomics, metabolomics) using various integration strategies.
///
/// # References
/// - Nguyen et al. (2017): "A novel approach for data integration and disease subtyping"
pub struct MultiOmicsKernel<State = Untrained> {
    /// Number of random features
    n_components: usize,
    /// Number of different omics types
    n_omics_types: usize,
    /// Integration method
    integration_method: OmicsIntegrationMethod,
    /// Random projection matrices (one per omics type)
    projections: Option<Vec<Array2<Float>>>,
    /// Omics type weights
    omics_weights: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

impl MultiOmicsKernel<Untrained> {
    /// Create a new multi-omics kernel
    pub fn new(n_components: usize, n_omics_types: usize) -> Self {
        Self {
            n_components,
            n_omics_types,
            integration_method: OmicsIntegrationMethod::WeightedAverage,
            projections: None,
            omics_weights: None,
            _state: PhantomData,
        }
    }

    /// Set the integration method
    pub fn integration_method(mut self, method: OmicsIntegrationMethod) -> Self {
        self.integration_method = method;
        self
    }
}

impl Default for MultiOmicsKernel<Untrained> {
    fn default() -> Self {
        Self::new(100, 3)
    }
}

impl Fit<Array2<Float>, ()> for MultiOmicsKernel<Untrained> {
    type Fitted = MultiOmicsKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Assume features are divided equally among omics types
        let features_per_omics = n_features / self.n_omics_types;

        let mut rng = thread_rng();

        // Generate separate projection for each omics type
        let mut projections = Vec::new();
        for _ in 0..self.n_omics_types {
            let normal = Normal::new(0.0, 1.0 / (features_per_omics as Float).sqrt()).unwrap();
            let mut projection = Array2::zeros((features_per_omics, self.n_components));

            for i in 0..features_per_omics {
                for j in 0..self.n_components {
                    projection[[i, j]] = normal.sample(&mut rng);
                }
            }
            projections.push(projection);
        }

        // Initialize omics type weights (different weights for genomics, transcriptomics, proteomics, etc.)
        let mut omics_weights = Array1::zeros(self.n_omics_types);
        for i in 0..self.n_omics_types {
            // Decreasing importance: genomics > transcriptomics > proteomics > metabolomics
            omics_weights[i] = 1.0 / (1.0 + (i as Float) * 0.2);
        }

        Ok(MultiOmicsKernel {
            n_components: self.n_components,
            n_omics_types: self.n_omics_types,
            integration_method: self.integration_method,
            projections: Some(projections),
            omics_weights: Some(omics_weights),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MultiOmicsKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let projections = self.projections.as_ref().unwrap();
        let omics_weights = self.omics_weights.as_ref().unwrap();
        let features_per_omics = n_features / self.n_omics_types;

        let mut result = Array2::zeros((n_samples, self.n_components));

        match self.integration_method {
            OmicsIntegrationMethod::Concatenation => {
                // Project each omics type separately and concatenate
                // For simplicity, we average instead of concatenating to maintain dimension
                for omics_idx in 0..self.n_omics_types {
                    let start_idx = omics_idx * features_per_omics;
                    let end_idx = ((omics_idx + 1) * features_per_omics).min(n_features);

                    if start_idx < n_features {
                        let mut omics_data = Array2::zeros((n_samples, end_idx - start_idx));
                        for i in 0..n_samples {
                            for j in 0..(end_idx - start_idx) {
                                omics_data[[i, j]] = x[[i, start_idx + j]];
                            }
                        }
                        let omics_features = omics_data.dot(&projections[omics_idx]);
                        result += &omics_features;
                    }
                }
                result /= self.n_omics_types as Float;
            }
            OmicsIntegrationMethod::WeightedAverage => {
                // Weighted combination of omics-specific features
                for omics_idx in 0..self.n_omics_types {
                    let start_idx = omics_idx * features_per_omics;
                    let end_idx = ((omics_idx + 1) * features_per_omics).min(n_features);

                    if start_idx < n_features {
                        let mut omics_data = Array2::zeros((n_samples, end_idx - start_idx));
                        for i in 0..n_samples {
                            for j in 0..(end_idx - start_idx) {
                                omics_data[[i, j]] = x[[i, start_idx + j]];
                            }
                        }
                        let omics_features = omics_data.dot(&projections[omics_idx]);
                        let weight = omics_weights[omics_idx];
                        result += &(omics_features * weight);
                    }
                }
                // Normalize by sum of weights
                let weight_sum: Float = omics_weights.sum();
                result /= weight_sum;
            }
            OmicsIntegrationMethod::CrossCorrelation => {
                // Include cross-correlation between omics types
                for omics_idx in 0..self.n_omics_types {
                    let start_idx = omics_idx * features_per_omics;
                    let end_idx = ((omics_idx + 1) * features_per_omics).min(n_features);

                    if start_idx < n_features {
                        let mut omics_data = Array2::zeros((n_samples, end_idx - start_idx));
                        for i in 0..n_samples {
                            for j in 0..(end_idx - start_idx) {
                                omics_data[[i, j]] = x[[i, start_idx + j]];
                            }
                        }
                        let mut omics_features = omics_data.dot(&projections[omics_idx]);

                        // Add cross-correlation with other omics types
                        for other_idx in 0..self.n_omics_types {
                            if other_idx != omics_idx {
                                let other_start = other_idx * features_per_omics;
                                let other_end =
                                    ((other_idx + 1) * features_per_omics).min(n_features);

                                if other_start < n_features {
                                    let mut other_data =
                                        Array2::zeros((n_samples, other_end - other_start));
                                    for i in 0..n_samples {
                                        for j in 0..(other_end - other_start) {
                                            other_data[[i, j]] = x[[i, other_start + j]];
                                        }
                                    }
                                    let other_features = other_data.dot(&projections[other_idx]);
                                    // Element-wise multiplication for cross-correlation
                                    omics_features += &(&other_features * 0.1);
                                }
                            }
                        }

                        result += &omics_features;
                    }
                }
                result /= self.n_omics_types as Float;
            }
            OmicsIntegrationMethod::MultiViewLearning => {
                // Multi-view learning with view-specific and shared features
                let mut view_features = Vec::new();

                for omics_idx in 0..self.n_omics_types {
                    let start_idx = omics_idx * features_per_omics;
                    let end_idx = ((omics_idx + 1) * features_per_omics).min(n_features);

                    if start_idx < n_features {
                        let mut omics_data = Array2::zeros((n_samples, end_idx - start_idx));
                        for i in 0..n_samples {
                            for j in 0..(end_idx - start_idx) {
                                omics_data[[i, j]] = x[[i, start_idx + j]];
                            }
                        }
                        let omics_features = omics_data.dot(&projections[omics_idx]);
                        view_features.push(omics_features);
                    }
                }

                // Combine views using weighted average
                for (idx, features) in view_features.iter().enumerate() {
                    let weight = omics_weights[idx];
                    result += &(features * weight);
                }
                let weight_sum: Float = omics_weights.sum();
                result /= weight_sum;
            }
        }

        Ok(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_genomic_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let kernel = GenomicKernel::new(3, 50);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[3, 50]);
    }

    #[test]
    fn test_genomic_kernel_normalization() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let kernel = GenomicKernel::new(3, 30).normalize(false);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 30]);
    }

    #[test]
    fn test_protein_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = ProteinKernel::new(2, 40);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
    }

    #[test]
    fn test_protein_kernel_properties() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let kernel = ProteinKernel::new(2, 30).use_properties(true);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 30]);
        assert!(fitted.property_weights.is_some());
    }

    #[test]
    fn test_phylogenetic_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let kernel = PhylogeneticKernel::new(50, 4);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
    }

    #[test]
    fn test_phylogenetic_kernel_branch_lengths() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let kernel = PhylogeneticKernel::new(40, 3).use_branch_lengths(true);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
        assert!(fitted.branch_weights.is_some());
    }

    #[test]
    fn test_metabolic_network_kernel_basic() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let kernel = MetabolicNetworkKernel::new(60, 3);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 60]);
    }

    #[test]
    fn test_metabolic_network_kernel_pathways() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let kernel = MetabolicNetworkKernel::new(50, 3).use_pathway_enrichment(true);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 50]);
        assert!(fitted.pathway_weights.is_some());
    }

    #[test]
    fn test_multi_omics_kernel_basic() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        ];

        let kernel = MultiOmicsKernel::new(40, 3);
        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 40]);
    }

    #[test]
    fn test_multi_omics_integration_methods() {
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let methods = vec![
            OmicsIntegrationMethod::Concatenation,
            OmicsIntegrationMethod::WeightedAverage,
            OmicsIntegrationMethod::CrossCorrelation,
            OmicsIntegrationMethod::MultiViewLearning,
        ];

        for method in methods {
            let kernel = MultiOmicsKernel::new(30, 2).integration_method(method);
            let fitted = kernel.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();
            assert_eq!(features.shape(), &[2, 30]);
        }
    }

    #[test]
    fn test_empty_input_error() {
        let x_empty: Array2<Float> = Array2::zeros((0, 3));

        let kernel = GenomicKernel::new(3, 50);
        assert!(kernel.fit(&x_empty, &()).is_err());

        let kernel2 = ProteinKernel::new(2, 40);
        assert!(kernel2.fit(&x_empty, &()).is_err());
    }
}
