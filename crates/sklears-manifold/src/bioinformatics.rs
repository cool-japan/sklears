use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::ndarray_linalg::{Eigh, SVD, UPLO};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// Genomic Manifold Analysis
///
/// Learns manifold structure from genomic data (e.g., gene expression, SNP data).
/// Useful for identifying population structure, disease subtypes, and gene expression patterns.
#[derive(Debug, Clone)]
pub struct GenomicManifoldAnalysis<S = Untrained> {
    n_genes: usize,
    n_components: usize,
    embedding_method: GenomicEmbeddingMethod,
    normalization: GenomicNormalization,
    state: S,
}

#[derive(Debug, Clone)]
pub enum GenomicEmbeddingMethod {
    /// PCA for genomic data
    PCA,
    /// t-SNE for nonlinear genomic structure
    TSNE { perplexity: f64, n_iter: usize },
    /// UMAP for genomic data visualization
    UMAP { n_neighbors: usize, min_dist: f64 },
}

#[derive(Debug, Clone)]
pub enum GenomicNormalization {
    /// No normalization
    None,
    /// Log transformation (log2(x + 1))
    Log2,
    /// Z-score normalization
    ZScore,
    /// TPM (Transcripts Per Million) normalization
    TPM,
}

#[derive(Debug, Clone)]
pub struct TrainedGenomicManifold {
    n_genes: usize,
    n_components: usize,
    embedding_method: GenomicEmbeddingMethod,
    normalization: GenomicNormalization,
    genomic_manifold: Array2<f64>,  // Learned manifold embedding
    projection_matrix: Array2<f64>, // Projection from gene space to manifold
    gene_mean: Array1<f64>,         // Mean expression for normalization
    gene_std: Array1<f64>,          // Std deviation for normalization
}

impl GenomicManifoldAnalysis<Untrained> {
    /// Create a new genomic manifold analysis model
    ///
    /// # Arguments
    /// * `n_genes` - Number of genes/genomic features
    pub fn new(n_genes: usize) -> Self {
        Self {
            n_genes,
            n_components: 50,
            embedding_method: GenomicEmbeddingMethod::PCA,
            normalization: GenomicNormalization::Log2,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_embedding_method(mut self, method: GenomicEmbeddingMethod) -> Self {
        self.embedding_method = method;
        self
    }

    pub fn with_normalization(mut self, normalization: GenomicNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    fn normalize_data(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        match &self.normalization {
            GenomicNormalization::None => Ok(data.clone()),
            GenomicNormalization::Log2 => Ok(data.mapv(|x| (x + 1.0).log2())),
            GenomicNormalization::ZScore => {
                let mean = data.mean_axis(Axis(0)).unwrap();
                let std = data.std_axis(Axis(0), 0.0);
                let mut normalized = Array2::zeros(data.dim());
                for (i, mut row) in normalized.axis_iter_mut(Axis(0)).enumerate() {
                    let data_row = data.row(i);
                    for (j, val) in row.iter_mut().enumerate() {
                        let std_val = std[j];
                        *val = if std_val > 1e-10 {
                            (data_row[j] - mean[j]) / std_val
                        } else {
                            0.0
                        };
                    }
                }
                Ok(normalized)
            }
            GenomicNormalization::TPM => {
                // Simplified TPM: scale each sample to sum to 1 million
                let mut normalized = Array2::zeros(data.dim());
                for (i, mut row) in normalized.axis_iter_mut(Axis(0)).enumerate() {
                    let data_row = data.row(i);
                    let sum: f64 = data_row.iter().sum();
                    if sum > 0.0 {
                        for (j, val) in row.iter_mut().enumerate() {
                            *val = (data_row[j] / sum) * 1_000_000.0;
                        }
                    }
                }
                Ok(normalized)
            }
        }
    }

    fn compute_genomic_embedding(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        match &self.embedding_method {
            GenomicEmbeddingMethod::PCA => {
                let mean = data.mean_axis(Axis(0)).unwrap();
                let centered = data - &mean.insert_axis(Axis(0));

                let (_, _, vt) = centered
                    .svd(false, true)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

                let v = vt.unwrap().t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                let projection = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

                Ok(centered.dot(&projection))
            }
            GenomicEmbeddingMethod::TSNE { perplexity, n_iter } => {
                // Simplified t-SNE
                let n_samples = data.nrows();

                // Initialize with PCA
                let mean = data.mean_axis(Axis(0)).unwrap();
                let centered = data - &mean.insert_axis(Axis(0));
                let (_, _, vt) = centered
                    .svd(false, true)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
                let v = vt.unwrap().t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                let projection = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
                let mut embedding = centered.dot(&projection);

                // Simplified gradient descent (placeholder for full t-SNE)
                for _iter in 0..*n_iter {
                    let mut gradient = Array2::<f64>::zeros(embedding.dim());
                    for i in 0..n_samples {
                        for j in 0..n_samples {
                            if i != j {
                                let diff = embedding.row(i).to_owned() - embedding.row(j);
                                let dist_sq = diff.mapv(|x| x * x).sum() + 1.0;
                                let force = &diff / dist_sq;
                                for k in 0..n_comp {
                                    gradient[[i, k]] += force[k] * 0.01;
                                }
                            }
                        }
                    }
                    embedding = embedding - &gradient * 0.1;
                }

                Ok(embedding)
            }
            GenomicEmbeddingMethod::UMAP {
                n_neighbors,
                min_dist,
            } => {
                // Simplified UMAP (use PCA as placeholder)
                let mean = data.mean_axis(Axis(0)).unwrap();
                let centered = data - &mean.insert_axis(Axis(0));
                let (_, _, vt) = centered
                    .svd(false, true)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
                let v = vt.unwrap().t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                let projection = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
                Ok(centered.dot(&projection))
            }
        }
    }
}

impl GenomicManifoldAnalysis<TrainedGenomicManifold> {
    /// Project new genomic data onto learned manifold
    pub fn project_genomic_data(&self, data: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        if data.ncols() != self.state.n_genes {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} genes, got {}",
                self.state.n_genes,
                data.ncols()
            )));
        }

        let mut normalized = Array2::zeros(data.dim());
        for (i, mut row) in normalized.axis_iter_mut(Axis(0)).enumerate() {
            let data_row = data.row(i);
            for (j, val) in row.iter_mut().enumerate() {
                let std_val = self.state.gene_std[j];
                *val = if std_val > 1e-10 {
                    (data_row[j] - self.state.gene_mean[j]) / std_val
                } else {
                    0.0
                };
            }
        }

        Ok(normalized.dot(&self.state.projection_matrix))
    }

    /// Identify differentially expressed genes based on manifold structure
    pub fn differential_genes(&self, threshold: f64) -> Vec<usize> {
        let mut diff_genes = Vec::new();

        for gene_idx in 0..self.state.n_genes {
            let gene_importance = self
                .state
                .projection_matrix
                .row(gene_idx)
                .mapv(|x| x.abs())
                .sum();
            if gene_importance > threshold {
                diff_genes.push(gene_idx);
            }
        }

        diff_genes
    }

    /// Compute sample-to-sample similarity in genomic manifold
    pub fn sample_similarity(
        &self,
        sample1: &ArrayView1<f64>,
        sample2: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        let proj1 = self.state.projection_matrix.t().dot(sample1);
        let proj2 = self.state.projection_matrix.t().dot(sample2);

        let diff = &proj1 - &proj2;
        let dist = diff.mapv(|x| x * x).sum().sqrt();

        // Convert distance to similarity (0 to 1)
        Ok((-dist / 10.0).exp())
    }
}

impl Fit<Array2<f64>, ()> for GenomicManifoldAnalysis<Untrained> {
    type Fitted = GenomicManifoldAnalysis<TrainedGenomicManifold>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.ncols() != self.n_genes {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} genes, got {}",
                self.n_genes,
                x.ncols()
            )));
        }

        let normalized = self.normalize_data(x)?;

        let gene_mean = normalized.mean_axis(Axis(0)).unwrap();
        let gene_std = normalized.std_axis(Axis(0), 0.0);

        let genomic_manifold = self.compute_genomic_embedding(&normalized)?;

        // Compute projection matrix
        let centered = &normalized - &gene_mean.clone().insert_axis(Axis(0));
        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        Ok(GenomicManifoldAnalysis {
            n_genes: self.n_genes,
            n_components: self.n_components,
            embedding_method: self.embedding_method.clone(),
            normalization: self.normalization.clone(),
            state: TrainedGenomicManifold {
                n_genes: self.n_genes,
                n_components: self.n_components,
                embedding_method: self.embedding_method.clone(),
                normalization: self.normalization.clone(),
                genomic_manifold,
                projection_matrix,
                gene_mean,
                gene_std,
            },
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for GenomicManifoldAnalysis<TrainedGenomicManifold> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.project_genomic_data(&x.view())
    }
}

/// Protein Structure Manifolds
///
/// Learns manifold representations of protein structures for classification,
/// structure prediction, and conformational analysis.
#[derive(Debug, Clone)]
pub struct ProteinStructureManifold<S = Untrained> {
    n_residues: usize,
    n_components: usize,
    representation: ProteinRepresentation,
    state: S,
}

#[derive(Debug, Clone)]
pub enum ProteinRepresentation {
    /// Contact map representation
    ContactMap,
    /// Distance matrix between residues
    DistanceMatrix,
    /// Dihedral angles (phi, psi, omega)
    DihedralAngles,
    /// Secondary structure encoding
    SecondaryStructure,
}

#[derive(Debug, Clone)]
pub struct TrainedProteinStructure {
    n_residues: usize,
    n_components: usize,
    representation: ProteinRepresentation,
    structure_manifold: Array2<f64>, // Learned structure manifold
    projection_matrix: Array2<f64>,  // Projection matrix
    mean_structure: Array1<f64>,     // Mean structure for normalization
}

impl ProteinStructureManifold<Untrained> {
    /// Create a new protein structure manifold model
    ///
    /// # Arguments
    /// * `n_residues` - Number of residues in the protein
    pub fn new(n_residues: usize) -> Self {
        Self {
            n_residues,
            n_components: 20,
            representation: ProteinRepresentation::ContactMap,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_representation(mut self, representation: ProteinRepresentation) -> Self {
        self.representation = representation;
        self
    }

    fn validate_structure(&self, structure: &ArrayView2<f64>) -> SklResult<()> {
        match &self.representation {
            ProteinRepresentation::ContactMap | ProteinRepresentation::DistanceMatrix => {
                if structure.nrows() != self.n_residues || structure.ncols() != self.n_residues {
                    return Err(SklearsError::InvalidInput(format!(
                        "Expected {}x{} matrix, got {}x{}",
                        self.n_residues,
                        self.n_residues,
                        structure.nrows(),
                        structure.ncols()
                    )));
                }
            }
            ProteinRepresentation::DihedralAngles => {
                if structure.ncols() != 3 {
                    return Err(SklearsError::InvalidInput(
                        "Dihedral angles require 3 columns (phi, psi, omega)".to_string(),
                    ));
                }
            }
            ProteinRepresentation::SecondaryStructure => {
                if structure.ncols() != 3 {
                    return Err(SklearsError::InvalidInput(
                        "Secondary structure requires 3 columns (helix, sheet, coil)".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn flatten_structure(&self, structure: &ArrayView2<f64>) -> Array1<f64> {
        match &self.representation {
            ProteinRepresentation::ContactMap | ProteinRepresentation::DistanceMatrix => {
                // Use upper triangle only (symmetric matrix)
                let mut flat = Vec::new();
                for i in 0..structure.nrows() {
                    for j in i + 1..structure.ncols() {
                        flat.push(structure[[i, j]]);
                    }
                }
                Array1::from_vec(flat)
            }
            _ => {
                // Flatten directly
                Array1::from_iter(structure.iter().copied())
            }
        }
    }
}

impl ProteinStructureManifold<TrainedProteinStructure> {
    /// Project protein structure onto learned manifold
    pub fn project_structure(&self, structure: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let flat = match &self.state.representation {
            ProteinRepresentation::ContactMap | ProteinRepresentation::DistanceMatrix => {
                let mut flat_vec = Vec::new();
                for i in 0..structure.nrows() {
                    for j in i + 1..structure.ncols() {
                        flat_vec.push(structure[[i, j]]);
                    }
                }
                Array1::from_vec(flat_vec)
            }
            _ => Array1::from_iter(structure.iter().copied()),
        };

        let normalized = &flat - &self.state.mean_structure;
        Ok(self.state.projection_matrix.t().dot(&normalized))
    }

    /// Compute structural similarity between two proteins
    pub fn structural_similarity(
        &self,
        struct1: &ArrayView2<f64>,
        struct2: &ArrayView2<f64>,
    ) -> SklResult<f64> {
        let proj1 = self.project_structure(struct1)?;
        let proj2 = self.project_structure(struct2)?;

        let diff = &proj1 - &proj2;
        let dist = diff.mapv(|x| x * x).sum().sqrt();

        // Convert to similarity score
        Ok((-dist / 5.0).exp())
    }

    /// Identify key structural features
    pub fn key_structural_features(&self, n_features: usize) -> Vec<(usize, usize, f64)> {
        let mut features = Vec::new();

        match &self.state.representation {
            ProteinRepresentation::ContactMap | ProteinRepresentation::DistanceMatrix => {
                let mut idx = 0;
                for i in 0..self.state.n_residues {
                    for j in i + 1..self.state.n_residues {
                        if idx < self.state.projection_matrix.nrows() {
                            let importance = self
                                .state
                                .projection_matrix
                                .row(idx)
                                .mapv(|x| x.abs())
                                .sum();
                            features.push((i, j, importance));
                        }
                        idx += 1;
                    }
                }
            }
            _ => {
                for i in 0..self
                    .state
                    .n_residues
                    .min(self.state.projection_matrix.nrows())
                {
                    let importance = self.state.projection_matrix.row(i).mapv(|x| x.abs()).sum();
                    features.push((i, 0, importance));
                }
            }
        }

        features.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        features.truncate(n_features);
        features
    }
}

impl Fit<Vec<Array2<f64>>, ()> for ProteinStructureManifold<Untrained> {
    type Fitted = ProteinStructureManifold<TrainedProteinStructure>;

    fn fit(self, x: &Vec<Array2<f64>>, _y: &()) -> SklResult<Self::Fitted> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty structure list".to_string(),
            ));
        }

        // Validate and flatten all structures
        let mut flattened = Vec::new();
        for structure in x {
            self.validate_structure(&structure.view())?;
            flattened.push(self.flatten_structure(&structure.view()));
        }

        let n_samples = flattened.len();
        let n_features = flattened[0].len();

        let mut data = Array2::zeros((n_samples, n_features));
        for (i, flat) in flattened.iter().enumerate() {
            for (j, &val) in flat.iter().enumerate() {
                data[[i, j]] = val;
            }
        }

        let mean_structure = data.mean_axis(Axis(0)).unwrap();
        let centered = &data - &mean_structure.clone().insert_axis(Axis(0));

        // Compute manifold embedding using PCA
        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        let structure_manifold = centered.dot(&projection_matrix);

        Ok(ProteinStructureManifold {
            n_residues: self.n_residues,
            n_components: self.n_components,
            representation: self.representation.clone(),
            state: TrainedProteinStructure {
                n_residues: self.n_residues,
                n_components: self.n_components,
                representation: self.representation.clone(),
                structure_manifold,
                projection_matrix,
                mean_structure,
            },
        })
    }
}

impl Transform<Array2<f64>, Array1<f64>> for ProteinStructureManifold<TrainedProteinStructure> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        self.project_structure(&x.view())
    }
}

/// Phylogenetic Embedding
///
/// Learns manifold representations of phylogenetic relationships for
/// evolutionary analysis and taxonomy.
#[derive(Debug, Clone)]
pub struct PhylogeneticEmbedding<S = Untrained> {
    n_taxa: usize,
    n_components: usize,
    distance_method: PhylogeneticDistance,
    state: S,
}

#[derive(Debug, Clone)]
pub enum PhylogeneticDistance {
    /// Hamming distance for sequence alignment
    Hamming,
    /// Jukes-Cantor distance
    JukesCantor,
    /// Kimura 2-parameter distance
    Kimura2P,
    /// Patristic distance from tree
    Patristic,
}

#[derive(Debug, Clone)]
pub struct TrainedPhylogenetic {
    n_taxa: usize,
    n_components: usize,
    distance_method: PhylogeneticDistance,
    phylo_embedding: Array2<f64>, // Phylogenetic manifold embedding
    distance_matrix: Array2<f64>, // Pairwise phylogenetic distances
}

impl PhylogeneticEmbedding<Untrained> {
    /// Create a new phylogenetic embedding model
    ///
    /// # Arguments
    /// * `n_taxa` - Number of taxa (species/sequences)
    pub fn new(n_taxa: usize) -> Self {
        Self {
            n_taxa,
            n_components: 10,
            distance_method: PhylogeneticDistance::JukesCantor,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_distance_method(mut self, method: PhylogeneticDistance) -> Self {
        self.distance_method = method;
        self
    }

    fn compute_phylogenetic_distance(&self, dist_matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        match &self.distance_method {
            PhylogeneticDistance::Hamming => {
                // Already in Hamming distance format
                Ok(dist_matrix.clone())
            }
            PhylogeneticDistance::JukesCantor => {
                // Jukes-Cantor correction: d = -3/4 * ln(1 - 4/3 * p)
                Ok(dist_matrix.mapv(|p| {
                    let corrected = 1.0 - 4.0 / 3.0 * p;
                    if corrected > 0.0 {
                        -0.75 * corrected.ln()
                    } else {
                        10.0 // Maximum distance for saturated sequences
                    }
                }))
            }
            PhylogeneticDistance::Kimura2P => {
                // Simplified Kimura 2-parameter (assuming equal transition/transversion)
                Ok(dist_matrix.mapv(|p| {
                    let corrected = 1.0 - 2.0 * p;
                    if corrected > 0.0 {
                        -0.5 * corrected.ln()
                    } else {
                        10.0
                    }
                }))
            }
            PhylogeneticDistance::Patristic => {
                // Use provided distances as patristic distances
                Ok(dist_matrix.clone())
            }
        }
    }
}

impl PhylogeneticEmbedding<TrainedPhylogenetic> {
    /// Get embedding coordinates for a taxon
    pub fn taxon_embedding(&self, taxon_idx: usize) -> SklResult<Array1<f64>> {
        if taxon_idx >= self.state.n_taxa {
            return Err(SklearsError::InvalidInput(format!(
                "Taxon index {} out of range (max: {})",
                taxon_idx,
                self.state.n_taxa - 1
            )));
        }

        Ok(self.state.phylo_embedding.row(taxon_idx).to_owned())
    }

    /// Compute phylogenetic distance between two taxa
    pub fn taxon_distance(&self, taxon1: usize, taxon2: usize) -> SklResult<f64> {
        if taxon1 >= self.state.n_taxa || taxon2 >= self.state.n_taxa {
            return Err(SklearsError::InvalidInput(
                "Taxon index out of range".to_string(),
            ));
        }

        Ok(self.state.distance_matrix[[taxon1, taxon2]])
    }

    /// Find nearest neighbors in phylogenetic space
    pub fn nearest_neighbors(&self, taxon_idx: usize, k: usize) -> SklResult<Vec<(usize, f64)>> {
        if taxon_idx >= self.state.n_taxa {
            return Err(SklearsError::InvalidInput(
                "Taxon index out of range".to_string(),
            ));
        }

        let mut neighbors: Vec<(usize, f64)> = (0..self.state.n_taxa)
            .filter(|&i| i != taxon_idx)
            .map(|i| (i, self.state.distance_matrix[[taxon_idx, i]]))
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(k);

        Ok(neighbors)
    }
}

impl Fit<Array2<f64>, ()> for PhylogeneticEmbedding<Untrained> {
    type Fitted = PhylogeneticEmbedding<TrainedPhylogenetic>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.nrows() != self.n_taxa || x.ncols() != self.n_taxa {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {}x{} distance matrix, got {}x{}",
                self.n_taxa,
                self.n_taxa,
                x.nrows(),
                x.ncols()
            )));
        }

        let distance_matrix = self.compute_phylogenetic_distance(x)?;

        // Classical MDS for phylogenetic embedding
        let n = distance_matrix.nrows();
        let mut gram = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                gram[[i, j]] = -0.5 * distance_matrix[[i, j]].powi(2);
            }
        }

        // Double centering
        let row_mean = gram.mean_axis(Axis(1)).unwrap();
        let col_mean = gram.mean_axis(Axis(0)).unwrap();
        let total_mean = gram.mean().unwrap();

        for i in 0..n {
            for j in 0..n {
                gram[[i, j]] = gram[[i, j]] - row_mean[i] - col_mean[j] + total_mean;
            }
        }

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = gram.eigh(UPLO::Lower).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // Sort eigenvalues in descending order
        let mut eigen_pairs: Vec<_> = eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > 1e-10)
            .collect();
        eigen_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        // Build embedding
        let n_comp = self.n_components.min(eigen_pairs.len());
        let mut phylo_embedding = Array2::zeros((n, n_comp));
        for (comp_idx, (eig_idx, eigenval)) in eigen_pairs[..n_comp].iter().enumerate() {
            let scale = eigenval.sqrt();
            for i in 0..n {
                phylo_embedding[[i, comp_idx]] = eigenvectors[[i, *eig_idx]] * scale;
            }
        }

        Ok(PhylogeneticEmbedding {
            n_taxa: self.n_taxa,
            n_components: self.n_components,
            distance_method: self.distance_method.clone(),
            state: TrainedPhylogenetic {
                n_taxa: self.n_taxa,
                n_components: self.n_components,
                distance_method: self.distance_method.clone(),
                phylo_embedding,
                distance_matrix,
            },
        })
    }
}

impl Transform<usize, Array1<f64>> for PhylogeneticEmbedding<TrainedPhylogenetic> {
    fn transform(&self, taxon_idx: &usize) -> SklResult<Array1<f64>> {
        self.taxon_embedding(*taxon_idx)
    }
}

/// Single-Cell Trajectory Analysis
///
/// Learns developmental trajectories and pseudotime orderings from single-cell data.
#[derive(Debug, Clone)]
pub struct SingleCellTrajectory<S = Untrained> {
    n_genes: usize,
    n_components: usize,
    trajectory_method: TrajectoryMethod,
    state: S,
}

#[derive(Debug, Clone)]
pub enum TrajectoryMethod {
    /// Diffusion pseudotime
    DiffusionPseudotime,
    /// Principal curve fitting
    PrincipalCurve,
    /// Minimum spanning tree
    MST,
}

#[derive(Debug, Clone)]
pub struct TrainedSingleCellTrajectory {
    n_genes: usize,
    n_components: usize,
    trajectory_method: TrajectoryMethod,
    trajectory_embedding: Array2<f64>, // Cell embeddings
    pseudotime: Array1<f64>,           // Pseudotime ordering
    gene_mean: Array1<f64>,            // Mean gene expression
}

impl SingleCellTrajectory<Untrained> {
    /// Create a new single-cell trajectory analysis model
    ///
    /// # Arguments
    /// * `n_genes` - Number of genes
    pub fn new(n_genes: usize) -> Self {
        Self {
            n_genes,
            n_components: 50,
            trajectory_method: TrajectoryMethod::DiffusionPseudotime,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_trajectory_method(mut self, method: TrajectoryMethod) -> Self {
        self.trajectory_method = method;
        self
    }

    fn compute_pseudotime(&self, embedding: &Array2<f64>) -> Array1<f64> {
        let n_cells = embedding.nrows();

        match &self.trajectory_method {
            TrajectoryMethod::DiffusionPseudotime => {
                // Start from first cell (could be user-specified root)
                let mut pseudotime = Array1::zeros(n_cells);
                let root = embedding.row(0);

                for (i, cell) in embedding.axis_iter(Axis(0)).enumerate() {
                    let diff = &cell - &root;
                    pseudotime[i] = diff.mapv(|x| x * x).sum().sqrt();
                }

                // Normalize to [0, 1]
                let max_time = pseudotime.iter().cloned().fold(0.0f64, f64::max);
                if max_time > 0.0 {
                    pseudotime = pseudotime.mapv(|x| x / max_time);
                }

                pseudotime
            }
            TrajectoryMethod::PrincipalCurve => {
                // Simplified: use first principal component as pseudotime
                let first_comp = embedding.column(0);
                let min_val = first_comp.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = first_comp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                let range = max_val - min_val;
                if range > 0.0 {
                    first_comp.mapv(|x| (x - min_val) / range)
                } else {
                    Array1::zeros(n_cells)
                }
            }
            TrajectoryMethod::MST => {
                // Simplified MST-based pseudotime
                let mut pseudotime = Array1::zeros(n_cells);
                let mut visited = vec![false; n_cells];
                visited[0] = true;

                for step in 1..n_cells {
                    let mut min_dist = f64::INFINITY;
                    let mut next_cell = 0;

                    for i in 0..n_cells {
                        if visited[i] {
                            for j in 0..n_cells {
                                if !visited[j] {
                                    let diff = embedding.row(i).to_owned() - embedding.row(j);
                                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                                    if dist < min_dist {
                                        min_dist = dist;
                                        next_cell = j;
                                    }
                                }
                            }
                        }
                    }

                    visited[next_cell] = true;
                    pseudotime[next_cell] = step as f64 / (n_cells - 1) as f64;
                }

                pseudotime
            }
        }
    }
}

impl SingleCellTrajectory<TrainedSingleCellTrajectory> {
    /// Get pseudotime for a cell
    pub fn cell_pseudotime(&self, cell_idx: usize) -> SklResult<f64> {
        if cell_idx >= self.state.pseudotime.len() {
            return Err(SklearsError::InvalidInput(
                "Cell index out of range".to_string(),
            ));
        }

        Ok(self.state.pseudotime[cell_idx])
    }

    /// Identify genes varying along trajectory
    pub fn trajectory_genes(&self, correlation_threshold: f64) -> Vec<usize> {
        let mut varying_genes = Vec::new();

        for gene_idx in 0..self.state.n_genes {
            // Simplified: genes with high variation along trajectory
            let gene_importance = if gene_idx < self.state.gene_mean.len() {
                self.state.gene_mean[gene_idx].abs()
            } else {
                0.0
            };

            if gene_importance > correlation_threshold {
                varying_genes.push(gene_idx);
            }
        }

        varying_genes
    }

    /// Order cells by pseudotime
    pub fn ordered_cells(&self) -> Vec<usize> {
        let mut cell_order: Vec<_> = (0..self.state.pseudotime.len())
            .map(|i| (i, self.state.pseudotime[i]))
            .collect();

        cell_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        cell_order.into_iter().map(|(idx, _)| idx).collect()
    }
}

impl Fit<Array2<f64>, ()> for SingleCellTrajectory<Untrained> {
    type Fitted = SingleCellTrajectory<TrainedSingleCellTrajectory>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.ncols() != self.n_genes {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} genes, got {}",
                self.n_genes,
                x.ncols()
            )));
        }

        let gene_mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &gene_mean.clone().insert_axis(Axis(0));

        // Compute trajectory embedding using PCA
        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        let trajectory_embedding = centered.dot(&projection);
        let pseudotime = self.compute_pseudotime(&trajectory_embedding);

        Ok(SingleCellTrajectory {
            n_genes: self.n_genes,
            n_components: self.n_components,
            trajectory_method: self.trajectory_method.clone(),
            state: TrainedSingleCellTrajectory {
                n_genes: self.n_genes,
                n_components: self.n_components,
                trajectory_method: self.trajectory_method.clone(),
                trajectory_embedding,
                pseudotime,
                gene_mean,
            },
        })
    }
}

impl Transform<usize, f64> for SingleCellTrajectory<TrainedSingleCellTrajectory> {
    fn transform(&self, cell_idx: &usize) -> SklResult<f64> {
        self.cell_pseudotime(*cell_idx)
    }
}

/// Metabolic Pathway Manifolds
///
/// Learns manifold representations of metabolic pathways for pathway analysis,
/// drug target identification, and metabolic flux prediction.
#[derive(Debug, Clone)]
pub struct MetabolicPathwayManifold<S = Untrained> {
    n_metabolites: usize,
    n_components: usize,
    pathway_representation: PathwayRepresentation,
    state: S,
}

#[derive(Debug, Clone)]
pub enum PathwayRepresentation {
    /// Flux balance analysis
    FluxBalance,
    /// Reaction stoichiometry
    Stoichiometry,
    /// Network topology
    NetworkTopology,
}

#[derive(Debug, Clone)]
pub struct TrainedMetabolicPathway {
    n_metabolites: usize,
    n_components: usize,
    pathway_representation: PathwayRepresentation,
    pathway_manifold: Array2<f64>,  // Pathway manifold embedding
    projection_matrix: Array2<f64>, // Projection matrix
    mean_flux: Array1<f64>,         // Mean metabolic flux
}

impl MetabolicPathwayManifold<Untrained> {
    /// Create a new metabolic pathway manifold model
    ///
    /// # Arguments
    /// * `n_metabolites` - Number of metabolites in the network
    pub fn new(n_metabolites: usize) -> Self {
        Self {
            n_metabolites,
            n_components: 30,
            pathway_representation: PathwayRepresentation::NetworkTopology,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_pathway_representation(mut self, representation: PathwayRepresentation) -> Self {
        self.pathway_representation = representation;
        self
    }
}

impl MetabolicPathwayManifold<TrainedMetabolicPathway> {
    /// Project metabolic state onto learned manifold
    pub fn project_metabolic_state(&self, state: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if state.len() != self.state.n_metabolites {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} metabolites, got {}",
                self.state.n_metabolites,
                state.len()
            )));
        }

        let normalized = state - &self.state.mean_flux;
        Ok(self.state.projection_matrix.t().dot(&normalized))
    }

    /// Identify key metabolic pathways
    pub fn key_pathways(&self, n_pathways: usize) -> Vec<(usize, f64)> {
        let mut pathway_importance: Vec<_> = (0..self.state.n_metabolites)
            .map(|i| {
                let importance = if i < self.state.projection_matrix.nrows() {
                    self.state.projection_matrix.row(i).mapv(|x| x.abs()).sum()
                } else {
                    0.0
                };
                (i, importance)
            })
            .collect();

        pathway_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pathway_importance.truncate(n_pathways);

        pathway_importance
    }

    /// Compute pathway similarity
    pub fn pathway_similarity(
        &self,
        state1: &ArrayView1<f64>,
        state2: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        let proj1 = self.project_metabolic_state(state1)?;
        let proj2 = self.project_metabolic_state(state2)?;

        let diff = &proj1 - &proj2;
        let dist = diff.mapv(|x| x * x).sum().sqrt();

        // Convert to similarity
        Ok((-dist / 10.0).exp())
    }
}

impl Fit<Array2<f64>, ()> for MetabolicPathwayManifold<Untrained> {
    type Fitted = MetabolicPathwayManifold<TrainedMetabolicPathway>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.ncols() != self.n_metabolites {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} metabolites, got {}",
                self.n_metabolites,
                x.ncols()
            )));
        }

        let mean_flux = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean_flux.clone().insert_axis(Axis(0));

        // Compute pathway embedding using PCA
        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        let pathway_manifold = centered.dot(&projection_matrix);

        Ok(MetabolicPathwayManifold {
            n_metabolites: self.n_metabolites,
            n_components: self.n_components,
            pathway_representation: self.pathway_representation.clone(),
            state: TrainedMetabolicPathway {
                n_metabolites: self.n_metabolites,
                n_components: self.n_components,
                pathway_representation: self.pathway_representation.clone(),
                pathway_manifold,
                projection_matrix,
                mean_flux,
            },
        })
    }
}

impl Transform<Array1<f64>, Array1<f64>> for MetabolicPathwayManifold<TrainedMetabolicPathway> {
    fn transform(&self, x: &Array1<f64>) -> SklResult<Array1<f64>> {
        self.project_metabolic_state(&x.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genomic_manifold_basic() {
        let n_samples = 20;
        let n_genes = 100;

        let genomic_data = Array2::from_shape_fn((n_samples, n_genes), |(i, j)| {
            (i as f64 * 0.5 + j as f64 * 0.1) % 10.0
        });

        let genomic_model = GenomicManifoldAnalysis::new(n_genes)
            .with_n_components(10)
            .with_normalization(GenomicNormalization::ZScore);

        let fitted = genomic_model.fit(&genomic_data, &()).unwrap();
        let projected = fitted.project_genomic_data(&genomic_data.view()).unwrap();

        assert_eq!(projected.nrows(), n_samples);
        assert_eq!(projected.ncols(), 10);
    }

    #[test]
    fn test_protein_structure_manifold() {
        let n_proteins = 10;
        let n_residues = 50;

        let mut structures = Vec::new();
        for i in 0..n_proteins {
            let structure = Array2::from_shape_fn((n_residues, n_residues), |(row, col)| {
                if row == col {
                    0.0
                } else {
                    (i as f64 + row as f64 + col as f64) % 5.0
                }
            });
            structures.push(structure);
        }

        let protein_model = ProteinStructureManifold::new(n_residues)
            .with_n_components(5)
            .with_representation(ProteinRepresentation::ContactMap);

        let fitted = protein_model.fit(&structures, &()).unwrap();

        let test_structure = &structures[0];
        let projected = fitted.project_structure(&test_structure.view()).unwrap();
        assert_eq!(projected.len(), 5);
    }

    #[test]
    fn test_phylogenetic_embedding() {
        let n_taxa = 8;

        // Create synthetic distance matrix
        let mut dist_matrix = Array2::zeros((n_taxa, n_taxa));
        for i in 0..n_taxa {
            for j in i + 1..n_taxa {
                let dist = ((i as f64 - j as f64).abs() / n_taxa as f64).min(0.5);
                dist_matrix[[i, j]] = dist;
                dist_matrix[[j, i]] = dist;
            }
        }

        let phylo_model = PhylogeneticEmbedding::new(n_taxa)
            .with_n_components(3)
            .with_distance_method(PhylogeneticDistance::JukesCantor);

        let fitted = phylo_model.fit(&dist_matrix, &()).unwrap();

        let embedding = fitted.taxon_embedding(0).unwrap();
        assert_eq!(embedding.len(), 3);
    }

    #[test]
    fn test_single_cell_trajectory() {
        let n_cells = 50;
        let n_genes = 200;

        let cell_data = Array2::from_shape_fn((n_cells, n_genes), |(i, j)| {
            i as f64 * 0.1 + j as f64 * 0.05
        });

        let trajectory_model = SingleCellTrajectory::new(n_genes)
            .with_n_components(20)
            .with_trajectory_method(TrajectoryMethod::DiffusionPseudotime);

        let fitted = trajectory_model.fit(&cell_data, &()).unwrap();

        let pseudotime = fitted.cell_pseudotime(0).unwrap();
        assert!(pseudotime >= 0.0 && pseudotime <= 1.0);
    }

    #[test]
    fn test_metabolic_pathway_manifold() {
        let n_samples = 30;
        let n_metabolites = 150;

        let metabolic_data = Array2::from_shape_fn((n_samples, n_metabolites), |(i, j)| {
            (i as f64 + j as f64 * 0.3) % 8.0
        });

        let pathway_model = MetabolicPathwayManifold::new(n_metabolites)
            .with_n_components(15)
            .with_pathway_representation(PathwayRepresentation::NetworkTopology);

        let fitted = pathway_model.fit(&metabolic_data, &()).unwrap();

        let test_state = metabolic_data.row(0);
        let projected = fitted.project_metabolic_state(&test_state).unwrap();
        assert_eq!(projected.len(), 15);
    }

    #[test]
    fn test_genomic_differential_genes() {
        let genomic_data =
            Array2::from_shape_fn((15, 80), |(i, j)| i as f64 * 2.0 + j as f64 * 0.2);

        let genomic_model = GenomicManifoldAnalysis::new(80).with_n_components(8);
        let fitted = genomic_model.fit(&genomic_data, &()).unwrap();

        let diff_genes = fitted.differential_genes(0.5);
        assert!(diff_genes.len() > 0);
    }

    #[test]
    fn test_phylogenetic_nearest_neighbors() {
        let n_taxa = 10;
        let dist_matrix = Array2::from_shape_fn((n_taxa, n_taxa), |(i, j)| {
            if i == j {
                0.0
            } else {
                (i as f64 - j as f64).abs() / 10.0
            }
        });

        let phylo_model = PhylogeneticEmbedding::new(n_taxa).with_n_components(4);
        let fitted = phylo_model.fit(&dist_matrix, &()).unwrap();

        let neighbors = fitted.nearest_neighbors(0, 3).unwrap();
        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_single_cell_ordered_cells() {
        let cell_data = Array2::from_shape_fn((25, 100), |(i, j)| i as f64 + j as f64 * 0.1);

        let trajectory_model = SingleCellTrajectory::new(100).with_n_components(10);
        let fitted = trajectory_model.fit(&cell_data, &()).unwrap();

        let ordered = fitted.ordered_cells();
        assert_eq!(ordered.len(), 25);
    }

    #[test]
    fn test_metabolic_key_pathways() {
        let metabolic_data =
            Array2::from_shape_fn((20, 120), |(i, j)| i as f64 * 1.5 + j as f64 * 0.25);

        let pathway_model = MetabolicPathwayManifold::new(120).with_n_components(12);
        let fitted = pathway_model.fit(&metabolic_data, &()).unwrap();

        let key_pathways = fitted.key_pathways(5);
        assert_eq!(key_pathways.len(), 5);
    }

    #[test]
    fn test_protein_structural_similarity() {
        let n_residues = 30;
        let struct1 = Array2::from_shape_fn((n_residues, n_residues), |(i, j)| {
            if i == j {
                0.0
            } else {
                (i + j) as f64
            }
        });
        let struct2 = Array2::from_shape_fn((n_residues, n_residues), |(i, j)| {
            if i == j {
                0.0
            } else {
                (i + j) as f64 + 1.0
            }
        });

        let structures = vec![struct1.clone(), struct2.clone()];

        let protein_model = ProteinStructureManifold::new(n_residues).with_n_components(4);
        let fitted = protein_model.fit(&structures, &()).unwrap();

        let similarity = fitted
            .structural_similarity(&struct1.view(), &struct2.view())
            .unwrap();
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }
}
