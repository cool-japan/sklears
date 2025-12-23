//! Bioinformatics-specific imputation methods
//!
//! This module provides specialized imputation methods for biological data types
//! including single-cell RNA sequencing, genomics, proteomics, and metabolomics.

use crate::core::{ImputationError, ImputationResult};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;

/// Single-cell RNA sequencing imputation using zero-inflated models
///
/// Single-cell RNA-seq data has unique characteristics:
/// - High sparsity (many zeros)
/// - Zero-inflation (technical and biological zeros)
/// - Cell-specific and gene-specific dropout patterns
/// - Need to preserve biological heterogeneity
#[derive(Debug, Clone)]
pub struct SingleCellRNASeqImputer {
    /// Minimum expression threshold to distinguish zeros from dropouts
    pub min_expression_threshold: f64,
    /// Zero-inflation parameter (probability of technical zero)
    pub zero_inflation_rate: f64,
    /// Method for imputation: "magic", "scimpute", "saver", "dca"
    pub method: String,
    /// Number of principal components for dimensionality reduction
    pub n_components: usize,
    /// K-nearest neighbors for cell similarity
    pub n_neighbors: usize,
    /// Use batch correction
    pub batch_correction: bool,
    /// Cell type labels for supervised imputation
    pub cell_types: Option<Vec<String>>,
}

impl Default for SingleCellRNASeqImputer {
    fn default() -> Self {
        Self {
            min_expression_threshold: 0.1,
            zero_inflation_rate: 0.7,
            method: "magic".to_string(),
            n_components: 50,
            n_neighbors: 15,
            batch_correction: false,
            cell_types: None,
        }
    }
}

impl SingleCellRNASeqImputer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    pub fn with_batch_correction(mut self, batch_correction: bool) -> Self {
        self.batch_correction = batch_correction;
        self
    }

    /// Impute single-cell RNA-seq expression data
    ///
    /// Uses specialized methods for handling zero-inflation and preserving
    /// biological structure in single-cell data.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        match self.method.as_str() {
            "magic" => self.magic_imputation(X),
            "scimpute" => self.scimpute_imputation(X),
            "saver" => self.saver_imputation(X),
            "dca" => self.dca_imputation(X),
            _ => Err(ImputationError::InvalidParameter(format!(
                "Unknown method: {}",
                self.method
            ))),
        }
    }

    /// MAGIC (Markov Affinity-based Graph Imputation of Cells) imputation
    fn magic_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_cells, n_genes) = X.dim();
        let mut imputed = X.to_owned();

        // Build cell-cell affinity matrix
        let affinity_matrix = self.build_cell_affinity_matrix(X)?;

        // Power iteration for diffusion
        let mut diffused_data = imputed.clone();
        for _ in 0..3 {
            // Power iterations
            diffused_data = affinity_matrix.dot(&diffused_data);
        }

        // Preserve zeros that are likely biological (high expression genes)
        for i in 0..n_cells {
            for j in 0..n_genes {
                let original_value = X[[i, j]];
                if original_value == 0.0 {
                    // Check if this is likely a technical dropout
                    let gene_mean = X.column(j).mean().unwrap_or(0.0);
                    if gene_mean > self.min_expression_threshold {
                        imputed[[i, j]] = diffused_data[[i, j]];
                    }
                }
            }
        }

        Ok(imputed)
    }

    /// scImpute imputation using dropouts identification
    fn scimpute_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_cells, n_genes) = X.dim();
        let mut imputed = X.to_owned();

        // Identify likely dropouts vs true zeros
        let dropout_indicators = self.identify_dropouts(X)?;

        // For each cell, impute dropouts using similar cells
        for i in 0..n_cells {
            let cell_vector = X.row(i);
            let similar_cells = self.find_similar_cells(&cell_vector, X, self.n_neighbors)?;

            for j in 0..n_genes {
                if dropout_indicators[[i, j]] {
                    // Impute using similar cells
                    let mut values = Vec::new();
                    for &similar_cell in &similar_cells {
                        let val = X[[similar_cell, j]];
                        if val > 0.0 {
                            values.push(val);
                        }
                    }

                    if !values.is_empty() {
                        imputed[[i, j]] = values.iter().sum::<f64>() / values.len() as f64;
                    }
                }
            }
        }

        Ok(imputed)
    }

    /// SAVER (Single-cell Analysis Via Expression Recovery) imputation
    fn saver_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_cells, n_genes) = X.dim();
        let mut imputed = X.to_owned();

        // Fit Poisson-Gamma model for each gene
        for j in 0..n_genes {
            let gene_expression = X.column(j);
            let (alpha, beta) = self.fit_poisson_gamma_model(&gene_expression)?;

            // Posterior prediction for each cell
            for i in 0..n_cells {
                if X[[i, j]] == 0.0 {
                    // Posterior mean under Poisson-Gamma model
                    let posterior_mean = (alpha + X[[i, j]]) / (beta + 1.0);
                    imputed[[i, j]] = posterior_mean;
                }
            }
        }

        Ok(imputed)
    }

    /// Deep Count Autoencoder (DCA) imputation
    fn dca_imputation(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        // Simplified DCA - in practice would use a neural network
        let (n_cells, n_genes) = X.dim();
        let mut imputed = X.to_owned();

        // Library size normalization
        for i in 0..n_cells {
            let library_size: f64 = X.row(i).sum();
            if library_size > 0.0 {
                let median_library_size = self.compute_median_library_size(X);
                let scale_factor = median_library_size / library_size;

                for j in 0..n_genes {
                    if imputed[[i, j]] > 0.0 {
                        imputed[[i, j]] *= scale_factor;
                    }
                }
            }
        }

        // Zero-inflated negative binomial smoothing (simplified)
        for j in 0..n_genes {
            let gene_values: Vec<f64> = X.column(j).iter().filter(|&&x| x > 0.0).cloned().collect();
            if !gene_values.is_empty() {
                let gene_mean = gene_values.iter().sum::<f64>() / gene_values.len() as f64;
                let _gene_var = gene_values
                    .iter()
                    .map(|&x| (x - gene_mean).powi(2))
                    .sum::<f64>()
                    / gene_values.len() as f64;

                for i in 0..n_cells {
                    if X[[i, j]] == 0.0 {
                        // Impute based on zero-inflated model
                        let prob_non_zero = 1.0 - self.zero_inflation_rate;
                        if Random::default().gen::<f64>() < prob_non_zero {
                            imputed[[i, j]] = Random::default().gen_range(0.0..gene_mean * 0.1);
                        }
                    }
                }
            }
        }

        Ok(imputed)
    }

    /// Build cell-cell affinity matrix for MAGIC
    fn build_cell_affinity_matrix(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let n_cells = X.nrows();
        let mut affinity = Array2::zeros((n_cells, n_cells));

        for i in 0..n_cells {
            let distances = self.compute_cell_distances(X.row(i), X)?;
            let mut indices: Vec<usize> = (0..n_cells).collect();
            indices.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());

            // Keep only k nearest neighbors
            for &idx in indices.iter().take(self.n_neighbors) {
                if idx != i {
                    affinity[[i, idx]] = (-distances[idx] / (2.0 * 1.0)).exp(); // sigma = 1.0
                }
            }
        }

        // Row normalize
        for i in 0..n_cells {
            let row_sum: f64 = affinity.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_cells {
                    affinity[[i, j]] /= row_sum;
                }
            }
        }

        Ok(affinity)
    }

    /// Compute distances between cells
    fn compute_cell_distances(
        &self,
        cell: ArrayView1<f64>,
        X: &ArrayView2<f64>,
    ) -> ImputationResult<Vec<f64>> {
        let n_cells = X.nrows();
        let mut distances = Vec::with_capacity(n_cells);

        for i in 0..n_cells {
            let other_cell = X.row(i);
            let distance = cell
                .iter()
                .zip(other_cell.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            distances.push(distance);
        }

        Ok(distances)
    }

    /// Identify likely dropouts vs biological zeros
    fn identify_dropouts(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<bool>> {
        let (n_cells, n_genes) = X.dim();
        let mut dropouts = Array2::from_elem((n_cells, n_genes), false);

        for j in 0..n_genes {
            let gene_values: Vec<f64> = X.column(j).iter().cloned().collect();
            let non_zero_values: Vec<f64> =
                gene_values.iter().filter(|&&x| x > 0.0).cloned().collect();

            if non_zero_values.len() > 5 {
                // Need enough non-zero values
                let mean_expr = non_zero_values.iter().sum::<f64>() / non_zero_values.len() as f64;
                let detection_rate = non_zero_values.len() as f64 / n_cells as f64;

                // High expression, high detection rate genes are more likely to have dropouts
                if mean_expr > self.min_expression_threshold && detection_rate > 0.1 {
                    for i in 0..n_cells {
                        if X[[i, j]] == 0.0 {
                            dropouts[[i, j]] = true;
                        }
                    }
                }
            }
        }

        Ok(dropouts)
    }

    /// Find similar cells based on expression profiles
    fn find_similar_cells(
        &self,
        cell: &ArrayView1<f64>,
        X: &ArrayView2<f64>,
        k: usize,
    ) -> ImputationResult<Vec<usize>> {
        let n_cells = X.nrows();
        let mut similarities = Vec::new();

        for i in 0..n_cells {
            let other_cell = X.row(i);
            let correlation = self.compute_correlation(cell, &other_cell);
            similarities.push((i, correlation));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(similarities.iter().take(k).map(|(idx, _)| *idx).collect())
    }

    /// Compute Pearson correlation between two cells
    fn compute_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let n = x.len();
        if n != y.len() {
            return 0.0;
        }

        let mean_x = x.sum() / n as f64;
        let mean_y = y.sum() / n as f64;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Fit Poisson-Gamma model parameters
    fn fit_poisson_gamma_model(&self, data: &ArrayView1<f64>) -> ImputationResult<(f64, f64)> {
        let non_zero_data: Vec<f64> = data.iter().filter(|&&x| x > 0.0).cloned().collect();

        if non_zero_data.is_empty() {
            return Ok((1.0, 1.0)); // Default parameters
        }

        let mean = non_zero_data.iter().sum::<f64>() / non_zero_data.len() as f64;
        let variance = non_zero_data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / non_zero_data.len() as f64;

        // Method of moments estimation for Gamma distribution
        let alpha = mean.powi(2) / variance.max(0.1); // Avoid division by zero
        let beta = mean / variance.max(0.1);

        Ok((alpha, beta))
    }

    /// Compute median library size across all cells
    fn compute_median_library_size(&self, X: &ArrayView2<f64>) -> f64 {
        let mut library_sizes: Vec<f64> = Vec::new();

        for i in 0..X.nrows() {
            let size = X.row(i).sum();
            library_sizes.push(size);
        }

        library_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = library_sizes.len();
        if n % 2 == 0 {
            (library_sizes[n / 2 - 1] + library_sizes[n / 2]) / 2.0
        } else {
            library_sizes[n / 2]
        }
    }
}

/// Genomic data imputation for SNP and CNV data
#[derive(Debug, Clone)]
pub struct GenomicImputer {
    /// Minimum allele frequency threshold
    pub min_allele_freq: f64,
    /// Linkage disequilibrium window size
    pub ld_window: usize,
    /// Population structure principal components
    pub n_population_pcs: usize,
    /// Reference panel for imputation
    pub use_reference_panel: bool,
    /// Hardy-Weinberg equilibrium p-value threshold
    pub hwe_pvalue_threshold: f64,
}

impl Default for GenomicImputer {
    fn default() -> Self {
        Self {
            min_allele_freq: 0.01,
            ld_window: 1000,
            n_population_pcs: 10,
            use_reference_panel: false,
            hwe_pvalue_threshold: 1e-6,
        }
    }
}

impl GenomicImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute genomic variants using linkage disequilibrium patterns
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_samples, n_variants) = X.dim();
        let mut imputed = X.to_owned();

        // Calculate allele frequencies
        let allele_freqs = self.calculate_allele_frequencies(X)?;

        // Quality control: filter variants
        let valid_variants = self.quality_control_variants(X, &allele_freqs)?;

        // Impute missing variants using LD patterns
        for variant_idx in 0..n_variants {
            if !valid_variants[variant_idx] {
                continue;
            }

            for sample_idx in 0..n_samples {
                if X[[sample_idx, variant_idx]].is_nan() {
                    let imputed_value =
                        self.impute_variant(sample_idx, variant_idx, X, &allele_freqs)?;
                    imputed[[sample_idx, variant_idx]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// Calculate allele frequencies for each variant
    fn calculate_allele_frequencies(&self, X: &ArrayView2<f64>) -> ImputationResult<Vec<f64>> {
        let n_variants = X.ncols();
        let mut freqs = Vec::with_capacity(n_variants);

        for j in 0..n_variants {
            let variant_data = X.column(j);
            let valid_genotypes: Vec<f64> = variant_data
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if valid_genotypes.is_empty() {
                freqs.push(0.0);
            } else {
                let allele_count: f64 = valid_genotypes.iter().sum();
                let total_alleles = valid_genotypes.len() as f64 * 2.0; // Diploid
                freqs.push(allele_count / total_alleles);
            }
        }

        Ok(freqs)
    }

    /// Quality control filtering for variants
    fn quality_control_variants(
        &self,
        X: &ArrayView2<f64>,
        allele_freqs: &[f64],
    ) -> ImputationResult<Vec<bool>> {
        let mut valid = Vec::with_capacity(allele_freqs.len());

        for (j, &freq) in allele_freqs.iter().enumerate() {
            let is_valid = freq >= self.min_allele_freq
                && freq <= (1.0 - self.min_allele_freq)
                && self.test_hardy_weinberg_equilibrium(X.column(j), freq);
            valid.push(is_valid);
        }

        Ok(valid)
    }

    /// Test Hardy-Weinberg equilibrium
    fn test_hardy_weinberg_equilibrium(
        &self,
        genotypes: ArrayView1<f64>,
        allele_freq: f64,
    ) -> bool {
        let valid_genotypes: Vec<f64> = genotypes
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .collect();

        if valid_genotypes.len() < 10 {
            return false; // Need sufficient sample size
        }

        let n = valid_genotypes.len() as f64;
        let p = allele_freq;
        let q = 1.0 - p;

        // Expected genotype frequencies under HWE
        let exp_aa = n * p * p;
        let exp_ab = n * 2.0 * p * q;
        let exp_bb = n * q * q;

        // Observed genotype counts
        let mut obs_aa = 0.0;
        let mut obs_ab = 0.0;
        let mut obs_bb = 0.0;

        for &genotype in &valid_genotypes {
            match genotype as i32 {
                0 => obs_bb += 1.0,
                1 => obs_ab += 1.0,
                2 => obs_aa += 1.0,
                _ => {} // Invalid genotype
            }
        }

        // Chi-square test statistic
        let chi_square = (obs_aa - exp_aa).powi(2) / exp_aa
            + (obs_ab - exp_ab).powi(2) / exp_ab
            + (obs_bb - exp_bb).powi(2) / exp_bb;

        // Approximate p-value (df = 1)
        let p_value = 1.0 - self.chi_square_cdf(chi_square, 1);

        p_value > self.hwe_pvalue_threshold
    }

    /// Approximate chi-square CDF for df=1
    fn chi_square_cdf(&self, x: f64, _df: i32) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // Simple approximation for df=1
        2.0 * (1.0 - (-x / 2.0).exp().sqrt())
    }

    /// Impute a single variant using LD patterns
    fn impute_variant(
        &self,
        sample_idx: usize,
        variant_idx: usize,
        X: &ArrayView2<f64>,
        allele_freqs: &[f64],
    ) -> ImputationResult<f64> {
        let n_variants = X.ncols();

        // Find variants in LD window
        let window_start = variant_idx.saturating_sub(self.ld_window / 2);
        let window_end = (variant_idx + self.ld_window / 2).min(n_variants);

        let mut ld_scores = Vec::new();
        for j in window_start..window_end {
            if j != variant_idx {
                let ld_r2 = self.calculate_ld_r_squared(X.column(variant_idx), X.column(j));
                if ld_r2 > 0.1 && !X[[sample_idx, j]].is_nan() {
                    ld_scores.push((j, ld_r2, X[[sample_idx, j]]));
                }
            }
        }

        if ld_scores.is_empty() {
            // Fall back to population frequency
            return Ok(allele_freqs[variant_idx] * 2.0);
        }

        // Weighted average based on LD
        let total_weight: f64 = ld_scores.iter().map(|(_, r2, _)| r2).sum();
        let weighted_sum: f64 = ld_scores
            .iter()
            .map(|(_, r2, genotype)| r2 * genotype)
            .sum();

        Ok(weighted_sum / total_weight)
    }

    /// Calculate linkage disequilibrium r-squared between two variants
    fn calculate_ld_r_squared(&self, var1: ArrayView1<f64>, var2: ArrayView1<f64>) -> f64 {
        let pairs: Vec<(f64, f64)> = var1
            .iter()
            .zip(var2.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();

        if pairs.len() < 10 {
            return 0.0;
        }

        let n = pairs.len() as f64;
        let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = pairs.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

        if denominator > 1e-10 {
            (numerator / denominator).powi(2)
        } else {
            0.0
        }
    }
}

/// Protein expression data imputation
#[derive(Debug, Clone)]
pub struct ProteinExpressionImputer {
    /// Normalization method: "log2", "zscore", "quantile"
    pub normalization: String,
    /// Missing value mechanism detection
    pub detect_missing_mechanism: bool,
    /// Protein interaction network information
    pub use_protein_interactions: bool,
    /// Pathway enrichment for guided imputation
    pub pathway_guided: bool,
}

impl Default for ProteinExpressionImputer {
    fn default() -> Self {
        Self {
            normalization: "log2".to_string(),
            detect_missing_mechanism: true,
            use_protein_interactions: false,
            pathway_guided: false,
        }
    }
}

impl ProteinExpressionImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute protein expression data with biological constraints
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let mut data = X.to_owned();

        // Apply normalization
        data = self.normalize_data(&data)?;

        // Detect missing value patterns
        if self.detect_missing_mechanism {
            let _missing_pattern = self.analyze_missing_pattern(&data)?;
            // Use pattern information to guide imputation strategy
        }

        // Perform imputation
        let imputed = self.impute_with_biological_constraints(&data)?;

        Ok(imputed)
    }

    /// Normalize protein expression data
    fn normalize_data(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut normalized = X.clone();

        match self.normalization.as_str() {
            "log2" => {
                // Log2 transformation with pseudocount
                for value in normalized.iter_mut() {
                    if *value > 0.0 {
                        *value = (*value + 1.0).log2();
                    }
                }
            }
            "zscore" => {
                // Z-score normalization per protein
                for j in 0..X.ncols() {
                    let column = X.column(j);
                    let valid_values: Vec<f64> = column
                        .iter()
                        .filter(|&&x| !x.is_nan() && x.is_finite())
                        .cloned()
                        .collect();

                    if !valid_values.is_empty() {
                        let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                        let variance = valid_values
                            .iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f64>()
                            / valid_values.len() as f64;
                        let std_dev = variance.sqrt();

                        if std_dev > 1e-10 {
                            for i in 0..X.nrows() {
                                if !X[[i, j]].is_nan() {
                                    normalized[[i, j]] = (X[[i, j]] - mean) / std_dev;
                                }
                            }
                        }
                    }
                }
            }
            "quantile" => {
                // Quantile normalization
                normalized = self.quantile_normalize(&normalized)?;
            }
            _ => {
                return Err(ImputationError::InvalidParameter(format!(
                    "Unknown normalization method: {}",
                    self.normalization
                )));
            }
        }

        Ok(normalized)
    }

    /// Perform quantile normalization
    fn quantile_normalize(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_samples, n_proteins) = X.dim();
        let mut normalized = X.clone();

        // Collect all non-missing values
        let mut all_values = Vec::new();
        for i in 0..n_samples {
            for j in 0..n_proteins {
                let val = X[[i, j]];
                if !val.is_nan() && val.is_finite() {
                    all_values.push(val);
                }
            }
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Create quantile mapping
        for i in 0..n_samples {
            for j in 0..n_proteins {
                if !X[[i, j]].is_nan() {
                    let value = X[[i, j]];
                    let rank = all_values
                        .iter()
                        .position(|&x| x >= value)
                        .unwrap_or(all_values.len() - 1);
                    let quantile = rank as f64 / all_values.len() as f64;
                    normalized[[i, j]] = quantile;
                }
            }
        }

        Ok(normalized)
    }

    /// Analyze missing value patterns in protein data
    fn analyze_missing_pattern(&self, X: &Array2<f64>) -> ImputationResult<HashMap<String, f64>> {
        let (n_samples, n_proteins) = X.dim();
        let mut pattern_stats = HashMap::new();

        // Calculate missingness rates
        let mut missing_counts = Vec::new();
        for j in 0..n_proteins {
            let missing_count = X.column(j).iter().filter(|&&x| x.is_nan()).count();
            missing_counts.push(missing_count as f64 / n_samples as f64);
        }

        pattern_stats.insert(
            "mean_missing_rate".to_string(),
            missing_counts.iter().sum::<f64>() / n_proteins as f64,
        );

        pattern_stats.insert(
            "max_missing_rate".to_string(),
            missing_counts.iter().fold(0.0, |a, &b| a.max(b)),
        );

        // Analyze missing pattern correlations
        let mut missing_correlations = Vec::new();
        for i in 0..n_proteins {
            for j in (i + 1)..n_proteins {
                let correlation = self.compute_missing_correlation(&X.column(i), &X.column(j));
                missing_correlations.push(correlation.abs());
            }
        }

        if !missing_correlations.is_empty() {
            pattern_stats.insert(
                "mean_missing_correlation".to_string(),
                missing_correlations.iter().sum::<f64>() / missing_correlations.len() as f64,
            );
        }

        Ok(pattern_stats)
    }

    /// Compute correlation between missing patterns of two proteins
    fn compute_missing_correlation(&self, col1: &ArrayView1<f64>, col2: &ArrayView1<f64>) -> f64 {
        let missing1: Vec<f64> = col1
            .iter()
            .map(|&x| if x.is_nan() { 1.0 } else { 0.0 })
            .collect();
        let missing2: Vec<f64> = col2
            .iter()
            .map(|&x| if x.is_nan() { 1.0 } else { 0.0 })
            .collect();

        let n = missing1.len() as f64;
        let mean1 = missing1.iter().sum::<f64>() / n;
        let mean2 = missing2.iter().sum::<f64>() / n;

        let numerator: f64 = missing1
            .iter()
            .zip(missing2.iter())
            .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
            .sum();

        let var1: f64 = missing1.iter().map(|&x| (x - mean1).powi(2)).sum();
        let var2: f64 = missing2.iter().map(|&x| (x - mean2).powi(2)).sum();

        let denominator = (var1 * var2).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Impute with biological constraints
    fn impute_with_biological_constraints(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_samples, n_proteins) = X.dim();
        let mut imputed = X.clone();

        // For each missing value, use biological similarity for imputation
        for i in 0..n_samples {
            for j in 0..n_proteins {
                if X[[i, j]].is_nan() {
                    let imputed_value = self.impute_single_value(i, j, X)?;
                    imputed[[i, j]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// Impute a single missing value using protein similarity
    fn impute_single_value(
        &self,
        sample_idx: usize,
        protein_idx: usize,
        X: &Array2<f64>,
    ) -> ImputationResult<f64> {
        let n_proteins = X.ncols();

        // Find similar proteins based on expression correlation
        let mut protein_similarities = Vec::new();
        let target_protein = X.column(protein_idx);

        for j in 0..n_proteins {
            if j != protein_idx {
                let other_protein = X.column(j);
                let correlation = self.compute_protein_correlation(&target_protein, &other_protein);
                if correlation.abs() > 0.3 && !X[[sample_idx, j]].is_nan() {
                    protein_similarities.push((j, correlation.abs(), X[[sample_idx, j]]));
                }
            }
        }

        if protein_similarities.is_empty() {
            // Fall back to protein mean
            let valid_values: Vec<f64> = target_protein
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if !valid_values.is_empty() {
                return Ok(valid_values.iter().sum::<f64>() / valid_values.len() as f64);
            } else {
                return Ok(0.0);
            }
        }

        // Weight by similarity
        protein_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_similarities: Vec<_> = protein_similarities.into_iter().take(5).collect();

        let total_weight: f64 = top_similarities.iter().map(|(_, weight, _)| weight).sum();
        let weighted_sum: f64 = top_similarities
            .iter()
            .map(|(_, weight, value)| weight * value)
            .sum();

        Ok(weighted_sum / total_weight)
    }

    /// Compute correlation between two protein expression profiles
    fn compute_protein_correlation(
        &self,
        protein1: &ArrayView1<f64>,
        protein2: &ArrayView1<f64>,
    ) -> f64 {
        let pairs: Vec<(f64, f64)> = protein1
            .iter()
            .zip(protein2.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();

        if pairs.len() < 3 {
            return 0.0;
        }

        let n = pairs.len() as f64;
        let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = pairs.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

/// Metabolomics data imputation
#[derive(Debug, Clone)]
pub struct MetabolomicsImputer {
    /// Method for handling left-censored data (below detection limit)
    pub censoring_method: String,
    /// Metabolic pathway information for guided imputation
    pub pathway_guided: bool,
    /// Mass spectrometry specific handling
    pub ms_specific: bool,
    /// Detection limit threshold
    pub detection_limit: f64,
}

impl Default for MetabolomicsImputer {
    fn default() -> Self {
        Self {
            censoring_method: "half_min".to_string(),
            pathway_guided: false,
            ms_specific: true,
            detection_limit: 0.0,
        }
    }
}

impl MetabolomicsImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute metabolomics data with specialized handling for detection limits
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let mut imputed = X.to_owned();

        // Handle left-censored data (below detection limit)
        imputed = self.handle_censored_data(&imputed)?;

        // Apply metabolite-specific imputation
        imputed = self.metabolite_specific_imputation(&imputed)?;

        Ok(imputed)
    }

    /// Handle left-censored (below detection limit) metabolomics data
    fn handle_censored_data(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut processed = X.clone();

        match self.censoring_method.as_str() {
            "half_min" => {
                // Replace missing with half of minimum detected value
                for j in 0..X.ncols() {
                    let column = X.column(j);
                    let detected_values: Vec<f64> = column
                        .iter()
                        .filter(|&&x| !x.is_nan() && x > self.detection_limit)
                        .cloned()
                        .collect();

                    if !detected_values.is_empty() {
                        let min_detected =
                            detected_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let replacement_value = min_detected / 2.0;

                        for i in 0..X.nrows() {
                            if X[[i, j]].is_nan() || X[[i, j]] <= self.detection_limit {
                                processed[[i, j]] = replacement_value;
                            }
                        }
                    }
                }
            }
            "random_forest" => {
                // Use random forest for imputation (simplified version)
                processed = self.random_forest_imputation(&processed)?;
            }
            "quantile_regression" => {
                // Use quantile regression for censored data
                processed = self.quantile_regression_imputation(&processed)?;
            }
            _ => {
                return Err(ImputationError::InvalidParameter(format!(
                    "Unknown censoring method: {}",
                    self.censoring_method
                )));
            }
        }

        Ok(processed)
    }

    /// Random forest imputation for metabolomics
    fn random_forest_imputation(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        // Simplified random forest - in practice would use a proper implementation
        let mut imputed = X.clone();
        let (_n_samples, n_metabolites) = X.dim();

        for j in 0..n_metabolites {
            let target_metabolite = X.column(j);
            let missing_indices: Vec<usize> = target_metabolite
                .iter()
                .enumerate()
                .filter(|(_, &val)| val.is_nan())
                .map(|(idx, _)| idx)
                .collect();

            if missing_indices.is_empty() {
                continue;
            }

            // Find predictive metabolites
            let mut predictors = Vec::new();
            for k in 0..n_metabolites {
                if k != j {
                    let correlation =
                        self.compute_metabolite_correlation(&X.column(j), &X.column(k));
                    if correlation.abs() > 0.3 {
                        predictors.push(k);
                    }
                }
            }

            // Impute missing values
            for &missing_idx in &missing_indices {
                let mut predictor_values = Vec::new();
                let mut weights = Vec::new();

                for &pred_idx in &predictors {
                    if !X[[missing_idx, pred_idx]].is_nan() {
                        predictor_values.push(X[[missing_idx, pred_idx]]);
                        weights.push(1.0); // Equal weights for simplicity
                    }
                }

                if !predictor_values.is_empty() {
                    // Find similar samples based on predictor values
                    let similar_samples =
                        self.find_similar_metabolic_profiles(missing_idx, j, &predictors, X, 5)?;

                    if !similar_samples.is_empty() {
                        let mean_value =
                            similar_samples.iter().map(|&idx| X[[idx, j]]).sum::<f64>()
                                / similar_samples.len() as f64;
                        imputed[[missing_idx, j]] = mean_value;
                    }
                }
            }
        }

        Ok(imputed)
    }

    /// Quantile regression imputation for censored data
    fn quantile_regression_imputation(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut imputed = X.clone();
        let (n_samples, n_metabolites) = X.dim();

        for j in 0..n_metabolites {
            let metabolite_values = X.column(j);
            let valid_values: Vec<f64> = metabolite_values
                .iter()
                .filter(|&&x| !x.is_nan() && x > self.detection_limit)
                .cloned()
                .collect();

            if valid_values.is_empty() {
                continue;
            }

            // Estimate lower quantiles for censored imputation
            let mut sorted_values = valid_values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q10 = self.quantile(&sorted_values, 0.10);
            let _q25 = self.quantile(&sorted_values, 0.25);

            for i in 0..n_samples {
                if X[[i, j]].is_nan() || X[[i, j]] <= self.detection_limit {
                    // Impute with values between detection limit and 10th percentile
                    let random_factor = Random::default().gen_range(0.5..1.0);
                    imputed[[i, j]] =
                        self.detection_limit + random_factor * (q10 - self.detection_limit);
                }
            }
        }

        Ok(imputed)
    }

    /// Calculate quantile of sorted values
    fn quantile(&self, sorted_values: &[f64], q: f64) -> f64 {
        let n = sorted_values.len();
        if n == 0 {
            return 0.0;
        }

        let index = (q * (n - 1) as f64) as usize;
        if index >= n {
            sorted_values[n - 1]
        } else {
            sorted_values[index]
        }
    }

    /// Metabolite-specific imputation using biological knowledge
    fn metabolite_specific_imputation(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut imputed = X.clone();

        if self.pathway_guided {
            imputed = self.pathway_guided_imputation(&imputed)?;
        }

        if self.ms_specific {
            imputed = self.mass_spec_specific_processing(&imputed)?;
        }

        Ok(imputed)
    }

    /// Pathway-guided imputation using metabolic networks
    fn pathway_guided_imputation(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        // Simplified pathway-guided imputation
        // In practice, would use pathway databases like KEGG
        let mut imputed = X.clone();
        let (n_samples, n_metabolites) = X.dim();

        // Group metabolites by assumed pathways (simplified)
        let pathway_groups = self.create_pathway_groups(n_metabolites);

        for group in pathway_groups {
            for &metabolite_idx in &group {
                for sample_idx in 0..n_samples {
                    if X[[sample_idx, metabolite_idx]].is_nan() {
                        // Impute using other metabolites in the same pathway
                        let pathway_values: Vec<f64> = group
                            .iter()
                            .filter(|&&idx| idx != metabolite_idx)
                            .filter_map(|&idx| {
                                let val = X[[sample_idx, idx]];
                                if !val.is_nan() {
                                    Some(val)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !pathway_values.is_empty() {
                            let pathway_mean =
                                pathway_values.iter().sum::<f64>() / pathway_values.len() as f64;
                            imputed[[sample_idx, metabolite_idx]] = pathway_mean;
                        }
                    }
                }
            }
        }

        Ok(imputed)
    }

    /// Create simplified pathway groupings
    fn create_pathway_groups(&self, n_metabolites: usize) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let group_size = 5; // Simplified: groups of 5 metabolites

        for i in (0..n_metabolites).step_by(group_size) {
            let end = (i + group_size).min(n_metabolites);
            groups.push((i..end).collect());
        }

        groups
    }

    /// Mass spectrometry specific processing
    fn mass_spec_specific_processing(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut processed = X.clone();

        // Apply log transformation for MS data (common preprocessing)
        for value in processed.iter_mut() {
            if *value > 0.0 && !value.is_nan() {
                *value = value.ln();
            }
        }

        // Account for technical artifacts in MS data
        processed = self.correct_technical_artifacts(&processed)?;

        Ok(processed)
    }

    /// Correct technical artifacts in mass spectrometry data
    fn correct_technical_artifacts(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut corrected = X.clone();

        // Batch effect correction (simplified)
        // In practice, would use methods like ComBat
        corrected = self.simple_batch_correction(&corrected)?;

        Ok(corrected)
    }

    /// Simple batch effect correction
    fn simple_batch_correction(&self, X: &Array2<f64>) -> ImputationResult<Array2<f64>> {
        let mut corrected = X.clone();
        let (n_samples, n_metabolites) = X.dim();

        // Assume first half and second half are different batches (simplified)
        let batch_size = n_samples / 2;

        for j in 0..n_metabolites {
            let batch1_values: Vec<f64> = X
                .column(j)
                .iter()
                .take(batch_size)
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            let batch2_values: Vec<f64> = X
                .column(j)
                .iter()
                .skip(batch_size)
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if !batch1_values.is_empty() && !batch2_values.is_empty() {
                let batch1_mean = batch1_values.iter().sum::<f64>() / batch1_values.len() as f64;
                let batch2_mean = batch2_values.iter().sum::<f64>() / batch2_values.len() as f64;
                let overall_mean = (batch1_mean + batch2_mean) / 2.0;

                // Center each batch to overall mean
                let batch1_correction = overall_mean - batch1_mean;
                let batch2_correction = overall_mean - batch2_mean;

                for i in 0..batch_size {
                    if !X[[i, j]].is_nan() {
                        corrected[[i, j]] += batch1_correction;
                    }
                }

                for i in batch_size..n_samples {
                    if !X[[i, j]].is_nan() {
                        corrected[[i, j]] += batch2_correction;
                    }
                }
            }
        }

        Ok(corrected)
    }

    /// Find samples with similar metabolic profiles
    fn find_similar_metabolic_profiles(
        &self,
        target_sample: usize,
        target_metabolite: usize,
        predictor_metabolites: &[usize],
        X: &Array2<f64>,
        k: usize,
    ) -> ImputationResult<Vec<usize>> {
        let n_samples = X.nrows();
        let mut similarities = Vec::new();

        let target_profile: Vec<f64> = predictor_metabolites
            .iter()
            .map(|&idx| X[[target_sample, idx]])
            .collect();

        for i in 0..n_samples {
            if i != target_sample && !X[[i, target_metabolite]].is_nan() {
                let sample_profile: Vec<f64> = predictor_metabolites
                    .iter()
                    .map(|&idx| X[[i, idx]])
                    .collect();

                let similarity = self.compute_profile_similarity(&target_profile, &sample_profile);
                similarities.push((i, similarity));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(similarities
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect())
    }

    /// Compute similarity between metabolic profiles
    fn compute_profile_similarity(&self, profile1: &[f64], profile2: &[f64]) -> f64 {
        if profile1.len() != profile2.len() {
            return 0.0;
        }

        let valid_pairs: Vec<(f64, f64)> = profile1
            .iter()
            .zip(profile2.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();

        if valid_pairs.len() < 2 {
            return 0.0;
        }

        // Compute Euclidean distance (convert to similarity)
        let distance: f64 = valid_pairs
            .iter()
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();

        // Convert distance to similarity
        (-distance).exp()
    }

    /// Compute correlation between metabolite profiles
    fn compute_metabolite_correlation(
        &self,
        metabolite1: &ArrayView1<f64>,
        metabolite2: &ArrayView1<f64>,
    ) -> f64 {
        let pairs: Vec<(f64, f64)> = metabolite1
            .iter()
            .zip(metabolite2.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();

        if pairs.len() < 3 {
            return 0.0;
        }

        let n = pairs.len() as f64;
        let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = pairs.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

/// Phylogenetic tree-guided imputation for evolutionary data
#[derive(Debug, Clone)]
pub struct PhylogeneticImputer {
    /// Phylogenetic tree structure (simplified as distance matrix)
    pub phylogenetic_distances: Option<Array2<f64>>,
    /// Evolutionary model: "jukes_cantor", "kimura", "gtr"
    pub evolution_model: String,
    /// Branch length scaling factor
    pub branch_scaling: f64,
    /// Use molecular clock assumption
    pub molecular_clock: bool,
}

impl Default for PhylogeneticImputer {
    fn default() -> Self {
        Self {
            phylogenetic_distances: None,
            evolution_model: "jukes_cantor".to_string(),
            branch_scaling: 1.0,
            molecular_clock: false,
        }
    }
}

impl PhylogeneticImputer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_phylogeny(mut self, distances: Array2<f64>) -> Self {
        self.phylogenetic_distances = Some(distances);
        self
    }

    /// Impute using phylogenetic relationships
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let (n_species, n_traits) = X.dim();
        let mut imputed = X.to_owned();

        let computed_distances = self.compute_genetic_distances(X)?;
        let distances = self
            .phylogenetic_distances
            .as_ref()
            .unwrap_or(&computed_distances);

        // Phylogenetic imputation for each trait
        for trait_idx in 0..n_traits {
            for species_idx in 0..n_species {
                if X[[species_idx, trait_idx]].is_nan() {
                    let imputed_value =
                        self.phylogenetic_impute_trait(species_idx, trait_idx, X, distances)?;
                    imputed[[species_idx, trait_idx]] = imputed_value;
                }
            }
        }

        Ok(imputed)
    }

    /// Compute genetic distances if not provided
    fn compute_genetic_distances(&self, X: &ArrayView2<f64>) -> ImputationResult<Array2<f64>> {
        let n_species = X.nrows();
        let mut distances = Array2::zeros((n_species, n_species));

        for i in 0..n_species {
            for j in (i + 1)..n_species {
                let distance = self.compute_pairwise_distance(&X.row(i), &X.row(j));
                distances[[i, j]] = distance;
                distances[[j, i]] = distance;
            }
        }

        Ok(distances)
    }

    /// Compute pairwise genetic distance between species
    fn compute_pairwise_distance(
        &self,
        species1: &ArrayView1<f64>,
        species2: &ArrayView1<f64>,
    ) -> f64 {
        let valid_pairs: Vec<(f64, f64)> = species1
            .iter()
            .zip(species2.iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .collect();

        if valid_pairs.is_empty() {
            return f64::INFINITY;
        }

        match self.evolution_model.as_str() {
            "jukes_cantor" => {
                let differences = valid_pairs
                    .iter()
                    .filter(|(x, y)| (x - y).abs() > 1e-10)
                    .count() as f64;
                let total_comparisons = valid_pairs.len() as f64;
                let p = differences / total_comparisons;

                if p < 0.75 {
                    -0.75 * (1.0 - 4.0 * p / 3.0).ln()
                } else {
                    f64::INFINITY
                }
            }
            "kimura" => {
                // Simplified Kimura 2-parameter model
                let hamming_distance: f64 = valid_pairs.iter().map(|(x, y)| (x - y).abs()).sum();
                hamming_distance / valid_pairs.len() as f64
            }
            _ => {
                // Default: Euclidean distance
                valid_pairs
                    .iter()
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
        }
    }

    /// Impute a single trait value using phylogenetic relationships
    fn phylogenetic_impute_trait(
        &self,
        species_idx: usize,
        trait_idx: usize,
        X: &ArrayView2<f64>,
        distances: &Array2<f64>,
    ) -> ImputationResult<f64> {
        let n_species = X.nrows();

        // Collect related species with known trait values
        let mut related_species = Vec::new();
        for i in 0..n_species {
            if i != species_idx && !X[[i, trait_idx]].is_nan() {
                let phylo_distance = distances[[species_idx, i]];
                if phylo_distance.is_finite() {
                    // Weight inversely by phylogenetic distance
                    let weight = (-phylo_distance * self.branch_scaling).exp();
                    related_species.push((i, weight, X[[i, trait_idx]]));
                }
            }
        }

        if related_species.is_empty() {
            // Fall back to overall trait mean
            let trait_values: Vec<f64> = X
                .column(trait_idx)
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if !trait_values.is_empty() {
                return Ok(trait_values.iter().sum::<f64>() / trait_values.len() as f64);
            } else {
                return Ok(0.0);
            }
        }

        // Sort by phylogenetic closeness (higher weight = closer)
        related_species.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Use top related species for imputation
        let top_related: Vec<_> = related_species.into_iter().take(10).collect();

        // Weighted average based on phylogenetic closeness
        let total_weight: f64 = top_related.iter().map(|(_, weight, _)| weight).sum();
        let weighted_sum: f64 = top_related
            .iter()
            .map(|(_, weight, value)| weight * value)
            .sum();

        if total_weight > 1e-10 {
            Ok(weighted_sum / total_weight)
        } else {
            Ok(top_related[0].2) // Use closest species value
        }
    }
}
