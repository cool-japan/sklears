//! Gene-Environment Interaction Analysis
//!
//! This module provides methods for analyzing gene-environment interactions
//! using cross-decomposition techniques.

use crate::cca::CCA;
use crate::multi_omics::GenomicsError;
use crate::pls::PLSRegression;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::traits::{Fit, Predict, Trained};
use sklears_core::types::Float;

/// Gene-Environment Interaction Analysis
pub struct GeneEnvironmentInteraction {
    /// Number of components to extract
    n_components: usize,
    /// Regularization parameter
    alpha: Float,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: Float,
    /// Whether to scale the data
    scale: bool,
    /// Interaction strength threshold
    interaction_threshold: Float,
}

impl GeneEnvironmentInteraction {
    /// Create a new gene-environment interaction analysis
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            alpha: 0.1,
            max_iter: 500,
            tol: 1e-6,
            scale: true,
            interaction_threshold: 0.1,
        }
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the interaction strength threshold
    pub fn interaction_threshold(mut self, threshold: Float) -> Self {
        self.interaction_threshold = threshold;
        self
    }

    /// Fit the gene-environment interaction model
    pub fn fit(
        &self,
        gene_data: ArrayView2<Float>,
        env_data: ArrayView2<Float>,
        phenotype: ArrayView1<Float>,
    ) -> Result<FittedGeneEnvironmentInteraction, GenomicsError> {
        if gene_data.nrows() != env_data.nrows() || gene_data.nrows() != phenotype.len() {
            return Err(GenomicsError::InvalidDimensions(
                "Gene data, environment data, and phenotype must have same number of samples"
                    .to_string(),
            ));
        }

        // Use PLS regression to model gene-environment interactions
        let pls = PLSRegression::new(self.n_components);

        // Combine gene and environment data
        let mut combined_data =
            Array2::zeros((gene_data.nrows(), gene_data.ncols() + env_data.ncols()));
        combined_data
            .slice_mut(s![.., ..gene_data.ncols()])
            .assign(&gene_data);
        combined_data
            .slice_mut(s![.., gene_data.ncols()..])
            .assign(&env_data);

        let phenotype_2d = phenotype.to_shape((phenotype.len(), 1)).unwrap().to_owned();
        let fitted_pls = pls.fit(&combined_data, &phenotype_2d)?;

        // Compute interaction effects
        let interaction_effects =
            self.compute_interaction_effects(&gene_data, &env_data, &fitted_pls)?;
        let significant_interactions =
            self.identify_significant_interactions(&interaction_effects)?;

        Ok(FittedGeneEnvironmentInteraction {
            fitted_pls,
            interaction_effects,
            significant_interactions,
            n_components: self.n_components,
        })
    }

    fn compute_interaction_effects(
        &self,
        gene_data: &ArrayView2<Float>,
        env_data: &ArrayView2<Float>,
        fitted_pls: &PLSRegression<Trained>,
    ) -> Result<Array2<Float>, GenomicsError> {
        let n_genes = gene_data.ncols();
        let n_env = env_data.ncols();
        let mut interaction_effects = Array2::zeros((n_genes, n_env));

        // Compute interaction effects as correlation between gene and environment effects
        for i in 0..n_genes {
            for j in 0..n_env {
                let gene_column = gene_data.column(i);
                let env_column = env_data.column(j);

                // Compute Pearson correlation as interaction effect
                let correlation = self.compute_correlation(&gene_column, &env_column)?;
                interaction_effects[(i, j)] = correlation;
            }
        }

        Ok(interaction_effects)
    }

    pub fn compute_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> Result<Float, GenomicsError> {
        if x.len() != y.len() {
            return Err(GenomicsError::InvalidDimensions(
                "Arrays must have same length".to_string(),
            ));
        }

        let n = x.len() as Float;
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            sum_xy = sum_xy + dx * dy;
            sum_x2 = sum_x2 + dx * dx;
            sum_y2 = sum_y2 + dy * dy;
        }

        let denominator = (sum_x2 * sum_y2).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(sum_xy / denominator)
        }
    }

    fn identify_significant_interactions(
        &self,
        interaction_effects: &Array2<Float>,
    ) -> Result<Vec<(usize, usize, Float)>, GenomicsError> {
        let mut significant = Vec::new();

        for ((i, j), &effect) in interaction_effects.indexed_iter() {
            if effect.abs() > self.interaction_threshold {
                significant.push((i, j, effect));
            }
        }

        // Sort by interaction strength (descending)
        significant.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

        Ok(significant)
    }
}

/// Fitted gene-environment interaction model
pub struct FittedGeneEnvironmentInteraction {
    fitted_pls: PLSRegression<Trained>,
    interaction_effects: Array2<Float>,
    significant_interactions: Vec<(usize, usize, Float)>,
    n_components: usize,
}

impl FittedGeneEnvironmentInteraction {
    /// Get the interaction effects matrix
    pub fn interaction_effects(&self) -> &Array2<Float> {
        &self.interaction_effects
    }

    /// Get the significant interactions
    pub fn significant_interactions(&self) -> &[(usize, usize, Float)] {
        &self.significant_interactions
    }

    /// Predict phenotype for new gene-environment data
    pub fn predict(
        &self,
        gene_data: ArrayView2<Float>,
        env_data: ArrayView2<Float>,
    ) -> Result<Array2<Float>, GenomicsError> {
        if gene_data.nrows() != env_data.nrows() {
            return Err(GenomicsError::InvalidDimensions(
                "Gene data and environment data must have same number of samples".to_string(),
            ));
        }

        // Combine gene and environment data
        let mut combined_data =
            Array2::zeros((gene_data.nrows(), gene_data.ncols() + env_data.ncols()));
        combined_data
            .slice_mut(s![.., ..gene_data.ncols()])
            .assign(&gene_data);
        combined_data
            .slice_mut(s![.., gene_data.ncols()..])
            .assign(&env_data);

        Ok(self.fitted_pls.predict(&combined_data)?)
    }
}
