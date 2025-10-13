//! Multi-block Partial Least Squares

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Random};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Block scaling strategy for Multi-block PLS
#[derive(Debug, Clone, PartialEq)]
pub enum BlockScaling {
    /// No scaling
    None,
    /// Scale each block to unit variance
    UnitVariance,
    /// Scale each block to unit sum-of-squares
    UnitSumSquares,
    /// Scale each block by its first singular value
    SingularValue,
}

/// Multi-block Partial Least Squares (MB-PLS)
///
/// Multi-block PLS extends traditional PLS regression to handle multiple blocks
/// (views) of predictor variables simultaneously. This is particularly useful
/// when dealing with heterogeneous data sources (e.g., genomics, metabolomics,
/// clinical data) that need to be integrated to predict a response.
///
/// The algorithm finds latent components that maximize the covariance between
/// the combined predictor blocks and the response variables, while accounting
/// for the block structure of the data.
///
/// # Parameters
///
/// * `n_components` - Number of components to extract
/// * `block_scaling` - How to scale individual blocks
/// * `max_iter` - Maximum number of iterations for NIPALS algorithm
/// * `tol` - Tolerance for convergence
/// * `scale_y` - Whether to scale the response variables
/// * `copy` - Whether to copy input arrays
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::{MultiBlockPLS, BlockScaling};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// // Multiple blocks of predictors
/// let X1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let X2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
/// let X3 = array![[0.1], [0.3], [0.5], [0.7]];
/// let X_blocks = vec![X1, X2, X3];
///
/// let Y = array![[1.5], [3.5], [5.5], [7.5]];
///
/// let mbpls = MultiBlockPLS::new(2)
///     .block_scaling(BlockScaling::UnitVariance)
///     .max_iter(500);
///
/// let fitted = mbpls.fit(&X_blocks, &Y).unwrap();
/// let predictions = fitted.predict(&X_blocks).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiBlockPLS<State = Untrained> {
    /// Number of components to extract
    pub n_components: usize,
    /// Block scaling strategy
    pub block_scaling: BlockScaling,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to scale Y
    pub scale_y: bool,
    /// Whether to copy input arrays
    pub copy: bool,

    // Fitted attributes
    x_weights_: Option<Vec<Array2<Float>>>,
    y_weights_: Option<Array2<Float>>,
    x_loadings_: Option<Vec<Array2<Float>>>,
    y_loadings_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,
    super_weights_: Option<Array1<Float>>,
    coef_: Option<Array2<Float>>,
    x_means_: Option<Vec<Array1<Float>>>,
    y_mean_: Option<Array1<Float>>,
    x_stds_: Option<Vec<Array1<Float>>>,
    y_std_: Option<Array1<Float>>,
    block_scales_: Option<Vec<Float>>,
    n_iter_: Option<Vec<usize>>,
    explained_variance_: Option<Array1<Float>>,
    explained_variance_ratio_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl MultiBlockPLS<Untrained> {
    /// Create a new Multi-block PLS model
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            block_scaling: BlockScaling::UnitVariance,
            max_iter: 500,
            tol: 1e-6,
            scale_y: true,
            copy: true,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            super_weights_: None,
            coef_: None,
            x_means_: None,
            y_mean_: None,
            x_stds_: None,
            y_std_: None,
            block_scales_: None,
            n_iter_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            _state: PhantomData,
        }
    }

    /// Set the block scaling strategy
    pub fn block_scaling(mut self, block_scaling: BlockScaling) -> Self {
        self.block_scaling = block_scaling;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to scale Y
    pub fn scale_y(mut self, scale_y: bool) -> Self {
        self.scale_y = scale_y;
        self
    }

    /// Set whether to copy input arrays
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl MultiBlockPLS<Trained> {
    /// Get the block weights for each predictor block
    pub fn x_weights(&self) -> &Vec<Array2<Float>> {
        self.x_weights_.as_ref().unwrap()
    }

    /// Get the Y weights
    pub fn y_weights(&self) -> &Array2<Float> {
        self.y_weights_.as_ref().unwrap()
    }

    /// Get the block loadings for each predictor block
    pub fn x_loadings(&self) -> &Vec<Array2<Float>> {
        self.x_loadings_.as_ref().unwrap()
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> &Array2<Float> {
        self.y_loadings_.as_ref().unwrap()
    }

    /// Get the super weights (combination weights for blocks)
    pub fn super_weights(&self) -> &Array1<Float> {
        self.super_weights_.as_ref().unwrap()
    }

    /// Get the regression coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the number of iterations for each component
    pub fn n_iter(&self) -> &Vec<usize> {
        self.n_iter_.as_ref().unwrap()
    }

    /// Get the explained variance for each component
    pub fn explained_variance(&self) -> &Array1<Float> {
        self.explained_variance_.as_ref().unwrap()
    }

    /// Get the explained variance ratio for each component
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        self.explained_variance_ratio_.as_ref().unwrap()
    }

    /// Scale and center blocks according to the fitted scaling parameters
    fn scale_blocks(&self, x_blocks: &[Array2<Float>]) -> Result<Vec<Array2<Float>>> {
        let mut scaled_blocks = Vec::new();

        for (i, block) in x_blocks.iter().enumerate() {
            let x_mean = &self.x_means_.as_ref().unwrap()[i];
            let x_std = &self.x_stds_.as_ref().unwrap()[i];
            let block_scale = self.block_scales_.as_ref().unwrap()[i];

            let mut scaled_block = block.clone();

            // Center the block
            for (mut row, mean) in scaled_block.axis_iter_mut(Axis(0)).zip(x_mean.iter()) {
                row -= *mean;
            }

            // Scale features within block
            for (mut row, std) in scaled_block.axis_iter_mut(Axis(0)).zip(x_std.iter()) {
                if *std > 0.0 {
                    row /= *std;
                }
            }

            // Apply block-level scaling
            scaled_block /= block_scale;

            scaled_blocks.push(scaled_block);
        }

        Ok(scaled_blocks)
    }

    /// Scale Y according to fitted parameters
    fn scale_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.scale_y {
            return Ok(y.clone());
        }

        let y_mean = self.y_mean_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();

        let mut y_scaled = y.clone();

        // Center Y
        for (mut row, mean) in y_scaled.axis_iter_mut(Axis(0)).zip(y_mean.iter()) {
            row -= *mean;
        }

        // Scale Y
        for (mut row, std) in y_scaled.axis_iter_mut(Axis(0)).zip(y_std.iter()) {
            if *std > 0.0 {
                row /= *std;
            }
        }

        Ok(y_scaled)
    }

    /// Transform blocks to get super scores
    fn transform_blocks(&self, x_blocks: &[Array2<Float>]) -> Result<Array2<Float>> {
        let scaled_blocks = self.scale_blocks(x_blocks)?;
        let n_samples = scaled_blocks[0].nrows();
        let mut super_scores = Array2::zeros((n_samples, self.n_components));

        for comp in 0..self.n_components {
            let mut block_scores = Vec::new();

            // Compute scores for each block
            for (block_idx, block) in scaled_blocks.iter().enumerate() {
                let weights = &self.x_weights_.as_ref().unwrap()[block_idx];
                let scores = block.dot(&weights.column(comp));
                block_scores.push(scores);
            }

            // Combine block scores using super weights
            let mut combined_scores = Array1::zeros(n_samples);
            for (block_idx, scores) in block_scores.iter().enumerate() {
                combined_scores += &(scores * self.super_weights_.as_ref().unwrap()[block_idx]);
            }

            super_scores.column_mut(comp).assign(&combined_scores);
        }

        Ok(super_scores)
    }
}

impl Fit<Vec<Array2<Float>>, Array2<Float>> for MultiBlockPLS<Untrained> {
    type Fitted = MultiBlockPLS<Trained>;
    fn fit(mut self, x_blocks: &Vec<Array2<Float>>, y: &Array2<Float>) -> Result<Self::Fitted> {
        if x_blocks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one X block is required".into(),
            ));
        }

        let n_samples = x_blocks[0].nrows();
        let n_targets = y.ncols();

        // Validate all blocks have same number of samples
        for (i, block) in x_blocks.iter().enumerate() {
            if block.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Block {} has different number of samples",
                    i
                )));
            }
        }

        if y.nrows() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Y must have same number of samples as X blocks".into(),
            ));
        }

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput("Need at least 2 samples".into()));
        }

        // Compute means and standard deviations for each block
        let mut x_means = Vec::new();
        let mut x_stds = Vec::new();
        for block in x_blocks {
            let mean = block.mean_axis(Axis(0)).unwrap();
            let std = block.std_axis(Axis(0), 1.0);
            x_means.push(mean);
            x_stds.push(std);
        }

        // Compute Y statistics
        let y_mean = if self.scale_y {
            Some(y.mean_axis(Axis(0)).unwrap())
        } else {
            Some(Array1::zeros(n_targets))
        };

        let y_std = if self.scale_y {
            Some(y.std_axis(Axis(0), 1.0))
        } else {
            Some(Array1::ones(n_targets))
        };

        // Center and scale blocks
        let mut scaled_blocks = Vec::new();
        let mut block_scales = Vec::new();

        for (i, block) in x_blocks.iter().enumerate() {
            let mut scaled_block = block.clone();

            // Center the block
            for (mut row, mean) in scaled_block.axis_iter_mut(Axis(0)).zip(x_means[i].iter()) {
                row -= *mean;
            }

            // Scale features within block
            for (mut row, std) in scaled_block.axis_iter_mut(Axis(0)).zip(x_stds[i].iter()) {
                if *std > 0.0 {
                    row /= *std;
                }
            }

            // Compute block-level scaling
            let block_scale = match self.block_scaling {
                BlockScaling::None => 1.0,
                BlockScaling::UnitVariance => {
                    let var = scaled_block.var_axis(Axis(0), 1.0).sum().sqrt();
                    if var > 0.0 {
                        var
                    } else {
                        1.0
                    }
                }
                BlockScaling::UnitSumSquares => {
                    (scaled_block.mapv(|x| x * x).sum() / (n_samples as Float)).sqrt()
                }
                BlockScaling::SingularValue => {
                    // Approximation using Frobenius norm
                    (scaled_block.mapv(|x| x * x).sum()
                        / (n_samples as Float * scaled_block.ncols() as Float))
                        .sqrt()
                }
            };

            scaled_block /= block_scale;
            block_scales.push(block_scale);
            scaled_blocks.push(scaled_block);
        }

        // Scale Y
        let mut y_scaled = y.clone();
        if self.scale_y {
            for (mut row, mean) in y_scaled
                .axis_iter_mut(Axis(0))
                .zip(y_mean.as_ref().unwrap().iter())
            {
                row -= *mean;
            }
            for (mut row, std) in y_scaled
                .axis_iter_mut(Axis(0))
                .zip(y_std.as_ref().unwrap().iter())
            {
                if *std > 0.0 {
                    row /= *std;
                }
            }
        }

        // Initialize fitted attributes
        let n_blocks = x_blocks.len();
        let mut x_weights = Vec::new();
        let mut x_loadings = Vec::new();

        for block in &scaled_blocks {
            x_weights.push(Array2::zeros((block.ncols(), self.n_components)));
            x_loadings.push(Array2::zeros((block.ncols(), self.n_components)));
        }

        let mut y_weights = Array2::zeros((n_targets, self.n_components));
        let mut y_loadings = Array2::zeros((n_targets, self.n_components));
        let mut super_weights = Array1::ones(n_blocks) / (n_blocks as Float);
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));
        let mut n_iter = Vec::new();

        // Make mutable copies for deflation
        let mut x_residuals = scaled_blocks.clone();
        let mut y_residual = y_scaled.clone();

        // NIPALS algorithm for each component
        for comp in 0..self.n_components {
            let mut iter_count = 0;
            let mut converged = false;

            // Initialize Y weights as first column of Y residual
            let mut u = y_residual.column(0).to_owned();
            let mut u_old = Array1::zeros(n_samples);

            while iter_count < self.max_iter && !converged {
                // Step 1: Compute block weights and scores
                let mut block_scores = Vec::new();

                for (block_idx, block) in x_residuals.iter().enumerate() {
                    // Compute block weight
                    let weight = block.t().dot(&u);
                    let weight_norm = (weight.dot(&weight)).sqrt();
                    let normalized_weight = if weight_norm > 0.0 {
                        &weight / weight_norm
                    } else {
                        weight
                    };

                    x_weights[block_idx]
                        .column_mut(comp)
                        .assign(&normalized_weight);

                    // Compute block score
                    let score = block.dot(&normalized_weight);
                    block_scores.push(score);
                }

                // Step 2: Compute super score as weighted combination of block scores
                let mut t: Array1<Float> = Array1::zeros(n_samples);
                for (block_idx, score) in block_scores.iter().enumerate() {
                    t += &(score * super_weights[block_idx]);
                }

                // Normalize super score
                let t_norm = (t.dot(&t)).sqrt();
                if t_norm > 0.0 {
                    t /= t_norm;
                }

                x_scores.column_mut(comp).assign(&t);

                // Step 3: Compute Y weight and score
                let y_weight = y_residual.t().dot(&t);
                let y_weight_norm = (y_weight.dot(&y_weight)).sqrt();
                let normalized_y_weight = if y_weight_norm > 0.0 {
                    &y_weight / y_weight_norm
                } else {
                    y_weight
                };

                y_weights.column_mut(comp).assign(&normalized_y_weight);

                u = y_residual.dot(&normalized_y_weight);
                let u_norm = (u.dot(&u)).sqrt();
                if u_norm > 0.0 {
                    u /= u_norm;
                }

                y_scores.column_mut(comp).assign(&u);

                // Check convergence
                let diff = (&u - &u_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    converged = true;
                }

                u_old = u.clone();
                iter_count += 1;
            }

            n_iter.push(iter_count);

            // Compute loadings
            for (block_idx, block) in x_residuals.iter().enumerate() {
                let loading = block.t().dot(&x_scores.column(comp));
                x_loadings[block_idx].column_mut(comp).assign(&loading);
            }

            let y_loading = y_residual.t().dot(&y_scores.column(comp));
            y_loadings.column_mut(comp).assign(&y_loading);

            // Deflate X blocks and Y
            for (block_idx, block) in x_residuals.iter_mut().enumerate() {
                let deflation = x_scores
                    .column(comp)
                    .insert_axis(Axis(1))
                    .dot(&x_loadings[block_idx].column(comp).insert_axis(Axis(0)));
                *block -= &deflation;
            }

            let y_deflation = y_scores
                .column(comp)
                .insert_axis(Axis(1))
                .dot(&y_loadings.column(comp).insert_axis(Axis(0)));
            y_residual -= &y_deflation;
        }

        // Compute regression coefficients (simplified approach for multi-block)
        let mut coef = Array2::zeros((x_blocks.iter().map(|b| b.ncols()).sum(), n_targets));
        let mut start_idx = 0;

        for (block_idx, block) in x_blocks.iter().enumerate() {
            let end_idx = start_idx + block.ncols();
            // Use simpler coefficient calculation: weights * y_loadings.t()
            let block_coef = x_weights[block_idx].dot(&y_loadings.t());

            coef.slice_mut(scirs2_core::ndarray::s![start_idx..end_idx, ..])
                .assign(&block_coef);
            start_idx = end_idx;
        }

        // Compute explained variance
        let total_var = y.var_axis(Axis(0), 1.0).sum();
        let mut explained_var = Array1::zeros(self.n_components);
        let mut explained_var_ratio = Array1::zeros(self.n_components);

        for comp in 0..self.n_components {
            let comp_var = y_scores.column(comp).var(1.0);
            explained_var[comp] = comp_var;
            explained_var_ratio[comp] = comp_var / total_var;
        }

        // Store fitted attributes
        self.x_weights_ = Some(x_weights);
        self.y_weights_ = Some(y_weights);
        self.x_loadings_ = Some(x_loadings);
        self.y_loadings_ = Some(y_loadings);
        self.x_scores_ = Some(x_scores);
        self.y_scores_ = Some(y_scores);
        self.super_weights_ = Some(super_weights);
        self.coef_ = Some(coef);
        self.x_means_ = Some(x_means);
        self.y_mean_ = y_mean;
        self.x_stds_ = Some(x_stds);
        self.y_std_ = y_std;
        self.block_scales_ = Some(block_scales);
        self.n_iter_ = Some(n_iter);
        self.explained_variance_ = Some(explained_var);
        self.explained_variance_ratio_ = Some(explained_var_ratio);

        Ok(MultiBlockPLS {
            n_components: self.n_components,
            block_scaling: self.block_scaling,
            max_iter: self.max_iter,
            tol: self.tol,
            scale_y: self.scale_y,
            copy: self.copy,
            x_weights_: self.x_weights_,
            y_weights_: self.y_weights_,
            x_loadings_: self.x_loadings_,
            y_loadings_: self.y_loadings_,
            x_scores_: self.x_scores_,
            y_scores_: self.y_scores_,
            super_weights_: self.super_weights_,
            coef_: self.coef_,
            x_means_: self.x_means_,
            y_mean_: self.y_mean_,
            x_stds_: self.x_stds_,
            y_std_: self.y_std_,
            block_scales_: self.block_scales_,
            n_iter_: self.n_iter_,
            explained_variance_: self.explained_variance_,
            explained_variance_ratio_: self.explained_variance_ratio_,
            _state: PhantomData,
        })
    }
}

impl Transform<Vec<Array2<Float>>, Array2<Float>> for MultiBlockPLS<Trained> {
    /// Transform X blocks to latent space
    fn transform(&self, x_blocks: &Vec<Array2<Float>>) -> Result<Array2<Float>> {
        self.transform_blocks(x_blocks)
    }
}

impl Predict<Vec<Array2<Float>>, Array2<Float>> for MultiBlockPLS<Trained> {
    /// Predict Y using Multi-block PLS
    fn predict(&self, x_blocks: &Vec<Array2<Float>>) -> Result<Array2<Float>> {
        let scaled_blocks = self.scale_blocks(x_blocks)?;
        let n_samples = scaled_blocks[0].nrows();

        // Concatenate scaled blocks
        let total_features: usize = scaled_blocks.iter().map(|b| b.ncols()).sum();
        let mut x_concat = Array2::zeros((n_samples, total_features));
        let mut start_idx = 0;

        for block in &scaled_blocks {
            let end_idx = start_idx + block.ncols();
            x_concat
                .slice_mut(scirs2_core::ndarray::s![.., start_idx..end_idx])
                .assign(block);
            start_idx = end_idx;
        }

        // Predict using concatenated features
        let mut y_pred = x_concat.dot(self.coef());

        // Unscale Y if necessary
        if self.scale_y {
            let y_mean = self.y_mean_.as_ref().unwrap();
            let y_std = self.y_std_.as_ref().unwrap();

            for (mut row, (mean, std)) in y_pred
                .axis_iter_mut(Axis(0))
                .zip(y_mean.iter().zip(y_std.iter()))
            {
                if *std > 0.0 {
                    row *= *std;
                }
                row += *mean;
            }
        }

        Ok(y_pred)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multiblock_pls_basic() {
        let x1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let x2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let x_blocks = vec![x1, x2];
        let y = array![[1.5], [3.5], [5.5], [7.5]];

        let mbpls = MultiBlockPLS::new(1).max_iter(100);

        let fitted = mbpls.fit(&x_blocks, &y).unwrap();
        let predictions = fitted.predict(&x_blocks).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
        assert_eq!(fitted.x_weights().len(), 2); // Two blocks
    }

    #[test]
    fn test_block_scaling_strategies() {
        let x1 = array![
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0],
            [700.0, 800.0]
        ];
        let x2 = array![[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]];
        let x_blocks = vec![x1, x2];
        let y = array![[1.0], [2.0], [3.0], [4.0]];

        // Test different scaling strategies
        for scaling in &[
            BlockScaling::None,
            BlockScaling::UnitVariance,
            BlockScaling::UnitSumSquares,
            BlockScaling::SingularValue,
        ] {
            let mbpls = MultiBlockPLS::new(1)
                .block_scaling(scaling.clone())
                .max_iter(50);

            let fitted = mbpls.fit(&x_blocks, &y).unwrap();
            let predictions = fitted.predict(&x_blocks).unwrap();

            assert_eq!(predictions.shape(), &[4, 1]);
        }
    }

    #[test]
    fn test_multiblock_transform() {
        let x1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let x2 = array![
            [0.5, 1.5, 2.5],
            [2.5, 3.5, 4.5],
            [4.5, 5.5, 6.5],
            [6.5, 7.5, 8.5]
        ];
        let x_blocks = vec![x1, x2];
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mbpls = MultiBlockPLS::new(2).max_iter(100);

        let fitted = mbpls.fit(&x_blocks, &y).unwrap();
        let transformed = fitted.transform(&x_blocks).unwrap();

        assert_eq!(transformed.shape(), &[4, 2]);
    }

    #[test]
    fn test_explained_variance() {
        let x1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let x2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let x_blocks = vec![x1, x2];
        let y = array![[1.5], [3.5], [5.5], [7.5]];

        let mbpls = MultiBlockPLS::new(1).max_iter(100);

        let fitted = mbpls.fit(&x_blocks, &y).unwrap();

        assert_eq!(fitted.explained_variance().len(), 1);
        assert_eq!(fitted.explained_variance_ratio().len(), 1);
        assert!(fitted.explained_variance_ratio()[0] >= 0.0);
        assert!(fitted.explained_variance_ratio()[0] <= 1.0);
    }

    proptest! {
        #[test]
        fn test_multiblock_properties(
            n_samples in 4..20usize,
            n_features1 in 2..8usize,
            n_features2 in 2..8usize,
            n_targets in 1..4usize,
            n_components in 1..3usize,
        ) {
            let x1 = Array2::from_shape_fn((n_samples, n_features1), |_| {

                let mut rng = thread_rng();
                rng.gen_range(-1.0..1.0)
            });
            let x2 = Array2::from_shape_fn((n_samples, n_features2), |_| {

                let mut rng = thread_rng();
                rng.gen_range(-1.0..1.0)
            });
            let x_blocks = vec![x1, x2];
            let y = Array2::from_shape_fn((n_samples, n_targets), |_| {

                let mut rng = thread_rng();
                rng.gen_range(-1.0..1.0)
            });

            let n_comp = n_components.min((n_features1 + n_features2).min(n_targets));
            let mbpls = MultiBlockPLS::new(n_comp).max_iter(10);

            if let Ok(fitted) = mbpls.fit(&x_blocks, &y) {
                let predictions = fitted.predict(&x_blocks).unwrap();
                let transformed = fitted.transform(&x_blocks).unwrap();

                // Check output dimensions
                prop_assert_eq!(predictions.shape(), &[n_samples, n_targets]);
                prop_assert_eq!(transformed.shape(), &[n_samples, n_comp]);

                // Check that weights have correct dimensions
                prop_assert_eq!(fitted.x_weights().len(), 2);
                prop_assert_eq!(fitted.x_weights()[0].shape(), &[n_features1, n_comp]);
                prop_assert_eq!(fitted.x_weights()[1].shape(), &[n_features2, n_comp]);

                // Check explained variance
                prop_assert_eq!(fitted.explained_variance().len(), n_comp);
                prop_assert_eq!(fitted.explained_variance_ratio().len(), n_comp);
            }
        }
    }
}
