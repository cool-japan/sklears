//! B-spline basis function transformations
//!
//! This module provides B-spline basis function transformations for smooth regression
//! and non-linear feature generation.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for SplineTransformer
#[derive(Debug, Clone)]
pub struct SplineTransformerConfig {
    /// Number of splines (number of knots - 1)
    pub n_splines: usize,
    /// Degree of the spline polynomial
    pub degree: usize,
    /// Knot strategy
    pub knots: KnotStrategy,
    /// Whether to include bias (intercept) term
    pub include_bias: bool,
    /// Extrapolation strategy for values outside training range
    pub extrapolation: ExtrapolationStrategy,
}

impl Default for SplineTransformerConfig {
    fn default() -> Self {
        Self {
            n_splines: 5,
            degree: 3,
            knots: KnotStrategy::Uniform,
            include_bias: true,
            extrapolation: ExtrapolationStrategy::Continue,
        }
    }
}

/// Strategy for placing knots
#[derive(Debug, Clone, Copy)]
pub enum KnotStrategy {
    /// Place knots uniformly between min and max
    Uniform,
    /// Place knots at quantiles of the data
    Quantile,
}

/// Strategy for handling extrapolation
#[derive(Debug, Clone, Copy)]
pub enum ExtrapolationStrategy {
    /// Continue the spline beyond the boundary knots
    Continue,
    /// Set values to zero outside the boundary
    Zero,
    /// Raise an error for out-of-bounds values
    Error,
}

/// SplineTransformer generates B-spline basis functions
///
/// This transformer generates univariate B-spline basis functions for each feature
/// in X. B-splines are piecewise polynomials that are smooth at the boundaries
/// between pieces (knots).
#[derive(Debug, Clone)]
pub struct SplineTransformer<State = Untrained> {
    config: SplineTransformerConfig,
    state: PhantomData<State>,
    // Fitted parameters
    n_features_in_: Option<usize>,
    n_output_features_: Option<usize>,
    knots_: Option<Array2<Float>>,        // knots for each feature
    bsplines_: Option<Vec<BSplineBasis>>, // B-spline basis for each feature
}

/// B-spline basis for a single feature
#[derive(Debug, Clone)]
struct BSplineBasis {
    knots: Array1<Float>,
    degree: usize,
    n_splines: usize,
}

impl BSplineBasis {
    fn new(knots: Array1<Float>, degree: usize) -> Self {
        let n_splines = knots.len() - degree - 1;
        Self {
            knots,
            degree,
            n_splines,
        }
    }

    /// Evaluate B-spline basis functions for given values
    fn evaluate(&self, x: &Array1<Float>) -> Array2<Float> {
        let n_samples = x.len();
        let mut basis_values = Array2::<Float>::zeros((n_samples, self.n_splines));

        for (i, &val) in x.iter().enumerate() {
            for j in 0..self.n_splines {
                basis_values[[i, j]] = self.b_spline_basis(val, j, self.degree);
            }
        }

        basis_values
    }

    /// Cox-de Boor recursion formula for B-spline basis functions
    fn b_spline_basis(&self, x: Float, i: usize, p: usize) -> Float {
        if p == 0 {
            // Base case: B-spline of degree 0 is a step function
            if i < self.knots.len() - 1 && x >= self.knots[i] && x < self.knots[i + 1] {
                1.0
            } else if i == self.knots.len() - 2 && x == self.knots[i + 1] {
                // Special case for right boundary
                1.0
            } else {
                0.0
            }
        } else {
            // Recursive case: Cox-de Boor formula
            let mut result = 0.0;

            // First term
            if i + p < self.knots.len() {
                let denom = self.knots[i + p] - self.knots[i];
                if denom.abs() > 1e-12 {
                    result += (x - self.knots[i]) / denom * self.b_spline_basis(x, i, p - 1);
                }
            }

            // Second term
            if i + 1 < self.knots.len() - p {
                let denom = self.knots[i + p + 1] - self.knots[i + 1];
                if denom.abs() > 1e-12 {
                    result +=
                        (self.knots[i + p + 1] - x) / denom * self.b_spline_basis(x, i + 1, p - 1);
                }
            }

            result
        }
    }
}

impl SplineTransformer<Untrained> {
    /// Create a new SplineTransformer
    pub fn new() -> Self {
        Self {
            config: SplineTransformerConfig::default(),
            state: PhantomData,
            n_features_in_: None,
            n_output_features_: None,
            knots_: None,
            bsplines_: None,
        }
    }

    /// Set the number of splines
    pub fn n_splines(mut self, n_splines: usize) -> Self {
        self.config.n_splines = n_splines;
        self
    }

    /// Set the degree of the spline
    pub fn degree(mut self, degree: usize) -> Self {
        self.config.degree = degree;
        self
    }

    /// Set the knot strategy
    pub fn knots(mut self, knots: KnotStrategy) -> Self {
        self.config.knots = knots;
        self
    }

    /// Set whether to include bias
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.config.include_bias = include_bias;
        self
    }

    /// Set the extrapolation strategy
    pub fn extrapolation(mut self, extrapolation: ExtrapolationStrategy) -> Self {
        self.config.extrapolation = extrapolation;
        self
    }

    /// Generate knots for a feature based on the strategy
    fn generate_knots(&self, feature_values: &Array1<Float>) -> Array1<Float> {
        // For B-splines: num_knots = num_splines + degree + 1
        // We want num_splines B-spline basis functions
        let n_internal_knots = self.config.n_splines - self.config.degree - 1;
        let mut knots = Vec::new();

        let min_val = feature_values
            .iter()
            .fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = feature_values
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        // Add left boundary knots (repeated degree+1 times)
        for _ in 0..=self.config.degree {
            knots.push(min_val);
        }

        // Add internal knots
        if n_internal_knots > 0 {
            match self.config.knots {
                KnotStrategy::Uniform => {
                    for i in 1..=n_internal_knots {
                        let t = i as Float / (n_internal_knots + 1) as Float;
                        knots.push(min_val + t * (max_val - min_val));
                    }
                }
                KnotStrategy::Quantile => {
                    let mut sorted_values = feature_values.to_vec();
                    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    for i in 1..=n_internal_knots {
                        let quantile = i as Float / (n_internal_knots + 1) as Float;
                        let idx = ((sorted_values.len() - 1) as Float * quantile) as usize;
                        knots.push(sorted_values[idx]);
                    }
                }
            }
        }

        // Add right boundary knots (repeated degree+1 times)
        for _ in 0..=self.config.degree {
            knots.push(max_val);
        }

        Array1::from_vec(knots)
    }
}

impl SplineTransformer<Trained> {
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("SplineTransformer should be fitted")
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_
            .expect("SplineTransformer should be fitted")
    }

    /// Get the knots for each feature
    pub fn knots(&self) -> &Array2<Float> {
        self.knots_
            .as_ref()
            .expect("SplineTransformer should be fitted")
    }
}

impl Default for SplineTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for SplineTransformer<Untrained> {
    type Fitted = SplineTransformer<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit SplineTransformer on empty dataset".to_string(),
            ));
        }

        if self.config.n_splines == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_splines".to_string(),
                reason: "Number of splines must be positive".to_string(),
            });
        }

        // Generate knots and B-spline bases for each feature
        let mut bsplines = Vec::new();
        let mut max_knots = 0;

        for j in 0..n_features {
            let feature_column = x.column(j).to_owned();
            let knots = self.generate_knots(&feature_column);
            max_knots = max_knots.max(knots.len());

            let bspline = BSplineBasis::new(knots.clone(), self.config.degree);
            bsplines.push(bspline);
        }

        // Store knots in a matrix (pad with NaN for shorter knot vectors)
        let mut knots_matrix = Array2::<Float>::from_elem((n_features, max_knots), Float::NAN);
        for (j, bspline) in bsplines.iter().enumerate() {
            for (k, &knot) in bspline.knots.iter().enumerate() {
                knots_matrix[[j, k]] = knot;
            }
        }

        let n_splines_per_feature = self.config.n_splines;
        let n_output_features = if self.config.include_bias {
            n_features * (n_splines_per_feature + 1)
        } else {
            n_features * n_splines_per_feature
        };

        Ok(SplineTransformer {
            config: self.config,
            state: PhantomData,
            n_features_in_: Some(n_features),
            n_output_features_: Some(n_output_features),
            knots_: Some(knots_matrix),
            bsplines_: Some(bsplines),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for SplineTransformer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let bsplines = self
            .bsplines_
            .as_ref()
            .expect("SplineTransformer should be fitted");
        let n_output = self.n_output_features();
        let mut result = Array2::<Float>::zeros((n_samples, n_output));

        let mut output_col = 0;

        for (j, bspline) in bsplines.iter().enumerate().take(n_features) {
            let feature_column = x.column(j).to_owned();

            // Add bias term if requested
            if self.config.include_bias {
                result.column_mut(output_col).fill(1.0);
                output_col += 1;
            }

            // Evaluate B-spline basis functions
            let basis_values = bspline.evaluate(&feature_column);

            for k in 0..bspline.n_splines {
                result
                    .column_mut(output_col)
                    .assign(&basis_values.column(k));
                output_col += 1;
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_spline_transformer_basic() -> Result<()> {
        let x = array![[0.0], [0.5], [1.0]];
        let spline = SplineTransformer::new()
            .n_splines(3)
            .degree(2)
            .include_bias(false);

        let fitted = spline.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Should have 3 B-spline basis functions for 1 input feature
        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), 3);

        Ok(())
    }

    #[test]
    fn test_spline_transformer_with_bias() -> Result<()> {
        let x = array![[0.0], [1.0]];
        let spline = SplineTransformer::new()
            .n_splines(2)
            .degree(1)
            .include_bias(true);

        let fitted = spline.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Should have bias + 2 B-spline basis functions = 3 features
        assert_eq!(transformed.ncols(), 3);

        // First column should be all ones (bias)
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 1.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_spline_transformer_multiple_features() -> Result<()> {
        let x = array![[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]];
        let spline = SplineTransformer::new()
            .n_splines(2)
            .degree(1)
            .include_bias(false);

        let fitted = spline.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Should have 2 B-spline basis functions per feature = 4 total features
        assert_eq!(transformed.ncols(), 4);

        Ok(())
    }

    #[test]
    fn test_quantile_knots() -> Result<()> {
        let x = array![[0.0], [0.1], [0.5], [0.9], [1.0]];
        let spline = SplineTransformer::new()
            .n_splines(3)
            .degree(1)
            .knots(KnotStrategy::Quantile);

        let fitted = spline.fit(&x, &())?;

        // Should fit without errors
        assert_eq!(fitted.n_features_in(), 1);

        Ok(())
    }

    #[test]
    fn test_bspline_basis_degree_0() {
        let knots = array![0.0, 0.5, 1.0];
        let basis = BSplineBasis::new(knots, 0);

        // Degree 0 basis functions are step functions
        assert_abs_diff_eq!(basis.b_spline_basis(0.25, 0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(basis.b_spline_basis(0.75, 1, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(basis.b_spline_basis(0.25, 1, 0), 0.0, epsilon = 1e-10);
    }
}
