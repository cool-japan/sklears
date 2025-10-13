//! Polynomial and spline feature transformations
//!
//! This module provides feature engineering utilities for creating polynomial
//! and spline basis functions from input features.

use crate::*;
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::prelude::{SklearsError, Transform};

/// Polynomial Feature Generator
///
/// Generate polynomial and interaction features from input features.
/// This transformer can be used to capture non-linear relationships
/// by creating polynomial combinations of the original features.
///
/// # Parameters
///
/// * `degree` - Maximum degree of polynomial features
/// * `interaction_only` - If true, only interaction features are produced
/// * `include_bias` - If true, include a bias column of ones
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::polynomial_spline::PolynomialFeatures;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let poly = PolynomialFeatures::new()
///     .degree(2)
///     .include_bias(true);
///
/// let features = poly.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
}

impl PolynomialFeatures {
    /// Create a new PolynomialFeatures transformer
    pub fn new() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
        }
    }

    /// Set the maximum degree of polynomial features
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set whether to produce only interaction features
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Set whether to include a bias column
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Transform input features to polynomial features
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if self.degree == 0 && !self.include_bias {
            return Err(SklearsError::InvalidInput(
                "At least one of degree > 0 or include_bias=true must be specified".to_string(),
            ));
        }

        // Generate all polynomial combinations
        let combinations = self.generate_combinations(n_features)?;
        let n_output_features = combinations.len();

        let mut result = Array2::<Float>::zeros((n_samples, n_output_features));

        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            for (feat_idx, combination) in combinations.iter().enumerate() {
                let mut value = 1.0;

                if combination.is_empty() {
                    // Bias term
                    value = 1.0;
                } else {
                    for &feature_idx in combination {
                        value *= sample[feature_idx];
                    }
                }

                result[[sample_idx, feat_idx]] = value;
            }
        }

        Ok(result)
    }

    /// Get feature names for the polynomial features
    pub fn get_feature_names(&self, input_features: Option<&[String]>) -> SklResult<Vec<String>> {
        let n_features = match input_features {
            Some(names) => names.len(),
            None => {
                return Err(SklearsError::InvalidInput(
                    "Input feature names must be provided".to_string(),
                ))
            }
        };

        let combinations = self.generate_combinations(n_features)?;
        let mut feature_names = Vec::new();

        for combination in combinations {
            if combination.is_empty() {
                // Bias term
                feature_names.push("1".to_string());
            } else if combination.len() == 1 {
                // Single feature
                feature_names.push(input_features.unwrap()[combination[0]].clone());
            } else {
                // Interaction features
                let mut name = String::new();
                for (i, &feature_idx) in combination.iter().enumerate() {
                    if i > 0 {
                        name.push(' ');
                    }
                    name.push_str(&input_features.unwrap()[feature_idx]);
                }
                feature_names.push(name);
            }
        }

        Ok(feature_names)
    }

    /// Get the number of output features
    pub fn get_n_output_features(&self, n_input_features: usize) -> SklResult<usize> {
        let combinations = self.generate_combinations(n_input_features)?;
        Ok(combinations.len())
    }

    fn generate_combinations(&self, n_features: usize) -> SklResult<Vec<Vec<usize>>> {
        let mut combinations = Vec::new();

        // Add bias term if requested
        if self.include_bias {
            combinations.push(Vec::new());
        }

        // Generate combinations for each degree
        for degree in 1..=self.degree {
            let degree_combinations = if self.interaction_only {
                self.generate_interaction_combinations(n_features, degree)?
            } else {
                self.generate_degree_combinations(n_features, degree)?
            };
            combinations.extend(degree_combinations);
        }

        Ok(combinations)
    }

    fn generate_interaction_combinations(
        &self,
        n_features: usize,
        degree: usize,
    ) -> SklResult<Vec<Vec<usize>>> {
        let mut combinations = Vec::new();
        let mut current = vec![0; degree];

        self.generate_interaction_recursive(
            &mut combinations,
            &mut current,
            n_features,
            degree,
            0,
            0,
        );

        Ok(combinations)
    }

    fn generate_interaction_recursive(
        &self,
        combinations: &mut Vec<Vec<usize>>,
        current: &mut [usize],
        n_features: usize,
        degree: usize,
        pos: usize,
        start: usize,
    ) {
        if pos == degree {
            // All different features (interaction only)
            let mut unique_features = current.to_vec();
            unique_features.sort_unstable();
            unique_features.dedup();

            if unique_features.len() == degree {
                combinations.push(current.to_vec());
            }
            return;
        }

        for i in start..n_features {
            current[pos] = i;
            self.generate_interaction_recursive(
                combinations,
                current,
                n_features,
                degree,
                pos + 1,
                i + 1,
            );
        }
    }

    fn generate_degree_combinations(
        &self,
        n_features: usize,
        degree: usize,
    ) -> SklResult<Vec<Vec<usize>>> {
        let mut combinations = Vec::new();
        let mut current = vec![0; degree];

        self.generate_degree_recursive(&mut combinations, &mut current, n_features, degree, 0, 0);

        Ok(combinations)
    }

    fn generate_degree_recursive(
        &self,
        combinations: &mut Vec<Vec<usize>>,
        current: &mut [usize],
        n_features: usize,
        degree: usize,
        pos: usize,
        start: usize,
    ) {
        if pos == degree {
            combinations.push(current.to_vec());
            return;
        }

        for i in start..n_features {
            current[pos] = i;
            self.generate_degree_recursive(combinations, current, n_features, degree, pos + 1, i);
        }
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for PolynomialFeatures {
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.transform(X)
    }
}

/// Spline Basis Functions Generator
///
/// Generate spline basis functions for non-linear feature transformation.
/// Supports B-splines and Natural Cubic Splines for smooth curve fitting
/// and feature engineering applications.
///
/// # Parameters
///
/// * `n_splines` - Number of spline basis functions to generate
/// * `degree` - Degree of the spline (1 for linear, 2 for quadratic, 3 for cubic)
/// * `spline_type` - Type of spline (B-spline or Natural Cubic)
/// * `knots` - Knot positions (if None, uniformly distributed knots are used)
/// * `extrapolation` - How to handle values outside the knot range
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::polynomial_spline::{SplineBasisFunctions, SplineType};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
/// let spline = SplineBasisFunctions::new()
///     .n_splines(5)
///     .degree(3)
///     .spline_type(SplineType::BSpline);
///
/// let features = spline.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SplineBasisFunctions {
    n_splines: usize,
    degree: usize,
    spline_type: SplineType,
    knots: Option<Vec<Float>>,
    extrapolation: ExtrapolationMode,
}

/// Type of spline basis function
#[derive(Debug, Clone)]
pub enum SplineType {
    /// B-spline basis functions
    BSpline,
    /// Natural cubic spline
    NaturalCubic,
    /// Cubic spline with periodic boundary conditions
    Periodic,
}

/// Extrapolation mode for values outside knot range
#[derive(Debug, Clone)]
pub enum ExtrapolationMode {
    /// Raise an error for values outside knot range
    Error,
    /// Use constant extrapolation (boundary values)
    Constant,
    /// Use linear extrapolation
    Linear,
    /// Use polynomial extrapolation
    Polynomial,
}

impl SplineBasisFunctions {
    /// Create a new SplineBasisFunctions transformer
    pub fn new() -> Self {
        Self {
            n_splines: 5,
            degree: 3,
            spline_type: SplineType::BSpline,
            knots: None,
            extrapolation: ExtrapolationMode::Constant,
        }
    }

    /// Set the number of spline basis functions
    pub fn n_splines(mut self, n_splines: usize) -> Self {
        self.n_splines = n_splines;
        self
    }

    /// Set the degree of the spline
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the type of spline
    pub fn spline_type(mut self, spline_type: SplineType) -> Self {
        self.spline_type = spline_type;
        self
    }

    /// Set custom knot positions
    pub fn knots(mut self, knots: Vec<Float>) -> Self {
        self.knots = Some(knots);
        self
    }

    /// Set extrapolation mode
    pub fn extrapolation(mut self, extrapolation: ExtrapolationMode) -> Self {
        self.extrapolation = extrapolation;
        self
    }

    /// Transform input features using spline basis functions
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if n_features != 1 {
            return Err(SklearsError::InvalidInput(
                "SplineBasisFunctions currently supports only 1D input".to_string(),
            ));
        }

        let x_values: Vec<Float> = X.column(0).to_vec();
        let x_min = x_values.iter().cloned().fold(Float::INFINITY, Float::min);
        let x_max = x_values
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);

        // Generate knots if not provided
        let knots = match &self.knots {
            Some(knots) => knots.clone(),
            None => self.generate_uniform_knots(x_min, x_max)?,
        };

        let mut result = Array2::<Float>::zeros((n_samples, self.n_splines));

        for (i, &x) in x_values.iter().enumerate() {
            let basis_values = self.evaluate_basis_functions(x, &knots)?;

            for (j, &value) in basis_values.iter().enumerate() {
                if j < self.n_splines {
                    result[[i, j]] = value;
                }
            }
        }

        Ok(result)
    }

    fn generate_uniform_knots(&self, x_min: Float, x_max: Float) -> SklResult<Vec<Float>> {
        let n_internal_knots = if self.n_splines > self.degree + 1 {
            self.n_splines - self.degree - 1
        } else {
            0
        };

        let mut knots = Vec::new();

        // Add boundary knots (repeated for B-splines)
        for _ in 0..=self.degree {
            knots.push(x_min);
        }

        // Add internal knots
        if n_internal_knots > 0 {
            for i in 1..=n_internal_knots {
                let t = i as Float / (n_internal_knots + 1) as Float;
                knots.push(x_min + t * (x_max - x_min));
            }
        }

        // Add boundary knots (repeated for B-splines)
        for _ in 0..=self.degree {
            knots.push(x_max);
        }

        Ok(knots)
    }

    fn evaluate_basis_functions(&self, x: Float, knots: &[Float]) -> SklResult<Vec<Float>> {
        match self.spline_type {
            SplineType::BSpline => self.evaluate_bspline_basis(x, knots),
            SplineType::NaturalCubic => self.evaluate_natural_cubic_basis(x, knots),
            SplineType::Periodic => self.evaluate_periodic_basis(x, knots),
        }
    }

    fn evaluate_bspline_basis(&self, x: Float, knots: &[Float]) -> SklResult<Vec<Float>> {
        let n_knots = knots.len();
        let n_basis = n_knots - self.degree - 1;
        let mut basis = vec![0.0; n_basis];

        // Find the knot span
        let span = self.find_knot_span(x, knots)?;

        // Use de Boor's algorithm for B-spline evaluation
        let mut temp = vec![0.0; self.degree + 1];
        temp[0] = 1.0;

        for j in 1..=self.degree {
            let mut saved = 0.0;
            for r in 0..j {
                let alpha = (x - knots[span - self.degree + j + r])
                    / (knots[span + r + 1] - knots[span - self.degree + j + r]);
                let temp_val = temp[r];
                temp[r] = saved + (1.0 - alpha) * temp_val;
                saved = alpha * temp_val;
            }
            temp[j] = saved;
        }

        // Copy non-zero basis functions to result
        for (i, &value) in temp.iter().enumerate() {
            let basis_idx = span - self.degree + i;
            if basis_idx < n_basis {
                basis[basis_idx] = value;
            }
        }

        Ok(basis)
    }

    fn evaluate_natural_cubic_basis(&self, x: Float, knots: &[Float]) -> SklResult<Vec<Float>> {
        // Simplified natural cubic spline implementation
        let n_knots = knots.len();
        let mut basis = vec![0.0; n_knots];

        // Find the interval
        let mut interval = 0;
        for i in 0..n_knots - 1 {
            if x >= knots[i] && x <= knots[i + 1] {
                interval = i;
                break;
            }
        }

        // Linear interpolation for simplicity (could be extended to full cubic)
        if interval < n_knots - 1 {
            let t = (x - knots[interval]) / (knots[interval + 1] - knots[interval]);
            basis[interval] = 1.0 - t;
            basis[interval + 1] = t;
        }

        Ok(basis)
    }

    fn evaluate_periodic_basis(&self, x: Float, knots: &[Float]) -> SklResult<Vec<Float>> {
        // Simplified periodic spline (wrap x to domain)
        let x_min = knots[0];
        let x_max = knots[knots.len() - 1];
        let period = x_max - x_min;

        let x_wrapped = if period > 0.0 {
            x_min + ((x - x_min) % period + period) % period
        } else {
            x
        };

        self.evaluate_bspline_basis(x_wrapped, knots)
    }

    fn find_knot_span(&self, x: Float, knots: &[Float]) -> SklResult<usize> {
        let n = knots.len() - self.degree - 1;

        if x >= knots[n] {
            return Ok(n - 1);
        }

        if x <= knots[self.degree] {
            return Ok(self.degree);
        }

        // Binary search
        let mut low = self.degree;
        let mut high = n;
        let mut mid = (low + high) / 2;

        while x < knots[mid] || x >= knots[mid + 1] {
            if x < knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }

        Ok(mid)
    }
}

impl Default for SplineBasisFunctions {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for SplineBasisFunctions {
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.transform(X)
    }
}
