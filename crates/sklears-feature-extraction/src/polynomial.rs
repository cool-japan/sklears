//! Polynomial and spline feature generation
//!
//! This module provides transformers for generating polynomial features and
//! spline basis functions for non-linear feature transformation.

use crate::*;
// use rayon::prelude::*;
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
/// use sklears_feature_extraction::polynomial::PolynomialFeatures;
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
/// use sklears_feature_extraction::polynomial::{SplineBasisFunctions, SplineType};
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

#[derive(Debug, Clone, Copy)]
pub enum SplineType {
    /// B-spline basis functions
    BSpline,
    /// Natural cubic spline basis functions
    NaturalCubic,
    /// Truncated power basis functions
    TruncatedPower,
}

#[derive(Debug, Clone, Copy)]
pub enum ExtrapolationMode {
    /// Constant extrapolation (edge values)
    Constant,
    /// Linear extrapolation
    Linear,
    /// Zero outside the knot range
    Zero,
    /// Raise error for out-of-bounds values
    Error,
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

    /// Set the degree of the splines
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

    /// Transform input data using spline basis functions
    pub fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != 1 {
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

        match self.spline_type {
            SplineType::BSpline => self.compute_bspline_basis(&x_values, &knots),
            SplineType::NaturalCubic => self.compute_natural_cubic_basis(&x_values, &knots),
            SplineType::TruncatedPower => self.compute_truncated_power_basis(&x_values, &knots),
        }
    }

    fn generate_uniform_knots(&self, x_min: Float, x_max: Float) -> SklResult<Vec<Float>> {
        if x_min >= x_max {
            return Err(SklearsError::InvalidInput(
                "Invalid range for knot generation".to_string(),
            ));
        }

        let mut knots = Vec::new();

        match self.spline_type {
            SplineType::BSpline => {
                // For B-splines, we need additional knots at the boundaries
                let n_interior_knots = self.n_splines - self.degree - 1;
                let _total_knots = n_interior_knots + 2 * (self.degree + 1);

                let step = (x_max - x_min) / (n_interior_knots + 1) as Float;

                // Add boundary knots with multiplicity
                for _ in 0..=self.degree {
                    knots.push(x_min);
                }

                // Add interior knots
                for i in 1..=n_interior_knots {
                    knots.push(x_min + i as Float * step);
                }

                // Add boundary knots with multiplicity
                for _ in 0..=self.degree {
                    knots.push(x_max);
                }
            }
            SplineType::NaturalCubic | SplineType::TruncatedPower => {
                // For natural cubic and truncated power, use simple uniform spacing
                let step = (x_max - x_min) / (self.n_splines - 1) as Float;
                for i in 0..self.n_splines {
                    knots.push(x_min + i as Float * step);
                }
            }
        }

        Ok(knots)
    }

    fn compute_bspline_basis(
        &self,
        x_values: &[Float],
        knots: &[Float],
    ) -> SklResult<Array2<Float>> {
        let n_samples = x_values.len();
        let n_basis = self.n_splines;
        let mut basis_matrix = Array2::zeros((n_samples, n_basis));

        for (sample_idx, &x) in x_values.iter().enumerate() {
            let basis_values = self.evaluate_bspline_basis(x, knots)?;

            for (basis_idx, &value) in basis_values.iter().enumerate() {
                if basis_idx < n_basis {
                    basis_matrix[[sample_idx, basis_idx]] = value;
                }
            }
        }

        Ok(basis_matrix)
    }

    fn evaluate_bspline_basis(&self, x: Float, knots: &[Float]) -> SklResult<Vec<Float>> {
        let n_knots = knots.len();
        let degree = self.degree;
        let n_basis = n_knots - degree - 1;

        if n_basis == 0 {
            return Ok(vec![]);
        }

        let mut basis = vec![0.0; n_basis];

        // Handle extrapolation
        let (x_clamped, is_extrapolated, extrapolation_factor) = match self.extrapolation {
            ExtrapolationMode::Constant => (x.max(knots[0]).min(knots[n_knots - 1]), false, 0.0),
            ExtrapolationMode::Zero => {
                if x < knots[0] || x > knots[n_knots - 1] {
                    return Ok(basis); // All zeros
                }
                (x, false, 0.0)
            }
            ExtrapolationMode::Error => {
                if x < knots[0] || x > knots[n_knots - 1] {
                    return Err(SklearsError::InvalidInput(
                        "Value outside knot range".to_string(),
                    ));
                }
                (x, false, 0.0)
            }
            ExtrapolationMode::Linear => {
                if x < knots[0] {
                    // Linear extrapolation to the left
                    let boundary = knots[0];
                    let next_knot = if n_knots > 1 {
                        knots[1]
                    } else {
                        knots[0] + 1.0
                    };
                    let slope = 1.0 / (next_knot - boundary);
                    let factor = (x - boundary) * slope;
                    (boundary, true, factor)
                } else if x > knots[n_knots - 1] {
                    // Linear extrapolation to the right
                    let boundary = knots[n_knots - 1];
                    let prev_knot = if n_knots > 1 {
                        knots[n_knots - 2]
                    } else {
                        knots[n_knots - 1] - 1.0
                    };
                    let slope = 1.0 / (boundary - prev_knot);
                    let factor = (x - boundary) * slope;
                    (boundary, true, factor)
                } else {
                    (x, false, 0.0)
                }
            }
        };

        // De Boor's algorithm for B-spline evaluation
        // Start with degree 0 (piecewise constant)
        for i in 0..n_basis {
            if i + degree + 1 < n_knots && x_clamped >= knots[i] && x_clamped < knots[i + 1] {
                basis[i] = 1.0;
            }
        }

        // Handle the right boundary
        if (x_clamped - knots[n_knots - 1]).abs() < 1e-10 && n_basis > 0 {
            basis[n_basis - 1] = 1.0;
        }

        // Recursive computation for higher degrees
        for d in 1..=degree {
            let mut new_basis = vec![0.0; n_basis];

            for i in 0..n_basis {
                if i + d < n_knots && (knots[i + d] - knots[i]).abs() > 1e-10 {
                    let alpha = (x_clamped - knots[i]) / (knots[i + d] - knots[i]);
                    new_basis[i] += alpha * basis[i];
                }

                if i > 0 && i + d < n_knots && (knots[i + d] - knots[i]).abs() > 1e-10 {
                    let alpha = (knots[i + d] - x_clamped) / (knots[i + d] - knots[i]);
                    new_basis[i] += alpha * basis[i - 1];
                }
            }

            basis = new_basis;
        }

        // Apply linear extrapolation if needed
        if is_extrapolated && extrapolation_factor != 0.0 {
            // For linear extrapolation, scale the basis functions by the extrapolation factor
            // This provides a linear extension of the spline beyond the knot boundaries
            for value in basis.iter_mut() {
                *value *= 1.0 + extrapolation_factor.abs().min(10.0); // Cap the factor for numerical stability
            }
        }

        Ok(basis)
    }

    fn compute_natural_cubic_basis(
        &self,
        x_values: &[Float],
        knots: &[Float],
    ) -> SklResult<Array2<Float>> {
        let n_samples = x_values.len();
        let n_knots = knots.len();
        let n_basis = n_knots;
        let mut basis_matrix = Array2::zeros((n_samples, n_basis));

        if n_knots < 4 {
            return Err(SklearsError::InvalidInput(
                "Natural cubic splines require at least 4 knots".to_string(),
            ));
        }

        for (sample_idx, &x) in x_values.iter().enumerate() {
            // Linear basis functions
            basis_matrix[[sample_idx, 0]] = 1.0;
            basis_matrix[[sample_idx, 1]] = x;

            // Cubic basis functions
            for k in 2..n_basis {
                if k - 2 < n_knots - 2 {
                    let knot = knots[k - 2];
                    let _x_min = knots[0];
                    let x_max = knots[n_knots - 1];

                    let d_k = self.natural_cubic_basis_function(x, knot, x_max);
                    let d_n = self.natural_cubic_basis_function(x, x_max, x_max);

                    basis_matrix[[sample_idx, k]] = d_k - d_n;
                }
            }
        }

        Ok(basis_matrix)
    }

    fn natural_cubic_basis_function(&self, x: Float, knot: Float, _x_max: Float) -> Float {
        if x <= knot {
            0.0
        } else {
            let diff = x - knot;
            diff.powi(3) / 6.0
        }
    }

    fn compute_truncated_power_basis(
        &self,
        x_values: &[Float],
        knots: &[Float],
    ) -> SklResult<Array2<Float>> {
        let n_samples = x_values.len();
        let n_basis = self.n_splines;
        let mut basis_matrix = Array2::zeros((n_samples, n_basis));

        for (sample_idx, &x) in x_values.iter().enumerate() {
            // Polynomial terms
            for d in 0..=self.degree.min(n_basis - 1) {
                basis_matrix[[sample_idx, d]] = x.powi(d as i32);
            }

            // Truncated power functions
            let mut basis_idx = self.degree + 1;
            for &knot in knots.iter() {
                if basis_idx >= n_basis {
                    break;
                }

                if x > knot {
                    basis_matrix[[sample_idx, basis_idx]] = (x - knot).powi(self.degree as i32);
                }
                basis_idx += 1;
            }
        }

        Ok(basis_matrix)
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
