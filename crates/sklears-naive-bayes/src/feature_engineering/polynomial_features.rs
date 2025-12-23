//! Polynomial feature generation and transformation
//!
//! This module provides comprehensive polynomial feature generation implementations
//! including interaction features, polynomial features, spline features, and
//! basis function expansion. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Polynomial feature generation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolynomialMethod {
    /// Standard
    Standard,
    /// Interaction
    Interaction,
    /// Legendre
    Legendre,
    /// Chebyshev
    Chebyshev,
    /// Hermite
    Hermite,
    /// Laguerre
    Laguerre,
}

/// Configuration for polynomial features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialConfig {
    pub method: PolynomialMethod,
    pub degree: usize,
    pub include_bias: bool,
    pub interaction_only: bool,
    pub include_input_features: bool,
    pub order: String,
}

impl Default for PolynomialConfig {
    fn default() -> Self {
        Self {
            method: PolynomialMethod::Standard,
            degree: 2,
            include_bias: true,
            interaction_only: false,
            include_input_features: true,
            order: "C".to_string(),
        }
    }
}

/// Trait for polynomial feature generation
pub trait PolynomialFeatureGenerator<T>
where
    T: Clone + Copy + std::fmt::Debug + Default + std::ops::Mul<Output = T> + From<f64>,
{
    fn fit(&mut self, x: &ArrayView2<T>) -> Result<()>;
    fn transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>>;
    fn fit_transform(&mut self, x: &ArrayView2<T>) -> Result<Array2<T>>;
    fn get_feature_names(&self) -> Option<&Vec<String>>;
}

/// Standard polynomial features generator
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    config: PolynomialConfig,
    feature_names: Option<Vec<String>>,
    feature_combinations: Option<Vec<Vec<usize>>>,
    n_input_features: Option<usize>,
    n_output_features: Option<usize>,
}

impl PolynomialFeatures {
    pub fn new(config: PolynomialConfig) -> Self {
        Self {
            config,
            feature_names: None,
            feature_combinations: None,
            n_input_features: None,
            n_output_features: None,
        }
    }

    /// Fit polynomial features to data
    pub fn fit<T>(&mut self, x: &ArrayView2<T>) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug,
    {
        let (_, n_features) = x.dim();
        self.n_input_features = Some(n_features);

        let combinations = self.generate_combinations(n_features)?;
        self.n_output_features = Some(combinations.len());
        self.feature_combinations = Some(combinations);

        let feature_names = self.generate_feature_names(n_features)?;
        self.feature_names = Some(feature_names);

        Ok(())
    }

    /// Transform data using polynomial features
    pub fn transform<T>(&self, x: &ArrayView2<T>) -> Result<Array2<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Default + std::ops::Mul<Output = T> + From<f64>,
    {
        let combinations =
            self.feature_combinations
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "PolynomialFeatures not fitted".to_string(),
                })?;

        let (n_samples, _) = x.dim();
        let n_output_features = combinations.len();
        let mut result = Array2::default((n_samples, n_output_features));

        for (feature_idx, combination) in combinations.iter().enumerate() {
            for sample_idx in 0..n_samples {
                let mut feature_value = T::from(1.0);
                for &input_idx in combination {
                    if input_idx < x.ncols() {
                        feature_value = feature_value * x[(sample_idx, input_idx)];
                    }
                }
                result[(sample_idx, feature_idx)] = feature_value;
            }
        }

        Ok(result)
    }

    /// Generate feature combinations
    fn generate_combinations(&self, n_features: usize) -> Result<Vec<Vec<usize>>> {
        let mut combinations = Vec::new();

        if self.config.include_bias {
            combinations.push(Vec::new()); // Bias term (empty combination)
        }

        if self.config.include_input_features {
            for i in 0..n_features {
                combinations.push(vec![i]);
            }
        }

        // Generate polynomial combinations
        for degree in 2..=self.config.degree {
            let degree_combinations = self.generate_degree_combinations(n_features, degree)?;
            combinations.extend(degree_combinations);
        }

        Ok(combinations)
    }

    /// Generate combinations for a specific degree
    fn generate_degree_combinations(
        &self,
        n_features: usize,
        degree: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let mut combinations = Vec::new();

        if self.config.interaction_only && degree > 1 {
            // Only interaction terms (no repeated features)
            self.generate_interaction_combinations(
                n_features,
                degree,
                &mut Vec::new(),
                0,
                &mut combinations,
            );
        } else {
            // Full polynomial terms
            self.generate_polynomial_combinations(
                n_features,
                degree,
                &mut Vec::new(),
                &mut combinations,
            );
        }

        Ok(combinations)
    }

    /// Generate interaction combinations recursively
    #[allow(clippy::only_used_in_recursion)]
    fn generate_interaction_combinations(
        &self,
        n_features: usize,
        remaining_degree: usize,
        current: &mut Vec<usize>,
        start_idx: usize,
        result: &mut Vec<Vec<usize>>,
    ) {
        if remaining_degree == 0 {
            result.push(current.clone());
            return;
        }

        for i in start_idx..n_features {
            current.push(i);
            self.generate_interaction_combinations(
                n_features,
                remaining_degree - 1,
                current,
                i + 1,
                result,
            );
            current.pop();
        }
    }

    /// Generate full polynomial combinations
    #[allow(clippy::only_used_in_recursion)]
    fn generate_polynomial_combinations(
        &self,
        n_features: usize,
        degree: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == degree {
            result.push(current.clone());
            return;
        }

        let start_idx = current.last().copied().unwrap_or(0);
        for i in start_idx..n_features {
            current.push(i);
            self.generate_polynomial_combinations(n_features, degree, current, result);
            current.pop();
        }
    }

    /// Generate feature names
    fn generate_feature_names(&self, n_features: usize) -> Result<Vec<String>> {
        let combinations = self
            .feature_combinations
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("No combinations generated".to_string()))?;

        let mut names = Vec::new();

        for combination in combinations {
            if combination.is_empty() {
                names.push("1".to_string()); // Bias term
            } else {
                let name = combination
                    .iter()
                    .map(|&i| format!("x{}", i))
                    .collect::<Vec<_>>()
                    .join(" ");
                names.push(name);
            }
        }

        Ok(names)
    }

    pub fn feature_names(&self) -> Option<&Vec<String>> {
        self.feature_names.as_ref()
    }

    pub fn n_input_features(&self) -> Option<usize> {
        self.n_input_features
    }

    pub fn n_output_features(&self) -> Option<usize> {
        self.n_output_features
    }
}

impl<T> PolynomialFeatureGenerator<T> for PolynomialFeatures
where
    T: Clone + Copy + std::fmt::Debug + Default + std::ops::Mul<Output = T> + From<f64>,
{
    fn fit(&mut self, x: &ArrayView2<T>) -> Result<()> {
        self.fit(x)
    }

    fn transform(&self, x: &ArrayView2<T>) -> Result<Array2<T>> {
        self.transform(x)
    }

    fn fit_transform(&mut self, x: &ArrayView2<T>) -> Result<Array2<T>> {
        self.fit(x)?;
        self.transform(x)
    }

    fn get_feature_names(&self) -> Option<&Vec<String>> {
        self.feature_names.as_ref()
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new(PolynomialConfig::default())
    }
}

/// Interaction features generator
#[derive(Debug, Clone)]
pub struct InteractionFeatures {
    config: PolynomialConfig,
    interaction_pairs: Option<Vec<(usize, usize)>>,
    feature_names: Option<Vec<String>>,
}

impl InteractionFeatures {
    pub fn new(config: PolynomialConfig) -> Self {
        Self {
            config,
            interaction_pairs: None,
            feature_names: None,
        }
    }

    pub fn fit<T>(&mut self, x: &ArrayView2<T>) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug,
    {
        let (_, n_features) = x.dim();
        let mut pairs = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                pairs.push((i, j));
            }
        }

        self.interaction_pairs = Some(pairs);

        let feature_names = self.generate_interaction_names(n_features)?;
        self.feature_names = Some(feature_names);

        Ok(())
    }

    pub fn transform<T>(&self, x: &ArrayView2<T>) -> Result<Array2<T>>
    where
        T: Clone + Copy + std::fmt::Debug + Default + std::ops::Mul<Output = T>,
    {
        let pairs = self
            .interaction_pairs
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "InteractionFeatures not fitted".to_string(),
            })?;

        let (n_samples, _) = x.dim();
        let n_interactions = pairs.len();
        let mut result = Array2::default((n_samples, n_interactions));

        for (pair_idx, &(i, j)) in pairs.iter().enumerate() {
            for sample_idx in 0..n_samples {
                result[(sample_idx, pair_idx)] = x[(sample_idx, i)] * x[(sample_idx, j)];
            }
        }

        Ok(result)
    }

    fn generate_interaction_names(&self, n_features: usize) -> Result<Vec<String>> {
        let mut names = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                names.push(format!("x{} x{}", i, j));
            }
        }

        Ok(names)
    }

    pub fn interaction_pairs(&self) -> Option<&Vec<(usize, usize)>> {
        self.interaction_pairs.as_ref()
    }

    pub fn feature_names(&self) -> Option<&Vec<String>> {
        self.feature_names.as_ref()
    }
}

/// Spline features generator
#[derive(Debug, Clone)]
pub struct SplineFeatures {
    config: PolynomialConfig,
    knots: Option<HashMap<usize, Array1<f64>>>,
    spline_degree: usize,
}

impl SplineFeatures {
    pub fn new(config: PolynomialConfig, spline_degree: usize) -> Self {
        Self {
            config,
            knots: None,
            spline_degree,
        }
    }

    pub fn fit<T>(&mut self, x: &ArrayView2<T>) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        let (_, n_features) = x.dim();
        let mut knots = HashMap::new();

        for feature_idx in 0..n_features {
            let feature_knots = self.generate_knots(&x.column(feature_idx))?;
            knots.insert(feature_idx, feature_knots);
        }

        self.knots = Some(knots);
        Ok(())
    }

    pub fn transform<T>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let knots = self.knots.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "SplineFeatures not fitted".to_string(),
        })?;

        let (n_samples, n_features) = x.dim();
        let n_spline_features = n_features * self.config.degree;
        let mut result = Array2::zeros((n_samples, n_spline_features));

        for feature_idx in 0..n_features {
            if let Some(feature_knots) = knots.get(&feature_idx) {
                for sample_idx in 0..n_samples {
                    let value: f64 = x[(sample_idx, feature_idx)].into();
                    let spline_values = self.compute_spline_basis(value, feature_knots)?;

                    for (spline_idx, spline_value) in spline_values.iter().enumerate() {
                        let output_idx = feature_idx * self.config.degree + spline_idx;
                        if output_idx < n_spline_features {
                            result[(sample_idx, output_idx)] = *spline_value;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn generate_knots<T>(&self, column: &ArrayView1<T>) -> Result<Array1<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified knot generation
        let n_knots = self.config.degree + 1;
        let mut knots = Array1::zeros(n_knots);

        for i in 0..n_knots {
            knots[i] = i as f64 / (n_knots - 1) as f64;
        }

        Ok(knots)
    }

    fn compute_spline_basis(&self, value: f64, knots: &Array1<f64>) -> Result<Vec<f64>> {
        // Simplified B-spline basis computation
        let mut basis = Vec::with_capacity(self.config.degree);

        for i in 0..self.config.degree {
            let t = i as f64 / (self.config.degree - 1) as f64;
            let basis_value = (value - t).abs().max(0.0);
            basis.push(basis_value);
        }

        Ok(basis)
    }

    pub fn knots(&self) -> Option<&HashMap<usize, Array1<f64>>> {
        self.knots.as_ref()
    }
}

/// Basis function features
#[derive(Debug, Clone)]
pub struct BasisFunctionFeatures {
    config: PolynomialConfig,
    basis_type: BasisType,
    centers: Option<Array2<f64>>,
    scales: Option<Array1<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasisType {
    /// Gaussian
    Gaussian,
    /// Sigmoid
    Sigmoid,
    /// Polynomial
    Polynomial,
    /// Fourier
    Fourier,
}

impl BasisFunctionFeatures {
    pub fn new(config: PolynomialConfig, basis_type: BasisType) -> Self {
        Self {
            config,
            basis_type,
            centers: None,
            scales: None,
        }
    }

    pub fn fit<T>(&mut self, x: &ArrayView2<T>) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug + Into<f64>,
    {
        let (n_samples, n_features) = x.dim();
        let n_centers = self.config.degree;

        // Generate random centers
        let mut centers = Array2::zeros((n_centers, n_features));
        for i in 0..n_centers {
            for j in 0..n_features {
                centers[(i, j)] = (i + j) as f64; // Simplified center generation
            }
        }

        // Compute scales
        let scales = Array1::ones(n_centers);

        self.centers = Some(centers);
        self.scales = Some(scales);

        Ok(())
    }

    pub fn transform<T>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + Into<f64>,
    {
        let centers = self
            .centers
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "BasisFunctionFeatures not fitted".to_string(),
            })?;
        let scales = self
            .scales
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "BasisFunctionFeatures not fitted".to_string(),
            })?;

        let (n_samples, _) = x.dim();
        let n_centers = centers.nrows();
        let mut result = Array2::zeros((n_samples, n_centers));

        for sample_idx in 0..n_samples {
            for center_idx in 0..n_centers {
                let distance = self.compute_distance(x, sample_idx, centers, center_idx)?;
                let basis_value = self.apply_basis_function(distance, scales[center_idx])?;
                result[(sample_idx, center_idx)] = basis_value;
            }
        }

        Ok(result)
    }

    fn compute_distance<T>(
        &self,
        x: &ArrayView2<T>,
        sample_idx: usize,
        centers: &Array2<f64>,
        center_idx: usize,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + Into<f64>,
    {
        let mut distance = 0.0;
        for feature_idx in 0..x.ncols() {
            let diff = x[(sample_idx, feature_idx)].into() - centers[(center_idx, feature_idx)];
            distance += diff * diff;
        }
        Ok(distance.sqrt())
    }

    fn apply_basis_function(&self, distance: f64, scale: f64) -> Result<f64> {
        match self.basis_type {
            BasisType::Gaussian => Ok((-0.5 * (distance / scale).powi(2)).exp()),
            BasisType::Sigmoid => Ok(1.0 / (1.0 + (-distance / scale).exp())),
            BasisType::Polynomial => Ok(distance.powf(scale)),
            BasisType::Fourier => Ok((distance * scale).cos()),
        }
    }

    pub fn centers(&self) -> Option<&Array2<f64>> {
        self.centers.as_ref()
    }

    pub fn scales(&self) -> Option<&Array1<f64>> {
        self.scales.as_ref()
    }
}

/// Polynomial feature validator
#[derive(Debug, Clone)]
pub struct PolynomialValidator;

impl PolynomialValidator {
    pub fn validate_config(config: &PolynomialConfig) -> Result<()> {
        if config.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "Degree must be greater than 0".to_string(),
            ));
        }
        if config.degree > 10 {
            return Err(SklearsError::InvalidInput(
                "Degree too high (>10), may cause memory issues".to_string(),
            ));
        }
        Ok(())
    }

    pub fn validate_input<T>(x: &ArrayView2<T>) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug,
    {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Polynomial feature analyzer
#[derive(Debug, Clone)]
pub struct PolynomialAnalyzer {
    analysis_results: HashMap<String, f64>,
}

impl PolynomialAnalyzer {
    pub fn new() -> Self {
        Self {
            analysis_results: HashMap::new(),
        }
    }

    pub fn analyze_polynomial_expansion<T>(
        &mut self,
        original: &ArrayView2<T>,
        expanded: &Array2<T>,
    ) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug,
    {
        let original_features = original.ncols();
        let expanded_features = expanded.ncols();
        let expansion_ratio = expanded_features as f64 / original_features as f64;

        self.analysis_results
            .insert("original_features".to_string(), original_features as f64);
        self.analysis_results
            .insert("expanded_features".to_string(), expanded_features as f64);
        self.analysis_results
            .insert("expansion_ratio".to_string(), expansion_ratio);

        Ok(())
    }

    pub fn analysis_results(&self) -> &HashMap<String, f64> {
        &self.analysis_results
    }
}

impl Default for PolynomialAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_features() {
        let config = PolynomialConfig::default();
        let mut poly = PolynomialFeatures::new(config);

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert!(poly.fit(&x.view()).is_ok());
        assert!(poly.n_input_features() == Some(2));
        assert!(poly.feature_names().is_some());

        let transformed = poly.transform(&x.view()).unwrap();
        assert!(transformed.nrows() == 3);
        assert!(transformed.ncols() > 2); // Should have more features
    }

    #[test]
    fn test_interaction_features() {
        let config = PolynomialConfig::default();
        let mut interactions = InteractionFeatures::new(config);

        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert!(interactions.fit(&x.view()).is_ok());
        assert!(interactions.interaction_pairs().is_some());

        let transformed = interactions.transform(&x.view()).unwrap();
        assert_eq!(transformed.nrows(), 2);
        assert_eq!(transformed.ncols(), 3); // 3 interaction pairs for 3 features
    }

    #[test]
    fn test_polynomial_validator() {
        let valid_config = PolynomialConfig::default();
        assert!(PolynomialValidator::validate_config(&valid_config).is_ok());

        let mut invalid_config = PolynomialConfig::default();
        invalid_config.degree = 0;
        assert!(PolynomialValidator::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_basis_function_features() {
        let config = PolynomialConfig::default();
        let mut basis = BasisFunctionFeatures::new(config, BasisType::Gaussian);

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(basis.fit(&x.view()).is_ok());
        assert!(basis.centers().is_some());
        assert!(basis.scales().is_some());

        let transformed = basis.transform(&x.view()).unwrap();
        assert_eq!(transformed.nrows(), 2);
    }
}
