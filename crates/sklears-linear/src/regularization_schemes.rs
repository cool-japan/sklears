//! Composable Regularization Schemes for Linear Models
//!
//! This module implements various regularization schemes that can be composed
//! and used with the modular framework. All regularization schemes implement
//! the Regularization trait for consistency and pluggability.

use crate::modular_framework::Regularization;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// L2 (Ridge) regularization: α/2 * ||w||²₂
#[derive(Debug, Clone)]
pub struct L2Regularization {
    /// Regularization strength
    pub alpha: Float,
}

impl L2Regularization {
    /// Create a new L2 regularization with the specified strength
    pub fn new(alpha: Float) -> Result<Self> {
        if alpha < 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: format!(
                    "Regularization strength must be non-negative, got {}",
                    alpha
                ),
            });
        }
        Ok(Self { alpha })
    }
}

impl Regularization for L2Regularization {
    fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float> {
        let norm_squared = coefficients.mapv(|x| x * x).sum();
        Ok(0.5 * self.alpha * norm_squared)
    }

    fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>> {
        Ok(self.alpha * coefficients)
    }

    fn proximal_operator(
        &self,
        coefficients: &Array1<Float>,
        step_size: Float,
    ) -> Result<Array1<Float>> {
        // For L2: prox(x) = x / (1 + α * step_size)
        let shrinkage_factor = 1.0 / (1.0 + self.alpha * step_size);
        Ok(coefficients * shrinkage_factor)
    }

    fn strength(&self) -> Float {
        self.alpha
    }

    fn name(&self) -> &'static str {
        "L2Regularization"
    }
}

/// L1 (Lasso) regularization: α * ||w||₁
#[derive(Debug, Clone)]
pub struct L1Regularization {
    /// Regularization strength
    pub alpha: Float,
}

impl L1Regularization {
    /// Create a new L1 regularization with the specified strength
    pub fn new(alpha: Float) -> Result<Self> {
        if alpha < 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: format!(
                    "Regularization strength must be non-negative, got {}",
                    alpha
                ),
            });
        }
        Ok(Self { alpha })
    }
}

impl Regularization for L1Regularization {
    fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float> {
        let l1_norm = coefficients.mapv(|x| x.abs()).sum();
        Ok(self.alpha * l1_norm)
    }

    fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>> {
        // L1 penalty is not differentiable at 0, so we return the subgradient
        let subgradient = coefficients.mapv(|x| {
            if x > 0.0 {
                self.alpha
            } else if x < 0.0 {
                -self.alpha
            } else {
                0.0 // Could be any value in [-α, α]
            }
        });
        Ok(subgradient)
    }

    fn proximal_operator(
        &self,
        coefficients: &Array1<Float>,
        step_size: Float,
    ) -> Result<Array1<Float>> {
        // Soft thresholding operator
        let threshold = self.alpha * step_size;
        let result = coefficients.mapv(|x| {
            if x > threshold {
                x - threshold
            } else if x < -threshold {
                x + threshold
            } else {
                0.0
            }
        });
        Ok(result)
    }

    fn is_non_smooth(&self) -> bool {
        true
    }

    fn strength(&self) -> Float {
        self.alpha
    }

    fn name(&self) -> &'static str {
        "L1Regularization"
    }
}

/// Elastic Net regularization: α * (ρ * ||w||₁ + (1-ρ)/2 * ||w||²₂)
#[derive(Debug, Clone)]
pub struct ElasticNetRegularization {
    /// Total regularization strength
    pub alpha: Float,
    /// L1 ratio (ρ): 0 = Ridge, 1 = Lasso
    pub l1_ratio: Float,
}

impl ElasticNetRegularization {
    /// Create a new Elastic Net regularization
    pub fn new(alpha: Float, l1_ratio: Float) -> Result<Self> {
        if alpha < 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: format!(
                    "Regularization strength must be non-negative, got {}",
                    alpha
                ),
            });
        }
        if l1_ratio < 0.0 || l1_ratio > 1.0 {
            return Err(SklearsError::InvalidParameter {
                name: "l1_ratio".to_string(),
                reason: format!("L1 ratio must be between 0 and 1, got {}", l1_ratio),
            });
        }
        Ok(Self { alpha, l1_ratio })
    }

    /// Get the L1 regularization strength
    pub fn l1_strength(&self) -> Float {
        self.alpha * self.l1_ratio
    }

    /// Get the L2 regularization strength
    pub fn l2_strength(&self) -> Float {
        self.alpha * (1.0 - self.l1_ratio)
    }
}

impl Regularization for ElasticNetRegularization {
    fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float> {
        let l1_norm = coefficients.mapv(|x| x.abs()).sum();
        let l2_norm_squared = coefficients.mapv(|x| x * x).sum();

        let l1_penalty = self.l1_strength() * l1_norm;
        let l2_penalty = 0.5 * self.l2_strength() * l2_norm_squared;

        Ok(l1_penalty + l2_penalty)
    }

    fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>> {
        let l1_strength = self.l1_strength();
        let l2_strength = self.l2_strength();

        let gradient = coefficients.mapv(|x| {
            let l1_subgrad = if x > 0.0 {
                l1_strength
            } else if x < 0.0 {
                -l1_strength
            } else {
                0.0
            };
            let l2_grad = l2_strength * x;
            l1_subgrad + l2_grad
        });

        Ok(gradient)
    }

    fn proximal_operator(
        &self,
        coefficients: &Array1<Float>,
        step_size: Float,
    ) -> Result<Array1<Float>> {
        let l1_strength = self.l1_strength();
        let l2_strength = self.l2_strength();

        // Elastic net proximal operator combines soft thresholding with L2 shrinkage
        let threshold = l1_strength * step_size;
        let shrinkage_factor = 1.0 / (1.0 + l2_strength * step_size);

        let result = coefficients.mapv(|x| {
            let soft_thresholded = if x > threshold {
                x - threshold
            } else if x < -threshold {
                x + threshold
            } else {
                0.0
            };
            soft_thresholded * shrinkage_factor
        });

        Ok(result)
    }

    fn is_non_smooth(&self) -> bool {
        self.l1_ratio > 0.0
    }

    fn strength(&self) -> Float {
        self.alpha
    }

    fn name(&self) -> &'static str {
        "ElasticNetRegularization"
    }
}

/// Group Lasso regularization for grouped features
#[derive(Debug, Clone)]
pub struct GroupLassoRegularization {
    /// Regularization strength
    pub alpha: Float,
    /// Group assignment for each feature (group_id for each coefficient)
    pub groups: Vec<usize>,
}

impl GroupLassoRegularization {
    /// Create a new Group Lasso regularization
    pub fn new(alpha: Float, groups: Vec<usize>) -> Result<Self> {
        if alpha < 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: format!(
                    "Regularization strength must be non-negative, got {}",
                    alpha
                ),
            });
        }
        Ok(Self { alpha, groups })
    }
}

impl Regularization for GroupLassoRegularization {
    fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float> {
        if coefficients.len() != self.groups.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: self.groups.len(),
                actual: coefficients.len(),
            });
        }

        // Group the coefficients and compute L2 norm for each group
        let max_group = *self.groups.iter().max().unwrap_or(&0);
        let mut group_norms = vec![0.0; max_group + 1];

        for (i, &group_id) in self.groups.iter().enumerate() {
            group_norms[group_id] += coefficients[i] * coefficients[i];
        }

        let penalty = group_norms
            .iter()
            .map(|&norm_sq| norm_sq.sqrt())
            .sum::<Float>();
        Ok(self.alpha * penalty)
    }

    fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>> {
        if coefficients.len() != self.groups.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: self.groups.len(),
                actual: coefficients.len(),
            });
        }

        let max_group = *self.groups.iter().max().unwrap_or(&0);
        let mut group_norms = vec![0.0; max_group + 1];

        // Compute group norms
        for (i, &group_id) in self.groups.iter().enumerate() {
            group_norms[group_id] += coefficients[i] * coefficients[i];
        }

        // Convert to L2 norms
        for norm_sq in &mut group_norms {
            *norm_sq = norm_sq.sqrt();
        }

        // Compute subgradient
        let mut gradient = Array1::zeros(coefficients.len());
        for (i, &group_id) in self.groups.iter().enumerate() {
            if group_norms[group_id] > 0.0 {
                gradient[i] = self.alpha * coefficients[i] / group_norms[group_id];
            } else {
                gradient[i] = 0.0; // Subgradient at 0
            }
        }

        Ok(gradient)
    }

    fn proximal_operator(
        &self,
        coefficients: &Array1<Float>,
        step_size: Float,
    ) -> Result<Array1<Float>> {
        if coefficients.len() != self.groups.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: self.groups.len(),
                actual: coefficients.len(),
            });
        }

        let max_group = *self.groups.iter().max().unwrap_or(&0);
        let mut group_norms = vec![0.0; max_group + 1];

        // Compute group norms
        for (i, &group_id) in self.groups.iter().enumerate() {
            group_norms[group_id] += coefficients[i] * coefficients[i];
        }

        for norm_sq in &mut group_norms {
            *norm_sq = norm_sq.sqrt();
        }

        // Apply group soft thresholding
        let threshold = self.alpha * step_size;
        let mut result = coefficients.clone();

        for (i, &group_id) in self.groups.iter().enumerate() {
            let group_norm = group_norms[group_id];
            if group_norm > threshold {
                let shrinkage_factor = (group_norm - threshold) / group_norm;
                result[i] *= shrinkage_factor;
            } else {
                result[i] = 0.0;
            }
        }

        Ok(result)
    }

    fn is_non_smooth(&self) -> bool {
        true
    }

    fn strength(&self) -> Float {
        self.alpha
    }

    fn name(&self) -> &'static str {
        "GroupLassoRegularization"
    }
}

/// Composite regularization that combines multiple regularization schemes
#[derive(Debug)]
pub struct CompositeRegularization {
    /// List of regularization schemes with their weights
    regularizations: Vec<(Float, Box<dyn Regularization>)>,
}

impl CompositeRegularization {
    /// Create a new composite regularization
    pub fn new() -> Self {
        Self {
            regularizations: Vec::new(),
        }
    }

    /// Add a regularization scheme with a weight
    pub fn add_regularization(
        mut self,
        weight: Float,
        regularization: Box<dyn Regularization>,
    ) -> Self {
        self.regularizations.push((weight, regularization));
        self
    }

    /// Add L1 regularization
    pub fn add_l1(self, alpha: Float) -> Result<Self> {
        Ok(self.add_regularization(1.0, Box::new(L1Regularization::new(alpha)?)))
    }

    /// Add L2 regularization
    pub fn add_l2(self, alpha: Float) -> Result<Self> {
        Ok(self.add_regularization(1.0, Box::new(L2Regularization::new(alpha)?)))
    }

    /// Add Group Lasso regularization
    pub fn add_group_lasso(self, alpha: Float, groups: Vec<usize>) -> Result<Self> {
        Ok(self.add_regularization(1.0, Box::new(GroupLassoRegularization::new(alpha, groups)?)))
    }

    /// Check if any component is non-smooth
    pub fn is_any_non_smooth(&self) -> bool {
        self.regularizations
            .iter()
            .any(|(_, reg)| reg.is_non_smooth())
    }
}

impl Regularization for CompositeRegularization {
    fn penalty(&self, coefficients: &Array1<Float>) -> Result<Float> {
        let mut total_penalty = 0.0;
        for (weight, regularization) in &self.regularizations {
            total_penalty += weight * regularization.penalty(coefficients)?;
        }
        Ok(total_penalty)
    }

    fn penalty_gradient(&self, coefficients: &Array1<Float>) -> Result<Array1<Float>> {
        let mut total_gradient = Array1::zeros(coefficients.len());
        for (weight, regularization) in &self.regularizations {
            let grad = regularization.penalty_gradient(coefficients)?;
            total_gradient = total_gradient + *weight * grad;
        }
        Ok(total_gradient)
    }

    fn proximal_operator(
        &self,
        coefficients: &Array1<Float>,
        step_size: Float,
    ) -> Result<Array1<Float>> {
        // For composite regularization, we apply proximal operators sequentially
        // This is an approximation - the exact proximal operator is generally not available
        let mut result = coefficients.clone();
        for (weight, regularization) in &self.regularizations {
            result = regularization.proximal_operator(&result, weight * step_size)?;
        }
        Ok(result)
    }

    fn is_non_smooth(&self) -> bool {
        self.is_any_non_smooth()
    }

    fn strength(&self) -> Float {
        // Return the sum of weighted strengths
        self.regularizations
            .iter()
            .map(|(weight, reg)| weight * reg.strength())
            .sum()
    }

    fn name(&self) -> &'static str {
        "CompositeRegularization"
    }
}

/// Factory for creating common regularization schemes
pub struct RegularizationFactory;

impl RegularizationFactory {
    /// Create L1 (Lasso) regularization
    pub fn l1(alpha: Float) -> Result<Box<dyn Regularization>> {
        Ok(Box::new(L1Regularization::new(alpha)?))
    }

    /// Create L2 (Ridge) regularization
    pub fn l2(alpha: Float) -> Result<Box<dyn Regularization>> {
        Ok(Box::new(L2Regularization::new(alpha)?))
    }

    /// Create Elastic Net regularization
    pub fn elastic_net(alpha: Float, l1_ratio: Float) -> Result<Box<dyn Regularization>> {
        Ok(Box::new(ElasticNetRegularization::new(alpha, l1_ratio)?))
    }

    /// Create Group Lasso regularization
    pub fn group_lasso(alpha: Float, groups: Vec<usize>) -> Result<Box<dyn Regularization>> {
        Ok(Box::new(GroupLassoRegularization::new(alpha, groups)?))
    }

    /// Create a composite regularization builder
    pub fn composite() -> CompositeRegularization {
        CompositeRegularization::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_l2_regularization() {
        let reg = L2Regularization::new(0.5).unwrap();
        let coefficients = Array::from_vec(vec![1.0, -2.0, 3.0]);

        let penalty = reg.penalty(&coefficients).unwrap();
        let expected = 0.5 * 0.5 * (1.0 + 4.0 + 9.0); // α/2 * ||w||²
        assert!((penalty - expected).abs() < 1e-10);

        let gradient = reg.penalty_gradient(&coefficients).unwrap();
        let expected_grad = Array::from_vec(vec![0.5, -1.0, 1.5]); // α * w
        for (actual, expected) in gradient.iter().zip(expected_grad.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_l1_regularization() {
        let reg = L1Regularization::new(0.3).unwrap();
        let coefficients = Array::from_vec(vec![1.0, -2.0, 3.0]);

        let penalty = reg.penalty(&coefficients).unwrap();
        let expected = 0.3 * (1.0 + 2.0 + 3.0); // α * ||w||₁
        assert!((penalty - expected).abs() < 1e-10);

        assert!(reg.is_non_smooth());
    }

    #[test]
    fn test_l1_proximal_operator() {
        let reg = L1Regularization::new(1.0).unwrap();
        let coefficients = Array::from_vec(vec![2.0, -1.0, 0.5]);
        let step_size = 1.0;

        let result = reg.proximal_operator(&coefficients, step_size).unwrap();
        // Soft thresholding with threshold = 1.0 * 1.0 = 1.0
        let expected = Array::from_vec(vec![1.0, 0.0, 0.0]);
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_elastic_net_regularization() {
        let reg = ElasticNetRegularization::new(1.0, 0.7).unwrap();
        let coefficients = Array::from_vec(vec![1.0, -1.0]);

        let penalty = reg.penalty(&coefficients).unwrap();
        let l1_penalty = 0.7 * (1.0 + 1.0); // l1_ratio * α * ||w||₁
        let l2_penalty = 0.5 * 0.3 * (1.0 + 1.0); // (1-l1_ratio) * α/2 * ||w||²
        let expected = l1_penalty + l2_penalty;
        assert!((penalty - expected).abs() < 1e-10);

        assert!(reg.is_non_smooth()); // Because l1_ratio > 0
    }

    #[test]
    fn test_group_lasso_regularization() {
        let groups = vec![0, 0, 1, 1]; // Two groups: {0,1} and {2,3}
        let reg = GroupLassoRegularization::new(1.0, groups).unwrap();
        let coefficients = Array::from_vec(vec![3.0, 4.0, 0.0, 0.0]); // First group has norm 5.0, second group is zero

        let penalty = reg.penalty(&coefficients).unwrap();
        let expected = 5.0 + 0.0; // Sum of group L2 norms
        assert!((penalty - expected).abs() < 1e-10);

        assert!(reg.is_non_smooth());
    }

    #[test]
    fn test_composite_regularization() {
        let composite = CompositeRegularization::new()
            .add_l1(0.1)
            .unwrap()
            .add_l2(0.2)
            .unwrap();

        let coefficients = Array::from_vec(vec![1.0, -2.0]);

        let penalty = composite.penalty(&coefficients).unwrap();
        let l1_penalty = 0.1 * (1.0 + 2.0);
        let l2_penalty = 0.5 * 0.2 * (1.0 + 4.0);
        let expected = l1_penalty + l2_penalty;
        assert!((penalty - expected).abs() < 1e-10);

        assert!(composite.is_non_smooth()); // Because it includes L1
    }

    #[test]
    fn test_regularization_factory() {
        let l1 = RegularizationFactory::l1(0.5).unwrap();
        assert_eq!(l1.name(), "L1Regularization");

        let l2 = RegularizationFactory::l2(0.3).unwrap();
        assert_eq!(l2.name(), "L2Regularization");

        let elastic_net = RegularizationFactory::elastic_net(1.0, 0.8).unwrap();
        assert_eq!(elastic_net.name(), "ElasticNetRegularization");
    }

    #[test]
    fn test_invalid_parameters() {
        // Negative alpha should fail
        assert!(L1Regularization::new(-1.0).is_err());
        assert!(L2Regularization::new(-0.1).is_err());

        // Invalid l1_ratio should fail
        assert!(ElasticNetRegularization::new(1.0, -0.1).is_err());
        assert!(ElasticNetRegularization::new(1.0, 1.5).is_err());
    }
}
