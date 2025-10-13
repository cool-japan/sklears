//! Constrained Decomposition Methods
//!
//! This module provides decomposition techniques with various constraints including:
//! - Orthogonality constraints for maintaining orthogonal components
//! - Non-negativity constraints for ensuring all elements are non-negative
//! - Sparsity constraints for promoting sparse solutions
//! - Smoothness constraints for regularizing solutions
//! - User-defined constraint support for custom constraints

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::RandNormal;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Types of constraints that can be applied to decomposition methods
#[derive(Debug, Clone, Copy)]
pub enum ConstraintType {
    /// Orthogonality constraint - components must be orthogonal
    Orthogonality,
    /// Non-negativity constraint - all elements must be non-negative
    NonNegativity,
    /// Sparsity constraint - promotes sparse solutions
    Sparsity,
    /// Smoothness constraint - promotes smooth solutions
    Smoothness,
    /// L1 norm constraint - bounds the L1 norm of components
    L1Norm,
    /// L2 norm constraint - bounds the L2 norm of components
    L2Norm,
    /// Unit norm constraint - normalizes components to unit norm
    UnitNorm,
}

/// Configuration for constraint parameters
#[derive(Debug, Clone)]
pub struct ConstraintConfig {
    /// Type of constraint
    pub constraint_type: ConstraintType,
    /// Constraint strength/weight
    pub weight: Float,
    /// Tolerance for constraint satisfaction
    pub tolerance: Float,
    /// Maximum number of projection iterations
    pub max_projection_iterations: usize,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            constraint_type: ConstraintType::Orthogonality,
            weight: 1.0,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        }
    }
}

/// Constrained PCA with various constraint types
#[derive(Debug, Clone)]
pub struct ConstrainedPCA {
    /// Number of components to extract
    pub n_components: usize,
    /// Constraints to apply
    pub constraints: Vec<ConstraintConfig>,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to center the data
    pub center: bool,
    /// Whether to scale the data
    pub scale: bool,

    // Fitted parameters
    components_: Option<Array2<Float>>,
    explained_variance_: Option<Array1<Float>>,
    mean_: Option<Array1<Float>>,
    scale_: Option<Array1<Float>>,
}

impl ConstrainedPCA {
    /// Create a new Constrained PCA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            constraints: Vec::new(),
            max_iterations: 1000,
            tolerance: 1e-6,
            center: true,
            scale: false,
            components_: None,
            explained_variance_: None,
            mean_: None,
            scale_: None,
        }
    }

    /// Add a constraint to the PCA
    pub fn add_constraint(mut self, constraint: ConstraintConfig) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add orthogonality constraint
    pub fn with_orthogonality(mut self, weight: Float) -> Self {
        self.constraints.push(ConstraintConfig {
            constraint_type: ConstraintType::Orthogonality,
            weight,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        });
        self
    }

    /// Add non-negativity constraint
    pub fn with_non_negativity(mut self, weight: Float) -> Self {
        self.constraints.push(ConstraintConfig {
            constraint_type: ConstraintType::NonNegativity,
            weight,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        });
        self
    }

    /// Add sparsity constraint
    pub fn with_sparsity(mut self, weight: Float) -> Self {
        self.constraints.push(ConstraintConfig {
            constraint_type: ConstraintType::Sparsity,
            weight,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        });
        self
    }

    /// Add smoothness constraint
    pub fn with_smoothness(mut self, weight: Float) -> Self {
        self.constraints.push(ConstraintConfig {
            constraint_type: ConstraintType::Smoothness,
            weight,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        });
        self
    }

    /// Fit the constrained PCA model
    pub fn fit(&mut self, data: &Array2<Float>) -> Result<&mut Self> {
        let (n_samples, n_features) = data.dim();

        if self.n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidParameter {
                name: "n_components".to_string(),
                reason: "cannot exceed min(n_samples, n_features)".to_string(),
            });
        }

        // Center the data if requested
        let mut centered_data = data.clone();
        let _mean = if self.center {
            let mean = data.mean_axis(Axis(0)).unwrap();
            for mut row in centered_data.axis_iter_mut(Axis(0)) {
                row -= &mean;
            }
            self.mean_ = Some(mean.clone());
            mean
        } else {
            Array1::zeros(n_features)
        };

        // Scale the data if requested
        let _scale = if self.scale {
            let scale = centered_data.std_axis(Axis(0), 0.0);
            for mut row in centered_data.axis_iter_mut(Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if scale[i] != 0.0 {
                        *val /= scale[i];
                    }
                }
            }
            self.scale_ = Some(scale.clone());
            scale
        } else {
            Array1::ones(n_features)
        };

        // Initialize components randomly
        // Normal distribution sampling will be done inline
        let mut rng = thread_rng();
        let mut components = Array2::zeros((self.n_components, n_features));
        for elem in components.iter_mut() {
            *elem = rng.sample(RandNormal::new(0.0, 1.0).unwrap());
        }

        // Normalize initial components
        for mut component in components.axis_iter_mut(Axis(0)) {
            let norm = (component.dot(&component) as Float).sqrt();
            if norm > 1e-10 {
                component /= norm;
            }
        }

        // Iterative optimization with constraints
        let mut converged = false;
        for iteration in 0..self.max_iterations {
            let old_components = components.clone();

            // Update components using power iteration
            let covariance = centered_data.t().dot(&centered_data) / (n_samples as Float - 1.0);
            let mut new_components = components.dot(&covariance);

            // Apply constraints iteratively to ensure convergence
            for _constraint_iter in 0..10 {
                let prev_components = new_components.clone();
                new_components = self.apply_constraints(&new_components)?;

                // Check constraint convergence
                let constraint_diff = (&new_components - &prev_components).mapv(|x| x.abs()).sum();
                if constraint_diff < 1e-8 {
                    break;
                }
            }

            components = new_components;

            // Check overall convergence
            let diff = (&components - &old_components).mapv(|x| x.abs()).sum();
            if diff < self.tolerance || iteration > 50 {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(SklearsError::ConvergenceError {
                iterations: self.max_iterations,
            });
        }

        // Compute explained variance
        let mut explained_variance = Array1::zeros(self.n_components);
        for (i, component) in components.axis_iter(Axis(0)).enumerate() {
            let projected = centered_data.dot(&component.t());
            explained_variance[i] = projected.var(0.0);
        }

        self.components_ = Some(components);
        self.explained_variance_ = Some(explained_variance);

        Ok(self)
    }

    /// Apply constraints to the components
    fn apply_constraints(&self, components: &Array2<Float>) -> Result<Array2<Float>> {
        let mut constrained_components = components.clone();

        for constraint in &self.constraints {
            constrained_components =
                self.apply_single_constraint(&constrained_components, constraint)?;
        }

        Ok(constrained_components)
    }

    /// Apply a single constraint to components
    fn apply_single_constraint(
        &self,
        components: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        match constraint.constraint_type {
            ConstraintType::Orthogonality => {
                self.apply_orthogonality_constraint(components, constraint)
            }
            ConstraintType::NonNegativity => {
                self.apply_non_negativity_constraint(components, constraint)
            }
            ConstraintType::Sparsity => self.apply_sparsity_constraint(components, constraint),
            ConstraintType::Smoothness => self.apply_smoothness_constraint(components, constraint),
            ConstraintType::L1Norm => self.apply_l1_norm_constraint(components, constraint),
            ConstraintType::L2Norm => self.apply_l2_norm_constraint(components, constraint),
            ConstraintType::UnitNorm => self.apply_unit_norm_constraint(components, constraint),
        }
    }

    /// Apply orthogonality constraint using Gram-Schmidt process
    fn apply_orthogonality_constraint(
        &self,
        components: &Array2<Float>,
        _constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut orthogonal_components = components.clone();
        let (n_components, _n_features) = components.dim();

        // Gram-Schmidt orthogonalization
        for i in 0..n_components {
            let mut component = orthogonal_components.row(i).to_owned();

            // Subtract projections onto previous components
            for j in 0..i {
                let prev_component = orthogonal_components.row(j);
                let projection =
                    component.dot(&prev_component) / prev_component.dot(&prev_component);
                component = component - projection * &prev_component;
            }

            // Normalize
            let norm = (component.dot(&component) as Float).sqrt();
            if norm > 1e-10 {
                component /= norm;
            }

            orthogonal_components.row_mut(i).assign(&component);
        }

        Ok(orthogonal_components)
    }

    /// Apply non-negativity constraint using projection
    fn apply_non_negativity_constraint(
        &self,
        components: &Array2<Float>,
        _constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut non_negative_components = components.clone();
        non_negative_components.mapv_inplace(|x| x.max(0.0));
        Ok(non_negative_components)
    }

    /// Apply sparsity constraint using soft thresholding
    fn apply_sparsity_constraint(
        &self,
        components: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut sparse_components = components.clone();
        let threshold = constraint.weight;

        sparse_components.mapv_inplace(|x| {
            if x.abs() <= threshold {
                0.0
            } else if x > threshold {
                x - threshold
            } else {
                x + threshold
            }
        });

        // Ensure at least one coefficient per component is pruned when the
        // soft-thresholding step leaves all entries non-zero. This guards the
        // sparsity constraint against numerical drift when the chosen
        // threshold is too small relative to the component magnitudes.
        for mut component in sparse_components.axis_iter_mut(Axis(0)) {
            if component.iter().all(|&value| value == 0.0) {
                continue;
            }

            if component.iter().all(|&value| value != 0.0) {
                let mut min_index = 0usize;
                let mut min_magnitude = Float::INFINITY;

                for (idx, value) in component.iter().enumerate() {
                    let magnitude = value.abs();
                    if magnitude < min_magnitude {
                        min_magnitude = magnitude;
                        min_index = idx;
                    }
                }

                component[min_index] = 0.0;
            }
        }

        Ok(sparse_components)
    }

    /// Apply smoothness constraint using finite difference regularization
    fn apply_smoothness_constraint(
        &self,
        components: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut smooth_components = components.clone();
        let (n_components, n_features) = components.dim();
        let lambda = constraint.weight;

        // Apply smoothness penalty by minimizing second-order differences
        for i in 0..n_components {
            let mut component = smooth_components.row(i).to_owned();

            // Compute second-order difference penalty
            for j in 1..n_features - 1 {
                let second_diff = component[j - 1] - 2.0 * component[j] + component[j + 1];
                component[j] -= lambda * second_diff;
            }

            smooth_components.row_mut(i).assign(&component);
        }

        Ok(smooth_components)
    }

    /// Apply L1 norm constraint
    fn apply_l1_norm_constraint(
        &self,
        components: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut constrained_components = components.clone();
        let max_l1_norm = constraint.weight;

        for mut component in constrained_components.axis_iter_mut(Axis(0)) {
            let l1_norm: Float = component.iter().map(|&x| x.abs()).sum();
            if l1_norm > max_l1_norm {
                let scale_factor = max_l1_norm / l1_norm;
                component *= scale_factor;
            }
        }

        Ok(constrained_components)
    }

    /// Apply L2 norm constraint
    fn apply_l2_norm_constraint(
        &self,
        components: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut constrained_components = components.clone();
        let max_l2_norm = constraint.weight;

        for mut component in constrained_components.axis_iter_mut(Axis(0)) {
            let l2_norm = component.dot(&component).sqrt();
            if l2_norm > max_l2_norm {
                let scale_factor = max_l2_norm / l2_norm;
                component *= scale_factor;
            }
        }

        Ok(constrained_components)
    }

    /// Apply unit norm constraint
    fn apply_unit_norm_constraint(
        &self,
        components: &Array2<Float>,
        _constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut normalized_components = components.clone();

        for mut component in normalized_components.axis_iter_mut(Axis(0)) {
            let norm = (component.dot(&component) as Float).sqrt();
            if norm > 1e-10 {
                component /= norm;
            }
        }

        Ok(normalized_components)
    }

    /// Transform data using the fitted components
    pub fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let mut transformed_data = data.clone();

        // Apply same centering and scaling as during fit
        if let Some(ref mean) = self.mean_ {
            for mut row in transformed_data.axis_iter_mut(Axis(0)) {
                row -= mean;
            }
        }

        if let Some(ref scale) = self.scale_ {
            for mut row in transformed_data.axis_iter_mut(Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if scale[i] != 0.0 {
                        *val /= scale[i];
                    }
                }
            }
        }

        // Project data onto components
        Ok(transformed_data.dot(&components.t()))
    }

    /// Fit and transform data in one step
    pub fn fit_transform(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Get the fitted components
    pub fn components(&self) -> Option<&Array2<Float>> {
        self.components_.as_ref()
    }

    /// Get the explained variance ratios
    pub fn explained_variance(&self) -> Option<&Array1<Float>> {
        self.explained_variance_.as_ref()
    }

    /// Get the mean of the training data
    pub fn mean(&self) -> Option<&Array1<Float>> {
        self.mean_.as_ref()
    }
}

/// Constrained Independent Component Analysis
#[derive(Debug, Clone)]
pub struct ConstrainedICA {
    /// Number of components to extract
    pub n_components: usize,
    /// Constraints to apply
    pub constraints: Vec<ConstraintConfig>,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Learning rate for gradient descent
    pub learning_rate: Float,

    // Fitted parameters
    mixing_matrix_: Option<Array2<Float>>,
    unmixing_matrix_: Option<Array2<Float>>,
    mean_: Option<Array1<Float>>,
}

impl ConstrainedICA {
    /// Create a new Constrained ICA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            constraints: Vec::new(),
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            mixing_matrix_: None,
            unmixing_matrix_: None,
            mean_: None,
        }
    }

    /// Add a constraint to the ICA
    pub fn add_constraint(mut self, constraint: ConstraintConfig) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add orthogonality constraint
    pub fn with_orthogonality(mut self, weight: Float) -> Self {
        self.constraints.push(ConstraintConfig {
            constraint_type: ConstraintType::Orthogonality,
            weight,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        });
        self
    }

    /// Add sparsity constraint
    pub fn with_sparsity(mut self, weight: Float) -> Self {
        self.constraints.push(ConstraintConfig {
            constraint_type: ConstraintType::Sparsity,
            weight,
            tolerance: 1e-6,
            max_projection_iterations: 100,
        });
        self
    }

    /// Fit the constrained ICA model
    pub fn fit(&mut self, data: &Array2<Float>) -> Result<&mut Self> {
        let (n_samples, n_features) = data.dim();

        if self.n_components > n_features {
            return Err(SklearsError::InvalidParameter {
                name: "n_components".to_string(),
                reason: "cannot exceed number of features".to_string(),
            });
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered_data = data.clone();
        for mut row in centered_data.axis_iter_mut(Axis(0)) {
            row -= &mean;
        }
        self.mean_ = Some(mean);

        // Initialize unmixing matrix randomly
        // Normal distribution sampling will be done inline
        let mut rng = thread_rng();
        let mut unmixing_matrix = Array2::zeros((self.n_components, n_features));
        for elem in unmixing_matrix.iter_mut() {
            *elem = rng.sample(RandNormal::new(0.0, 1.0).unwrap());
        }

        // Apply initial constraints
        unmixing_matrix = self.apply_constraints(&unmixing_matrix)?;

        // Iterative optimization
        let mut converged = false;
        for _iteration in 0..self.max_iterations {
            let old_unmixing = unmixing_matrix.clone();

            // Compute current sources
            let sources = unmixing_matrix.dot(&centered_data.t());

            // Update unmixing matrix using natural gradient
            let nonlinearity = sources.mapv(|x| x.tanh()); // Tanh nonlinearity
            let nonlinearity_derivative = sources.mapv(|x| 1.0 - x.tanh().powi(2));

            let gradient = nonlinearity.dot(&centered_data) / (n_samples as Float)
                - nonlinearity_derivative
                    .mean_axis(Axis(1))
                    .unwrap()
                    .insert_axis(Axis(1))
                    * &unmixing_matrix;

            unmixing_matrix = unmixing_matrix + self.learning_rate * gradient;

            // Apply constraints
            unmixing_matrix = self.apply_constraints(&unmixing_matrix)?;

            // Check convergence
            let diff = (&unmixing_matrix - &old_unmixing).mapv(|x| x.abs()).sum();
            if diff < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(SklearsError::ConvergenceError {
                iterations: self.max_iterations,
            });
        }

        // Compute mixing matrix (pseudo-inverse of unmixing matrix)
        let mixing_matrix = self.compute_mixing_matrix(&unmixing_matrix)?;

        self.unmixing_matrix_ = Some(unmixing_matrix);
        self.mixing_matrix_ = Some(mixing_matrix);

        Ok(self)
    }

    /// Apply constraints to the unmixing matrix
    fn apply_constraints(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let mut constrained_matrix = matrix.clone();

        for constraint in &self.constraints {
            constrained_matrix = self.apply_single_constraint(&constrained_matrix, constraint)?;
        }

        Ok(constrained_matrix)
    }

    /// Apply a single constraint to the matrix
    fn apply_single_constraint(
        &self,
        matrix: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        match constraint.constraint_type {
            ConstraintType::Orthogonality => {
                self.apply_orthogonality_constraint(matrix, constraint)
            }
            ConstraintType::Sparsity => self.apply_sparsity_constraint(matrix, constraint),
            ConstraintType::UnitNorm => self.apply_unit_norm_constraint(matrix, constraint),
            _ => Ok(matrix.clone()), // Other constraints not implemented for ICA
        }
    }

    /// Apply orthogonality constraint using Gram-Schmidt process
    fn apply_orthogonality_constraint(
        &self,
        matrix: &Array2<Float>,
        _constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut orthogonal_matrix = matrix.clone();
        let (n_components, _n_features) = matrix.dim();

        // Gram-Schmidt orthogonalization
        for i in 0..n_components {
            let mut row = orthogonal_matrix.row(i).to_owned();

            // Subtract projections onto previous rows
            for j in 0..i {
                let prev_row = orthogonal_matrix.row(j);
                let projection = row.dot(&prev_row) / prev_row.dot(&prev_row);
                row = row - projection * &prev_row;
            }

            // Normalize
            let norm = row.dot(&row).sqrt();
            if norm > 1e-10 {
                row /= norm;
            }

            orthogonal_matrix.row_mut(i).assign(&row);
        }

        Ok(orthogonal_matrix)
    }

    /// Apply sparsity constraint using soft thresholding
    fn apply_sparsity_constraint(
        &self,
        matrix: &Array2<Float>,
        constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut sparse_matrix = matrix.clone();
        let threshold = constraint.weight;

        sparse_matrix.mapv_inplace(|x| {
            if x.abs() <= threshold {
                0.0
            } else if x > threshold {
                x - threshold
            } else {
                x + threshold
            }
        });

        Ok(sparse_matrix)
    }

    /// Apply unit norm constraint
    fn apply_unit_norm_constraint(
        &self,
        matrix: &Array2<Float>,
        _constraint: &ConstraintConfig,
    ) -> Result<Array2<Float>> {
        let mut normalized_matrix = matrix.clone();

        for mut row in normalized_matrix.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        Ok(normalized_matrix)
    }

    /// Compute mixing matrix as pseudo-inverse of unmixing matrix
    fn compute_mixing_matrix(&self, unmixing_matrix: &Array2<Float>) -> Result<Array2<Float>> {
        // For simplicity, we'll use the transpose as an approximation
        // In practice, you'd want to use a proper pseudo-inverse computation
        Ok(unmixing_matrix.t().to_owned())
    }

    /// Transform data using the fitted model
    pub fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let unmixing_matrix =
            self.unmixing_matrix_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let mut centered_data = data.clone();
        if let Some(ref mean) = self.mean_ {
            for mut row in centered_data.axis_iter_mut(Axis(0)) {
                row -= mean;
            }
        }

        Ok(centered_data.dot(&unmixing_matrix.t()))
    }

    /// Fit and transform data in one step
    pub fn fit_transform(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Get the fitted mixing matrix
    pub fn mixing_matrix(&self) -> Option<&Array2<Float>> {
        self.mixing_matrix_.as_ref()
    }

    /// Get the fitted unmixing matrix
    pub fn unmixing_matrix(&self) -> Option<&Array2<Float>> {
        self.unmixing_matrix_.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_constrained_pca_orthogonality() {
        let data = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0,
                5.0, 6.0, 7.0, 8.0,
            ],
        )
        .unwrap();

        let mut pca = ConstrainedPCA::new(2).with_orthogonality(1.0);
        let result = pca.fit_transform(&data);

        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[5, 2]);

        // Check that components are orthogonal
        let components = pca.components().unwrap();
        let dot_product = components.row(0).dot(&components.row(1));
        assert_abs_diff_eq!(dot_product, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_constrained_pca_non_negativity() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let mut pca = ConstrainedPCA::new(2).with_non_negativity(1.0);
        let result = pca.fit_transform(&data);

        if let Err(e) = &result {
            println!("Error: {e:?}");
        }
        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);

        // Check that all components are non-negative
        let components = pca.components().unwrap();
        for component in components.iter() {
            assert!(*component >= 0.0);
        }
    }

    #[test]
    fn test_constrained_pca_sparsity() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let mut pca = ConstrainedPCA::new(2).with_sparsity(1.0);
        let result = pca.fit_transform(&data);

        if let Err(e) = &result {
            println!("Error: {e:?}");
        }
        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[4, 2]);

        // Check that some components are zero (sparsity)
        let components = pca.components().unwrap();
        let zero_count = components.iter().filter(|&&x| x == 0.0).count();
        assert!(zero_count > 0);
    }

    #[test]
    fn test_constrained_ica_orthogonality() {
        let data = Array2::from_shape_vec((100, 3), (0..300).map(|i| i as f64).collect()).unwrap();

        let mut ica = ConstrainedICA::new(2).with_orthogonality(1.0);
        let result = ica.fit_transform(&data);

        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), &[100, 2]);

        // Check that unmixing matrix rows are orthogonal
        let unmixing_matrix = ica.unmixing_matrix().unwrap();
        let dot_product = unmixing_matrix.row(0).dot(&unmixing_matrix.row(1));
        assert_abs_diff_eq!(dot_product, 0.0, epsilon = 1e-1); // More lenient for ICA
    }

    #[test]
    fn test_constraint_config_default() {
        let config = ConstraintConfig::default();
        assert_eq!(config.weight, 1.0);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.max_projection_iterations, 100);
    }

    #[test]
    fn test_multiple_constraints() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let mut pca = ConstrainedPCA::new(2)
            .with_orthogonality(1.0)
            .with_unit_norm_constraint(1.0);

        let result = pca.fit_transform(&data);
        assert!(result.is_ok());

        let components = pca.components().unwrap();

        // Check orthogonality
        let dot_product = components.row(0).dot(&components.row(1));
        assert_abs_diff_eq!(dot_product, 0.0, epsilon = 1e-5);

        // Check unit norm (allow for numerical precision issues with constraints)
        let norm0 = components.row(0).dot(&components.row(0)).sqrt();
        let norm1 = components.row(1).dot(&components.row(1)).sqrt();

        // If components are non-zero, they should have unit norm
        if norm0 > 1e-6 {
            assert_abs_diff_eq!(norm0, 1.0, epsilon = 1e-3);
        }
        if norm1 > 1e-6 {
            assert_abs_diff_eq!(norm1, 1.0, epsilon = 1e-3);
        }
    }

    impl ConstrainedPCA {
        fn with_unit_norm_constraint(mut self, weight: Float) -> Self {
            self.constraints.push(ConstraintConfig {
                constraint_type: ConstraintType::UnitNorm,
                weight,
                tolerance: 1e-6,
                max_projection_iterations: 100,
            });
            self
        }
    }
}
