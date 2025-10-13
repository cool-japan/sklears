//! Tensor product polynomial features for multi-dimensional feature interactions

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Tensor product ordering for multi-dimensional polynomial features
#[derive(Debug, Clone)]
/// TensorOrdering
pub enum TensorOrdering {
    /// Lexicographic ordering (default)
    Lexicographic,
    /// Graded lexicographic ordering
    GradedLexicographic,
    /// Reverse graded lexicographic ordering
    ReversedGradedLexicographic,
}

/// Tensor contraction method for reducing dimensionality
#[derive(Debug, Clone)]
/// ContractionMethod
pub enum ContractionMethod {
    /// No contraction (full tensor)
    None,
    /// Contract over specified indices
    Indices(Vec<usize>),
    /// Contract to specified rank
    Rank(usize),
    /// Symmetric contraction
    Symmetric,
}

/// Tensor Product Polynomial Features
///
/// Generates tensor product polynomial features for multi-dimensional data.
/// This captures higher-order interactions between features across multiple
/// dimensions and feature groups.
///
/// # Parameters
///
/// * `degree` - Maximum degree of polynomial features (default: 2)
/// * `n_dimensions` - Number of tensor dimensions (default: 2)
/// * `include_bias` - Include bias term (default: true)
/// * `interaction_only` - Include only interaction terms (default: false)
/// * `tensor_ordering` - Ordering scheme for tensor indices
/// * `contraction_method` - Method for tensor contraction
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::tensor_polynomial::TensorPolynomialFeatures;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let tensor_poly = TensorPolynomialFeatures::new(2, 2);
/// let fitted_tensor = tensor_poly.fit(&X, &()).unwrap();
/// let X_transformed = fitted_tensor.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
/// TensorPolynomialFeatures
pub struct TensorPolynomialFeatures<State = Untrained> {
    /// Maximum degree of polynomial features
    pub degree: u32,
    /// Number of tensor dimensions
    pub n_dimensions: usize,
    /// Include bias term
    pub include_bias: bool,
    /// Include only interaction terms
    pub interaction_only: bool,
    /// Tensor index ordering
    pub tensor_ordering: TensorOrdering,
    /// Tensor contraction method
    pub contraction_method: ContractionMethod,

    // Fitted attributes
    n_input_features_: Option<usize>,
    n_output_features_: Option<usize>,
    tensor_indices_: Option<Vec<Vec<Vec<u32>>>>,
    contraction_map_: Option<Vec<Vec<usize>>>,

    _state: PhantomData<State>,
}

impl TensorPolynomialFeatures<Untrained> {
    /// Create a new tensor polynomial features transformer
    pub fn new(degree: u32, n_dimensions: usize) -> Self {
        Self {
            degree,
            n_dimensions,
            include_bias: true,
            interaction_only: false,
            tensor_ordering: TensorOrdering::Lexicographic,
            contraction_method: ContractionMethod::None,
            n_input_features_: None,
            n_output_features_: None,
            tensor_indices_: None,
            contraction_map_: None,
            _state: PhantomData,
        }
    }

    /// Set include_bias parameter
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Set interaction_only parameter
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Set tensor ordering
    pub fn tensor_ordering(mut self, ordering: TensorOrdering) -> Self {
        self.tensor_ordering = ordering;
        self
    }

    /// Set contraction method
    pub fn contraction_method(mut self, method: ContractionMethod) -> Self {
        self.contraction_method = method;
        self
    }
}

impl Estimator for TensorPolynomialFeatures<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for TensorPolynomialFeatures<Untrained> {
    type Fitted = TensorPolynomialFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "degree must be positive".to_string(),
            ));
        }

        if self.n_dimensions == 0 {
            return Err(SklearsError::InvalidInput(
                "n_dimensions must be positive".to_string(),
            ));
        }

        // Generate tensor indices
        let tensor_indices = self.generate_tensor_indices(n_features)?;

        // Apply contraction if specified
        let (final_indices, contraction_map) = self.apply_contraction(&tensor_indices)?;

        let n_output_features = final_indices.len();

        Ok(TensorPolynomialFeatures {
            degree: self.degree,
            n_dimensions: self.n_dimensions,
            include_bias: self.include_bias,
            interaction_only: self.interaction_only,
            tensor_ordering: self.tensor_ordering,
            contraction_method: self.contraction_method,
            n_input_features_: Some(n_features),
            n_output_features_: Some(n_output_features),
            tensor_indices_: Some(final_indices),
            contraction_map_: Some(contraction_map),
            _state: PhantomData,
        })
    }
}

impl TensorPolynomialFeatures<Untrained> {
    /// Generate tensor indices for all combinations
    fn generate_tensor_indices(&self, n_features: usize) -> Result<Vec<Vec<Vec<u32>>>> {
        let mut tensor_indices = Vec::new();

        // Add bias term if requested
        if self.include_bias {
            let bias_tensor = vec![vec![0; n_features]; self.n_dimensions];
            tensor_indices.push(bias_tensor);
        }

        // Generate all tensor combinations up to degree
        for total_degree in 1..=self.degree {
            let mut degree_indices =
                self.generate_tensor_combinations_with_degree(n_features, total_degree);

            // Apply ordering
            self.apply_tensor_ordering(&mut degree_indices);

            tensor_indices.extend(degree_indices);
        }

        Ok(tensor_indices)
    }

    /// Generate tensor combinations for a specific total degree
    fn generate_tensor_combinations_with_degree(
        &self,
        n_features: usize,
        total_degree: u32,
    ) -> Vec<Vec<Vec<u32>>> {
        let mut combinations = Vec::new();

        // For each dimension, generate all combinations
        let mut current_tensor = vec![vec![0; n_features]; self.n_dimensions];
        self.generate_recursive_tensor_combinations(
            n_features,
            total_degree,
            0, // dimension index
            0, // feature index
            &mut current_tensor,
            &mut combinations,
        );

        // Filter based on interaction_only setting
        if self.interaction_only {
            combinations.retain(|tensor| self.is_valid_tensor_for_interaction_only(tensor));
        }

        combinations
    }

    /// Recursively generate tensor combinations
    fn generate_recursive_tensor_combinations(
        &self,
        n_features: usize,
        remaining_degree: u32,
        dim_idx: usize,
        feature_idx: usize,
        current_tensor: &mut Vec<Vec<u32>>,
        combinations: &mut Vec<Vec<Vec<u32>>>,
    ) {
        if dim_idx >= self.n_dimensions {
            // Check if we've used all the degree
            let total_degree: u32 = current_tensor
                .iter()
                .map(|dim| dim.iter().sum::<u32>())
                .sum();

            if total_degree == self.degree {
                combinations.push(current_tensor.clone());
            }
            return;
        }

        if feature_idx >= n_features {
            // Move to next dimension
            self.generate_recursive_tensor_combinations(
                n_features,
                remaining_degree,
                dim_idx + 1,
                0,
                current_tensor,
                combinations,
            );
            return;
        }

        // Current degree sum for this dimension
        let current_dim_degree: u32 = current_tensor[dim_idx].iter().sum();
        let max_power = remaining_degree.min(self.degree - current_dim_degree);

        // Try different powers for current feature in current dimension
        for power in 0..=max_power {
            current_tensor[dim_idx][feature_idx] = power;

            self.generate_recursive_tensor_combinations(
                n_features,
                remaining_degree,
                dim_idx,
                feature_idx + 1,
                current_tensor,
                combinations,
            );
        }

        current_tensor[dim_idx][feature_idx] = 0;
    }

    /// Check if tensor is valid for interaction_only mode
    fn is_valid_tensor_for_interaction_only(&self, tensor: &[Vec<u32>]) -> bool {
        for dimension in tensor {
            let non_zero_count = dimension.iter().filter(|&&p| p > 0).count();
            let max_power = dimension.iter().max().unwrap_or(&0);

            if non_zero_count == 1 {
                // Single variable: valid only if power is 1
                if *max_power != 1 {
                    return false;
                }
            } else if non_zero_count > 1 {
                // Multiple variables: valid only if all powers are 1
                if *max_power != 1 {
                    return false;
                }
            }
        }
        true
    }

    /// Apply tensor ordering to indices
    fn apply_tensor_ordering(&self, indices: &mut Vec<Vec<Vec<u32>>>) {
        match self.tensor_ordering {
            TensorOrdering::Lexicographic => {
                indices.sort_by(|a, b| {
                    for (dim_a, dim_b) in a.iter().zip(b.iter()) {
                        for (pow_a, pow_b) in dim_a.iter().zip(dim_b.iter()) {
                            match pow_a.cmp(pow_b) {
                                std::cmp::Ordering::Equal => continue,
                                other => return other,
                            }
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }
            TensorOrdering::GradedLexicographic => {
                indices.sort_by(|a, b| {
                    let degree_a: u32 = a.iter().map(|dim| dim.iter().sum::<u32>()).sum();
                    let degree_b: u32 = b.iter().map(|dim| dim.iter().sum::<u32>()).sum();

                    match degree_a.cmp(&degree_b) {
                        std::cmp::Ordering::Equal => {
                            // Same degree, use lexicographic
                            for (dim_a, dim_b) in a.iter().zip(b.iter()) {
                                for (pow_a, pow_b) in dim_a.iter().zip(dim_b.iter()) {
                                    match pow_a.cmp(pow_b) {
                                        std::cmp::Ordering::Equal => continue,
                                        other => return other,
                                    }
                                }
                            }
                            std::cmp::Ordering::Equal
                        }
                        other => other,
                    }
                });
            }
            TensorOrdering::ReversedGradedLexicographic => {
                indices.sort_by(|a, b| {
                    let degree_a: u32 = a.iter().map(|dim| dim.iter().sum::<u32>()).sum();
                    let degree_b: u32 = b.iter().map(|dim| dim.iter().sum::<u32>()).sum();

                    match degree_a.cmp(&degree_b) {
                        std::cmp::Ordering::Equal => {
                            // Same degree, use reverse lexicographic
                            for (dim_a, dim_b) in a.iter().zip(b.iter()).rev() {
                                for (pow_a, pow_b) in dim_a.iter().zip(dim_b.iter()).rev() {
                                    match pow_b.cmp(pow_a) {
                                        std::cmp::Ordering::Equal => continue,
                                        other => return other,
                                    }
                                }
                            }
                            std::cmp::Ordering::Equal
                        }
                        other => other,
                    }
                });
            }
        }
    }

    /// Apply tensor contraction method
    fn apply_contraction(
        &self,
        tensor_indices: &[Vec<Vec<u32>>],
    ) -> Result<(Vec<Vec<Vec<u32>>>, Vec<Vec<usize>>)> {
        match &self.contraction_method {
            ContractionMethod::None => {
                let identity_map: Vec<Vec<usize>> =
                    (0..tensor_indices.len()).map(|i| vec![i]).collect();
                Ok((tensor_indices.to_vec(), identity_map))
            }
            ContractionMethod::Indices(indices) => {
                self.contract_by_indices(tensor_indices, indices)
            }
            ContractionMethod::Rank(target_rank) => {
                self.contract_by_rank(tensor_indices, *target_rank)
            }
            ContractionMethod::Symmetric => self.contract_symmetric(tensor_indices),
        }
    }

    /// Contract tensor by specific indices
    fn contract_by_indices(
        &self,
        tensor_indices: &[Vec<Vec<u32>>],
        contraction_indices: &[usize],
    ) -> Result<(Vec<Vec<Vec<u32>>>, Vec<Vec<usize>>)> {
        let mut contracted_indices = Vec::new();
        let mut contraction_map = Vec::new();

        for (i, tensor) in tensor_indices.iter().enumerate() {
            let mut contracted_tensor = tensor.clone();

            // Contract specified dimensions by summing them
            for &contract_idx in contraction_indices {
                if contract_idx < contracted_tensor.len() && contracted_tensor.len() > 1 {
                    if contract_idx + 1 < contracted_tensor.len() {
                        // Collect values to add before mutating
                        let values_to_add: Vec<(usize, u32)> = contracted_tensor[contract_idx]
                            .iter()
                            .enumerate()
                            .map(|(j, &val)| (j, val))
                            .collect();

                        // Add the contracted dimension to the next one
                        for (j, val) in values_to_add {
                            if j < contracted_tensor[contract_idx + 1].len() {
                                contracted_tensor[contract_idx + 1][j] += val;
                            }
                        }
                    }
                    contracted_tensor.remove(contract_idx);
                }
            }

            contracted_indices.push(contracted_tensor);
            contraction_map.push(vec![i]);
        }

        Ok((contracted_indices, contraction_map))
    }

    /// Contract tensor to specified rank
    fn contract_by_rank(
        &self,
        tensor_indices: &[Vec<Vec<u32>>],
        target_rank: usize,
    ) -> Result<(Vec<Vec<Vec<u32>>>, Vec<Vec<usize>>)> {
        if target_rank >= tensor_indices.len() {
            let identity_map: Vec<Vec<usize>> =
                (0..tensor_indices.len()).map(|i| vec![i]).collect();
            return Ok((tensor_indices.to_vec(), identity_map));
        }

        // Simple rank reduction: take first target_rank tensors
        let contracted_indices = tensor_indices[..target_rank].to_vec();
        let contraction_map: Vec<Vec<usize>> = (0..target_rank).map(|i| vec![i]).collect();

        Ok((contracted_indices, contraction_map))
    }

    /// Apply symmetric contraction
    fn contract_symmetric(
        &self,
        tensor_indices: &[Vec<Vec<u32>>],
    ) -> Result<(Vec<Vec<Vec<u32>>>, Vec<Vec<usize>>)> {
        let mut contracted_indices = Vec::new();
        let mut contraction_map = Vec::new();
        let mut used = vec![false; tensor_indices.len()];

        for i in 0..tensor_indices.len() {
            if used[i] {
                continue;
            }

            let mut symmetric_group = vec![i];
            used[i] = true;

            // Find symmetric tensors (same structure across dimensions)
            for j in (i + 1)..tensor_indices.len() {
                if used[j] {
                    continue;
                }

                if self.are_tensors_symmetric(&tensor_indices[i], &tensor_indices[j]) {
                    symmetric_group.push(j);
                    used[j] = true;
                }
            }

            // Create averaged tensor
            let mut averaged_tensor = tensor_indices[i].clone();
            for &group_idx in &symmetric_group[1..] {
                for (dim_idx, dimension) in tensor_indices[group_idx].iter().enumerate() {
                    for (feat_idx, &power) in dimension.iter().enumerate() {
                        if dim_idx < averaged_tensor.len()
                            && feat_idx < averaged_tensor[dim_idx].len()
                        {
                            averaged_tensor[dim_idx][feat_idx] += power;
                        }
                    }
                }
            }

            // Average the powers
            let group_size = symmetric_group.len() as u32;
            for dimension in &mut averaged_tensor {
                for power in dimension {
                    *power /= group_size;
                }
            }

            contracted_indices.push(averaged_tensor);
            contraction_map.push(symmetric_group);
        }

        Ok((contracted_indices, contraction_map))
    }

    /// Check if two tensors are symmetric
    fn are_tensors_symmetric(&self, tensor_a: &[Vec<u32>], tensor_b: &[Vec<u32>]) -> bool {
        if tensor_a.len() != tensor_b.len() {
            return false;
        }

        for (dim_a, dim_b) in tensor_a.iter().zip(tensor_b.iter()) {
            if dim_a.len() != dim_b.len() {
                return false;
            }

            let sum_a: u32 = dim_a.iter().sum();
            let sum_b: u32 = dim_b.iter().sum();

            if sum_a != sum_b {
                return false;
            }
        }

        true
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TensorPolynomialFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let n_input_features = self.n_input_features_.unwrap();
        let n_output_features = self.n_output_features_.unwrap();
        let tensor_indices = self.tensor_indices_.as_ref().unwrap();

        if n_features != n_input_features {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but TensorPolynomialFeatures was fitted with {} features",
                n_features, n_input_features
            )));
        }

        let mut result = Array2::zeros((n_samples, n_output_features));

        for i in 0..n_samples {
            for (j, tensor) in tensor_indices.iter().enumerate() {
                let feature_value = self.compute_tensor_feature_value(&x.row(i), tensor);
                result[[i, j]] = feature_value;
            }
        }

        Ok(result)
    }
}

impl TensorPolynomialFeatures<Trained> {
    /// Compute tensor feature value for a single sample
    fn compute_tensor_feature_value(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        tensor: &[Vec<u32>],
    ) -> Float {
        let mut tensor_value = 1.0;

        for dimension in tensor {
            let mut dim_value = 1.0;
            for (feature_idx, &power) in dimension.iter().enumerate() {
                if power > 0 && feature_idx < sample.len() {
                    dim_value *= sample[feature_idx].powi(power as i32);
                }
            }
            tensor_value *= dim_value;
        }

        tensor_value
    }

    /// Get the number of input features
    pub fn n_input_features(&self) -> usize {
        self.n_input_features_.unwrap()
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_.unwrap()
    }

    /// Get the tensor indices
    pub fn tensor_indices(&self) -> &[Vec<Vec<u32>>] {
        self.tensor_indices_.as_ref().unwrap()
    }

    /// Get the contraction map
    pub fn contraction_map(&self) -> &[Vec<usize>] {
        self.contraction_map_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_tensor_polynomial_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let tensor_poly = TensorPolynomialFeatures::new(2, 2);
        let fitted = tensor_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        assert!(x_transformed.ncols() > 0);
    }

    #[test]
    fn test_tensor_polynomial_no_bias() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let tensor_poly = TensorPolynomialFeatures::new(2, 2).include_bias(false);
        let fitted = tensor_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        assert!(x_transformed.ncols() > 0);
    }

    #[test]
    fn test_tensor_polynomial_interaction_only() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let tensor_poly = TensorPolynomialFeatures::new(2, 2).interaction_only(true);
        let fitted = tensor_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        assert!(x_transformed.ncols() > 0);
    }

    #[test]
    fn test_tensor_polynomial_different_orderings() {
        let x = array![[1.0, 2.0]];

        let orderings = vec![
            TensorOrdering::Lexicographic,
            TensorOrdering::GradedLexicographic,
            TensorOrdering::ReversedGradedLexicographic,
        ];

        for ordering in orderings {
            let tensor_poly = TensorPolynomialFeatures::new(2, 2).tensor_ordering(ordering);
            let fitted = tensor_poly.fit(&x, &()).unwrap();
            let x_transformed = fitted.transform(&x).unwrap();

            assert_eq!(x_transformed.nrows(), 1);
            assert!(x_transformed.ncols() > 0);
        }
    }

    #[test]
    fn test_tensor_polynomial_contraction_methods() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let methods = vec![
            ContractionMethod::None,
            ContractionMethod::Rank(5),
            ContractionMethod::Symmetric,
        ];

        for method in methods {
            let tensor_poly = TensorPolynomialFeatures::new(2, 3).contraction_method(method);
            let fitted = tensor_poly.fit(&x, &()).unwrap();
            let x_transformed = fitted.transform(&x).unwrap();

            assert_eq!(x_transformed.nrows(), 2);
            assert!(x_transformed.ncols() > 0);
        }
    }

    #[test]
    fn test_tensor_polynomial_different_dimensions() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        for n_dims in 1..=4 {
            let tensor_poly = TensorPolynomialFeatures::new(2, n_dims);
            let fitted = tensor_poly.fit(&x, &()).unwrap();
            let x_transformed = fitted.transform(&x).unwrap();

            assert_eq!(x_transformed.nrows(), 2);
            assert!(x_transformed.ncols() > 0);
        }
    }

    #[test]
    fn test_tensor_polynomial_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Different number of features

        let tensor_poly = TensorPolynomialFeatures::new(2, 2);
        let fitted = tensor_poly.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_polynomial_zero_degree() {
        let x = array![[1.0, 2.0]];
        let tensor_poly = TensorPolynomialFeatures::new(0, 2);
        let result = tensor_poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_polynomial_zero_dimensions() {
        let x = array![[1.0, 2.0]];
        let tensor_poly = TensorPolynomialFeatures::new(2, 0);
        let result = tensor_poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_polynomial_single_feature() {
        let x = array![[2.0], [3.0]];

        let tensor_poly = TensorPolynomialFeatures::new(3, 2);
        let fitted = tensor_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        assert!(x_transformed.ncols() > 0);
    }

    #[test]
    fn test_tensor_polynomial_contraction_map() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let tensor_poly =
            TensorPolynomialFeatures::new(2, 2).contraction_method(ContractionMethod::Symmetric);
        let fitted = tensor_poly.fit(&x, &()).unwrap();

        let contraction_map = fitted.contraction_map();
        assert!(!contraction_map.is_empty());

        // Each group in the contraction map should contain valid indices
        // The contraction map length should match the number of output features
        assert_eq!(contraction_map.len(), fitted.n_output_features());

        // Each group should be non-empty
        for group in contraction_map {
            assert!(!group.is_empty());
        }
    }
}
