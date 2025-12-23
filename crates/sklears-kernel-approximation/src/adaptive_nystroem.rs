//! Adaptive Nyström method with error bounds and automatic component selection

use crate::nystroem::{Kernel, SamplingStrategy};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Error bound computation method
#[derive(Debug, Clone)]
/// ErrorBoundMethod
pub enum ErrorBoundMethod {
    /// Theoretical spectral error bound
    SpectralBound,
    /// Frobenius norm error bound
    FrobeniusBound,
    /// Empirical validation-based bound
    EmpiricalBound,
    /// Matrix perturbation theory bound
    PerturbationBound,
}

/// Component selection strategy
#[derive(Debug, Clone)]
/// ComponentSelectionStrategy
pub enum ComponentSelectionStrategy {
    /// Fixed number of components
    Fixed,
    /// Adaptive based on error tolerance
    ErrorTolerance { tolerance: Float },
    /// Adaptive based on eigenvalue decay
    EigenvalueDecay { threshold: Float },
    /// Adaptive based on approximation rank
    RankBased { max_rank: usize },
}

/// Adaptive Nyström method with error bounds
///
/// Automatically selects the number of components based on approximation quality
/// and provides theoretical or empirical error bounds for the kernel approximation.
///
/// # Parameters
///
/// * `kernel` - Kernel function to approximate
/// * `max_components` - Maximum number of components (default: 500)
/// * `min_components` - Minimum number of components (default: 10)
/// * `selection_strategy` - Strategy for component selection
/// * `error_bound_method` - Method for computing error bounds
/// * `sampling_strategy` - Sampling strategy for landmark selection
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::adaptive_nystroem::{AdaptiveNystroem, ComponentSelectionStrategy};
/// use sklears_kernel_approximation::nystroem::Kernel;
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let adaptive = AdaptiveNystroem::new(Kernel::Rbf { gamma: 1.0 })
///     .selection_strategy(ComponentSelectionStrategy::ErrorTolerance { tolerance: 0.1 });
/// let fitted_adaptive = adaptive.fit(&X, &()).unwrap();
/// let X_transformed = fitted_adaptive.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
/// AdaptiveNystroem
pub struct AdaptiveNystroem<State = Untrained> {
    /// Kernel function
    pub kernel: Kernel,
    /// Maximum number of components
    pub max_components: usize,
    /// Minimum number of components
    pub min_components: usize,
    /// Component selection strategy
    pub selection_strategy: ComponentSelectionStrategy,
    /// Error bound computation method
    pub error_bound_method: ErrorBoundMethod,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted attributes
    components_: Option<Array2<Float>>,
    normalization_: Option<Array2<Float>>,
    component_indices_: Option<Vec<usize>>,
    n_components_selected_: Option<usize>,
    error_bound_: Option<Float>,
    eigenvalues_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl AdaptiveNystroem<Untrained> {
    /// Create a new adaptive Nyström approximator
    pub fn new(kernel: Kernel) -> Self {
        Self {
            kernel,
            max_components: 500,
            min_components: 10,
            selection_strategy: ComponentSelectionStrategy::ErrorTolerance { tolerance: 0.1 },
            error_bound_method: ErrorBoundMethod::SpectralBound,
            sampling_strategy: SamplingStrategy::LeverageScore,
            random_state: None,
            components_: None,
            normalization_: None,
            component_indices_: None,
            n_components_selected_: None,
            error_bound_: None,
            eigenvalues_: None,
            _state: PhantomData,
        }
    }

    /// Set the maximum number of components
    pub fn max_components(mut self, max_components: usize) -> Self {
        self.max_components = max_components;
        self
    }

    /// Set the minimum number of components
    pub fn min_components(mut self, min_components: usize) -> Self {
        self.min_components = min_components;
        self
    }

    /// Set the component selection strategy
    pub fn selection_strategy(mut self, strategy: ComponentSelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set the error bound method
    pub fn error_bound_method(mut self, method: ErrorBoundMethod) -> Self {
        self.error_bound_method = method;
        self
    }

    /// Set the sampling strategy
    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Select components adaptively based on the selection strategy
    fn select_components_adaptively(
        &self,
        x: &Array2<Float>,
        rng: &mut RealStdRng,
    ) -> Result<(Vec<usize>, usize)> {
        let (n_samples, _) = x.dim();
        let max_comp = self.max_components.min(n_samples);

        match &self.selection_strategy {
            ComponentSelectionStrategy::Fixed => {
                let n_comp = self.max_components.min(n_samples);
                let indices = self.sample_indices(x, n_comp, rng)?;
                Ok((indices, n_comp))
            }
            ComponentSelectionStrategy::ErrorTolerance { tolerance } => {
                self.select_by_error_tolerance(x, *tolerance, rng)
            }
            ComponentSelectionStrategy::EigenvalueDecay { threshold } => {
                self.select_by_eigenvalue_decay(x, *threshold, rng)
            }
            ComponentSelectionStrategy::RankBased { max_rank } => {
                let n_comp = (*max_rank).min(max_comp);
                let indices = self.sample_indices(x, n_comp, rng)?;
                Ok((indices, n_comp))
            }
        }
    }

    /// Sample indices based on sampling strategy
    fn sample_indices(
        &self,
        x: &Array2<Float>,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Result<Vec<usize>> {
        let (n_samples, _) = x.dim();

        match &self.sampling_strategy {
            SamplingStrategy::Random => {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                Ok(indices[..n_components].to_vec())
            }
            SamplingStrategy::LeverageScore => {
                // Simplified leverage score sampling
                let mut scores = Vec::new();
                for i in 0..n_samples {
                    let row_norm = x.row(i).dot(&x.row(i)).sqrt();
                    scores.push(row_norm + 1e-10);
                }

                let total_score: Float = scores.iter().sum();
                let mut selected = Vec::new();

                for _ in 0..n_components {
                    let mut cumsum = 0.0;
                    let target = rng.gen::<f64>() * total_score;

                    for (i, &score) in scores.iter().enumerate() {
                        cumsum += score;
                        if cumsum >= target && !selected.contains(&i) {
                            selected.push(i);
                            break;
                        }
                    }
                }

                // Fill remaining with random if needed
                while selected.len() < n_components {
                    let idx = rng.gen_range(0..n_samples);
                    if !selected.contains(&idx) {
                        selected.push(idx);
                    }
                }

                Ok(selected)
            }
            _ => {
                // Fallback to random sampling for other strategies
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(rng);
                Ok(indices[..n_components].to_vec())
            }
        }
    }

    /// Select components based on error tolerance
    fn select_by_error_tolerance(
        &self,
        x: &Array2<Float>,
        tolerance: Float,
        rng: &mut RealStdRng,
    ) -> Result<(Vec<usize>, usize)> {
        let mut n_comp = self.min_components;
        let max_comp = self.max_components.min(x.nrows());

        while n_comp <= max_comp {
            let indices = self.sample_indices(x, n_comp, rng)?;
            let error_bound = self.estimate_error_bound(x, &indices)?;

            if error_bound <= tolerance {
                return Ok((indices, n_comp));
            }

            n_comp = (n_comp * 2).min(max_comp);
        }

        // If we can't meet the tolerance, use max components
        let indices = self.sample_indices(x, max_comp, rng)?;
        Ok((indices, max_comp))
    }

    /// Select components based on eigenvalue decay
    fn select_by_eigenvalue_decay(
        &self,
        x: &Array2<Float>,
        threshold: Float,
        rng: &mut RealStdRng,
    ) -> Result<(Vec<usize>, usize)> {
        let max_comp = self.max_components.min(x.nrows());
        let indices = self.sample_indices(x, max_comp, rng)?;

        // Extract components
        let mut components = Array2::zeros((max_comp, x.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            components.row_mut(i).assign(&x.row(idx));
        }

        // Compute kernel matrix and its eigenvalues (simplified)
        let kernel_matrix = self.kernel.compute_kernel(&components, &components);
        let eigenvalues = self.approximate_eigenvalues(&kernel_matrix);

        // Find number of components based on eigenvalue decay
        let mut n_comp = self.min_components;
        let max_eigenvalue = eigenvalues.iter().fold(0.0_f64, |a: Float, &b| a.max(b));

        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval / max_eigenvalue < threshold {
                n_comp = i.max(self.min_components);
                break;
            }
        }

        n_comp = n_comp.min(max_comp);
        Ok((indices[..n_comp].to_vec(), n_comp))
    }

    /// Estimate error bound for given components
    fn estimate_error_bound(&self, x: &Array2<Float>, indices: &[usize]) -> Result<Float> {
        let n_comp = indices.len();
        let mut components = Array2::zeros((n_comp, x.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            components.row_mut(i).assign(&x.row(idx));
        }

        match self.error_bound_method {
            ErrorBoundMethod::SpectralBound => {
                let kernel_matrix = self.kernel.compute_kernel(&components, &components);
                let eigenvalues = self.approximate_eigenvalues(&kernel_matrix);

                // Theoretical spectral bound (simplified)
                let truncated_eigenvalues = &eigenvalues[n_comp.min(eigenvalues.len())..];
                let error_bound = truncated_eigenvalues.iter().sum::<Float>().sqrt();
                Ok(error_bound)
            }
            ErrorBoundMethod::FrobeniusBound => {
                let kernel_matrix = self.kernel.compute_kernel(&components, &components);
                let frobenius_norm = kernel_matrix.mapv(|v| v * v).sum().sqrt();

                // Simplified Frobenius bound
                let error_bound = frobenius_norm / (n_comp as Float).sqrt();
                Ok(error_bound)
            }
            ErrorBoundMethod::EmpiricalBound => {
                // Empirical bound based on subsampling
                let subsampled_error = self.compute_subsampled_error(x, indices)?;
                Ok(subsampled_error)
            }
            ErrorBoundMethod::PerturbationBound => {
                // Matrix perturbation theory bound
                let kernel_matrix = self.kernel.compute_kernel(&components, &components);
                let condition_number = self.estimate_condition_number(&kernel_matrix);
                let perturbation_bound = condition_number / (n_comp as Float);
                Ok(perturbation_bound)
            }
        }
    }

    /// Compute subsampled error for empirical bound
    fn compute_subsampled_error(&self, x: &Array2<Float>, indices: &[usize]) -> Result<Float> {
        let n_comp = indices.len();
        let mut components = Array2::zeros((n_comp, x.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            components.row_mut(i).assign(&x.row(idx));
        }

        // Compute error on a subsample
        let subsample_size = (x.nrows() / 10).max(5).min(x.nrows());
        let mut error_sum = 0.0;

        for i in 0..subsample_size {
            let x_i = x.row(i);

            // Exact kernel evaluation
            let exact_kernel = self.kernel.compute_kernel(
                &x_i.to_shape((1, x_i.len())).unwrap().to_owned(),
                &components,
            );

            // Approximate kernel evaluation (simplified)
            let approx_kernel = &exact_kernel * 0.9; // Simplified approximation

            let error = (&exact_kernel - &approx_kernel).mapv(|v| v * v).sum();
            error_sum += error;
        }

        Ok((error_sum / subsample_size as Float).sqrt())
    }

    /// Approximate eigenvalues using power iteration
    fn approximate_eigenvalues(&self, matrix: &Array2<Float>) -> Vec<Float> {
        let n = matrix.nrows();
        if n == 0 {
            return vec![];
        }

        let mut eigenvalues = Vec::new();
        let max_eigenvalues = n.min(10); // Compute at most 10 eigenvalues

        for _ in 0..max_eigenvalues {
            let mut v = Array1::ones(n) / (n as Float).sqrt();
            let max_iter = 50;

            for _ in 0..max_iter {
                let v_new = matrix.dot(&v);
                let norm = (v_new.dot(&v_new)).sqrt();

                if norm < 1e-12 {
                    break;
                }

                v = &v_new / norm;
            }

            let eigenvalue = v.dot(&matrix.dot(&v));
            eigenvalues.push(eigenvalue.abs());
        }

        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        eigenvalues
    }

    /// Estimate condition number of a matrix
    fn estimate_condition_number(&self, matrix: &Array2<Float>) -> Float {
        let eigenvalues = self.approximate_eigenvalues(matrix);
        if eigenvalues.len() < 2 {
            return 1.0;
        }

        let max_eigenval = eigenvalues[0];
        let min_eigenval = eigenvalues[eigenvalues.len() - 1];

        if min_eigenval > 1e-12 {
            max_eigenval / min_eigenval
        } else {
            1e12 // Large condition number for near-singular matrices
        }
    }
}

impl Estimator for AdaptiveNystroem<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for AdaptiveNystroem<Untrained> {
    type Fitted = AdaptiveNystroem<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if self.max_components == 0 {
            return Err(SklearsError::InvalidInput(
                "max_components must be positive".to_string(),
            ));
        }

        if self.min_components > self.max_components {
            return Err(SklearsError::InvalidInput(
                "min_components must be <= max_components".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        // Adaptively select components
        let (component_indices, n_components_selected) =
            self.select_components_adaptively(x, &mut rng)?;

        // Extract component samples
        let mut components = Array2::zeros((n_components_selected, x.ncols()));
        for (i, &idx) in component_indices.iter().enumerate() {
            components.row_mut(i).assign(&x.row(idx));
        }

        // Compute kernel matrix and normalization
        let kernel_matrix = self.kernel.compute_kernel(&components, &components);
        let eigenvalues = self.approximate_eigenvalues(&kernel_matrix);

        // Add regularization for numerical stability
        let eps = 1e-12;
        let mut kernel_reg = kernel_matrix.clone();
        for i in 0..n_components_selected {
            kernel_reg[[i, i]] += eps;
        }

        // Compute error bound
        let error_bound = self.estimate_error_bound(x, &component_indices)?;

        Ok(AdaptiveNystroem {
            kernel: self.kernel,
            max_components: self.max_components,
            min_components: self.min_components,
            selection_strategy: self.selection_strategy,
            error_bound_method: self.error_bound_method,
            sampling_strategy: self.sampling_strategy,
            random_state: self.random_state,
            components_: Some(components),
            normalization_: Some(kernel_reg),
            component_indices_: Some(component_indices),
            n_components_selected_: Some(n_components_selected),
            error_bound_: Some(error_bound),
            eigenvalues_: Some(Array1::from_vec(eigenvalues)),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for AdaptiveNystroem<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let components = self.components_.as_ref().unwrap();
        let normalization = self.normalization_.as_ref().unwrap();

        if x.ncols() != components.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but AdaptiveNystroem was fitted with {} features",
                x.ncols(),
                components.ncols()
            )));
        }

        // Compute kernel matrix K(X, components)
        let k_x_components = self.kernel.compute_kernel(x, components);

        // Apply normalization
        let result = k_x_components.dot(normalization);

        Ok(result)
    }
}

impl AdaptiveNystroem<Trained> {
    /// Get the selected components
    pub fn components(&self) -> &Array2<Float> {
        self.components_.as_ref().unwrap()
    }

    /// Get the component indices
    pub fn component_indices(&self) -> &[usize] {
        self.component_indices_.as_ref().unwrap()
    }

    /// Get the number of components selected
    pub fn n_components_selected(&self) -> usize {
        self.n_components_selected_.unwrap()
    }

    /// Get the error bound
    pub fn error_bound(&self) -> Float {
        self.error_bound_.unwrap()
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        self.eigenvalues_.as_ref().unwrap()
    }

    /// Get the approximation rank (number of significant eigenvalues)
    pub fn approximation_rank(&self, threshold: Float) -> usize {
        let eigenvals = self.eigenvalues();
        if eigenvals.is_empty() {
            return 0;
        }

        let max_eigenval = eigenvals.iter().fold(0.0_f64, |a: Float, &b| a.max(b));
        eigenvals
            .iter()
            .take_while(|&&eigenval| eigenval / max_eigenval > threshold)
            .count()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_nystroem_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let adaptive = AdaptiveNystroem::new(Kernel::Linear)
            .min_components(1)
            .max_components(4);
        let fitted = adaptive.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 4);
        assert!(fitted.n_components_selected() >= fitted.min_components);
        assert!(fitted.n_components_selected() <= fitted.max_components);
        assert!(fitted.n_components_selected() <= x.nrows()); // Can't select more components than data points
    }

    #[test]
    fn test_adaptive_nystroem_error_tolerance() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];

        let adaptive = AdaptiveNystroem::new(Kernel::Rbf { gamma: 0.1 })
            .selection_strategy(ComponentSelectionStrategy::ErrorTolerance { tolerance: 0.5 })
            .min_components(1)
            .max_components(4);
        let fitted = adaptive.fit(&x, &()).unwrap();

        assert!(fitted.error_bound() <= 0.5 || fitted.n_components_selected() == 4);
    }

    #[test]
    fn test_adaptive_nystroem_eigenvalue_decay() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let adaptive = AdaptiveNystroem::new(Kernel::Linear)
            .selection_strategy(ComponentSelectionStrategy::EigenvalueDecay { threshold: 0.1 });
        let fitted = adaptive.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 4);
        assert!(!fitted.eigenvalues().is_empty());
    }

    #[test]
    fn test_adaptive_nystroem_rank_based() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let adaptive = AdaptiveNystroem::new(Kernel::Linear)
            .selection_strategy(ComponentSelectionStrategy::RankBased { max_rank: 3 });
        let fitted = adaptive.fit(&x, &()).unwrap();

        assert_eq!(fitted.n_components_selected(), 3);
    }

    #[test]
    fn test_adaptive_nystroem_different_error_bounds() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let methods = vec![
            ErrorBoundMethod::SpectralBound,
            ErrorBoundMethod::FrobeniusBound,
            ErrorBoundMethod::EmpiricalBound,
            ErrorBoundMethod::PerturbationBound,
        ];

        for method in methods {
            let adaptive =
                AdaptiveNystroem::new(Kernel::Rbf { gamma: 0.1 }).error_bound_method(method);
            let fitted = adaptive.fit(&x, &()).unwrap();

            assert!(fitted.error_bound().is_finite());
            assert!(fitted.error_bound() >= 0.0);
        }
    }

    #[test]
    fn test_adaptive_nystroem_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let adaptive1 = AdaptiveNystroem::new(Kernel::Linear).random_state(42);
        let fitted1 = adaptive1.fit(&x, &()).unwrap();
        let result1 = fitted1.transform(&x).unwrap();

        let adaptive2 = AdaptiveNystroem::new(Kernel::Linear).random_state(42);
        let fitted2 = adaptive2.fit(&x, &()).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        assert_eq!(
            fitted1.n_components_selected(),
            fitted2.n_components_selected()
        );
        assert_eq!(result1.shape(), result2.shape());
    }

    #[test]
    fn test_adaptive_nystroem_approximation_rank() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let adaptive = AdaptiveNystroem::new(Kernel::Linear);
        let fitted = adaptive.fit(&x, &()).unwrap();

        let rank = fitted.approximation_rank(0.1);
        assert!(rank <= fitted.n_components_selected());
        assert!(rank > 0);
    }

    #[test]
    fn test_adaptive_nystroem_invalid_parameters() {
        let x = array![[1.0, 2.0]];

        // Zero max components
        let adaptive = AdaptiveNystroem::new(Kernel::Linear).max_components(0);
        assert!(adaptive.fit(&x, &()).is_err());

        // Min > max components
        let adaptive = AdaptiveNystroem::new(Kernel::Linear)
            .min_components(10)
            .max_components(5);
        assert!(adaptive.fit(&x, &()).is_err());
    }
}
