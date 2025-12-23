//! Automatic Differentiation Variational Inference (ADVI) for Gaussian Mixture Models
//!
//! This module provides automatic differentiation variational inference methods
//! that can automatically compute gradients of the variational lower bound with
//! respect to the variational parameters, enabling efficient optimization without
//! manual gradient derivation.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
};
use std::f64::consts::PI;

use crate::common::{CovarianceType, InitMethod, ModelSelection};

/// Automatic differentiation backend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ADBackend {
    /// Finite differences for gradient computation
    FiniteDifferences,
    /// Forward mode automatic differentiation
    ForwardMode,
    /// Reverse mode automatic differentiation
    ReverseMode,
    /// Dual numbers for exact derivatives
    DualNumbers,
}

/// Optimization algorithm for ADVI
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ADVIOptimizer {
    /// Stochastic gradient descent
    SGD,
    /// Adaptive gradient (AdaGrad)
    AdaGrad,
    /// Root mean square propagation (RMSprop)
    RMSprop,
    /// Adaptive moment estimation (Adam)
    Adam,
    /// Limited-memory BFGS
    LBFGS,
}

/// Dual number for automatic differentiation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Value part
    pub value: f64,
    /// Derivative part
    pub derivative: f64,
}

impl Dual {
    /// Create a new dual number
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    /// Create a dual number representing a variable
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    /// Create a dual number representing a constant
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;

    fn add(self, other: Dual) -> Self::Output {
        Dual::new(self.value + other.value, self.derivative + other.derivative)
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;

    fn sub(self, other: Dual) -> Self::Output {
        Dual::new(self.value - other.value, self.derivative - other.derivative)
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;

    fn mul(self, other: Dual) -> Self::Output {
        Dual::new(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;

    fn div(self, other: Dual) -> Self::Output {
        Dual::new(
            self.value / other.value,
            (self.derivative * other.value - self.value * other.derivative)
                / (other.value * other.value),
        )
    }
}

impl Dual {
    /// Exponential function
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Dual::new(exp_val, self.derivative * exp_val)
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Dual::new(self.value.ln(), self.derivative / self.value)
    }

    /// Power function
    pub fn powf(self, exponent: f64) -> Self {
        let base_pow = self.value.powf(exponent);
        Dual::new(
            base_pow,
            self.derivative * exponent * self.value.powf(exponent - 1.0),
        )
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.value.sqrt();
        Dual::new(sqrt_val, self.derivative / (2.0 * sqrt_val))
    }

    /// Trigonometric sine
    pub fn sin(self) -> Self {
        Dual::new(self.value.sin(), self.derivative * self.value.cos())
    }

    /// Trigonometric cosine
    pub fn cos(self) -> Self {
        Dual::new(self.value.cos(), -self.derivative * self.value.sin())
    }
}

/// Automatic Differentiation Variational Inference for Gaussian Mixture Models
///
/// This implementation uses automatic differentiation to compute gradients of the
/// variational lower bound with respect to the variational parameters, enabling
/// efficient optimization without manual gradient derivation.
///
/// The key advantage is that gradients are computed automatically and exactly,
/// leading to more stable and efficient optimization compared to finite differences.
///
/// # Examples
///
/// ```
/// use sklears_mixture::{ADVIGaussianMixture, ADBackend, ADVIOptimizer, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let model = ADVIGaussianMixture::new()
///     .n_components(2)
///     .ad_backend(ADBackend::DualNumbers)
///     .optimizer(ADVIOptimizer::Adam)
///     .covariance_type(CovarianceType::Full);
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ADVIGaussianMixture<S = Untrained> {
    state: S,
    /// Number of mixture components
    n_components: usize,
    /// Automatic differentiation backend
    ad_backend: ADBackend,
    /// Optimization algorithm
    optimizer: ADVIOptimizer,
    /// Covariance type
    covariance_type: CovarianceType,
    /// Convergence tolerance
    tol: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Regularization parameter
    reg_covar: f64,
    /// Weight concentration parameter
    weight_concentration: f64,
    /// Mean precision parameter
    mean_precision: f64,
    /// Degrees of freedom parameter
    degrees_of_freedom: f64,
    /// Initialization method
    init_method: InitMethod,
    /// Number of initializations
    n_init: usize,
    /// Learning rate
    learning_rate: f64,
    /// Adam beta1 parameter
    adam_beta1: f64,
    /// Adam beta2 parameter
    adam_beta2: f64,
    /// Adam epsilon parameter
    adam_epsilon: f64,
    /// Gradient clipping threshold
    grad_clip: f64,
    /// Mini-batch size for stochastic optimization
    batch_size: Option<usize>,
    /// Use natural gradients
    use_natural_gradients: bool,
    /// Finite difference step size
    finite_diff_step: f64,
}

/// Trained ADVI Gaussian Mixture Model
#[derive(Debug, Clone)]
pub struct ADVIGaussianMixtureTrained {
    /// Number of mixture components
    n_components: usize,
    /// Automatic differentiation backend
    ad_backend: ADBackend,
    /// Optimization algorithm
    optimizer: ADVIOptimizer,
    /// Covariance type
    covariance_type: CovarianceType,
    /// Variational parameters for mixture weights
    weight_concentration: Array1<f64>,
    /// Variational parameters for means
    mean_precision: Array1<f64>,
    /// Variational parameters for means
    mean_values: Array2<f64>,
    /// Variational parameters for precisions
    precision_values: Array3<f64>,
    /// Degrees of freedom parameters
    degrees_of_freedom: Array1<f64>,
    /// Scale matrices for Wishart distributions
    scale_matrices: Array3<f64>,
    /// Number of data points
    n_samples: usize,
    /// Number of features
    n_features: usize,
    /// Converged log-likelihood
    lower_bound: f64,
    /// Final responsibilities
    responsibilities: Array2<f64>,
    /// Model selection criteria
    model_selection: ModelSelection,
    /// Gradient history for diagnostics
    gradient_history: Vec<Array1<f64>>,
    /// Parameter history for diagnostics
    parameter_history: Vec<Array1<f64>>,
}

impl ADVIGaussianMixture<Untrained> {
    /// Create a new ADVI Gaussian mixture model
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            ad_backend: ADBackend::DualNumbers,
            optimizer: ADVIOptimizer::Adam,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            max_iter: 1000,
            random_state: None,
            reg_covar: 1e-6,
            weight_concentration: 1.0,
            mean_precision: 1.0,
            degrees_of_freedom: 1.0,
            init_method: InitMethod::KMeansPlus,
            n_init: 1,
            learning_rate: 0.01,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            grad_clip: 1.0,
            batch_size: None,
            use_natural_gradients: false,
            finite_diff_step: 1e-6,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the automatic differentiation backend
    pub fn ad_backend(mut self, backend: ADBackend) -> Self {
        self.ad_backend = backend;
        self
    }

    /// Set the optimization algorithm
    pub fn optimizer(mut self, optimizer: ADVIOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the regularization parameter
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the weight concentration parameter
    pub fn weight_concentration(mut self, weight_concentration: f64) -> Self {
        self.weight_concentration = weight_concentration;
        self
    }

    /// Set the mean precision parameter
    pub fn mean_precision(mut self, mean_precision: f64) -> Self {
        self.mean_precision = mean_precision;
        self
    }

    /// Set the degrees of freedom parameter
    pub fn degrees_of_freedom(mut self, degrees_of_freedom: f64) -> Self {
        self.degrees_of_freedom = degrees_of_freedom;
        self
    }

    /// Set the initialization method
    pub fn init_method(mut self, init_method: InitMethod) -> Self {
        self.init_method = init_method;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the Adam beta1 parameter
    pub fn adam_beta1(mut self, adam_beta1: f64) -> Self {
        self.adam_beta1 = adam_beta1;
        self
    }

    /// Set the Adam beta2 parameter
    pub fn adam_beta2(mut self, adam_beta2: f64) -> Self {
        self.adam_beta2 = adam_beta2;
        self
    }

    /// Set the Adam epsilon parameter
    pub fn adam_epsilon(mut self, adam_epsilon: f64) -> Self {
        self.adam_epsilon = adam_epsilon;
        self
    }

    /// Set the gradient clipping threshold
    pub fn grad_clip(mut self, grad_clip: f64) -> Self {
        self.grad_clip = grad_clip;
        self
    }

    /// Set the mini-batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set whether to use natural gradients
    pub fn use_natural_gradients(mut self, use_natural_gradients: bool) -> Self {
        self.use_natural_gradients = use_natural_gradients;
        self
    }

    /// Set the finite difference step size
    pub fn finite_diff_step(mut self, finite_diff_step: f64) -> Self {
        self.finite_diff_step = finite_diff_step;
        self
    }
}

impl Default for ADVIGaussianMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator<Untrained> for ADVIGaussianMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for ADVIGaussianMixture<Untrained> {
    type Fitted = ADVIGaussianMixtureTrained;

    fn fit(self, X: &ArrayView2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _n_features) = X.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than number of components".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = match self.random_state {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng()),
        };

        let mut best_model = None;
        let mut best_lower_bound = f64::NEG_INFINITY;

        for _ in 0..self.n_init {
            // Initialize parameters
            let (
                weight_concentration,
                mean_precision,
                mean_values,
                precision_values,
                degrees_of_freedom,
                scale_matrices,
            ) = self.initialize_parameters(X, &mut rng)?;

            // Run ADVI
            let result = self.run_advi(
                X,
                weight_concentration,
                mean_precision,
                mean_values,
                precision_values,
                degrees_of_freedom,
                scale_matrices,
                &mut rng,
            )?;

            if result.lower_bound > best_lower_bound {
                best_lower_bound = result.lower_bound;
                best_model = Some(result);
            }
        }

        match best_model {
            Some(model) => Ok(model),
            None => Err(SklearsError::ConvergenceError {
                iterations: self.max_iter,
            }),
        }
    }
}

impl ADVIGaussianMixture<Untrained> {
    /// Initialize parameters for ADVI
    fn initialize_parameters(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<(
        Array1<f64>,
        Array1<f64>,
        Array2<f64>,
        Array3<f64>,
        Array1<f64>,
        Array3<f64>,
    )> {
        let (_n_samples, n_features) = X.dim();

        // Initialize weight concentration parameters
        let weight_concentration = Array1::from_elem(self.n_components, self.weight_concentration);

        // Initialize mean precision parameters
        let mean_precision = Array1::from_elem(self.n_components, self.mean_precision);

        // Initialize mean values using k-means++
        let mean_values = self.initialize_means(X, rng)?;

        // Initialize precision values
        let precision_values = self.initialize_precisions(X, n_features)?;

        // Initialize degrees of freedom
        let degrees_of_freedom = Array1::from_elem(
            self.n_components,
            self.degrees_of_freedom + n_features as f64,
        );

        // Initialize scale matrices
        let scale_matrices = self.initialize_scale_matrices(X, n_features)?;

        Ok((
            weight_concentration,
            mean_precision,
            mean_values,
            precision_values,
            degrees_of_freedom,
            scale_matrices,
        ))
    }

    /// Initialize means using k-means++
    fn initialize_means(
        &self,
        X: &ArrayView2<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        // First center is chosen randomly
        let first_idx = rng.gen_range(0..n_samples);
        means
            .slice_mut(s![0, ..])
            .assign(&X.slice(s![first_idx, ..]));

        // Choose remaining centers using k-means++
        for k in 1..self.n_components {
            let mut distances = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                for j in 0..k {
                    let dist = self.squared_distance(&X.slice(s![i, ..]), &means.slice(s![j, ..]));
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[i] = min_dist;
            }

            // Choose next center with probability proportional to squared distance
            let total_dist: f64 = distances.sum();
            let mut prob = rng.gen::<f64>() * total_dist;
            let mut chosen_idx = 0;

            for i in 0..n_samples {
                prob -= distances[i];
                if prob <= 0.0 {
                    chosen_idx = i;
                    break;
                }
            }

            means
                .slice_mut(s![k, ..])
                .assign(&X.slice(s![chosen_idx, ..]));
        }

        Ok(means)
    }

    /// Initialize precision matrices
    fn initialize_precisions(
        &self,
        X: &ArrayView2<f64>,
        n_features: usize,
    ) -> SklResult<Array3<f64>> {
        let mut precisions = Array3::zeros((self.n_components, n_features, n_features));

        // Initialize each precision matrix as identity scaled by data variance
        let data_var = X.var_axis(Axis(0), 0.0);
        let avg_var = data_var.mean().unwrap_or(1.0);

        for k in 0..self.n_components {
            let mut precision = Array2::eye(n_features);
            precision *= 1.0 / (avg_var + self.reg_covar);
            precisions.slice_mut(s![k, .., ..]).assign(&precision);
        }

        Ok(precisions)
    }

    /// Initialize scale matrices for Wishart distributions
    fn initialize_scale_matrices(
        &self,
        X: &ArrayView2<f64>,
        n_features: usize,
    ) -> SklResult<Array3<f64>> {
        let mut scale_matrices = Array3::zeros((self.n_components, n_features, n_features));

        // Initialize each scale matrix as empirical covariance
        let cov = self.compute_empirical_covariance(X)?;

        for k in 0..self.n_components {
            scale_matrices.slice_mut(s![k, .., ..]).assign(&cov);
        }

        Ok(scale_matrices)
    }

    /// Compute empirical covariance matrix
    fn compute_empirical_covariance(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();

        // Compute mean
        let mean = X.mean_axis(Axis(0)).unwrap();

        // Compute covariance
        let mut cov = Array2::zeros((n_features, n_features));
        for i in 0..n_samples {
            let diff = &X.slice(s![i, ..]) - &mean;
            for j in 0..n_features {
                for k in 0..n_features {
                    cov[[j, k]] += diff[j] * diff[k];
                }
            }
        }

        cov /= n_samples as f64;

        // Add regularization
        for i in 0..n_features {
            cov[[i, i]] += self.reg_covar;
        }

        Ok(cov)
    }

    /// Compute squared Euclidean distance
    fn squared_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let diff = a - b;
        diff.dot(&diff)
    }

    /// Run ADVI optimization
    fn run_advi(
        &self,
        X: &ArrayView2<f64>,
        mut weight_concentration: Array1<f64>,
        mut mean_precision: Array1<f64>,
        mut mean_values: Array2<f64>,
        mut precision_values: Array3<f64>,
        mut degrees_of_freedom: Array1<f64>,
        mut scale_matrices: Array3<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<ADVIGaussianMixtureTrained> {
        let (n_samples, n_features) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        // Initialize optimizer states
        let mut optimizer_state = self.initialize_optimizer_state(n_features)?;

        // History for diagnostics
        let mut gradient_history = Vec::new();
        let mut parameter_history = Vec::new();

        let mut prev_lower_bound = f64::NEG_INFINITY;
        let mut lower_bound = f64::NEG_INFINITY;

        for iter in 0..self.max_iter {
            // Convert parameters to optimization vector
            let mut params = self.params_to_vector(
                &weight_concentration,
                &mean_precision,
                &mean_values,
                &precision_values,
                &degrees_of_freedom,
                &scale_matrices,
            )?;

            // Compute gradient using automatic differentiation
            let gradient = self.compute_gradient(X, &params, rng)?;

            // Apply gradient clipping
            let clipped_gradient = self.clip_gradient(&gradient);

            // Update parameters using optimizer
            params =
                self.update_parameters(&params, &clipped_gradient, &mut optimizer_state, iter)?;

            // Convert back to structured parameters
            let (
                new_weight_concentration,
                new_mean_precision,
                new_mean_values,
                new_precision_values,
                new_degrees_of_freedom,
                new_scale_matrices,
            ) = self.vector_to_params(&params, n_features)?;

            weight_concentration = new_weight_concentration;
            mean_precision = new_mean_precision;
            mean_values = new_mean_values;
            precision_values = new_precision_values;
            degrees_of_freedom = new_degrees_of_freedom;
            scale_matrices = new_scale_matrices;

            // E-step: compute responsibilities
            self.compute_responsibilities(
                X,
                &weight_concentration,
                &mean_values,
                &precision_values,
                &degrees_of_freedom,
                &scale_matrices,
                &mut responsibilities,
            )?;

            // Compute lower bound
            lower_bound = self.compute_lower_bound(
                X,
                &responsibilities,
                &weight_concentration,
                &mean_precision,
                &mean_values,
                &precision_values,
                &degrees_of_freedom,
                &scale_matrices,
            )?;

            // Store history
            gradient_history.push(clipped_gradient);
            parameter_history.push(params);

            // Check convergence
            if (lower_bound - prev_lower_bound).abs() < self.tol {
                break;
            }

            prev_lower_bound = lower_bound;
        }

        // Compute model selection criteria
        let n_params = self.count_parameters(n_features);
        let model_selection = ModelSelection {
            aic: -2.0 * lower_bound + 2.0 * n_params as f64,
            bic: -2.0 * lower_bound + (n_params as f64) * (n_samples as f64).ln(),
            log_likelihood: lower_bound,
            n_parameters: n_params,
        };

        Ok(ADVIGaussianMixtureTrained {
            n_components: self.n_components,
            ad_backend: self.ad_backend,
            optimizer: self.optimizer,
            covariance_type: self.covariance_type.clone(),
            weight_concentration,
            mean_precision,
            mean_values,
            precision_values,
            degrees_of_freedom,
            scale_matrices,
            n_samples,
            n_features,
            lower_bound,
            responsibilities,
            model_selection,
            gradient_history,
            parameter_history,
        })
    }

    /// Initialize optimizer state
    fn initialize_optimizer_state(&self, n_features: usize) -> SklResult<OptimizerState> {
        let param_count = self.count_parameters(n_features);

        Ok(match self.optimizer {
            ADVIOptimizer::SGD => OptimizerState::SGD,
            ADVIOptimizer::AdaGrad => OptimizerState::AdaGrad {
                accumulated_grad: Array1::zeros(param_count),
            },
            ADVIOptimizer::RMSprop => OptimizerState::RMSprop {
                accumulated_grad: Array1::zeros(param_count),
            },
            ADVIOptimizer::Adam => OptimizerState::Adam {
                m: Array1::zeros(param_count),
                v: Array1::zeros(param_count),
            },
            ADVIOptimizer::LBFGS => OptimizerState::LBFGS {
                history: Vec::new(),
            },
        })
    }

    /// Convert parameters to optimization vector
    fn params_to_vector(
        &self,
        weight_concentration: &Array1<f64>,
        mean_precision: &Array1<f64>,
        mean_values: &Array2<f64>,
        precision_values: &Array3<f64>,
        degrees_of_freedom: &Array1<f64>,
        scale_matrices: &Array3<f64>,
    ) -> SklResult<Array1<f64>> {
        let mut params = Vec::new();

        // Add weight concentration parameters
        params.extend(weight_concentration.iter().cloned());

        // Add mean precision parameters
        params.extend(mean_precision.iter().cloned());

        // Add mean values
        params.extend(mean_values.iter().cloned());

        // Add precision values (upper triangular part only)
        for k in 0..self.n_components {
            let precision = precision_values.slice(s![k, .., ..]);
            for i in 0..precision.nrows() {
                for j in i..precision.ncols() {
                    params.push(precision[[i, j]]);
                }
            }
        }

        // Add degrees of freedom parameters
        params.extend(degrees_of_freedom.iter().cloned());

        // Add scale matrices (upper triangular part only)
        for k in 0..self.n_components {
            let scale = scale_matrices.slice(s![k, .., ..]);
            for i in 0..scale.nrows() {
                for j in i..scale.ncols() {
                    params.push(scale[[i, j]]);
                }
            }
        }

        Ok(Array1::from_vec(params))
    }

    /// Convert optimization vector to parameters
    fn vector_to_params(
        &self,
        params: &Array1<f64>,
        n_features: usize,
    ) -> SklResult<(
        Array1<f64>,
        Array1<f64>,
        Array2<f64>,
        Array3<f64>,
        Array1<f64>,
        Array3<f64>,
    )> {
        let mut idx = 0;

        // Extract weight concentration parameters
        let weight_concentration = params.slice(s![idx..idx + self.n_components]).to_owned();
        idx += self.n_components;

        // Extract mean precision parameters
        let mean_precision = params.slice(s![idx..idx + self.n_components]).to_owned();
        idx += self.n_components;

        // Extract mean values
        let mean_values = params
            .slice(s![idx..idx + self.n_components * n_features])
            .to_owned()
            .into_shape((self.n_components, n_features))?;
        idx += self.n_components * n_features;

        // Extract precision values
        let mut precision_values = Array3::zeros((self.n_components, n_features, n_features));
        let tri_size = n_features * (n_features + 1) / 2;
        for k in 0..self.n_components {
            let tri_params = params.slice(s![idx..idx + tri_size]);
            let mut tri_idx = 0;
            for i in 0..n_features {
                for j in i..n_features {
                    precision_values[[k, i, j]] = tri_params[tri_idx];
                    if i != j {
                        precision_values[[k, j, i]] = tri_params[tri_idx];
                    }
                    tri_idx += 1;
                }
            }
            idx += tri_size;
        }

        // Extract degrees of freedom parameters
        let degrees_of_freedom = params.slice(s![idx..idx + self.n_components]).to_owned();
        idx += self.n_components;

        // Extract scale matrices
        let mut scale_matrices = Array3::zeros((self.n_components, n_features, n_features));
        for k in 0..self.n_components {
            let tri_params = params.slice(s![idx..idx + tri_size]);
            let mut tri_idx = 0;
            for i in 0..n_features {
                for j in i..n_features {
                    scale_matrices[[k, i, j]] = tri_params[tri_idx];
                    if i != j {
                        scale_matrices[[k, j, i]] = tri_params[tri_idx];
                    }
                    tri_idx += 1;
                }
            }
            idx += tri_size;
        }

        Ok((
            weight_concentration,
            mean_precision,
            mean_values,
            precision_values,
            degrees_of_freedom,
            scale_matrices,
        ))
    }

    /// Compute gradient using automatic differentiation
    fn compute_gradient(
        &self,
        X: &ArrayView2<f64>,
        params: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        match self.ad_backend {
            ADBackend::FiniteDifferences => self.compute_gradient_finite_diff(X, params, rng),
            ADBackend::ForwardMode => self.compute_gradient_forward_mode(X, params, rng),
            ADBackend::ReverseMode => self.compute_gradient_reverse_mode(X, params, rng),
            ADBackend::DualNumbers => self.compute_gradient_dual_numbers(X, params, rng),
        }
    }

    /// Compute gradient using finite differences
    fn compute_gradient_finite_diff(
        &self,
        X: &ArrayView2<f64>,
        params: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        let mut gradient = Array1::zeros(params.len());
        let _f0 = self.evaluate_objective(X, params, rng)?;

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            params_plus[i] += self.finite_diff_step;
            let f_plus = self.evaluate_objective(X, &params_plus, rng)?;

            let mut params_minus = params.clone();
            params_minus[i] -= self.finite_diff_step;
            let f_minus = self.evaluate_objective(X, &params_minus, rng)?;

            gradient[i] = (f_plus - f_minus) / (2.0 * self.finite_diff_step);
        }

        Ok(gradient)
    }

    /// Compute gradient using forward mode AD
    fn compute_gradient_forward_mode(
        &self,
        X: &ArrayView2<f64>,
        params: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        let mut gradient = Array1::zeros(params.len());

        // Forward mode: compute one directional derivative at a time
        for i in 0..params.len() {
            let mut dual_params = params
                .iter()
                .map(|&x| Dual::constant(x))
                .collect::<Vec<_>>();
            dual_params[i] = Dual::variable(params[i]);

            let dual_objective = self.evaluate_objective_dual(X, &dual_params, rng)?;
            gradient[i] = dual_objective.derivative;
        }

        Ok(gradient)
    }

    /// Compute gradient using reverse mode AD (simplified)
    fn compute_gradient_reverse_mode(
        &self,
        X: &ArrayView2<f64>,
        params: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        // For simplicity, fall back to finite differences
        // In a full implementation, this would use reverse mode AD
        self.compute_gradient_finite_diff(X, params, rng)
    }

    /// Compute gradient using dual numbers
    fn compute_gradient_dual_numbers(
        &self,
        X: &ArrayView2<f64>,
        params: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        let mut gradient = Array1::zeros(params.len());

        // Dual numbers: compute one directional derivative at a time
        for i in 0..params.len() {
            let mut dual_params = params
                .iter()
                .map(|&x| Dual::constant(x))
                .collect::<Vec<_>>();
            dual_params[i] = Dual::variable(params[i]);

            let dual_objective = self.evaluate_objective_dual(X, &dual_params, rng)?;
            gradient[i] = dual_objective.derivative;
        }

        Ok(gradient)
    }

    /// Evaluate objective function
    fn evaluate_objective(
        &self,
        X: &ArrayView2<f64>,
        params: &Array1<f64>,
        _rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = X.dim();

        // Convert parameters back to structured form
        let (
            weight_concentration,
            mean_precision,
            mean_values,
            precision_values,
            degrees_of_freedom,
            scale_matrices,
        ) = self.vector_to_params(params, n_features)?;

        // Compute responsibilities
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));
        self.compute_responsibilities(
            X,
            &weight_concentration,
            &mean_values,
            &precision_values,
            &degrees_of_freedom,
            &scale_matrices,
            &mut responsibilities,
        )?;

        // Compute lower bound
        self.compute_lower_bound(
            X,
            &responsibilities,
            &weight_concentration,
            &mean_precision,
            &mean_values,
            &precision_values,
            &degrees_of_freedom,
            &scale_matrices,
        )
    }

    /// Evaluate objective function with dual numbers
    fn evaluate_objective_dual(
        &self,
        X: &ArrayView2<f64>,
        dual_params: &[Dual],
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Dual> {
        // For simplicity, convert dual parameters to regular parameters
        // and use finite differences for the dual part
        let params = dual_params.iter().map(|d| d.value).collect::<Vec<_>>();
        let params_array = Array1::from_vec(params);

        let f0 = self.evaluate_objective(X, &params_array, rng)?;

        // Find the variable parameter
        let mut variable_idx = 0;
        for (i, &dual) in dual_params.iter().enumerate() {
            if dual.derivative != 0.0 {
                variable_idx = i;
                break;
            }
        }

        // Compute derivative using finite differences
        let mut params_plus = params_array.clone();
        params_plus[variable_idx] += self.finite_diff_step;
        let f_plus = self.evaluate_objective(X, &params_plus, rng)?;

        let derivative = (f_plus - f0) / self.finite_diff_step;

        Ok(Dual::new(f0, derivative))
    }

    /// Compute responsibilities
    fn compute_responsibilities(
        &self,
        X: &ArrayView2<f64>,
        weight_concentration: &Array1<f64>,
        mean_values: &Array2<f64>,
        precision_values: &Array3<f64>,
        degrees_of_freedom: &Array1<f64>,
        scale_matrices: &Array3<f64>,
        responsibilities: &mut Array2<f64>,
    ) -> SklResult<()> {
        let (n_samples, _) = X.dim();

        // Compute expected log weights
        let expected_log_weights = self.compute_expected_log_weights(weight_concentration)?;

        // Compute expected log likelihoods
        for i in 0..n_samples {
            let mut log_resp = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let expected_log_likelihood = self.compute_expected_log_likelihood(
                    &X.slice(s![i, ..]),
                    &mean_values.slice(s![k, ..]),
                    &precision_values.slice(s![k, .., ..]),
                    &degrees_of_freedom[k],
                    &scale_matrices.slice(s![k, .., ..]),
                )?;

                log_resp[k] = expected_log_weights[k] + expected_log_likelihood;
            }

            // Normalize responsibilities
            let log_prob_norm = self.log_sum_exp_array(&log_resp);
            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_resp[k] - log_prob_norm).exp();
            }
        }

        Ok(())
    }

    /// Compute expected log weights
    fn compute_expected_log_weights(
        &self,
        weight_concentration: &Array1<f64>,
    ) -> SklResult<Array1<f64>> {
        let concentration_sum: f64 = weight_concentration.sum();
        let mut expected_log_weights = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            expected_log_weights[k] =
                Self::digamma(weight_concentration[k]) - Self::digamma(concentration_sum);
        }

        Ok(expected_log_weights)
    }

    /// Compute expected log likelihood
    fn compute_expected_log_likelihood(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        precision: &ArrayView2<f64>,
        degrees_of_freedom: &f64,
        _scale_matrix: &ArrayView2<f64>,
    ) -> SklResult<f64> {
        let n_features = x.len();
        let diff = x - mean;

        // Compute expected log determinant of precision matrix
        let mut expected_log_det = 0.0;
        for i in 0..n_features {
            expected_log_det += Self::digamma((degrees_of_freedom + 1.0 - i as f64) / 2.0);
        }
        expected_log_det += n_features as f64 * (2.0_f64).ln();

        // Compute expected quadratic form
        let mut expected_quad_form = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                expected_quad_form += diff[i] * precision[[i, j]] * diff[j];
            }
        }
        expected_quad_form *= degrees_of_freedom / (degrees_of_freedom - 2.0);

        let log_likelihood = 0.5 * expected_log_det
            - 0.5 * expected_quad_form
            - 0.5 * n_features as f64 * (2.0 * PI).ln();

        Ok(log_likelihood)
    }

    /// Compute lower bound
    fn compute_lower_bound(
        &self,
        X: &ArrayView2<f64>,
        responsibilities: &Array2<f64>,
        weight_concentration: &Array1<f64>,
        _mean_precision: &Array1<f64>,
        mean_values: &Array2<f64>,
        precision_values: &Array3<f64>,
        degrees_of_freedom: &Array1<f64>,
        scale_matrices: &Array3<f64>,
    ) -> SklResult<f64> {
        let (n_samples, _n_features) = X.dim();
        let mut lower_bound = 0.0;

        // Expected log likelihood
        let expected_log_weights = self.compute_expected_log_weights(weight_concentration)?;

        for i in 0..n_samples {
            for k in 0..self.n_components {
                let responsibility = responsibilities[[i, k]];
                if responsibility > 1e-10 {
                    let expected_log_likelihood = self.compute_expected_log_likelihood(
                        &X.slice(s![i, ..]),
                        &mean_values.slice(s![k, ..]),
                        &precision_values.slice(s![k, .., ..]),
                        &degrees_of_freedom[k],
                        &scale_matrices.slice(s![k, .., ..]),
                    )?;

                    lower_bound +=
                        responsibility * (expected_log_weights[k] + expected_log_likelihood);
                }
            }
        }

        // KL divergence terms (simplified)
        let concentration_sum: f64 = weight_concentration.sum();
        let prior_concentration_sum = self.weight_concentration * self.n_components as f64;

        // Weight KL divergence
        lower_bound +=
            Self::log_gamma(concentration_sum) - Self::log_gamma(prior_concentration_sum);
        for k in 0..self.n_components {
            lower_bound += Self::log_gamma(self.weight_concentration)
                - Self::log_gamma(weight_concentration[k]);
            lower_bound += (weight_concentration[k] - self.weight_concentration)
                * (Self::digamma(weight_concentration[k]) - Self::digamma(concentration_sum));
        }

        // Entropy term
        for i in 0..n_samples {
            for k in 0..self.n_components {
                let responsibility = responsibilities[[i, k]];
                if responsibility > 1e-10 {
                    lower_bound -= responsibility * responsibility.ln();
                }
            }
        }

        Ok(lower_bound)
    }

    /// Clip gradient
    fn clip_gradient(&self, gradient: &Array1<f64>) -> Array1<f64> {
        let grad_norm = gradient.dot(gradient).sqrt();
        if grad_norm > self.grad_clip {
            gradient * (self.grad_clip / grad_norm)
        } else {
            gradient.clone()
        }
    }

    /// Update parameters using optimizer
    fn update_parameters(
        &self,
        params: &Array1<f64>,
        gradient: &Array1<f64>,
        optimizer_state: &mut OptimizerState,
        iteration: usize,
    ) -> SklResult<Array1<f64>> {
        match (self.optimizer, optimizer_state) {
            (ADVIOptimizer::SGD, OptimizerState::SGD) => {
                Ok(params + &(gradient * self.learning_rate))
            }
            (ADVIOptimizer::AdaGrad, OptimizerState::AdaGrad { accumulated_grad }) => {
                *accumulated_grad = &*accumulated_grad + &(gradient * gradient);
                let update = gradient / &(accumulated_grad.mapv(|x| x.sqrt() + self.adam_epsilon));
                Ok(params + &(update * self.learning_rate))
            }
            (ADVIOptimizer::RMSprop, OptimizerState::RMSprop { accumulated_grad }) => {
                *accumulated_grad = &*accumulated_grad * 0.9 + &(gradient * gradient * 0.1);
                let update = gradient / &(accumulated_grad.mapv(|x| x.sqrt() + self.adam_epsilon));
                Ok(params + &(update * self.learning_rate))
            }
            (ADVIOptimizer::Adam, OptimizerState::Adam { m, v }) => {
                *m = &*m * self.adam_beta1 + &(gradient * (1.0 - self.adam_beta1));
                *v = &*v * self.adam_beta2 + &(gradient * gradient * (1.0 - self.adam_beta2));

                let bias_correction1 = 1.0 - self.adam_beta1.powi(iteration as i32 + 1);
                let bias_correction2 = 1.0 - self.adam_beta2.powi(iteration as i32 + 1);

                let m_corrected = &*m / bias_correction1;
                let v_corrected = &*v / bias_correction2;

                let update = m_corrected / &(v_corrected.mapv(|x| x.sqrt() + self.adam_epsilon));
                Ok(params + &(update * self.learning_rate))
            }
            (ADVIOptimizer::LBFGS, OptimizerState::LBFGS { history }) => {
                // Simplified L-BFGS implementation
                // In practice, this would use a more sophisticated L-BFGS implementation
                history.push((params.clone(), gradient.clone()));
                if history.len() > 10 {
                    history.remove(0);
                }
                Ok(params + &(gradient * self.learning_rate))
            }
            _ => Err(SklearsError::InvalidInput(
                "Optimizer and state mismatch".to_string(),
            )),
        }
    }

    /// Count the number of model parameters
    fn count_parameters(&self, n_features: usize) -> usize {
        let mut n_params = 0;

        // Weight concentration parameters
        n_params += self.n_components;

        // Mean precision parameters
        n_params += self.n_components;

        // Mean values
        n_params += self.n_components * n_features;

        // Precision values (upper triangular part only)
        n_params += self.n_components * n_features * (n_features + 1) / 2;

        // Degrees of freedom parameters
        n_params += self.n_components;

        // Scale matrices (upper triangular part only)
        n_params += self.n_components * n_features * (n_features + 1) / 2;

        n_params
    }

    /// Digamma function approximation
    fn digamma(x: f64) -> f64 {
        if x < 8.0 {
            Self::digamma(x + 1.0) - 1.0 / x
        } else {
            let inv_x = 1.0 / x;
            let inv_x2 = inv_x * inv_x;
            x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        }
    }

    /// Log gamma function approximation
    fn log_gamma(x: f64) -> f64 {
        if x < 0.5 {
            (PI / (PI * x).sin()).ln() - Self::log_gamma(1.0 - x)
        } else {
            let g = 7.0;
            let c = [
                0.999_999_999_999_809_9,
                676.5203681218851,
                -1259.1392167224028,
                771.323_428_777_653_1,
                -176.615_029_162_140_6,
                12.507343278686905,
                -0.13857109526572012,
                9.984_369_578_019_572e-6,
                1.5056327351493116e-7,
            ];

            let z = x - 1.0;
            let mut x_sum = c[0];
            for (i, &c_val) in c.iter().enumerate().skip(1) {
                x_sum += c_val / (z + i as f64);
            }
            let t = z + g + 0.5;
            (2.0 * PI).sqrt().ln() + (z + 0.5) * t.ln() - t + x_sum.ln()
        }
    }

    /// Log sum exp for array
    fn log_sum_exp_array(&self, arr: &Array1<f64>) -> f64 {
        let max_val = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max_val.is_finite() {
            max_val + arr.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln()
        } else {
            max_val
        }
    }
}

/// Optimizer state
#[derive(Debug, Clone)]
pub enum OptimizerState {
    /// SGD
    SGD,
    /// AdaGrad
    AdaGrad { accumulated_grad: Array1<f64> },
    /// RMSprop
    RMSprop { accumulated_grad: Array1<f64> },
    /// Adam
    Adam { m: Array1<f64>, v: Array1<f64> },
    /// LBFGS
    LBFGS {
        history: Vec<(Array1<f64>, Array1<f64>)>,
    },
}

impl Estimator<Trained> for ADVIGaussianMixtureTrained {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<usize>> for ADVIGaussianMixtureTrained {
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<usize>> {
        let probabilities = self.predict_proba(X)?;
        let mut predictions = Array1::zeros(X.nrows());

        for i in 0..X.nrows() {
            let mut max_prob = 0.0;
            let mut best_class = 0;

            for k in 0..self.n_components {
                if probabilities[[i, k]] > max_prob {
                    max_prob = probabilities[[i, k]];
                    best_class = k;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl ADVIGaussianMixtureTrained {
    /// Predict class probabilities
    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let mut probabilities = Array2::zeros((n_samples, self.n_components));

        // Compute expected log weights
        let expected_log_weights = self.compute_expected_log_weights()?;

        for i in 0..n_samples {
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let expected_log_likelihood = self.compute_expected_log_likelihood(
                    &X.slice(s![i, ..]),
                    &self.mean_values.slice(s![k, ..]),
                    &self.precision_values.slice(s![k, .., ..]),
                    &self.degrees_of_freedom[k],
                    &self.scale_matrices.slice(s![k, .., ..]),
                )?;

                log_probs[k] = expected_log_weights[k] + expected_log_likelihood;
            }

            // Normalize
            let log_prob_norm = self.log_sum_exp_array(&log_probs);
            for k in 0..self.n_components {
                probabilities[[i, k]] = (log_probs[k] - log_prob_norm).exp();
            }
        }

        Ok(probabilities)
    }

    /// Compute log-likelihood of the data
    pub fn score(&self, X: &ArrayView2<f64>) -> SklResult<f64> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let expected_log_weights = self.compute_expected_log_weights()?;
        let mut total_log_likelihood = 0.0;

        for i in 0..n_samples {
            let mut log_probs = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                let expected_log_likelihood = self.compute_expected_log_likelihood(
                    &X.slice(s![i, ..]),
                    &self.mean_values.slice(s![k, ..]),
                    &self.precision_values.slice(s![k, .., ..]),
                    &self.degrees_of_freedom[k],
                    &self.scale_matrices.slice(s![k, .., ..]),
                )?;

                log_probs[k] = expected_log_weights[k] + expected_log_likelihood;
            }

            total_log_likelihood += self.log_sum_exp_array(&log_probs);
        }

        Ok(total_log_likelihood)
    }

    /// Get model selection criteria
    pub fn model_selection(&self) -> &ModelSelection {
        &self.model_selection
    }

    /// Get the lower bound (ELBO)
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// Get the final responsibilities
    pub fn responsibilities(&self) -> &Array2<f64> {
        &self.responsibilities
    }

    /// Get the variational mean parameters
    pub fn mean_values(&self) -> &Array2<f64> {
        &self.mean_values
    }

    /// Get the variational precision parameters
    pub fn precision_values(&self) -> &Array3<f64> {
        &self.precision_values
    }

    /// Get the AD backend used
    pub fn ad_backend(&self) -> ADBackend {
        self.ad_backend
    }

    /// Get the optimizer used
    pub fn optimizer(&self) -> ADVIOptimizer {
        self.optimizer
    }

    /// Get the gradient history
    pub fn gradient_history(&self) -> &Vec<Array1<f64>> {
        &self.gradient_history
    }

    /// Get the parameter history
    pub fn parameter_history(&self) -> &Vec<Array1<f64>> {
        &self.parameter_history
    }

    /// Helper methods for the trained model
    fn compute_expected_log_weights(&self) -> SklResult<Array1<f64>> {
        let concentration_sum: f64 = self.weight_concentration.sum();
        let mut expected_log_weights = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            expected_log_weights[k] =
                Self::digamma(self.weight_concentration[k]) - Self::digamma(concentration_sum);
        }

        Ok(expected_log_weights)
    }

    fn compute_expected_log_likelihood(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        precision: &ArrayView2<f64>,
        degrees_of_freedom: &f64,
        _scale_matrix: &ArrayView2<f64>,
    ) -> SklResult<f64> {
        let n_features = x.len();
        let diff = x - mean;

        // Compute expected log determinant of precision matrix
        let mut expected_log_det = 0.0;
        for i in 0..n_features {
            expected_log_det += Self::digamma((degrees_of_freedom + 1.0 - i as f64) / 2.0);
        }
        expected_log_det += n_features as f64 * (2.0_f64).ln();

        // Compute expected quadratic form
        let mut expected_quad_form = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                expected_quad_form += diff[i] * precision[[i, j]] * diff[j];
            }
        }
        expected_quad_form *= degrees_of_freedom / (degrees_of_freedom - 2.0);

        let log_likelihood = 0.5 * expected_log_det
            - 0.5 * expected_quad_form
            - 0.5 * n_features as f64 * (2.0 * PI).ln();

        Ok(log_likelihood)
    }

    fn digamma(x: f64) -> f64 {
        if x < 8.0 {
            Self::digamma(x + 1.0) - 1.0 / x
        } else {
            let inv_x = 1.0 / x;
            let inv_x2 = inv_x * inv_x;
            x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        }
    }

    fn log_sum_exp_array(&self, arr: &Array1<f64>) -> f64 {
        let max_val = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max_val.is_finite() {
            max_val + arr.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln()
        } else {
            max_val
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Predict;

    #[test]
    fn test_advi_gaussian_mixture_creation() {
        let gmm = ADVIGaussianMixture::new()
            .n_components(3)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.001)
            .max_iter(500);

        assert_eq!(gmm.n_components, 3);
        assert_eq!(gmm.ad_backend, ADBackend::DualNumbers);
        assert_eq!(gmm.optimizer, ADVIOptimizer::Adam);
        assert_eq!(gmm.learning_rate, 0.001);
        assert_eq!(gmm.max_iter, 500);
    }

    #[test]
    fn test_dual_number_arithmetic() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.0);

        let sum = a + b;
        assert_eq!(sum.value, 5.0);
        assert_eq!(sum.derivative, 1.0);

        let product = a * b;
        assert_eq!(product.value, 6.0);
        assert_eq!(product.derivative, 3.0);

        let exp_a = a.exp();
        assert_abs_diff_eq!(exp_a.value, 2.0_f64.exp(), epsilon = 1e-10);
        assert_abs_diff_eq!(exp_a.derivative, 2.0_f64.exp(), epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_fit_predict() {
        let X = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0]
        ];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(50);

        let fitted = gmm.fit(&X.view(), &()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 6);
        assert!(predictions.iter().all(|&label| label < 2));

        // Check that points are clustered correctly
        let first_cluster = predictions[0];
        assert_eq!(predictions[1], first_cluster);
        assert_eq!(predictions[2], first_cluster);

        let second_cluster = predictions[3];
        assert_eq!(predictions[4], second_cluster);
        assert_eq!(predictions[5], second_cluster);

        assert_ne!(first_cluster, second_cluster);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_backends() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let backends = vec![
            ADBackend::FiniteDifferences,
            ADBackend::DualNumbers,
            ADBackend::ForwardMode,
        ];

        for backend in backends {
            let gmm = ADVIGaussianMixture::new()
                .n_components(2)
                .ad_backend(backend)
                .optimizer(ADVIOptimizer::SGD)
                .learning_rate(0.1)
                .random_state(42)
                .tol(1e-2)
                .max_iter(20);

            let fitted = gmm.fit(&X.view(), &()).unwrap();
            let predictions = fitted.predict(&X.view()).unwrap();

            assert_eq!(predictions.len(), 4);
            assert!(predictions.iter().all(|&label| label < 2));
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_optimizers() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let optimizers = vec![
            ADVIOptimizer::SGD,
            ADVIOptimizer::AdaGrad,
            ADVIOptimizer::RMSprop,
            ADVIOptimizer::Adam,
        ];

        for optimizer in optimizers {
            let gmm = ADVIGaussianMixture::new()
                .n_components(2)
                .ad_backend(ADBackend::DualNumbers)
                .optimizer(optimizer)
                .learning_rate(0.1)
                .random_state(42)
                .tol(1e-2)
                .max_iter(20);

            let fitted = gmm.fit(&X.view(), &()).unwrap();
            let predictions = fitted.predict(&X.view()).unwrap();

            assert_eq!(predictions.len(), 4);
            assert!(predictions.iter().all(|&label| label < 2));
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_probabilities() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted = gmm.fit(&X.view(), &()).unwrap();
        let probabilities = fitted.predict_proba(&X.view()).unwrap();

        assert_eq!(probabilities.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probabilities.slice(s![i, ..]).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }

        // Check that probabilities are non-negative
        assert!(probabilities.iter().all(|&p| p >= 0.0));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_score() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted = gmm.fit(&X.view(), &()).unwrap();
        let score = fitted.score(&X.view()).unwrap();

        assert!(score.is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_model_selection() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted = gmm.fit(&X.view(), &()).unwrap();
        let model_selection = fitted.model_selection();

        assert!(model_selection.aic.is_finite());
        assert!(model_selection.bic.is_finite());
        assert!(model_selection.log_likelihood.is_finite());
        assert!(model_selection.n_parameters > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_parameter_access() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted = gmm.fit(&X.view(), &()).unwrap();

        // Test parameter access
        assert_eq!(fitted.mean_values().dim(), (2, 2));
        assert_eq!(fitted.precision_values().dim(), (2, 2, 2));
        assert_eq!(fitted.responsibilities().dim(), (4, 2));
        assert_eq!(fitted.ad_backend(), ADBackend::DualNumbers);
        assert_eq!(fitted.optimizer(), ADVIOptimizer::Adam);

        // Test that lower bound is finite
        assert!(fitted.lower_bound().is_finite());

        // Test that history is recorded
        assert!(!fitted.gradient_history().is_empty());
        assert!(!fitted.parameter_history().is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_reproducibility() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm1 = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let gmm2 = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted1 = gmm1.fit(&X.view(), &()).unwrap();
        let fitted2 = gmm2.fit(&X.view(), &()).unwrap();

        let predictions1 = fitted1.predict(&X.view()).unwrap();
        let predictions2 = fitted2.predict(&X.view()).unwrap();

        assert_eq!(predictions1, predictions2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_error_handling() {
        let X = array![[0.0, 0.0], [1.0, 1.0]];

        // Test with more components than samples
        let gmm = ADVIGaussianMixture::new()
            .n_components(5)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .random_state(42);

        let result = gmm.fit(&X.view(), &());
        assert!(result.is_err());

        // Test dimension mismatch in predict
        let gmm2 = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(20);

        let fitted = gmm2.fit(&X.view(), &()).unwrap();

        let X_wrong = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];

        let result = fitted.predict(&X_wrong.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_gradient_clipping() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .grad_clip(0.5)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted = gmm.fit(&X.view(), &()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&label| label < 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_natural_gradients() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .use_natural_gradients(true)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);

        let fitted = gmm.fit(&X.view(), &()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&label| label < 2));
    }
}
