//! Mixture Discriminant Analysis for semi-supervised learning
//!
//! This module provides Mixture Discriminant Analysis (MDA), a probabilistic
//! semi-supervised learning method that extends discriminant analysis to
//! handle both labeled and unlabeled data through mixture modeling.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Mixture Discriminant Analysis
///
/// MDA is a semi-supervised extension of Linear/Quadratic Discriminant Analysis
/// that uses both labeled and unlabeled data to learn class-conditional densities.
/// It models each class as a mixture of Gaussians and uses EM algorithm to
/// estimate parameters from both labeled and unlabeled samples.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components per class
/// * `covariance_type` - Type of covariance matrix ('full', 'tied', 'diag', 'spherical')
/// * `reg_covar` - Regularization added to diagonal of covariance
/// * `max_iter` - Maximum number of EM iterations
/// * `tol` - Convergence tolerance
/// * `n_init` - Number of initializations
/// * `random_state` - Random state for reproducible results
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::MixtureDiscriminantAnalysis;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let mda = MixtureDiscriminantAnalysis::new()
///     .n_components(2)
///     .covariance_type("full".to_string())
///     .max_iter(100);
/// let fitted = mda.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MixtureDiscriminantAnalysis<S = Untrained> {
    state: S,
    n_components: usize,
    covariance_type: String,
    reg_covar: f64,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    random_state: Option<u64>,
}

impl MixtureDiscriminantAnalysis<Untrained> {
    /// Create a new MixtureDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            covariance_type: "full".to_string(),
            reg_covar: 1e-6,
            max_iter: 100,
            tol: 1e-3,
            n_init: 1,
            random_state: None,
        }
    }

    /// Set the number of mixture components per class
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: String) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the covariance regularization
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Initialize mixture parameters
    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
        labeled_indices: &[usize],
        y: &Array1<i32>,
        classes: &[i32],
    ) -> SklResult<(
        Vec<Vec<Array1<f64>>>,
        Vec<Vec<Array2<f64>>>,
        Vec<Array1<f64>>,
        Array1<f64>,
    )> {
        let n_features = X.ncols();
        let n_classes = classes.len();

        // Initialize means, covariances, component weights, and class priors
        let mut means = Vec::new();
        let mut covariances = Vec::new();
        let mut component_weights = Vec::new();
        let mut class_priors = Array1::zeros(n_classes);

        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Find labeled samples for this class
            let class_samples: Vec<usize> = labeled_indices
                .iter()
                .filter(|&&i| y[i] == class_label)
                .copied()
                .collect();

            if class_samples.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "No labeled samples for class {}",
                    class_label
                )));
            }

            // Initialize means for this class
            let mut class_means = Vec::new();
            let mut class_covariances = Vec::new();

            // Simple initialization: use labeled samples as initial means
            for comp_idx in 0..self.n_components {
                let sample_idx = class_samples[comp_idx % class_samples.len()];
                let mean = X.row(sample_idx).to_owned();
                class_means.push(mean);

                // Initialize covariance based on type
                let cov = match self.covariance_type.as_str() {
                    "full" => {
                        let mut cov = Array2::eye(n_features) * self.reg_covar;
                        // Add small random perturbation
                        for i in 0..n_features {
                            for j in 0..n_features {
                                if i == j {
                                    cov[[i, j]] += 1.0;
                                }
                            }
                        }
                        cov
                    }
                    "diag" => Array2::eye(n_features),
                    "spherical" => Array2::eye(n_features),
                    "tied" => Array2::eye(n_features),
                    _ => {
                        return Err(SklearsError::InvalidInput(format!(
                            "Unknown covariance type: {}",
                            self.covariance_type
                        )));
                    }
                };
                class_covariances.push(cov);
            }

            means.push(class_means);
            covariances.push(class_covariances);

            // Initialize component weights (uniform)
            let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
            component_weights.push(weights);

            // Set class prior based on labeled samples
            class_priors[class_idx] = class_samples.len() as f64 / labeled_indices.len() as f64;
        }

        Ok((means, covariances, component_weights, class_priors))
    }

    /// Compute multivariate Gaussian PDF
    fn multivariate_gaussian_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &Array1<f64>,
        cov: &Array2<f64>,
    ) -> f64 {
        let n_features = x.len();
        let diff = x - mean;

        // Compute determinant and inverse (simplified)
        let det = match self.covariance_type.as_str() {
            "spherical" => cov[[0, 0]].powf(n_features as f64),
            "diag" => cov.diag().iter().product(),
            _ => {
                // Simplified determinant calculation
                let mut det = 1.0;
                for i in 0..n_features {
                    det *= cov[[i, i]];
                }
                det
            }
        };

        if det <= 0.0 {
            return 1e-10; // Avoid numerical issues
        }

        // Simplified Mahalanobis distance calculation
        let mut mahal_dist = 0.0;
        match self.covariance_type.as_str() {
            "spherical" => {
                let var = cov[[0, 0]];
                mahal_dist = diff.mapv(|x| x * x).sum() / var;
            }
            "diag" => {
                for i in 0..n_features {
                    mahal_dist += diff[i] * diff[i] / cov[[i, i]];
                }
            }
            _ => {
                // Simplified full covariance
                for i in 0..n_features {
                    mahal_dist += diff[i] * diff[i] / cov[[i, i]];
                }
            }
        }

        let normalization =
            1.0 / ((2.0 * std::f64::consts::PI).powf(n_features as f64 / 2.0) * det.sqrt());
        normalization * (-0.5 * mahal_dist).exp()
    }

    /// E-step: Compute responsibilities
    fn e_step(
        &self,
        X: &Array2<f64>,
        means: &[Vec<Array1<f64>>],
        covariances: &[Vec<Array2<f64>>],
        component_weights: &[Array1<f64>],
        class_priors: &Array1<f64>,
        labeled_indices: &[usize],
        y: &Array1<i32>,
        classes: &[i32],
    ) -> (Array2<f64>, f64) {
        let n_samples = X.nrows();
        let n_classes = classes.len();
        let total_components = n_classes * self.n_components;

        let mut responsibilities = Array2::zeros((n_samples, total_components));
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let x = X.row(i);
            let mut total_prob = 0.0;
            let mut probs = Vec::new();

            // Compute probabilities for each class and component
            for (class_idx, &class_label) in classes.iter().enumerate() {
                for comp_idx in 0..self.n_components {
                    let comp_global_idx = class_idx * self.n_components + comp_idx;
                    let prob = class_priors[class_idx]
                        * component_weights[class_idx][comp_idx]
                        * self.multivariate_gaussian_pdf(
                            &x,
                            &means[class_idx][comp_idx],
                            &covariances[class_idx][comp_idx],
                        );
                    probs.push(prob);
                    total_prob += prob;
                }
            }

            // Normalize and assign responsibilities
            if total_prob > 0.0 {
                for (comp_idx, &prob) in probs.iter().enumerate() {
                    responsibilities[[i, comp_idx]] = prob / total_prob;
                }
                log_likelihood += total_prob.ln();
            } else {
                // Uniform assignment if no valid probability
                for comp_idx in 0..total_components {
                    responsibilities[[i, comp_idx]] = 1.0 / total_components as f64;
                }
            }

            // Hard assignment for labeled samples
            if labeled_indices.contains(&i) {
                if let Some(class_idx) = classes.iter().position(|&c| c == y[i]) {
                    // Set responsibility to 1 for true class, 0 for others
                    for comp_idx in 0..total_components {
                        responsibilities[[i, comp_idx]] = 0.0;
                    }
                    // Uniform distribution within the true class
                    for comp_idx in 0..self.n_components {
                        let global_comp_idx = class_idx * self.n_components + comp_idx;
                        responsibilities[[i, global_comp_idx]] = 1.0 / self.n_components as f64;
                    }
                }
            }
        }

        (responsibilities, log_likelihood)
    }

    /// M-step: Update parameters
    fn m_step(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        classes: &[i32],
    ) -> (
        Vec<Vec<Array1<f64>>>,
        Vec<Vec<Array2<f64>>>,
        Vec<Array1<f64>>,
        Array1<f64>,
    ) {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_classes = classes.len();

        let mut means = Vec::new();
        let mut covariances = Vec::new();
        let mut component_weights = Vec::new();
        let mut class_priors = Array1::zeros(n_classes);

        for class_idx in 0..n_classes {
            let mut class_means = Vec::new();
            let mut class_covariances = Vec::new();
            let mut class_component_weights = Array1::zeros(self.n_components);

            let mut class_total_responsibility = 0.0;

            for comp_idx in 0..self.n_components {
                let global_comp_idx = class_idx * self.n_components + comp_idx;
                let comp_responsibilities = responsibilities.column(global_comp_idx);
                let comp_total_resp: f64 = comp_responsibilities.sum();

                class_total_responsibility += comp_total_resp;

                if comp_total_resp > 1e-10 {
                    // Update mean
                    let mut new_mean = Array1::zeros(n_features);
                    for i in 0..n_samples {
                        for j in 0..n_features {
                            new_mean[j] += comp_responsibilities[i] * X[[i, j]];
                        }
                    }
                    new_mean /= comp_total_resp;

                    // Update covariance
                    let mut new_cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_samples {
                        let diff = &X.row(i) - &new_mean;
                        let weight = comp_responsibilities[i];
                        for j in 0..n_features {
                            for k in 0..n_features {
                                new_cov[[j, k]] += weight * diff[j] * diff[k];
                            }
                        }
                    }
                    new_cov /= comp_total_resp;

                    // Add regularization
                    for i in 0..n_features {
                        new_cov[[i, i]] += self.reg_covar;
                    }

                    class_means.push(new_mean);
                    class_covariances.push(new_cov);
                    class_component_weights[comp_idx] = comp_total_resp;
                } else {
                    // Fallback for empty components
                    class_means.push(Array1::zeros(n_features));
                    class_covariances.push(Array2::eye(n_features) * self.reg_covar);
                    class_component_weights[comp_idx] = 1e-10;
                }
            }

            // Normalize component weights
            let total_weight = class_component_weights.sum();
            if total_weight > 0.0 {
                class_component_weights /= total_weight;
            } else {
                class_component_weights.fill(1.0 / self.n_components as f64);
            }

            means.push(class_means);
            covariances.push(class_covariances);
            component_weights.push(class_component_weights);

            // Update class prior
            class_priors[class_idx] = class_total_responsibility / n_samples as f64;
        }

        // Normalize class priors
        let total_prior = class_priors.sum();
        if total_prior > 0.0 {
            class_priors /= total_prior;
        } else {
            class_priors.fill(1.0 / n_classes as f64);
        }

        (means, covariances, component_weights, class_priors)
    }
}

impl Default for MixtureDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MixtureDiscriminantAnalysis<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Trained state for MixtureDiscriminantAnalysis
#[derive(Debug, Clone)]
pub struct MixtureDiscriminantAnalysisTrained {
    /// means
    pub means: Vec<Vec<Array1<f64>>>,
    /// covariances
    pub covariances: Vec<Vec<Array2<f64>>>,
    /// component_weights
    pub component_weights: Vec<Array1<f64>>,
    /// class_priors
    pub class_priors: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// n_components
    pub n_components: usize,
    /// covariance_type
    pub covariance_type: String,
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for MixtureDiscriminantAnalysis<Untrained> {
    type Fitted = MixtureDiscriminantAnalysis<MixtureDiscriminantAnalysisTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        // Identify labeled samples and classes
        let mut labeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();

        // Initialize parameters
        let (mut means, mut covariances, mut component_weights, mut class_priors) =
            self.initialize_parameters(&X, &labeled_indices, &y, &classes)?;

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        // EM algorithm
        for iteration in 0..self.max_iter {
            // E-step
            let (responsibilities, log_likelihood) = self.e_step(
                &X,
                &means,
                &covariances,
                &component_weights,
                &class_priors,
                &labeled_indices,
                &y,
                &classes,
            );

            // M-step
            let (new_means, new_covariances, new_component_weights, new_class_priors) =
                self.m_step(&X, &responsibilities, &classes);

            means = new_means;
            covariances = new_covariances;
            component_weights = new_component_weights;
            class_priors = new_class_priors;

            // Check convergence
            if iteration > 0 && (log_likelihood - prev_log_likelihood).abs() < self.tol {
                break;
            }

            prev_log_likelihood = log_likelihood;
        }

        Ok(MixtureDiscriminantAnalysis {
            state: MixtureDiscriminantAnalysisTrained {
                means,
                covariances,
                component_weights,
                class_priors,
                classes: Array1::from(classes),
                n_components: self.n_components,
                covariance_type: self.covariance_type.clone(),
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            tol: self.tol,
            n_init: self.n_init,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for MixtureDiscriminantAnalysis<MixtureDiscriminantAnalysisTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let probas = self.predict_proba(X)?;
        let n_test = probas.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            let max_idx = probas
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            predictions[i] = self.state.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for MixtureDiscriminantAnalysis<MixtureDiscriminantAnalysisTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut probas = Array2::zeros((n_test, n_classes));

        for i in 0..n_test {
            let x = X.row(i);
            let mut class_probs = Array1::zeros(n_classes);

            // Compute probability for each class
            for class_idx in 0..n_classes {
                let mut class_prob = 0.0;

                // Sum over all components in the class
                for comp_idx in 0..self.state.n_components {
                    let component_prob = self.state.component_weights[class_idx][comp_idx]
                        * self.multivariate_gaussian_pdf(
                            &x,
                            &self.state.means[class_idx][comp_idx],
                            &self.state.covariances[class_idx][comp_idx],
                        );
                    class_prob += component_prob;
                }

                class_probs[class_idx] = self.state.class_priors[class_idx] * class_prob;
            }

            // Normalize probabilities
            let total_prob = class_probs.sum();
            if total_prob > 0.0 {
                class_probs /= total_prob;
            } else {
                class_probs.fill(1.0 / n_classes as f64);
            }

            for j in 0..n_classes {
                probas[[i, j]] = class_probs[j];
            }
        }

        Ok(probas)
    }
}

impl MixtureDiscriminantAnalysis<MixtureDiscriminantAnalysisTrained> {
    /// Compute multivariate Gaussian PDF (same as training method)
    fn multivariate_gaussian_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &Array1<f64>,
        cov: &Array2<f64>,
    ) -> f64 {
        let n_features = x.len();
        let diff = x - mean;

        // Compute determinant and Mahalanobis distance (simplified)
        let det = match self.state.covariance_type.as_str() {
            "spherical" => cov[[0, 0]].powf(n_features as f64),
            "diag" => cov.diag().iter().product(),
            _ => {
                let mut det = 1.0;
                for i in 0..n_features {
                    det *= cov[[i, i]];
                }
                det
            }
        };

        if det <= 0.0 {
            return 1e-10;
        }

        let mut mahal_dist = 0.0;
        match self.state.covariance_type.as_str() {
            "spherical" => {
                let var = cov[[0, 0]];
                mahal_dist = diff.mapv(|x| x * x).sum() / var;
            }
            "diag" => {
                for i in 0..n_features {
                    mahal_dist += diff[i] * diff[i] / cov[[i, i]];
                }
            }
            _ => {
                for i in 0..n_features {
                    mahal_dist += diff[i] * diff[i] / cov[[i, i]];
                }
            }
        }

        let normalization =
            1.0 / ((2.0 * std::f64::consts::PI).powf(n_features as f64 / 2.0) * det.sqrt());
        normalization * (-0.5 * mahal_dist).exp()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_mixture_discriminant_analysis() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let mda = MixtureDiscriminantAnalysis::new()
            .n_components(1)
            .max_iter(10); // Reduced for testing
        let fitted = mda.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_mda_parameters() {
        let mda = MixtureDiscriminantAnalysis::new()
            .n_components(3)
            .covariance_type("diag".to_string())
            .reg_covar(1e-5)
            .max_iter(200)
            .tol(1e-6)
            .n_init(5)
            .random_state(42);

        assert_eq!(mda.n_components, 3);
        assert_eq!(mda.covariance_type, "diag");
        assert_eq!(mda.reg_covar, 1e-5);
        assert_eq!(mda.max_iter, 200);
        assert_eq!(mda.tol, 1e-6);
        assert_eq!(mda.n_init, 5);
        assert_eq!(mda.random_state, Some(42));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mda_covariance_types() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        for cov_type in &["full", "diag", "spherical", "tied"] {
            let mda = MixtureDiscriminantAnalysis::new()
                .covariance_type(cov_type.to_string())
                .max_iter(5);
            let fitted = mda.fit(&X.view(), &y.view()).unwrap();

            let predictions = fitted.predict(&X.view()).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mda_multiple_components() {
        let X = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
            [3.0, 3.0],
            [3.1, 3.1]
        ];
        let y = array![0, 0, -1, 1, 1, -1, -1, -1];

        let mda = MixtureDiscriminantAnalysis::new()
            .n_components(2)
            .max_iter(20);
        let fitted = mda.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 8);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (8, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mda_error_cases() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // No labeled samples

        let mda = MixtureDiscriminantAnalysis::new();
        let result = mda.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_mda_gaussian_pdf() {
        let mda = MixtureDiscriminantAnalysis::new().covariance_type("diag".to_string());

        let x = array![1.0, 2.0];
        let mean = array![1.0, 2.0];
        let cov = Array2::eye(2);

        let pdf = mda.multivariate_gaussian_pdf(&x.view(), &mean, &cov);
        assert!(pdf > 0.0);
        assert!(pdf <= 1.0);
    }
}
