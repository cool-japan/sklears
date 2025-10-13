//! Information-Theoretic Kernel Methods
//!
//! This module implements kernel approximation methods based on information theory,
//! including mutual information kernels, entropy-based feature selection,
//! and KL-divergence kernel features.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::error::Result;

const LOG2: f64 = std::f64::consts::LN_2;

/// Mutual Information Kernel Approximation
///
/// Approximates kernels based on mutual information between features and targets.
/// Uses histogram-based MI estimation with adaptive binning.
#[derive(Debug, Clone)]
/// MutualInformationKernel
pub struct MutualInformationKernel {
    n_components: usize,
    n_bins: usize,
    sigma: f64,
    random_state: Option<u64>,
}

impl MutualInformationKernel {
    /// Create a new Mutual Information Kernel
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_bins: 10,
            sigma: 1.0,
            random_state: None,
        }
    }

    /// Set number of bins for histogram estimation
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set bandwidth parameter
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted Mutual Information Kernel
#[derive(Debug, Clone)]
/// FittedMutualInformationKernel
pub struct FittedMutualInformationKernel {
    feature_weights: Array1<f64>,
    random_features: Array2<f64>,
    mi_scores: Array1<f64>,
}

impl sklears_core::traits::Fit<Array2<f64>, Array1<f64>> for MutualInformationKernel {
    type Fitted = FittedMutualInformationKernel;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let (n_samples, n_features) = x.dim();

        // Compute mutual information scores for each feature
        let mut mi_scores = Array1::zeros(n_features);
        for (i, score) in mi_scores.iter_mut().enumerate() {
            let feature = x.column(i);
            *score = compute_mutual_information(&feature.view(), &y.view(), self.n_bins)?;
        }

        // Normalize MI scores to get feature weights
        let max_mi = mi_scores.fold(0.0_f64, |acc, &x| acc.max(x));
        let feature_weights = if max_mi > 0.0 {
            &mi_scores / max_mi
        } else {
            Array1::ones(n_features) / n_features as f64
        };

        // Generate random features weighted by MI scores
        let normal = RandNormal::new(0.0, 1.0 / self.sigma).unwrap();
        let mut random_features = Array2::zeros((self.n_components, n_features));

        for i in 0..self.n_components {
            for j in 0..n_features {
                let weight = feature_weights[j].sqrt();
                random_features[[i, j]] = rng.sample(normal) * weight;
            }
        }

        Ok(FittedMutualInformationKernel {
            feature_weights,
            random_features,
            mi_scores,
        })
    }
}

impl sklears_core::traits::Transform<Array2<f64>, Array2<f64>> for FittedMutualInformationKernel {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut features = Array2::zeros((n_samples, self.random_features.nrows() * 2));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            for (j, random_feature) in self.random_features.axis_iter(Axis(0)).enumerate() {
                let projection = sample.dot(&random_feature);
                features[[i, 2 * j]] = projection.cos();
                features[[i, 2 * j + 1]] = projection.sin();
            }
        }

        Ok(features)
    }
}

/// Entropy-based Feature Selection
///
/// Selects features based on their entropy and mutual information with targets.
#[derive(Debug, Clone)]
/// EntropyFeatureSelector
pub struct EntropyFeatureSelector {
    n_features: usize,
    selection_method: EntropySelectionMethod,
    n_bins: usize,
}

/// Methods for entropy-based feature selection
#[derive(Debug, Clone)]
/// EntropySelectionMethod
pub enum EntropySelectionMethod {
    /// Select features with highest entropy
    MaxEntropy,
    /// Select features with highest mutual information
    MaxMutualInformation,
    /// Select features optimizing information gain
    InformationGain,
    /// Select features using minimum redundancy maximum relevance
    MRMR,
}

impl EntropyFeatureSelector {
    /// Create a new entropy-based feature selector
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            selection_method: EntropySelectionMethod::MaxMutualInformation,
            n_bins: 10,
        }
    }

    /// Set selection method
    pub fn selection_method(mut self, method: EntropySelectionMethod) -> Self {
        self.selection_method = method;
        self
    }

    /// Set number of bins for entropy estimation
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }
}

/// Fitted Entropy Feature Selector
#[derive(Debug, Clone)]
/// FittedEntropyFeatureSelector
pub struct FittedEntropyFeatureSelector {
    selected_features: Vec<usize>,
    feature_scores: Array1<f64>,
    entropies: Array1<f64>,
}

impl sklears_core::traits::Fit<Array2<f64>, Array1<f64>> for EntropyFeatureSelector {
    type Fitted = FittedEntropyFeatureSelector;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        // Compute entropies and scores for all features
        let mut entropies = Array1::zeros(n_features);
        let mut feature_scores = Array1::zeros(n_features);

        for i in 0..n_features {
            let feature = x.column(i);
            entropies[i] = compute_entropy(&feature.to_owned(), self.n_bins)?;

            feature_scores[i] = match self.selection_method {
                EntropySelectionMethod::MaxEntropy => entropies[i],
                EntropySelectionMethod::MaxMutualInformation => {
                    compute_mutual_information(&feature.view(), &y.view(), self.n_bins)?
                }
                EntropySelectionMethod::InformationGain => {
                    let mi = compute_mutual_information(&feature.view(), &y.view(), self.n_bins)?;
                    let target_entropy = compute_entropy(&y.to_owned(), self.n_bins)?;
                    mi / target_entropy.max(1e-8)
                }
                EntropySelectionMethod::MRMR => {
                    // For MRMR, we'll compute relevance - redundancy
                    let relevance =
                        compute_mutual_information(&feature.view(), &y.view(), self.n_bins)?;
                    // Simplified redundancy computation (would need iterative selection for full MRMR)
                    relevance - entropies[i] * 0.1
                }
            };
        }

        // Select top k features based on scores
        let mut indexed_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_features: Vec<usize> = indexed_scores
            .iter()
            .take(self.n_features)
            .map(|(i, _)| *i)
            .collect();

        Ok(FittedEntropyFeatureSelector {
            selected_features,
            feature_scores,
            entropies,
        })
    }
}

impl sklears_core::traits::Transform<Array2<f64>, Array2<f64>> for FittedEntropyFeatureSelector {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut selected_x = Array2::zeros((n_samples, self.selected_features.len()));

        for (j, &feature_idx) in self.selected_features.iter().enumerate() {
            selected_x.column_mut(j).assign(&x.column(feature_idx));
        }

        Ok(selected_x)
    }
}

/// KL-Divergence Kernel Features
///
/// Generates kernel features based on KL-divergence approximation between distributions.
#[derive(Debug, Clone)]
/// KLDivergenceKernel
pub struct KLDivergenceKernel {
    n_components: usize,
    reference_distribution: KLReferenceDistribution,
    n_bins: usize,
    random_state: Option<u64>,
}

/// Reference distributions for KL-divergence computation
#[derive(Debug, Clone)]
/// KLReferenceDistribution
pub enum KLReferenceDistribution {
    /// Gaussian reference distribution
    Gaussian { mean: f64, std: f64 },
    /// Uniform reference distribution
    Uniform { low: f64, high: f64 },
    /// Empirical distribution from training data
    Empirical,
}

impl KLDivergenceKernel {
    /// Create a new KL-divergence kernel
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            reference_distribution: KLReferenceDistribution::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            n_bins: 20,
            random_state: None,
        }
    }

    /// Set reference distribution
    pub fn reference_distribution(mut self, dist: KLReferenceDistribution) -> Self {
        self.reference_distribution = dist;
        self
    }

    /// Set number of bins for histogram estimation
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted KL-Divergence Kernel
#[derive(Debug, Clone)]
/// FittedKLDivergenceKernel
pub struct FittedKLDivergenceKernel {
    random_projections: Array2<f64>,
    reference_histograms: Vec<Array1<f64>>,
    kl_weights: Array1<f64>,
}

impl sklears_core::traits::Fit<Array2<f64>, ()> for KLDivergenceKernel {
    type Fitted = FittedKLDivergenceKernel;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let (_, n_features) = x.dim();

        // Generate random projections
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut random_projections = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            for j in 0..n_features {
                random_projections[[i, j]] = rng.sample(normal);
            }
        }

        // Compute reference histograms and KL weights
        let mut reference_histograms = Vec::new();
        let mut kl_weights = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            let projection = x.dot(&random_projections.row(i));
            let (hist, bins) = compute_histogram(&projection, self.n_bins)?;

            let reference_hist = match &self.reference_distribution {
                KLReferenceDistribution::Gaussian { mean, std } => {
                    compute_gaussian_reference_histogram(&bins, *mean, *std)
                }
                KLReferenceDistribution::Uniform { low, high } => {
                    compute_uniform_reference_histogram(&bins, *low, *high)
                }
                KLReferenceDistribution::Empirical => hist.clone(),
            };

            let kl_div = compute_kl_divergence(&hist, &reference_hist)?;
            kl_weights[i] = (-kl_div).exp(); // Convert to similarity weight

            reference_histograms.push(reference_hist);
        }

        // Normalize weights
        let weight_sum = kl_weights.sum();
        if weight_sum > 0.0 {
            kl_weights /= weight_sum;
        }

        Ok(FittedKLDivergenceKernel {
            random_projections,
            reference_histograms,
            kl_weights,
        })
    }
}

impl sklears_core::traits::Transform<Array2<f64>, Array2<f64>> for FittedKLDivergenceKernel {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut features = Array2::zeros((n_samples, self.random_projections.nrows()));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            for j in 0..self.random_projections.nrows() {
                let projection = sample.dot(&self.random_projections.row(j));
                let weight = self.kl_weights[j];
                features[[i, j]] = projection * weight;
            }
        }

        Ok(features)
    }
}

/// Information Bottleneck Feature Extractor
///
/// Implements the Information Bottleneck principle for feature extraction,
/// finding representations that maximize information about targets while
/// minimizing information about inputs.
#[derive(Debug, Clone)]
/// InformationBottleneckExtractor
pub struct InformationBottleneckExtractor {
    n_components: usize,
    beta: f64,
    n_bins: usize,
    max_iterations: usize,
    tolerance: f64,
    random_state: Option<u64>,
}

impl InformationBottleneckExtractor {
    /// Create a new Information Bottleneck extractor
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            beta: 1.0,
            n_bins: 10,
            max_iterations: 100,
            tolerance: 1e-6,
            random_state: None,
        }
    }

    /// Set beta parameter (trade-off between compression and prediction)
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set number of bins for discretization
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set maximum iterations for optimization
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted Information Bottleneck Extractor
#[derive(Debug, Clone)]
/// FittedInformationBottleneckExtractor
pub struct FittedInformationBottleneckExtractor {
    cluster_centers: Array2<f64>,
    assignment_probs: Array2<f64>,
    information_values: Array1<f64>,
    n_components: usize,
}

impl sklears_core::traits::Fit<Array2<f64>, Array1<f64>> for InformationBottleneckExtractor {
    type Fitted = FittedInformationBottleneckExtractor;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        let (n_samples, n_features) = x.dim();

        // Initialize cluster centers randomly
        let mut cluster_centers = Array2::zeros((self.n_components, n_features));
        let uniform = RandUniform::new(0, n_samples).unwrap();
        for i in 0..self.n_components {
            let sample_idx = rng.sample(uniform);
            cluster_centers.row_mut(i).assign(&x.row(sample_idx));
        }

        let mut assignment_probs = Array2::zeros((n_samples, self.n_components));
        let mut prev_objective = f64::NEG_INFINITY;

        // Iterate until convergence
        for iteration in 0..self.max_iterations {
            // E-step: Update assignment probabilities
            self.update_assignment_probs(x, &cluster_centers, &mut assignment_probs)?;

            // M-step: Update cluster centers
            self.update_cluster_centers(x, &assignment_probs, &mut cluster_centers)?;

            // Compute objective function
            let objective = self.compute_objective(x, y, &cluster_centers, &assignment_probs)?;

            if (objective - prev_objective).abs() < self.tolerance {
                break;
            }
            prev_objective = objective;
        }

        // Compute information values for each component
        let mut information_values = Array1::zeros(self.n_components);
        for i in 0..self.n_components {
            let cluster_assignments = assignment_probs.column(i);
            information_values[i] =
                compute_mutual_information(&cluster_assignments.view(), &y.view(), self.n_bins)?;
        }

        Ok(FittedInformationBottleneckExtractor {
            cluster_centers,
            assignment_probs: assignment_probs.clone(),
            information_values,
            n_components: self.n_components,
        })
    }
}

impl InformationBottleneckExtractor {
    fn update_assignment_probs(
        &self,
        x: &Array2<f64>,
        cluster_centers: &Array2<f64>,
        assignment_probs: &mut Array2<f64>,
    ) -> Result<()> {
        let (n_samples, _) = x.dim();

        for i in 0..n_samples {
            let sample = x.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            for j in 0..self.n_components {
                let center = cluster_centers.row(j);
                let distance = (&sample - &center).mapv(|x| x * x).sum().sqrt();
                log_probs[j] = -self.beta * distance;
            }

            // Softmax normalization
            let max_log_prob = log_probs.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            log_probs -= max_log_prob;
            let exp_probs = log_probs.mapv(|x| x.exp());
            let prob_sum = exp_probs.sum();

            for j in 0..self.n_components {
                assignment_probs[[i, j]] = exp_probs[j] / prob_sum;
            }
        }

        Ok(())
    }

    fn update_cluster_centers(
        &self,
        x: &Array2<f64>,
        assignment_probs: &Array2<f64>,
        cluster_centers: &mut Array2<f64>,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        for j in 0..self.n_components {
            let mut weighted_sum = Array1::zeros(n_features);
            let mut weight_sum = 0.0;

            for i in 0..n_samples {
                let weight = assignment_probs[[i, j]];
                weighted_sum += &(&x.row(i) * weight);
                weight_sum += weight;
            }

            if weight_sum > 1e-8 {
                cluster_centers
                    .row_mut(j)
                    .assign(&(weighted_sum / weight_sum));
            }
        }

        Ok(())
    }

    fn compute_objective(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        cluster_centers: &Array2<f64>,
        assignment_probs: &Array2<f64>,
    ) -> Result<f64> {
        let (n_samples, _) = x.dim();
        let mut compression_term = 0.0;
        let mut prediction_term = 0.0;

        // Compression term: -I(X; T)
        for i in 0..n_samples {
            for j in 0..self.n_components {
                let prob = assignment_probs[[i, j]];
                if prob > 1e-8 {
                    compression_term -= prob * prob.ln();
                }
            }
        }

        // Prediction term: I(T; Y) - approximate with cluster assignments
        for j in 0..self.n_components {
            let cluster_assignments = assignment_probs.column(j);
            let mi =
                compute_mutual_information(&cluster_assignments.view(), &y.view(), self.n_bins)?;
            prediction_term += mi;
        }

        Ok(prediction_term - self.beta * compression_term)
    }
}

impl sklears_core::traits::Transform<Array2<f64>, Array2<f64>>
    for FittedInformationBottleneckExtractor
{
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let mut features = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = x.row(i);

            for j in 0..self.n_components {
                let center = self.cluster_centers.row(j);
                let distance = (&sample - &center).mapv(|x| x * x).sum().sqrt();
                let weight = self.information_values[j];
                features[[i, j]] = (-distance).exp() * weight;
            }
        }

        Ok(features)
    }
}

// Helper functions for information-theoretic computations

fn compute_entropy(data: &Array1<f64>, n_bins: usize) -> Result<f64> {
    let (hist, _) = compute_histogram(data, n_bins)?;
    let mut entropy = 0.0;

    for &prob in hist.iter() {
        if prob > 1e-12 {
            entropy -= prob * prob.ln();
        }
    }

    Ok(entropy / LOG2) // Convert to bits
}

fn compute_mutual_information(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    n_bins: usize,
) -> Result<f64> {
    let (hist_x, bins_x) = compute_histogram(&x.to_owned(), n_bins)?;
    let (hist_y, bins_y) = compute_histogram(&y.to_owned(), n_bins)?;
    let joint_hist = compute_joint_histogram(x, y, &bins_x, &bins_y)?;

    let mut mi = 0.0;

    for i in 0..n_bins {
        for j in 0..n_bins {
            let p_xy = joint_hist[[i, j]];
            let p_x = hist_x[i];
            let p_y = hist_y[j];

            if p_xy > 1e-12 && p_x > 1e-12 && p_y > 1e-12 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }

    Ok(mi / LOG2) // Convert to bits
}

fn compute_histogram(data: &Array1<f64>, n_bins: usize) -> Result<(Array1<f64>, Array1<f64>)> {
    let min_val = data.fold(f64::INFINITY, |acc, &x| acc.min(x));
    let max_val = data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

    if (max_val - min_val).abs() < 1e-12 {
        let mut hist = Array1::zeros(n_bins);
        hist[0] = 1.0;
        let bins = Array1::linspace(min_val - 0.5, max_val + 0.5, n_bins + 1);
        return Ok((hist, bins));
    }

    let bin_width = (max_val - min_val) / n_bins as f64;
    let bins = Array1::linspace(min_val, max_val + bin_width, n_bins + 1);
    let mut hist = Array1::zeros(n_bins);

    for &value in data.iter() {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        hist[bin_idx] += 1.0;
    }

    hist /= data.len() as f64;
    Ok((hist, bins))
}

fn compute_joint_histogram(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    bins_x: &Array1<f64>,
    bins_y: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_bins_x = bins_x.len() - 1;
    let n_bins_y = bins_y.len() - 1;
    let mut joint_hist = Array2::zeros((n_bins_x, n_bins_y));

    let min_x = bins_x[0];
    let max_x = bins_x[n_bins_x];
    let min_y = bins_y[0];
    let max_y = bins_y[n_bins_y];

    let bin_width_x = (max_x - min_x) / n_bins_x as f64;
    let bin_width_y = (max_y - min_y) / n_bins_y as f64;

    for (&val_x, &val_y) in x.iter().zip(y.iter()) {
        let bin_x = ((val_x - min_x) / bin_width_x).floor() as usize;
        let bin_y = ((val_y - min_y) / bin_width_y).floor() as usize;

        let bin_x = bin_x.min(n_bins_x - 1);
        let bin_y = bin_y.min(n_bins_y - 1);

        joint_hist[[bin_x, bin_y]] += 1.0;
    }

    joint_hist /= x.len() as f64;
    Ok(joint_hist)
}

fn compute_kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64> {
    let mut kl = 0.0;

    for (&p_i, &q_i) in p.iter().zip(q.iter()) {
        if p_i > 1e-12 {
            if q_i > 1e-12 {
                kl += p_i * (p_i / q_i).ln();
            } else {
                return Ok(f64::INFINITY); // KL divergence is infinite
            }
        }
    }

    Ok(kl)
}

fn compute_gaussian_reference_histogram(bins: &Array1<f64>, mean: f64, std: f64) -> Array1<f64> {
    let n_bins = bins.len() - 1;
    let mut hist = Array1::zeros(n_bins);
    let normal = RandNormal::new(mean, std).unwrap();

    for i in 0..n_bins {
        let bin_center = (bins[i] + bins[i + 1]) / 2.0;
        // Use probability density function approximation
        let variance = std * std;
        let exp_part = -0.5 * (bin_center - mean).powi(2) / variance;
        hist[i] = (1.0 / (std * (2.0 * std::f64::consts::PI).sqrt())) * exp_part.exp();
    }

    let hist_sum = hist.sum();
    if hist_sum > 0.0 {
        hist /= hist_sum;
    }

    hist
}

fn compute_uniform_reference_histogram(bins: &Array1<f64>, low: f64, high: f64) -> Array1<f64> {
    let n_bins = bins.len() - 1;
    let mut hist = Array1::zeros(n_bins);
    let uniform_density = 1.0 / (high - low);

    for i in 0..n_bins {
        let bin_start = bins[i];
        let bin_end = bins[i + 1];

        if bin_end <= low || bin_start >= high {
            hist[i] = 0.0;
        } else {
            let overlap_start = bin_start.max(low);
            let overlap_end = bin_end.min(high);
            let overlap_length = overlap_end - overlap_start;
            hist[i] = overlap_length * uniform_density;
        }
    }

    let hist_sum = hist.sum();
    if hist_sum > 0.0 {
        hist /= hist_sum;
    }

    hist
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Transform};

    #[test]
    fn test_mutual_information_kernel() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let mi_kernel = MutualInformationKernel::new(10)
            .n_bins(5)
            .sigma(1.0)
            .random_state(42);

        let fitted = mi_kernel.fit(&x, &y).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[4, 20]); // 10 components * 2 (cos, sin)
        assert!(fitted.mi_scores.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_entropy_feature_selector() {
        let x = array![
            [1.0, 0.1, 100.0],
            [2.0, 0.2, 200.0],
            [3.0, 0.3, 300.0],
            [4.0, 0.4, 400.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let selector = EntropyFeatureSelector::new(2)
            .selection_method(EntropySelectionMethod::MaxMutualInformation)
            .n_bins(3);

        let fitted = selector.fit(&x, &y).unwrap();
        let selected_features = fitted.transform(&x).unwrap();

        assert_eq!(selected_features.shape(), &[4, 2]);
        assert_eq!(fitted.selected_features.len(), 2);
    }

    #[test]
    fn test_kl_divergence_kernel() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let kl_kernel = KLDivergenceKernel::new(5)
            .reference_distribution(KLReferenceDistribution::Gaussian {
                mean: 0.0,
                std: 1.0,
            })
            .n_bins(10)
            .random_state(42);

        let fitted = kl_kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[4, 5]);
        assert!(fitted.kl_weights.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_information_bottleneck_extractor() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let ib_extractor = InformationBottleneckExtractor::new(2)
            .beta(1.0)
            .n_bins(2)
            .max_iterations(10)
            .random_state(42);

        let fitted = ib_extractor.fit(&x, &y).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[4, 2]);
        assert!(fitted.information_values.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_entropy_computation() {
        let data = array![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let entropy = compute_entropy(&data.to_owned(), 3).unwrap();

        // For uniform distribution over 3 bins: H = log2(3) ≈ 1.585
        assert!((entropy - 1.585).abs() < 0.1);
    }

    #[test]
    fn test_mutual_information_computation() {
        let x = array![1.0, 1.0, 2.0, 2.0];
        let y = array![1.0, 1.0, 2.0, 2.0]; // Perfect correlation

        let mi = compute_mutual_information(&x.view(), &y.view(), 2).unwrap();

        // Perfect correlation should give high MI
        assert!(mi > 0.5);
    }
}
