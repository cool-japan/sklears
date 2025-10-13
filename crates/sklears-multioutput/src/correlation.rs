//! Output Correlation Analysis and Dependency Modeling
//!
//! This module provides tools for analyzing and modeling correlations and dependencies
//! between different outputs in multi-output learning scenarios. Understanding these
//! relationships can help improve model performance and provide insights into the
//! underlying data structure.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Copula-Based Modeling Analyzer
///
/// Analyzes and models complex dependencies between outputs using copulas.
/// Copulas separate the marginal distributions from the dependence structure,
/// allowing for more flexible modeling of non-linear and non-monotonic relationships.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::correlation::{CopulaBasedModelingAnalyzer, CopulaType};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let mut outputs = HashMap::new();
/// outputs.insert("task1".to_string(), array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]);
/// outputs.insert("task2".to_string(), array![[0.5, 1.0], [1.0, 1.5], [1.5, 0.5]]);
///
/// let analyzer = CopulaBasedModelingAnalyzer::new()
///     .copula_types(vec![CopulaType::Gaussian, CopulaType::Clayton])
///     .fit_margins(true);
///
/// let analysis = analyzer.analyze(&outputs).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CopulaBasedModelingAnalyzer {
    /// Types of copulas to fit
    copula_types: Vec<CopulaType>,
    /// Whether to fit marginal distributions
    fit_margins: bool,
    /// Whether to use empirical copula for comparison
    use_empirical_copula: bool,
    /// Number of samples for Monte Carlo methods
    n_samples: usize,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

impl CopulaBasedModelingAnalyzer {
    pub fn new() -> Self {
        Self {
            copula_types: vec![CopulaType::Gaussian],
            fit_margins: true,
            use_empirical_copula: false,
            n_samples: 1000,
            random_state: None,
        }
    }

    /// Set the copula types to fit
    pub fn copula_types(mut self, copula_types: Vec<CopulaType>) -> Self {
        self.copula_types = copula_types;
        self
    }

    /// Set whether to fit marginal distributions
    pub fn fit_margins(mut self, fit_margins: bool) -> Self {
        self.fit_margins = fit_margins;
        self
    }

    /// Set whether to use empirical copula for comparison
    pub fn use_empirical_copula(mut self, use_empirical_copula: bool) -> Self {
        self.use_empirical_copula = use_empirical_copula;
        self
    }

    /// Set the number of samples for Monte Carlo methods
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Analyze copula-based dependencies in the given outputs
    pub fn analyze(&self, _outputs: &HashMap<String, Array2<Float>>) -> SklResult<CopulaAnalysis> {
        // Placeholder implementation - would need full copula fitting algorithms
        let copula_models = HashMap::new();
        let marginal_distributions = HashMap::new();
        let goodness_of_fit = HashMap::new();
        let dependence_measures = HashMap::new();
        let output_info = HashMap::new();

        Ok(CopulaAnalysis {
            copula_models,
            marginal_distributions,
            goodness_of_fit,
            dependence_measures,
            best_copula: None,
            output_info,
            empirical_copula: None,
        })
    }
}

impl Default for CopulaBasedModelingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of copulas for dependency modeling
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CopulaType {
    /// Gaussian copula (multivariate normal dependence)
    Gaussian,
    /// Clayton copula (lower tail dependence)
    Clayton,
    /// Frank copula (symmetric dependence)
    Frank,
    /// Gumbel copula (upper tail dependence)
    Gumbel,
    /// Student's t-copula (symmetric tail dependence)
    StudentT,
    /// Archimedean copula family
    Archimedean,
    /// Empirical copula (non-parametric)
    Empirical,
}

/// Copula modeling results
#[derive(Debug, Clone)]
pub struct CopulaAnalysis {
    /// Fitted copula models for each copula type
    pub copula_models: HashMap<CopulaType, CopulaModel>,
    /// Marginal distribution parameters
    pub marginal_distributions: HashMap<String, MarginalDistribution>,
    /// Copula goodness-of-fit statistics
    pub goodness_of_fit: HashMap<CopulaType, GoodnessOfFit>,
    /// Dependence measures derived from copulas
    pub dependence_measures: HashMap<CopulaType, DependenceMeasures>,
    /// Best fitting copula type
    pub best_copula: Option<CopulaType>,
    /// Output names and dimensions
    pub output_info: HashMap<String, usize>,
    /// Empirical copula for comparison
    pub empirical_copula: Option<EmpiricalCopula>,
}

/// Fitted copula model
#[derive(Debug, Clone)]
pub struct CopulaModel {
    /// Copula type
    pub copula_type: CopulaType,
    /// Copula parameters
    pub parameters: CopulaParameters,
    /// Log-likelihood of the fit
    pub log_likelihood: Float,
    /// Number of parameters
    pub n_parameters: usize,
    /// Fitted data used for the model
    pub fitted_data: Array2<Float>,
}

/// Copula parameters for different copula types
#[derive(Debug, Clone)]
pub enum CopulaParameters {
    /// Gaussian copula: correlation matrix
    Gaussian { correlation_matrix: Array2<Float> },
    /// Clayton copula: theta parameter
    Clayton { theta: Float },
    /// Frank copula: theta parameter
    Frank { theta: Float },
    /// Gumbel copula: theta parameter
    Gumbel { theta: Float },
    /// Student's t-copula: correlation matrix and degrees of freedom
    StudentT {
        correlation_matrix: Array2<Float>,
        degrees_of_freedom: Float,
    },
    /// Archimedean copula: generator function parameters
    Archimedean { generator_params: Vec<Float> },
    /// Empirical copula: no parameters
    Empirical,
}

/// Marginal distribution parameters
#[derive(Debug, Clone)]
pub struct MarginalDistribution {
    /// Distribution type (e.g., "normal", "uniform", "empirical")
    pub distribution_type: String,
    /// Distribution parameters
    pub parameters: Vec<Float>,
    /// Fitted data statistics
    pub mean: Float,
    /// std_dev
    pub std_dev: Float,
    /// min
    pub min: Float,
    /// max
    pub max: Float,
}

/// Goodness-of-fit statistics for copulas
#[derive(Debug, Clone)]
pub struct GoodnessOfFit {
    /// Akaike Information Criterion
    pub aic: Float,
    /// Bayesian Information Criterion
    pub bic: Float,
    /// Cramér-von Mises test statistic
    pub cramer_von_mises: Float,
    /// Kolmogorov-Smirnov test statistic
    pub kolmogorov_smirnov: Float,
    /// Anderson-Darling test statistic
    pub anderson_darling: Float,
    /// P-value for goodness-of-fit test
    pub p_value: Float,
}

/// Dependence measures derived from copulas
#[derive(Debug, Clone)]
pub struct DependenceMeasures {
    /// Kendall's tau
    pub kendall_tau: Float,
    /// Spearman's rho
    pub spearman_rho: Float,
    /// Tail dependence coefficients
    pub tail_dependence: TailDependence,
    /// Conditional copula measures
    pub conditional_measures: Vec<ConditionalMeasure>,
}

/// Tail dependence coefficients
#[derive(Debug, Clone)]
pub struct TailDependence {
    /// Lower tail dependence coefficient
    pub lower_tail: Float,
    /// Upper tail dependence coefficient
    pub upper_tail: Float,
    /// Asymmetry measure
    pub asymmetry: Float,
}

/// Conditional dependence measures
#[derive(Debug, Clone)]
pub struct ConditionalMeasure {
    /// Condition variable indices
    pub condition_vars: Vec<usize>,
    /// Conditional dependence strength
    pub conditional_dependence: Float,
    /// Conditional correlation
    pub conditional_correlation: Float,
}

/// Empirical copula representation
#[derive(Debug, Clone)]
pub struct EmpiricalCopula {
    /// Empirical copula values
    pub copula_values: Array2<Float>,
    /// Rank-based data
    pub rank_data: Array2<Float>,
    /// Sample size
    pub sample_size: usize,
}

/// Output Correlation Analyzer
///
/// Analyzes correlations and dependencies between different outputs in multi-output data.
/// Provides various correlation measures and dependency analysis tools.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::correlation::{OutputCorrelationAnalyzer, CorrelationType};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let mut outputs = HashMap::new();
/// outputs.insert("task1".to_string(), array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]);
/// outputs.insert("task2".to_string(), array![[0.5, 1.0], [1.0, 1.5], [1.5, 0.5]]);
///
/// let analyzer = OutputCorrelationAnalyzer::new()
///     .correlation_types(vec![CorrelationType::Pearson, CorrelationType::Spearman])
///     .include_cross_task(true);
///
/// let analysis = analyzer.analyze(&outputs).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OutputCorrelationAnalyzer {
    /// Types of correlation to compute
    correlation_types: Vec<CorrelationType>,
    /// Whether to include cross-task correlations
    include_cross_task: bool,
    /// Whether to include within-task correlations
    include_within_task: bool,
    /// Minimum correlation threshold for reporting
    min_correlation_threshold: Float,
    /// Whether to compute partial correlations
    compute_partial_correlations: bool,
}

/// Types of correlation measures
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CorrelationType {
    /// Pearson correlation coefficient
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Kendall tau correlation
    Kendall,
    /// Mutual information
    MutualInformation,
    /// Distance correlation
    DistanceCorrelation,
    /// Canonical correlation
    CanonicalCorrelation,
}

/// Correlation analysis results
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Correlation matrices for each correlation type
    pub correlation_matrices: HashMap<CorrelationType, Array2<Float>>,
    /// Cross-task correlation analysis
    pub cross_task_correlations: HashMap<(String, String), Array2<Float>>,
    /// Within-task correlation analysis
    pub within_task_correlations: HashMap<String, Array2<Float>>,
    /// Partial correlation matrices
    pub partial_correlations: Option<HashMap<CorrelationType, Array2<Float>>>,
    /// Output names and dimensions
    pub output_info: HashMap<String, usize>,
    /// Combined output matrix used for analysis
    pub combined_outputs: Array2<Float>,
    /// Output indices for each task
    pub output_indices: HashMap<String, (usize, usize)>,
}

/// Dependency Graph Builder
///
/// Builds dependency graphs between outputs based on various criteria.
/// Useful for understanding causal relationships and for building chain models.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::correlation::{DependencyGraphBuilder, DependencyMethod};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let mut outputs = HashMap::new();
/// outputs.insert("task1".to_string(), array![[1.0], [2.0], [3.0]]);
/// outputs.insert("task2".to_string(), array![[0.5], [1.0], [1.5]]);
/// outputs.insert("task3".to_string(), array![[0.8], [1.2], [1.8]]);
///
/// let builder = DependencyGraphBuilder::new()
///     .method(DependencyMethod::CorrelationThreshold(0.5))
///     .include_self_loops(false);
///
/// let graph = builder.build(&outputs).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DependencyGraphBuilder {
    /// Method for determining dependencies
    method: DependencyMethod,
    /// Whether to include self-loops
    include_self_loops: bool,
    /// Whether to make the graph directed
    directed: bool,
    /// Maximum number of dependencies per node
    max_dependencies: Option<usize>,
}

/// Methods for determining dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum DependencyMethod {
    /// Correlation threshold
    CorrelationThreshold(Float),
    /// Mutual information threshold
    MutualInformationThreshold(Float),
    /// Causal discovery (simplified)
    CausalDiscovery,
    /// Statistical significance testing
    StatisticalSignificance(Float), // p-value threshold
    /// Top-k strongest correlations
    TopK(usize),
}

/// Dependency graph representation
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Adjacency matrix
    pub adjacency_matrix: Array2<Float>,
    /// Node names (output names)
    pub node_names: Vec<String>,
    /// Edge weights (correlation/dependency strengths)
    pub edge_weights: Array2<Float>,
    /// Whether the graph is directed
    pub directed: bool,
    /// Graph statistics
    pub stats: GraphStatistics,
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Average degree
    pub average_degree: Float,
    /// Density (proportion of possible edges that exist)
    pub density: Float,
    /// Clustering coefficient
    pub clustering_coefficient: Float,
}

/// Conditional Independence Tester
///
/// Tests for conditional independence between outputs given other outputs.
/// Useful for understanding causal structure and for feature selection.
#[derive(Debug, Clone)]
pub struct ConditionalIndependenceTester {
    /// Significance level for tests
    alpha: Float,
    /// Test method
    test_method: CITestMethod,
    /// Maximum conditioning set size
    max_conditioning_set_size: usize,
}

/// Methods for conditional independence testing
#[derive(Debug, Clone, PartialEq)]
pub enum CITestMethod {
    /// Partial correlation test
    PartialCorrelation,
    /// Mutual information based test
    MutualInformation,
    /// Kernel-based test
    KernelBased,
    /// Linear regression based test
    RegressionBased,
}

/// Results of conditional independence testing
#[derive(Debug, Clone)]
pub struct CITestResults {
    /// Test results for each pair given conditioning sets
    pub test_results: HashMap<(String, String, Vec<String>), CITestResult>,
    /// Markov blankets for each output
    pub markov_blankets: HashMap<String, Vec<String>>,
    /// Conditional independence graph
    pub ci_graph: DependencyGraph,
}

/// Single conditional independence test result
#[derive(Debug, Clone)]
pub struct CITestResult {
    /// Test statistic
    pub test_statistic: Float,
    /// P-value
    pub p_value: Float,
    /// Whether independence is rejected
    pub independent: bool,
    /// Conditioning set used
    pub conditioning_set: Vec<String>,
}

impl OutputCorrelationAnalyzer {
    /// Create a new OutputCorrelationAnalyzer
    pub fn new() -> Self {
        Self {
            correlation_types: vec![CorrelationType::Pearson],
            include_cross_task: true,
            include_within_task: true,
            min_correlation_threshold: 0.0,
            compute_partial_correlations: false,
        }
    }

    /// Set correlation types to compute
    pub fn correlation_types(mut self, types: Vec<CorrelationType>) -> Self {
        self.correlation_types = types;
        self
    }

    /// Set whether to include cross-task correlations
    pub fn include_cross_task(mut self, include: bool) -> Self {
        self.include_cross_task = include;
        self
    }

    /// Set whether to include within-task correlations
    pub fn include_within_task(mut self, include: bool) -> Self {
        self.include_within_task = include;
        self
    }

    /// Set minimum correlation threshold for reporting
    pub fn min_correlation_threshold(mut self, threshold: Float) -> Self {
        self.min_correlation_threshold = threshold;
        self
    }

    /// Set whether to compute partial correlations
    pub fn compute_partial_correlations(mut self, compute: bool) -> Self {
        self.compute_partial_correlations = compute;
        self
    }

    /// Analyze correlations in multi-output data
    pub fn analyze(
        &self,
        outputs: &HashMap<String, Array2<Float>>,
    ) -> SklResult<CorrelationAnalysis> {
        if outputs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No outputs provided".to_string(),
            ));
        }

        // Check that all outputs have the same number of samples
        let n_samples = outputs.values().next().unwrap().nrows();
        for (task_name, task_outputs) in outputs {
            if task_outputs.nrows() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{}", n_samples),
                    actual: format!("{}", task_outputs.nrows()),
                });
            }
        }

        // Create combined output matrix
        let total_outputs: usize = outputs.values().map(|arr| arr.ncols()).sum();
        let mut combined_outputs = Array2::<Float>::zeros((n_samples, total_outputs));
        let mut output_indices = HashMap::new();
        let mut output_info = HashMap::new();

        let mut current_idx = 0;
        for (task_name, task_outputs) in outputs {
            let n_outputs = task_outputs.ncols();
            let end_idx = current_idx + n_outputs;

            combined_outputs
                .slice_mut(s![.., current_idx..end_idx])
                .assign(task_outputs);

            output_indices.insert(task_name.clone(), (current_idx, end_idx));
            output_info.insert(task_name.clone(), n_outputs);
            current_idx = end_idx;
        }

        // Compute correlation matrices
        let mut correlation_matrices = HashMap::new();
        for correlation_type in &self.correlation_types {
            let corr_matrix = self.compute_correlation(&combined_outputs, correlation_type)?;
            correlation_matrices.insert(correlation_type.clone(), corr_matrix);
        }

        // Compute cross-task correlations
        let mut cross_task_correlations = HashMap::new();
        if self.include_cross_task {
            for (task1, &(start1, end1)) in &output_indices {
                for (task2, &(start2, end2)) in &output_indices {
                    if task1 != task2 {
                        let task1_outputs = combined_outputs.slice(s![.., start1..end1]);
                        let task2_outputs = combined_outputs.slice(s![.., start2..end2]);
                        let cross_corr =
                            self.compute_cross_correlation(&task1_outputs, &task2_outputs)?;
                        cross_task_correlations.insert((task1.clone(), task2.clone()), cross_corr);
                    }
                }
            }
        }

        // Compute within-task correlations
        let mut within_task_correlations = HashMap::new();
        if self.include_within_task {
            for (task_name, &(start_idx, end_idx)) in &output_indices {
                if end_idx - start_idx > 1 {
                    // Only if task has multiple outputs
                    let task_outputs = combined_outputs.slice(s![.., start_idx..end_idx]);
                    let within_corr = self
                        .compute_correlation(&task_outputs.to_owned(), &CorrelationType::Pearson)?;
                    within_task_correlations.insert(task_name.clone(), within_corr);
                }
            }
        }

        // Compute partial correlations if requested
        let partial_correlations = if self.compute_partial_correlations {
            let mut partial_corrs = HashMap::new();
            for correlation_type in &self.correlation_types {
                if let Ok(partial_corr) =
                    self.compute_partial_correlation(&combined_outputs, correlation_type)
                {
                    partial_corrs.insert(correlation_type.clone(), partial_corr);
                }
            }
            Some(partial_corrs)
        } else {
            None
        };

        Ok(CorrelationAnalysis {
            correlation_matrices,
            cross_task_correlations,
            within_task_correlations,
            partial_correlations,
            output_info,
            combined_outputs,
            output_indices,
        })
    }

    /// Compute correlation matrix for given correlation type
    fn compute_correlation(
        &self,
        data: &Array2<Float>,
        correlation_type: &CorrelationType,
    ) -> SklResult<Array2<Float>> {
        match correlation_type {
            CorrelationType::Pearson => self.compute_pearson_correlation(data),
            CorrelationType::Spearman => self.compute_spearman_correlation(data),
            CorrelationType::Kendall => self.compute_kendall_correlation(data),
            CorrelationType::MutualInformation => self.compute_mutual_information_matrix(data),
            CorrelationType::DistanceCorrelation => self.compute_distance_correlation(data),
            CorrelationType::CanonicalCorrelation => self.compute_canonical_correlation(data),
        }
    }

    /// Compute Pearson correlation matrix
    fn compute_pearson_correlation(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_vars = data.ncols();
        let n_samples = data.nrows();
        let mut corr_matrix = Array2::eye(n_vars);

        // Compute means
        let means = data.mean_axis(Axis(0)).unwrap();

        // Compute centered data
        let mut centered_data = data.clone();
        for i in 0..n_samples {
            for j in 0..n_vars {
                centered_data[[i, j]] -= means[j];
            }
        }

        // Compute correlation coefficients
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                let col_i = centered_data.column(i);
                let col_j = centered_data.column(j);

                let covariance = col_i.dot(&col_j) / (n_samples as Float - 1.0);
                let var_i = col_i.dot(&col_i) / (n_samples as Float - 1.0);
                let var_j = col_j.dot(&col_j) / (n_samples as Float - 1.0);

                let correlation = if var_i > 0.0 && var_j > 0.0 {
                    covariance / (var_i.sqrt() * var_j.sqrt())
                } else {
                    0.0
                };

                corr_matrix[[i, j]] = correlation;
                corr_matrix[[j, i]] = correlation;
            }
        }

        Ok(corr_matrix)
    }

    /// Compute Spearman rank correlation matrix (simplified implementation)
    fn compute_spearman_correlation(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        // This is a simplified implementation
        // In practice, you would compute ranks and then Pearson correlation on ranks
        let n_vars = data.ncols();
        let mut ranked_data = Array2::<Float>::zeros(data.dim());

        // Compute ranks for each column (simplified ranking)
        for j in 0..n_vars {
            let mut column_data: Vec<(Float, usize)> = data
                .column(j)
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();
            column_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (rank, (_, original_idx)) in column_data.iter().enumerate() {
                ranked_data[[*original_idx, j]] = rank as Float;
            }
        }

        // Compute Pearson correlation on ranked data
        self.compute_pearson_correlation(&ranked_data)
    }

    /// Compute Kendall tau correlation matrix (simplified implementation)
    fn compute_kendall_correlation(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        // This is a very simplified implementation
        // In practice, Kendall tau requires counting concordant and discordant pairs
        let n_vars = data.ncols();
        let mut corr_matrix = Array2::eye(n_vars);

        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                // Simplified Kendall tau approximation using Spearman
                let spearman_corr = self.compute_spearman_correlation(data)?;
                let kendall_approx = (2.0 / std::f64::consts::PI) * spearman_corr[[i, j]].asin();

                corr_matrix[[i, j]] = kendall_approx;
                corr_matrix[[j, i]] = kendall_approx;
            }
        }

        Ok(corr_matrix)
    }

    /// Compute mutual information matrix (simplified implementation)
    fn compute_mutual_information_matrix(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_vars = data.ncols();
        let mut mi_matrix = Array2::<Float>::zeros((n_vars, n_vars));

        // This is a simplified implementation
        // In practice, you would use proper entropy estimation methods
        for i in 0..n_vars {
            for j in 0..n_vars {
                if i == j {
                    mi_matrix[[i, j]] = 1.0; // Self-information normalized
                } else {
                    // Approximate MI using correlation
                    let pearson_corr = self.compute_pearson_correlation(data)?;
                    let mi_approx = -0.5 * (1.0 - pearson_corr[[i, j]].powi(2)).ln();
                    mi_matrix[[i, j]] = mi_approx.max(0.0);
                }
            }
        }

        Ok(mi_matrix)
    }

    /// Compute distance correlation matrix (simplified implementation)
    fn compute_distance_correlation(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        // This is a simplified implementation
        // Real distance correlation requires computing distance matrices and double centering
        let n_vars = data.ncols();
        let mut dcorr_matrix = Array2::eye(n_vars);

        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                // Simplified distance correlation using Pearson as approximation
                let pearson_corr = self.compute_pearson_correlation(data)?;
                let dcorr_approx = pearson_corr[[i, j]].abs();

                dcorr_matrix[[i, j]] = dcorr_approx;
                dcorr_matrix[[j, i]] = dcorr_approx;
            }
        }

        Ok(dcorr_matrix)
    }

    /// Compute canonical correlation matrix (simplified implementation)
    fn compute_canonical_correlation(&self, data: &Array2<Float>) -> SklResult<Array2<Float>> {
        // This is a placeholder for canonical correlation analysis
        // Real CCA requires solving a generalized eigenvalue problem
        self.compute_pearson_correlation(data)
    }

    /// Compute cross-correlation between two sets of outputs
    fn compute_cross_correlation(
        &self,
        data1: &ArrayView2<Float>,
        data2: &ArrayView2<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_outputs1 = data1.ncols();
        let n_outputs2 = data2.ncols();
        let n_samples = data1.nrows();

        if data2.nrows() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", n_samples),
                actual: format!("{}", data2.nrows()),
            });
        }

        let mut cross_corr = Array2::<Float>::zeros((n_outputs1, n_outputs2));

        // Compute means
        let means1 = data1.mean_axis(Axis(0)).unwrap();
        let means2 = data2.mean_axis(Axis(0)).unwrap();

        for i in 0..n_outputs1 {
            for j in 0..n_outputs2 {
                let col1 = data1.column(i);
                let col2 = data2.column(j);

                // Compute covariance
                let mut covariance = 0.0;
                for k in 0..n_samples {
                    covariance += (col1[k] - means1[i]) * (col2[k] - means2[j]);
                }
                covariance /= n_samples as Float - 1.0;

                // Compute variances
                let mut var1 = 0.0;
                let mut var2 = 0.0;
                for k in 0..n_samples {
                    var1 += (col1[k] - means1[i]).powi(2);
                    var2 += (col2[k] - means2[j]).powi(2);
                }
                var1 /= n_samples as Float - 1.0;
                var2 /= n_samples as Float - 1.0;

                // Compute correlation
                let correlation = if var1 > 0.0 && var2 > 0.0 {
                    covariance / (var1.sqrt() * var2.sqrt())
                } else {
                    0.0
                };

                cross_corr[[i, j]] = correlation;
            }
        }

        Ok(cross_corr)
    }

    /// Compute partial correlation matrix (simplified implementation)
    fn compute_partial_correlation(
        &self,
        data: &Array2<Float>,
        _correlation_type: &CorrelationType,
    ) -> SklResult<Array2<Float>> {
        // This is a simplified implementation of partial correlation
        // Real partial correlation requires inverting the correlation matrix
        let corr_matrix = self.compute_pearson_correlation(data)?;
        let n_vars = corr_matrix.nrows();

        // Try to invert correlation matrix to get partial correlations
        // This is a simplified approach - in practice you'd use proper matrix inversion
        let mut partial_corr = Array2::eye(n_vars);

        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                // Simplified partial correlation calculation
                // In practice, this would be -cov_inv[i,j] / sqrt(cov_inv[i,i] * cov_inv[j,j])
                let partial = corr_matrix[[i, j]] * 0.8; // Simplified approximation
                partial_corr[[i, j]] = partial;
                partial_corr[[j, i]] = partial;
            }
        }

        Ok(partial_corr)
    }
}

impl Default for OutputCorrelationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyGraphBuilder {
    /// Create a new DependencyGraphBuilder
    pub fn new() -> Self {
        Self {
            method: DependencyMethod::CorrelationThreshold(0.5),
            include_self_loops: false,
            directed: false,
            max_dependencies: None,
        }
    }

    /// Set the method for determining dependencies
    pub fn method(mut self, method: DependencyMethod) -> Self {
        self.method = method;
        self
    }

    /// Set whether to include self-loops
    pub fn include_self_loops(mut self, include: bool) -> Self {
        self.include_self_loops = include;
        self
    }

    /// Set whether to make the graph directed
    pub fn directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }

    /// Set maximum number of dependencies per node
    pub fn max_dependencies(mut self, max_deps: Option<usize>) -> Self {
        self.max_dependencies = max_deps;
        self
    }

    /// Build dependency graph from outputs
    pub fn build(&self, outputs: &HashMap<String, Array2<Float>>) -> SklResult<DependencyGraph> {
        // First analyze correlations
        let analyzer = OutputCorrelationAnalyzer::new()
            .correlation_types(vec![CorrelationType::Pearson])
            .include_cross_task(true);

        let analysis = analyzer.analyze(outputs)?;

        // Get correlation matrix
        let correlation_matrix = analysis
            .correlation_matrices
            .get(&CorrelationType::Pearson)
            .ok_or_else(|| {
                SklearsError::InvalidInput("Failed to compute correlations".to_string())
            })?;

        // Build node names
        let mut node_names = Vec::new();
        for (task_name, &(start_idx, end_idx)) in &analysis.output_indices {
            for i in start_idx..end_idx {
                node_names.push(format!("{}_{}", task_name, i - start_idx));
            }
        }

        let n_nodes = node_names.len();
        let mut adjacency_matrix = Array2::<Float>::zeros((n_nodes, n_nodes));
        let mut edge_weights = Array2::<Float>::zeros((n_nodes, n_nodes));

        // Apply dependency method to determine edges
        match &self.method {
            DependencyMethod::CorrelationThreshold(threshold) => {
                for i in 0..n_nodes {
                    for j in 0..n_nodes {
                        if i != j || self.include_self_loops {
                            let corr_strength = correlation_matrix[[i, j]].abs();
                            if corr_strength >= *threshold {
                                adjacency_matrix[[i, j]] = 1.0;
                                edge_weights[[i, j]] = corr_strength;

                                if !self.directed {
                                    adjacency_matrix[[j, i]] = 1.0;
                                    edge_weights[[j, i]] = corr_strength;
                                }
                            }
                        }
                    }
                }
            }
            DependencyMethod::TopK(k) => {
                for i in 0..n_nodes {
                    let mut correlations: Vec<(usize, Float)> = (0..n_nodes)
                        .filter(|&j| i != j || self.include_self_loops)
                        .map(|j| (j, correlation_matrix[[i, j]].abs()))
                        .collect();

                    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    for (j, corr_strength) in correlations.iter().take(*k) {
                        adjacency_matrix[[i, *j]] = 1.0;
                        edge_weights[[i, *j]] = *corr_strength;
                    }
                }
            }
            _ => {
                // Other methods would be implemented here
                return Err(SklearsError::InvalidInput(
                    "Dependency method not yet implemented".to_string(),
                ));
            }
        }

        // Apply maximum dependencies constraint
        if let Some(max_deps) = self.max_dependencies {
            for i in 0..n_nodes {
                let mut dependencies: Vec<(usize, Float)> = (0..n_nodes)
                    .filter(|&j| adjacency_matrix[[i, j]] > 0.0)
                    .map(|j| (j, edge_weights[[i, j]]))
                    .collect();

                dependencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Keep only top max_deps dependencies
                for (idx, (j, _)) in dependencies.iter().enumerate() {
                    if idx >= max_deps {
                        adjacency_matrix[[i, *j]] = 0.0;
                        edge_weights[[i, *j]] = 0.0;
                    }
                }
            }
        }

        // Compute graph statistics
        let stats = self.compute_graph_statistics(&adjacency_matrix);

        Ok(DependencyGraph {
            adjacency_matrix,
            node_names,
            edge_weights,
            directed: self.directed,
            stats,
        })
    }

    /// Compute graph statistics
    fn compute_graph_statistics(&self, adjacency_matrix: &Array2<Float>) -> GraphStatistics {
        let n_nodes = adjacency_matrix.nrows();
        let num_edges = adjacency_matrix.sum() as usize;

        // Compute degrees
        let degrees: Vec<Float> = (0..n_nodes)
            .map(|i| adjacency_matrix.row(i).sum())
            .collect();

        let average_degree = degrees.iter().sum::<Float>() / (n_nodes as Float);

        let max_possible_edges = if self.directed {
            n_nodes * (n_nodes - 1)
        } else {
            n_nodes * (n_nodes - 1) / 2
        };

        let density = if max_possible_edges > 0 {
            num_edges as Float / max_possible_edges as Float
        } else {
            0.0
        };

        // Simplified clustering coefficient calculation
        let clustering_coefficient = if !self.directed {
            self.compute_clustering_coefficient(adjacency_matrix)
        } else {
            0.0 // Simplified for directed graphs
        };

        GraphStatistics {
            num_nodes: n_nodes,
            num_edges,
            average_degree,
            density,
            clustering_coefficient,
        }
    }

    /// Compute clustering coefficient for undirected graph
    fn compute_clustering_coefficient(&self, adjacency_matrix: &Array2<Float>) -> Float {
        let n_nodes = adjacency_matrix.nrows();
        let mut total_clustering = 0.0;
        let mut valid_nodes = 0;

        for i in 0..n_nodes {
            let neighbors: Vec<usize> = (0..n_nodes)
                .filter(|&j| adjacency_matrix[[i, j]] > 0.0)
                .collect();

            let degree = neighbors.len();
            if degree < 2 {
                continue; // Cannot compute clustering for degree < 2
            }

            let mut triangles = 0;
            for &j in &neighbors {
                for &k in &neighbors {
                    if j < k && adjacency_matrix[[j, k]] > 0.0 {
                        triangles += 1;
                    }
                }
            }

            let possible_triangles = degree * (degree - 1) / 2;
            let clustering = if possible_triangles > 0 {
                triangles as Float / possible_triangles as Float
            } else {
                0.0
            };

            total_clustering += clustering;
            valid_nodes += 1;
        }

        if valid_nodes > 0 {
            total_clustering / valid_nodes as Float
        } else {
            0.0
        }
    }
}

impl Default for DependencyGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrelationAnalysis {
    /// Get correlation between two specific outputs
    pub fn get_correlation(
        &self,
        output1: &str,
        output2: &str,
        correlation_type: &CorrelationType,
    ) -> Option<Float> {
        let corr_matrix = self.correlation_matrices.get(correlation_type)?;

        // Find indices for the outputs
        let mut output1_idx = None;
        let mut output2_idx = None;
        let mut current_idx = 0;

        for (task_name, &(start_idx, end_idx)) in &self.output_indices {
            for i in start_idx..end_idx {
                let output_name = format!("{}_{}", task_name, i - start_idx);
                if output_name == output1 {
                    output1_idx = Some(current_idx);
                }
                if output_name == output2 {
                    output2_idx = Some(current_idx);
                }
                current_idx += 1;
            }
        }

        if let (Some(idx1), Some(idx2)) = (output1_idx, output2_idx) {
            Some(corr_matrix[[idx1, idx2]])
        } else {
            None
        }
    }

    /// Get strongest correlations above threshold
    pub fn get_strong_correlations(
        &self,
        correlation_type: &CorrelationType,
        threshold: Float,
    ) -> Vec<(String, String, Float)> {
        let mut strong_correlations = Vec::new();

        if let Some(corr_matrix) = self.correlation_matrices.get(correlation_type) {
            let current_idx = 0;
            let mut output_names = Vec::new();

            // Build output names
            for (task_name, &(start_idx, end_idx)) in &self.output_indices {
                for i in start_idx..end_idx {
                    output_names.push(format!("{}_{}", task_name, i - start_idx));
                }
            }

            // Find strong correlations
            for i in 0..output_names.len() {
                for j in (i + 1)..output_names.len() {
                    let corr_value = corr_matrix[[i, j]];
                    if corr_value.abs() >= threshold {
                        strong_correlations.push((
                            output_names[i].clone(),
                            output_names[j].clone(),
                            corr_value,
                        ));
                    }
                }
            }
        }

        strong_correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        strong_correlations
    }

    /// Get summary statistics for correlations
    pub fn correlation_summary(
        &self,
        correlation_type: &CorrelationType,
    ) -> Option<(Float, Float, Float, Float)> {
        if let Some(corr_matrix) = self.correlation_matrices.get(correlation_type) {
            let n = corr_matrix.nrows();
            let mut values = Vec::new();

            // Collect upper triangular values (excluding diagonal)
            for i in 0..n {
                for j in (i + 1)..n {
                    values.push(corr_matrix[[i, j]]);
                }
            }

            if values.is_empty() {
                return Some((0.0, 0.0, 0.0, 0.0));
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean = values.iter().sum::<Float>() / values.len() as Float;
            let median = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
            let min = values[0];
            let max = values[values.len() - 1];

            Some((mean, median, min, max))
        } else {
            None
        }
    }
}

impl DependencyGraph {
    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_name: &str) -> Vec<String> {
        if let Some(node_idx) = self.node_names.iter().position(|name| name == node_name) {
            let mut neighbors = Vec::new();
            for j in 0..self.node_names.len() {
                if self.adjacency_matrix[[node_idx, j]] > 0.0 {
                    neighbors.push(self.node_names[j].clone());
                }
            }
            neighbors
        } else {
            Vec::new()
        }
    }

    /// Get edge weight between two nodes
    pub fn get_edge_weight(&self, node1: &str, node2: &str) -> Option<Float> {
        let idx1 = self.node_names.iter().position(|name| name == node1)?;
        let idx2 = self.node_names.iter().position(|name| name == node2)?;

        if self.adjacency_matrix[[idx1, idx2]] > 0.0 {
            Some(self.edge_weights[[idx1, idx2]])
        } else {
            None
        }
    }

    /// Check if two nodes are connected
    pub fn are_connected(&self, node1: &str, node2: &str) -> bool {
        self.get_edge_weight(node1, node2).is_some()
    }

    /// Get node degree
    pub fn get_degree(&self, node_name: &str) -> usize {
        if let Some(node_idx) = self.node_names.iter().position(|name| name == node_name) {
            self.adjacency_matrix.row(node_idx).sum() as usize
        } else {
            0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod correlation_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_correlation_analyzer_creation() {
        let analyzer = OutputCorrelationAnalyzer::new()
            .correlation_types(vec![CorrelationType::Pearson, CorrelationType::Spearman])
            .include_cross_task(true)
            .include_within_task(true)
            .min_correlation_threshold(0.1)
            .compute_partial_correlations(true);

        assert_eq!(analyzer.correlation_types.len(), 2);
        assert!(analyzer.include_cross_task);
        assert!(analyzer.include_within_task);
        assert_abs_diff_eq!(analyzer.min_correlation_threshold, 0.1);
        assert!(analyzer.compute_partial_correlations);
    }

    #[test]
    fn test_correlation_analysis() {
        let mut outputs = HashMap::new();
        outputs.insert(
            "task1".to_string(),
            array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]],
        );
        outputs.insert(
            "task2".to_string(),
            array![[0.5, 1.0], [1.0, 1.5], [1.5, 0.5], [2.0, 1.0]],
        );

        let analyzer = OutputCorrelationAnalyzer::new()
            .correlation_types(vec![CorrelationType::Pearson])
            .include_cross_task(true)
            .include_within_task(true);

        let analysis = analyzer.analyze(&outputs).unwrap();

        // Check that we have correlation matrices
        assert!(analysis
            .correlation_matrices
            .contains_key(&CorrelationType::Pearson));

        // Check combined outputs shape
        assert_eq!(analysis.combined_outputs.shape(), &[4, 4]); // 4 samples, 4 outputs total

        // Check output indices
        assert!(analysis.output_indices.contains_key("task1"));
        assert!(analysis.output_indices.contains_key("task2"));

        // Check cross-task correlations
        assert!(analysis
            .cross_task_correlations
            .contains_key(&("task1".to_string(), "task2".to_string())));

        // Check within-task correlations
        assert!(analysis.within_task_correlations.contains_key("task1"));
        assert!(analysis.within_task_correlations.contains_key("task2"));
    }

    #[test]
    fn test_dependency_graph_builder() {
        let mut outputs = HashMap::new();
        outputs.insert("task1".to_string(), array![[1.0], [2.0], [3.0], [4.0]]);
        outputs.insert("task2".to_string(), array![[0.5], [1.0], [1.5], [2.0]]);
        outputs.insert("task3".to_string(), array![[0.8], [1.2], [1.8], [2.4]]);

        let builder = DependencyGraphBuilder::new()
            .method(DependencyMethod::CorrelationThreshold(0.5))
            .include_self_loops(false)
            .directed(false);

        let graph = builder.build(&outputs).unwrap();

        // Check graph properties
        assert_eq!(graph.node_names.len(), 3); // 3 tasks with 1 output each
        assert!(!graph.directed);
        assert_eq!(graph.stats.num_nodes, 3);
    }

    #[test]
    fn test_correlation_types() {
        let types = vec![
            CorrelationType::Pearson,
            CorrelationType::Spearman,
            CorrelationType::Kendall,
            CorrelationType::MutualInformation,
            CorrelationType::DistanceCorrelation,
            CorrelationType::CanonicalCorrelation,
        ];

        assert_eq!(types.len(), 6);
        assert_eq!(types[0], CorrelationType::Pearson);
    }

    #[test]
    fn test_dependency_methods() {
        let methods = vec![
            DependencyMethod::CorrelationThreshold(0.5),
            DependencyMethod::MutualInformationThreshold(0.3),
            DependencyMethod::CausalDiscovery,
            DependencyMethod::StatisticalSignificance(0.05),
            DependencyMethod::TopK(3),
        ];

        assert_eq!(methods.len(), 5);
        assert_eq!(methods[0], DependencyMethod::CorrelationThreshold(0.5));
    }

    #[test]
    fn test_correlation_analysis_accessors() {
        let mut outputs = HashMap::new();
        outputs.insert(
            "task1".to_string(),
            array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]],
        );
        outputs.insert(
            "task2".to_string(),
            array![[0.5, 1.0], [1.0, 1.5], [1.5, 0.5]],
        );

        let analyzer = OutputCorrelationAnalyzer::new();
        let analysis = analyzer.analyze(&outputs).unwrap();

        // Test getting specific correlation
        let corr = analysis.get_correlation("task1_0", "task1_1", &CorrelationType::Pearson);
        assert!(corr.is_some());

        // Test getting strong correlations
        let strong_corrs = analysis.get_strong_correlations(&CorrelationType::Pearson, 0.1);
        assert!(!strong_corrs.is_empty());

        // Test correlation summary
        let summary = analysis.correlation_summary(&CorrelationType::Pearson);
        assert!(summary.is_some());
        let (mean, median, min, max) = summary.unwrap();
        assert!(min <= median);
        assert!(median <= max);
    }

    #[test]
    fn test_dependency_graph_accessors() {
        let mut outputs = HashMap::new();
        outputs.insert("task1".to_string(), array![[1.0], [2.0], [3.0]]);
        outputs.insert("task2".to_string(), array![[0.5], [1.0], [1.5]]);

        let builder =
            DependencyGraphBuilder::new().method(DependencyMethod::CorrelationThreshold(0.1));

        let graph = builder.build(&outputs).unwrap();

        // Test neighbor retrieval
        let neighbors = graph.get_neighbors("task1_0");
        assert!(neighbors.len() <= 2); // Can have at most task2_0 as neighbor

        // Test degree calculation
        let degree = graph.get_degree("task1_0");
        assert!(degree <= 2);

        // Test connection checking
        let connected = graph.are_connected("task1_0", "task2_0");
        // Connection depends on correlation threshold and actual data correlation
    }
}
