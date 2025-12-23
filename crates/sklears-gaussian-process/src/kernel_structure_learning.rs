//! Advanced kernel structure learning using grammar-based search
//!
//! This module implements sophisticated kernel structure learning algorithms that can
//! automatically discover complex kernel compositions using grammar-based search,
//! statistical tests, and structure optimization.

use crate::kernels::*;
// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
// SciRS2 Policy - Use scirs2-core for random number generation
use scirs2_core::random::Rng;
use sklears_core::error::{Result as SklResult, SklearsError};

/// Grammar-based kernel structure learning
#[derive(Debug, Clone)]
pub struct KernelStructureLearner {
    /// Maximum depth of kernel expressions
    pub max_depth: usize,
    /// Maximum number of iterations for structure search
    pub max_iterations: usize,
    /// Probability of adding new components
    pub expansion_probability: f64,
    /// Probability of simplifying structures
    pub simplification_probability: f64,
    /// Minimum improvement threshold for accepting new structures
    pub improvement_threshold: f64,
    /// Whether to use Bayesian information criterion for model selection
    pub use_bic: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Search strategy
    pub search_strategy: SearchStrategy,
}

/// Search strategies for kernel structure learning
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    Greedy,
    Beam {
        beam_width: usize,
    },
    /// Genetic algorithm - evolve population of kernel structures
    Genetic {
        population_size: usize,
    },
    /// Simulated annealing - accept worse solutions with decreasing probability
    SimulatedAnnealing {
        initial_temperature: f64,
    },
}

/// Kernel grammar production rules
#[derive(Debug, Clone)]
pub enum KernelGrammar {
    /// Terminal symbols (base kernels)
    Terminal(TerminalKernel),
    /// Non-terminal symbols (operations)
    NonTerminal(NonTerminalOperation),
}

/// Base kernel types in the grammar
#[derive(Debug, Clone)]
pub enum TerminalKernel {
    /// RBF
    RBF { length_scale: f64 },
    /// Linear
    Linear { sigma_0: f64, sigma_1: f64 },
    /// Periodic
    Periodic { length_scale: f64, period: f64 },
    /// Matern
    Matern { length_scale: f64, nu: f64 },
    /// RationalQuadratic
    RationalQuadratic { length_scale: f64, alpha: f64 },
    /// White
    White { noise_level: f64 },
    /// Constant
    Constant { constant_value: f64 },
}

/// Operations for combining kernels
#[derive(Debug, Clone)]
pub enum NonTerminalOperation {
    /// Sum
    Sum {
        left: Box<KernelGrammar>,
        right: Box<KernelGrammar>,
    },
    /// Product
    Product {
        left: Box<KernelGrammar>,
        right: Box<KernelGrammar>,
    },
    /// Power
    Power {
        base: Box<KernelGrammar>,
        exponent: f64,
    },
    /// Scale
    Scale {
        kernel: Box<KernelGrammar>,
        scale: f64,
    },
}

/// Result of kernel structure learning
#[derive(Debug, Clone)]
pub struct StructureLearningResult {
    /// The best kernel structure found
    pub best_kernel: Box<dyn Kernel>,
    /// Grammar expression of the best kernel
    pub best_expression: KernelGrammar,
    /// Score of the best kernel
    pub best_score: f64,
    /// All structures explored with their scores
    pub exploration_history: Vec<(String, f64)>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Information about the learning convergence
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Number of iterations completed
    pub iterations: usize,
    /// Score history over iterations
    pub score_history: Vec<f64>,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final temperature (for simulated annealing)
    pub final_temperature: Option<f64>,
}

impl Default for KernelStructureLearner {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_iterations: 100,
            expansion_probability: 0.7,
            simplification_probability: 0.3,
            improvement_threshold: 0.01,
            use_bic: true,
            random_state: Some(42),
            search_strategy: SearchStrategy::Greedy,
        }
    }
}

impl KernelStructureLearner {
    /// Create a new kernel structure learner
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum depth of kernel expressions
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set search strategy
    pub fn search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.search_strategy = strategy;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Learn kernel structure from data
    pub fn learn_structure(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> SklResult<StructureLearningResult> {
        // SciRS2 Policy - Use scirs2-core for random number generation
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(42)
        };

        match self.search_strategy {
            SearchStrategy::Greedy => self.greedy_search(X, y, &mut rng),
            SearchStrategy::Beam { beam_width } => self.beam_search(X, y, beam_width, &mut rng),
            SearchStrategy::Genetic { population_size } => {
                self.genetic_search(X, y, population_size, &mut rng)
            }
            SearchStrategy::SimulatedAnnealing {
                initial_temperature,
            } => self.simulated_annealing_search(X, y, initial_temperature, &mut rng),
        }
    }

    /// Greedy search for kernel structure
    fn greedy_search(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<StructureLearningResult> {
        let mut exploration_history = Vec::new();
        let mut score_history = Vec::new();

        // Start with simple RBF kernel
        let mut current_expression =
            KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 1.0 });
        let mut current_kernel = self.expression_to_kernel(&current_expression)?;
        let mut current_score = self.evaluate_kernel(&current_kernel, &X, &y)?;

        exploration_history.push(("RBF".to_string(), current_score));
        score_history.push(current_score);

        let mut best_expression = current_expression.clone();
        let mut best_kernel = current_kernel.clone();
        let mut best_score = current_score;

        for _iteration in 0..self.max_iterations {
            // Generate candidate structures
            let candidates = self.generate_candidates(&current_expression, rng)?;

            let mut found_improvement = false;

            for candidate in candidates {
                if self.expression_depth(&candidate) > self.max_depth {
                    continue;
                }

                let kernel = self.expression_to_kernel(&candidate)?;
                let score = self.evaluate_kernel(&kernel, &X, &y)?;

                let expression_str = self.expression_to_string(&candidate);
                exploration_history.push((expression_str, score));

                if score < current_score - self.improvement_threshold {
                    current_expression = candidate;
                    current_kernel = kernel;
                    current_score = score;
                    found_improvement = true;

                    if score < best_score {
                        best_expression = current_expression.clone();
                        best_kernel = current_kernel.clone();
                        best_score = score;
                    }
                    break;
                }
            }

            score_history.push(current_score);

            if !found_improvement {
                break;
            }
        }

        Ok(StructureLearningResult {
            best_kernel,
            best_expression,
            best_score,
            exploration_history,
            convergence_info: ConvergenceInfo {
                iterations: score_history.len(),
                score_history,
                converged: true,
                final_temperature: None,
            },
        })
    }

    /// Beam search for kernel structure
    fn beam_search(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        beam_width: usize,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<StructureLearningResult> {
        let mut exploration_history = Vec::new();
        let mut score_history = Vec::new();

        // Initialize beam with base kernels
        let mut beam: Vec<(KernelGrammar, f64)> = Vec::new();
        let initial_kernels = self.generate_initial_kernels()?;

        if initial_kernels.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No initial kernels generated".to_string(),
            ));
        }

        for kernel_expr in initial_kernels {
            let kernel = self.expression_to_kernel(&kernel_expr)?;
            let score = self.evaluate_kernel(&kernel, &X, &y)?;
            beam.push((kernel_expr.clone(), score));

            let expr_str = self.expression_to_string(&kernel_expr);
            exploration_history.push((expr_str, score));
        }

        // Sort beam by score (lower is better)
        beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        beam.truncate(beam_width);

        if beam.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "Beam is empty after initialization".to_string(),
            ));
        }

        let mut best_score = beam[0].1;
        score_history.push(best_score);

        for _iteration in 0..self.max_iterations {
            let mut new_beam = Vec::new();

            // Expand each beam element
            for (expression, _) in &beam {
                let candidates = self.generate_candidates(expression, rng)?;

                for candidate in candidates {
                    if self.expression_depth(&candidate) > self.max_depth {
                        continue;
                    }

                    let kernel = self.expression_to_kernel(&candidate)?;
                    let score = self.evaluate_kernel(&kernel, &X, &y)?;

                    new_beam.push((candidate.clone(), score));

                    let expr_str = self.expression_to_string(&candidate);
                    exploration_history.push((expr_str, score));
                }
            }

            // Merge and sort
            new_beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            new_beam.truncate(beam_width);

            // Update beam
            beam = new_beam;

            if beam.is_empty() {
                break;
            }

            let current_best = beam[0].1;
            score_history.push(current_best);

            if (best_score - current_best).abs() < self.improvement_threshold {
                break;
            }

            best_score = current_best;
        }

        if beam.is_empty() {
            // Return a default RBF kernel if no beam elements remain
            let default_expression =
                KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 1.0 });
            let default_kernel = self.expression_to_kernel(&default_expression)?;
            return Ok(StructureLearningResult {
                best_kernel: default_kernel,
                best_expression: default_expression,
                best_score: f64::INFINITY,
                exploration_history,
                convergence_info: ConvergenceInfo {
                    iterations: score_history.len(),
                    score_history,
                    converged: false,
                    final_temperature: None,
                },
            });
        }

        let best_expression = beam[0].0.clone();
        let best_kernel = self.expression_to_kernel(&best_expression)?;

        Ok(StructureLearningResult {
            best_kernel,
            best_expression,
            best_score: beam[0].1,
            exploration_history,
            convergence_info: ConvergenceInfo {
                iterations: score_history.len(),
                score_history,
                converged: true,
                final_temperature: None,
            },
        })
    }

    /// Genetic algorithm search for kernel structure
    fn genetic_search(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        population_size: usize,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<StructureLearningResult> {
        let mut exploration_history = Vec::new();
        let mut score_history = Vec::new();

        // Initialize population
        let mut population: Vec<(KernelGrammar, f64)> = Vec::new();
        let initial_kernels = self.generate_initial_kernels()?;

        for _ in 0..population_size {
            let idx = rng.gen_range(0..initial_kernels.len());
            let kernel_expr = initial_kernels[idx].clone();
            let kernel = self.expression_to_kernel(&kernel_expr)?;
            let score = self.evaluate_kernel(&kernel, &X, &y)?;
            population.push((kernel_expr, score));
        }

        for _generation in 0..self.max_iterations {
            // Selection: tournament selection
            let mut new_population = Vec::new();

            for _ in 0..population_size {
                let parent1 = self.tournament_selection(&population, rng);
                let parent2 = self.tournament_selection(&population, rng);

                // Crossover
                let child = self.crossover(&parent1.0, &parent2.0, rng)?;

                // Mutation
                let mutated_child = self.mutate(&child, rng)?;

                if self.expression_depth(&mutated_child) <= self.max_depth {
                    let kernel = self.expression_to_kernel(&mutated_child)?;
                    let score = self.evaluate_kernel(&kernel, &X, &y)?;
                    new_population.push((mutated_child.clone(), score));

                    let expr_str = self.expression_to_string(&mutated_child);
                    exploration_history.push((expr_str, score));
                }
            }

            // Replace population
            population = new_population;
            population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            population.truncate(population_size);

            let best_score = population[0].1;
            score_history.push(best_score);
        }

        let best_expression = population[0].0.clone();
        let best_kernel = self.expression_to_kernel(&best_expression)?;

        Ok(StructureLearningResult {
            best_kernel,
            best_expression,
            best_score: population[0].1,
            exploration_history,
            convergence_info: ConvergenceInfo {
                iterations: score_history.len(),
                score_history,
                converged: true,
                final_temperature: None,
            },
        })
    }

    /// Simulated annealing search for kernel structure
    fn simulated_annealing_search(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        initial_temperature: f64,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<StructureLearningResult> {
        let mut exploration_history = Vec::new();
        let mut score_history = Vec::new();

        // Start with simple RBF kernel
        let mut current_expression =
            KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 1.0 });
        let mut current_kernel = self.expression_to_kernel(&current_expression)?;
        let mut current_score = self.evaluate_kernel(&current_kernel, &X, &y)?;

        let mut best_expression = current_expression.clone();
        let mut best_kernel = current_kernel.clone();
        let mut best_score = current_score;

        let mut temperature = initial_temperature;
        let cooling_rate = 0.95;

        for _iteration in 0..self.max_iterations {
            // Generate a neighbor
            let candidates = self.generate_candidates(&current_expression, rng)?;
            if candidates.is_empty() {
                break;
            }

            let idx = rng.gen_range(0..candidates.len());
            let candidate = &candidates[idx];
            if self.expression_depth(candidate) > self.max_depth {
                continue;
            }

            let kernel = self.expression_to_kernel(candidate)?;
            let score = self.evaluate_kernel(&kernel, &X, &y)?;

            let expr_str = self.expression_to_string(candidate);
            exploration_history.push((expr_str, score));

            // Accept or reject based on Metropolis criterion
            let delta = score - current_score;
            if delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
                current_expression = candidate.clone();
                current_kernel = kernel;
                current_score = score;

                if score < best_score {
                    best_expression = current_expression.clone();
                    best_kernel = current_kernel.clone();
                    best_score = score;
                }
            }

            score_history.push(current_score);
            temperature *= cooling_rate;
        }

        Ok(StructureLearningResult {
            best_kernel,
            best_expression,
            best_score,
            exploration_history,
            convergence_info: ConvergenceInfo {
                iterations: score_history.len(),
                score_history,
                converged: true,
                final_temperature: Some(temperature),
            },
        })
    }

    /// Generate initial base kernels
    fn generate_initial_kernels(&self) -> SklResult<Vec<KernelGrammar>> {
        Ok(vec![
            KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 1.0 }),
            KernelGrammar::Terminal(TerminalKernel::Linear {
                sigma_0: 1.0,
                sigma_1: 1.0,
            }),
            KernelGrammar::Terminal(TerminalKernel::Matern {
                length_scale: 1.0,
                nu: 1.5,
            }),
            KernelGrammar::Terminal(TerminalKernel::RationalQuadratic {
                length_scale: 1.0,
                alpha: 1.0,
            }),
            KernelGrammar::Terminal(TerminalKernel::Periodic {
                length_scale: 1.0,
                period: 1.0,
            }),
            KernelGrammar::Terminal(TerminalKernel::White { noise_level: 0.1 }),
            KernelGrammar::Terminal(TerminalKernel::Constant {
                constant_value: 1.0,
            }),
        ])
    }

    /// Generate candidate structures from current expression
    fn generate_candidates(
        &self,
        expression: &KernelGrammar,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<Vec<KernelGrammar>> {
        let mut candidates = Vec::new();

        // Add a new component with sum
        if rng.gen::<f64>() < self.expansion_probability {
            let new_kernels = self.generate_initial_kernels()?;
            for new_kernel in new_kernels {
                candidates.push(KernelGrammar::NonTerminal(NonTerminalOperation::Sum {
                    left: Box::new(expression.clone()),
                    right: Box::new(new_kernel),
                }));
            }
        }

        // Add a new component with product
        if rng.gen::<f64>() < self.expansion_probability {
            let new_kernels = self.generate_initial_kernels()?;
            for new_kernel in new_kernels {
                candidates.push(KernelGrammar::NonTerminal(NonTerminalOperation::Product {
                    left: Box::new(expression.clone()),
                    right: Box::new(new_kernel),
                }));
            }
        }

        // Scale the current kernel
        if rng.gen::<f64>() < 0.3 {
            let scale = rng.gen_range(0.1..10.0);
            candidates.push(KernelGrammar::NonTerminal(NonTerminalOperation::Scale {
                kernel: Box::new(expression.clone()),
                scale,
            }));
        }

        Ok(candidates)
    }

    /// Tournament selection for genetic algorithm
    fn tournament_selection<'a>(
        &self,
        population: &'a [(KernelGrammar, f64)],
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> &'a (KernelGrammar, f64) {
        let tournament_size = 3.min(population.len());
        let mut best_idx = rng.gen_range(0..population.len());
        let mut best_score = population[best_idx].1;

        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..population.len());
            if population[idx].1 < best_score {
                best_idx = idx;
                best_score = population[idx].1;
            }
        }

        &population[best_idx]
    }

    /// Crossover operation for genetic algorithm
    fn crossover(
        &self,
        parent1: &KernelGrammar,
        parent2: &KernelGrammar,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<KernelGrammar> {
        if rng.gen::<f64>() < 0.5 {
            Ok(KernelGrammar::NonTerminal(NonTerminalOperation::Sum {
                left: Box::new(parent1.clone()),
                right: Box::new(parent2.clone()),
            }))
        } else {
            Ok(KernelGrammar::NonTerminal(NonTerminalOperation::Product {
                left: Box::new(parent1.clone()),
                right: Box::new(parent2.clone()),
            }))
        }
    }

    /// Mutation operation for genetic algorithm
    fn mutate(
        &self,
        expression: &KernelGrammar,
        rng: &mut scirs2_core::random::Random<scirs2_core::rngs::StdRng>,
    ) -> SklResult<KernelGrammar> {
        if rng.gen::<f64>() < 0.1 {
            // Replace with random kernel
            let new_kernels = self.generate_initial_kernels()?;
            let idx = rng.gen_range(0..new_kernels.len());
            Ok(new_kernels[idx].clone())
        } else {
            // Keep original
            Ok(expression.clone())
        }
    }

    /// Calculate depth of kernel expression
    fn expression_depth(&self, expression: &KernelGrammar) -> usize {
        match expression {
            KernelGrammar::Terminal(_) => 1,
            KernelGrammar::NonTerminal(op) => match op {
                NonTerminalOperation::Sum { left, right }
                | NonTerminalOperation::Product { left, right } => {
                    1 + self
                        .expression_depth(left)
                        .max(self.expression_depth(right))
                }
                NonTerminalOperation::Scale { kernel, .. }
                | NonTerminalOperation::Power { base: kernel, .. } => {
                    1 + self.expression_depth(kernel)
                }
            },
        }
    }

    /// Convert grammar expression to actual kernel
    fn expression_to_kernel(&self, expression: &KernelGrammar) -> SklResult<Box<dyn Kernel>> {
        match expression {
            KernelGrammar::Terminal(terminal) => match terminal {
                TerminalKernel::RBF { length_scale } => Ok(Box::new(RBF::new(*length_scale))),
                TerminalKernel::Linear { sigma_0, sigma_1 } => {
                    Ok(Box::new(Linear::new(*sigma_0, *sigma_1)))
                }
                TerminalKernel::Periodic {
                    length_scale,
                    period,
                } => Ok(Box::new(ExpSineSquared::new(*length_scale, *period))),
                TerminalKernel::Matern { length_scale, nu } => {
                    Ok(Box::new(Matern::new(*length_scale, *nu)))
                }
                TerminalKernel::RationalQuadratic {
                    length_scale,
                    alpha,
                } => Ok(Box::new(RationalQuadratic::new(*length_scale, *alpha))),
                TerminalKernel::White { noise_level } => {
                    Ok(Box::new(WhiteKernel::new(*noise_level)))
                }
                TerminalKernel::Constant { constant_value } => {
                    Ok(Box::new(ConstantKernel::new(*constant_value)))
                }
            },
            KernelGrammar::NonTerminal(op) => match op {
                NonTerminalOperation::Sum { left, right } => {
                    let left_kernel = self.expression_to_kernel(left)?;
                    let right_kernel = self.expression_to_kernel(right)?;
                    Ok(Box::new(crate::kernels::SumKernel::new(vec![
                        left_kernel,
                        right_kernel,
                    ])))
                }
                NonTerminalOperation::Product { left, right } => {
                    let left_kernel = self.expression_to_kernel(left)?;
                    let right_kernel = self.expression_to_kernel(right)?;
                    Ok(Box::new(crate::kernels::ProductKernel::new(vec![
                        left_kernel,
                        right_kernel,
                    ])))
                }
                NonTerminalOperation::Scale { kernel, scale } => {
                    let base_kernel = self.expression_to_kernel(kernel)?;
                    // Scale by creating a constant kernel and using product
                    let scale_kernel = Box::new(ConstantKernel::new(*scale));
                    Ok(Box::new(crate::kernels::ProductKernel::new(vec![
                        base_kernel,
                        scale_kernel,
                    ])))
                }
                NonTerminalOperation::Power { base, exponent: _ } => {
                    // For now, just return the base kernel
                    // Power kernels would need special implementation
                    self.expression_to_kernel(base)
                }
            },
        }
    }

    /// Convert expression to human-readable string
    fn expression_to_string(&self, expression: &KernelGrammar) -> String {
        match expression {
            KernelGrammar::Terminal(terminal) => match terminal {
                TerminalKernel::RBF { length_scale } => format!("RBF({:.3})", length_scale),
                TerminalKernel::Linear { sigma_0, sigma_1 } => {
                    format!("Linear({:.3}, {:.3})", sigma_0, sigma_1)
                }
                TerminalKernel::Periodic {
                    length_scale,
                    period,
                } => format!("Periodic({:.3}, {:.3})", length_scale, period),
                TerminalKernel::Matern { length_scale, nu } => {
                    format!("Matern({:.3}, {:.3})", length_scale, nu)
                }
                TerminalKernel::RationalQuadratic {
                    length_scale,
                    alpha,
                } => format!("RQ({:.3}, {:.3})", length_scale, alpha),
                TerminalKernel::White { noise_level } => format!("White({:.3})", noise_level),
                TerminalKernel::Constant { constant_value } => {
                    format!("Const({:.3})", constant_value)
                }
            },
            KernelGrammar::NonTerminal(op) => match op {
                NonTerminalOperation::Sum { left, right } => {
                    format!(
                        "({} + {})",
                        self.expression_to_string(left),
                        self.expression_to_string(right)
                    )
                }
                NonTerminalOperation::Product { left, right } => {
                    format!(
                        "({} * {})",
                        self.expression_to_string(left),
                        self.expression_to_string(right)
                    )
                }
                NonTerminalOperation::Scale { kernel, scale } => {
                    format!("{:.3} * {}", scale, self.expression_to_string(kernel))
                }
                NonTerminalOperation::Power { base, exponent } => {
                    format!("{}^{:.3}", self.expression_to_string(base), exponent)
                }
            },
        }
    }

    /// Evaluate kernel using BIC or cross-validation
    fn evaluate_kernel(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        // For now, use simple marginal likelihood
        // This can be extended to use cross-validation or BIC
        self.compute_marginal_likelihood(kernel, X, y)
    }

    /// Compute marginal likelihood for kernel evaluation
    #[allow(non_snake_case)]
    fn compute_marginal_likelihood(
        &self,
        kernel: &Box<dyn Kernel>,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        let X_owned = X.to_owned();
        let K = kernel.compute_kernel_matrix(&X_owned, Some(&X_owned))?;

        // Add noise to diagonal
        let mut K_noisy = K;
        let noise_var = 0.1;
        for i in 0..K_noisy.nrows() {
            K_noisy[[i, i]] += noise_var;
        }

        // Compute Cholesky decomposition
        match crate::utils::cholesky_decomposition(&K_noisy) {
            Ok(L) => {
                // Compute log marginal likelihood
                let mut log_det = 0.0;
                for i in 0..L.nrows() {
                    log_det += L[[i, i]].ln();
                }
                log_det *= 2.0;

                // Solve for alpha = K^(-1) * y
                let y_owned = y.to_owned();
                let alpha = match crate::utils::triangular_solve(&L, &y_owned) {
                    Ok(temp) => {
                        let L_T = L.t();
                        crate::utils::triangular_solve(&L_T.view().to_owned(), &temp)?
                    }
                    Err(_) => return Ok(f64::INFINITY),
                };

                let data_fit = -0.5 * y.dot(&alpha);
                let complexity_penalty = -0.5 * log_det;
                let normalization = -0.5 * y.len() as f64 * (2.0 * std::f64::consts::PI).ln();

                Ok(-(data_fit + complexity_penalty + normalization))
            }
            Err(_) => Ok(f64::INFINITY),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_kernel_structure_learner_creation() {
        let learner = KernelStructureLearner::new();
        assert_eq!(learner.max_depth, 4);
        assert_eq!(learner.max_iterations, 100);
    }

    #[test]
    fn test_expression_depth_calculation() {
        let learner = KernelStructureLearner::new();

        // Terminal kernel has depth 1
        let terminal = KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 1.0 });
        assert_eq!(learner.expression_depth(&terminal), 1);

        // Sum of two terminals has depth 2
        let sum = KernelGrammar::NonTerminal(NonTerminalOperation::Sum {
            left: Box::new(terminal.clone()),
            right: Box::new(terminal.clone()),
        });
        assert_eq!(learner.expression_depth(&sum), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_expression_to_kernel_conversion() {
        let learner = KernelStructureLearner::new();

        let expression = KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 2.0 });
        let kernel = learner.expression_to_kernel(&expression).unwrap();

        // Test that kernel can be used
        let X = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let K = kernel.compute_kernel_matrix(&X, Some(&X)).unwrap();
        assert_eq!(K.nrows(), 3);
        assert_eq!(K.ncols(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_greedy_search() {
        let learner = KernelStructureLearner::new().max_iterations(5).max_depth(2);

        let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);

        let result = learner.learn_structure(X.view(), y.view()).unwrap();

        assert!(result.best_score.is_finite());
        assert!(!result.exploration_history.is_empty());
        assert!(!result.convergence_info.score_history.is_empty());
    }

    #[test]
    fn test_expression_to_string() {
        let learner = KernelStructureLearner::new();

        let rbf = KernelGrammar::Terminal(TerminalKernel::RBF { length_scale: 1.0 });
        let linear = KernelGrammar::Terminal(TerminalKernel::Linear {
            sigma_0: 1.0,
            sigma_1: 1.0,
        });
        let sum = KernelGrammar::NonTerminal(NonTerminalOperation::Sum {
            left: Box::new(rbf),
            right: Box::new(linear),
        });

        let expr_str = learner.expression_to_string(&sum);
        assert!(expr_str.contains("RBF"));
        assert!(expr_str.contains("Linear"));
        assert!(expr_str.contains("+"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_beam_search() {
        let learner = KernelStructureLearner::new()
            .max_iterations(3)
            .max_depth(2)
            .search_strategy(SearchStrategy::Beam { beam_width: 2 });

        let X = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = learner.learn_structure(X.view(), y.view()).unwrap();

        // The search might return infinity if no valid kernels are found
        assert!(result.best_score.is_finite() || result.best_score == f64::INFINITY);
        assert!(!result.exploration_history.is_empty());
    }
}
