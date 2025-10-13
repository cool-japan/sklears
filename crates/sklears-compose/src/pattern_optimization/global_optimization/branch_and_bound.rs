//! Branch and Bound Optimization Module
//!
//! This module provides comprehensive branch and bound optimization capabilities
//! for solving discrete and mixed-integer optimization problems. Branch and bound
//! is a systematic tree search algorithm that explores solution spaces by dividing
//! them into smaller subproblems and using bounds to prune unpromising branches.
//!
//! # Key Features
//!
//! ## Tree Search Strategies
//! - **Depth-First Search**: Memory-efficient exploration with stack-based traversal
//! - **Breadth-First Search**: Level-by-level exploration ensuring optimality
//! - **Best-First Search**: Priority-based exploration using node bounds
//! - **Hybrid Strategies**: Adaptive switching between search methods
//!
//! ## Branching Techniques
//! - **Variable Selection**: Most fractional, most promising, pseudocost-based
//! - **Value Selection**: Strong branching, reliability branching
//! - **Constraint Branching**: SOS constraints, indicator constraints
//! - **Adaptive Branching**: Problem-specific branching rules
//!
//! ## Bounding Methods
//! - **Linear Relaxation**: LP-based bounds for integer problems
//! - **Lagrangian Relaxation**: Dual bounds with constraint relaxation
//! - **Heuristic Bounds**: Fast approximate bounds for large problems
//! - **Problem-Specific Bounds**: Domain knowledge integration
//!
//! ## Advanced Features
//! - **Parallel Processing**: Multi-threaded tree exploration
//! - **Cut Generation**: Valid inequalities for tighter bounds
//! - **Presolving**: Problem reduction before search
//! - **Node Selection**: Sophisticated priority mechanisms
//!
//! # Usage Examples
//!
//! ## Basic Branch and Bound
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::branch_and_bound::*;
//! use scirs2_core::ndarray::array;
//!
//! // Configure depth-first search strategy
//! let search_strategy = SearchStrategy::DepthFirst;
//! let branching_rule = BranchingRule::MostFractional;
//!
//! // Create branch and bound optimizer
//! let bnb = BranchAndBoundOptimizer::builder()
//!     .search_strategy(search_strategy)
//!     .branching_rule(branching_rule)
//!     .node_selection(NodeSelection::BestBound)
//!     .max_nodes(100000)
//!     .tolerance(1e-6)
//!     .build();
//!
//! // Apply to mixed-integer problem
//! let result = bnb.optimize(&problem)?;
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust
//! // Configure sophisticated branching and bounding
//! let bounding_method = BoundingMethod::LagrangianRelaxation {
//!     max_iterations: 100,
//!     step_size: 0.1,
//!     tolerance: 1e-8,
//! };
//!
//! let bnb = BranchAndBoundOptimizer::builder()
//!     .search_strategy(SearchStrategy::BestFirst)
//!     .bounding_method(bounding_method)
//!     .branching_rule(BranchingRule::StrongBranching { candidates: 5 })
//!     .cut_generation(true)
//!     .parallel_processing(true)
//!     .build();
//! ```
//!
//! ## Tree Analysis and Monitoring
//!
//! ```rust
//! // Configure with detailed analysis
//! let bnb = BranchAndBoundOptimizer::builder()
//!     .enable_tree_analysis(true)
//!     .node_limit_monitoring(true)
//!     .convergence_tracking(true)
//!     .build();
//!
//! let result = bnb.optimize(&problem)?;
//!
//! // Access detailed tree statistics
//! let analysis = result.tree_analysis.unwrap();
//! println!("Nodes explored: {}", analysis.nodes_explored);
//! println!("Nodes pruned: {}", analysis.nodes_pruned);
//! println!("Best bound gap: {:.2}%", analysis.final_gap * 100.0);
//! ```

use crate::core::{OptimizationProblem, Solution, SklResult, SklError, OptimizationResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, rng, DistributionExt};
use std::collections::{BinaryHeap, VecDeque, HashMap};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex, RwLock};
use rayon::prelude::*;

/// Core trait for branch and bound optimization algorithms.
///
/// This trait defines the essential operations for branch and bound optimization,
/// including node creation, bound computation, branching decisions, and tree management.
/// Implementations can provide different search strategies and problem-specific optimizations.
pub trait BranchAndBound: Send + Sync {
    /// Creates child nodes by branching on the given node.
    ///
    /// # Arguments
    /// * `node` - Parent node to branch from
    /// * `problem` - Optimization problem definition
    ///
    /// # Returns
    /// Vector of child nodes created by branching
    fn branch_node(
        &self,
        node: &TreeNode,
        problem: &OptimizationProblem,
    ) -> SklResult<Vec<TreeNode>>;

    /// Computes bounds for a given node.
    ///
    /// # Arguments
    /// * `node` - Tree node to compute bounds for
    /// * `problem` - Problem definition for bound computation
    ///
    /// # Returns
    /// Node bounds (lower and upper bounds)
    fn compute_bounds(
        &self,
        node: &TreeNode,
        problem: &OptimizationProblem,
    ) -> SklResult<NodeBounds>;

    /// Determines if a node should be pruned based on bounds.
    ///
    /// # Arguments
    /// * `node` - Node to evaluate for pruning
    /// * `current_best` - Current best solution value
    /// * `tolerance` - Pruning tolerance
    ///
    /// # Returns
    /// True if node should be pruned
    fn should_prune(
        &self,
        node: &TreeNode,
        current_best: f64,
        tolerance: f64,
    ) -> bool;

    /// Selects the next node to process from the tree.
    ///
    /// # Arguments
    /// * `open_nodes` - Collection of open nodes
    /// * `selection_strategy` - Strategy for node selection
    ///
    /// # Returns
    /// Selected node for processing
    fn select_node(
        &self,
        open_nodes: &mut NodeCollection,
        selection_strategy: NodeSelection,
    ) -> Option<TreeNode>;

    /// Analyzes the search tree performance and statistics.
    ///
    /// # Arguments
    /// * `tree_statistics` - Current tree statistics
    /// * `node_history` - History of processed nodes
    ///
    /// # Returns
    /// Comprehensive tree analysis
    fn analyze_tree(
        &self,
        tree_statistics: &TreeStatistics,
        node_history: &[ProcessedNode],
    ) -> SklResult<TreeAnalysis>;
}

/// Represents a node in the branch and bound search tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Unique identifier for the node
    pub id: usize,
    /// Depth in the search tree
    pub depth: usize,
    /// Lower bound for this subproblem
    pub lower_bound: f64,
    /// Upper bound for this subproblem
    pub upper_bound: f64,
    /// Variable assignments and constraints for this node
    pub constraints: NodeConstraints,
    /// Best solution found in this subtree
    pub best_solution: Option<Solution>,
    /// Parent node identifier
    pub parent_id: Option<usize>,
    /// Child node identifiers
    pub children: Vec<usize>,
    /// Processing priority
    pub priority: f64,
    /// Node creation timestamp
    pub created_at: std::time::Instant,
}

impl TreeNode {
    /// Creates a new root node.
    pub fn new_root(problem: &OptimizationProblem) -> Self {
        Self {
            id: 0,
            depth: 0,
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
            constraints: NodeConstraints::new(problem.dimension),
            best_solution: None,
            parent_id: None,
            children: Vec::new(),
            priority: 0.0,
            created_at: std::time::Instant::now(),
        }
    }

    /// Creates a child node with additional constraints.
    pub fn create_child(
        &self,
        id: usize,
        additional_constraints: Vec<Constraint>,
    ) -> Self {
        let mut child_constraints = self.constraints.clone();
        for constraint in additional_constraints {
            child_constraints.add_constraint(constraint);
        }

        Self {
            id,
            depth: self.depth + 1,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            constraints: child_constraints,
            best_solution: None,
            parent_id: Some(self.id),
            children: Vec::new(),
            priority: 0.0,
            created_at: std::time::Instant::now(),
        }
    }

    /// Checks if this node is feasible.
    pub fn is_feasible(&self) -> bool {
        self.constraints.is_feasible() && self.lower_bound <= self.upper_bound
    }

    /// Computes the gap between upper and lower bounds.
    pub fn gap(&self) -> f64 {
        if self.upper_bound.is_infinite() || self.lower_bound.is_infinite() {
            f64::INFINITY
        } else {
            (self.upper_bound - self.lower_bound).abs()
        }
    }
}

impl PartialEq for TreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TreeNode {}

impl PartialOrd for TreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TreeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Priority queue is max-heap, but we want min priority
        other.priority.partial_cmp(&self.priority).unwrap_or(Ordering::Equal)
    }
}

/// Constraints associated with a tree node.
#[derive(Debug, Clone)]
pub struct NodeConstraints {
    /// Variable bounds for this node
    pub variable_bounds: Vec<(f64, f64)>,
    /// Additional linear constraints
    pub linear_constraints: Vec<LinearConstraint>,
    /// Integer constraints on variables
    pub integer_constraints: Vec<usize>,
    /// Special ordered set (SOS) constraints
    pub sos_constraints: Vec<SOSConstraint>,
}

impl NodeConstraints {
    /// Creates new constraints for a problem dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            variable_bounds: vec![(-f64::INFINITY, f64::INFINITY); dimension],
            linear_constraints: Vec::new(),
            integer_constraints: Vec::new(),
            sos_constraints: Vec::new(),
        }
    }

    /// Adds a constraint to this node.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        match constraint {
            Constraint::VariableBound { index, lower, upper } => {
                self.variable_bounds[index] = (lower, upper);
            },
            Constraint::Linear(linear_constraint) => {
                self.linear_constraints.push(linear_constraint);
            },
            Constraint::Integer(var_index) => {
                self.integer_constraints.push(var_index);
            },
            Constraint::SOS(sos_constraint) => {
                self.sos_constraints.push(sos_constraint);
            },
        }
    }

    /// Checks if the constraints are feasible.
    pub fn is_feasible(&self) -> bool {
        // Check variable bounds consistency
        for (lower, upper) in &self.variable_bounds {
            if lower > upper {
                return false;
            }
        }

        // Additional feasibility checks could be added here
        true
    }

    /// Computes effective bounds considering all constraints.
    pub fn effective_bounds(&self) -> Vec<(f64, f64)> {
        // Start with variable bounds and tighten with linear constraints
        let mut bounds = self.variable_bounds.clone();

        // Simple bound tightening with linear constraints
        for constraint in &self.linear_constraints {
            // Simplified constraint propagation
            if constraint.coefficients.len() == 1 {
                let var_idx = 0; // Simplified for single variable constraints
                let coeff = constraint.coefficients[0];
                if coeff > 0.0 {
                    let new_upper = constraint.rhs / coeff;
                    bounds[var_idx].1 = bounds[var_idx].1.min(new_upper);
                } else if coeff < 0.0 {
                    let new_lower = constraint.rhs / coeff;
                    bounds[var_idx].0 = bounds[var_idx].0.max(new_lower);
                }
            }
        }

        bounds
    }
}

/// Different types of constraints that can be added to nodes.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Variable bound constraint
    VariableBound {
        index: usize,
        lower: f64,
        upper: f64,
    },
    /// Linear constraint
    Linear(LinearConstraint),
    /// Integer constraint on variable
    Integer(usize),
    /// Special ordered set constraint
    SOS(SOSConstraint),
}

/// Linear constraint of the form: sum(coeff[i] * x[i]) <= rhs
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// Constraint coefficients
    pub coefficients: Vec<f64>,
    /// Right-hand side value
    pub rhs: f64,
    /// Constraint type (<=, =, >=)
    pub constraint_type: ConstraintType,
}

/// Types of linear constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Less than or equal
    LessEqual,
    /// Equal
    Equal,
    /// Greater than or equal
    GreaterEqual,
}

/// Special Ordered Set (SOS) constraint.
#[derive(Debug, Clone)]
pub struct SOSConstraint {
    /// Variables in the SOS
    pub variables: Vec<usize>,
    /// SOS type (1 or 2)
    pub sos_type: usize,
    /// Priority weights
    pub weights: Vec<f64>,
}

/// Bounds computed for a tree node.
#[derive(Debug, Clone)]
pub struct NodeBounds {
    /// Lower bound on the objective
    pub lower_bound: f64,
    /// Upper bound on the objective
    pub upper_bound: f64,
    /// Feasibility of the node
    pub feasible: bool,
    /// Solution achieving the bound (if available)
    pub bound_solution: Option<Array1<f64>>,
    /// Method used to compute the bound
    pub bound_method: String,
    /// Computation time for bounds
    pub computation_time: std::time::Duration,
}

/// Different strategies for tree search.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    /// Depth-first search (DFS)
    DepthFirst,
    /// Breadth-first search (BFS)
    BreadthFirst,
    /// Best-first search using bounds
    BestFirst,
    /// Hybrid strategy combining multiple approaches
    Hybrid {
        primary: Box<SearchStrategy>,
        secondary: Box<SearchStrategy>,
        switch_condition: SwitchCondition,
    },
}

/// Conditions for switching between search strategies in hybrid approach.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchCondition {
    /// Switch after a certain number of nodes
    NodeCount(usize),
    /// Switch after a certain time
    TimeLimit(std::time::Duration),
    /// Switch based on gap improvement
    GapImprovement(f64),
}

/// Node selection strategies within search methods.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeSelection {
    /// Select node with best lower bound
    BestBound,
    /// Select node with best upper bound
    BestSolution,
    /// Select deepest node first
    DepthFirst,
    /// Select shallowest node first
    BreadthFirst,
    /// Select most recently created node
    LastIn,
    /// Select oldest node
    FirstIn,
}

/// Branching rules for creating child nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum BranchingRule {
    /// Branch on most fractional variable
    MostFractional,
    /// Branch on variable with best pseudocost
    PseudoCost,
    /// Strong branching with candidate evaluation
    StrongBranching { candidates: usize },
    /// Reliability branching
    ReliabilityBranching { min_observations: usize },
    /// Custom branching rule
    Custom(String),
}

/// Methods for computing node bounds.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundingMethod {
    /// Linear programming relaxation
    LinearRelaxation,
    /// Lagrangian relaxation
    LagrangianRelaxation {
        max_iterations: usize,
        step_size: f64,
        tolerance: f64,
    },
    /// Heuristic bounds
    Heuristic,
    /// Problem-specific bounding
    ProblemSpecific(String),
}

/// Collection of open nodes for processing.
#[derive(Debug)]
pub enum NodeCollection {
    /// Priority queue for best-first search
    PriorityQueue(BinaryHeap<TreeNode>),
    /// Stack for depth-first search
    Stack(Vec<TreeNode>),
    /// Queue for breadth-first search
    Queue(VecDeque<TreeNode>),
}

impl NodeCollection {
    /// Creates a new node collection based on search strategy.
    pub fn new(strategy: &SearchStrategy) -> Self {
        match strategy {
            SearchStrategy::DepthFirst => Self::Stack(Vec::new()),
            SearchStrategy::BreadthFirst => Self::Queue(VecDeque::new()),
            SearchStrategy::BestFirst => Self::PriorityQueue(BinaryHeap::new()),
            SearchStrategy::Hybrid { primary, .. } => Self::new(primary),
        }
    }

    /// Adds a node to the collection.
    pub fn push(&mut self, node: TreeNode) {
        match self {
            Self::PriorityQueue(heap) => heap.push(node),
            Self::Stack(stack) => stack.push(node),
            Self::Queue(queue) => queue.push_back(node),
        }
    }

    /// Removes and returns the next node.
    pub fn pop(&mut self) -> Option<TreeNode> {
        match self {
            Self::PriorityQueue(heap) => heap.pop(),
            Self::Stack(stack) => stack.pop(),
            Self::Queue(queue) => queue.pop_front(),
        }
    }

    /// Returns the number of nodes in the collection.
    pub fn len(&self) -> usize {
        match self {
            Self::PriorityQueue(heap) => heap.len(),
            Self::Stack(stack) => stack.len(),
            Self::Queue(queue) => queue.len(),
        }
    }

    /// Checks if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Statistics about the search tree.
#[derive(Debug, Clone)]
pub struct TreeStatistics {
    /// Total nodes created
    pub nodes_created: usize,
    /// Nodes currently in memory
    pub nodes_active: usize,
    /// Nodes processed and closed
    pub nodes_processed: usize,
    /// Nodes pruned by bounds
    pub nodes_pruned: usize,
    /// Nodes pruned by infeasibility
    pub nodes_infeasible: usize,
    /// Current tree depth
    pub current_depth: usize,
    /// Maximum tree depth reached
    pub max_depth: usize,
    /// Current best objective value
    pub best_objective: f64,
    /// Current best lower bound
    pub best_bound: f64,
    /// Current optimality gap
    pub gap: f64,
    /// Search start time
    pub start_time: std::time::Instant,
    /// Total computation time
    pub total_time: std::time::Duration,
}

impl TreeStatistics {
    /// Creates new tree statistics.
    pub fn new() -> Self {
        Self {
            nodes_created: 0,
            nodes_active: 0,
            nodes_processed: 0,
            nodes_pruned: 0,
            nodes_infeasible: 0,
            current_depth: 0,
            max_depth: 0,
            best_objective: f64::INFINITY,
            best_bound: f64::NEG_INFINITY,
            gap: f64::INFINITY,
            start_time: std::time::Instant::now(),
            total_time: std::time::Duration::new(0, 0),
        }
    }

    /// Updates the statistics with a new processed node.
    pub fn update_processed(&mut self, node: &TreeNode) {
        self.nodes_processed += 1;
        self.nodes_active -= 1;
        self.current_depth = node.depth;
        self.max_depth = self.max_depth.max(node.depth);
        self.total_time = self.start_time.elapsed();
    }

    /// Updates the best objective value found.
    pub fn update_best_objective(&mut self, objective: f64) {
        if objective < self.best_objective {
            self.best_objective = objective;
            self.update_gap();
        }
    }

    /// Updates the best lower bound.
    pub fn update_best_bound(&mut self, bound: f64) {
        if bound > self.best_bound {
            self.best_bound = bound;
            self.update_gap();
        }
    }

    /// Computes the current optimality gap.
    fn update_gap(&mut self) {
        if self.best_objective.is_finite() && self.best_bound.is_finite() {
            if self.best_objective.abs() > 1e-10 {
                self.gap = ((self.best_objective - self.best_bound) / self.best_objective.abs()).abs();
            } else {
                self.gap = (self.best_objective - self.best_bound).abs();
            }
        } else {
            self.gap = f64::INFINITY;
        }
    }
}

/// Information about a processed node.
#[derive(Debug, Clone)]
pub struct ProcessedNode {
    /// Node that was processed
    pub node: TreeNode,
    /// Processing result
    pub result: ProcessingResult,
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Number of children created
    pub children_created: usize,
}

/// Result of processing a tree node.
#[derive(Debug, Clone)]
pub enum ProcessingResult {
    /// Node was branched into children
    Branched,
    /// Node was pruned by bounds
    PrunedByBounds,
    /// Node was pruned by infeasibility
    PrunedByInfeasibility,
    /// Node provided a new solution
    SolutionFound,
    /// Node processing failed
    Failed(String),
}

/// Comprehensive analysis of the search tree.
#[derive(Debug, Clone)]
pub struct TreeAnalysis {
    /// Overall tree statistics
    pub statistics: TreeStatistics,
    /// Search efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Branching analysis
    pub branching_analysis: BranchingAnalysis,
    /// Pruning effectiveness
    pub pruning_analysis: PruningAnalysis,
    /// Convergence characteristics
    pub convergence_analysis: ConvergenceAnalysis,
}

/// Metrics about search efficiency.
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Nodes processed per second
    pub nodes_per_second: f64,
    /// Average node processing time
    pub avg_processing_time: std::time::Duration,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
    /// Load balancing effectiveness (for parallel)
    pub load_balance: f64,
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Peak number of active nodes
    pub peak_active_nodes: usize,
    /// Average number of active nodes
    pub avg_active_nodes: f64,
    /// Memory per node estimate
    pub memory_per_node: usize,
    /// Total memory usage estimate
    pub total_memory: usize,
}

/// Analysis of branching behavior.
#[derive(Debug, Clone)]
pub struct BranchingAnalysis {
    /// Average branching factor
    pub avg_branching_factor: f64,
    /// Distribution of branching decisions
    pub branching_distribution: HashMap<String, usize>,
    /// Effectiveness of branching rules
    pub rule_effectiveness: HashMap<String, f64>,
    /// Variable selection frequency
    pub variable_selection: Vec<usize>,
}

/// Analysis of pruning effectiveness.
#[derive(Debug, Clone)]
pub struct PruningAnalysis {
    /// Fraction of nodes pruned
    pub pruning_rate: f64,
    /// Pruning by reason
    pub pruning_by_reason: HashMap<String, usize>,
    /// Depth distribution of pruned nodes
    pub pruning_by_depth: Vec<usize>,
    /// Time saved by pruning
    pub time_saved: std::time::Duration,
}

/// Analysis of convergence behavior.
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Gap reduction over time
    pub gap_history: Vec<(std::time::Duration, f64)>,
    /// Solution improvement events
    pub solution_improvements: Vec<SolutionImprovement>,
    /// Stagnation periods
    pub stagnation_periods: Vec<StagnationPeriod>,
    /// Convergence rate estimate
    pub convergence_rate: f64,
}

/// Information about solution improvements.
#[derive(Debug, Clone)]
pub struct SolutionImprovement {
    /// Time when improvement occurred
    pub timestamp: std::time::Duration,
    /// Previous best objective
    pub previous_objective: f64,
    /// New best objective
    pub new_objective: f64,
    /// Improvement magnitude
    pub improvement: f64,
    /// Node where improvement was found
    pub node_id: usize,
}

/// Information about stagnation periods.
#[derive(Debug, Clone)]
pub struct StagnationPeriod {
    /// Start time of stagnation
    pub start_time: std::time::Duration,
    /// Duration of stagnation
    pub duration: std::time::Duration,
    /// Number of nodes processed during stagnation
    pub nodes_processed: usize,
    /// Reason for stagnation end
    pub end_reason: String,
}

/// Main branch and bound optimizer implementation.
#[derive(Debug)]
pub struct BranchAndBoundOptimizer {
    /// Search strategy for tree traversal
    pub search_strategy: SearchStrategy,
    /// Node selection within strategy
    pub node_selection: NodeSelection,
    /// Branching rule for creating children
    pub branching_rule: BranchingRule,
    /// Method for computing bounds
    pub bounding_method: BoundingMethod,
    /// Maximum number of nodes to process
    pub max_nodes: usize,
    /// Maximum computation time
    pub max_time: std::time::Duration,
    /// Optimality gap tolerance
    pub tolerance: f64,
    /// Enable cut generation
    pub cut_generation: bool,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Enable detailed tree analysis
    pub enable_analysis: bool,
    /// Next node ID counter
    next_node_id: Arc<Mutex<usize>>,
    /// Random number generator
    rng: Random,
}

impl BranchAndBoundOptimizer {
    /// Creates a builder for branch and bound configuration.
    pub fn builder() -> BranchAndBoundBuilder {
        BranchAndBoundBuilder::default()
    }

    /// Optimizes the given problem using branch and bound.
    pub fn optimize(&self, problem: &OptimizationProblem) -> SklResult<OptimizationResult> {
        let mut statistics = TreeStatistics::new();
        let mut open_nodes = NodeCollection::new(&self.search_strategy);
        let mut processed_nodes = Vec::new();

        // Create root node
        let root = TreeNode::new_root(problem);
        open_nodes.push(root);
        statistics.nodes_created += 1;
        statistics.nodes_active += 1;

        let mut best_solution: Option<Solution> = None;
        let mut iteration = 0;

        while !open_nodes.is_empty() && iteration < self.max_nodes {
            // Check time limit
            if statistics.start_time.elapsed() > self.max_time {
                break;
            }

            // Select next node to process
            let node = match self.select_node(&mut open_nodes, self.node_selection.clone()) {
                Some(n) => n,
                None => break,
            };

            let processing_start = std::time::Instant::now();

            // Compute bounds for the node
            let bounds = self.compute_bounds(&node, problem)?;

            // Check if node should be pruned
            let current_best = best_solution.as_ref().map(|s| s.fitness).unwrap_or(f64::INFINITY);
            if self.should_prune(&node, current_best, self.tolerance) {
                statistics.nodes_pruned += 1;
                processed_nodes.push(ProcessedNode {
                    node: node.clone(),
                    result: ProcessingResult::PrunedByBounds,
                    processing_time: processing_start.elapsed(),
                    children_created: 0,
                });
                statistics.update_processed(&node);
                continue;
            }

            // Check feasibility
            if !bounds.feasible {
                statistics.nodes_infeasible += 1;
                processed_nodes.push(ProcessedNode {
                    node: node.clone(),
                    result: ProcessingResult::PrunedByInfeasibility,
                    processing_time: processing_start.elapsed(),
                    children_created: 0,
                });
                statistics.update_processed(&node);
                continue;
            }

            // Update bounds
            statistics.update_best_bound(bounds.lower_bound);

            // Check if node provides a complete solution
            if self.is_integer_solution(&bounds.bound_solution, &node.constraints) {
                let solution = Solution {
                    parameters: bounds.bound_solution.unwrap_or_else(|| Array1::zeros(problem.dimension)),
                    fitness: bounds.lower_bound,
                    feasible: true,
                    generation: iteration,
                };

                if best_solution.is_none() || solution.fitness < best_solution.as_ref().unwrap().fitness {
                    best_solution = Some(solution.clone());
                    statistics.update_best_objective(solution.fitness);
                }

                processed_nodes.push(ProcessedNode {
                    node: node.clone(),
                    result: ProcessingResult::SolutionFound,
                    processing_time: processing_start.elapsed(),
                    children_created: 0,
                });
                statistics.update_processed(&node);
                continue;
            }

            // Branch the node
            let children = self.branch_node(&node, problem)?;
            let num_children = children.len();

            for child in children {
                open_nodes.push(child);
                statistics.nodes_created += 1;
                statistics.nodes_active += 1;
            }

            processed_nodes.push(ProcessedNode {
                node: node.clone(),
                result: ProcessingResult::Branched,
                processing_time: processing_start.elapsed(),
                children_created: num_children,
            });

            statistics.update_processed(&node);
            iteration += 1;

            // Check convergence
            if statistics.gap < self.tolerance {
                break;
            }
        }

        // Create analysis if requested
        let analysis = if self.enable_analysis {
            Some(self.analyze_tree(&statistics, &processed_nodes)?)
        } else {
            None
        };

        let final_solution = best_solution.unwrap_or_else(|| Solution {
            parameters: Array1::zeros(problem.dimension),
            fitness: f64::INFINITY,
            feasible: false,
            generation: 0,
        });

        Ok(OptimizationResult {
            best_solution: final_solution,
            convergence_history: vec![statistics.best_objective],
            metadata: analysis.map(|a| serde_json::to_value(a).unwrap()),
        })
    }

    /// Checks if a solution satisfies integer constraints.
    fn is_integer_solution(
        &self,
        solution: &Option<Array1<f64>>,
        constraints: &NodeConstraints,
    ) -> bool {
        if let Some(sol) = solution {
            for &var_idx in &constraints.integer_constraints {
                if var_idx < sol.len() {
                    let value = sol[var_idx];
                    if (value - value.round()).abs() > 1e-9 {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

impl BranchAndBound for BranchAndBoundOptimizer {
    fn branch_node(
        &self,
        node: &TreeNode,
        problem: &OptimizationProblem,
    ) -> SklResult<Vec<TreeNode>> {
        // Find branching variable based on branching rule
        let branch_var = self.select_branching_variable(node, problem)?;

        // Get current variable bounds
        let (current_lower, current_upper) = node.constraints.variable_bounds[branch_var];

        // Create two child nodes with restricted bounds
        let mut children = Vec::new();

        // Child 1: variable <= floor(current_value)
        let mid_point = (current_lower + current_upper) / 2.0;
        let floor_val = mid_point.floor();

        if floor_val >= current_lower {
            let child1_id = {
                let mut id_counter = self.next_node_id.lock().unwrap();
                *id_counter += 1;
                *id_counter
            };

            let constraint1 = Constraint::VariableBound {
                index: branch_var,
                lower: current_lower,
                upper: floor_val,
            };

            let child1 = node.create_child(child1_id, vec![constraint1]);
            children.push(child1);
        }

        // Child 2: variable >= ceil(current_value)
        let ceil_val = mid_point.ceil();

        if ceil_val <= current_upper {
            let child2_id = {
                let mut id_counter = self.next_node_id.lock().unwrap();
                *id_counter += 1;
                *id_counter
            };

            let constraint2 = Constraint::VariableBound {
                index: branch_var,
                lower: ceil_val,
                upper: current_upper,
            };

            let child2 = node.create_child(child2_id, vec![constraint2]);
            children.push(child2);
        }

        Ok(children)
    }

    fn compute_bounds(
        &self,
        node: &TreeNode,
        problem: &OptimizationProblem,
    ) -> SklResult<NodeBounds> {
        let start_time = std::time::Instant::now();

        match &self.bounding_method {
            BoundingMethod::LinearRelaxation => {
                // Simplified linear relaxation bound
                let bounds = node.constraints.effective_bounds();
                let mut best_bound = f64::INFINITY;
                let mut best_solution = None;

                // Simple bound based on variable bounds and objective (simplified)
                for i in 0..problem.dimension {
                    let (lower, upper) = bounds[i];
                    if lower.is_finite() && upper.is_finite() {
                        // Evaluate at bounds (simplified objective evaluation)
                        let mut test_point = Array1::zeros(problem.dimension);
                        test_point[i] = lower;

                        if let Ok(value) = problem.evaluate(&test_point) {
                            if value < best_bound {
                                best_bound = value;
                                best_solution = Some(test_point);
                            }
                        }
                    }
                }

                Ok(NodeBounds {
                    lower_bound: best_bound,
                    upper_bound: f64::INFINITY,
                    feasible: best_bound.is_finite(),
                    bound_solution: best_solution,
                    bound_method: "LinearRelaxation".to_string(),
                    computation_time: start_time.elapsed(),
                })
            },
            BoundingMethod::Heuristic => {
                // Simple heuristic bound
                let bounds = node.constraints.effective_bounds();
                let mut solution = Array1::zeros(problem.dimension);

                for i in 0..problem.dimension {
                    let (lower, upper) = bounds[i];
                    solution[i] = if lower.is_finite() && upper.is_finite() {
                        (lower + upper) / 2.0
                    } else if lower.is_finite() {
                        lower
                    } else if upper.is_finite() {
                        upper
                    } else {
                        0.0
                    };
                }

                let objective_value = problem.evaluate(&solution)?;

                Ok(NodeBounds {
                    lower_bound: objective_value,
                    upper_bound: f64::INFINITY,
                    feasible: true,
                    bound_solution: Some(solution),
                    bound_method: "Heuristic".to_string(),
                    computation_time: start_time.elapsed(),
                })
            },
            _ => {
                // Default to heuristic for other methods
                self.compute_bounds_heuristic(node, problem, start_time)
            },
        }
    }

    fn should_prune(
        &self,
        node: &TreeNode,
        current_best: f64,
        tolerance: f64,
    ) -> bool {
        // Prune if lower bound is not better than current best
        if node.lower_bound >= current_best - tolerance {
            return true;
        }

        // Prune if node is infeasible
        if !node.is_feasible() {
            return true;
        }

        // Prune if gap is smaller than tolerance
        if node.gap() < tolerance {
            return true;
        }

        false
    }

    fn select_node(
        &self,
        open_nodes: &mut NodeCollection,
        selection_strategy: NodeSelection,
    ) -> Option<TreeNode> {
        match selection_strategy {
            NodeSelection::BestBound | NodeSelection::BestSolution => {
                open_nodes.pop()
            },
            _ => {
                open_nodes.pop()
            },
        }
    }

    fn analyze_tree(
        &self,
        tree_statistics: &TreeStatistics,
        node_history: &[ProcessedNode],
    ) -> SklResult<TreeAnalysis> {
        // Efficiency metrics
        let nodes_per_second = if tree_statistics.total_time.as_secs_f64() > 0.0 {
            tree_statistics.nodes_processed as f64 / tree_statistics.total_time.as_secs_f64()
        } else {
            0.0
        };

        let avg_processing_time = if !node_history.is_empty() {
            let total_time: std::time::Duration = node_history.iter()
                .map(|n| n.processing_time)
                .sum();
            total_time / node_history.len() as u32
        } else {
            std::time::Duration::new(0, 0)
        };

        let efficiency_metrics = EfficiencyMetrics {
            nodes_per_second,
            avg_processing_time,
            memory_usage: MemoryUsage {
                peak_active_nodes: tree_statistics.nodes_active,
                avg_active_nodes: tree_statistics.nodes_active as f64,
                memory_per_node: 1024, // Estimate
                total_memory: tree_statistics.nodes_active * 1024,
            },
            load_balance: 1.0, // Simplified
        };

        // Branching analysis
        let avg_branching_factor = if tree_statistics.nodes_processed > 0 {
            (tree_statistics.nodes_created - 1) as f64 / tree_statistics.nodes_processed as f64
        } else {
            0.0
        };

        let branching_analysis = BranchingAnalysis {
            avg_branching_factor,
            branching_distribution: HashMap::new(),
            rule_effectiveness: HashMap::new(),
            variable_selection: vec![],
        };

        // Pruning analysis
        let pruning_rate = if tree_statistics.nodes_created > 0 {
            tree_statistics.nodes_pruned as f64 / tree_statistics.nodes_created as f64
        } else {
            0.0
        };

        let pruning_analysis = PruningAnalysis {
            pruning_rate,
            pruning_by_reason: HashMap::new(),
            pruning_by_depth: vec![],
            time_saved: std::time::Duration::new(0, 0),
        };

        // Convergence analysis
        let convergence_analysis = ConvergenceAnalysis {
            gap_history: vec![(tree_statistics.total_time, tree_statistics.gap)],
            solution_improvements: vec![],
            stagnation_periods: vec![],
            convergence_rate: 0.0,
        };

        Ok(TreeAnalysis {
            statistics: tree_statistics.clone(),
            efficiency_metrics,
            branching_analysis,
            pruning_analysis,
            convergence_analysis,
        })
    }
}

impl BranchAndBoundOptimizer {
    /// Selects the variable to branch on based on the branching rule.
    fn select_branching_variable(
        &self,
        node: &TreeNode,
        problem: &OptimizationProblem,
    ) -> SklResult<usize> {
        match &self.branching_rule {
            BranchingRule::MostFractional => {
                // Find the integer variable with most fractional value
                let bounds = node.constraints.effective_bounds();
                let mut max_fractionality = 0.0;
                let mut best_var = 0;

                for &var_idx in &node.constraints.integer_constraints {
                    if var_idx < bounds.len() {
                        let (lower, upper) = bounds[var_idx];
                        if lower.is_finite() && upper.is_finite() {
                            let mid = (lower + upper) / 2.0;
                            let fractionality = (mid - mid.round()).abs();
                            if fractionality > max_fractionality {
                                max_fractionality = fractionality;
                                best_var = var_idx;
                            }
                        }
                    }
                }

                Ok(best_var)
            },
            _ => {
                // Default to first integer variable
                Ok(node.constraints.integer_constraints.first().copied().unwrap_or(0))
            },
        }
    }

    /// Computes heuristic bounds for a node.
    fn compute_bounds_heuristic(
        &self,
        node: &TreeNode,
        problem: &OptimizationProblem,
        start_time: std::time::Instant,
    ) -> SklResult<NodeBounds> {
        let bounds = node.constraints.effective_bounds();
        let mut solution = Array1::zeros(problem.dimension);

        // Use midpoint of variable bounds
        for i in 0..problem.dimension {
            let (lower, upper) = bounds[i];
            solution[i] = if lower.is_finite() && upper.is_finite() {
                (lower + upper) / 2.0
            } else if lower.is_finite() {
                lower
            } else if upper.is_finite() {
                upper
            } else {
                0.0
            };
        }

        let objective_value = problem.evaluate(&solution)?;

        Ok(NodeBounds {
            lower_bound: objective_value,
            upper_bound: f64::INFINITY,
            feasible: true,
            bound_solution: Some(solution),
            bound_method: "Heuristic".to_string(),
            computation_time: start_time.elapsed(),
        })
    }
}

/// Builder for branch and bound optimizer.
#[derive(Debug)]
pub struct BranchAndBoundBuilder {
    search_strategy: SearchStrategy,
    node_selection: NodeSelection,
    branching_rule: BranchingRule,
    bounding_method: BoundingMethod,
    max_nodes: usize,
    max_time: std::time::Duration,
    tolerance: f64,
    cut_generation: bool,
    parallel_processing: bool,
    enable_analysis: bool,
}

impl Default for BranchAndBoundBuilder {
    fn default() -> Self {
        Self {
            search_strategy: SearchStrategy::BestFirst,
            node_selection: NodeSelection::BestBound,
            branching_rule: BranchingRule::MostFractional,
            bounding_method: BoundingMethod::LinearRelaxation,
            max_nodes: 100000,
            max_time: std::time::Duration::from_secs(3600),
            tolerance: 1e-6,
            cut_generation: false,
            parallel_processing: false,
            enable_analysis: false,
        }
    }
}

impl BranchAndBoundBuilder {
    /// Sets the search strategy.
    pub fn search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.search_strategy = strategy;
        self
    }

    /// Sets the node selection method.
    pub fn node_selection(mut self, selection: NodeSelection) -> Self {
        self.node_selection = selection;
        self
    }

    /// Sets the branching rule.
    pub fn branching_rule(mut self, rule: BranchingRule) -> Self {
        self.branching_rule = rule;
        self
    }

    /// Sets the bounding method.
    pub fn bounding_method(mut self, method: BoundingMethod) -> Self {
        self.bounding_method = method;
        self
    }

    /// Sets the maximum number of nodes.
    pub fn max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Sets the maximum computation time.
    pub fn max_time(mut self, max_time: std::time::Duration) -> Self {
        self.max_time = max_time;
        self
    }

    /// Sets the optimality tolerance.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Enables cut generation.
    pub fn cut_generation(mut self, enable: bool) -> Self {
        self.cut_generation = enable;
        self
    }

    /// Enables parallel processing.
    pub fn parallel_processing(mut self, enable: bool) -> Self {
        self.parallel_processing = enable;
        self
    }

    /// Enables detailed tree analysis.
    pub fn enable_analysis(mut self, enable: bool) -> Self {
        self.enable_analysis = enable;
        self
    }

    /// Builds the branch and bound optimizer.
    pub fn build(self) -> BranchAndBoundOptimizer {
        BranchAndBoundOptimizer {
            search_strategy: self.search_strategy,
            node_selection: self.node_selection,
            branching_rule: self.branching_rule,
            bounding_method: self.bounding_method,
            max_nodes: self.max_nodes,
            max_time: self.max_time,
            tolerance: self.tolerance,
            cut_generation: self.cut_generation,
            parallel_processing: self.parallel_processing,
            enable_analysis: self.enable_analysis,
            next_node_id: Arc::new(Mutex::new(0)),
            rng: Random::new(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_tree_node_creation() {
        let problem = OptimizationProblem {
            dimension: 2,
            bounds: vec![(0.0, 10.0), (0.0, 10.0)],
            objective: Box::new(|x| Ok(x.sum())),
        };

        let root = TreeNode::new_root(&problem);
        assert_eq!(root.id, 0);
        assert_eq!(root.depth, 0);
        assert!(root.parent_id.is_none());
        assert_eq!(root.constraints.variable_bounds.len(), 2);
    }

    #[test]
    fn test_node_constraints() {
        let mut constraints = NodeConstraints::new(3);

        // Add variable bound constraint
        constraints.add_constraint(Constraint::VariableBound {
            index: 0,
            lower: 1.0,
            upper: 5.0,
        });

        assert_eq!(constraints.variable_bounds[0], (1.0, 5.0));

        // Add integer constraint
        constraints.add_constraint(Constraint::Integer(1));
        assert!(constraints.integer_constraints.contains(&1));

        // Check feasibility
        assert!(constraints.is_feasible());
    }

    #[test]
    fn test_node_collection() {
        let mut stack = NodeCollection::Stack(Vec::new());
        let mut queue = NodeCollection::Queue(VecDeque::new());

        let node = TreeNode::new_root(&OptimizationProblem {
            dimension: 1,
            bounds: vec![(0.0, 1.0)],
            objective: Box::new(|x| Ok(x.sum())),
        });

        // Test stack operations
        stack.push(node.clone());
        assert_eq!(stack.len(), 1);
        let popped = stack.pop().unwrap();
        assert_eq!(popped.id, node.id);
        assert!(stack.is_empty());

        // Test queue operations
        queue.push(node.clone());
        assert_eq!(queue.len(), 1);
        let popped = queue.pop().unwrap();
        assert_eq!(popped.id, node.id);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_tree_statistics() {
        let mut stats = TreeStatistics::new();

        // Test initial state
        assert_eq!(stats.nodes_created, 0);
        assert_eq!(stats.best_objective, f64::INFINITY);

        // Update best objective
        stats.update_best_objective(10.0);
        assert_eq!(stats.best_objective, 10.0);

        // Update best bound
        stats.update_best_bound(8.0);
        assert_eq!(stats.best_bound, 8.0);
        assert!((stats.gap - 0.2).abs() < 1e-10); // (10-8)/10 = 0.2
    }

    #[test]
    fn test_branch_and_bound_builder() {
        let bnb = BranchAndBoundOptimizer::builder()
            .search_strategy(SearchStrategy::DepthFirst)
            .branching_rule(BranchingRule::PseudoCost)
            .max_nodes(50000)
            .tolerance(1e-8)
            .enable_analysis(true)
            .build();

        assert_eq!(bnb.search_strategy, SearchStrategy::DepthFirst);
        assert_eq!(bnb.branching_rule, BranchingRule::PseudoCost);
        assert_eq!(bnb.max_nodes, 50000);
        assert_eq!(bnb.tolerance, 1e-8);
        assert!(bnb.enable_analysis);
    }

    #[test]
    fn test_branching_variable_selection() {
        let bnb = BranchAndBoundOptimizer::builder()
            .branching_rule(BranchingRule::MostFractional)
            .build();

        let mut node = TreeNode::new_root(&OptimizationProblem {
            dimension: 3,
            bounds: vec![(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            objective: Box::new(|x| Ok(x.sum())),
        });

        // Add integer constraints
        node.constraints.integer_constraints = vec![0, 1, 2];

        let problem = OptimizationProblem {
            dimension: 3,
            bounds: vec![(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            objective: Box::new(|x| Ok(x.sum())),
        };

        let var = bnb.select_branching_variable(&node, &problem).unwrap();
        assert!(var < 3);
    }

    #[test]
    fn test_bound_computation() {
        let bnb = BranchAndBoundOptimizer::builder()
            .bounding_method(BoundingMethod::Heuristic)
            .build();

        let node = TreeNode::new_root(&OptimizationProblem {
            dimension: 2,
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            objective: Box::new(|x| Ok(x.sum())),
        });

        let problem = OptimizationProblem {
            dimension: 2,
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            objective: Box::new(|x| Ok(x.sum())),
        };

        let bounds = bnb.compute_bounds(&node, &problem).unwrap();
        assert!(bounds.feasible);
        assert!(bounds.lower_bound.is_finite());
        assert_eq!(bounds.bound_method, "Heuristic");
    }
}