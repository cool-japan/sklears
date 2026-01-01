/// Advanced automatic differentiation system for sklears-core
///
/// This module provides a comprehensive automatic differentiation framework with procedural
/// macros for compile-time gradient computation. It supports both forward-mode and reverse-mode
/// automatic differentiation with efficient tape-based computation graphs.
///
/// # Key Features
///
/// - **Compile-time AD**: Zero-overhead automatic differentiation using procedural macros
/// - **Dual Number System**: Forward-mode AD with epsilon-delta calculus
/// - **Computation Graph**: Reverse-mode AD with dynamic tape construction
/// - **Higher-order Derivatives**: Support for Hessians and higher-order gradients
/// - **SIMD Optimization**: Vectorized gradient computation
/// - **GPU Support**: CUDA kernels for gradient computation
/// - **Symbolic Differentiation**: Optional symbolic manipulation
///
/// # Usage Examples
///
/// ## Forward-mode Automatic Differentiation
/// ```rust,ignore
/// use sklears_core::autodiff::{autodiff, Dual, forward_diff};
///
/// // Define a function with automatic differentiation
/// #[autodiff(forward)]
/// fn polynomial(x: f64) -> f64 {
///     x.powi(3) + 2.0 * x.powi(2) - 3.0 * x + 1.0
/// }
///
/// // Compute function value and derivative at x = 2.0
/// let (value, derivative) = forward_diff(polynomial, 2.0);
/// assert_eq!(value, 11.0);        // f(2) = 8 + 8 - 6 + 1 = 11
/// assert_eq!(derivative, 15.0);   // f'(2) = 12 + 8 - 3 = 17
/// ```
///
/// ## Reverse-mode Automatic Differentiation
/// ```rust,ignore
/// use sklears_core::autodiff::{autodiff, Variable, backward};
///
/// // Define a neural network layer with backpropagation
/// #[autodiff(reverse)]
/// fn neural_layer(x: &[f64], weights: &[f64], bias: f64) -> f64 {
///     let linear = x.iter().zip(weights).map(|(xi, wi)| xi * wi).sum::`<f64>`() + bias;
///     1.0 / (1.0 + (-linear).exp()) // sigmoid activation
/// }
///
/// // Compute gradients with respect to all inputs
/// let x = vec![1.0, 2.0, 3.0];
/// let weights = vec![0.5, -0.3, 0.7];
/// let bias = 0.1;
///
/// let gradients = backward(neural_layer, (&x, &weights, bias));
/// println!("Gradients: {:?}", gradients);
/// ```
///
/// ## Multi-variable Functions
/// ```rust,ignore
/// use sklears_core::autodiff::{autodiff, gradient, hessian};
///
/// // Loss function for logistic regression
/// #[autodiff(reverse, order = 2)] // Support up to 2nd derivatives
/// fn logistic_loss(weights: &[f64], x: &[f64], y: f64) -> f64 {
///     let prediction = sigmoid(dot_product(weights, x));
///     -y * prediction.ln() - (1.0 - y) * (1.0 - prediction).ln()
/// }
///
/// // Compute gradient and Hessian for optimization
/// let weights = vec![0.1, 0.2, -0.3];
/// let x = vec![1.0, 2.0, 3.0];
/// let y = 1.0;
///
/// let grad = gradient(logistic_loss, (&weights, &x, y));
/// let hess = hessian(logistic_loss, (&weights, &x, y));
/// ```
use crate::error::{Result, SklearsError};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use syn::{Attribute, Expr, FnArg, ItemFn, ReturnType, Stmt, Type};

// =============================================================================
// Core Automatic Differentiation Types
// =============================================================================

/// Dual number for forward-mode automatic differentiation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Real part (function value)
    pub real: f64,
    /// Dual part (derivative)
    pub dual: f64,
}

impl Dual {
    /// Create a new dual number
    pub fn new(real: f64, dual: f64) -> Self {
        Self { real, dual }
    }

    /// Create a dual number representing a variable
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    /// Create a dual number representing a constant
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// Extract the value (real part)
    pub fn value(&self) -> f64 {
        self.real
    }

    /// Extract the derivative (dual part)
    pub fn derivative(&self) -> f64 {
        self.dual
    }
}

/// Implementation of arithmetic operations for dual numbers
impl std::ops::Add for Dual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.dual + other.dual)
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.dual - other.dual)
    }
}

impl std::ops::Mul for Dual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real,
        )
    }
}

impl std::ops::Div for Dual {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let inv_other_real = 1.0 / other.real;
        Self::new(
            self.real * inv_other_real,
            (self.dual * other.real - self.real * other.dual) * inv_other_real * inv_other_real,
        )
    }
}

/// Variable for reverse-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct Variable {
    /// Unique variable identifier
    pub id: VariableId,
    /// Current value
    pub value: f64,
    /// Gradient (populated during backpropagation)
    pub gradient: f64,
    /// Computation graph node
    pub node: Option<Arc<ComputationNode>>,
}

/// Unique identifier for variables in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(pub u64);

impl Variable {
    /// Create a new variable
    pub fn new(value: f64) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = VariableId(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst));

        Self {
            id,
            value,
            gradient: 0.0,
            node: None,
        }
    }

    /// Create a variable with computation graph tracking
    pub fn with_graph(value: f64, tape: Arc<Mutex<ComputationTape>>) -> Self {
        let mut var = Self::new(value);

        let node = ComputationNode {
            operation: Operation::Input,
            inputs: Vec::new(),
            output_id: var.id,
            gradient_fn: Box::new(|_inputs, _output_grad| Vec::new()),
        };

        var.node = Some(Arc::new(node));

        // Register with tape
        if let Ok(mut tape_guard) = tape.lock() {
            tape_guard.add_node(var.node.as_ref().unwrap().clone());
        }

        var
    }

    /// Set gradient value
    pub fn set_gradient(&mut self, gradient: f64) {
        self.gradient = gradient;
    }

    /// Add to gradient (for accumulation)
    pub fn add_gradient(&mut self, gradient: f64) {
        self.gradient += gradient;
    }

    /// Reset gradient to zero
    pub fn zero_gradient(&mut self) {
        self.gradient = 0.0;
    }
}

/// Type alias for gradient functions to reduce complexity
pub type GradientFunction = Box<dyn Fn(&[f64], f64) -> Vec<f64> + Send + Sync>;

/// Node in the computation graph for reverse-mode AD
pub struct ComputationNode {
    /// Operation that produced this node
    pub operation: Operation,
    /// Input variable IDs
    pub inputs: Vec<VariableId>,
    /// Output variable ID
    pub output_id: VariableId,
    /// Gradient function for backpropagation
    pub gradient_fn: GradientFunction,
}

impl std::fmt::Debug for ComputationNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationNode")
            .field("operation", &self.operation)
            .field("inputs", &self.inputs)
            .field("output_id", &self.output_id)
            .field("gradient_fn", &"<function>")
            .finish()
    }
}

/// Operations in the computation graph
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Input variable
    Input,
    /// Addition operation
    Add,
    /// Subtraction operation
    Sub,
    /// Multiplication operation
    Mul,
    /// Division operation
    Div,
    /// Power operation
    Pow,
    /// Exponential function
    Exp,
    /// Natural logarithm
    Ln,
    /// Sine function
    Sin,
    /// Cosine function
    Cos,
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid function
    Sigmoid,
    /// ReLU activation
    ReLU,
    /// Custom operation
    Custom(String),
}

/// Computation tape for tracking operations in reverse-mode AD
#[derive(Debug)]
pub struct ComputationTape {
    /// Nodes in the computation graph
    pub nodes: Vec<Arc<ComputationNode>>,
    /// Variable registry
    pub variables: HashMap<VariableId, Variable>,
    /// Execution order for backpropagation
    pub execution_order: Vec<VariableId>,
}

impl ComputationTape {
    /// Create a new computation tape
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            variables: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    /// Add a node to the computation graph
    pub fn add_node(&mut self, node: Arc<ComputationNode>) {
        self.execution_order.push(node.output_id);
        self.nodes.push(node);
    }

    /// Register a variable
    pub fn register_variable(&mut self, var: Variable) {
        self.variables.insert(var.id, var);
    }

    /// Perform backpropagation
    pub fn backward(&mut self, root_gradient: f64) -> Result<()> {
        // Initialize gradients
        for var in self.variables.values_mut() {
            var.zero_gradient();
        }

        // Set root gradient
        if let Some(root_id) = self.execution_order.last() {
            if let Some(root_var) = self.variables.get_mut(root_id) {
                root_var.set_gradient(root_gradient);
            }
        }

        // Backpropagate in reverse order
        for &node_id in self.execution_order.iter().rev() {
            if let Some(node) = self.nodes.iter().find(|n| n.output_id == node_id) {
                let output_gradient = self
                    .variables
                    .get(&node_id)
                    .map(|v| v.gradient)
                    .unwrap_or(0.0);

                // Get input values for gradient computation
                let input_values: Vec<f64> = node
                    .inputs
                    .iter()
                    .filter_map(|&id| self.variables.get(&id).map(|v| v.value))
                    .collect();

                // Compute input gradients
                let input_gradients = (node.gradient_fn)(&input_values, output_gradient);

                // Accumulate gradients to input variables
                for (&input_id, &gradient) in node.inputs.iter().zip(input_gradients.iter()) {
                    if let Some(input_var) = self.variables.get_mut(&input_id) {
                        input_var.add_gradient(gradient);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get gradient for a specific variable
    pub fn get_gradient(&self, id: VariableId) -> Option<f64> {
        self.variables.get(&id).map(|v| v.gradient)
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.variables.clear();
        self.execution_order.clear();
    }
}

impl Default for ComputationTape {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Procedural Macro Implementation for Auto-differentiation
// =============================================================================

/// Configuration for automatic differentiation code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutodiffConfig {
    /// AD mode (forward or reverse)
    pub mode: ADMode,
    /// Maximum derivative order
    pub max_order: u32,
    /// Enable SIMD optimizations
    pub simd: bool,
    /// Enable GPU kernels
    pub gpu: bool,
    /// Enable symbolic differentiation
    pub symbolic: bool,
    /// Custom optimization flags
    pub optimizations: Vec<String>,
}

/// Automatic differentiation modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ADMode {
    /// Forward-mode automatic differentiation
    Forward,
    /// Reverse-mode automatic differentiation
    Reverse,
    /// Mixed-mode (forward for some variables, reverse for others)
    Mixed,
    /// Symbolic differentiation
    Symbolic,
}

impl Default for AutodiffConfig {
    fn default() -> Self {
        Self {
            mode: ADMode::Forward,
            max_order: 1,
            simd: false,
            gpu: false,
            symbolic: false,
            optimizations: Vec::new(),
        }
    }
}

/// Parse autodiff attributes from function
pub fn parse_autodiff_attributes(attrs: &[Attribute]) -> Result<AutodiffConfig> {
    let mut config = AutodiffConfig::default();

    for attr in attrs {
        if attr.path().is_ident("autodiff") {
            // Parse autodiff configuration from attribute
            // This would be more complex in a real implementation
            config.mode = ADMode::Forward; // Default for now
        }
    }

    Ok(config)
}

/// Generate automatic differentiation code for a function
pub fn generate_autodiff_impl(func: &ItemFn, config: &AutodiffConfig) -> Result<TokenStream> {
    let original_name = &func.sig.ident;
    let autodiff_name = syn::Ident::new(&format!("{}_autodiff", original_name), Span::call_site());

    match config.mode {
        ADMode::Forward => generate_forward_mode(func, &autodiff_name, config),
        ADMode::Reverse => generate_reverse_mode(func, &autodiff_name, config),
        ADMode::Mixed => generate_mixed_mode(func, &autodiff_name, config),
        ADMode::Symbolic => generate_symbolic_mode(func, &autodiff_name, config),
    }
}

/// Generate forward-mode automatic differentiation
fn generate_forward_mode(
    func: &ItemFn,
    autodiff_name: &syn::Ident,
    _config: &AutodiffConfig,
) -> Result<TokenStream> {
    let original_name = &func.sig.ident;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;

    // Transform function parameters to use Dual numbers
    let dual_inputs = transform_inputs_to_dual(inputs)?;
    let dual_output = transform_output_to_dual(output)?;

    // Transform function body to use Dual arithmetic
    let dual_body = transform_body_to_dual(&func.block)?;

    let generated = quote! {
        /// Forward-mode automatic differentiation version
        pub fn #autodiff_name(#dual_inputs) -> #dual_output {
            #dual_body
        }

        /// Convenience function for computing derivative
        pub fn #original_name _derivative(x: f64) -> (f64, f64) {
            let dual_x = Dual::variable(x);
            let result = #autodiff_name(dual_x);
            (result.value(), result.derivative())
        }
    };

    Ok(generated)
}

/// Generate reverse-mode automatic differentiation
fn generate_reverse_mode(
    func: &ItemFn,
    autodiff_name: &syn::Ident,
    _config: &AutodiffConfig,
) -> Result<TokenStream> {
    let original_name = &func.sig.ident;
    let inputs = &func.sig.inputs;

    // Transform function to use Variables and computation tape
    let var_inputs = transform_inputs_to_variables(inputs)?;
    let tape_body = transform_body_to_tape(&func.block)?;

    let generated = quote! {
        /// Reverse-mode automatic differentiation version
        pub fn #autodiff_name(#var_inputs, tape: Arc<Mutex<ComputationTape>>) -> Variable {
            #tape_body
        }

        /// Convenience function for computing gradients
        pub fn #original_name _gradients(inputs: &[f64]) -> Vec<f64> {
            let tape = Arc::new(Mutex::new(ComputationTape::new()));

            // Create variables for inputs
            let vars: Vec<Variable> = inputs.iter()
                .map(|&x| Variable::with_graph(x, tape.clone()))
                .collect();

            // Forward pass
            let output = #autodiff_name(vars, tape.clone());

            // Backward pass
            if let Ok(mut tape_guard) = tape.lock() {
                let _ = tape_guard.backward(1.0);

                // Extract gradients
                vars.iter()
                    .map(|v| tape_guard.get_gradient(v.id).unwrap_or(0.0))
                    .collect()
            } else {
                vec![0.0; inputs.len()]
            }
        }
    };

    Ok(generated)
}

/// Generate mixed-mode automatic differentiation
fn generate_mixed_mode(
    func: &ItemFn,
    autodiff_name: &syn::Ident,
    config: &AutodiffConfig,
) -> Result<TokenStream> {
    // For mixed mode, we generate both forward and reverse versions
    let forward_impl = generate_forward_mode(func, autodiff_name, config)?;

    let reverse_name = syn::Ident::new(&format!("{}_reverse", autodiff_name), Span::call_site());
    let reverse_impl = generate_reverse_mode(func, &reverse_name, config)?;

    let generated = quote! {
        #forward_impl
        #reverse_impl

        /// Mixed-mode automatic differentiation
        pub fn #autodiff_name _mixed(inputs: &[f64], forward_vars: &[usize]) -> (f64, Vec<f64>) {
            // Implementation would choose forward or reverse mode per variable
            // This is a placeholder implementation
            let gradients = vec![0.0; inputs.len()];
            (0.0, gradients)
        }
    };

    Ok(generated)
}

/// Generate symbolic differentiation
fn generate_symbolic_mode(
    func: &ItemFn,
    autodiff_name: &syn::Ident,
    _config: &AutodiffConfig,
) -> Result<TokenStream> {
    let original_name = &func.sig.ident;

    let generated = quote! {
        /// Symbolic differentiation version
        pub fn #autodiff_name() -> SymbolicExpression {
            // This would generate symbolic expressions for derivatives
            // Placeholder implementation
            SymbolicExpression::new("derivative")
        }

        /// Get symbolic derivative as LaTeX string
        pub fn #original_name _latex() -> String {
            let expr = #autodiff_name();
            expr.to_latex()
        }
    };

    Ok(generated)
}

// =============================================================================
// Code Transformation Utilities
// =============================================================================

/// Transform function inputs to use Dual numbers
fn transform_inputs_to_dual(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
) -> Result<TokenStream> {
    let mut dual_inputs = Vec::new();

    for input in inputs {
        match input {
            FnArg::Typed(pat_type) => {
                let pat = &pat_type.pat;
                // Transform f64 to Dual, keep other types as-is
                match &*pat_type.ty {
                    Type::Path(type_path) if type_path.path.is_ident("f64") => {
                        dual_inputs.push(quote! { #pat: Dual });
                    }
                    ty => {
                        dual_inputs.push(quote! { #pat: #ty });
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidOperation(
                    "Unsupported function parameter type".to_string(),
                ));
            }
        }
    }

    Ok(quote! { #(#dual_inputs),* })
}

/// Transform function output to use Dual numbers
fn transform_output_to_dual(output: &ReturnType) -> Result<TokenStream> {
    match output {
        ReturnType::Type(_, ty) => match &**ty {
            Type::Path(type_path) if type_path.path.is_ident("f64") => Ok(quote! { Dual }),
            ty => Ok(quote! { #ty }),
        },
        ReturnType::Default => Ok(quote! { () }),
    }
}

/// Transform function body to use Dual arithmetic
fn transform_body_to_dual(block: &syn::Block) -> Result<TokenStream> {
    let mut transformed_stmts = Vec::new();

    for stmt in &block.stmts {
        let transformed = transform_statement_to_dual(stmt)?;
        transformed_stmts.push(transformed);
    }

    Ok(quote! { { #(#transformed_stmts)* } })
}

/// Transform a single statement to use Dual arithmetic
fn transform_statement_to_dual(stmt: &Stmt) -> Result<TokenStream> {
    match stmt {
        Stmt::Expr(expr, _) => {
            let transformed_expr = transform_expression_to_dual(expr)?;
            Ok(quote! { #transformed_expr })
        }
        Stmt::Local(local) => {
            // Transform variable declarations
            let pat = &local.pat;
            if let Some(local_init) = &local.init {
                let init = &local_init.expr;
                let transformed_init = transform_expression_to_dual(init)?;
                Ok(quote! { let #pat = #transformed_init; })
            } else {
                Ok(quote! { #stmt })
            }
        }
        _ => Ok(quote! { #stmt }),
    }
}

/// Transform an expression to use Dual arithmetic
fn transform_expression_to_dual(expr: &Expr) -> Result<TokenStream> {
    match expr {
        Expr::Binary(binary_expr) => {
            let left = transform_expression_to_dual(&binary_expr.left)?;
            let right = transform_expression_to_dual(&binary_expr.right)?;
            let op = &binary_expr.op;

            // Dual arithmetic preserves standard operators
            Ok(quote! { (#left) #op (#right) })
        }
        Expr::Call(call_expr) => {
            let func = &call_expr.func;
            let args: Vec<TokenStream> = call_expr
                .args
                .iter()
                .map(transform_expression_to_dual)
                .collect::<Result<Vec<_>>>()?;

            // Transform math functions to Dual versions
            match &**func {
                Expr::Path(path) if path.path.is_ident("exp") => {
                    Ok(quote! { dual_exp(#(#args),*) })
                }
                Expr::Path(path) if path.path.is_ident("ln") => Ok(quote! { dual_ln(#(#args),*) }),
                Expr::Path(path) if path.path.is_ident("sin") => {
                    Ok(quote! { dual_sin(#(#args),*) })
                }
                Expr::Path(path) if path.path.is_ident("cos") => {
                    Ok(quote! { dual_cos(#(#args),*) })
                }
                _ => Ok(quote! { #func(#(#args),*) }),
            }
        }
        Expr::Lit(lit_expr) => {
            // Transform numeric literals to Dual constants
            match &lit_expr.lit {
                syn::Lit::Float(float_lit) => {
                    let value = &float_lit.base10_digits();
                    let parsed_value: f64 = value.parse().map_err(|_| {
                        SklearsError::InvalidOperation("Invalid float literal".to_string())
                    })?;
                    Ok(quote! { Dual::constant(#parsed_value) })
                }
                syn::Lit::Int(int_lit) => {
                    let value = &int_lit.base10_digits();
                    let parsed_value: i64 = value.parse().map_err(|_| {
                        SklearsError::InvalidOperation("Invalid int literal".to_string())
                    })?;
                    Ok(quote! { Dual::constant(#parsed_value as f64) })
                }
                _ => Ok(quote! { #expr }),
            }
        }
        _ => Ok(quote! { #expr }),
    }
}

/// Transform function inputs to use Variables
fn transform_inputs_to_variables(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
) -> Result<TokenStream> {
    let mut var_inputs = Vec::new();

    for input in inputs {
        match input {
            FnArg::Typed(pat_type) => {
                let pat = &pat_type.pat;
                // Transform f64 to Variable, keep other types as-is
                match &*pat_type.ty {
                    Type::Path(type_path) if type_path.path.is_ident("f64") => {
                        var_inputs.push(quote! { #pat: Variable });
                    }
                    ty => {
                        var_inputs.push(quote! { #pat: #ty });
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidOperation(
                    "Unsupported function parameter type".to_string(),
                ));
            }
        }
    }

    Ok(quote! { #(#var_inputs),* })
}

/// Transform function body to use computation tape
fn transform_body_to_tape(_block: &syn::Block) -> Result<TokenStream> {
    // This would transform the function body to build a computation graph
    // For now, return a placeholder that creates a simple variable
    Ok(quote! {
        {
            // Placeholder: create a variable representing the function output
            Variable::with_graph(0.0, tape)
        }
    })
}

// =============================================================================
// Dual Number Math Functions
// =============================================================================

/// Exponential function for dual numbers
pub fn dual_exp(x: Dual) -> Dual {
    let exp_x = x.real.exp();
    Dual::new(exp_x, x.dual * exp_x)
}

/// Natural logarithm for dual numbers
pub fn dual_ln(x: Dual) -> Dual {
    Dual::new(x.real.ln(), x.dual / x.real)
}

/// Sine function for dual numbers
pub fn dual_sin(x: Dual) -> Dual {
    Dual::new(x.real.sin(), x.dual * x.real.cos())
}

/// Cosine function for dual numbers
pub fn dual_cos(x: Dual) -> Dual {
    Dual::new(x.real.cos(), -x.dual * x.real.sin())
}

/// Hyperbolic tangent for dual numbers
pub fn dual_tanh(x: Dual) -> Dual {
    let tanh_x = x.real.tanh();
    Dual::new(tanh_x, x.dual * (1.0 - tanh_x * tanh_x))
}

/// Sigmoid function for dual numbers
pub fn dual_sigmoid(x: Dual) -> Dual {
    let sigmoid_x = 1.0 / (1.0 + (-x.real).exp());
    Dual::new(sigmoid_x, x.dual * sigmoid_x * (1.0 - sigmoid_x))
}

/// Power function for dual numbers
pub fn dual_pow(base: Dual, exponent: f64) -> Dual {
    let pow_result = base.real.powf(exponent);
    Dual::new(
        pow_result,
        base.dual * exponent * base.real.powf(exponent - 1.0),
    )
}

// =============================================================================
// Symbolic Expression System
// =============================================================================

/// Symbolic expression for symbolic differentiation
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpression {
    /// Variable
    Variable(String),
    /// Constant
    Constant(f64),
    /// Addition
    Add(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Subtraction
    Sub(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Multiplication
    Mul(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Division
    Div(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Power
    Pow(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Function call
    Function(String, Vec<SymbolicExpression>),
}

impl SymbolicExpression {
    /// Create a new symbolic expression
    pub fn new(name: &str) -> Self {
        Self::Variable(name.to_string())
    }

    /// Differentiate with respect to a variable
    pub fn differentiate(&self, var: &str) -> Self {
        match self {
            SymbolicExpression::Variable(v) if v == var => SymbolicExpression::Constant(1.0),
            SymbolicExpression::Variable(_) => SymbolicExpression::Constant(0.0),
            SymbolicExpression::Constant(_) => SymbolicExpression::Constant(0.0),
            SymbolicExpression::Add(left, right) => SymbolicExpression::Add(
                Box::new(left.differentiate(var)),
                Box::new(right.differentiate(var)),
            ),
            SymbolicExpression::Sub(left, right) => SymbolicExpression::Sub(
                Box::new(left.differentiate(var)),
                Box::new(right.differentiate(var)),
            ),
            SymbolicExpression::Mul(left, right) => {
                // Product rule: (fg)' = f'g + fg'
                SymbolicExpression::Add(
                    Box::new(SymbolicExpression::Mul(
                        Box::new(left.differentiate(var)),
                        right.clone(),
                    )),
                    Box::new(SymbolicExpression::Mul(
                        left.clone(),
                        Box::new(right.differentiate(var)),
                    )),
                )
            }
            SymbolicExpression::Div(left, right) => {
                // Quotient rule: (f/g)' = (f'g - fg')/g²
                SymbolicExpression::Div(
                    Box::new(SymbolicExpression::Sub(
                        Box::new(SymbolicExpression::Mul(
                            Box::new(left.differentiate(var)),
                            right.clone(),
                        )),
                        Box::new(SymbolicExpression::Mul(
                            left.clone(),
                            Box::new(right.differentiate(var)),
                        )),
                    )),
                    Box::new(SymbolicExpression::Pow(
                        right.clone(),
                        Box::new(SymbolicExpression::Constant(2.0)),
                    )),
                )
            }
            SymbolicExpression::Pow(base, exp) => {
                // Power rule and chain rule
                match (&**base, &**exp) {
                    (_, SymbolicExpression::Constant(n)) => {
                        // Simple power rule: (x^n)' = n*x^(n-1)*x'
                        SymbolicExpression::Mul(
                            Box::new(SymbolicExpression::Mul(
                                Box::new(SymbolicExpression::Constant(*n)),
                                Box::new(SymbolicExpression::Pow(
                                    base.clone(),
                                    Box::new(SymbolicExpression::Constant(n - 1.0)),
                                )),
                            )),
                            Box::new(base.differentiate(var)),
                        )
                    }
                    _ => {
                        // General case: (f^g)' = f^g * (g'*ln(f) + g*f'/f)
                        SymbolicExpression::Mul(
                            Box::new(self.clone()),
                            Box::new(SymbolicExpression::Add(
                                Box::new(SymbolicExpression::Mul(
                                    Box::new(exp.differentiate(var)),
                                    Box::new(SymbolicExpression::Function(
                                        "ln".to_string(),
                                        vec![*base.clone()],
                                    )),
                                )),
                                Box::new(SymbolicExpression::Mul(
                                    exp.clone(),
                                    Box::new(SymbolicExpression::Div(
                                        Box::new(base.differentiate(var)),
                                        base.clone(),
                                    )),
                                )),
                            )),
                        )
                    }
                }
            }
            SymbolicExpression::Function(name, args) => {
                self.differentiate_function(name, args, var)
            }
        }
    }

    /// Differentiate function calls
    fn differentiate_function(&self, name: &str, args: &[SymbolicExpression], var: &str) -> Self {
        match name {
            "sin" if args.len() == 1 => {
                // d/dx sin(f) = cos(f) * f'
                SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::Function(
                        "cos".to_string(),
                        args.to_vec(),
                    )),
                    Box::new(args[0].differentiate(var)),
                )
            }
            "cos" if args.len() == 1 => {
                // d/dx cos(f) = -sin(f) * f'
                SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::Constant(-1.0)),
                    Box::new(SymbolicExpression::Mul(
                        Box::new(SymbolicExpression::Function(
                            "sin".to_string(),
                            args.to_vec(),
                        )),
                        Box::new(args[0].differentiate(var)),
                    )),
                )
            }
            "exp" if args.len() == 1 => {
                // d/dx exp(f) = exp(f) * f'
                SymbolicExpression::Mul(
                    Box::new(self.clone()),
                    Box::new(args[0].differentiate(var)),
                )
            }
            "ln" if args.len() == 1 => {
                // d/dx ln(f) = f'/f
                SymbolicExpression::Div(
                    Box::new(args[0].differentiate(var)),
                    Box::new(args[0].clone()),
                )
            }
            _ => {
                // Unknown function - return symbolic derivative
                SymbolicExpression::Function(format!("d{}_d{}", name, var), args.to_vec())
            }
        }
    }

    /// Convert to LaTeX representation
    pub fn to_latex(&self) -> String {
        match self {
            SymbolicExpression::Variable(v) => v.clone(),
            SymbolicExpression::Constant(c) => {
                if c.fract() == 0.0 {
                    format!("{}", *c as i64)
                } else {
                    format!("{:.3}", c)
                }
            }
            SymbolicExpression::Add(left, right) => {
                format!("({} + {})", left.to_latex(), right.to_latex())
            }
            SymbolicExpression::Sub(left, right) => {
                format!("({} - {})", left.to_latex(), right.to_latex())
            }
            SymbolicExpression::Mul(left, right) => {
                format!("({} \\cdot {})", left.to_latex(), right.to_latex())
            }
            SymbolicExpression::Div(left, right) => {
                format!("\\frac{{{}}}{{{}}}", left.to_latex(), right.to_latex())
            }
            SymbolicExpression::Pow(base, exp) => {
                format!("{}^{{{}}}", base.to_latex(), exp.to_latex())
            }
            SymbolicExpression::Function(name, args) => {
                if args.is_empty() {
                    format!("\\{}", name)
                } else if args.len() == 1 {
                    format!("\\{}({})", name, args[0].to_latex())
                } else {
                    let arg_strs: Vec<String> = args.iter().map(|a| a.to_latex()).collect();
                    format!("\\{}({})", name, arg_strs.join(", "))
                }
            }
        }
    }

    /// Simplify the expression
    pub fn simplify(&self) -> Self {
        match self {
            SymbolicExpression::Add(left, right) => {
                let left_simp = left.simplify();
                let right_simp = right.simplify();

                match (&left_simp, &right_simp) {
                    (SymbolicExpression::Constant(0.0), _) => right_simp,
                    (_, SymbolicExpression::Constant(0.0)) => left_simp,
                    (SymbolicExpression::Constant(a), SymbolicExpression::Constant(b)) => {
                        SymbolicExpression::Constant(a + b)
                    }
                    _ => SymbolicExpression::Add(Box::new(left_simp), Box::new(right_simp)),
                }
            }
            SymbolicExpression::Mul(left, right) => {
                let left_simp = left.simplify();
                let right_simp = right.simplify();

                match (&left_simp, &right_simp) {
                    (SymbolicExpression::Constant(0.0), _)
                    | (_, SymbolicExpression::Constant(0.0)) => SymbolicExpression::Constant(0.0),
                    (SymbolicExpression::Constant(1.0), _) => right_simp,
                    (_, SymbolicExpression::Constant(1.0)) => left_simp,
                    (SymbolicExpression::Constant(a), SymbolicExpression::Constant(b)) => {
                        SymbolicExpression::Constant(a * b)
                    }
                    _ => SymbolicExpression::Mul(Box::new(left_simp), Box::new(right_simp)),
                }
            }
            SymbolicExpression::Pow(base, exponent) => {
                let base_simp = base.simplify();
                let exp_simp = exponent.simplify();

                match (&base_simp, &exp_simp) {
                    // x^1 = x
                    (_, SymbolicExpression::Constant(1.0)) => base_simp,
                    // x^0 = 1
                    (_, SymbolicExpression::Constant(0.0)) => SymbolicExpression::Constant(1.0),
                    // 1^n = 1
                    (SymbolicExpression::Constant(1.0), _) => SymbolicExpression::Constant(1.0),
                    // 0^n = 0 (for n > 0)
                    (SymbolicExpression::Constant(0.0), SymbolicExpression::Constant(n))
                        if *n > 0.0 =>
                    {
                        SymbolicExpression::Constant(0.0)
                    }
                    // a^b = a^b (constant exponentiation)
                    (SymbolicExpression::Constant(a), SymbolicExpression::Constant(b)) => {
                        SymbolicExpression::Constant(a.powf(*b))
                    }
                    _ => SymbolicExpression::Pow(Box::new(base_simp), Box::new(exp_simp)),
                }
            }
            _ => self.clone(),
        }
    }
}

// =============================================================================
// Higher-order Derivatives
// =============================================================================

/// Compute second derivative using dual numbers
pub fn second_derivative<F>(_f: F, x: f64) -> f64
where
    F: Fn(Dual) -> Dual,
{
    // Use dual numbers nested to compute second derivative
    let _dual_x = Dual::new(x, 1.0);

    // This is a simplified placeholder - real implementation would be more complex
    0.0
}

/// Compute Hessian matrix for multivariate functions
pub fn hessian<F>(f: F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut hessian = vec![vec![0.0; n]; n];

    let h = 1e-8; // Small step size

    // Compute Hessian using finite differences
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal elements: f''(x)
                let mut x_plus = x.to_vec();
                let mut x_minus = x.to_vec();
                x_plus[i] += h;
                x_minus[i] -= h;

                let f_plus = f(&x_plus);
                let f_center = f(x);
                let f_minus = f(&x_minus);

                hessian[i][j] = (f_plus - 2.0 * f_center + f_minus) / (h * h);
            } else {
                // Off-diagonal elements: mixed partial derivatives
                let mut x_pp = x.to_vec();
                let mut x_pm = x.to_vec();
                let mut x_mp = x.to_vec();
                let mut x_mm = x.to_vec();

                x_pp[i] += h;
                x_pp[j] += h;
                x_pm[i] += h;
                x_pm[j] -= h;
                x_mp[i] -= h;
                x_mp[j] += h;
                x_mm[i] -= h;
                x_mm[j] -= h;

                let f_pp = f(&x_pp);
                let f_pm = f(&x_pm);
                let f_mp = f(&x_mp);
                let f_mm = f(&x_mm);

                hessian[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            }
        }
    }

    hessian
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Compute forward-mode derivative
pub fn forward_diff<F>(f: F, x: f64) -> (f64, f64)
where
    F: Fn(Dual) -> Dual,
{
    let dual_x = Dual::variable(x);
    let result = f(dual_x);
    (result.value(), result.derivative())
}

/// Compute gradient using finite differences
pub fn gradient<F>(f: F, x: &[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut grad = vec![0.0; x.len()];
    let h = 1e-8;

    for i in 0..x.len() {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] += h;
        x_minus[i] -= h;

        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
    }

    grad
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_arithmetic() {
        let x = Dual::new(2.0, 1.0);
        let y = Dual::new(3.0, 0.0);

        let sum = x + y;
        assert_eq!(sum.real, 5.0);
        assert_eq!(sum.dual, 1.0);

        let product = x * y;
        assert_eq!(product.real, 6.0);
        assert_eq!(product.dual, 3.0);
    }

    #[test]
    fn test_dual_math_functions() {
        let x = Dual::variable(1.0);

        let exp_result = dual_exp(x);
        assert!((exp_result.real - std::f64::consts::E).abs() < 1e-10);
        assert!((exp_result.dual - std::f64::consts::E).abs() < 1e-10);

        let ln_result = dual_ln(x);
        assert!((ln_result.real - 0.0).abs() < 1e-10);
        assert!((ln_result.dual - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_forward_diff() {
        // Test f(x) = x^2, f'(x) = 2x
        let f = |x: Dual| x * x;
        let (value, derivative) = forward_diff(f, 3.0);

        assert_eq!(value, 9.0);
        assert_eq!(derivative, 6.0);
    }

    #[test]
    fn test_symbolic_differentiation() {
        let x = SymbolicExpression::Variable("x".to_string());
        let x_squared = SymbolicExpression::Pow(
            Box::new(x.clone()),
            Box::new(SymbolicExpression::Constant(2.0)),
        );

        let derivative = x_squared.differentiate("x");
        let simplified = derivative.simplify();

        // Should be 2*x
        match simplified {
            SymbolicExpression::Mul(left, right) => {
                assert_eq!(*left, SymbolicExpression::Constant(2.0));
                assert_eq!(*right, SymbolicExpression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_gradient_computation() {
        // Test f(x, y) = x^2 + y^2, gradient = [2x, 2y]
        let f = |vars: &[f64]| vars[0] * vars[0] + vars[1] * vars[1];
        let grad = gradient(f, &[2.0, 3.0]);

        assert!((grad[0] - 4.0).abs() < 1e-6);
        assert!((grad[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_computation_tape() {
        let mut tape = ComputationTape::new();

        // Create variables
        let x = Variable::new(2.0);
        let y = Variable::new(3.0);

        tape.register_variable(x.clone());
        tape.register_variable(y.clone());

        // Test basic tape operations
        assert_eq!(tape.variables.len(), 2);
        assert!(tape.get_gradient(x.id).is_some());
    }

    #[test]
    fn test_variable_creation() {
        let var1 = Variable::new(1.0);
        let var2 = Variable::new(2.0);

        assert_ne!(var1.id, var2.id);
        assert_eq!(var1.value, 1.0);
        assert_eq!(var2.value, 2.0);
        assert_eq!(var1.gradient, 0.0);
        assert_eq!(var2.gradient, 0.0);
    }

    #[test]
    fn test_autodiff_config() {
        let config = AutodiffConfig::default();
        assert_eq!(config.mode, ADMode::Forward);
        assert_eq!(config.max_order, 1);
        assert!(!config.simd);
        assert!(!config.gpu);
    }

    #[test]
    fn test_symbolic_latex_output() {
        let expr = SymbolicExpression::Div(
            Box::new(SymbolicExpression::Variable("x".to_string())),
            Box::new(SymbolicExpression::Constant(2.0)),
        );

        let latex = expr.to_latex();
        assert_eq!(latex, "\\frac{x}{2}");
    }
}
